"""RunPod serverless entrypoint for the Cl1pper GPU render worker.

Two input modes:

  selftest  — synthesize a 16:9 ``testsrc2`` source + use the bundled
              transcript/plan fixtures, render N variants, and report timings
              plus the NVENC capability probe. No external inputs. This is the
              "is the GPU box wired up correctly" smoke test.

  job       — real render: download ``source_url``, render the supplied
              ``segments`` + ``transcript`` into the requested variants.

Either way the response carries the ffmpeg capability probe so you can confirm
``h264_nvenc`` is actually being used on the box.

RunPod builds this image from the repo Dockerfile; ``CMD python3 -u handler.py``
runs ``runpod.serverless.start`` at the bottom of this file. The render code is
import-clean of runpod, so this module can be imported and ``handler({...})``
called directly in a local CPU smoke test.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from engine.clipmaker import parse_format, pick_style_descriptors
from engine.encoder import render_gpu_enabled
from render_job import Segment, run_render_job

_HERE = Path(__file__).resolve().parent
_FIXTURES = _HERE / "fixtures"


# ---------------------------------------------------------------------------
# ffmpeg capability probe
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ffmpeg_caps() -> dict:
    """Probe whether NVENC + the libass filter are actually available.

    ``nvenc_encode_ok`` is the real signal — listing the encoder doesn't mean
    the box has a usable GPU. We do a 0.2s null-muxed test encode to confirm.
    """
    caps: dict = {"render_gpu_env": render_gpu_enabled()}

    try:
        enc = _run(["ffmpeg", "-hide_banner", "-encoders"]).stdout
        caps["nvenc_listed"] = "h264_nvenc" in enc
    except Exception as e:
        caps["nvenc_listed"] = False
        caps["nvenc_listed_error"] = str(e)

    try:
        filt = _run(["ffmpeg", "-hide_banner", "-filters"]).stdout
        caps["ass_filter"] = any(
            line.split()[1] == "ass"
            for line in filt.splitlines()
            if len(line.split()) > 1
        )
    except Exception as e:
        caps["ass_filter"] = False
        caps["ass_filter_error"] = str(e)

    try:
        probe = _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=15:duration=0.2",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ])
        caps["nvenc_encode_ok"] = probe.returncode == 0
        if probe.returncode != 0:
            caps["nvenc_encode_error"] = (probe.stderr or "")[-400:]
    except Exception as e:
        caps["nvenc_encode_ok"] = False
        caps["nvenc_encode_error"] = str(e)

    return caps


# ---------------------------------------------------------------------------
# source acquisition
# ---------------------------------------------------------------------------

def _synth_source(dst: Path, seconds: float) -> None:
    """Synthesize a 1280x720 (16:9) test clip with motion + tone.

    Lets the self-test exercise the full crop/scale/stamp pipeline without
    shipping a binary fixture. testsrc2 has moving content so the compose
    filtergraph isn't a no-op.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds:.2f}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(dst),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"source synth failed (rc={rc})")


def _download(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def _probe_duration(path: Path) -> float:
    try:
        r = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ])
        return round(float(r.stdout.strip()), 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    inp = (event or {}).get("input") or {}
    selftest = bool(inp.get("selftest", False))
    return_video = bool(inp.get("return_video", False))
    out_w, out_h = parse_format(inp.get("format"))
    fps = float(inp.get("fps", 30))

    job_t0 = time.time()
    work_dir = Path(tempfile.mkdtemp(prefix="cl1pper_job_"))

    try:
        src = work_dir / "source.mp4"

        if selftest:
            seconds = float(inp.get("seconds", 15))
            _synth_source(src, seconds)
            transcript_path = _FIXTURES / "transcript.json"
            plan = json.loads((_FIXTURES / "plan.json").read_text())
            seg_end = min(seconds, _probe_duration(src) or seconds)
            segments = [Segment(plan=plan, t_start=0.0, t_end=seg_end)]
            n = int(inp.get("variants_count", 3))
            variants = [(d.pack, d.subtitle_mode) for d in pick_style_descriptors(n)]
        else:
            source_url = inp.get("source_url")
            if not source_url:
                raise ValueError("job mode requires 'source_url'")
            _download(source_url, src)

            transcript = inp.get("transcript")
            if not transcript:
                raise ValueError("job mode requires 'transcript'")
            transcript_path = work_dir / "transcript.json"
            transcript_path.write_text(json.dumps(transcript), encoding="utf-8")

            raw_segments = inp.get("segments") or []
            if not raw_segments:
                raise ValueError("job mode requires non-empty 'segments'")
            segments = [
                Segment(plan=s["plan"], t_start=float(s["t_start"]), t_end=float(s["t_end"]))
                for s in raw_segments
            ]

            raw_variants = inp.get("variants")
            if raw_variants:
                variants = [(v[0], v[1]) for v in raw_variants]
            else:
                n = int(inp.get("variants_count", 3))
                variants = [(d.pack, d.subtitle_mode) for d in pick_style_descriptors(n)]

        results = run_render_job(
            src=src,
            segments=segments,
            transcript_path=transcript_path,
            variants=variants,
            out_w=out_w,
            out_h=out_h,
            fps=fps,
            work_dir=work_dir,
            brand_is_watermark=True,
            brand_channel=inp.get("brand_channel"),
        )

        for r in results:
            p = Path(r["path"])
            r["bytes"] = p.stat().st_size if p.exists() else 0
            r["duration_s"] = _probe_duration(p)
            if return_video and p.exists():
                r["mp4_base64"] = base64.b64encode(p.read_bytes()).decode("ascii")
            r.pop("path", None)

        return {
            "ok": True,
            "mode": "selftest" if selftest else "job",
            "canvas": [out_w, out_h],
            "variants": results,
            "total_s": round(time.time() - job_t0, 2),
            "ffmpeg_caps": _ffmpeg_caps(),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "ffmpeg_caps": _ffmpeg_caps(),
        }
    finally:
        if not return_video:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
