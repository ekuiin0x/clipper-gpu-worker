"""RunPod serverless entrypoint for the Cl1pper GPU render worker.

Two input modes:

  selftest  — synthesize a 16:9 ``testsrc2`` source + use the bundled
              transcript/plan fixtures, render N variants, and report timings
              plus the NVENC capability probe. No external inputs. This is the
              "is the GPU box wired up correctly" smoke test.

  job       — real render: read the source, render the supplied ``segments``
              + ``transcript`` into the requested variants. The source arrives
              one of two ways: ``source_key`` (the preferred path — an object
              on the RunPod network volume mounted at /runpod-volume, read as a
              local file; each variant is written back under ``output_prefix``
              and its key returned) or the legacy ``source_url`` (downloaded
              from a temp host, variants uploaded back via ``upload_results``).

  volume_ls — diagnostic: list the mounted network volume and return the tree.
              Used to verify that an object PUT at S3 key ``clipper/x/y.mp4``
              lands at ``/runpod-volume/clipper/x/y.mp4``.

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
            "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=15:duration=0.2",
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


def _upload_litterbox(path: Path, ttl: str = "24h") -> str:
    """Upload a rendered variant to litterbox and return its temp URL.

    Real 1080x1920 outputs are tens of MB each — far over RunPod's inline
    result limit — so when the caller asks for ``upload_results`` we push each
    mp4 to a short-lived host and return URLs the app downloads. litterbox is a
    no-auth temp host (1h/12h/24h/72h TTL); production should point this at
    private object storage instead.
    """
    r = _run([
        "curl", "-s", "--max-time", "300",
        "-F", "reqtype=fileupload",
        "-F", f"time={ttl}",
        "-F", f"fileToUpload=@{path}",
        "https://litterbox.catbox.moe/resources/internals/api.php",
    ], timeout=320)
    url = (r.stdout or "").strip()
    if not url.startswith("http"):
        raise RuntimeError(
            f"litterbox upload failed (rc={r.returncode}): "
            f"out={url[:200]!r} err={(r.stderr or '')[:200]!r}"
        )
    return url


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
# network volume (S3-backed) transport
# ---------------------------------------------------------------------------

# The RunPod network volume mounts here; the app PUTs source objects to the
# same volume over the S3 gateway and reads variants back by key.
VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))


def _list_tree(root: Path, limit: int = 300) -> dict:
    """List files under ``root`` (relative paths + sizes) for diagnostics."""
    info: dict = {"root": str(root), "exists": root.exists(), "files": []}
    if not root.exists():
        return info
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            sz = -1
        info["files"].append({"key": str(p.relative_to(root)).replace("\\", "/"),
                              "bytes": sz})
        n += 1
        if n >= limit:
            info["truncated"] = True
            break
    return info


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    inp = (event or {}).get("input") or {}

    # Cheap diagnostic: confirm the network volume is mounted and inspect the
    # S3-key -> volume-path mapping without rendering anything.
    if inp.get("volume_ls"):
        return {"ok": True, "mode": "volume_ls",
                "volume": _list_tree(VOLUME_ROOT)}

    selftest = bool(inp.get("selftest", False))
    return_video = bool(inp.get("return_video", False))
    upload_results = bool(inp.get("upload_results", False))
    result_ttl = str(inp.get("result_ttl", "24h"))
    output_prefix = inp.get("output_prefix")
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
            source_key = inp.get("source_key")
            source_url = inp.get("source_url")
            if source_key:
                # Preferred path: the source is an object on the shared volume.
                vol_src = VOLUME_ROOT / source_key
                if not vol_src.exists():
                    # Surface the real mount layout so a broken S3-key ->
                    # volume-path assumption is diagnosable, not a silent miss.
                    return {
                        "ok": False,
                        "error": f"source_key not found on volume: {vol_src}",
                        "volume": _list_tree(VOLUME_ROOT),
                        "ffmpeg_caps": _ffmpeg_caps(),
                    }
                src = vol_src
            elif source_url:
                _download(source_url, src)
            else:
                raise ValueError(
                    "job mode requires 'source_key' or 'source_url'")

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
            if output_prefix and p.exists():
                # Volume mode: write the variant back onto the shared volume and
                # return its key; the app GETs it over the S3 gateway.
                idx = int(r.get("index", 0))
                pack = r.get("pack", "")
                rel = f"{output_prefix}/variant_{idx:02d}_{pack}.mp4"
                dst = VOLUME_ROOT / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                r["key"] = rel
            elif upload_results and p.exists():
                r["url"] = _upload_litterbox(p, ttl=result_ttl)
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
