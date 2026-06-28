"""Render-job orchestration for the GPU worker.

One job = one short moment + N style variants. The expensive 9:16 compose
(crop/scale/stack filtergraph) is shared across variants that produce the
same panel ordering, so it runs ONCE per distinct base; each variant then
gets its own libass caption stamp (the cheap pass).

Pipeline per job:

    segments + plan  --compose-->  base.mp4   (one per distinct order-key)
    base.mp4 + pack  --stamp---->  variant_k.mp4

Both ffmpeg passes pick up the NVENC codec automatically when ``RENDER_GPU=1``
(see ``engine.encoder``). This module is codec-agnostic — it just calls the
vendored render functions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from engine.auto_style import preset_for_pack
from engine.captions_ass import burn_styled_ffmpeg
from engine.clipmaker import permute_panels, render_variant_base, variant_orderkey
from engine.styling import load_subtitles_json


@dataclass
class Segment:
    """One planner segment: a render-ready v11 plan over a time window.

    Field names match what ``engine.clipmaker`` expects from a
    ``V11Segment`` (``.plan``, ``.t_start``, ``.t_end``).
    """
    plan: dict
    t_start: float
    t_end: float


def _clip_duration(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    return max(0.0, segments[-1].t_end - segments[0].t_start)


def compose_base_only(
    *,
    src: Path,
    segments: list[Segment],
    out_w: int,
    out_h: int,
    fps: float,
    work_dir: Path,
) -> list[dict]:
    """Compose the 9:16 base ONCE and return it unstyled (no caption stamp).

    This is the single-clip ``/generate`` offload: the app plans + transcribes
    locally, ships the source + segments here for the expensive NVENC compose,
    and burns its chosen style locally on the returned base. Result shape mirrors
    :func:`run_render_job` (one entry) so the handler's volume write-back loop is
    unchanged; ``pack`` is empty and ``stamp_s`` is 0.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_path = work_dir / "base_00.mp4"
    t0 = time.time()
    render_variant_base(
        src=Path(src),
        segments=segments,
        variant_index=0,
        out_w=out_w,
        out_h=out_h,
        fps=fps,
        work_dir=work_dir / "base_00_work",
        dst=base_path,
    )
    compose_s = time.time() - t0
    return [{
        "index": 0,
        "pack": "",
        "subtitle_mode": "",
        "path": str(base_path),
        "compose_s": round(compose_s, 2),
        "stamp_s": 0.0,
        "reused_base": False,
        "compose_only": True,
    }]


def run_render_job(
    *,
    src: Path,
    segments: list[Segment],
    transcript_path: Path,
    variants: list[tuple[str, str]],
    out_w: int,
    out_h: int,
    fps: float,
    work_dir: Path,
    brand_is_watermark: bool = True,
    brand_channel: str | None = None,
) -> list[dict]:
    """Compose + stamp every variant. Returns a per-variant result list.

    ``variants`` is a list of ``(pack_name, subtitle_mode)`` pairs. Bases are
    rendered once per distinct panel-order signature (``variant_orderkey``) and
    reused, so single-panel clips compose exactly one base for the batch.

    Each result dict: ``{index, pack, subtitle_mode, path, compose_s, stamp_s,
    reused_base}``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    subs = load_subtitles_json(transcript_path)
    rep_plan = segments[0].plan  # representative plan for Y-position derivation

    base_by_key: dict[tuple, tuple[Path, float]] = {}
    results: list[dict] = []

    for idx, (pack, subtitle_mode) in enumerate(variants):
        key = variant_orderkey(segments, idx)

        reused = key in base_by_key
        if reused:
            base_path, compose_s = base_by_key[key]
        else:
            base_path = work_dir / f"base_{len(base_by_key):02d}.mp4"
            t0 = time.time()
            render_variant_base(
                src=Path(src),
                segments=segments,
                variant_index=idx,
                out_w=out_w,
                out_h=out_h,
                fps=fps,
                work_dir=work_dir / f"base_{len(base_by_key):02d}_work",
                dst=base_path,
            )
            compose_s = time.time() - t0
            base_by_key[key] = (base_path, compose_s)

        # Y positions follow the permuted representative plan for this variant.
        _spec, preset = preset_for_pack(
            pack,
            plan=permute_panels(rep_plan, idx),
            hook_enabled=False,
        )

        dst = work_dir / f"variant_{idx:02d}.mp4"
        stamp_dir = work_dir / f"stamp_{idx:02d}"
        t1 = time.time()
        burn_styled_ffmpeg(
            src=base_path,
            dst=dst,
            preset=preset,
            subs=subs,
            hook=None,
            subtitle_mode=subtitle_mode,
            work_dir=stamp_dir,
            brand_is_watermark=brand_is_watermark,
            brand_channel=brand_channel,
            out_w=out_w,
            out_h=out_h,
        )
        stamp_s = time.time() - t1

        results.append({
            "index": idx,
            "pack": pack,
            "subtitle_mode": subtitle_mode,
            "path": str(dst),
            "compose_s": round(compose_s, 2),
            "stamp_s": round(stamp_s, 2),
            "reused_base": reused,
        })

    return results
