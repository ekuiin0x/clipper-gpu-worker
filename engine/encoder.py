"""Encoder selection: CPU libx264 locally, NVENC on the RunPod GPU worker.

The whole render is ffmpeg filtergraphs (compose in v11_render, caption stamp in
captions_ass), so going GPU is a one-line codec swap at each ffmpeg call. Flip it
with ``RENDER_GPU=1`` (the Dockerfile sets this for the GPU image).
"""
from __future__ import annotations

import os


def render_gpu_enabled() -> bool:
    return os.getenv("RENDER_GPU", "0").strip().lower() in ("1", "true", "yes", "on")


def video_encoder_args(cpu_args: list[str]) -> list[str]:
    """Swap the CPU codec args for NVENC when RENDER_GPU is on.

    ``cpu_args`` is the existing libx264 invocation
    (e.g. ``["-c:v","libx264","-preset","veryfast","-crf","20"]``). On GPU we
    return an h264_nvenc equivalent; ``-cq`` is NVENC's constant-quality knob,
    roughly the CRF analogue. ``-pix_fmt`` is left to the caller (unchanged).
    """
    if not render_gpu_enabled():
        return cpu_args
    return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
