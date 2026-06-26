# Cl1pper GPU render worker — NVENC-accelerated 9:16 compose + libass stamp.
#
# Base: CUDA runtime on Ubuntu 22.04. We rely on the distro ffmpeg, which is
# built with --enable-nvenc and --enable-libass, so no custom ffmpeg build is
# needed. The build FAILS LOUDLY if either capability is missing, so a broken
# base image can never ship a worker that silently falls back to libx264.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    RENDER_GPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fail the build if this ffmpeg can't do NVENC or libass — the whole point of
# the GPU worker. Cheaper to catch here than after a RunPod deploy.
RUN ffmpeg -hide_banner -encoders | grep -q h264_nvenc \
    && ffmpeg -hide_banner -filters  | grep -qw ass \
    || (echo "FATAL: ffmpeg missing h264_nvenc or ass filter" && exit 1)

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY engine ./engine
COPY fixtures ./fixtures
COPY handler.py render_job.py ./

CMD ["python3", "-u", "handler.py"]
