# clipper-gpu-worker

NVENC-accelerated render worker for Cl1pper's **Clip Maker**: given one short
moment + a render-ready 9:16 plan + a transcript, it composes the vertical
frame and burns libass captions for **N style variants** of the same moment.

The whole render is ffmpeg filtergraphs (compose in `engine/v11_render.py`,
caption stamp in `engine/captions_ass.py`), so going GPU is a one-line codec
swap per ffmpeg call — `RENDER_GPU=1` flips `libx264` → `h264_nvenc`
(`engine/encoder.py`). The Docker image sets `RENDER_GPU=1`.

This repo is the **render closure only** — no VLM planning, transcription, bot,
web, or payment code. Composition plans are computed upstream and handed to the
worker render-ready.

## Deploy on RunPod Serverless (from GitHub)

1. Push this repo (public).
2. RunPod → Serverless → New Endpoint → **Source: GitHub**, point it at this
   repo. RunPod builds the image from the `Dockerfile`.
3. Pick a GPU type (any NVENC-capable NVIDIA card; a cheap one is fine — the
   bottleneck is encode, not compute).
4. Deploy, then hit the endpoint with the self-test payload below.

The Docker build **hard-fails** if its ffmpeg lacks `h264_nvenc` or the `ass`
filter, so a bad base image can't ship a silently-CPU worker.

## Self-test (no inputs)

`test_input.json`:

```json
{ "input": { "selftest": true, "variants_count": 3, "seconds": 15, "return_video": false } }
```

The worker synthesizes a 16:9 `testsrc2` source + tone, renders 3 variants
through the real compose+stamp pipeline, and returns timings plus the NVENC
capability probe. Check `ffmpeg_caps.nvenc_encode_ok` — that's the proof the
GPU encoder actually ran (listing the encoder isn't enough).

Response shape:

```json
{
  "ok": true,
  "mode": "selftest",
  "canvas": [1080, 1920],
  "variants": [
    { "index": 0, "pack": "...", "subtitle_mode": "word_only",
      "compose_s": 1.2, "stamp_s": 0.6, "reused_base": false,
      "bytes": 812345, "duration_s": 15.0 }
  ],
  "total_s": 9.1,
  "ffmpeg_caps": { "render_gpu_env": true, "nvenc_listed": true,
                   "ass_filter": true, "nvenc_encode_ok": true }
}
```

## Real job

```json
{
  "input": {
    "source_url": "https://.../clip.mp4",
    "transcript": { "phrases": [ { "text": "...", "t_start": 0.0, "t_end": 2.1, "words": [] } ] },
    "segments": [ { "plan": { "composition": "single_view", "panels": [] }, "t_start": 0.0, "t_end": 15.0 } ],
    "variants_count": 3,
    "format": "9x16",
    "fps": 30,
    "return_video": true
  }
}
```

- `source_url` — direct video URL the worker downloads.
- `transcript` — `{ "phrases": [...] }`, same schema as `fixtures/transcript.json`.
- `segments` — one or more `{ plan, t_start, t_end }`; `plan` is a render-ready
  v11 plan (`composition` + pixel-box `panels`).
- `variants` — explicit `[[pack, subtitle_mode], ...]`, OR omit and pass
  `variants_count` (3–10) to auto-pick distinct packs.
- `format` — `9x16` (default), `3x4`, or `4x5`.
- `return_video` — when `true`, each variant carries `mp4_base64`.

## Local smoke test (CPU)

```bash
pip install -r requirements.txt   # runpod not required for the render path
python smoke_test.py              # libx264 fallback, proves the import closure
```
