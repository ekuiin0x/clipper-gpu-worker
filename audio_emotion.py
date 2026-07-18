"""SenseVoice-Small audio analysis: per-window emotion + audio-event tags.

This is the audio-analysis probe for the detection rework. Given a VOD audio
file it produces a coarse timeline of:

  * emotion   — one of HAPPY / SAD / ANGRY / NEUTRAL / FEARFUL / DISGUSTED /
                SURPRISED / EMO_UNKNOWN  (SenseVoice SER head)
  * events    — Speech / Laughter / Applause / BGM / Cry / Cough / ...
                (SenseVoice audio-event detection head)
  * language  — zh / en / yue / ja / ko / nospeech

SenseVoice-Small (FunASR ``iic/SenseVoiceSmall``) emits all of these plus the
ASR text in one non-autoregressive forward pass. We run it per fixed window so
the output is a deterministic tag timeline, then tally the tags. No VAD/timestamp
guessing — fixed windows keep the shape predictable for a first test.

Model is loaded lazily and cached module-level so a warm RunPod worker reuses it
across invocations.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# SenseVoice rich-transcription tag vocabularies. Everything the model emits is
# wrapped as ``<|TOKEN|>``; we bucket each token by which vocab it belongs to.
EMOTION_TOKENS = {
    "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED",
    "EMO_UNKNOWN",
}
EVENT_TOKENS = {
    "Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough",
}
LANG_TOKENS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
# textnorm markers we drop
_NOISE_TOKENS = {"withitn", "woitn"}

_TAG_RE = re.compile(r"<\|([^|>]+)\|>")

_MODEL = None


def _load_model():
    """Lazily construct the FunASR SenseVoice model (cached module-level)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from funasr import AutoModel

    device = os.getenv("AUDIO_EMOTION_DEVICE", "")
    if not device:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    _MODEL = AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        device=device,
    )
    return _MODEL


def _decode_16k_mono(path: Path) -> np.ndarray:
    """ffmpeg decode -> float32 mono @ 16 kHz in [-1, 1]."""
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not on PATH")
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed: {proc.stderr[:300].decode('utf-8', 'replace')}")
    raw = np.frombuffer(proc.stdout, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def _parse_tags(text: str) -> dict:
    """Split SenseVoice rich text into (emotion, events, lang, clean_text)."""
    emotion = None
    events: list[str] = []
    lang = None
    for tok in _TAG_RE.findall(text):
        if tok in EMOTION_TOKENS:
            emotion = tok
        elif tok in EVENT_TOKENS:
            if tok not in events:
                events.append(tok)
        elif tok in LANG_TOKENS:
            lang = tok
        # else: textnorm/unknown -> ignore
    clean = _TAG_RE.sub("", text).strip()
    return {"emotion": emotion, "events": events, "lang": lang, "text": clean}


def analyze_emotions(
    audio_path: Path,
    *,
    window_s: float = 15.0,
    max_seconds: float | None = None,
) -> dict:
    """Run SenseVoice over fixed windows and return a tag timeline + tallies.

    ``window_s``   — analysis granularity (one emotion/event verdict per window).
    ``max_seconds`` — cap analysis to the first N seconds (None = whole file).
    """
    samples = _decode_16k_mono(Path(audio_path))
    if max_seconds:
        samples = samples[: int(max_seconds * SAMPLE_RATE)]
    total_s = len(samples) / SAMPLE_RATE

    model = _load_model()
    win = int(window_s * SAMPLE_RATE)
    min_win = int(0.3 * SAMPLE_RATE)  # skip sub-300ms tail fragments

    timeline: list[dict] = []
    emotion_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}

    for start in range(0, len(samples), win):
        chunk = samples[start:start + win]
        if len(chunk) < min_win:
            break
        t_start = round(start / SAMPLE_RATE, 2)
        t_end = round((start + len(chunk)) / SAMPLE_RATE, 2)
        res = model.generate(
            input=chunk, fs=SAMPLE_RATE, cache={},
            language="auto", use_itn=False,
        )
        raw = res[0]["text"] if res else ""
        parsed = _parse_tags(raw)
        parsed["t_start"] = t_start
        parsed["t_end"] = t_end
        timeline.append(parsed)

        if parsed["emotion"]:
            emotion_counts[parsed["emotion"]] = emotion_counts.get(parsed["emotion"], 0) + 1
        for ev in parsed["events"]:
            event_counts[ev] = event_counts.get(ev, 0) + 1

    def _stamps(pred) -> list[float]:
        return [w["t_start"] for w in timeline if pred(w)]

    highlights = {
        "surprise": _stamps(lambda w: w["emotion"] == "SURPRISED"),
        "anger":    _stamps(lambda w: w["emotion"] == "ANGRY"),
        "laughter": _stamps(lambda w: "Laughter" in w["events"]),
        "applause": _stamps(lambda w: "Applause" in w["events"]),
        "bgm":      _stamps(lambda w: "BGM" in w["events"]),
    }

    return {
        "window_s": window_s,
        "analyzed_s": round(total_s, 1),
        "n_windows": len(timeline),
        "emotion_counts": emotion_counts,
        "event_counts": event_counts,
        "highlights": highlights,
        "timeline": timeline,
    }
