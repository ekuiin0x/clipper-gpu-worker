"""Post-pass styling: hook caption + subtitles + logo overlay.

Composites overlays via PIL onto each frame in a PyAV decode/encode loop.
No ASS, no libass -- pure Python.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import os
import re
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .schema import COMPOSITION_RENDER_MODE, Composition, RenderMode
from .planning import panel_dest_boxes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SubtitleMode(str, Enum):
    PHRASE = "phrase"          # whole phrase fades in/out
    WORD = "word"              # active word changes color + scale-pop
    WORD_BG = "word_bg"        # active word gets a colored rounded-rect background
    WORD_REVEAL = "word_reveal"  # words appear one-by-one, retained after appearing
    WORD_ONLY = "word_only"    # ONLY the current word is shown, replacing prev (karaoke)


@dataclass(frozen=True)
class HookCaption:
    text: str
    t_start: float
    t_end: float


@dataclass(frozen=True)
class TimedWord:
    text: str
    t_start: float
    t_end: float


@dataclass(frozen=True)
class TimedPhrase:
    text: str
    t_start: float
    t_end: float
    words: tuple[TimedWord, ...] = ()


@dataclass(frozen=True)
class Subtitles:
    phrases: tuple[TimedPhrase, ...]


# Whisper occasionally emits sub-100ms word durations on fast speech, which
# causes chaotic flicker in word-by-word highlighting (the active word
# changes faster than the eye can register, producing strobe-like color
# pops). Floor every word at 0.25s — short enough to feel responsive,
# long enough to be readable. Capped at the next word's start in
# ``_clamp_word_durations`` so we never overlap into the following word.
MIN_WORD_DUR = 0.25

# Whisper's word ``end`` timestamps frequently overshoot actual phonation
# — for word-final consonants and clauses ending in a long silence the
# segment-end gets attributed to the last word, producing a "hang" where
# a phrase stays on screen for hundreds of ms after the speaker stopped.
# Cap each word at a realistic max display duration derived from its
# character count + a small fade-out tail. ~13 chars/second is a fast but
# normal speaking rate; the tail buffer is the subtitle fade window so
# the word doesn't disappear abruptly mid-syllable.
_MAX_WORD_CHAR_DUR_S = 0.075     # per-character cap (~13 chars/s)
_MAX_WORD_TAIL_BUFFER_S = 0.20   # fade-out tail allowed past speech end
_MAX_WORD_FLOOR_S = 0.30         # absolute minimum cap (very short words)


def _clamp_word_durations(words: list[dict]) -> list[dict]:
    """Enforce ``MIN_WORD_DUR`` AND a realistic upper cap on each word's
    display duration, then ensure no word extends past the next word's
    start.

    Whisper overshoots word ``end`` timestamps during silences and clause
    boundaries, which causes the rendered subtitle to "hang" on the last
    word of a phrase. This cap pulls the visible end back to a
    char-count-derived natural maximum so the subtitle disappears around
    when the speaker actually stops.

    Operates on dicts of shape ``{"text", "t_start", "t_end"}``. Returns
    a new list (does not mutate input). Idempotent.
    """
    n = len(words)
    out: list[dict] = []
    for i, w in enumerate(words):
        t_start = float(w["t_start"])
        t_end = float(w["t_end"])
        text = str(w.get("text", "")).strip()
        # Upper cap: derive a "natural" max display end from char count,
        # then clamp t_end if Whisper overshot. Strip leading/trailing
        # whitespace before counting so " word " doesn't get inflated.
        natural_max = (
            t_start
            + max(_MAX_WORD_FLOOR_S, len(text) * _MAX_WORD_CHAR_DUR_S)
            + _MAX_WORD_TAIL_BUFFER_S
        )
        if t_end > natural_max:
            t_end = natural_max
        # Lower floor: MIN_WORD_DUR for legibility on fast speech.
        if t_end - t_start < MIN_WORD_DUR:
            t_end = t_start + MIN_WORD_DUR
        # Never bleed into the next word — cap at its start.
        if i + 1 < n:
            next_start = float(words[i + 1]["t_start"])
            if t_end > next_start:
                t_end = next_start
        # Degenerate case: next word starts before this word's start.
        if t_end < t_start:
            t_end = t_start
        out.append({**w, "t_start": t_start, "t_end": t_end})
    return out


def _censor_hook(hook: HookCaption | None) -> HookCaption | None:
    if hook is None:
        return None
    from .profanity import censor_text
    return HookCaption(text=censor_text(hook.text),
                       t_start=hook.t_start, t_end=hook.t_end)


def _censor_subtitles(subs: Subtitles | None) -> Subtitles | None:
    if subs is None:
        return None
    from .profanity import censor_text
    return Subtitles(phrases=tuple(
        TimedPhrase(
            text=censor_text(p.text), t_start=p.t_start, t_end=p.t_end,
            words=tuple(
                TimedWord(text=censor_text(w.text),
                          t_start=w.t_start, t_end=w.t_end)
                for w in p.words
            ),
        )
        for p in subs.phrases
    ))


@dataclass(frozen=True)
class StyleSegment:
    """Per-segment composition info passed to the styler so it can place
    overlays in zones that don't cover panel content."""
    t_start: float
    t_end: float
    composition: Composition


def load_subtitles_json(path: Path) -> Subtitles:
    data = json.loads(Path(path).read_text())
    phrases = []
    for p in data.get("phrases", []):
        raw_words = list(p.get("words", []))
        clamped = _clamp_word_durations(raw_words) if raw_words else []
        words = tuple(
            TimedWord(text=w["text"], t_start=float(w["t_start"]), t_end=float(w["t_end"]))
            for w in clamped
        )
        phrases.append(TimedPhrase(
            text=p["text"], t_start=float(p["t_start"]), t_end=float(p["t_end"]),
            words=words,
        ))
    return Subtitles(phrases=tuple(phrases))


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StylePreset:
    """Typography-only preset. Position is derived from composition at render time."""
    name: str
    hook_font_path: str
    hook_font_size: int
    subtitle_font_path: str
    subtitle_font_size: int
    hook_font_axes: tuple[int, ...] | None = None        # for variable fonts (e.g. (900,) = Black)
    subtitle_font_axes: tuple[int, ...] | None = None
    emoji_font_path: str | None = None
    # Hook
    hook_fill: tuple[int, int, int] = (255, 255, 255)
    hook_outline: tuple[int, int, int] = (0, 0, 0)
    hook_outline_width: int = 4
    hook_bg_color: tuple[int, int, int] | None = None
    hook_bg_padding: int = 26
    hook_bg_radius: int = 24
    hook_bg_alpha: int = 220
    # Subtitle
    subtitle_fill: tuple[int, int, int] = (255, 255, 255)
    subtitle_highlight: tuple[int, int, int] = (255, 230, 0)
    subtitle_outline: tuple[int, int, int] = (0, 0, 0)
    subtitle_outline_width: int = 4
    subtitle_bg_color: tuple[int, int, int] | None = None
    subtitle_bg_padding: int = 26
    subtitle_bg_radius: int = 24
    subtitle_bg_alpha: int = 220
    # PHRASE-level subtitle background: a rounded rect drawn behind the
    # full subtitle line (independent of the per-word pill). Used to give
    # legibility to thin fonts (Inter / Marker / Montserrat) that would
    # otherwise vanish into busy clip backgrounds. None = no phrase bg.
    subtitle_phrase_bg_color: tuple[int, int, int] | None = None
    subtitle_phrase_bg_alpha: int = 128         # ~50% opacity
    subtitle_phrase_bg_padding: int = 22
    subtitle_phrase_bg_radius: int = 18
    # Layout (only fallback values; per-segment layout overrides at render time)
    hook_y_pct: float = 0.05
    hook_max_width_pct: float = 0.86
    subtitle_y_pct_default: float = 0.85
    subtitle_max_width_pct: float = 0.85
    line_spacing: float = 1.10
    # Animation
    hook_fade_s: float = 0.30
    subtitle_fade_s: float = 0.12
    word_pop_scale: float = 1.15
    word_pop_dur_s: float = 0.15
    # Hook entrance motion, layered ON TOP of the fade-in. "fade" = plain
    # opacity ramp (the historical behaviour); "pop" = scale-in with a small
    # ease-out-back overshoot; "slide" = rise up into place; "pop_slide" =
    # both. Entrance plays over the first ``hook_anim_s`` seconds of the hook.
    hook_anim: str = "fade"
    hook_anim_s: float = 0.45
    # Render the subtitle phrase + hook in ALL CAPS regardless of source casing.
    # Used by the Hormozi / reaction_pop / comedy_pop visual identities — the
    # canonical viral look depends on uppercase. False = render as transcribed.
    uppercase: bool = False


def _scale_preset(p: "StylePreset", factor: float) -> "StylePreset":
    """Return a copy of ``p`` with every absolute-pixel field scaled by ``factor``.

    Presets are written assuming a 1080×1920 frame; ``stylize_clip`` calls
    this so a preset still looks right when rendered at e.g. 720×1280.
    Floats (pcts, fade times, animation curves) are NOT scaled — those are
    already resolution-agnostic. Only the pixel fields move.

    Font sizes are clamped to a minimum of 1 (PIL won't load font@0). All
    other pixel fields (outlines, padding, radius) are clamped at 0 — a
    preset with no outline must STAY at zero after scaling, not bump up
    to 1 (the 1-px line would visually appear as a faint halo).
    """
    from dataclasses import replace
    if abs(factor - 1.0) < 0.01:
        return p
    sf = lambda v: max(1, int(round(v * factor)))          # font: floor at 1
    sp = lambda v: max(0, int(round(v * factor)))          # padding/outline: floor at 0
    return replace(
        p,
        hook_font_size=sf(p.hook_font_size),
        hook_outline_width=sp(p.hook_outline_width),
        hook_bg_padding=sp(p.hook_bg_padding),
        hook_bg_radius=sp(p.hook_bg_radius),
        subtitle_font_size=sf(p.subtitle_font_size),
        subtitle_outline_width=sp(p.subtitle_outline_width),
        subtitle_bg_padding=sp(p.subtitle_bg_padding),
        subtitle_bg_radius=sp(p.subtitle_bg_radius),
        subtitle_phrase_bg_padding=sp(p.subtitle_phrase_bg_padding),
        subtitle_phrase_bg_radius=sp(p.subtitle_phrase_bg_radius),
    )


@dataclass(frozen=True)
class StyleScript:
    """A timeline mapping time ranges to StylePreset variants.

    Used by the per-beat dynamic styling path so the hook, setup, and
    payoff phases of a Short can each get a tuned look — e.g. the
    payoff swaps to a larger font so the punchline reads bigger. Built
    by ``auto_style.pick_style_script_for_video``; consumed by
    ``stylize_clip`` via the ``style_script=`` parameter.

    ``entries`` is expected to be sorted by ``t_start`` and cover
    ``[0, clip_duration)`` without gaps; ``active_at`` returns ``None``
    for times outside any range (the caller falls back to the base preset).
    Adjacent ranges may reference the SAME preset object — the font /
    layer caches in ``stylize_clip`` key on ``id(preset)``, so reuse hits
    the cache rather than re-rendering.
    """
    entries: tuple[tuple[float, float, "StylePreset"], ...]

    def active_at(self, t: float) -> "StylePreset | None":
        for t_start, t_end, preset in self.entries:
            if t_start <= t < t_end:
                return preset
        return None


_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


def _resolve_emoji_font() -> str | None:
    """Pick an emoji font that exists on the running OS AND scales properly
    in PIL.

    PIL's ``ImageFont.truetype`` rejects bitmap-only fonts (e.g. Noto Color
    Emoji ships at fixed 109/136px and raises "invalid pixel size" on
    other sizes), so we prefer vector fonts even if they're monochrome.
    Order:

      1. Segoe UI Emoji (Windows; vector + color)
      2. Apple Color Emoji (macOS; vector + color)
      3. Noto Emoji non-color (Linux; vector, monochrome)
      4. Symbola (Linux; vector, monochrome, good coverage)
      5. Noto Color Emoji (Linux; bitmap — last resort, may crash PIL)

    Returns None when nothing matches — the renderer treats that as "no
    emoji glyphs", which is fine for plain-ASCII hook/subtitle text.
    """
    candidates = [
        "C:/Windows/Fonts/seguiemj.ttf",                              # Windows
        "/System/Library/Fonts/Apple Color Emoji.ttc",                # macOS
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",       # Debian/Ubuntu (vector, mono)
        "/usr/share/fonts/noto/NotoEmoji-Regular.ttf",                # other Linux
        "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf", # Debian Symbola
        "/usr/share/fonts/truetype/symbola/Symbola.ttf",              # alt Symbola path
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",          # Debian (bitmap; last resort)
        "/usr/share/fonts/noto/NotoColorEmoji.ttf",                   # other Linux (bitmap)
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    return None


_WIN_EMOJI = _resolve_emoji_font()


def _resolve_cjk_font() -> str | None:
    """Pick a system CJK font for Chinese/Japanese/Korean transcripts.

    We don't bundle a CJK font (Noto Sans CJK is ~20 MB and pushes the
    install size up). When transcript contains CJK glyphs we fall back to
    whichever OS-installed font we can find. Returns None if nothing
    matches — caller decides what to do (likely keep the default font and
    accept tofu glyphs, which is what was happening before this fix).
    """
    candidates = [
        # Windows — YaHei is shipped with every modern build.
        "C:/Windows/Fonts/msyh.ttc",          # Microsoft YaHei (CJK)
        "C:/Windows/Fonts/msyhbd.ttc",        # Microsoft YaHei Bold
        "C:/Windows/Fonts/YuGothM.ttc",       # Yu Gothic Medium (JP)
        "C:/Windows/Fonts/malgun.ttf",        # Malgun Gothic (KR)
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    return None


_CJK_FONT = _resolve_cjk_font()
# Inter-VF covers Latin, Latin-Extended, Cyrillic, Greek, Vietnamese — broad
# enough for most non-CJK scripts that show up in livestream transcripts.
# CJK falls through to the OS font picked by _resolve_cjk_font() above.
_UNICODE_FONT = str(_FONT_DIR / "Inter-VF.ttf")
_UNICODE_FONT_AXES = (700,)  # Bold weight to roughly match Anton/Bebas thickness


def _f(name: str) -> str:
    return str(_FONT_DIR / name)


def text_needs_unicode_font(text: str) -> bool:
    """True when text has a codepoint outside basic ASCII + Latin-1.

    Catches Cyrillic, Greek, Latin Extended (accents), Vietnamese, etc.
    The check is intentionally loose — we'd rather false-positive into
    Inter (still looks fine) than miss a Cyrillic phrase and render tofu.
    """
    return any(ord(c) > 0x017F for c in text)


def text_needs_cjk_font(text: str) -> bool:
    """True when text has any CJK ideograph / Hiragana / Katakana / Hangul."""
    for c in text:
        cp = ord(c)
        if (
            0x3040 <= cp <= 0x30FF  # Hiragana + Katakana
            or 0x3400 <= cp <= 0x9FFF  # CJK Unified Ideographs (incl. ext A)
            or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
        ):
            return True
    return False


def _subtitle_text_blob(subs: "Subtitles | None") -> str:
    if subs is None:
        return ""
    return " ".join(p.text for p in subs.phrases)


def apply_unicode_font_fallback(
    preset: "StylePreset",
    subtitle_text: str = "",
    hook_text: str = "",
) -> "StylePreset":
    """Swap preset's font_path fields when the text contains glyphs the
    preset's bundled font can't render (most condensed display fonts —
    Anton, Bebas, Bangers — are Latin-only).

    Scans the subtitle blob + hook for non-ASCII / CJK and replaces:
      * subtitle_font_path → Inter-VF for non-ASCII, CJK font for CJK
      * hook_font_path → same logic, only if hook text needs it

    Idempotent — applying it twice returns the same preset. Returns the
    original preset unchanged when text is ASCII-only.
    """
    from dataclasses import replace
    sub_path = preset.subtitle_font_path
    sub_axes = preset.subtitle_font_axes
    hook_path = preset.hook_font_path
    hook_axes = preset.hook_font_axes
    changed = False
    if text_needs_cjk_font(subtitle_text) and _CJK_FONT:
        sub_path, sub_axes, changed = _CJK_FONT, None, True
    elif text_needs_unicode_font(subtitle_text):
        sub_path, sub_axes, changed = _UNICODE_FONT, _UNICODE_FONT_AXES, True
    if text_needs_cjk_font(hook_text) and _CJK_FONT:
        hook_path, hook_axes, changed = _CJK_FONT, None, True
    elif text_needs_unicode_font(hook_text):
        hook_path, hook_axes, changed = _UNICODE_FONT, _UNICODE_FONT_AXES, True
    if not changed:
        return preset
    return replace(
        preset,
        subtitle_font_path=sub_path,
        subtitle_font_axes=sub_axes,
        hook_font_path=hook_path,
        hook_font_axes=hook_axes,
    )


PRESETS: dict[str, StylePreset] = {
    # ── Tuned for a 1080×1920 reference frame. _scale_preset() shrinks
    #    every absolute-px field proportionally at render time when the
    #    output resolution is smaller (e.g. 720×1280 → ×0.667). All 8
    #    presets converge on a tight font-size band (sub ≈ 76-92, hook ≈
    #    104-128 at 1080p) so the only visual variable across them is the
    #    typeface and color palette — predictable customisation.

    # 1. Classic viral: Anton condensed + warm yellow / white pop.
    "anton_yellow": StylePreset(
        name="anton_yellow",
        hook_font_path=_f("Anton-Regular.ttf"),
        hook_font_size=128,
        subtitle_font_path=_f("Anton-Regular.ttf"),
        subtitle_font_size=92,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 232, 64),
        hook_outline=(0, 0, 0),
        hook_outline_width=10,
        subtitle_fill=(255, 232, 64),
        subtitle_highlight=(255, 255, 255),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=8,
        word_pop_scale=1.18,
    ),

    # 2. Clean modern: Bebas Neue, white with charcoal outline, warm-yellow word pop.
    "bebas_clean": StylePreset(
        name="bebas_clean",
        hook_font_path=_f("BebasNeue-Regular.ttf"),
        hook_font_size=120,
        subtitle_font_path=_f("BebasNeue-Regular.ttf"),
        subtitle_font_size=88,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(15, 15, 22),
        hook_outline_width=6,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 220, 70),
        subtitle_outline=(15, 15, 22),
        subtitle_outline_width=5,
        word_pop_scale=1.12,
    ),

    # 3. Gen-Z hot: Archivo Black, white text with hot-pink active word.
    "archivo_pink": StylePreset(
        name="archivo_pink",
        hook_font_path=_f("ArchivoBlack-Regular.ttf"),
        hook_font_size=104,
        subtitle_font_path=_f("ArchivoBlack-Regular.ttf"),
        subtitle_font_size=80,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(28, 0, 70),
        hook_outline_width=9,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 56, 142),
        subtitle_outline=(28, 0, 70),
        subtitle_outline_width=7,
        word_pop_scale=1.20,
    ),

    # 4. Cartoon pop: Bangers, white text with lime-green word pop, navy outline.
    "bangers_pop": StylePreset(
        name="bangers_pop",
        hook_font_path=_f("Bangers-Regular.ttf"),
        hook_font_size=128,
        subtitle_font_path=_f("Bangers-Regular.ttf"),
        subtitle_font_size=92,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(28, 0, 70),
        hook_outline_width=10,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(140, 255, 80),
        subtitle_outline=(28, 0, 70),
        subtitle_outline_width=8,
        word_pop_scale=1.22,
    ),

    # 5. Handwritten lifestyle: Permanent Marker on warm-cream pill background.
    "marker_pill": StylePreset(
        name="marker_pill",
        hook_font_path=_f("PermanentMarker-Regular.ttf"),
        hook_font_size=104,
        subtitle_font_path=_f("PermanentMarker-Regular.ttf"),
        subtitle_font_size=80,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(20, 30, 50),
        hook_outline=(0, 0, 0),
        hook_outline_width=0,
        hook_bg_color=(255, 240, 220),
        hook_bg_padding=30,
        hook_bg_radius=28,
        hook_bg_alpha=235,
        subtitle_fill=(20, 30, 50),
        subtitle_highlight=(220, 50, 80),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_bg_color=(255, 240, 220),
        subtitle_bg_padding=26,
        subtitle_bg_radius=24,
        subtitle_bg_alpha=235,
        word_pop_scale=1.14,
    ),

    # 6. Lime IRL pill: punchy Archivo on lime-green rounded background.
    "compact_lime": StylePreset(
        name="compact_lime",
        hook_font_path=_f("ArchivoBlack-Regular.ttf"),
        hook_font_size=96,
        subtitle_font_path=_f("ArchivoBlack-Regular.ttf"),
        subtitle_font_size=76,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=6,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 232, 80),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_bg_color=(132, 204, 64),
        subtitle_bg_padding=22,
        subtitle_bg_radius=20,
        subtitle_bg_alpha=240,
        word_pop_scale=1.14,
    ),

    # 7. Premium minimal: Inter Black, white text on translucent black pill.
    "isolated_white": StylePreset(
        name="isolated_white",
        hook_font_path=_f("Inter-VF.ttf"),
        hook_font_size=104,
        hook_font_axes=(900,),
        subtitle_font_path=_f("Inter-VF.ttf"),
        subtitle_font_size=78,
        subtitle_font_axes=(900,),
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=0,
        hook_bg_color=(10, 10, 18),
        hook_bg_padding=24,
        hook_bg_radius=16,
        hook_bg_alpha=220,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 92, 110),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_bg_color=(10, 10, 18),
        subtitle_bg_padding=20,
        subtitle_bg_radius=14,
        subtitle_bg_alpha=215,
        word_pop_scale=1.10,
    ),

    # 8. Sleek minimal: Montserrat Black, white text on black rounded pill, no outline.
    "montserrat_minimal": StylePreset(
        name="montserrat_minimal",
        hook_font_path=_f("Montserrat-VF.ttf"),
        hook_font_size=96,
        hook_font_axes=(900,),
        subtitle_font_path=_f("Montserrat-VF.ttf"),
        subtitle_font_size=76,
        subtitle_font_axes=(900,),
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=0,
        hook_bg_color=(0, 0, 0),
        hook_bg_padding=26,
        hook_bg_radius=18,
        hook_bg_alpha=220,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 220, 60),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_bg_color=(0, 0, 0),
        subtitle_bg_padding=22,
        subtitle_bg_radius=16,
        subtitle_bg_alpha=220,
        word_pop_scale=1.10,
    ),

    # 9. IRL Bungee warm: chunky Bungee with warm coral active-word.
    "bungee_warm": StylePreset(
        name="bungee_warm",
        hook_font_path=_f("Bungee-Regular.ttf"),
        hook_font_size=110,
        subtitle_font_path=_f("Bungee-Regular.ttf"),
        subtitle_font_size=82,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 240, 230),
        hook_outline=(60, 20, 0),
        hook_outline_width=8,
        subtitle_fill=(255, 240, 230),
        subtitle_highlight=(255, 100, 70),
        subtitle_outline=(60, 20, 0),
        subtitle_outline_width=6,
        word_pop_scale=1.16,
    ),

    # 10. Podcast Inter quote: Inter SemiBold with phrase-bg dark block for legibility.
    "inter_quote": StylePreset(
        name="inter_quote",
        hook_font_path=_f("Inter-VF.ttf"),
        hook_font_size=92,
        hook_font_axes=(700,),
        subtitle_font_path=_f("Inter-VF.ttf"),
        subtitle_font_size=72,
        subtitle_font_axes=(700,),
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=0,
        hook_bg_color=(30, 30, 38),
        hook_bg_padding=22,
        hook_bg_radius=12,
        hook_bg_alpha=230,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(120, 200, 255),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_phrase_bg_color=(30, 30, 38),
        subtitle_phrase_bg_alpha=200,
        subtitle_phrase_bg_padding=22,
        subtitle_phrase_bg_radius=14,
        word_pop_scale=1.08,
    ),

    # 11. Gameplay BlackOps neon: military stencil font with neon-cyan word pop.
    "blackops_neon": StylePreset(
        name="blackops_neon",
        hook_font_path=_f("BlackOpsOne-Regular.ttf"),
        hook_font_size=118,
        subtitle_font_path=_f("BlackOpsOne-Regular.ttf"),
        subtitle_font_size=86,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=10,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(40, 255, 200),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=8,
        word_pop_scale=1.18,
    ),

    # 12. Chess/Strategy Bebas navy: smaller Bebas on slim navy pill — keeps
    # board / overlay graphics legible without big subtitle blocks competing.
    "bebas_navy": StylePreset(
        name="bebas_navy",
        hook_font_path=_f("BebasNeue-Regular.ttf"),
        hook_font_size=92,
        subtitle_font_path=_f("BebasNeue-Regular.ttf"),
        subtitle_font_size=72,
        emoji_font_path=_WIN_EMOJI,
        hook_fill=(255, 255, 255),
        hook_outline=(0, 0, 0),
        hook_outline_width=0,
        hook_bg_color=(15, 25, 60),
        hook_bg_padding=18,
        hook_bg_radius=10,
        hook_bg_alpha=235,
        subtitle_fill=(255, 255, 255),
        subtitle_highlight=(255, 200, 80),
        subtitle_outline=(0, 0, 0),
        subtitle_outline_width=0,
        subtitle_bg_color=(15, 25, 60),
        subtitle_bg_padding=16,
        subtitle_bg_radius=10,
        subtitle_bg_alpha=235,
        word_pop_scale=1.10,
    ),
}


# ---------------------------------------------------------------------------
# Text layout helpers
# ---------------------------------------------------------------------------

_LAYOUT_DRAW = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

_MEASURE_CACHE: dict[tuple[int, str, int], tuple[int, int]] = {}


def _load_font(path: str, size: int, axes: tuple[int, ...] | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if axes:
        try:
            f.set_variation_by_axes(list(axes))
        except Exception:
            pass
    return f


def _load_emoji_font(path: str | None, size: int) -> "ImageFont.FreeTypeFont | None":
    """Tolerant emoji-font loader.

    Bitmap-only emoji fonts (Noto Color Emoji) raise OSError when loaded
    at arbitrary sizes; this wrapper swallows that and returns None so
    the renderer skips emoji rendering instead of crashing the whole job.
    Plain text hooks render fine without it.
    """
    if not path:
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        # Try once more at the font's native size (common with bitmap fonts).
        for native in (109, 136, 64, 96):
            try:
                return ImageFont.truetype(path, native)
            except Exception:
                continue
        return None


# Emoji codepoint ranges (covers common modern emojis + ZWJ joiner + variation selector).
_EMOJI_RE = re.compile(
    "[\U00002300-\U000023FF]"          # misc technical (clocks, hourglasses)
    "|[\U00002460-\U000024FF]"          # enclosed alphanumerics (1.,2.,...)
    "|[\U000025A0-\U000025FF]"          # geometric shapes (squares, triangles)
    "|[\U00002600-\U000027BF]"          # misc symbols + dingbats
    "|[\U00002900-\U0000297F]"          # supplemental arrows
    "|[\U00002B00-\U00002BFF]"          # misc symbols and arrows
    "|[\U0001F000-\U0001F2FF]"          # mahjong, dominoes, playing cards
    "|[\U0001F300-\U0001FAFF]"          # modern emojis
    "|[\U0001F600-\U0001F64F]"          # emoticons
    "|[‍️]"                   # ZWJ + variation selector
)


def _split_emoji_runs(text: str) -> list[tuple[str, str]]:
    """Return list of (kind, substring) where kind is 'text' or 'emoji'."""
    runs: list[tuple[str, str]] = []
    last = 0
    for m in _EMOJI_RE.finditer(text):
        if m.start() > last:
            runs.append(("text", text[last:m.start()]))
        runs.append(("emoji", m.group(0)))
        last = m.end()
    if last < len(text):
        runs.append(("text", text[last:]))
    if not runs:
        runs.append(("text", text))
    return runs


def _measure_run(s: str, font: ImageFont.FreeTypeFont, kind: str, sw: int) -> tuple[int, int]:
    return _measure(s, font, sw if kind == "text" else 0)


def _measure_runs_total(runs: list[tuple[str, str]],
                         text_font: ImageFont.FreeTypeFont,
                         emoji_font: ImageFont.FreeTypeFont | None,
                         sw: int) -> int:
    total = 0
    for kind, s in runs:
        if kind == "emoji" and emoji_font is not None:
            total += _measure_run(s, emoji_font, kind, sw)[0]
        else:
            total += _measure_run(s, text_font, "text", sw)[0]
    return total


EMOJI_SCALE = 0.78  # emoji rendered smaller than text so it visually fits the cap-height line


def _draw_runs(draw: ImageDraw.ImageDraw, runs: list[tuple[str, str]],
               x: int, y: int,
               text_font: ImageFont.FreeTypeFont,
               emoji_font: ImageFont.FreeTypeFont | None,
               fill: tuple[int, int, int, int],
               outline: tuple[int, int, int, int],
               sw: int) -> int:
    """Draw runs side-by-side starting at (x,y); return new x.

    Emoji is rendered with its own scaled-down emoji font (EMOJI_SCALE of
    text size) and dropped vertically to share the midline with the text,
    so it visually fits inside the cap-height of caps-heavy display fonts.
    """
    if emoji_font is not None:
        emoji_y_offset = int(text_font.size * (1 - EMOJI_SCALE) / 2)
    else:
        emoji_y_offset = 0
    for kind, s in runs:
        if kind == "emoji" and emoji_font is not None:
            draw.text((x, y + emoji_y_offset), s, font=emoji_font, embedded_color=True)
            x += _measure_run(s, emoji_font, kind, 0)[0]
        else:
            draw.text((x, y), s, font=text_font, fill=fill,
                      stroke_width=sw, stroke_fill=outline)
            x += _measure_run(s, text_font, "text", sw)[0]
    return x


def _measure(text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 0) -> tuple[int, int]:
    key = (id(font), text, stroke_width)
    cached = _MEASURE_CACHE.get(key)
    if cached is not None:
        return cached
    bbox = _LAYOUT_DRAW.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    val = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    _MEASURE_CACHE[key] = val
    return val


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int, stroke_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if _measure(trial, font, stroke_width)[0] <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _alpha_envelope(t: float, t_start: float, t_end: float, fade_s: float) -> float:
    if t < t_start or t >= t_end:
        return 0.0
    rel = t - t_start
    dur = t_end - t_start
    if rel < fade_s:
        return rel / fade_s
    if rel > dur - fade_s:
        return max(0.0, (dur - rel) / fade_s)
    return 1.0


def _ease_out_back(u: float) -> float:
    """Ease-out with a small overshoot past 1.0 (peaks ~1.10 near u≈0.7)."""
    c1 = 1.70158
    c3 = c1 + 1.0
    x = u - 1.0
    return 1.0 + c3 * x * x * x + c1 * x * x


def _ease_out_cubic(u: float) -> float:
    x = 1.0 - u
    return 1.0 - x * x * x


def _hook_entrance(anim: str, t: float, t_start: float, dur_s: float) -> tuple[float, float]:
    """Return ``(scale, dy_frac)`` for the hook entrance at time ``t``.

    ``scale`` multiplies the hook's drawn size about its center; ``dy_frac`` is
    a downward offset as a fraction of the hook's line height (positive = the
    text starts below its resting spot and rises into place). Both settle to
    ``(1.0, 0.0)`` once the entrance window has elapsed.
    """
    if not anim or anim == "fade" or dur_s <= 0:
        return 1.0, 0.0
    rel = t - t_start
    if rel <= 0:
        # Pre-roll: start at the entrance extreme so frame 0 isn't a jump.
        u = 0.0
    elif rel >= dur_s:
        return 1.0, 0.0
    else:
        u = rel / dur_s
    scale, dy = 1.0, 0.0
    if anim in ("pop", "pop_slide"):
        scale = 0.62 + (1.0 - 0.62) * _ease_out_back(u)
    if anim in ("slide", "pop_slide"):
        dy = (1.0 - _ease_out_cubic(u)) * 0.9
    return scale, dy


def _transform_layer_region(
    layer: Image.Image,
    bbox: tuple[int, int, int, int],
    scale: float,
    dy_px: int,
) -> Image.Image:
    """Scale (about its center) and vertically shift the ``bbox`` region of a
    full-canvas RGBA ``layer``. Returns a new full-canvas layer."""
    if abs(scale - 1.0) < 0.002 and abs(dy_px) < 1:
        return layer
    w, h = layer.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return layer
    sub = layer.crop((x0, y0, x1, y1))
    bw, bh = sub.size
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if abs(scale - 1.0) >= 0.002:
        nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
        sub = sub.resize((nw, nh), Image.LANCZOS)
        bw, bh = nw, nh
    px = int(round(cx - bw / 2.0))
    py = int(round(cy - bh / 2.0 + dy_px))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.paste(sub, (px, py), sub)
    return out


# ---------------------------------------------------------------------------
# Composition-aware layout
# ---------------------------------------------------------------------------

def subtitle_y_pct_for_composition(c: Composition, out_w: int, out_h: int) -> float:
    """Decide subtitle vertical center as a fraction of frame height.

    For LETTERBOX-render compositions (whole source on a blurred canvas),
    return 0.92 -- subtitles sit in the bottom blur band.

    For STACKED compositions, return the seam y-position between the
    bottom-most panel and the panel above it. For 1-panel compositions
    (no seam), return 0.85 (lower mid).
    """
    if COMPOSITION_RENDER_MODE[c] == RenderMode.LETTERBOX:
        return 0.92
    boxes = panel_dest_boxes(c, out_w, out_h)
    if len(boxes) <= 1:
        return 0.85
    # Bottom seam = top of the last panel = bottom of the second-to-last panel.
    seam_y = boxes[-2][1] + boxes[-2][3]
    return seam_y / out_h


def hook_y_pct_for_composition(c: Composition, out_w: int, out_h: int) -> float:
    """Hook caption vertical position. Always near the top of the frame."""
    return 0.05


def _active_segment(t: float, segs: list[StyleSegment]) -> StyleSegment | None:
    for s in segs:
        if s.t_start <= t < s.t_end:
            return s
    return None


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------

def _corner_pos(frame_w: int, frame_h: int, w: int, h: int,
                corner: str, margin_pct: float) -> tuple[int, int]:
    margin = int(frame_w * margin_pct)
    if corner == "top-left":
        return margin, margin
    if corner == "top-right":
        return frame_w - w - margin, margin
    if corner == "bottom-left":
        return margin, frame_h - h - margin
    if corner == "bottom-right":
        return frame_w - w - margin, frame_h - h - margin
    raise ValueError(f"unknown corner: {corner!r}")


def _resolve_logo(
    logo_path: Path,
    frame_w: int,
    frame_h: int,
    corner: str,
    scale: float,
    margin_pct: float = 0.04,
) -> tuple[Image.Image | None, tuple[int, int] | None]:
    """Load + place a brand kit logo. Returns ``(None, None)`` on any failure
    so the caller can skip the overlay rather than crash the whole render.

    Failure modes seen in the wild:
      - User deleted the file after creating the brand kit.
      - DB pointed at a path that never existed (legacy / migration drift).
      - File is corrupt or in an unsupported format.
    """
    import logging as _lg
    _log = _lg.getLogger(__name__)

    try:
        img = Image.open(str(logo_path)).convert("RGBA")
    except (FileNotFoundError, OSError, ValueError) as exc:
        _log.warning("brand kit logo load failed (path=%s): %s", logo_path, exc)
        return None, None

    target_w = max(1, int(frame_w * scale))
    aspect = img.size[1] / max(1, img.size[0])
    target_h = max(1, int(target_w * aspect))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    pos = _corner_pos(frame_w, frame_h, target_w, target_h, corner, margin_pct)
    return img, pos


def _resolve_channel_name(
    text: str,
    font_path: str,
    frame_w: int,
    frame_h: int,
    corner: str,
    font_size: int = 44,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: int = 195,
    margin_pct: float = 0.04,
) -> tuple[Image.Image, tuple[int, int]]:
    """Render @channel-name to a small RGBA pill; return (image, position)."""
    font = _load_font(font_path, font_size)
    bbox = _LAYOUT_DRAW.textbbox((0, 0), text, font=font, stroke_width=0)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = max(16, font_size // 3), max(10, font_size // 5)
    img_w, img_h = tw + 2 * pad_x, th + 2 * pad_y
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img_w - 1, img_h - 1), radius=img_h // 2,
                         fill=(*bg_color, bg_alpha))
    d.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font,
           fill=(*text_color, 255))
    pos = _corner_pos(frame_w, frame_h, img_w, img_h, corner, margin_pct)
    return img, pos


# Free-tier watermark constants. Locked styling — Bebas Neue with the "1"
# rendered in brand pink. Caller supplies frame size + corner only; everything
# else is fixed so the mark looks identical across every preset.
_WATERMARK_TEXT = "cl1pper.com"
_WATERMARK_ACCENT_IDX = 2          # the "1" in cl1pper.com
_WATERMARK_PINK = (255, 47, 138)
_WATERMARK_WHITE = (255, 255, 255)
_WATERMARK_PILL_BG = (0, 0, 0)
_WATERMARK_PILL_ALPHA = 175        # readable but not dominant
_WATERMARK_FONT_FILE = "BebasNeue-Regular.ttf"


def _resolve_watermark(
    frame_w: int,
    frame_h: int,
    corner: str = "bottom-right",
    margin_pct: float = 0.04,
) -> tuple[Image.Image, tuple[int, int]]:
    """Render the free-tier 'cl1pper.com' watermark pill: dark translucent
    rounded-rect, white Bebas Neue text, the '1' in brand pink. Fixed look —
    not user-customisable so the brand mark is consistent across all presets.
    """
    # Scale the font with frame height so 720x1280 ≈ 48px (which is what the
    # mockup the user picked was rendered at). Smaller resolutions get a
    # proportional shrink rather than a hard-coded oversized pill.
    font_size = max(24, int(frame_h * (48 / 1280)))
    pad_x = max(16, font_size // 2)
    pad_y = max(8, font_size // 5)
    radius = max(10, font_size // 3)

    font = _load_font(str(_FONT_DIR / _WATERMARK_FONT_FILE), font_size)
    bbox = _LAYOUT_DRAW.textbbox((0, 0), _WATERMARK_TEXT, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img_w, img_h = tw + 2 * pad_x, th + 2 * pad_y

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img_w - 1, img_h - 1), radius=radius,
                        fill=(*_WATERMARK_PILL_BG, _WATERMARK_PILL_ALPHA))

    text_x = pad_x - bbox[0]
    text_y = pad_y - bbox[1]
    # White full text first, then re-draw just the "1" in pink. Re-rendering
    # a single char at the kerned offset preserves the font's natural spacing
    # — drawing per-glyph would break kerning.
    d.text((text_x, text_y), _WATERMARK_TEXT, font=font,
           fill=(*_WATERMARK_WHITE, 255))
    prefix_w = d.textlength(_WATERMARK_TEXT[:_WATERMARK_ACCENT_IDX], font=font)
    d.text(
        (text_x + prefix_w, text_y),
        _WATERMARK_TEXT[_WATERMARK_ACCENT_IDX],
        font=font, fill=(*_WATERMARK_PINK, 255),
    )

    pos = _corner_pos(frame_w, frame_h, img_w, img_h, corner, margin_pct)
    return img, pos


# ---------------------------------------------------------------------------
# Frame layout: optional "window" mode — inset video on a dark canvas.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameLayout:
    kind: str = "fill"                                    # "fill" | "window"
    bg_color: tuple[int, int, int] = (0, 0, 0)
    inset_w_pct: float = 0.92
    inset_h_pct: float = 0.55
    inset_y_pct: float = 0.45                             # vertical center of the inset
    corner_radius: int = 28
    border_color: tuple[int, int, int] | None = None
    border_width: int = 0

    def hook_y_pct(self) -> float:
        if self.kind != "window":
            return 0.05
        inset_top = self.inset_y_pct - self.inset_h_pct / 2
        return max(0.02, inset_top / 2 - 0.04)

    def subtitle_y_pct(self) -> float:
        if self.kind != "window":
            return 0.85
        inset_bottom = self.inset_y_pct + self.inset_h_pct / 2
        return min(0.97, inset_bottom + (1.0 - inset_bottom) / 2)


def _apply_window_frame(canvas: Image.Image, layout: FrameLayout) -> Image.Image:
    """Inset `canvas` into a rounded-rect window on a dark background canvas."""
    w, h = canvas.size
    out = Image.new("RGBA", (w, h), (*layout.bg_color, 255))
    inset_w = max(1, int(w * layout.inset_w_pct))
    inset_h = max(1, int(h * layout.inset_h_pct))
    src_aspect = w / h
    dst_aspect = inset_w / inset_h
    if src_aspect > dst_aspect:
        new_h = inset_h
        new_w = max(inset_w, int(round(w * inset_h / h)))
    else:
        new_w = inset_w
        new_h = max(inset_h, int(round(h * inset_w / w)))
    resized = canvas.resize((new_w, new_h), Image.LANCZOS)
    x_off = max(0, (new_w - inset_w) // 2)
    y_off = max(0, (new_h - inset_h) // 2)
    cropped = resized.crop((x_off, y_off, x_off + inset_w, y_off + inset_h))
    mask = Image.new("L", (inset_w, inset_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, inset_w, inset_h), radius=layout.corner_radius, fill=255
    )
    px = (w - inset_w) // 2
    py = int(h * layout.inset_y_pct - inset_h / 2)
    out.paste(cropped, (px, py), mask)
    if layout.border_color is not None and layout.border_width > 0:
        bd = ImageDraw.Draw(out)
        bd.rounded_rectangle(
            (px, py, px + inset_w, py + inset_h),
            radius=layout.corner_radius,
            outline=(*layout.border_color, 255),
            width=layout.border_width,
        )
    return out


# ---------------------------------------------------------------------------
# Hook drawing
# ---------------------------------------------------------------------------

def _wrap_runs(text: str,
               text_font: ImageFont.FreeTypeFont,
               emoji_font: ImageFont.FreeTypeFont | None,
               max_w: int, sw: int) -> list[list[tuple[str, str]]]:
    """Greedy word-wrap across mixed text+emoji runs. Returns lines of run-lists."""
    words = text.split(" ")
    lines: list[list[tuple[str, str]]] = []
    cur_runs: list[tuple[str, str]] = []
    cur_text = ""
    for word in words:
        candidate = (cur_text + " " + word) if cur_text else word
        candidate_runs = _split_emoji_runs(candidate)
        candidate_w = _measure_runs_total(candidate_runs, text_font, emoji_font, sw)
        if cur_runs and candidate_w > max_w:
            lines.append(cur_runs)
            cur_text = word
            cur_runs = _split_emoji_runs(word)
        else:
            cur_text = candidate
            cur_runs = candidate_runs
    if cur_runs:
        lines.append(cur_runs)
    return lines


def _draw_hook(
    canvas: Image.Image,
    hook: HookCaption,
    font: ImageFont.FreeTypeFont,
    emoji_font: ImageFont.FreeTypeFont | None,
    style: StylePreset,
    t: float,
    y_pct: float,
) -> None:
    alpha = _alpha_envelope(t, hook.t_start, hook.t_end, style.hook_fade_s)
    if alpha <= 0.001:
        return
    w, h = canvas.size
    max_w = int(w * style.hook_max_width_pct)
    sw = style.hook_outline_width
    hook_text = hook.text.upper() if style.uppercase else hook.text
    lines = _wrap_runs(hook_text, font, emoji_font, max_w, sw)
    line_h = int(font.size * style.line_spacing)
    y0 = int(h * y_pct)

    # Render the hook layer at FULL alpha; fade is applied to the whole layer's
    # alpha channel below so color-emoji glyphs fade with the rest.
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fill_a = (*style.hook_fill, 255)
    outline_a = (*style.hook_outline, 255)

    line_rects: list[tuple[int, int, int]] = []  # (x, y, line_w)
    for i, runs in enumerate(lines):
        line_w = _measure_runs_total(runs, font, emoji_font, sw)
        x = (w - line_w) // 2
        y = y0 + i * line_h
        line_rects.append((x, y, line_w))

    pad_x = style.hook_bg_padding
    pad_y = max(12, int(style.hook_bg_padding * 0.7))
    if style.hook_bg_color is not None:
        color = (*style.hook_bg_color, style.hook_bg_alpha)
        for (x, y, lw) in line_rects:
            d.rounded_rectangle(
                (x - pad_x, y - pad_y, x + lw + pad_x, y + line_h + pad_y // 2),
                radius=style.hook_bg_radius, fill=color,
            )

    for i, runs in enumerate(lines):
        x, y, _ = line_rects[i]
        _draw_runs(d, runs, x, y, font, emoji_font, fill_a, outline_a, sw)

    # Entrance motion (pop / slide) is a geometric transform of the drawn
    # layer; the fade-in opacity ramp below is applied afterwards so the two
    # compose. Build the content bbox from the line rects (+ bg pad/outline).
    scale, dy_frac = _hook_entrance(style.hook_anim, t, hook.t_start,
                                    style.hook_anim_s)
    if abs(scale - 1.0) >= 0.002 or abs(dy_frac) >= 0.002:
        margin = max(pad_x, sw) + 4
        bx0 = min(x for (x, _y, _lw) in line_rects) - margin
        bx1 = max(x + lw for (x, _y, lw) in line_rects) + margin
        by0 = line_rects[0][1] - pad_y - sw - 4
        by1 = line_rects[-1][1] + line_h + pad_y // 2 + sw + 4
        # Pad the crop for overshoot so a scale>1 frame isn't clipped.
        grow = int(round(max(bx1 - bx0, by1 - by0) * 0.20))
        bbox = (bx0 - grow, by0 - grow, bx1 + grow, by1 + grow)
        layer = _transform_layer_region(layer, bbox, scale,
                                        int(round(dy_frac * line_h)))

    if alpha < 1.0:
        arr = np.array(layer)
        arr[..., 3] = (arr[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        layer = Image.fromarray(arr)
    canvas.alpha_composite(layer)


# ---------------------------------------------------------------------------
# Subtitle drawing (phrase + word modes)
# ---------------------------------------------------------------------------

def _active_phrase(subs: Subtitles, t: float) -> TimedPhrase | None:
    for p in subs.phrases:
        if p.t_start <= t < p.t_end:
            return p
    return None


def _draw_subtitle_bg(
    layer_draw: ImageDraw.ImageDraw,
    style: StylePreset,
    line_rects: list[tuple[int, int, int, int]],
    alpha: float,
) -> None:
    if style.subtitle_bg_color is None:
        return
    pad_x = style.subtitle_bg_padding
    pad_y = max(12, int(style.subtitle_bg_padding * 0.7))
    bg_a = int(round(style.subtitle_bg_alpha * alpha))
    color = (*style.subtitle_bg_color, bg_a)
    for (x, y, w, h) in line_rects:
        layer_draw.rounded_rectangle(
            (x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y // 2),
            radius=style.subtitle_bg_radius, fill=color,
        )


def _draw_phrase_mode(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    alpha: float,
    y_pct: float,
) -> None:
    w, h = canvas.size
    sw = style.subtitle_outline_width
    line = phrase.text   # single line; overflow if wider than frame.
    line_h = int(font.size * style.line_spacing)
    line_w, line_h_px = _measure(line, font, sw)
    x = (w - line_w) // 2
    y = int(h * y_pct - line_h / 2)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = int(round(255 * alpha))
    fill_a = (*style.subtitle_fill, a8)
    outline_a = (*style.subtitle_outline, a8)

    _draw_subtitle_bg(d, style, [(x, y, line_w, line_h_px)], alpha)

    d.text((x, y), line, font=font, fill=fill_a,
           stroke_width=sw, stroke_fill=outline_a)
    canvas.alpha_composite(layer)


def _word_pop_scale(word: TimedWord, t: float, peak: float, dur_s: float) -> float:
    """Quick up-and-down pop on the word's first dur_s seconds."""
    if peak <= 1.0 or dur_s <= 0 or t < word.t_start:
        return 1.0
    rel = t - word.t_start
    if rel >= dur_s:
        return 1.0
    half = dur_s / 2
    if rel <= half:
        u = rel / half
    else:
        u = 1.0 - (rel - half) / half
    return 1.0 + (peak - 1.0) * u


# Multi-word subtitle layouts never stack taller than this — short-form
# reads worst when the line block grows past two rows.
_MAX_SUBTITLE_LINES = 2


def _fit_subtitle_lines(
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    canvas_w: int,
) -> tuple[ImageFont.FreeTypeFont, list[list[int]]]:
    """Lay a multi-word phrase out so no line spills past the frame's safe
    width (``style.subtitle_max_width_pct``).

    Returns ``(font, lines)`` where ``lines`` is 1 or 2 lists of word
    indices. ``font`` may be a smaller reload of the input when even a
    balanced two-line wrap can't fit at the original size.

    Keep the text as big as possible — prefer, in order:
      1. one line at the original size,
      2. a balanced two-line wrap at the original size,
      3. shrink until the wrap fits (floored at ~60% so a marginal
         overflow beats unreadably tiny text).
    """
    n = len(phrase.words)
    if n == 0:
        return font, []
    texts = [w.text for w in phrase.words]
    max_w = int(canvas_w * style.subtitle_max_width_pct)
    sw = style.subtitle_outline_width

    def widths_for(f: ImageFont.FreeTypeFont) -> tuple[list[int], int]:
        space_w = _measure(" ", f, sw)[0]
        return [_measure(tx, f, sw)[0] for tx in texts], space_w

    def line_width(idxs: list[int], widths: list[int], space_w: int) -> int:
        if not idxs:
            return 0
        return sum(widths[i] for i in idxs) + space_w * (len(idxs) - 1)

    def balanced_split(widths: list[int], space_w: int) -> list[list[int]]:
        half = line_width(list(range(n)), widths, space_w) / 2
        acc, split = 0, 1
        for i in range(n):
            acc += widths[i] + (space_w if i > 0 else 0)
            if acc >= half:
                split = i + 1
                break
        split = max(1, min(n - 1, split))
        return [list(range(split)), list(range(split, n))]

    widths, space_w = widths_for(font)
    if line_width(list(range(n)), widths, space_w) <= max_w:
        return font, [list(range(n))]
    if n >= 2:
        lines = balanced_split(widths, space_w)
        if all(line_width(idxs, widths, space_w) <= max_w for idxs in lines):
            return font, lines

    # Shrink until the chosen layout fits (or we hit the readability floor).
    floor = max(28, int(font.size * 0.6))
    size = font.size
    fitted_font, lines = font, ([list(range(n))] if n < 2 else balanced_split(widths, space_w))
    while size > floor:
        size = max(floor, int(size * 0.92))
        fitted_font = _load_font(style.subtitle_font_path, size,
                                 style.subtitle_font_axes)
        widths, space_w = widths_for(fitted_font)
        lines = [list(range(n))] if n < 2 else balanced_split(widths, space_w)
        if all(line_width(idxs, widths, space_w) <= max_w for idxs in lines):
            break
    return fitted_font, lines


def _draw_phrase_bg_block(
    canvas: Image.Image,
    style: StylePreset,
    line_rects: list[tuple[int, int, int, int]],
    alpha: float,
) -> None:
    """One rounded translucent pill spanning all subtitle lines, sized to
    the widest fitted line. Multi-word modes call this with their own
    fitted geometry so the legibility pill tracks the wrapped/shrunk text
    instead of a stale full-width single line."""
    if style.subtitle_phrase_bg_color is None or not line_rects:
        return
    x0 = min(x for (x, _y, _w, _h) in line_rects)
    x1 = max(x + w for (x, _y, w, _h) in line_rects)
    y0 = min(y for (_x, y, _w, _h) in line_rects)
    y1 = max(y + h for (_x, y, _w, h) in line_rects)
    pad_x = style.subtitle_phrase_bg_padding
    pad_y = max(4, pad_x // 2)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = max(0, min(255, int(round(style.subtitle_phrase_bg_alpha * alpha))))
    color = (*style.subtitle_phrase_bg_color, a8)
    d.rounded_rectangle(
        (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y),
        radius=style.subtitle_phrase_bg_radius, fill=color,
    )
    canvas.alpha_composite(layer)


def _draw_word_mode(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    t: float,
    alpha: float,
    y_pct: float,
    lines: list[list[int]],
) -> None:
    if not phrase.words:
        _draw_phrase_mode(canvas, phrase, font, style, alpha, y_pct)
        return

    w, h = canvas.size
    sw = style.subtitle_outline_width
    space_w, _ = _measure(" ", font, sw)
    word_sizes = [_measure(word.text, font, sw) for word in phrase.words]
    word_widths = [s[0] for s in word_sizes]

    line_h = int(font.size * style.line_spacing)
    block_h = line_h * len(lines)
    y0 = int(h * y_pct - block_h / 2)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = int(round(255 * alpha))
    fill_a = (*style.subtitle_fill, a8)
    outline_a = (*style.subtitle_outline, a8)
    highlight_a = (*style.subtitle_highlight, a8)

    # Precompute line layout (left x for each word).
    line_layouts: list[tuple[int, int, list[tuple[int, int]]]] = []  # y, total_w, [(idx, x)]
    for line_i, idxs in enumerate(lines):
        total = sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1)
        x = (w - total) // 2
        y = y0 + line_i * line_h
        positions = []
        for k, i in enumerate(idxs):
            positions.append((i, x))
            x += word_widths[i] + (space_w if k < len(idxs) - 1 else 0)
        line_layouts.append((y, total, positions))

    # Fitted phrase-level legibility pill behind the whole block.
    line_rects = [(int((w - tw) // 2), y, tw, line_h) for (y, tw, _pos) in line_layouts]
    _draw_phrase_bg_block(canvas, style, line_rects, alpha)
    # Optional rounded-rect background per line.
    _draw_subtitle_bg(d, style, line_rects, alpha)

    # Draw each word, with pop scale on the active word.
    for line_i, (y, _tw, positions) in enumerate(line_layouts):
        for i, x in positions:
            word = phrase.words[i]
            is_active = word.t_start <= t < word.t_end
            color = highlight_a if is_active else fill_a
            scale = _word_pop_scale(word, t, style.word_pop_scale, style.word_pop_dur_s) if is_active else 1.0
            if scale != 1.0:
                # Render word to its own layer, scale, paste centered on its baseline.
                ww, wh = word_sizes[i]
                # Pad layer for the stroke so it doesn't get clipped.
                pad = sw + 2
                wlayer = Image.new("RGBA", (ww + pad * 2, wh + pad * 2), (0, 0, 0, 0))
                wd = ImageDraw.Draw(wlayer)
                wd.text((pad, pad), word.text, font=font, fill=color,
                        stroke_width=sw, stroke_fill=outline_a)
                new_w = max(1, int(round(wlayer.size[0] * scale)))
                new_h = max(1, int(round(wlayer.size[1] * scale)))
                wlayer = wlayer.resize((new_w, new_h), Image.LANCZOS)
                # Paste so that the word's center stays in the same place.
                cx = x + ww // 2
                cy = y + wh // 2
                paste_x = cx - new_w // 2
                paste_y = cy - new_h // 2
                layer.alpha_composite(wlayer, (paste_x, paste_y))
            else:
                d.text((x, y), word.text, font=font, fill=color,
                       stroke_width=sw, stroke_fill=outline_a)

    canvas.alpha_composite(layer)


def _draw_word_bg_mode(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    t: float,
    alpha: float,
    y_pct: float,
    lines: list[list[int]],
) -> None:
    if not phrase.words:
        _draw_phrase_mode(canvas, phrase, font, style, alpha, y_pct)
        return

    w, h = canvas.size
    sw = style.subtitle_outline_width
    space_w, _ = _measure(" ", font, sw)
    word_sizes = [_measure(word.text, font, sw) for word in phrase.words]
    word_widths = [s[0] for s in word_sizes]

    line_h = int(font.size * style.line_spacing)
    block_h = line_h * len(lines)
    y0 = int(h * y_pct - block_h / 2)

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = int(round(255 * alpha))
    fill_a = (*style.subtitle_fill, a8)
    outline_a = (*style.subtitle_outline, a8)

    line_layouts: list[tuple[int, list[tuple[int, int]]]] = []
    for line_i, idxs in enumerate(lines):
        total = sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1)
        x = (w - total) // 2
        y = y0 + line_i * line_h
        positions: list[tuple[int, int]] = []
        for k, i in enumerate(idxs):
            positions.append((i, x))
            x += word_widths[i] + (space_w if k < len(idxs) - 1 else 0)
        line_layouts.append((y, positions))

    # Fitted phrase-level legibility pill behind the whole block.
    _draw_phrase_bg_block(
        canvas, style,
        [(int((w - (sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1))) // 2),
          y, sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1), line_h)
         for (y, _), idxs in zip(line_layouts, lines)],
        alpha,
    )

    # Optional phrase-level bg (preset-defined, e.g. tiktok_box).
    if style.subtitle_bg_color is not None:
        rects = [(int((w - sum(word_widths[i] for i in idxs) - space_w * (len(idxs) - 1)) // 2),
                   y, sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1), line_h)
                  for (y, _), idxs in zip(line_layouts, lines)]
        _draw_subtitle_bg(d, style, rects, alpha)

    # Per-word: colored bg behind active word. Symmetric padding so the pill
    # looks balanced around the glyph.
    pad_x = max(28, sw + 16)
    pad_y = max(28, sw + 16)
    radius = max(18, pad_y - 6)
    for (y, positions), idxs in zip(line_layouts, lines):
        for (i, x) in positions:
            word = phrase.words[i]
            ww, wh = word_sizes[i]
            if word.t_start <= t < word.t_end:
                bg_a = int(round(230 * alpha))
                bg_color = (*style.subtitle_highlight, bg_a)
                d.rounded_rectangle(
                    (x - pad_x, y - pad_y, x + ww + pad_x, y + wh + pad_y),
                    radius=radius, fill=bg_color,
                )
            d.text((x, y), word.text, font=font, fill=fill_a,
                   stroke_width=sw, stroke_fill=outline_a)
    canvas.alpha_composite(layer)


def _draw_word_reveal_mode(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    t: float,
    alpha: float,
    y_pct: float,
    lines: list[list[int]],
) -> None:
    if not phrase.words:
        _draw_phrase_mode(canvas, phrase, font, style, alpha, y_pct)
        return
    w, h = canvas.size
    sw = style.subtitle_outline_width
    space_w, _ = _measure(" ", font, sw)
    word_sizes = [_measure(word.text, font, sw) for word in phrase.words]
    word_widths = [s[0] for s in word_sizes]

    line_h = int(font.size * style.line_spacing)
    block_h = line_h * len(lines)
    y0 = int(h * y_pct - block_h / 2)

    # Fitted phrase-level legibility pill behind the whole block.
    _draw_phrase_bg_block(
        canvas, style,
        [(int((w - (sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1))) // 2),
          y0 + li * line_h,
          sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1), line_h)
         for li, idxs in enumerate(lines)],
        alpha,
    )

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = int(round(255 * alpha))
    fill_a = (*style.subtitle_fill, a8)
    outline_a = (*style.subtitle_outline, a8)
    highlight_a = (*style.subtitle_highlight, a8)

    for line_i, idxs in enumerate(lines):
        total = sum(word_widths[i] for i in idxs) + space_w * (len(idxs) - 1)
        x = (w - total) // 2
        y = y0 + line_i * line_h
        for k, i in enumerate(idxs):
            word = phrase.words[i]
            if word.t_start > t:
                # Future word: skip.
                x += word_widths[i] + (space_w if k < len(idxs) - 1 else 0)
                continue
            is_active = word.t_start <= t < word.t_end
            color = highlight_a if is_active else fill_a
            d.text((x, y), word.text, font=font, fill=color,
                   stroke_width=sw, stroke_fill=outline_a)
            x += word_widths[i] + (space_w if k < len(idxs) - 1 else 0)
    canvas.alpha_composite(layer)


def _draw_subtitle_phrase_bg(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    alpha: float,
    y_pct: float,
    sized_to: str | None = None,
) -> None:
    """Draw a rounded translucent rect behind the subtitle line — if
    ``style.subtitle_phrase_bg_color`` is set. Sized to fit the full
    phrase (or, when ``sized_to`` is a single word, just that word).
    """
    if style.subtitle_phrase_bg_color is None:
        return
    text = sized_to if sized_to is not None else phrase.text
    if not text.strip():
        return
    sw = style.subtitle_outline_width
    tw, th = _measure(text, font, sw)
    line_h = int(font.size * style.line_spacing)
    w, h = canvas.size
    cy = int(h * y_pct - line_h / 2)
    pad_x = style.subtitle_phrase_bg_padding
    pad_y = max(4, pad_x // 2)
    cx = w // 2
    x0 = cx - tw // 2 - pad_x
    x1 = cx + tw // 2 + pad_x
    y0 = cy - pad_y
    y1 = cy + th + pad_y
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = max(0, min(255, int(round(style.subtitle_phrase_bg_alpha * alpha))))
    color = (*style.subtitle_phrase_bg_color, a8)
    d.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=style.subtitle_phrase_bg_radius,
        fill=color,
    )
    canvas.alpha_composite(layer)


def _draw_word_only_mode(
    canvas: Image.Image,
    phrase: TimedPhrase,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    t: float,
    alpha: float,
    y_pct: float,
) -> None:
    """Render ONLY the current active word, centered. Each word replaces
    the previous — single-word karaoke rapid-fire. Renders the active
    word in the subtitle_highlight color (the "pop" color), since it's
    always the focal word. Hides between-word gaps."""
    if not phrase.words:
        _draw_phrase_mode(canvas, phrase, font, style, alpha, y_pct)
        return
    active = None
    for word in phrase.words:
        if word.t_start <= t < word.t_end:
            active = word
            break
    if active is None or not active.text.strip():
        return  # mid-gap or empty word — show nothing
    w, h = canvas.size
    sw = style.subtitle_outline_width
    # Shrink an over-wide single word so it never spills past the safe edge
    # (long words like CONGRATULATIONS at 78px otherwise clip the frame).
    max_w = int(w * style.subtitle_max_width_pct)
    if _measure(active.text, font, sw)[0] > max_w:
        floor = max(28, int(font.size * 0.6))
        size = font.size
        while size > floor:
            size = max(floor, int(size * 0.92))
            font = _load_font(style.subtitle_font_path, size,
                              style.subtitle_font_axes)
            if _measure(active.text, font, sw)[0] <= max_w:
                break
    # Phrase bg is sized to the ACTIVE word in this mode (tight, follows
    # the karaoke beat) rather than to the whole phrase (which would be
    # a static pill behind a single word — looks weird).
    _draw_subtitle_phrase_bg(canvas, phrase, font, style, alpha, y_pct,
                              sized_to=active.text)
    tw, th = _measure(active.text, font, sw)
    line_h = int(font.size * style.line_spacing)
    cy = int(h * y_pct - line_h / 2)
    x = (w - tw) // 2
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a8 = int(round(255 * alpha))
    fill_a = (*style.subtitle_highlight, a8)
    outline_a = (*style.subtitle_outline, a8)
    d.text((x, cy), active.text, font=font, fill=fill_a,
           stroke_width=sw, stroke_fill=outline_a)
    canvas.alpha_composite(layer)


def _draw_subtitles(
    canvas: Image.Image,
    subs: Subtitles,
    font: ImageFont.FreeTypeFont,
    style: StylePreset,
    t: float,
    mode: SubtitleMode,
    y_pct: float,
) -> None:
    phrase = _active_phrase(subs, t)
    if phrase is None:
        return
    if style.uppercase:
        phrase = TimedPhrase(
            text=phrase.text.upper(),
            t_start=phrase.t_start,
            t_end=phrase.t_end,
            words=tuple(
                TimedWord(text=w.text.upper(), t_start=w.t_start, t_end=w.t_end)
                for w in phrase.words
            ),
        )
    alpha = _alpha_envelope(t, phrase.t_start, phrase.t_end, style.subtitle_fade_s)
    if alpha <= 0.001:
        return
    # PHRASE shares one phrase-wide legibility pill drawn here. WORD_ONLY
    # sizes its own per-word pill internally. The multi-word modes (WORD,
    # WORD_BG, WORD_REVEAL) draw their own pill AFTER fitting, so it tracks
    # the wrapped/shrunk block instead of a stale full-width single line.
    if mode == SubtitleMode.PHRASE:
        _draw_subtitle_phrase_bg(canvas, phrase, font, style, alpha, y_pct)
        _draw_phrase_mode(canvas, phrase, font, style, alpha, y_pct)
    elif mode == SubtitleMode.WORD_ONLY:
        _draw_word_only_mode(canvas, phrase, font, style, t, alpha, y_pct)
    else:
        fit_font, lines = _fit_subtitle_lines(phrase, font, style, canvas.size[0])
        if mode == SubtitleMode.WORD_BG:
            _draw_word_bg_mode(canvas, phrase, fit_font, style, t, alpha, y_pct, lines)
        elif mode == SubtitleMode.WORD_REVEAL:
            _draw_word_reveal_mode(canvas, phrase, fit_font, style, t, alpha, y_pct, lines)
        else:  # WORD
            _draw_word_mode(canvas, phrase, fit_font, style, t, alpha, y_pct, lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_preview_frame(
    src: Path,
    dst: Path,
    style: StylePreset,
    t: float,
    hook: HookCaption | None = None,
    subtitles: Subtitles | None = None,
    logo: Path | None = None,
    channel_name: str | None = None,
    subtitle_mode: SubtitleMode = SubtitleMode.WORD,
    brand_corner: str = "bottom-right",
    logo_scale: float = 0.12,
    channel_font_size: int = 44,
    channel_text_color: tuple[int, int, int] = (255, 255, 255),
    channel_bg_color: tuple[int, int, int] = (0, 0, 0),
    channel_bg_alpha: int = 195,
    segments: list[StyleSegment] | None = None,
    frame_layout: "FrameLayout | None" = None,
    is_watermark: bool = False,
) -> None:
    """Render a single preview frame at time `t` to `dst` (JPG).

    Lets you iterate on style choices fast without re-encoding a whole video.
    """
    from .frames import frame_at

    # Match stylize_clip's profanity masking so the preview stays
    # pixel-identical to the rendered video.
    hook = _censor_hook(hook)
    subtitles = _censor_subtitles(subtitles)

    # Load the frame first so we know the actual output dims, then scale
    # the preset to match — mirrors what stylize_clip does for the full
    # video pipeline so the still preview pixel-matches the rendered clip.
    rgb = frame_at(src, t)
    canvas = Image.fromarray(rgb).convert("RGBA")
    if frame_layout is not None and frame_layout.kind == "window":
        canvas = _apply_window_frame(canvas, frame_layout)
    width, height = canvas.size
    style = _scale_preset(style, max(0.5, min(1.5, height / 1920)))

    hook_font = _load_font(style.hook_font_path, style.hook_font_size, style.hook_font_axes)
    sub_font = _load_font(style.subtitle_font_path, style.subtitle_font_size, style.subtitle_font_axes)
    emoji_font = _load_emoji_font(
        style.emoji_font_path, int(style.hook_font_size * EMOJI_SCALE),
    )

    brand_layer: tuple[Image.Image, tuple[int, int]] | None = None
    if is_watermark:
        # Free-tier mark — fixed Bebas Neue + colored "1". Ignores the
        # channel_* knobs because the brand mark must look identical across
        # presets. Caller still controls corner placement.
        brand_layer = _resolve_watermark(width, height, brand_corner)
    elif channel_name is not None:
        brand_layer = _resolve_channel_name(
            channel_name, style.subtitle_font_path, width, height, brand_corner,
            font_size=channel_font_size,
            text_color=channel_text_color,
            bg_color=channel_bg_color,
            bg_alpha=channel_bg_alpha,
        )
    elif logo is not None:
        lr, pos = _resolve_logo(logo, width, height, brand_corner, logo_scale)
        # _resolve_logo returns (None, None) on missing/corrupt files so we
        # silently skip the overlay rather than crash the render.
        brand_layer = (lr, pos) if lr is not None else None

    # Position resolution: window layout takes priority over composition seam.
    # For segments (stacked compositions), prefer the auto-style value when
    # it was explicitly set - auto_style.y_positions_from_plan reads the ACTUAL
    # panel heights from the VLM plan (e.g. 62/38) so its seam-aware position
    # is more accurate than subtitle_y_pct_for_composition which uses the
    # Composition enum's DEFAULT ratios (always 70/30 for split_screen_2).
    # 0.85 is the sentinel "not explicitly set, please compute" value.
    AUTO_DEFAULT_SUB_Y = 0.85
    if frame_layout is not None and frame_layout.kind == "window":
        hook_y = frame_layout.hook_y_pct()
        sub_y = frame_layout.subtitle_y_pct()
    elif segments:
        seg = _active_segment(t, segments)
        if seg is not None:
            hook_y = hook_y_pct_for_composition(seg.composition, width, height)
            if abs(style.subtitle_y_pct_default - AUTO_DEFAULT_SUB_Y) > 0.005:
                sub_y = style.subtitle_y_pct_default
            else:
                sub_y = subtitle_y_pct_for_composition(seg.composition, width, height)
        else:
            hook_y = style.hook_y_pct
            sub_y = style.subtitle_y_pct_default
    else:
        hook_y = style.hook_y_pct
        sub_y = style.subtitle_y_pct_default

    if brand_layer is not None:
        lr, pos = brand_layer
        canvas.alpha_composite(lr, pos)
    if hook is not None:
        _draw_hook(canvas, hook, hook_font, emoji_font, style, t, hook_y)
    if subtitles is not None:
        _draw_subtitles(canvas, subtitles, sub_font, style, t, subtitle_mode, sub_y)

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(str(dst), "JPEG", quality=92)


def stylize_clip(
    src: Path,
    dst: Path,
    style: StylePreset,
    hook: HookCaption | None = None,
    subtitles: Subtitles | None = None,
    logo: Path | None = None,
    channel_name: str | None = None,
    subtitle_mode: SubtitleMode = SubtitleMode.PHRASE,
    brand_corner: str = "bottom-right",
    logo_scale: float = 0.12,
    channel_font_size: int = 44,
    channel_text_color: tuple[int, int, int] = (255, 255, 255),
    channel_bg_color: tuple[int, int, int] = (0, 0, 0),
    channel_bg_alpha: int = 195,
    segments: list[StyleSegment] | None = None,
    frame_layout: "FrameLayout | None" = None,
    on_progress: "callable | None" = None,
    is_watermark: bool = False,
    style_script: StyleScript | None = None,
) -> None:
    """Decode `src`, composite overlays per frame, encode + remux audio to `dst`.

    Mirrors `render_preview_frame` exactly so a still preview at any timestamp
    matches the rendered video at the same timestamp pixel-for-pixel.

    When ``style_script`` is provided, the per-frame loop picks the active
    preset from the script's ``active_at(t)`` lookup; ``style`` becomes the
    fallback for times outside any script range. Fonts are cached per
    ``id(preset)`` so adjacent ranges sharing the same preset don't reload.
    """
    # Mask profanity in everything we burn into pixels (YT ToS + viewer
    # respect). Done here — the single choke point every render path
    # passes through — so user-supplied hooks are covered too.
    hook = _censor_hook(hook)
    subtitles = _censor_subtitles(subtitles)

    # Probe output dims first so the style scaling below sees them.
    import av  # lazy: only the legacy PIL stamp needs PyAV; the libass
    # render path (captions_ass.burn_styled_ffmpeg) never imports it.
    src_av = av.open(str(src))
    in_v = src_av.streams.video[0]
    width, height = in_v.width, in_v.height

    # Resolution-aware scaling. Presets are calibrated for a 1080×1920
    # reference frame; when the engine renders at 720×1280 (default since
    # the perf pass), every absolute-px field (font size, outline width,
    # padding, radius) needs to shrink proportionally — otherwise the
    # 96 px subtitle font that looks correct at 1080p looks 50 % too big
    # at 720p. The scale factor is keyed off frame height so anamorphic
    # outputs still get sensible typography.
    _REF_H = 1920
    _scale = max(0.5, min(1.5, height / _REF_H))
    style = _scale_preset(style, _scale)
    if style_script is not None:
        style_script = StyleScript(entries=tuple(
            (t0, t1, _scale_preset(p, _scale))
            for t0, t1, p in style_script.entries
        ))

    # Font cache keyed by id(preset). The script may repeat presets across
    # ranges (e.g. SETUP appears multiple times if beats split it), and we
    # want one font load per distinct preset object.
    _font_cache: dict[tuple[int, str], object] = {}

    def _fonts_for(p: StylePreset):
        kh = (id(p), "hook")
        if kh not in _font_cache:
            _font_cache[kh] = _load_font(
                p.hook_font_path, p.hook_font_size, p.hook_font_axes,
            )
            _font_cache[(id(p), "sub")] = _load_font(
                p.subtitle_font_path, p.subtitle_font_size, p.subtitle_font_axes,
            )
            _font_cache[(id(p), "emoji")] = _load_emoji_font(
                p.emoji_font_path, int(p.hook_font_size * EMOJI_SCALE),
            )
        return (
            _font_cache[(id(p), "hook")],
            _font_cache[(id(p), "sub")],
            _font_cache[(id(p), "emoji")],
        )

    hook_font, sub_font, emoji_font = _fonts_for(style)

    def _active_preset(t: float) -> StylePreset:
        if style_script is not None:
            p = style_script.active_at(t)
            if p is not None:
                return p
        return style
    src_fps = float(in_v.average_rate or in_v.guessed_rate or 30)
    duration = float(in_v.duration * in_v.time_base) if in_v.duration else 0.0

    # Cap output framerate. Shorts are watched on phones — 30 fps reads
    # identical to 60 fps to the human eye on a 6" screen, but doubles
    # the per-frame work in this PyAV decode/encode loop. The source's
    # 60 fps frames still get sampled at full rate during decode; we
    # just keep every 2nd frame in the encode (decimation handled by
    # frame_skip below). Env-overridable for users who want to preserve
    # 60 fps for non-Shorts use.
    max_fps = float(os.getenv("ENGINE_MAX_OUTPUT_FPS", "30"))
    out_fps = min(src_fps, max_fps) if max_fps > 0 else src_fps
    # Skip ratio: 60 fps source + 30 fps output → keep 1 of every 2.
    frame_skip = max(1, int(round(src_fps / out_fps)))

    tmp_video = dst.with_suffix(".video.mp4")
    out = av.open(str(tmp_video), mode="w")
    out_stream = out.add_stream("h264", rate=int(round(out_fps)))
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "22", "preset": "ultrafast", "threads": "0"}

    brand_layer: tuple[Image.Image, tuple[int, int]] | None = None
    if is_watermark:
        # Free-tier mark — fixed look, ignores the channel_* customisation
        # knobs so the brand stamp is identical across every preset.
        brand_layer = _resolve_watermark(width, height, brand_corner)
    elif channel_name is not None:
        brand_layer = _resolve_channel_name(
            channel_name, style.subtitle_font_path, width, height, brand_corner,
            font_size=channel_font_size,
            text_color=channel_text_color,
            bg_color=channel_bg_color,
            bg_alpha=channel_bg_alpha,
        )
    elif logo is not None:
        lr, pos = _resolve_logo(logo, width, height, brand_corner, logo_scale)
        # Skip logo overlay entirely if the file is missing/corrupt rather
        # than crashing the whole encode loop.
        brand_layer = (lr, pos) if lr is not None else None

    use_window = frame_layout is not None and frame_layout.kind == "window"
    layout_cache: dict[Composition, tuple[float, float]] = {}

    AUTO_DEFAULT_SUB_Y = 0.85
    explicit_sub_y = abs(style.subtitle_y_pct_default - AUTO_DEFAULT_SUB_Y) > 0.005

    def layout_for(t: float) -> tuple[float, float]:
        if use_window:
            return frame_layout.hook_y_pct(), frame_layout.subtitle_y_pct()
        if segments:
            seg = _active_segment(t, segments)
            if seg is not None:
                if seg.composition not in layout_cache:
                    # Prefer auto-style's explicit Y (computed from actual VLM
                    # panel heights) over the composition default (fixed 70/30).
                    sub_y = (style.subtitle_y_pct_default if explicit_sub_y
                             else subtitle_y_pct_for_composition(seg.composition, width, height))
                    layout_cache[seg.composition] = (
                        hook_y_pct_for_composition(seg.composition, width, height),
                        sub_y,
                    )
                return layout_cache[seg.composition]
        return style.hook_y_pct, style.subtitle_y_pct_default

    # Cache the rendered subtitle layer per (phrase, active_word, alpha-bucket,
    # active-preset-id) so we don't redo PIL text rendering for every frame
    # within the same word — the preset id is part of the key so per-beat
    # variants don't collide.
    sub_layer_cache: dict[tuple, Image.Image] = {}

    def make_sub_layer(
        t_now: float, sub_y: float,
        active_style: StylePreset, active_sub_font,
    ) -> Image.Image | None:
        if subtitles is None:
            return None
        phrase = _active_phrase(subtitles, t_now)
        if phrase is None:
            return None
        alpha = _alpha_envelope(
            t_now, phrase.t_start, phrase.t_end, active_style.subtitle_fade_s,
        )
        if alpha <= 0.001:
            return None
        active_idx = -1
        in_pop = False
        for i, w in enumerate(phrase.words):
            if w.t_start <= t_now < w.t_end:
                active_idx = i
                if (subtitle_mode == SubtitleMode.WORD
                        and active_style.word_pop_scale > 1.0
                        and t_now < w.t_start + active_style.word_pop_dur_s):
                    in_pop = True
                break
        if in_pop:
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            _draw_subtitles(
                layer, subtitles, active_sub_font, active_style,
                t_now, subtitle_mode, sub_y,
            )
            return layer
        alpha_bucket = round(alpha * 20) / 20
        key = (
            id(phrase), active_idx, alpha_bucket,
            round(sub_y, 3), subtitle_mode, id(active_style),
        )
        cached = sub_layer_cache.get(key)
        if cached is not None:
            return cached
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        _draw_subtitles(
            layer, subtitles, active_sub_font, active_style,
            t_now, subtitle_mode, sub_y,
        )
        sub_layer_cache[key] = layer
        return layer

    try:
        # frame_skip handles framerate decimation: with src=60 fps and
        # out=30 fps, we keep every 2nd decoded frame and drop the rest
        # before the (expensive) PIL overlay + encode work. Saves ~50 %
        # wall time on 60 fps sources for free.
        _skip_counter = 0
        for frame in src_av.decode(in_v):
            if frame.pts is None:
                continue
            if frame_skip > 1:
                if _skip_counter % frame_skip != 0:
                    _skip_counter += 1
                    continue
                _skip_counter += 1
            t = float(frame.pts * in_v.time_base)
            # Decode directly to RGBA — skips an explicit Image.convert("RGBA").
            rgba = frame.to_ndarray(format="rgba")
            canvas = Image.fromarray(rgba, "RGBA")
            if use_window:
                canvas = _apply_window_frame(canvas, frame_layout)

            hook_y, sub_y = layout_for(t)
            cur_style = _active_preset(t)
            cur_hook_font, cur_sub_font, cur_emoji_font = _fonts_for(cur_style)

            if brand_layer is not None:
                lr, pos = brand_layer
                canvas.alpha_composite(lr, pos)
            if hook is not None:
                _draw_hook(
                    canvas, hook, cur_hook_font, cur_emoji_font,
                    cur_style, t, hook_y,
                )
            sub_layer = make_sub_layer(t, sub_y, cur_style, cur_sub_font)
            if sub_layer is not None:
                canvas.alpha_composite(sub_layer)

            # Hand RGBA straight to PyAV; its native reformatter does the
            # alpha->YUV conversion in one C-level pass (much faster than
            # numpy slice-copy + RGB->YUV).
            rgba_arr = np.asarray(canvas)
            out_frame = av.VideoFrame.from_ndarray(rgba_arr, format="rgba")
            out_frame = out_frame.reformat(format="yuv420p")
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
            if on_progress is not None and duration > 0:
                on_progress(min(1.0, t / duration))
    finally:
        for packet in out_stream.encode():
            out.mux(packet)
        out.close()
        src_av.close()

    # Use the shared audio-mux helper: probes the source codec and picks
    # `-c:a copy` for the common AAC case (no re-encode = ~20-40s saved per
    # styled clip). Falls back to AAC 192k on copy failure (rare opus-in-webm).
    from .composition import _audio_mux_args  # local import — avoid cycle
    audio_args = _audio_mux_args(src)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_video),
        "-i", str(src),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", *audio_args,
        "-shortest", "-movflags", "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Copy failed (rare codec mismatch in mp4 container) — retry with
        # explicit AAC re-encode. Slower but always works.
        fallback = [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-i", str(src),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(dst),
        ]
        subprocess.run(fallback, check=True, capture_output=True)
    tmp_video.unlink(missing_ok=True)
