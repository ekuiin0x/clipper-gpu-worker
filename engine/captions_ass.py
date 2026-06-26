"""Generate libass .ass subtitle files + burn styled clips with one ffmpeg
call. Reproduces engine.styling's PIL word_only/phrase subtitles, hook, and
brand overlay, but as a single libass+overlay filtergraph (no per-frame PIL).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from fontTools import ttLib
from fontTools.varLib.instancer import instantiateVariableFont

from engine.encoder import video_encoder_args

if TYPE_CHECKING:
    from engine.styling import HookCaption, StylePreset, SubtitleMode, Subtitles


def ass_color(rgb: tuple[int, int, int]) -> str:
    """RGB → ASS &HAABBGGRR (AA=00 opaque)."""
    r, g, b = rgb
    return f"&H00{b:02X}{g:02X}{r:02X}"


def ass_ts(t: float) -> str:
    """Seconds → ASS H:MM:SS.cs (centiseconds), carrying on 100cs rounding."""
    if t < 0:
        t = 0.0
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60); t -= m * 60
    s = int(t); cs = int(round((t - s) * 100))
    if cs >= 100:
        s += 1; cs -= 100
    if s >= 60:
        m += 1; s -= 60
    if m >= 60:
        h += 1; m -= 60
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def instance_font(
    font_path: str, axes: tuple[int, ...] | None, fonts_dir: Path,
) -> tuple[str, str]:
    """Return (static_ttf_path, family_name) usable by libass.

    Variable fonts are instanced at the requested weight (axes[0]→wght) and
    written into ``fonts_dir``; static fonts are copied through. Cached by
    (font_path, axes) so repeated variants in a batch reuse one file.
    """
    fonts_dir.mkdir(parents=True, exist_ok=True)
    f = ttLib.TTFont(font_path)
    is_variable = "fvar" in f
    key = hashlib.md5(f"{font_path}|{axes}".encode()).hexdigest()[:10]
    dst = fonts_dir / f"face_{key}.ttf"
    if is_variable and axes:
        instantiateVariableFont(f, {"wght": float(axes[0])}, inplace=True)
    family = f["name"].getDebugName(1) or Path(font_path).stem
    if not dst.exists():
        f.save(str(dst))
    return str(dst), family


def _ass_escape(text: str) -> str:
    """Strip ASS override-block braces; collapse newlines to \\N."""
    return text.replace("{", "").replace("}", "").replace("\n", "\\N")


def subtitle_events(
    subs: Subtitles, preset: StylePreset, mode: SubtitleMode,
    *, y_pct: float, w: int, h: int,
) -> list[str]:
    """Dialogue lines for the `cap` style. word_only = one active word at a
    time (drawn in subtitle_highlight); phrase = one line per phrase."""
    from engine.styling import SubtitleMode
    cx = w // 2
    cy = int(h * y_pct)
    fade_ms = int(round(preset.subtitle_fade_s * 1000))
    up = preset.uppercase
    rows: list[str] = []
    for ph in subs.phrases:
        if mode == SubtitleMode.WORD_ONLY and ph.words:
            for word in ph.words:
                txt = (word.text or "").strip()
                if not txt:
                    continue
                txt = _ass_escape(txt.upper() if up else txt)
                rows.append(
                    f"Dialogue: 0,{ass_ts(word.t_start)},{ass_ts(word.t_end)},"
                    f"cap,,0,0,0,,{{\\an5\\pos({cx},{cy})\\fad({fade_ms},{fade_ms})}}{txt}"
                )
        else:  # PHRASE (or word_only with no word timings → whole phrase)
            txt = (ph.text or "").strip()
            if not txt:
                continue
            txt = _ass_escape(txt.upper() if up else txt)
            rows.append(
                f"Dialogue: 0,{ass_ts(ph.t_start)},{ass_ts(ph.t_end)},"
                f"cap,,0,0,0,,{{\\an5\\pos({cx},{cy})\\fad({fade_ms},{fade_ms})}}{txt}"
            )
    return rows


def hook_event(
    hook: HookCaption | None, preset: StylePreset,
    *, w: int, h: int, y_pct: float | None = None,
) -> str | None:
    """Dialogue line for the `hook` style. Hook is a headline at the top of
    the frame (anchor=top-center, \\an8) with fade-in/out (hook_fade_s) and
    entrance motion (pop/slide/pop_slide) over hook_anim_s.

    y_pct overrides preset.hook_y_pct when the caller pins a vertical position.
    Returns None if hook is None or has no text.
    """
    if hook is None or not (getattr(hook, "text", "") or "").strip():
        return None

    cx = w // 2
    y = int(h * (preset.hook_y_pct if y_pct is None else y_pct))
    fade_ms = int(round(preset.hook_fade_s * 1000))
    anim_ms = int(round(preset.hook_anim_s * 1000))

    txt = hook.text.upper() if preset.uppercase else hook.text
    txt = _ass_escape(txt)

    tags = [f"\\an8\\pos({cx},{y})", f"\\fad({fade_ms},{fade_ms})"]
    anim = (preset.hook_anim or "fade").lower()

    if anim in ("pop", "pop_slide"):
        # scale-in: start at 70%, ease to 100% over anim_ms (ASS \t lerps).
        tags.append(f"\\fscx70\\fscy70\\t(0,{anim_ms},\\fscx100\\fscy100)")

    if anim in ("slide", "pop_slide"):
        # rise: start ~half a line lower, move up into place.
        rise = int(round(preset.hook_font_size * 0.6))
        tags = [f"\\an8\\move({cx},{y + rise},{cx},{y},0,{anim_ms})",
                f"\\fad({fade_ms},{fade_ms})"] + tags[2:]

    return (f"Dialogue: 0,{ass_ts(hook.t_start)},{ass_ts(hook.t_end)},"
            f"hook,,0,0,0,,{{{''.join(tags)}}}{txt}")


def build_brand_png(
    width: int, height: int, dst: Path, *,
    is_watermark: bool = False, channel_name: str | None = None,
    logo: Path | None = None, brand_corner: str = "bottom-right",
    subtitle_font_path: str | None = None, logo_scale: float = 0.12,
) -> str | None:
    """Bake the brand mark into one full-frame RGBA PNG at dst. Returns the
    path, or None if there is nothing to brand. Reuses the PIL layer builders
    so the mark is byte-identical to the stylize_clip path.
    """
    from PIL import Image
    from engine.styling import _resolve_watermark, _resolve_channel_name, _resolve_logo

    layer = None
    if is_watermark:
        layer = _resolve_watermark(width, height, brand_corner)
    elif channel_name is not None:
        layer = _resolve_channel_name(
            channel_name, subtitle_font_path, width, height, brand_corner)
    elif logo is not None:
        img, pos = _resolve_logo(logo, width, height, brand_corner, logo_scale)
        layer = (img, pos) if img is not None else None

    if layer is None:
        return None

    img, pos = layer
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(img, pos)
    canvas.save(str(dst))
    return str(dst)


# ---------------------------------------------------------------------------
# ASS document assembly + one-call libass burn
# ---------------------------------------------------------------------------

_ASS_HEAD = (
    "[Script Info]\nScriptType: v4.00+\n"
    "PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\n"
    "ScaledBorderAndShadow: yes\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding\n"
    "{cap_style}\n{hook_style}\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n"
)


def _style_line(name, family, size, primary, outline_c, outline_w,
                border_style, back_c="&H00000000"):
    return (f"Style: {name},{family},{size},{primary},{primary},"
            f"{outline_c},{back_c},0,0,0,0,100,100,0,0,{border_style},"
            f"{outline_w},0,5,0,0,0,1")


def burn_styled_ffmpeg(
    *, src: Path, dst: Path, preset: StylePreset,
    subs: Subtitles | None, hook: HookCaption | None,
    subtitle_mode: str, work_dir: Path,
    brand_is_watermark: bool = False, brand_channel: str | None = None,
    brand_logo: Path | None = None, brand_corner: str = "bottom-right",
    subtitle_y_pct: float | None = None, hook_y_pct: float | None = None,
    out_w: int = 1080, out_h: int = 1920,
) -> str:
    """Burn hook + subtitles + brand onto src in ONE ffmpeg call.

    The .ass path is passed RELATIVE to work_dir (set as cwd on the subprocess)
    to avoid the Windows drive-colon filtergraph parse bug where 'C:\\...'
    triggers ffmpeg's option-separator parser inside a filtergraph string.

    Returns str(dst).
    """
    from engine.styling import SubtitleMode

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir = work_dir / "_fonts"

    # Instance (or copy-through) both fonts into a local fonts_dir.
    _, cap_fam = instance_font(
        preset.subtitle_font_path, preset.subtitle_font_axes, fonts_dir)
    _, hook_fam = instance_font(
        preset.hook_font_path, preset.hook_font_axes, fonts_dir)

    # Resolve y positions.
    sub_y = preset.subtitle_y_pct_default if subtitle_y_pct is None else subtitle_y_pct
    hk_y = preset.hook_y_pct if hook_y_pct is None else hook_y_pct

    mode = SubtitleMode(subtitle_mode)

    # Build ASS style lines.
    cap_primary = ass_color(
        preset.subtitle_highlight if mode == SubtitleMode.WORD_ONLY
        else preset.subtitle_fill
    )
    cap_style = _style_line(
        "cap", cap_fam, preset.subtitle_font_size, cap_primary,
        ass_color(preset.subtitle_outline), preset.subtitle_outline_width,
        border_style=1)

    hook_border = 3 if preset.hook_bg_color is not None else 1
    hook_back = ass_color(preset.hook_bg_color) if preset.hook_bg_color else "&H00000000"
    hook_style = _style_line(
        "hook", hook_fam, preset.hook_font_size, ass_color(preset.hook_fill),
        ass_color(preset.hook_outline), preset.hook_outline_width,
        border_style=hook_border, back_c=hook_back)

    # Collect event lines.
    events: list[str] = []
    if subs is not None:
        events += subtitle_events(subs, preset, mode, y_pct=sub_y, w=out_w, h=out_h)
    if hook is not None:
        hev = hook_event(hook, preset, w=out_w, h=out_h, y_pct=hk_y)
        if hev:
            events.append(hev)

    # Write the .ass document.
    ass_path = work_dir / "captions.ass"
    ass_path.write_text(
        _ASS_HEAD.format(w=out_w, h=out_h, cap_style=cap_style,
                         hook_style=hook_style)
        + "\n".join(events) + "\n",
        encoding="utf-8",
    )

    # Optional brand overlay PNG.
    brand_png = build_brand_png(
        out_w, out_h, work_dir / "brand.png",
        is_watermark=brand_is_watermark, channel_name=brand_channel,
        logo=brand_logo, brand_corner=brand_corner,
        subtitle_font_path=preset.subtitle_font_path)

    # Build filtergraph. Use paths relative to work_dir (the subprocess cwd)
    # so the Windows drive-colon never appears inside the filtergraph string.
    rel_ass = ass_path.name          # "captions.ass"
    rel_fonts = fonts_dir.name       # "_fonts"
    vf = f"ass={rel_ass}:fontsdir={rel_fonts}"

    inputs = ["-i", str(src.resolve())]
    if brand_png:
        inputs += ["-i", str(Path(brand_png).resolve())]
        fc = f"[0:v][1:v]overlay=0:0[bg];[bg]{vf}[v]"
        graph = ["-filter_complex", fc]
        maps = ["-map", "[v]", "-map", "0:a?"]
    else:
        graph = ["-vf", vf]
        maps = ["-map", "0:v", "-map", "0:a?"]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs, *graph, *maps,
        *video_encoder_args(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]),
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst.resolve()),
    ]
    proc = subprocess.run(
        cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(f"libass burn failed (rc={proc.returncode}): {tail}")
    return str(dst)
