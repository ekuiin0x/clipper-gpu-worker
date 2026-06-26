"""Auto-style (GPU-worker subset) — deterministic pack → StylePreset.

This is the slim render-only cut of the app's ``auto_style`` module. The
worker is handed an EXPLICIT style pack name per variant (the VLM pack-picker
lives in the private app repo), so everything VLM-related — OpenRouter calls,
frame extraction, the prompt, per-beat variants — has been stripped out.

What remains is the pure, side-effect-free translation:

  pack name + v11 plan  ->  StyleSpec  ->  engine StylePreset

  1. ``y_positions_from_plan`` derives subtitle / hook Y bands from the
     composition + per-panel geometry (faces / HUDs stay uncovered).
  2. ``pack_to_preset`` bakes a pack's font + colors + subtitle mode into
     the renderer's ``StylePreset``.
  3. ``preset_for_pack`` is the one-call entry the worker uses.
"""
from __future__ import annotations

from dataclasses import dataclass

from .style_packs import DEFAULT_PACK, get_pack, font_path_for_pack, font_axes_for_pack


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StyleSpec:
    pack: str = DEFAULT_PACK
    subtitle_y_pct: int = 85
    caption_y_pct: int = 8
    reasoning: str = ""
    model: str = ""
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Y positions derived from v11 plan
# ---------------------------------------------------------------------------

def y_positions_from_plan(plan: dict) -> tuple[int, int]:
    """Return ``(subtitle_y_pct, caption_y_pct)`` for a given v11 plan.

    Thin wrapper around ``engine.safe_zones`` — the actual geometry table
    lives there so the styling layer and any future debug UI pull from a
    single source of truth.
    """
    from .safe_zones import subtitle_y_pct, caption_y_pct
    comp = (plan or {}).get("composition", "")
    panels = (plan or {}).get("panels", []) or []
    return subtitle_y_pct(comp, panels), caption_y_pct(comp)


# ---------------------------------------------------------------------------
# Pack + Y → engine StylePreset
# ---------------------------------------------------------------------------

def pack_to_preset(
    spec: StyleSpec,
    *,
    hook_enabled: bool = True,
):
    """Translate a (pack_name, Y) StyleSpec into the engine StylePreset.

    word_bg subtitle mode wires the active word as white text on the
    pack's pill color. All other modes use the pack's highlight colour
    directly on the active word and have no pill.
    """
    from .styling import StylePreset

    pack = get_pack(spec.pack)
    font_path = font_path_for_pack(pack)
    font_axes = font_axes_for_pack(pack)

    if pack.subtitle_mode == "word_bg":
        subtitle_bg = pack.subtitle_bg or pack.subtitle_highlight
        subtitle_active_fill = (255, 255, 255)
    else:
        subtitle_bg = None
        subtitle_active_fill = pack.subtitle_highlight

    return StylePreset(
        name=f"auto_{pack.name}",
        hook_font_path=font_path,
        hook_font_size=pack.hook_font_size,
        hook_font_axes=font_axes,
        subtitle_font_path=font_path,
        subtitle_font_size=pack.subtitle_font_size,
        subtitle_font_axes=font_axes,
        hook_fill=pack.caption_fill if hook_enabled else (0, 0, 0),
        hook_outline=(0, 0, 0),
        # A pack with a caption_bg is a "card" hook (white-card podcast_clean /
        # cinema_aesthetic): the card supplies contrast, so drop the black text
        # outline the borderless colored-text packs rely on for legibility.
        hook_outline_width=(0 if pack.caption_bg is not None else 4) if hook_enabled else 0,
        hook_bg_color=pack.caption_bg if hook_enabled else None,
        hook_bg_padding=22,
        hook_bg_radius=36,
        hook_bg_alpha=pack.hook_bg_alpha,
        subtitle_fill=pack.subtitle_fill,
        subtitle_highlight=subtitle_active_fill,
        subtitle_outline=pack.subtitle_outline_color,
        subtitle_outline_width=pack.subtitle_outline_width,
        subtitle_bg_color=subtitle_bg,
        subtitle_bg_padding=14,
        subtitle_bg_radius=22,
        subtitle_bg_alpha=240,
        # Phrase-level bg. Default = pack.subtitle_phrase_bg if explicitly
        # set (legibility-shadow style). When the pack uses non-word_bg mode
        # AND defines subtitle_bg, treat subtitle_bg as the phrase-level pill
        # color (opaque) - that's how vlog_warm's white dialog pill and
        # esports_tactical's black commentary banner work.
        subtitle_phrase_bg_color=(
            pack.subtitle_bg if (pack.subtitle_mode != "word_bg" and pack.subtitle_bg is not None)
            else pack.subtitle_phrase_bg
        ),
        subtitle_phrase_bg_alpha=pack.subtitle_phrase_bg_alpha,
        subtitle_phrase_bg_padding=22,
        subtitle_phrase_bg_radius=18,
        hook_y_pct=spec.caption_y_pct / 100.0,
        hook_max_width_pct=0.78,
        subtitle_y_pct_default=spec.subtitle_y_pct / 100.0,
        subtitle_max_width_pct=0.80,
        word_pop_scale=(
            1.12 if pack.subtitle_mode in ("word_highlight", "word_bg") else 1.0
        ),
        uppercase=pack.uppercase,
        hook_anim=pack.hook_anim,
    )


def preset_for_pack(
    pack: str,
    *,
    plan: dict | None = None,
    hook_enabled: bool = True,
):
    """Build a StylePreset for an EXPLICIT ``pack`` (no VLM pick, no cost).

    Y positions come from the v11 ``plan``. Returns ``(StyleSpec, StylePreset)``.
    """
    sub_y, cap_y = y_positions_from_plan(plan or {})
    spec = StyleSpec(pack=pack, subtitle_y_pct=sub_y, caption_y_pct=cap_y)
    preset = pack_to_preset(spec, hook_enabled=hook_enabled)
    return spec, preset
