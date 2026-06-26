"""Composition-aware subtitle / caption safe zones.

Single source of truth for the Y bands subtitles and captions are allowed to
land in, derived from the composition the plan describes. ``auto_style``
delegates to ``subtitle_y_pct`` and ``caption_y_pct`` so the geometry math
lives in one place; ``shrink_panels_to_avoid_safezone`` is the inverse pass
that nudges Call A / Call C panel bboxes away from the subtitle band when
they'd put a face right under the caption.

All percentages refer to the 9:16 OUTPUT canvas height. ``subtitle_y_pct``
returns the BOTTOM-aligned baseline; ``caption_y_pct`` returns the
TOP-aligned baseline. Both are what ``engine.styling`` expects to plug into
``StylePreset.subtitle_y_pct_default`` / ``hook_y_pct``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafeZone:
    """Allowed vertical bands for subtitle and hook caption, in % of canvas.

    Each band is a ``(top_pct, bottom_pct)`` tuple. The subtitle baseline
    lands at ``bottom_pct`` (bottom-aligned text). The caption baseline lands
    at ``top_pct`` (top-aligned text). The mid of each band is the "default"
    used when the composition doesn't carry extra structural cues.
    """
    sub_y_pct_band: tuple[int, int]   # subtitle text band (top, bottom)
    hook_y_pct_band: tuple[int, int]  # hook caption band (top, bottom)


# ---------------------------------------------------------------------------
# Per-composition table (mirrors the spec).
# ---------------------------------------------------------------------------

ZONES_BY_COMPOSITION: dict[str, SafeZone] = {
    "single_view":       SafeZone(sub_y_pct_band=(82, 94), hook_y_pct_band=(2, 12)),
    "letterbox_irl":     SafeZone(sub_y_pct_band=(82, 94), hook_y_pct_band=(2, 12)),
    # Subtitle is centred on the seam in split_screen_2. The band here is the
    # FALLBACK used when we can't compute the real seam from the panels.
    "split_screen_2":    SafeZone(sub_y_pct_band=(46, 54), hook_y_pct_band=(2, 8)),
    # 2 stacked panels + corner PIP — same seam-centred subtitle as split_2.
    "split_screen_2_plus_pip": SafeZone(sub_y_pct_band=(46, 54), hook_y_pct_band=(10, 18)),
    # 2-column horizontal split — single bottom subtitle.
    "split_screen_2_horizontal": SafeZone(sub_y_pct_band=(84, 94), hook_y_pct_band=(2, 8)),
    # PIP in a corner — keep subtitle at the bottom; push caption below the PIP.
    "webcam_overlay":    SafeZone(sub_y_pct_band=(84, 94), hook_y_pct_band=(10, 18)),
    # 3-panel — subtitle centred inside the middle panel.
    "split_screen_3":    SafeZone(sub_y_pct_band=(46, 54), hook_y_pct_band=(2, 8)),
}

_DEFAULT_ZONE = SafeZone(sub_y_pct_band=(82, 94), hook_y_pct_band=(2, 12))


# Hand-tuned default Y per composition. Always inside the safe band above —
# these are the values the renderer plugs into the preset by default. The
# band is the allowed range; the default is the sweet spot within it.
_SUB_Y_DEFAULTS: dict[str, int] = {
    "single_view":               86,
    "letterbox_irl":             90,
    "webcam_overlay":            86,
    "split_screen_2_horizontal": 92,
    # split_screen_2 and split_screen_3 are computed from the seam / middle
    # panel — see subtitle_y_pct.
}

_HOOK_Y_DEFAULTS: dict[str, int] = {
    "single_view":               8,
    "letterbox_irl":             6,
    "webcam_overlay":            14,  # below the corner PIP
    "split_screen_2":            4,
    "split_screen_2_plus_pip":   14,  # below the corner PIP
    "split_screen_2_horizontal": 4,
    "split_screen_3":            4,
}


# ---------------------------------------------------------------------------
# Public geometry API
# ---------------------------------------------------------------------------

def _zone_for(comp: str) -> SafeZone:
    return ZONES_BY_COMPOSITION.get(comp, _DEFAULT_ZONE)


def subtitle_y_pct(comp: str, panels: list[dict] | None = None) -> int:
    """Return the BOTTOM-aligned baseline percentage for the subtitle band.

    For seam-based compositions (split_screen_2, split_screen_3) the position
    depends on the real panel heights so the baseline sits where text actually
    centres on the seam. For everything else it's the bottom of the safe band.

    ``panels`` is the post-conversion list of panel dicts (each with ``boxes``
    in source pixels and ``height_pct``); when omitted, the table fallback is
    used.
    """
    zone = _zone_for(comp)

    if comp in ("split_screen_2", "split_screen_2_plus_pip"):
        seam_pct = _real_seam_pct_split_screen_2(panels) if panels else 50.0
        # Subtitle text rises UPWARD from its baseline. Push the baseline
        # ~3% below the seam so the visible glyphs straddle the line instead
        # of hovering in the bottom of the upper panel.
        SUB_TEXT_HALF_PCT = 3.0
        y = seam_pct + SUB_TEXT_HALF_PCT
        # Clamp into a sane band so a wildly-allocated panel set can't push
        # the subtitle off the canvas.
        y = max(30.0, min(92.0, y))
        return int(round(y))

    if comp == "split_screen_3":
        if panels and len(panels) >= 3:
            heights = [int(p.get("height_pct", 0) or 0) for p in panels]
            top_h = heights[0] or 33
            mid_h = heights[1] or 33
            return top_h + mid_h // 2
        return 50

    # Look up the hand-tuned default, clamped into the safe band so we never
    # silently drift outside the documented range.
    default = _SUB_Y_DEFAULTS.get(comp, zone.sub_y_pct_band[1])
    return _clamp(default, zone.sub_y_pct_band)


def caption_y_pct(comp: str) -> int:
    """Return the TOP-aligned baseline percentage for the hook caption.

    Composition is the only input — captions live in a fixed band per comp
    (the hook is short and we don't need per-panel geometry to place it).
    """
    zone = _zone_for(comp)
    default = _HOOK_Y_DEFAULTS.get(comp, (zone.hook_y_pct_band[0]
                                          + zone.hook_y_pct_band[1]) // 2)
    return _clamp(default, zone.hook_y_pct_band)


def _clamp(value: int, band: tuple[int, int]) -> int:
    top, bottom = band
    return max(top, min(bottom, value))


def _real_seam_pct_split_screen_2(panels: list[dict]) -> float:
    """Compute the actual seam Y for a 2-panel vertical split.

    ``_compute_heights`` adapts panel heights to each panel's bbox aspect
    ratio (a tall-narrow chess board grabs more height than its 50% hint),
    so the rendered seam isn't necessarily at the plan's ``height_pct``.
    Falls back to the plan's height_pct if the renderer can't be imported
    or the panels list is malformed.
    """
    try:
        from .v11_render import _compute_heights, OUT_W, OUT_H
        real_heights = _compute_heights(panels, OUT_H, OUT_W)
        return (real_heights[0] / max(1, OUT_H)) * 100.0
    except Exception:
        h0 = panels[0].get("height_pct") if panels else None
        return float(h0 or 50)


# ---------------------------------------------------------------------------
# Shrink pass — push panel bboxes away from the subtitle band
# ---------------------------------------------------------------------------

# Max amount we'll lop off a panel's source bbox to avoid the subtitle zone,
# expressed as a fraction of source-frame height. 8% matches the spec.
_MAX_SHRINK_FRAC = 0.08

# Threshold: if the panel's source bbox extends below this fraction of the
# source frame, its content is "near the bottom" — likely to land under the
# subtitle band when rendered into a slot that fills the canvas.
_BOTTOM_HEAVY_THRESHOLD = 0.85


def shrink_panels_to_avoid_safezone(
    plan: dict,
    src_w: int,
    src_h: int,
) -> tuple[dict, int]:
    """Reduce panel-bbox heights so content sits clear of the subtitle band.

    Conservative v1: only shrinks single-bbox panels whose composition puts
    the subtitle at the bottom of the canvas (``single_view``, ``letterbox_irl``,
    ``webcam_overlay``) and whose bbox extends into the bottom
    ``1 - _BOTTOM_HEAVY_THRESHOLD`` of the source frame. Seam-based
    compositions (split_screen_2 / split_screen_3) are left untouched in v1
    — the seam math already centres the subtitle correctly when the panels
    keep their declared heights.

    Returns ``(new_plan, n_shrinks)`` — the modified plan and a count for
    telemetry. The input plan is never mutated.
    """
    comp = plan.get("composition", "")
    if comp not in ("single_view", "letterbox_irl", "webcam_overlay"):
        return plan, 0
    if src_h <= 0:
        return plan, 0

    panels = list(plan.get("panels", []) or [])
    if not panels:
        return plan, 0

    max_shrink_px = int(round(_MAX_SHRINK_FRAC * src_h))
    if max_shrink_px <= 0:
        return plan, 0

    new_panels: list[dict] = []
    shrinks = 0
    for p in panels:
        boxes = list(p.get("boxes", []) or [])
        if len(boxes) != 1:
            new_panels.append(p)
            continue
        x, y, w, h = boxes[0]
        if h <= 0 or w <= 0:
            new_panels.append(p)
            continue
        bbox_bottom_frac = (y + h) / src_h
        if bbox_bottom_frac < _BOTTOM_HEAVY_THRESHOLD:
            # Bbox doesn't reach the bottom of the source — face isn't near
            # the subtitle band when mapped. Leave it alone.
            new_panels.append(p)
            continue
        # Shrink by max_shrink_px, but never below half the original height
        # (the existing _plan_has_pathological_bbox aspect-ratio check still
        # has the final say, but we don't want to introduce a thin strip
        # on our own).
        new_h = max(h // 2 + 1, h - max_shrink_px)
        if new_h == h:
            new_panels.append(p)
            continue
        new_p = dict(p)
        new_p["boxes"] = [(x, y, w, new_h)]
        new_panels.append(new_p)
        shrinks += 1

    if shrinks == 0:
        return plan, 0
    new_plan = dict(plan)
    new_plan["panels"] = new_panels
    return new_plan, shrinks
