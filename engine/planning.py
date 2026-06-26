from __future__ import annotations
from dataclasses import dataclass
from .schema import (
    ClipPlan, Composition, PanelSpec, SegmentSpec,
    COMPOSITION_PANEL_GRID,
)


SAMPLE_SPACING_S = 2.0   # 1 frame every 2 seconds for Call A reference and Call B

# Below this duration, fall back to a single sample frame. plan_clip's
# `n == 1` branch then skips Call B+C entirely (one Call A only). Real
# composition changes inside a sub-8s window are rare — and even when one
# happens, the clip is too short to render two distinct segments anyway.
_SAMPLE_SINGLE_FRAME_THRESHOLD_S = 8.0

# Above this duration, stretch spacing so the total frame count caps at
# _SAMPLE_MAX_FRAMES. Below this duration we keep the SAMPLE_SPACING_S
# default (1 frame / 2s) — that's enough for Call B to spot a streamer
# toggling webcam in/out, switching cam-only -> gameplay, or any other
# real layout change in a 15-90s viral clip. The 2-second cadence halves
# Call B input tokens vs the original 1 Hz density, at the cost of cut
# precision: layout changes are flagged to within 2s rather than 1s,
# which is fine because the engine refines boundaries via histogram
# scan downstream.
_SAMPLE_FULL_DENSITY_THRESHOLD_S = 90.0
_SAMPLE_MAX_FRAMES = 60

# Panels wider than this aspect render with letterbox-fill (centered + blurred
# sides) instead of cover-resize. Avoids horizontal context bleed when the
# subject's natural bbox is much narrower than the panel — applies to webcam
# tiles (dest aspect ~1.875), commentator rows (~1.61), and the wider players
# bands in 3-panel chess/esports layouts (~2.25-2.56).
LETTERBOX_PANEL_ASPECT_THRESHOLD = 1.5

# Compositions whose panels must respect a hard horizontal boundary in the
# source (e.g. split_screen has a divider that the panels must not cross).
# These render with blur-fill so the bbox is used as-is and the cover-resize
# expansion can't bleed past the boundary.
HARD_BOUNDARY_COMPOSITIONS: frozenset[Composition] = frozenset({
    Composition.SPLIT_SCREEN,
})

# Roles whose VLM bbox tightly outlines a fixed inset region (gameplay area,
# browser viewport, webcam tile, chess board). For these, we must use the
# bbox as-is — derive_source_crop's aspect-expansion would bleed into
# adjacent UI/overlays/other inset regions. Compare with "expand-ok" roles
# like 'subject' or 'subjects' where expansion legitimately adds context.
HARD_BOUNDARY_ROLES: frozenset[str] = frozenset({
    "gameplay", "screencap", "browser", "webcam", "board",
    "players",       # esports/chess face-cam row — hard-bordered live inset.
                     # Without this, derive_source_crop expands the bbox to
                     # fill wide top-band panels (2.5:1) and bleeds into
                     # audience, standings, and adjacent broadcast UI.
    "commentators",  # commentator desk shot — bordered inset on most
                     # broadcasts. Same expansion-bleed risk as "players".
    "left", "right",  # split_screen halves; also covered by composition set
})

# For inset roles, switch from cover-resize to blur-fill when the bbox aspect
# diverges from the panel aspect AND the bbox is a small inset (covers less
# than INSET_AREA_FRACTION of the source). Both conditions together
# distinguish "small inset where edges matter" (chess board, score panel)
# from "large gameplay scene where center matters" (full-frame action).
# Cover-resize on a small inset would zoom + crop and lose edge content;
# cover-resize on a large scene fills the panel cleanly with the action
# centered.
BLUR_FILL_ASPECT_MISMATCH_THRESHOLD: float = 0.20
BLUR_FILL_INSET_AREA_FRACTION: float = 0.40


def panel_uses_letterbox_fill(dest_w: int, dest_h: int) -> bool:
    return (dest_w / max(1, dest_h)) > LETTERBOX_PANEL_ASPECT_THRESHOLD


def panel_uses_bbox_as_is(
    composition: Composition,
    dest_w: int,
    dest_h: int,
    role: str | None = None,
) -> bool:
    """Use the VLM bbox directly as the source crop, skipping the
    aspect-match expansion in derive_source_crop. Required to prevent bleed
    across split-screen dividers and into UI/overlays adjacent to inset
    panels (gameplay, webcam, browser, board)."""
    return (
        composition in HARD_BOUNDARY_COMPOSITIONS
        or (role is not None and role in HARD_BOUNDARY_ROLES)
        or panel_uses_letterbox_fill(dest_w, dest_h)
    )


def panel_uses_blur_fill(
    composition: Composition,
    dest_w: int,
    dest_h: int,
    role: str | None = None,
    src_aspect: float | None = None,
    bbox_area_fraction: float | None = None,
) -> bool:
    """Render with fit + blurred sides instead of cover-resize.

    Blur-fill kicks in for: (a) split-screen halves (HARD_BOUNDARY_COMPOSITIONS
    — bleed across the divider would be visible), (b) inset roles (gameplay,
    board, browser, screencap, webcam) when the bbox is a SMALL INSET
    (`bbox_area_fraction` < BLUR_FILL_INSET_AREA_FRACTION) AND its aspect
    mismatches the panel by more than BLUR_FILL_ASPECT_MISMATCH_THRESHOLD.
    Preserves edges of chess boards / score panels / icon-style content
    without forcing cover-resize off when the source is a full-frame
    gameplay scene that legitimately wants to fill the panel.

    Notably DOES NOT blur-fill for wide-aspect band panels (chess players
    row, commentators strip in a 3-panel broadcast) just because the
    panel itself is wider than tall. Cover-resize there fills the
    9:16 canvas width — the previous "panel aspect > 1.5 → blur-fill"
    rule made these renders look like 6:16 with black sides on the
    band, which the chess-broadcast users (t15) flagged."""
    if composition in HARD_BOUNDARY_COMPOSITIONS:
        return True
    if (
        role is not None
        and role in HARD_BOUNDARY_ROLES
        and src_aspect is not None
        and src_aspect > 0
        and bbox_area_fraction is not None
    ):
        dst_aspect = dest_w / max(1, dest_h)
        mismatch = abs(src_aspect / dst_aspect - 1.0)
        if (
            mismatch > BLUR_FILL_ASPECT_MISMATCH_THRESHOLD
            and bbox_area_fraction < BLUR_FILL_INSET_AREA_FRACTION
        ):
            return True
    return False


def sample_timestamps(duration_s: float, spacing_s: float = SAMPLE_SPACING_S) -> tuple[list[float], float]:
    """Pick frame timestamps for VLM Call A/B input. Returns (timestamps, spacing).

    Three regimes:
    * ``duration_s < _SAMPLE_SINGLE_FRAME_THRESHOLD_S`` (~8s): one frame at t=0.
      Triggers ``plan_clip``'s ``n == 1`` branch which skips Call B+C entirely.
      Sub-8s clips are too short for a second segment to be useful even if
      Call B did find a layout change.
    * ``duration_s <= _SAMPLE_FULL_DENSITY_THRESHOLD_S`` (~90s): keep the
      original 1-frame-per-second density. This is what Call B was designed
      for — spotting webcam toggles, cam-only ↔ gameplay switches, browser
      pop-ups, etc. inside a viral-clip-length window. Cheap on Ollama
      (a 60-frame batch is well within request limits) and the full
      density catches sub-second layout shifts that get baked into Call C
      boundary refinement.
    * Longer clips: stretch ``spacing`` so the count caps at
      ``_SAMPLE_MAX_FRAMES``. A 5-min source samples at 5s spacing instead
      of 1s — at that scale a layout typically holds for tens of seconds,
      so 5s sampling still detects every meaningful change.
    """
    if duration_s <= 0:
        return [0.0], spacing_s
    if duration_s < _SAMPLE_SINGLE_FRAME_THRESHOLD_S:
        return [0.0], duration_s
    if duration_s <= _SAMPLE_FULL_DENSITY_THRESHOLD_S:
        n = max(1, int(duration_s / spacing_s))
        return [round(i * spacing_s, 3) for i in range(n)], spacing_s
    # Long source: stretch spacing so we cap at _SAMPLE_MAX_FRAMES.
    n = _SAMPLE_MAX_FRAMES
    effective_spacing = duration_s / n
    return (
        [round(i * effective_spacing, 3) for i in range(n)],
        effective_spacing,
    )


def panel_dest_boxes(
    composition: Composition,
    out_w: int,
    out_h: int,
) -> list[tuple[int, int, int, int]]:
    """Stacked top-to-bottom dest panels for `composition`.

    Returns (x, y, w, h) per panel in output coords. Heights from
    COMPOSITION_PANEL_GRID; widths fill out_w.
    """
    grid = COMPOSITION_PANEL_GRID[composition]
    boxes: list[tuple[int, int, int, int]] = []
    y = 0
    remaining_h = out_h
    for i, ratio in enumerate(grid):
        if i == len(grid) - 1:
            h = remaining_h
        else:
            h = int(round(out_h * ratio))
            h = min(h, remaining_h - (len(grid) - i - 1))
        boxes.append((0, y, out_w, h))
        y += h
        remaining_h -= h
    return boxes


def derive_source_crop(
    bbox: tuple[int, int, int, int],
    src_w: int,
    src_h: int,
    dest_w: int,
    dest_h: int,
) -> tuple[int, int, int, int]:
    """Smallest crop CONTAINING bbox AND matching the dest panel aspect.

    Expands outward in one dimension to match panel aspect. If the expanded
    crop overflows the source, clamp to source dims (subject loses some
    context but stays visible). Position: centered on bbox, slid to fit.
    """
    if src_w <= 0 or src_h <= 0 or dest_w <= 0 or dest_h <= 0:
        raise ValueError("dimensions must be positive")
    bx, by, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise ValueError(f"bbox w,h must be positive, got ({bw},{bh})")

    panel_aspect = dest_w / dest_h
    bbox_aspect = bw / bh

    if bbox_aspect >= panel_aspect:
        crop_w = bw
        crop_h = int(round(crop_w / panel_aspect))
    else:
        crop_h = bh
        crop_w = int(round(crop_h * panel_aspect))

    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w / panel_aspect))
    if crop_h > src_h:
        crop_h = src_h
        crop_w = int(round(crop_h * panel_aspect))
    crop_w = max(1, min(crop_w, src_w))
    crop_h = max(1, min(crop_h, src_h))

    cx = bx + bw // 2
    cy = by + bh // 2
    x = cx - crop_w // 2
    y = cy - crop_h // 2
    x = max(0, min(src_w - crop_w, x))
    y = max(0, min(src_h - crop_h, y))
    return (x, y, crop_w, crop_h)


@dataclass(frozen=True)
class PanelLayout:
    role: str
    src_box: tuple[int, int, int, int]
    dest_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SegmentPlan:
    t_start: float
    t_end: float
    composition: Composition
    panels: list[PanelLayout]


def plan_segment(
    seg: SegmentSpec,
    spacing_s: float,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    clip_duration_s: float,
) -> SegmentPlan:
    """Resolve a SegmentSpec into a renderable SegmentPlan.

    Letterbox-render compositions get a single full-frame layout; the
    renderer ignores src_box/dest_box and blur-fills the whole source.
    Stacked compositions get per-panel src crops + dest boxes.
    """
    from .schema import COMPOSITION_RENDER_MODE, RenderMode

    grid_t_start = seg.frame_start * spacing_s
    grid_t_end = min(clip_duration_s, (seg.frame_end + 1) * spacing_s)
    t_start = seg.t_start if seg.t_start is not None else grid_t_start
    t_end = seg.t_end if seg.t_end is not None else grid_t_end
    t_end = min(t_end, clip_duration_s)

    if COMPOSITION_RENDER_MODE[seg.composition] == RenderMode.LETTERBOX:
        layouts = [PanelLayout(
            role=seg.panels[0].role,
            src_box=(0, 0, src_w, src_h),
            dest_box=(0, 0, out_w, out_h),
        )]
        return SegmentPlan(t_start=t_start, t_end=t_end, composition=seg.composition, panels=layouts)

    dest_boxes = panel_dest_boxes(seg.composition, out_w, out_h)
    if len(seg.panels) != len(dest_boxes):
        raise ValueError(
            f"segment has {len(seg.panels)} panels but composition "
            f"{seg.composition.value} expects {len(dest_boxes)}"
        )
    layouts: list[PanelLayout] = []
    for panel, dest in zip(seg.panels, dest_boxes):
        bx, by, bw, bh = panel.bbox
        bx = max(0, min(src_w - 1, bx))
        by = max(0, min(src_h - 1, by))
        bw = max(1, min(src_w - bx, bw))
        bh = max(1, min(src_h - by, bh))
        if panel_uses_bbox_as_is(seg.composition, dest[2], dest[3], role=panel.role):
            # Hard-boundary, inset role, or wide panel: keep tight bbox.
            # Skipping derive_source_crop's aspect-match expansion is what
            # prevents bleed across split-screen dividers and into UI/overlay
            # regions adjacent to gameplay/webcam insets. The renderer may
            # still cover-resize this bbox to fill the dest panel.
            src_box = (bx, by, bw, bh)
        else:
            src_box = derive_source_crop((bx, by, bw, bh), src_w, src_h, dest[2], dest[3])
        layouts.append(PanelLayout(role=panel.role, src_box=src_box, dest_box=dest))
    return SegmentPlan(t_start=t_start, t_end=t_end, composition=seg.composition, panels=layouts)


def segment_summary(plan: ClipPlan) -> list[tuple[str, int]]:
    """Compact (composition, n_frames) view — useful in logs."""
    return [(s.composition.value, s.n_frames) for s in plan.segments]
