from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

from engine.style_packs import PACKS  # 8-pack catalog, source of truth
from engine.v11_render import render_segment_ffmpeg, concat_and_mux

CANVAS_DIMS: dict[str, tuple[int, int]] = {
    "9x16": (1080, 1920),
    "3x4": (1080, 1440),
    "4x5": (1080, 1350),
}

DEFAULT_FORMAT = "9x16"


class FormatError(ValueError):
    pass


def parse_format(fmt: str | None) -> tuple[int, int]:
    if not fmt:
        return CANVAS_DIMS[DEFAULT_FORMAT]
    if fmt not in CANVAS_DIMS:
        allowed = ", ".join(sorted(CANVAS_DIMS))
        raise FormatError(f"Unknown format {fmt!r}; allowed: {allowed}")
    return CANVAS_DIMS[fmt]


# ---------------------------------------------------------------------------
# Style descriptor selection
# ---------------------------------------------------------------------------

# Ordered so the first 8 variants each use a distinct pack (max visual
# spread), then we cross with the second subtitle mode for variants 9-16.
_PACK_ORDER: list[str] = list(PACKS.keys())

# Live accepted tokens from engine.styling.SubtitleMode (str Enum):
#   "phrase"       - whole phrase fades in/out
#   "word"         - active word changes color + scale-pop
#   "word_bg"      - active word gets a colored rounded-rect background
#   "word_reveal"  - words appear one-by-one, retained after appearing
#   "word_only"    - ONLY the current word is shown, karaoke-swap
# All 8 packs default to "word_only". We pair it with "phrase" for the
# second subtitle mode so variants 9-16 use a distinct but valid render mode.
SUBTITLE_MODES: tuple[str, ...] = ("word_only", "phrase")

MIN_VARIANTS = 3
MAX_VARIANTS = 10


@dataclass(frozen=True)
class StyleDescriptor:
    """One variant's style: a pack name + subtitle render mode."""
    pack: str
    subtitle_mode: str


def _descriptor_catalog() -> list[StyleDescriptor]:
    # mode-major over packs: all packs in mode[0], then all packs in mode[1].
    out: list[StyleDescriptor] = []
    for mode in SUBTITLE_MODES:
        for pack in _PACK_ORDER:
            out.append(StyleDescriptor(pack=pack, subtitle_mode=mode))
    return out


def pick_style_descriptors(n: int) -> list[StyleDescriptor]:
    """Return ``n`` distinct (pack, subtitle_mode) descriptors. Deterministic.
    ``n`` must be in [MIN_VARIANTS, MAX_VARIANTS]."""
    if not (MIN_VARIANTS <= n <= MAX_VARIANTS):
        raise ValueError(f"n must be in [{MIN_VARIANTS},{MAX_VARIANTS}], got {n}")
    catalog = _descriptor_catalog()
    return catalog[:n]


# ---------------------------------------------------------------------------
# Panel-order permutation for multi-panel clip variants
# ---------------------------------------------------------------------------

def _orderings(n: int) -> list[tuple[int, ...]]:
    """All index orderings for ``n`` panels, with identity first so
    variant 0 keeps the planner's original top-to-bottom order."""
    if n <= 1:
        return [tuple(range(n))]
    identity = tuple(range(n))
    perms = [identity] + [p for p in permutations(range(n)) if p != identity]
    return perms


def permute_panels(plan: dict, variant_index: int) -> dict:
    """Return a deep-copied ``plan`` whose ``panels`` are reordered for
    ``variant_index``. No-op for compositions with <=1 panel. Orderings
    cycle (``variant_index`` is taken modulo the number of orderings)."""
    out = copy.deepcopy(plan)
    panels = out.get("panels") or []
    if len(panels) <= 1:
        return out
    orders = _orderings(len(panels))
    order = orders[variant_index % len(orders)]
    out["panels"] = [panels[i] for i in order]
    return out


def variant_orderkey(segments, variant_index: int) -> tuple:
    """Signature of the panel orderings ``variant_index`` produces across all
    segments. Two variants with the same key compose a byte-identical base, so
    the base can be rendered once and reused. Single-panel clips (the common
    case) collapse every variant to the same key — one base for the whole batch.
    """
    key: list[tuple[int, ...]] = []
    for seg in segments:
        panels = (seg.plan.get("panels") or [])
        n = len(panels)
        if n <= 1:
            key.append(tuple(range(n)))
        else:
            orders = _orderings(n)
            key.append(orders[variant_index % len(orders)])
    return tuple(key)


# ---------------------------------------------------------------------------
# Credit cost helpers (parity with bot.database.payments — no DB import)
# ---------------------------------------------------------------------------

def credits_for_short(duration_s: float) -> float:
    """Mirror bot.database.payments.credits_for_short: max(1, round(d/60,2)).
    Kept duplicated here so the pure module has no DB import."""
    return max(1.0, round(duration_s / 60.0, 2))


def clipmaker_credit_cost(*, duration_s: float, variants: int) -> float:
    """Total credits for a Clip Maker job: one short's cost per variant."""
    return round(variants * credits_for_short(duration_s), 2)


def render_variant_base(
    src: Path,
    segments,            # list[engine.vlm_video_direct.V11Segment]
    variant_index: int,
    *,
    out_w: int,
    out_h: int,
    fps: float,
    work_dir: Path,
    dst: Path,
) -> None:
    """Render one variant's composed base video (no captions) into ``dst``.

    For each planner segment we permute its panel order by ``variant_index``
    and render at the target canvas, then concat + mux the original audio.
    Mirrors the production render loop but adds the permutation + canvas params.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    for i, seg in enumerate(segments):
        permuted = permute_panels(seg.plan, variant_index)
        sp = work_dir / f"v{variant_index}_seg{i:03d}.mp4"
        render_segment_ffmpeg(
            src=src, t_start=seg.t_start, t_end=seg.t_end,
            plan=permuted, dst=sp, fps=fps, out_w=out_w, out_h=out_h,
        )
        seg_paths.append(sp)
    concat_and_mux(seg_paths, audio_src=src, dst=dst)
    for p in seg_paths:
        p.unlink(missing_ok=True)
