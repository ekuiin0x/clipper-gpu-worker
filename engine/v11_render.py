"""ffmpeg filter_complex renderer for v11 plans.

Produces the same pixel layout as scripts/debug_gpt_layout_v11.py::render_plan
but does it in one ffmpeg subprocess per segment instead of per-frame Python
cv2 ops. ~10× faster, same quality (libx264 vs cv2.VideoWriter mp4v).

Supports: single_view, split_screen_2, split_screen_3.
PIP composition (split_screen_2_plus_pip) falls back to the Python renderer
because circular alpha overlay is awkward in pure ffmpeg filters.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.encoder import video_encoder_args

OUT_W, OUT_H = 1080, 1920
# Maximum fraction of a bbox's long axis we're willing to discard when
# cover-resizing (zoom-to-fill). When filling a panel would crop away MORE
# than this, we instead FIT the whole bbox (show 100% of the content) over a
# blurred cover backdrop of the same content. crop_loss = 1 - 1/mismatch, so
# 0.20 ⟺ aspect mismatch ~1.25. Wide browser/chat regions therefore
# fit-with-backdrop instead of losing their side panels; only near-matching
# aspects (≤20% content loss) zoom edge-to-edge.
MAX_COVER_CROP = 0.20

# Gaussian blur sigma for the backdrop layer behind fitted content. The
# backdrop is a SEPARATE cover-scaled+blurred copy; the sharp fitted
# foreground is overlaid on top with a hard edge, so there is no halo
# (unlike blurring a single padded image, which bleeds the bar/content
# seam — the failure mode the old solid-pad fill was working around).
BACKDROP_BLUR_SIGMA = 18

# Retained: sampled+lightened pad color, still used for any solid-fill
# callers and surfaced via panel "bg_color".
BG_LIGHTEN = 0.70

DEFAULT_BG_HEX = "FFFFFF"


def sample_bg_color(img: np.ndarray, bbox_px: tuple[int, int, int, int],
                     lighten: float = BG_LIGHTEN) -> str:
    """Sample the mean color of ``bbox_px`` edge bands in ``img`` (BGR),
    mix it ``lighten`` of the way toward white, return ``RRGGBB`` (no #).

    Used as the panel pad-fill color so unfilled space looks like a
    soft tinted extension of the clip rather than a heavy blur.
    """
    H, W = img.shape[:2]
    x, y, w, h = bbox_px
    x0 = max(0, min(W - 1, int(x))); y0 = max(0, min(H - 1, int(y)))
    x1 = max(x0 + 1, min(W, int(x + w)))
    y1 = max(y0 + 1, min(H, int(y + h)))
    band = max(2, min(20, (x1 - x0) // 12, (y1 - y0) // 12))
    strips = []
    if y0 + band <= H: strips.append(img[y0:y0 + band, x0:x1])
    if y1 - band >= 0: strips.append(img[y1 - band:y1, x0:x1])
    if x0 + band <= W: strips.append(img[y0:y1, x0:x0 + band])
    if x1 - band >= 0: strips.append(img[y0:y1, x1 - band:x1])
    if not strips:
        return DEFAULT_BG_HEX
    flat = np.concatenate([s.reshape(-1, 3) for s in strips])
    b, g, r = flat.mean(axis=0).tolist()
    lighten = max(0.0, min(1.0, float(lighten)))
    r = r * (1 - lighten) + 255 * lighten
    g = g * (1 - lighten) + 255 * lighten
    b = b * (1 - lighten) + 255 * lighten
    return f"{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"


@dataclass(frozen=True)
class _Panel:
    boxes: list[tuple[int, int, int, int]]
    height_pct: int


def _compute_heights(panels, out_h: int, out_w: int) -> list[int]:
    """Adaptive panel heights from bbox aspects + per-panel area cap.

    Without the area cap, a small face-cam bbox (e.g. 200x250 in 1080-wide
    source) gets allocated a huge 1080-wide panel where it must upscale ~5×
    — the face fills the screen pixelated/giant. Capping panel area at
    MAX_AREA_UPSCALE × bbox area keeps the upscale ratio bounded so faces
    stay at a natural size.
    """
    # Linear upscale ~3× = area 9×. Acceptable visual quality bound.
    MAX_AREA_UPSCALE = 9.0

    ideal_heights = []
    max_heights = []
    for p in panels:
        n_boxes = len(p["boxes"])
        slot_w = out_w / max(1, n_boxes)
        aspects = [(bw / bh) if bh > 0 else 1.0 for _, _, bw, bh in p["boxes"]]
        aspects.sort()
        mid = aspects[len(aspects) // 2] if aspects else 1.0
        ideal_h = slot_w / mid if mid > 0 else slot_w
        hint = p["height_pct"] / 100.0
        ideal_pct = ideal_h / out_h
        blended = ideal_pct * 0.8 + hint * 0.2
        ideal_heights.append(blended * out_h)

        # Per-slot area cap: panel_area = slot_w × ph. We want
        #   slot_w × ph <= MAX_AREA_UPSCALE × bbox_w × bbox_h
        # taking the SMALLEST bbox in this panel (the binding constraint).
        slot_min_area = min((bw * bh) for _, _, bw, bh in p["boxes"])
        max_h = (MAX_AREA_UPSCALE * slot_min_area) / max(1, slot_w)
        max_heights.append(max_h)

    capped = [min(h, mh) for h, mh in zip(ideal_heights, max_heights)]
    total = sum(capped) or 1.0
    # If caps prevented filling out_h, allow scale-up (effectively relaxing
    # the cap for under-constrained plans like single-panel single_view).
    scale = out_h / total
    scaled = [max(2, (int(round(h * scale)) // 2) * 2) for h in capped]
    diff = out_h - sum(scaled)
    scaled[-1] = max(2, scaled[-1] + diff)
    return scaled


def _fit_blurred_backdrop(in_label: str,
                          box: tuple[int, int, int, int] | None,
                          dst_w: int, dst_h: int, out_label: str,
                          setsar: bool = False) -> str:
    """Fit 100% of ``box`` into ``dst_w``×``dst_h`` over a blurred
    cover-scaled backdrop of the SAME content. ``box=None`` means use the
    whole input frame (no crop).

    Two independent layers:
      - backdrop: (crop bbox →) scale to COVER dst (fill, center-crop
        overflow) → gaussian blur. Fills the whole slot, no bars.
      - foreground: (crop bbox →) scale to FIT dst (contain, no crop) →
        overlaid centered. Shows every edge of the content, sharp.
    The foreground's border is a hard edge against the blurred backdrop,
    so there is no halo (the seam is never blurred across).
    """
    ol = out_label
    tail = ",setsar=1" if setsar else ""
    if box is None:
        crop = ""
    else:
        bx, by, bw, bh = box
        bw = max(1, bw); bh = max(1, bh)
        crop = f"crop={bw}:{bh}:{bx}:{by},"
    return (
        f"[{in_label}]split=2[{ol}_bg][{ol}_fg];"
        f"[{ol}_bg]{crop}"
        f"scale={dst_w}:{dst_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={dst_w}:{dst_h},gblur=sigma={BACKDROP_BLUR_SIGMA}[{ol}_bgb];"
        f"[{ol}_fg]{crop}"
        f"scale={dst_w}:{dst_h}:force_original_aspect_ratio=decrease:flags=lanczos[{ol}_fgf];"
        f"[{ol}_bgb][{ol}_fgf]overlay=(W-w)/2:(H-h)/2{tail}[{out_label}]"
    )


def _slot_filter(in_label: str, box: tuple[int, int, int, int],
                 slot_w: int, slot_h: int, out_label: str,
                 bg_hex: str = DEFAULT_BG_HEX) -> str:
    """One source bbox → one output panel slot.

    Picks fit-with-blurred-backdrop (shows 100% of the bbox) vs
    cover-resize (zoom to fill, center-crop overflow) based on three
    signals:
      1. Cover-resize would crop away more than MAX_COVER_CROP of the
         long axis (e.g. a wide browser/chat region into a square-ish
         panel) → fit. Cropping would lose the side content.
      2. Bbox is near-square (chess boards, Wordle grids, scoreboards) →
         fit. Cover-resize would clip edge files/ranks/letters.
      3. Bbox area is much smaller than the panel slot → fit.
         Cover-resize would zoom heavily into a small bbox, going blocky.
    Otherwise cover-resize so near-matching content fills edge-to-edge.

    ``bg_hex`` is retained for call-site compatibility but unused on the
    fit path now that leftover space is a blurred extension of the content
    rather than a solid pad.
    """
    bx, by, bw, bh = box
    bw = max(1, bw); bh = max(1, bh)
    src_aspect = bw / bh
    dst_aspect = slot_w / slot_h
    mismatch = (max(src_aspect, dst_aspect)
                / max(0.001, min(src_aspect, dst_aspect)))
    # Fraction of the long axis cover-resize would discard to fill the slot.
    crop_loss = 1.0 - 1.0 / mismatch
    is_squareish_bbox = 0.85 <= src_aspect <= 1.20
    bbox_area = bw * bh
    slot_area = slot_w * slot_h
    panel_to_bbox = slot_area / max(1, bbox_area)
    bbox_much_smaller = panel_to_bbox > 8.0
    use_fit = (
        crop_loss > MAX_COVER_CROP
        or is_squareish_bbox
        or bbox_much_smaller
    )
    if not use_fit:
        return (
            f"[{in_label}]crop={bw}:{bh}:{bx}:{by},"
            f"scale={slot_w}:{slot_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={slot_w}:{slot_h}[{out_label}]"
        )
    return _fit_blurred_backdrop(in_label, box, slot_w, slot_h, out_label)


def build_filter_complex(plan: dict, out_w: int = OUT_W,
                          out_h: int = OUT_H) -> tuple[str, int]:
    """Build the -filter_complex string for a v11 plan.

    Returns (filter_str, n_split_branches). The source input is split into
    `n_split_branches` copies so each panel/slot can crop independently.
    """
    comp = plan["composition"]
    panels = plan["panels"]

    if comp == "single_view":
        # Cover-resize the panel's source bbox (or the whole frame, if the
        # panel's bbox is full-frame) to fill 9:16.
        box = panels[0]["boxes"][0] if panels else None
        if box is None:
            f = (
                f"[0:v]scale={out_w}:{out_h}:"
                f"force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={out_w}:{out_h},setsar=1[v]"
            )
        else:
            bx, by, bw, bh = box
            f = (
                f"[0:v]crop={bw}:{bh}:{bx}:{by},"
                f"scale={out_w}:{out_h}:"
                f"force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={out_w}:{out_h},setsar=1[v]"
            )
        return f, 1

    if comp == "letterbox_irl":
        # Source bbox laid in 9:16, leftover space filled by a blurred
        # cover backdrop of the same content (no white/solid bars).
        box = panels[0]["boxes"][0] if panels else None
        f = _fit_blurred_backdrop("0:v", box, out_w, out_h, "v", setsar=True)
        return f, 1

    if comp == "webcam_overlay":
        # Full-frame cover-resize of bg region + alpha-blended webcam tile
        # at its detected corner. Used when the detector returns a
        # gameplay/content region that fills the source plus a webcam tile
        # overlaid on top (e.g. CS gameplay with a corner facecam).
        if not panels:
            raise ValueError("webcam_overlay needs at least one panel")
        bg_box = panels[0]["boxes"][0]
        bg_x, bg_y, bg_w, bg_h = bg_box
        pip = plan.get("pip") or {}
        pip_box = pip.get("bbox")
        if not pip_box:
            # No PIP info — degrade to single_view cover-resize.
            return (
                f"[0:v]crop={bg_w}:{bg_h}:{bg_x}:{bg_y},"
                f"scale={out_w}:{out_h}:"
                f"force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={out_w}:{out_h},setsar=1[v]"
            ), 1

        px, py, pw, ph = pip_box
        # size_pct is integer percent (e.g. 30) — matches debug script.
        size_frac = float(pip.get("size_pct", 30)) / 100.0
        pip_dst_w = max(64, int(out_w * size_frac))
        pip_dst_w -= pip_dst_w % 2
        # preserve webcam aspect
        pip_aspect = pw / max(1, ph)
        pip_dst_h = max(64, int(pip_dst_w / max(0.01, pip_aspect)))
        pip_dst_h -= pip_dst_h % 2

        corner = str(pip.get("corner", "bottom-right"))
        margin = 32
        pip_y_out = margin if "top" in corner else out_h - pip_dst_h - margin
        pip_x_out = margin if "left" in corner else out_w - pip_dst_w - margin

        f = (
            f"[0:v]split=2[wo_bg_src][wo_pip_src];"
            f"[wo_bg_src]crop={bg_w}:{bg_h}:{bg_x}:{bg_y},"
            f"scale={out_w}:{out_h}:"
            f"force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={out_w}:{out_h}[wo_bg];"
            f"[wo_pip_src]crop={pw}:{ph}:{px}:{py},"
            f"scale={pip_dst_w}:{pip_dst_h}:flags=lanczos[wo_pip];"
            f"[wo_bg][wo_pip]overlay={pip_x_out}:{pip_y_out},setsar=1[v]"
        )
        return f, 1

    if comp == "split_screen_2_horizontal":
        # Side-by-side hstack: 2 panels, each `out_w/2` wide × `out_h` tall.
        # Used when both source regions are taller-than-wide (e.g. webcam
        # + wordle grid). Reuses _slot_filter for cover-resize vs blur-fill
        # decision per panel.
        if len(panels) != 2:
            raise ValueError(
                f"split_screen_2_horizontal needs 2 panels, got {len(panels)}"
            )
        slot_w = out_w // 2
        slot_h = out_h
        box0 = panels[0]["boxes"][0]
        box1 = panels[1]["boxes"][0]
        bg0 = panels[0].get("bg_color", DEFAULT_BG_HEX)
        bg1 = panels[1].get("bg_color", DEFAULT_BG_HEX)
        f = (
            f"[0:v]split=2[hs_l_src][hs_r_src];"
            + _slot_filter("hs_l_src", box0, slot_w, slot_h, "hs_l", bg0) + ";"
            + _slot_filter("hs_r_src", box1, slot_w, slot_h, "hs_r", bg1) + ";"
            + f"[hs_l][hs_r]hstack=inputs=2,setsar=1[v]"
        )
        return f, 2

    if comp == "split_screen_2_plus_pip":
        raise NotImplementedError("PIP composition not supported by ffmpeg path")

    if comp not in ("split_screen_2", "split_screen_3"):
        raise ValueError(f"unknown composition: {comp}")

    heights = _compute_heights(panels, out_h, out_w)
    panel_labels: list[str] = []
    slot_filters: list[str] = []
    slot_total = sum(len(p["boxes"]) for p in panels)
    # Split source into one branch per slot (cover-resize uses 1, blur-fill
    # uses 2 — split for blur-fill is done in _slot_filter via split=2).
    # Top-level split: one branch per slot.
    src_branches = [f"src{i}" for i in range(slot_total)]
    split_str = f"[0:v]split={slot_total}[{']['.join(src_branches)}]"

    branch_iter = iter(src_branches)
    for pi, (panel, ph) in enumerate(zip(panels, heights)):
        n = len(panel["boxes"])
        slot_w = out_w // n
        slot_widths = [slot_w] * n
        # Last slot absorbs rounding remainder.
        slot_widths[-1] = out_w - slot_w * (n - 1)
        slot_outs: list[str] = []
        panel_bg = panel.get("bg_color", DEFAULT_BG_HEX)
        for si, box in enumerate(panel["boxes"]):
            sw = slot_widths[si]
            in_label = next(branch_iter)
            out_label = f"p{pi}s{si}"
            slot_filters.append(
                _slot_filter(in_label, box, sw, ph, out_label, panel_bg)
            )
            slot_outs.append(out_label)
        if n == 1:
            panel_labels.append(slot_outs[0])
        else:
            row_label = f"row{pi}"
            slot_filters.append(
                f"[{']['.join(slot_outs)}]hstack=inputs={n}[{row_label}]"
            )
            panel_labels.append(row_label)

    # vstack panels into the final canvas. setsar=1 forces square pixels
    # on the final output — without it, ffmpeg's scale chain leaks a
    # non-1:1 SAR onto the encoder and players stretch the 1080x1920
    # buffer back to 16:9.
    final = (
        split_str + ";"
        + ";".join(slot_filters) + ";"
        + f"[{']['.join(panel_labels)}]vstack=inputs={len(panel_labels)},setsar=1[v]"
    )
    return final, slot_total


def render_segment_ffmpeg(
    src: Path,
    t_start: float,
    t_end: float,
    plan: dict,
    dst: Path,
    fps: float,
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    debug: bool = False,
) -> None:
    """Render one segment of `src` from t_start..t_end with `plan` into `dst`.

    No audio — caller stitches segments + muxes audio in a separate pass.
    """
    duration = max(0.001, t_end - t_start)
    filter_complex, _ = build_filter_complex(plan, out_w, out_h)
    if debug:
        print(f"[v11_render] filter_complex:\n{filter_complex}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t_start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-r", f"{fps}",
        *video_encoder_args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
        "-pix_fmt", "yuv420p",
        "-an",  # no audio per-segment; muxed back at the end.
        str(dst),
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"ffmpeg segment render failed (rc={rc})")


def concat_and_mux(seg_paths: list[Path], audio_src: Path,
                    dst: Path) -> None:
    """Concat segment videos with concat demuxer, mux audio from audio_src."""
    if not seg_paths:
        raise ValueError("no segments to concat")

    # Write concat listfile. Use ABSOLUTE paths so ffmpeg's "-safe 0"
    # path resolution (relative to the listfile's directory) doesn't
    # accidentally prefix segment paths with the dst parent.
    listfile = dst.with_suffix(".concat.txt")
    listfile.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in seg_paths),
        encoding="utf-8",
    )

    if len(seg_paths) == 1:
        # Single segment — just remux + add audio.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(seg_paths[0]),
            "-i", str(audio_src),
            "-c:v", "copy",
            "-c:a", "aac", "-shortest",
            "-map", "0:v:0", "-map", "1:a:0?",
            str(dst),
        ]
    else:
        # Concat + mux audio.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-i", str(audio_src),
            "-c:v", "copy",
            "-c:a", "aac", "-shortest",
            "-map", "0:v:0", "-map", "1:a:0?",
            str(dst),
        ]
    rc = subprocess.run(cmd).returncode
    listfile.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"ffmpeg concat/mux failed (rc={rc})")
