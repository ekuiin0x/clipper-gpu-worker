from __future__ import annotations
from enum import Enum
try:
    from typing import Self  # Python 3.11+
except ImportError:  # 3.10 container ships typing_extensions via pydantic
    from typing_extensions import Self
from pydantic import BaseModel, Field, field_validator, model_validator


class Composition(str, Enum):
    SINGLE_PERSON = "single_person"
    IRL_BASIC = "IRL_basic"
    IRL_CROWDED = "IRL_crowded"
    GAMEPLAY_ONLY = "gameplay_only"
    SCREENCAP_ONLY = "screencap_only"
    LETTERBOX = "letterbox"
    WEBCAM_GAMEPLAY = "webcam+gameplay"
    WEBCAM_BROWSER = "webcam+browser"
    WEBCAMS_GAMEPLAY = "webcams+gameplay"
    COMMENTATORS_GAMEPLAY = "commentators+gameplay"
    PLAYERS_GAMEPLAY = "players+gameplay"
    INTERVIEW = "interview"
    SPLIT_SCREEN = "split_screen"
    PLAYERS_GAMEPLAY_COMMENTATORS = "players+gameplay+commentators"
    PLAYERS_BOARD_COMMENTATORS = "players+board+commentators"


COMPOSITION_CATALOG: dict[Composition, list[str]] = {
    Composition.SINGLE_PERSON:                  ["subject"],
    Composition.IRL_BASIC:                      ["subjects"],
    Composition.IRL_CROWDED:                    ["scene"],
    Composition.GAMEPLAY_ONLY:                  ["gameplay"],
    Composition.SCREENCAP_ONLY:                 ["screencap"],
    Composition.LETTERBOX:                      ["scene"],
    Composition.WEBCAM_GAMEPLAY:                ["gameplay", "webcam"],
    Composition.WEBCAM_BROWSER:                 ["browser", "webcam"],
    Composition.WEBCAMS_GAMEPLAY:               ["gameplay", "webcams"],
    Composition.COMMENTATORS_GAMEPLAY:          ["gameplay", "commentators"],
    Composition.PLAYERS_GAMEPLAY:               ["gameplay", "players"],
    Composition.INTERVIEW:                      ["person_1", "person_2"],
    Composition.SPLIT_SCREEN:                   ["left", "right"],
    Composition.PLAYERS_GAMEPLAY_COMMENTATORS:  ["players", "gameplay", "commentators"],
    Composition.PLAYERS_BOARD_COMMENTATORS:     ["players", "board", "commentators"],
}


# Vertical dest-panel heights as ratios of out_h, in role order from the catalog.
COMPOSITION_PANEL_GRID: dict[Composition, list[float]] = {
    Composition.SINGLE_PERSON:                  [1.00],
    Composition.IRL_BASIC:                      [1.00],
    Composition.IRL_CROWDED:                    [1.00],
    Composition.GAMEPLAY_ONLY:                  [1.00],
    Composition.SCREENCAP_ONLY:                 [1.00],
    Composition.LETTERBOX:                      [1.00],
    Composition.WEBCAM_GAMEPLAY:                [0.70, 0.30],
    Composition.WEBCAM_BROWSER:                 [0.70, 0.30],
    Composition.WEBCAMS_GAMEPLAY:               [0.70, 0.30],
    Composition.COMMENTATORS_GAMEPLAY:          [0.65, 0.35],
    Composition.PLAYERS_GAMEPLAY:               [0.75, 0.25],
    Composition.INTERVIEW:                      [0.50, 0.50],
    Composition.SPLIT_SCREEN:                   [0.50, 0.50],
    Composition.PLAYERS_GAMEPLAY_COMMENTATORS:  [0.22, 0.56, 0.22],
    Composition.PLAYERS_BOARD_COMMENTATORS:     [0.22, 0.56, 0.22],
}


class RenderMode(str, Enum):
    LETTERBOX = "letterbox"   # blur-fill the whole source frame
    STACKED = "stacked"       # crop each panel's bbox and stack vertically


COMPOSITION_RENDER_MODE: dict[Composition, RenderMode] = {
    # The two compositions that render the whole source as a centered band.
    Composition.IRL_CROWDED: RenderMode.LETTERBOX,
    Composition.LETTERBOX:   RenderMode.LETTERBOX,
    # Everything else stacks panels.
    Composition.SINGLE_PERSON:                  RenderMode.STACKED,
    Composition.IRL_BASIC:                      RenderMode.STACKED,
    Composition.GAMEPLAY_ONLY:                  RenderMode.STACKED,
    Composition.SCREENCAP_ONLY:                 RenderMode.STACKED,
    Composition.WEBCAM_GAMEPLAY:                RenderMode.STACKED,
    Composition.WEBCAM_BROWSER:                 RenderMode.STACKED,
    Composition.WEBCAMS_GAMEPLAY:               RenderMode.STACKED,
    Composition.COMMENTATORS_GAMEPLAY:          RenderMode.STACKED,
    Composition.PLAYERS_GAMEPLAY:               RenderMode.STACKED,
    Composition.INTERVIEW:                      RenderMode.STACKED,
    Composition.SPLIT_SCREEN:                   RenderMode.STACKED,
    Composition.PLAYERS_GAMEPLAY_COMMENTATORS:  RenderMode.STACKED,
    Composition.PLAYERS_BOARD_COMMENTATORS:     RenderMode.STACKED,
}


def panel_count(c: Composition) -> int:
    return len(COMPOSITION_CATALOG[c])


class PanelSpec(BaseModel):
    id: int = Field(ge=1)
    role: str
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in source pixels

    @model_validator(mode="after")
    def _bbox_positive(self) -> Self:
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"bbox w,h must be positive, got ({w},{h})")
        if x < 0 or y < 0:
            raise ValueError(f"bbox x,y must be non-negative, got ({x},{y})")
        return self


class SegmentSpec(BaseModel):
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    composition: Composition
    panels: list[PanelSpec] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    # Refined boundary times in seconds. When None, the renderer falls back
    # to grid-aligned times computed from frame_start/frame_end * spacing_s.
    # Set by boundary refinement passes (e.g. histogram-diff scene snap).
    t_start: float | None = Field(default=None, ge=0.0)
    t_end: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.frame_end < self.frame_start:
            raise ValueError(f"frame_end {self.frame_end} < frame_start {self.frame_start}")
        if self.t_start is not None and self.t_end is not None and self.t_end <= self.t_start:
            raise ValueError(f"t_end {self.t_end} <= t_start {self.t_start}")
        expected_roles = COMPOSITION_CATALOG[self.composition]
        got_roles = [p.role for p in self.panels]
        if got_roles != expected_roles:
            raise ValueError(
                f"composition {self.composition.value} expects roles {expected_roles}, "
                f"got {got_roles}"
            )
        return self

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start + 1


class ClipPlan(BaseModel):
    n_frames_seen: int = Field(ge=1)
    segments: list[SegmentSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _coverage(self) -> Self:
        cursor = 0
        for i, seg in enumerate(self.segments):
            if seg.frame_start != cursor:
                raise ValueError(
                    f"segment {i} starts at {seg.frame_start}, expected {cursor}"
                )
            cursor = seg.frame_end + 1
        if cursor != self.n_frames_seen:
            raise ValueError(
                f"segments end at frame {cursor - 1}, expected {self.n_frames_seen - 1}"
            )
        return self


def letterbox_clip_plan(n_frames: int, src_w: int = 1, src_h: int = 1) -> ClipPlan:
    """Whole-clip letterbox fallback. Used when Call A fails entirely."""
    return ClipPlan(
        n_frames_seen=n_frames,
        segments=[
            SegmentSpec(
                frame_start=0,
                frame_end=n_frames - 1,
                composition=Composition.LETTERBOX,
                panels=[PanelSpec(id=1, role="scene", bbox=(0, 0, src_w, src_h))],
                confidence=0.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Typed-template VLM pipeline (USE_TYPED_TEMPLATES=1)
# ---------------------------------------------------------------------------
class ContentType(str, Enum):
    IRL_PODCAST = "irl_podcast"
    CHESS_BROADCAST = "chess_broadcast"
    GAMEPLAY = "gameplay"
    MOBA_GAMEPLAY = "moba_gameplay"
    TALKING_HEAD = "talking_head"
    REACTION_VIDEO = "reaction_video"
    TUTORIAL_SCREENCAST = "tutorial_screencast"
    SPORTS_BROADCAST = "sports_broadcast"
    UNKNOWN = "unknown"


class Slot(BaseModel):
    label: str
    source_bbox: tuple[float, float, float, float]  # normalized (l, t, r, b)

    @field_validator("source_bbox")
    @classmethod
    def _bbox_well_formed(cls, v: tuple[float, float, float, float]):
        l, t, r, b = v
        if not (0.0 <= l < r <= 1.0):
            raise ValueError(f"source_bbox left/right out of range or inverted: {v}")
        if not (0.0 <= t < b <= 1.0):
            raise ValueError(f"source_bbox top/bottom out of range or inverted: {v}")
        return v


class ClassificationResult(BaseModel):
    content_type: ContentType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LayoutPlan(BaseModel):
    content_type: ContentType
    template_id: str
    slots: list[Slot]
    rationale: str
