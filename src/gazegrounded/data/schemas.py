"""Row schemas for the standardised dataset.

Every data source (synthetic generator, UEyes, the WebGazer study) is converted by an adapter
into these tables, so the rest of the pipeline never sees source-specific columns.

Conventions
-----------
* Gaze coordinates and fixation centroids are in **screenshot pixels** (origin top-left).
* Gaze timestamps are **milliseconds since stimulus onset** of the trial.
* AOI boxes are **normalised** to [0, 1] relative to the screenshot width/height.
* Participant IDs are pseudonymous codes (e.g. ``P07``); no names, e-mails or IPs.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FRICTION_LABELS: tuple[str, ...] = ("low", "medium", "high")
FRICTION_TO_ORDINAL: dict[str, int] = {label: i for i, label in enumerate(FRICTION_LABELS)}

FEEDBACK_CATEGORIES: tuple[str, ...] = (
    "findability",
    "visual_hierarchy",
    "ambiguous_wording",
    "information_overload",
    "feedback_uncertainty",
    "navigation",
    "no_friction",
    "other",
)

CONDITIONS: tuple[str, ...] = ("clean", "friction", "single", "free_viewing")


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class UIRecord(_Row):
    ui_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1, description="Relative to the dataset directory")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    domain: str | None = None
    condition: Literal["clean", "friction", "single", "free_viewing"] = "single"
    pair_id: str | None = None


class AOIRecord(_Row):
    ui_id: str
    aoi_id: str = Field(min_length=1)
    label: str | None = None  # visible text; None for unlabeled elements such as icons
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    semantic_role: str = "unknown"
    description: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> AOIRecord:
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError(f"AOI {self.aoi_id}: box must satisfy min < max")
        return self


class TaskRecord(_Row):
    task_id: str
    ui_id: str
    task_key: str
    task_description: str = Field(min_length=1)
    target_aoi_ids: str = Field(description="Semicolon-separated AOI ids relevant to the task")
    designed_friction_type: str | None = None

    def targets(self) -> list[str]:
        return [t for t in self.target_aoi_ids.split(";") if t]


class TrialRecord(_Row):
    participant_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    session_id: str
    trial_id: str
    ui_id: str
    task_id: str
    task_description: str
    condition: Literal["clean", "friction", "single", "free_viewing"]
    trial_order: int | None = None
    task_success: bool | None = None
    completion_time_ms: float | None = Field(default=None, ge=0)
    error_count: int | None = Field(default=None, ge=0)
    click_count: int | None = Field(default=None, ge=0)
    feedback_text: str | None = None
    feedback_category: str | None = None
    friction_label: Literal["low", "medium", "high"] | None = None
    label_source: str | None = None
    ease_rating: int | None = Field(default=None, ge=1, le=7)

    @field_validator("feedback_category")
    @classmethod
    def _known_category(cls, v: str | None) -> str | None:
        if v is not None and v not in FEEDBACK_CATEGORIES:
            raise ValueError(f"Unknown feedback category {v!r}")
        return v


class GazeSample(_Row):
    participant_id: str
    session_id: str
    trial_id: str
    ui_id: str
    timestamp_ms: float = Field(ge=0)
    gaze_x: float | None = None
    gaze_y: float | None = None
    validity: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _valid_has_coords(self) -> GazeSample:
        if self.validity == 1 and (
            self.gaze_x is None
            or self.gaze_y is None
            or math.isnan(self.gaze_x)
            or math.isnan(self.gaze_y)
        ):
            raise ValueError("valid gaze sample must have coordinates")
        return self


class FixationRecord(_Row):
    participant_id: str
    trial_id: str
    fixation_id: int = Field(ge=0)
    start_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)
    duration_ms: float = Field(gt=0)
    centroid_x: float
    centroid_y: float
    aoi_id: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> FixationRecord:
        if self.end_ms < self.start_ms:
            raise ValueError("fixation end_ms < start_ms")
        return self


TABLE_SCHEMAS: dict[str, type[_Row]] = {
    "uis": UIRecord,
    "aois": AOIRecord,
    "tasks": TaskRecord,
    "trials": TrialRecord,
    "gaze_samples": GazeSample,
    "fixations": FixationRecord,
}

REQUIRED_TABLES: tuple[str, ...] = ("uis", "aois", "tasks", "trials")
