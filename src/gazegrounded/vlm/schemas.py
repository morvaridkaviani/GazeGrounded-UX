"""Structured VLM output, validated with pydantic.

Boxes are normalised ``[0, 1]`` (x_min, y_min, x_max, y_max) relative to the screenshot.
Scores are the model's *estimates* of task relevance and attention priority, not measurements.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BBox(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("bbox must satisfy x_min < x_max and y_min < y_max")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class VLMElement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(min_length=1)
    semantic_role: str = "unknown"
    task_relevance: float = Field(ge=0.0, le=1.0)
    expected_attention_priority: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    reason: str = ""

    @property
    def weight(self) -> float:
        """Weight used for the relevance map: relevance x expected attention priority."""
        return self.task_relevance * self.expected_attention_priority


class VLMAnalysis(BaseModel):
    """Parsed result for one (screenshot, task) pair."""

    task: str
    elements: list[VLMElement] = Field(default_factory=list)
    provider: str = "unknown"
    model: str | None = None
    prompt_version: str = "v1"
    parse_status: Literal["ok", "partial", "failed"] = "ok"
    warnings: list[str] = Field(default_factory=list)
    image_sha256: str | None = None
    cache_key: str | None = None
    from_cache: bool = False

    def relevant(self, threshold: float = 0.5) -> list[VLMElement]:
        return [e for e in self.elements if e.task_relevance >= threshold]
