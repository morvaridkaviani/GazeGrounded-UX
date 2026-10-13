"""Robust parsing of VLM responses and grounding of VLM elements to annotated AOIs.

``parse_vlm_response`` never raises: malformed output yields ``parse_status="failed"`` with
warnings, and individually invalid elements are dropped (``"partial"``). This keeps a single
bad generation from crashing a batch run.

Tolerated deviations (seen in real model outputs)
-------------------------------------------------
* JSON wrapped in markdown fences or surrounded by prose
* a bare list of elements instead of ``{"elements": [...]}``
* trailing commas, smart quotes, Python-style ``True/False/None``/single quotes
* alternative keys: ``bbox_2d``/``box``, ``relevance``, ``priority``, ``name``/``element``
* boxes as lists ``[x0, y0, x1, y1]``, in pixels, in 0-1000 ("per mille") or swapped corners
* scores given as percentages (e.g. ``85``)
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from gazegrounded.gaze.aoi import box_iou
from gazegrounded.vlm.schemas import VLMAnalysis, VLMElement

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _balanced_block(text: str) -> str | None:
    """First balanced ``{...}`` or ``[...]`` block in ``text`` (string-aware)."""
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        close = "}" if ch == "{" else "]"
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == ch:
                depth += 1
            elif c == close:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None
    return None


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from free-form model output. Raises ValueError."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        block = _balanced_block(cand.strip())
        if block is None:
            continue
        for attempt in _repairs(block):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        try:  # Python-literal style output: single quotes, True/False/None
            return ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
    raise ValueError("no parseable JSON object found in model output")


def _repairs(block: str):
    yield block
    fixed = block.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    yield fixed
    yield re.sub(
        r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", fixed))
    )


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _score(value: Any) -> float:
    v = float(value)
    if 1.0 < v <= 100.0:  # percentage
        v /= 100.0
    return v


def _bbox(raw: Any, width: int | None, height: int | None, coordinate_space: str) -> dict:
    if isinstance(raw, dict):
        vals = [
            _first(raw, "x_min", "xmin", "x0", "left"),
            _first(raw, "y_min", "ymin", "y0", "top"),
            _first(raw, "x_max", "xmax", "x1", "right"),
            _first(raw, "y_max", "ymax", "y1", "bottom"),
        ]
        if any(v is None for v in vals) and {"x", "y", "width", "height"} <= set(raw):
            vals = [raw["x"], raw["y"], raw["x"] + raw["width"], raw["y"] + raw["height"]]
    elif isinstance(raw, list | tuple) and len(raw) == 4:
        vals = list(raw)
    else:
        raise ValueError(f"unrecognised bbox {raw!r}")
    x0, y0, x1, y1 = (float(v) for v in vals)
    space = coordinate_space
    if space == "auto":
        top = max(x0, y0, x1, y1)
        if top <= 1.0:
            space = "normalized"
        elif width and height and (x1 <= width * 1.05 and y1 <= height * 1.05):
            space = "pixels"
        elif top <= 1000.0:
            space = "per_mille"
        else:
            space = "pixels"
    if space == "pixels":
        if not (width and height):
            raise ValueError("pixel bbox but image size unknown")
        x0, x1, y0, y1 = x0 / width, x1 / width, y0 / height, y1 / height
    elif space == "per_mille":
        x0, y0, x1, y1 = x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return {"x_min": clip(x0), "y_min": clip(y0), "x_max": clip(x1), "y_max": clip(y1)}


def parse_vlm_response(
    raw_text: str,
    task: str,
    image_size: tuple[int, int] | None = None,
    coordinate_space: str = "auto",
    provider: str = "unknown",
    model: str | None = None,
    prompt_version: str = "v1",
) -> VLMAnalysis:
    """Parse raw model text into a validated :class:`VLMAnalysis` (never raises)."""
    meta = {"task": task, "provider": provider, "model": model, "prompt_version": prompt_version}
    width, height = image_size or (None, None)
    try:
        data = extract_json(raw_text or "")
    except ValueError as exc:
        return VLMAnalysis(**meta, parse_status="failed", warnings=[str(exc)])
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = _first(data, "elements", "regions", "items", "results", default=None)
        if items is None and ("bbox" in data or "bbox_2d" in data):
            items = [data]
    else:
        items = None
    if not isinstance(items, list):
        return VLMAnalysis(**meta, parse_status="failed", warnings=["JSON has no list of elements"])

    elements, warnings = [], []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"element {i}: not an object")
            continue
        try:
            relevance = _score(_first(item, "task_relevance", "relevance", "relevance_score"))
            priority_raw = _first(
                item,
                "expected_attention_priority",
                "attention_priority",
                "priority",
                default=relevance,
            )
            el = VLMElement(
                label=str(_first(item, "label", "name", "element", "text", default="")).strip()
                or f"element_{i}",
                semantic_role=str(_first(item, "semantic_role", "role", "type", default="unknown")),
                task_relevance=relevance,
                expected_attention_priority=_score(priority_raw),
                bbox=_bbox(
                    _first(item, "bbox", "bbox_2d", "box", "bounding_box"),
                    width,
                    height,
                    coordinate_space,
                ),
                reason=str(_first(item, "reason", "rationale", "explanation", default="")),
            )
            elements.append(el)
        except (ValueError, TypeError, ValidationError) as exc:
            msg = str(exc).splitlines()[0][:160]
            warnings.append(f"element {i}: dropped ({msg})")
    if not elements:
        status = "failed"
    elif warnings:
        status = "partial"
    else:
        status = "ok"
    return VLMAnalysis(**meta, elements=elements, parse_status=status, warnings=warnings)


def match_elements_to_aois(
    analysis: VLMAnalysis, aois: pd.DataFrame, min_iou: float = 0.1
) -> pd.DataFrame:
    """Match each VLM element to the annotated AOI with the highest IoU (or containing centre).

    Returns one row per element with ``matched_aoi_id`` (None if no AOI overlaps enough).
    """
    boxes = aois[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
    rows = []
    for el in analysis.elements:
        b = el.bbox.as_tuple()
        ious = np.array([box_iou(b, tuple(a)) for a in boxes]) if len(boxes) else np.array([])
        best = int(ious.argmax()) if len(ious) else -1
        matched = aois["aoi_id"].iloc[best] if best >= 0 and ious[best] >= min_iou else None
        if matched is None and len(boxes):  # fall back to smallest AOI containing the centre
            cx, cy = el.bbox.center
            inside = (
                (boxes[:, 0] <= cx)
                & (cx <= boxes[:, 2])
                & (boxes[:, 1] <= cy)
                & (cy <= boxes[:, 3])
            )
            if inside.any():
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                matched = aois["aoi_id"].iloc[int(np.where(inside, areas, np.inf).argmin())]
        rows.append(
            {
                "label": el.label,
                "semantic_role": el.semantic_role,
                "task_relevance": el.task_relevance,
                "expected_attention_priority": el.expected_attention_priority,
                "weight": el.weight,
                "x_min": b[0],
                "y_min": b[1],
                "x_max": b[2],
                "y_max": b[3],
                "matched_aoi_id": matched,
                "match_iou": float(ious[best]) if best >= 0 else 0.0,
                "reason": el.reason,
            }
        )
    return pd.DataFrame(rows)
