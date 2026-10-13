"""Validation of standardised tables: per-row schema checks plus cross-table integrity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import TypeAdapter, ValidationError

from gazegrounded.data.schemas import REQUIRED_TABLES, TABLE_SCHEMAS

if TYPE_CHECKING:
    from gazegrounded.data.loader import StandardizedDataset

# Column names that suggest personally identifiable information slipped into a table.
PII_COLUMN_HINTS = ("name", "email", "e_mail", "ip", "ip_address", "phone", "birth", "address")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def raise_if_errors(self) -> None:
        if self.errors:
            preview = "\n  - ".join(self.errors[:20])
            raise ValueError(
                f"Dataset validation failed ({len(self.errors)} errors):\n  - {preview}"
            )

    def summary(self) -> str:
        return f"{len(self.errors)} errors, {len(self.warnings)} warnings"


def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict(orient="records")


def validate_table(df: pd.DataFrame, table: str, max_errors: int = 20) -> ValidationReport:
    """Validate every row of ``df`` against the pydantic schema registered for ``table``."""
    report = ValidationReport()
    model = TABLE_SCHEMAS[table]
    required = [n for n, f in model.model_fields.items() if f.is_required()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"[{table}] missing required columns: {missing}")
        return report
    pii = [c for c in df.columns if c.lower() in PII_COLUMN_HINTS]
    if pii:
        report.errors.append(f"[{table}] columns look like PII and must be removed: {pii}")
    adapter = TypeAdapter(list[model])
    try:
        adapter.validate_python(_records(df))
    except ValidationError as exc:
        for err in exc.errors()[:max_errors]:
            loc = ".".join(str(p) for p in err["loc"])
            report.errors.append(f"[{table}] row {loc}: {err['msg']}")
        if exc.error_count() > max_errors:
            report.errors.append(f"[{table}] ... {exc.error_count() - max_errors} more errors")
    return report


def validate_dataset(ds: StandardizedDataset) -> ValidationReport:
    """Schema checks for all tables plus referential integrity and data-quality warnings."""
    report = ValidationReport()
    for table in REQUIRED_TABLES:
        df = getattr(ds, table)
        if df is None or df.empty:
            report.errors.append(f"[{table}] table is missing or empty")
        else:
            report.extend(validate_table(df, table))
    for table in ("gaze_samples", "fixations"):
        df = getattr(ds, table)
        if df is not None and not df.empty:
            report.extend(validate_table(df, table))
    if ds.gaze_samples is None and ds.fixations is None:
        report.errors.append("dataset has neither gaze_samples nor fixations")
    if not report.ok:
        return report

    ui_ids = set(ds.uis["ui_id"])
    _check_unique(report, ds.uis, ["ui_id"], "uis")
    _check_unique(report, ds.aois, ["ui_id", "aoi_id"], "aois")
    _check_unique(report, ds.tasks, ["task_id"], "tasks")
    _check_unique(report, ds.trials, ["trial_id"], "trials")
    _check_subset(report, ds.aois["ui_id"], ui_ids, "aois.ui_id", "uis")
    _check_subset(report, ds.tasks["ui_id"], ui_ids, "tasks.ui_id", "uis")
    _check_subset(report, ds.trials["ui_id"], ui_ids, "trials.ui_id", "uis")
    _check_subset(report, ds.trials["task_id"], set(ds.tasks["task_id"]), "trials.task_id", "tasks")

    aoi_keys = set(zip(ds.aois["ui_id"], ds.aois["aoi_id"], strict=True))
    for row in ds.tasks.itertuples():
        for target in str(row.target_aoi_ids).split(";"):
            if target and (row.ui_id, target) not in aoi_keys:
                report.errors.append(f"[tasks] {row.task_id}: target AOI {target!r} not in aois")

    for name in ("gaze_samples", "fixations"):
        df = getattr(ds, name)
        if df is not None:
            _check_subset(
                report, df["trial_id"], set(ds.trials["trial_id"]), f"{name}.trial_id", "trials"
            )

    if ds.gaze_samples is not None:
        _gaze_quality(report, ds)
    for image in ds.uis["image_path"]:
        if not (ds.root / image).exists():
            report.errors.append(f"[uis] screenshot not found: {ds.root / image}")
    return report


def _check_unique(report: ValidationReport, df: pd.DataFrame, cols: list[str], name: str) -> None:
    dup = df.duplicated(subset=cols)
    if dup.any():
        report.errors.append(f"[{name}] {int(dup.sum())} duplicate rows on {cols}")


def _check_subset(report, values: pd.Series, allowed: set, what: str, where: str) -> None:
    unknown = set(values.dropna().unique()) - allowed
    if unknown:
        report.errors.append(
            f"[{what}] {len(unknown)} ids not present in {where}: "
            f"{sorted(map(str, unknown))[:5]}"
        )


def _gaze_quality(report: ValidationReport, ds: StandardizedDataset) -> None:
    g = ds.gaze_samples
    sizes = ds.uis.set_index("ui_id")[["width", "height"]]
    merged = g.join(sizes, on="ui_id")
    valid = merged["validity"] == 1
    margin = 0.1  # allow 10% off-screen before flagging (calibration drift is common)
    off = valid & (
        (merged["gaze_x"] < -margin * merged["width"])
        | (merged["gaze_x"] > (1 + margin) * merged["width"])
        | (merged["gaze_y"] < -margin * merged["height"])
        | (merged["gaze_y"] > (1 + margin) * merged["height"])
    )
    if off.any():
        report.warnings.append(f"[gaze_samples] {int(off.sum())} valid samples far off-screen")
    order_ok = g.groupby("trial_id")["timestamp_ms"].apply(lambda s: bool(np.all(np.diff(s) >= 0)))
    if not order_ok.all():
        bad = order_ok[~order_ok].index.tolist()[:5]
        report.errors.append(f"[gaze_samples] timestamps not sorted within trials: {bad}")
    ratio = g.groupby("trial_id")["validity"].mean()
    low = ratio[ratio < 0.6]
    if len(low):
        report.warnings.append(
            f"[gaze_samples] {len(low)} trials have <60% valid samples (e.g. {low.index[0]})"
        )
