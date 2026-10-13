"""Raw -> processed: run the configured adapter, validate, clean gaze samples, save."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.data.loader import StandardizedDataset, get_adapter
from gazegrounded.utils import get_logger, save_json

log = get_logger(__name__)


def clean_gaze_samples(
    samples: pd.DataFrame, uis: pd.DataFrame, offscreen_margin: float = 0.1
) -> pd.DataFrame:
    """Sort samples, drop exact duplicates and invalidate samples far outside the screenshot.

    Samples more than ``offscreen_margin`` (fraction of width/height) outside the image are set
    to ``validity = 0`` rather than clipped: clipping would pile spurious gaze onto the edges.
    """
    g = samples.sort_values(["trial_id", "timestamp_ms"]).drop_duplicates(
        subset=["trial_id", "timestamp_ms"]
    )
    sizes = uis.set_index("ui_id")[["width", "height"]]
    w = g["ui_id"].map(sizes["width"]).to_numpy(dtype=float)
    h = g["ui_id"].map(sizes["height"]).to_numpy(dtype=float)
    x = g["gaze_x"].to_numpy(dtype=float)
    y = g["gaze_y"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        off = (
            (x < -offscreen_margin * w)
            | (x > (1 + offscreen_margin) * w)
            | (y < -offscreen_margin * h)
            | (y > (1 + offscreen_margin) * h)
        )
    invalid = off | np.isnan(x) | np.isnan(y)
    g = g.copy()
    g["validity"] = np.where(invalid, 0, g["validity"].to_numpy()).astype(int)
    g.loc[invalid, ["gaze_x", "gaze_y"]] = np.nan
    return g.reset_index(drop=True)


def trial_quality(samples: pd.DataFrame, min_valid_ratio: float = 0.6) -> pd.DataFrame:
    """Per-trial data-quality summary: sample count, valid ratio, duration, sampling rate."""

    def _one(g: pd.DataFrame) -> pd.Series:
        t = g["timestamp_ms"].to_numpy()
        duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        dt = np.diff(t)
        return pd.Series(
            {
                "participant_id": g["participant_id"].iloc[0],
                "n_samples": len(g),
                "valid_ratio": float(g["validity"].mean()),
                "duration_ms": duration,
                "sampling_rate_hz": float(1000.0 / np.median(dt)) if len(dt) else np.nan,
            }
        )

    q = samples.groupby("trial_id", sort=True)[samples.columns.tolist()].apply(_one).reset_index()
    q["low_quality"] = q["valid_ratio"] < min_valid_ratio
    return q


def run_preprocessing(cfg) -> StandardizedDataset:
    adapter = get_adapter(
        cfg.data.adapter, cfg.raw_dir(), cfg.data.screen_width, cfg.data.screen_height
    )
    log.info("Loading raw data with %s from %s", type(adapter).__name__, adapter.raw_dir)
    ds, report = adapter.run()
    for w in report.warnings:
        log.warning(w)
    report.raise_if_errors()
    if ds.gaze_samples is not None:
        ds.gaze_samples = clean_gaze_samples(ds.gaze_samples, ds.uis)
        quality = trial_quality(ds.gaze_samples, cfg.gaze.min_valid_sample_ratio)
        n_low = int(quality["low_quality"].sum())
        if n_low:
            log.warning(
                "%d/%d trials below the valid-sample threshold (%.0f%%)",
                n_low,
                len(quality),
                100 * cfg.gaze.min_valid_sample_ratio,
            )
    else:
        quality = pd.DataFrame()
    out = ds.save(cfg.processed_dir())
    if not quality.empty:
        quality.to_csv(out / "trial_quality.csv", index=False)
    save_json({"errors": report.errors, "warnings": report.warnings}, out / "validation.json")
    log.info("Saved standardised dataset to %s (%s)", out, report.summary())
    return ds
