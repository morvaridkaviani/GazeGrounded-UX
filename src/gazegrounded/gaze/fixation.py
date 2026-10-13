"""Fixation detection with the dispersion-threshold algorithm (I-DT).

Reference: Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in
eye-tracking protocols. ETRA '00.

Algorithm
---------
A window of samples spanning at least ``minimum_duration_ms`` is a fixation candidate if its
dispersion ``(max(x) - min(x)) + (max(y) - min(y))`` is at most ``dispersion_threshold_px``.
The window is then grown sample by sample while the dispersion stays under the threshold; the
grown window becomes one fixation (centroid = mean position). Otherwise the window start
advances by one sample.

Limitations (documented, not hidden)
------------------------------------
* The threshold is in pixels, so its meaning in degrees of visual angle depends on screen size
  and viewing distance. Webcam gaze needs a larger threshold than lab trackers (see
  ``configs/experiment.yaml``), which also merges nearby fixations.
* Smooth pursuit and very short fixations (< ``minimum_duration_ms``) are not detected.
* Signal loss: valid samples separated by more than ``max_gap_ms`` are never joined into one
  fixation; shorter gaps (e.g. a single dropped sample) are bridged.
* Sample-to-sample noise inflates dispersion; very noisy data yields fragmented fixations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dispersion(xs: np.ndarray, ys: np.ndarray) -> float:
    return float((xs.max() - xs.min()) + (ys.max() - ys.min()))


def detect_fixations_idt(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray | None = None,
    dispersion_threshold_px: float = 80.0,
    minimum_duration_ms: float = 100.0,
    max_gap_ms: float = 75.0,
) -> pd.DataFrame:
    """Detect fixations in one continuous recording (one trial).

    Returns a DataFrame with ``start_ms, end_ms, duration_ms, centroid_x, centroid_y,
    n_samples``. ``end_ms`` includes the duration of the last sample (one sampling interval).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if valid is not None:
        ok &= np.asarray(valid).astype(bool)
    t, x, y = t[ok], x[ok], y[ok]
    columns = ["start_ms", "end_ms", "duration_ms", "centroid_x", "centroid_y", "n_samples"]
    if len(t) < 2:
        return pd.DataFrame(columns=columns)
    sample_dt = float(np.median(np.diff(t)))

    # Split into segments wherever signal was lost for longer than max_gap_ms.
    breaks = np.where(np.diff(t) > max_gap_ms + sample_dt)[0] + 1
    rows = []
    for seg in np.split(np.arange(len(t)), breaks):
        ts, xs, ys = t[seg], x[seg], y[seg]
        n = len(ts)
        i = 0
        while i < n:
            # Smallest window starting at i that spans the minimum duration.
            # (1e-6 tolerance absorbs rounding of timestamps to 0.01 ms.)
            j = int(
                np.searchsorted(ts, ts[i] + minimum_duration_ms - sample_dt - 1e-6, side="left")
            )
            if j >= n:
                break
            if _dispersion(xs[i : j + 1], ys[i : j + 1]) <= dispersion_threshold_px:
                x_min, x_max = xs[i : j + 1].min(), xs[i : j + 1].max()
                y_min, y_max = ys[i : j + 1].min(), ys[i : j + 1].max()
                while j + 1 < n:
                    nx_min, nx_max = min(x_min, xs[j + 1]), max(x_max, xs[j + 1])
                    ny_min, ny_max = min(y_min, ys[j + 1]), max(y_max, ys[j + 1])
                    if (nx_max - nx_min) + (ny_max - ny_min) > dispersion_threshold_px:
                        break
                    x_min, x_max, y_min, y_max = nx_min, nx_max, ny_min, ny_max
                    j += 1
                start, end = ts[i], ts[j] + sample_dt
                rows.append(
                    (start, end, end - start, xs[i : j + 1].mean(), ys[i : j + 1].mean(), j - i + 1)
                )
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows, columns=columns)


def detect_fixations(
    samples: pd.DataFrame,
    dispersion_threshold_px: float = 80.0,
    minimum_duration_ms: float = 100.0,
    max_gap_ms: float = 75.0,
) -> pd.DataFrame:
    """Run I-DT for every trial in a standard ``gaze_samples`` table.

    Output follows :class:`~gazegrounded.data.schemas.FixationRecord` (``aoi_id`` is left
    empty; see :func:`gazegrounded.gaze.aoi.assign_fixations_to_aois`).
    """
    out = []
    for (pid, trial_id, ui_id), g in samples.groupby(
        ["participant_id", "trial_id", "ui_id"], sort=False
    ):
        fx = detect_fixations_idt(
            g["timestamp_ms"].to_numpy(),
            g["gaze_x"].to_numpy(),
            g["gaze_y"].to_numpy(),
            g["validity"].to_numpy(),
            dispersion_threshold_px,
            minimum_duration_ms,
            max_gap_ms,
        )
        fx.insert(0, "fixation_id", np.arange(len(fx), dtype=int))
        fx.insert(0, "ui_id", ui_id)
        fx.insert(0, "trial_id", trial_id)
        fx.insert(0, "participant_id", pid)
        out.append(fx)
    if not out:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "trial_id",
                "ui_id",
                "fixation_id",
                "start_ms",
                "end_ms",
                "duration_ms",
                "centroid_x",
                "centroid_y",
                "n_samples",
                "aoi_id",
            ]
        )
    result = pd.concat(out, ignore_index=True)
    result["aoi_id"] = None
    return result
