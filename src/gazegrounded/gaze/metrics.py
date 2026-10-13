"""Trial-level and AOI-level gaze metrics.

Definitions (all times in ms from stimulus onset)
-------------------------------------------------
* **TTFF** (time to first fixation): ``start_ms`` of the first fixation on any task-relevant
  (target) AOI. ``NaN`` if the target was never fixated; ``ttff_censored_ms`` replaces NaN by
  the trial duration so models can use it (a right-censored value, flagged by
  ``target_fixated``).
* **Dwell time** of an AOI: sum of the durations of fixations assigned to it.
* **Visit**: a maximal run of consecutive fixations on the same AOI. **Revisits** of an AOI =
  ``max(visits - 1, 0)``; ``revisit_count`` sums this over AOIs.
* **AOI transitions**: number of changes between consecutive visits to *different* AOIs
  (fixations outside all AOIs are skipped).
* **Scanpath length**: summed distance between successive fixation centroids; also normalised
  by the screen diagonal.
* **Task-relevant attention**: share of total fixation time spent on target AOIs.
* **Task-relevant AOI coverage**: fraction of target AOIs fixated at least once.
* **Task-relevant sample proportion**: share of valid raw samples that fall inside target AOIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.gaze.aoi import assign_points_to_aois
from gazegrounded.gaze.scanpath import aoi_sequence, collapse_visits, scanpath_length_px


def aoi_level_metrics(
    fixations: pd.DataFrame, aois: pd.DataFrame, targets: list[str]
) -> pd.DataFrame:
    """One row per AOI of the interface with dwell, fixation count, visits, revisits, TTF."""
    f = fixations.sort_values("start_ms")
    total = float(f["duration_ms"].sum())
    visits = collapse_visits(aoi_sequence(f))
    visit_counts: dict[str, int] = {}
    for aoi, _ in visits:
        visit_counts[aoi] = visit_counts.get(aoi, 0) + 1
    rows = []
    for aoi_id in aois["aoi_id"]:
        on = f[f["aoi_id"] == aoi_id]
        n_visits = visit_counts.get(aoi_id, 0)
        dwell = float(on["duration_ms"].sum())
        rows.append(
            {
                "aoi_id": aoi_id,
                "is_target": aoi_id in targets,
                "dwell_ms": dwell,
                "dwell_share": dwell / total if total > 0 else 0.0,
                "fixation_count": len(on),
                "visit_count": n_visits,
                "revisit_count": max(n_visits - 1, 0),
                "time_to_first_fixation_ms": float(on["start_ms"].min()) if len(on) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def trial_gaze_metrics(
    fixations: pd.DataFrame,
    aois: pd.DataFrame,
    targets: list[str],
    screen_width: int,
    screen_height: int,
    trial_duration_ms: float,
    samples: pd.DataFrame | None = None,
    aoi_margin_px: float = 0.0,
) -> dict[str, float]:
    """Compute the trial-level gaze metrics described in the module docstring."""
    f = fixations.sort_values("start_ms")
    n = len(f)
    total = float(f["duration_ms"].sum()) if n else 0.0
    on_target = f["aoi_id"].isin(targets) if n else pd.Series(dtype=bool)
    ttff = float(f.loc[on_target, "start_ms"].min()) if n and on_target.any() else np.nan
    duration = max(float(trial_duration_ms), 1.0)
    visits = collapse_visits(aoi_sequence(f))
    visit_counts: dict[str, int] = {}
    for aoi, _ in visits:
        visit_counts[aoi] = visit_counts.get(aoi, 0) + 1
    length = scanpath_length_px(f)
    diag = float(np.hypot(screen_width, screen_height))
    fixated_targets = set(f.loc[on_target, "aoi_id"]) if n else set()

    m: dict[str, float] = {
        "fixation_count": n,
        "total_fixation_duration_ms": total,
        "mean_fixation_duration_ms": total / n if n else np.nan,
        "fixation_rate_per_s": n / (duration / 1000.0),
        "ttff_ms": ttff,
        "target_fixated": bool(not np.isnan(ttff)),
        "ttff_censored_ms": duration if np.isnan(ttff) else ttff,
        "ttff_norm": (duration if np.isnan(ttff) else ttff) / duration,
        "revisit_count": int(sum(max(v - 1, 0) for v in visit_counts.values())),
        "target_revisit_count": int(sum(max(visit_counts.get(t, 0) - 1, 0) for t in targets)),
        "aoi_transition_count": max(len(visits) - 1, 0),
        "scanpath_length_px": length,
        "scanpath_length_norm": length / diag,
        "task_relevant_attention": (
            float(f.loc[on_target, "duration_ms"].sum()) / total if total > 0 else 0.0
        ),
        "task_relevant_aoi_coverage": len(fixated_targets) / len(targets) if targets else np.nan,
        "fixations_outside_aoi_share": float(f["aoi_id"].isna().mean()) if n else np.nan,
    }
    if samples is not None and len(samples):
        valid = samples[samples["validity"] == 1]
        m["valid_sample_ratio"] = float(samples["validity"].mean())
        if len(valid):
            hits = assign_points_to_aois(
                valid["gaze_x"].to_numpy(),
                valid["gaze_y"].to_numpy(),
                aois,
                screen_width,
                screen_height,
                aoi_margin_px,
            )
            m["task_relevant_sample_proportion"] = float(np.isin(hits, targets).mean())
            m["in_any_aoi_sample_proportion"] = float(np.mean([h is not None for h in hits]))
        else:
            m["task_relevant_sample_proportion"] = np.nan
            m["in_any_aoi_sample_proportion"] = np.nan
    return m


def compute_gaze_metrics(
    ds, fixations: pd.DataFrame, aoi_margin_px: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial-level and AOI-level metrics for every trial of a standardised dataset.

    ``fixations`` must already have ``aoi_id`` assigned.
    """
    trial_rows, aoi_rows = [], []
    fix_by_trial = dict(tuple(fixations.groupby("trial_id")))
    samples_by_trial = (
        dict(tuple(ds.gaze_samples.groupby("trial_id"))) if ds.gaze_samples is not None else {}
    )
    for trial in ds.trials.itertuples(index=False):
        w, h = ds.ui_size(trial.ui_id)
        aois = ds.aois_for(trial.ui_id)
        targets = ds.targets_for_task(trial.task_id)
        f = fix_by_trial.get(trial.trial_id, fixations.iloc[0:0])
        s = samples_by_trial.get(trial.trial_id)
        duration = trial.completion_time_ms
        if duration is None or (isinstance(duration, float) and np.isnan(duration)):
            duration = (
                float(s["timestamp_ms"].max())
                if s is not None
                else float(f["end_ms"].max() if len(f) else 1.0)
            )
        m = trial_gaze_metrics(f, aois, targets, w, h, duration, s, aoi_margin_px)
        trial_rows.append(
            {
                "participant_id": trial.participant_id,
                "trial_id": trial.trial_id,
                "ui_id": trial.ui_id,
                "task_id": trial.task_id,
                **m,
            }
        )
        am = aoi_level_metrics(f, aois, targets)
        am.insert(0, "ui_id", trial.ui_id)
        am.insert(0, "trial_id", trial.trial_id)
        am.insert(0, "participant_id", trial.participant_id)
        aoi_rows.append(am)
    return pd.DataFrame(trial_rows), pd.concat(aoi_rows, ignore_index=True)
