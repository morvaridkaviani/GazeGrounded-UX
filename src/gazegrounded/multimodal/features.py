"""Trial-level multimodal feature table.

One row per trial, columns grouped by modality (see :data:`MODALITY_FEATURES`). Notes:

* **gaze** features use the researcher-annotated task-relevant AOIs (e.g. TTFF on the target).
* **interaction** features come from the task log.
* **vlm** features are of two kinds: interface-level estimates that use no human data
  (relevance statistics) and *alignment* features that compare the VLM map with this trial's
  gaze (similarity / divergence). Both are needed to test RQ2/RQ3.
* **text** is the raw feedback; it is converted to features inside each CV fold.
* **visual** features describe the screenshot only (UI-level). With few interfaces they act
  almost like an interface identifier, which is why the UI-grouped split matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.utils import get_logger

log = get_logger(__name__)

ID_COLUMNS = ["trial_id", "participant_id", "ui_id", "task_id", "condition"]
LABEL_COLUMN = "friction_label"
TEXT_COLUMNS = ["feedback_text", "feedback_category"]

MODALITY_FEATURES: dict[str, list[str]] = {
    "gaze": [
        "ttff_censored_ms",
        "target_fixated",
        "fixation_count",
        "mean_fixation_duration_ms",
        "fixation_rate_per_s",
        "revisit_count",
        "aoi_transition_count",
        "scanpath_length_norm",
        "task_relevant_attention",
        "task_relevant_aoi_coverage",
    ],
    "interaction": ["completion_time_ms", "click_count", "error_count", "task_success"],
    "vlm": [
        "vlm_human_similarity",
        "attention_divergence",
        "attention_divergence_early",
        "aoi_attention_divergence",
        "vlm_relevant_dwell_share",
        "task_relevant_region_count",
        "max_task_relevance",
        "mean_task_relevance",
    ],
    "text": [],  # handled by a text transformer on TEXT_COLUMNS
    "visual": ["ui_aoi_count", "ui_edge_density", "ui_saliency_entropy"],
}

FEATURE_SETS: dict[str, list[str]] = {
    "interaction": ["interaction"],
    "gaze": ["gaze"],
    "vlm": ["vlm"],
    "text": ["text"],
    "visual": ["visual"],
    "multimodal": ["gaze", "interaction", "vlm", "text"],
}


def numeric_columns(modalities: list[str]) -> list[str]:
    return [c for m in modalities for c in MODALITY_FEATURES[m]]


def ui_visual_features(image_path, aois: pd.DataFrame) -> dict[str, float]:
    """Screenshot descriptors: AOI count, Canny edge density, spectral-saliency entropy."""
    import cv2

    from gazegrounded.vision.baselines import spectral_residual_saliency

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, 50, 150)
    sal = spectral_residual_saliency(np.dstack([gray] * 3), (90, 144))
    p = sal / sal.sum() if sal.sum() > 0 else np.full_like(sal, 1 / sal.size)
    entropy = float(-(p * np.log2(p + 1e-12)).sum() / np.log2(p.size))
    return {
        "ui_aoi_count": len(aois),
        "ui_edge_density": float((edges > 0).mean()),
        "ui_saliency_entropy": entropy,
    }


def build_feature_table(cfg, ds) -> pd.DataFrame:
    """Join trials, gaze metrics, VLM features/alignment and visual features (one row/trial)."""
    interim = cfg.interim_dir()
    gaze = pd.read_csv(interim / "gaze_metrics.csv")
    align = pd.read_csv(interim / "vlm" / "vlm_alignment.csv")
    vlm_tasks = pd.read_csv(interim / "vlm" / "vlm_task_features.csv")

    t = ds.trials.copy()
    keep = ID_COLUMNS + [
        LABEL_COLUMN,
        "label_source",
        *TEXT_COLUMNS,
        "completion_time_ms",
        "click_count",
        "error_count",
        "task_success",
    ]
    t = t[[c for c in keep if c in t.columns]]
    df = t.merge(
        gaze.drop(columns=["participant_id", "ui_id", "task_id"]), on="trial_id", how="left"
    )
    df = df.merge(align.drop(columns=["participant_id", "task_id"]), on="trial_id", how="left")
    df = df.merge(vlm_tasks.drop(columns=["ui_id"]), on="task_id", how="left")
    visual = pd.DataFrame(
        [
            {"ui_id": u, **ui_visual_features(ds.image_path(u), ds.aois_for(u))}
            for u in ds.uis["ui_id"]
        ]
    )
    df = df.merge(visual, on="ui_id", how="left")
    for col in ("task_success", "target_fixated"):
        if col in df:
            df[col] = df[col].astype(float)
    missing = [c for c in numeric_columns(list(MODALITY_FEATURES)) if c not in df.columns]
    if missing:
        log.warning("Feature columns missing (will be skipped): %s", missing)
    return df


def available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Columns present and not entirely missing."""
    return [c for c in columns if c in df.columns and df[c].notna().any()]
