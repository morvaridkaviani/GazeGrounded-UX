"""Human gaze vs VLM task relevance: trial-level and task-level alignment.

Map level  : human duration-weighted fixation heatmap vs VLM Task-Relevance Map
             (metrics from :mod:`gazegrounded.analysis.saliency_metrics`).
AOI level  : human dwell distribution over AOIs (+ an "outside" bin) vs the share of VLM map
             mass falling in each AOI. ``aoi_attention_divergence`` = JSD (base 2) of the two.
Early      : ``attention_divergence_early`` = map JSD using only the first K fixations (after
             dropping the onset fixation, which reflects where gaze started). Whole-trial
             divergence is confounded by fixation count: a sparse map from a short trial is
             far from any smooth map, so easy trials look "divergent". The early window
             measures what the construct is about - where attention goes first.
Behavioural: ``vlm_relevant_dwell_share`` = share of the participant's fixation time spent
             inside boxes the VLM rated task-relevant (relevance >= threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.analysis.saliency_metrics import (
    compare_maps,
    distribution_divergence,
    fixations_to_grid,
    jsd,
    to_distribution,
)
from gazegrounded.gaze.aoi import assign_points_to_aois
from gazegrounded.gaze.heatmap import fixation_heatmap, grid_shape
from gazegrounded.vlm.schemas import VLMAnalysis

OUTSIDE = "__outside__"


def aoi_label_grid(aois: pd.DataFrame, width: int, height: int, scale: float) -> np.ndarray:
    """AOI id of each grid cell (cell centre, smallest containing AOI; OUTSIDE otherwise)."""
    rows, cols = grid_shape(width, height, scale)
    yy, xx = np.mgrid[0:rows, 0:cols]
    x = (xx.ravel() + 0.5) / scale
    y = (yy.ravel() + 0.5) / scale
    labels = assign_points_to_aois(x, y, aois, width, height)
    return np.array([OUTSIDE if a is None else a for a in labels], dtype=object).reshape(rows, cols)


def map_mass_by_aoi(m: np.ndarray, label_grid: np.ndarray) -> dict[str, float]:
    p = to_distribution(m)
    out: dict[str, float] = {}
    for label in np.unique(label_grid):
        out[str(label)] = float(p[label_grid == label].sum())
    return out


def human_dwell_distribution(fixations: pd.DataFrame) -> dict[str, float]:
    if fixations.empty:
        return {}
    aoi = fixations["aoi_id"].where(fixations["aoi_id"].notna(), OUTSIDE)
    dwell = fixations["duration_ms"].groupby(aoi).sum()
    return (dwell / dwell.sum()).to_dict()


def vlm_task_features(
    analysis: VLMAnalysis, elements: pd.DataFrame, targets: list[str], threshold: float
) -> dict[str, float]:
    """Interface-level VLM features (do not use gaze)."""
    rel = np.array([e.task_relevance for e in analysis.elements])
    relevant = elements[elements["task_relevance"] >= threshold] if len(elements) else elements
    top = elements.sort_values("weight", ascending=False).head(1) if len(elements) else elements
    return {
        "vlm_parse_status": analysis.parse_status,
        "vlm_element_count": len(analysis.elements),
        "task_relevant_region_count": int((rel >= threshold).sum()) if len(rel) else 0,
        "max_task_relevance": float(rel.max()) if len(rel) else 0.0,
        "mean_task_relevance": float(rel.mean()) if len(rel) else 0.0,
        "vlm_relevance_entropy": _entropy(rel),
        # Grounding checks against human-annotated target AOIs (evaluation of the VLM itself):
        "vlm_target_hit": bool(len(relevant) and relevant["matched_aoi_id"].isin(targets).any()),
        "vlm_top_is_target": bool(len(top) and top["matched_aoi_id"].isin(targets).any()),
    }


def _entropy(values: np.ndarray) -> float:
    if len(values) == 0 or values.sum() <= 0:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def trial_alignment(
    fixations: pd.DataFrame,
    vlm_map: np.ndarray,
    analysis: VLMAnalysis,
    label_grid: np.ndarray,
    vlm_aoi_mass: dict[str, float],
    width: int,
    height: int,
    sigma_px: float,
    scale: float,
    threshold: float,
    early_k: int = 8,
    drop_first: bool = True,
) -> dict[str, float]:
    """Alignment metrics between one trial's gaze and the VLM map for that (UI, task)."""
    if fixations.empty:
        return {"n_fixations_used": 0}
    fixations = fixations.sort_values("start_ms")
    human = fixation_heatmap(fixations, width, height, sigma_px, scale, normalize="sum")
    early = fixations.iloc[1 if drop_first and len(fixations) > 1 else 0 :][:early_k]
    early_map = fixation_heatmap(early, width, height, sigma_px, scale, normalize="sum")
    rows, cols = fixations_to_grid(fixations["centroid_x"], fixations["centroid_y"], scale)
    m = compare_maps(human, vlm_map, rows, cols)
    human_aoi = human_dwell_distribution(fixations)
    keys = sorted(set(human_aoi) | set(vlm_aoi_mass))
    h_vec = np.array([human_aoi.get(k, 0.0) for k in keys])
    v_vec = np.array([vlm_aoi_mass.get(k, 0.0) for k in keys])
    aoi_rank = (
        float(pd.Series(h_vec).corr(pd.Series(v_vec), method="spearman"))
        if len(keys) > 2
        else np.nan
    )
    # Fixation time inside boxes the VLM rated task-relevant.
    boxes = [e.bbox for e in analysis.elements if e.task_relevance >= threshold]
    x = fixations["centroid_x"].to_numpy() / width
    y = fixations["centroid_y"].to_numpy() / height
    inside = np.zeros(len(fixations), dtype=bool)
    for b in boxes:
        inside |= (x >= b.x_min) & (x <= b.x_max) & (y >= b.y_min) & (y <= b.y_max)
    dur = fixations["duration_ms"].to_numpy()
    return {
        "n_fixations_used": len(fixations),
        "vlm_human_cc": m["cc"],
        "vlm_human_spearman": m["spearman"],
        "vlm_human_cosine": m["cosine"],
        "vlm_human_similarity": m["sim"],
        "vlm_human_kl": m["kl"],
        "attention_divergence": m["jsd"],
        "attention_divergence_early": jsd(early_map, vlm_map),
        "vlm_nss": m.get("nss", np.nan),
        "aoi_attention_divergence": distribution_divergence(human_aoi, vlm_aoi_mass),
        "aoi_rank_agreement": aoi_rank,
        "vlm_relevant_dwell_share": float(dur[inside].sum() / dur.sum()) if dur.sum() else 0.0,
    }
