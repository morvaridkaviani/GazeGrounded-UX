"""Research-question analyses run by ``gazegrounded.cli evaluate``.

RQ1  How closely does task-conditioned VLM relevance align with human gaze, compared with
     task-agnostic baselines (A centre bias, B spectral residual, C deep saliency model)?
RQ2  Does human/VLM attention divergence co-vary with behavioural friction indicators?
RQ3  Do multimodal features predict friction better than single modalities? (reads the
     latest ``train-friction`` run)
RQ4  Which modalities contribute most? (ablation from the same run)
Plus an automatic failure-case table: tasks where the VLM's top element is not the
annotated target.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image

from gazegrounded.analysis.correlations import (
    participant_level_correlation,
    repeated_measures_correlation,
    spearman_correlation,
)
from gazegrounded.analysis.saliency_metrics import compare_maps, fixations_to_grid
from gazegrounded.analysis.statistics import holm_correction, paired_comparison
from gazegrounded.gaze.heatmap import fixation_heatmap, grid_shape, normalize_map
from gazegrounded.utils import get_logger, latest_run_dir
from gazegrounded.vision.baselines import center_bias_map, spectral_residual_saliency

log = get_logger(__name__)

BASELINE_NAMES = {
    "A": "A · centre bias",
    "B": "B · spectral residual",
    "C": "C · deep saliency model",
    "D": "D · task-aware VLM",
}

# Pre-specified RQ2 analysis plan (fix before collecting real data; Holm within primary family).
# The early-window divergence is primary because whole-trial divergence is confounded by the
# number of fixations (found while validating the pipeline on synthetic data, before any real
# data existed). Whole-trial and AOI-level divergences are secondary.
RQ2_PRIMARY = ["attention_divergence_early"]
RQ2_SECONDARY = ["attention_divergence", "aoi_attention_divergence"]
RQ2_DIVERGENCES = RQ2_PRIMARY + RQ2_SECONDARY
RQ2_OUTCOMES = ["completion_time_ms", "revisit_count", "ttff_censored_ms"]
RQ2_PAIRED_PRIMARY = ["attention_divergence_early", "completion_time_ms"]
RQ2_PAIRED_SECONDARY = ["attention_divergence", "aoi_attention_divergence"]


def _saliency_predictions(cfg, ds) -> tuple[dict[str, np.ndarray], str | None]:
    """Baseline C maps for held-out screenshots of the latest saliency run (else empty)."""
    run = latest_run_dir(cfg.experiments_dir(), "saliency")
    if run is None or not (run / "best_model.pt").exists():
        return {}, None
    import torch

    from gazegrounded.vision.image_utils import image_to_tensor, load_rgb
    from gazegrounded.vision.saliency_model import logits_to_distribution
    from gazegrounded.vision.train_saliency import load_saliency_model

    held_out = json.loads((run / "split.json").read_text())["val"]
    model, size = load_saliency_model(run, "cpu")
    preds = {}
    for ui_id in held_out:
        if ui_id not in set(ds.uis["ui_id"]):
            continue
        w, h = ds.ui_size(ui_id)
        with torch.no_grad():
            p = logits_to_distribution(
                model(image_to_tensor(load_rgb(ds.image_path(ui_id)), size)[None])
            )[0, 0].numpy()
        rows, cols = grid_shape(w, h, cfg.gaze.map_scale)
        resized = Image.fromarray(p.astype(np.float32)).resize(
            (cols, rows), Image.Resampling.BILINEAR
        )
        preds[ui_id] = normalize_map(np.asarray(resized, dtype=float), "max")
    return preds, run.name


def attention_baselines(cfg, ds, fixations, vlm_maps) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """RQ1: compare each baseline map with the pooled human map of every task."""
    sal_maps, sal_run = _saliency_predictions(cfg, ds)
    g = cfg.gaze
    rows = []
    for task in ds.tasks.itertuples():
        w, h = ds.ui_size(task.ui_id)
        shape = grid_shape(w, h, g.map_scale)
        trial_ids = ds.trials.loc[ds.trials["task_id"] == task.task_id, "trial_id"]
        f = fixations[fixations["trial_id"].isin(trial_ids)]
        if f.empty:
            continue
        human = fixation_heatmap(f, w, h, g.heatmap_sigma_px, g.map_scale, normalize="sum")
        r, c = fixations_to_grid(f["centroid_x"], f["centroid_y"], g.map_scale)
        maps = {
            "A": center_bias_map(shape),
            "B": spectral_residual_saliency(Image.open(ds.image_path(task.ui_id)), shape),
            "D": vlm_maps.get(task.task_id),
        }
        if task.ui_id in sal_maps:
            maps["C"] = sal_maps[task.ui_id]
        for key, m in maps.items():
            if m is None:
                continue
            rows.append(
                {
                    "task_id": task.task_id,
                    "ui_id": task.ui_id,
                    "baseline_key": key,
                    "baseline": BASELINE_NAMES[key],
                    "n_fixations": len(f),
                    **compare_maps(human, m, r, c),
                }
            )
    per_task = pd.DataFrame(rows)
    summary = (
        per_task.groupby(["baseline_key", "baseline"])
        .agg(
            n=("task_id", "nunique"),
            cc_mean=("cc", "mean"),
            cc_min=("cc", "min"),
            cc_max=("cc", "max"),
            sim_mean=("sim", "mean"),
            kl_mean=("kl", "mean"),
            kl_min=("kl", "min"),
            kl_max=("kl", "max"),
            nss_mean=("nss", "mean"),
            jsd_mean=("jsd", "mean"),
        )
        .reset_index()
    )
    # Exploratory paired comparison across tasks: VLM (D) vs centre bias (A).
    wide = per_task.pivot_table(index="task_id", columns="baseline_key", values="cc")
    paired = {}
    if {"A", "D"} <= set(wide.columns):
        both = wide[["A", "D"]].dropna()
        tmp = both.reset_index().melt(id_vars="task_id", var_name="cond", value_name="cc")
        paired = paired_comparison(
            tmp,
            "cc",
            subject="task_id",
            condition="cond",
            cond_a="A",
            cond_b="D",
            test=cfg.stats.paired_test,
        )
        paired["note"] = "exploratory: tasks as units; n is small"
    meta = {
        "saliency_run": sal_run,
        "saliency_heldout_note": "Baseline C is evaluated only on screenshots held out from "
        "its training run (n shown per baseline).",
    }
    return per_task, summary, {"paired_D_vs_A_cc": paired, **meta}


def rq2_statistics(cfg, features: pd.DataFrame) -> pd.DataFrame:
    """RQ2: repeated-measures correlations (primary), exploratory trial-level Spearman,
    participant-level correlations, and paired clean-vs-friction comparisons."""
    rows = []
    for x in RQ2_DIVERGENCES:
        level = "primary" if x in RQ2_PRIMARY else "secondary"
        for y in RQ2_OUTCOMES:
            for fn, role in (
                (repeated_measures_correlation, level),
                (
                    lambda d, a, b: spearman_correlation(d[a], d[b]),
                    "exploratory (non-independent trials)",
                ),
                (participant_level_correlation, "secondary"),
            ):
                res = fn(features, x, y)
                rows.append({"family": "rq2_correlation", "x": x, "y": y, "role": role, **res})
    if {"clean", "friction"} <= set(features["condition"]):
        for v in RQ2_PAIRED_PRIMARY + RQ2_PAIRED_SECONDARY:
            res = paired_comparison(features, v, test=cfg.stats.paired_test)
            role = "primary" if v in RQ2_PAIRED_PRIMARY else "secondary"
            rows.append(
                {
                    "family": "rq2_paired",
                    "x": "condition",
                    "y": v,
                    "role": role,
                    "method": res["test"],
                    **res,
                }
            )
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for family in df["family"].unique():
        mask = (df["family"] == family) & (df["role"] == "primary")
        if cfg.stats.correction == "holm":
            df.loc[mask, "p_holm"] = holm_correction(df.loc[mask, "p_value"].tolist())
    return df


def friction_results(cfg) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    run = latest_run_dir(cfg.experiments_dir(), "friction")
    if run is None:
        return None, None, None
    comp = pd.read_csv(run / "modality_comparison.csv")
    abl = pd.read_csv(run / "ablation.csv") if (run / "ablation.csv").exists() else None
    return comp, abl, run.name


def failure_cases(
    ds,
    features: pd.DataFrame,
    vlm_tasks: pd.DataFrame,
    elements: pd.DataFrame,
    aoi_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Tasks where the VLM's highest-weighted element is not an annotated target AOI."""
    rows = []
    for t in vlm_tasks.itertuples():
        if bool(t.vlm_top_is_target):
            continue
        el = elements[elements["task_id"] == t.task_id].sort_values("weight", ascending=False)
        targets = ds.targets_for_task(t.task_id)
        trial_ids = ds.trials.loc[ds.trials["task_id"] == t.task_id, "trial_id"]
        am = aoi_metrics[aoi_metrics["trial_id"].isin(trial_ids)]
        target_share = (
            float(am[am["aoi_id"].isin(targets)].groupby("trial_id")["dwell_share"].sum().mean())
            if len(am)
            else np.nan
        )
        top = el.iloc[0] if len(el) else None
        top_share = (
            float(am[am["aoi_id"] == top["matched_aoi_id"]]["dwell_share"].mean())
            if top is not None and isinstance(top["matched_aoi_id"], str)
            else np.nan
        )
        f = features[features["task_id"] == t.task_id]
        rows.append(
            {
                "task_id": t.task_id,
                "ui_id": t.ui_id,
                "task": ds.tasks.set_index("task_id").loc[t.task_id, "task_description"],
                "target_aois": ";".join(targets),
                "vlm_top_element": top["label"] if top is not None else None,
                "vlm_top_matched_aoi": top["matched_aoi_id"] if top is not None else None,
                "vlm_top_relevance": float(top["task_relevance"]) if top is not None else np.nan,
                "vlm_target_hit_any_relevant": bool(t.vlm_target_hit),
                "human_dwell_share_on_target": target_share,
                "human_dwell_share_on_vlm_top": top_share,
                "mean_attention_divergence": float(f["attention_divergence"].mean()),
            }
        )
    return pd.DataFrame(rows)
