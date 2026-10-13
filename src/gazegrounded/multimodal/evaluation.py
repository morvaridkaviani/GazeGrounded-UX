"""Friction-model experiments: modality comparison (RQ3), ablation (RQ4), explanations.

For every split strategy (participant-grouped; optionally interface-grouped), every model
type and every feature set, we collect out-of-fold predictions with GroupKFold and compute
macro-F1, weighted F1, accuracy, ROC-AUC (one-vs-rest) and the ordinal metrics MAE and
quadratic weighted kappa. A majority-class model gives the chance reference.

Outputs (``experiments/<timestamp>-friction/``): ``config.yaml``, ``metrics.json``,
``modality_comparison.csv``, ``ablation.csv``, ``predictions.csv``, ``feature_importance.csv``,
``contributions.csv`` and ``final_model.joblib`` (multimodal LR fitted on all data).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gazegrounded.analysis.model_evaluation import (
    align_proba,
    classification_metrics,
    grouped_cv_splits,
    summarize_folds,
)
from gazegrounded.config import config_to_dict
from gazegrounded.data.schemas import FRICTION_LABELS
from gazegrounded.multimodal.explainability import (
    fold_permutation_importance,
    forest_importances,
    linear_contributions,
    standardized_coefficients,
)
from gazegrounded.multimodal.features import FEATURE_SETS, LABEL_COLUMN
from gazegrounded.multimodal.fusion import make_pipeline_for
from gazegrounded.multimodal.models import make_estimator
from gazegrounded.utils import get_logger, make_run_dir, save_json_safe, save_yaml

log = get_logger(__name__)


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    cfg,
    group_by: str = "participant_id",
    explain: bool = False,
    modalities: list[str] | None = None,
) -> dict:
    """Grouped-CV evaluation of one (feature set, model) pair.

    ``modalities`` overrides the modalities of ``feature_set`` (used by the ablation).
    """
    labels = [lab for lab in FRICTION_LABELS if (df[LABEL_COLUMN] == lab).any()]
    y = df[LABEL_COLUMN].to_numpy()
    oof_pred = np.empty(len(df), dtype=object)
    oof_proba = np.zeros((len(df), len(labels)))
    fold_id = np.full(len(df), -1)
    fold_metrics, perm, contribs = [], [], []
    n_splits = cfg.friction_model.cv.n_splits
    for k, (tr, te) in enumerate(grouped_cv_splits(df[group_by].to_numpy(), n_splits, cfg.seed)):
        pipe, cols = make_pipeline_for(
            df, modalities or feature_set, make_estimator(model_name, cfg), cfg
        )
        pipe.fit(df.iloc[tr][cols], y[tr])
        X_te = df.iloc[te][cols]
        pred = pipe.predict(X_te)
        oof_pred[te] = pred
        oof_proba[te] = align_proba(pipe.predict_proba(X_te), pipe.classes_, labels)
        fold_id[te] = k
        fold_metrics.append(classification_metrics(y[te], pred, labels, ordinal=True))
        if explain:
            imp = fold_permutation_importance(
                pipe, X_te, y[te], cols, cfg.friction_model.permutation_repeats, cfg.seed
            )
            perm.append(imp.assign(fold=k))
            if model_name == "logistic_regression":
                c = linear_contributions(pipe, X_te, pred)
                c["trial_id"] = df.iloc[te]["trial_id"].to_numpy()[c["row"]]
                contribs.append(c.drop(columns="row"))
    overall = classification_metrics(y, oof_pred, labels, oof_proba, ordinal=True)
    result = {
        "feature_set": feature_set,
        "model": model_name,
        "group_by": group_by,
        "overall": overall,
        "folds": summarize_folds(fold_metrics),
        "predictions": pd.DataFrame(
            {
                "trial_id": df["trial_id"].to_numpy(),
                "true": y,
                "pred": oof_pred,
                "fold": fold_id,
                **{f"p_{lab}": oof_proba[:, j] for j, lab in enumerate(labels)},
            }
        ),
    }
    if explain:
        result["permutation_importance"] = (
            pd.concat(perm).groupby("feature", as_index=False)[["importance_mean"]].mean()
            if perm
            else pd.DataFrame()
        )
        result["contributions"] = pd.concat(contribs) if contribs else pd.DataFrame()
    return result


def _row(res: dict) -> dict:
    o, f = res["overall"], res["folds"]
    return {
        "group_by": res["group_by"],
        "model": res["model"],
        "modality": res["feature_set"],
        "n": o["n"],
        "macro_f1": o["macro_f1"],
        "weighted_f1": o["weighted_f1"],
        "accuracy": o["accuracy"],
        "roc_auc_ovr_macro": o.get("roc_auc_ovr_macro", np.nan),
        "mae_ordinal": o["mae_ordinal"],
        "quadratic_weighted_kappa": o["quadratic_weighted_kappa"],
        **{k: v for k, v in f.items() if k != "n_folds"},
        "n_folds": f["n_folds"],
    }


def run_friction_experiment(cfg, df: pd.DataFrame) -> Path:
    fm = cfg.friction_model
    df = df[df[LABEL_COLUMN].notna()].reset_index(drop=True)
    sources = sorted(df["label_source"].dropna().unique()) if "label_source" in df else []
    if fm.label_source == "human" and any(s != "human" for s in sources):
        log.warning("Config expects human labels but data has label_source=%s", sources)
    run_dir = make_run_dir(cfg.experiments_dir(), "friction")
    save_yaml(config_to_dict(cfg), run_dir / "config.yaml")

    splits = ["participant_id"] + (
        ["ui_id"] if fm.also_group_by_ui and df["ui_id"].nunique() > 2 else []
    )
    feature_sets = [m for m in fm.modalities if m in FEATURE_SETS]
    rows, preds, importances, contributions = [], [], [], []
    for group_by in splits:
        rows.append(
            _row(evaluate_feature_set(df, "interaction", "majority", cfg, group_by))
            | {"modality": "majority_baseline", "model": "majority"}
        )
        for model in fm.models:
            for fs in feature_sets:
                explain = fs == "multimodal" and group_by == "participant_id"
                res = evaluate_feature_set(df, fs, model, cfg, group_by, explain=explain)
                rows.append(_row(res))
                preds.append(res["predictions"].assign(model=model, modality=fs, group_by=group_by))
                if explain:
                    importances.append(
                        res["permutation_importance"].assign(
                            model=model, kind="permutation_macro_f1_drop"
                        )
                    )
                    if len(res["contributions"]):
                        contributions.append(res["contributions"].assign(model=model))
                log.info(
                    "%-12s %-20s %-12s macro-F1 %.2f (fold SD %.2f)",
                    group_by,
                    model,
                    fs,
                    res["overall"]["macro_f1"],
                    res["folds"].get("macro_f1_fold_sd", np.nan),
                )

    # RQ4 ablation: leave one modality out of the multimodal LR (participant split).
    ablation = []
    base_mods = FEATURE_SETS["multimodal"]
    for drop in base_mods:
        kept = [m for m in base_mods if m != drop]
        res = evaluate_feature_set(
            df, f"multimodal_minus_{drop}", "logistic_regression", cfg, modalities=kept
        )
        ablation.append({"dropped_modality": drop, **_row(res)})

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "modality_comparison.csv", index=False)
    pd.DataFrame(ablation).to_csv(run_dir / "ablation.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
    if contributions:
        pd.concat(contributions, ignore_index=True).to_csv(
            run_dir / "contributions.csv", index=False
        )

    # Global explanations from models fitted on all data (descriptive, not evaluation).
    final = {}
    for model in fm.models:
        pipe, cols = make_pipeline_for(df, "multimodal", make_estimator(model, cfg), cfg)
        pipe.fit(df[cols], df[LABEL_COLUMN])
        final[model] = pipe
        if model == "logistic_regression":
            importances.append(
                standardized_coefficients(pipe).assign(model=model, kind="standardized_coefficient")
            )
        else:
            importances.append(
                forest_importances(pipe).assign(model=model, kind="impurity_importance")
            )
    pd.concat(importances, ignore_index=True).to_csv(
        run_dir / "feature_importance.csv", index=False
    )
    joblib.dump(final, run_dir / "final_model.joblib")

    metrics = {
        "data_kind": cfg.data.data_kind,
        "label_source": sources,
        "n_trials": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "n_interfaces": int(df["ui_id"].nunique()),
        "class_counts": df[LABEL_COLUMN].value_counts().to_dict(),
        "splits": {g: f"GroupKFold by {g}" for g in splits},
        "results": results.to_dict(orient="records"),
        "ablation": ablation,
    }
    if cfg.is_synthetic:
        metrics["note"] = (
            "SYNTHETIC DEMO DATA - the generator built friction into gaze, "
            "interaction and text by design; these scores only show that the "
            "pipeline runs."
        )
    save_json_safe(metrics, run_dir / "metrics.json")
    _figures(cfg, run_dir, results, preds, importances)
    return run_dir


def _figures(cfg, run_dir: Path, results: pd.DataFrame, preds: list, importances: list) -> None:
    from gazegrounded.visualization.models import (
        confusion_matrix_figure,
        feature_importance_figure,
        modality_comparison_figure,
    )
    from gazegrounded.visualization.style import save_figure

    fig_dir = cfg.figures_dir() / "friction"
    synthetic = cfg.is_synthetic
    for group_by in results["group_by"].unique():
        r = results[(results["group_by"] == group_by) & (results["model"] != "majority")]
        chance = results[(results["group_by"] == group_by) & (results["model"] == "majority")][
            "macro_f1"
        ]
        who = "new participants" if group_by == "participant_id" else "new interfaces"
        fig = modality_comparison_figure(
            r,
            "macro_f1",
            f"Friction prediction by modality — generalising to {who}",
            chance=float(chance.iloc[0]) if len(chance) else None,
        )
        for target in (run_dir, fig_dir):
            save_figure(fig, target / f"modality_comparison_{group_by}.png", synthetic)
    p = pd.concat(preds)
    mm = p[
        (p["modality"] == "multimodal")
        & (p["model"] == "logistic_regression")
        & (p["group_by"] == "participant_id")
    ]
    if len(mm):
        labels = [lab for lab in FRICTION_LABELS if (mm["true"] == lab).any()]
        cm = classification_metrics(mm["true"], mm["pred"], labels)["confusion_matrix"]
        fig = confusion_matrix_figure(cm, labels, "Multimodal LR (out-of-fold, by participant)")
        for target in (run_dir, fig_dir):
            save_figure(fig, target / "confusion_matrix_multimodal.png", synthetic)
    imp = pd.concat(importances)
    perm = imp[
        (imp["kind"] == "permutation_macro_f1_drop") & (imp["model"] == "logistic_regression")
    ]
    if len(perm):
        fig = feature_importance_figure(
            perm,
            "importance_mean",
            "Permutation importance (macro-F1 drop, held-out folds)",
            signed=False,
        )
        for target in (run_dir, fig_dir):
            save_figure(fig, target / "feature_importance_permutation.png", synthetic)
    coefs = imp[(imp["kind"] == "standardized_coefficient") & (imp["class"] == "high")]
    if len(coefs):
        fig = feature_importance_figure(
            coefs,
            "coefficient",
            "Standardised LR coefficients for 'high' friction (red = pushes towards high)",
        )
        for target in (run_dir, fig_dir):
            save_figure(fig, target / "coefficients_high_friction.png", synthetic)
