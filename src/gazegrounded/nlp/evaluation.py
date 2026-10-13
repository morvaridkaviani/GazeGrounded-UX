"""End-to-end feedback analysis: classifier evaluation, top terms, exploratory clusters.

Small-sample caveat: with tens to a few hundred comments and eight categories, per-class
scores swing between folds. Fold-level SDs and per-class supports are always reported, and
categories with very few examples should be read as anecdotes, not results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gazegrounded.analysis.model_evaluation import classification_metrics, summarize_folds
from gazegrounded.config import config_to_dict
from gazegrounded.nlp.classifier import (
    cluster_feedback,
    cross_validated_predictions,
    make_feedback_classifier,
    top_terms_per_class,
)
from gazegrounded.nlp.preprocessing import prepare_feedback
from gazegrounded.utils import get_logger, make_run_dir, save_json_safe, save_yaml

log = get_logger(__name__)

MIN_EXAMPLES_PER_CLASS = 5


def run_nlp_analysis(cfg, trials: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    """Evaluate the feedback classifier with participant-grouped CV and save artefacts."""
    from gazegrounded.visualization.models import confusion_matrix_figure, top_terms_figure
    from gazegrounded.visualization.style import save_figure

    df = prepare_feedback(trials)
    labelled = df[df["has_feedback"] & df["feedback_category"].notna()].copy()
    counts = labelled["feedback_category"].value_counts()
    rare = counts[counts < MIN_EXAMPLES_PER_CLASS]
    if len(rare):
        log.warning(
            "Categories with < %d examples (unstable estimates): %s",
            MIN_EXAMPLES_PER_CLASS,
            rare.to_dict(),
        )
    run_dir = make_run_dir(cfg.experiments_dir(), "nlp-feedback")
    save_yaml(config_to_dict(cfg), run_dir / "config.yaml")

    oof, folds = cross_validated_predictions(
        labelled["feedback_clean"],
        labelled["feedback_category"],
        labelled["participant_id"],
        cfg.nlp,
        cfg.seed,
    )
    classes = sorted(labelled["feedback_category"].unique())
    proba = oof[[f"p_{c}" for c in classes]].to_numpy(dtype=float)
    overall = classification_metrics(labelled["feedback_category"], oof["pred"], classes, proba)
    per_fold = [classification_metrics(f["y_true"], f["y_pred"], classes) for f in folds]
    majority = float(counts.max() / counts.sum())
    metrics = {
        "data_kind": cfg.data.data_kind,
        "n_trials": int(len(df)),
        "n_with_feedback": int(df["has_feedback"].sum()),
        "n_labelled": int(len(labelled)),
        "class_counts": counts.to_dict(),
        "majority_class_accuracy": majority,
        "split": f"GroupKFold by participant_id ({len(folds)} folds)",
        "out_of_fold": overall,
        "folds": summarize_folds(per_fold, keys=("accuracy", "macro_f1", "weighted_f1")),
    }
    if cfg.is_synthetic:
        metrics["note"] = (
            "Synthetic feedback is generated from a handful of templates per "
            "category, so near-perfect scores are expected and meaningless."
        )
    save_json_safe(metrics, run_dir / "metrics.json")

    preds = labelled[["trial_id", "participant_id", "feedback_text", "feedback_category"]].join(oof)
    preds.to_csv(run_dir / "predictions.csv", index=False)

    final = make_feedback_classifier(cfg.nlp).fit(
        labelled["feedback_clean"], labelled["feedback_category"]
    )
    terms = top_terms_per_class(final)
    terms.to_csv(run_dir / "feature_importance.csv", index=False)

    cluster_labels, clusters = cluster_feedback(labelled["feedback_clean"], cfg.nlp, cfg.seed)
    clusters.to_csv(run_dir / "clusters.csv", index=False)
    preds.assign(cluster=cluster_labels).to_csv(run_dir / "predictions.csv", index=False)

    fig_dir = cfg.figures_dir() / "nlp"
    for target in (run_dir, fig_dir):
        save_figure(
            confusion_matrix_figure(
                overall["confusion_matrix"],
                classes,
                f"Feedback categories (out-of-fold, n={overall['n']})",
            ),
            target / "confusion_matrix.png",
            cfg.is_synthetic,
        )
        save_figure(
            top_terms_figure(terms, "Most indicative terms per category (TF-IDF + LR)"),
            target / "top_terms.png",
            cfg.is_synthetic,
        )
    log.info(
        "NLP: n=%d, out-of-fold macro-F1 %.2f (majority-class acc %.2f) -> %s",
        overall["n"],
        overall["macro_f1"],
        majority,
        run_dir,
    )
    return run_dir, preds
