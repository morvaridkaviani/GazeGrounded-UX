"""Leakage-safe cross-validation and the classification metrics used across the project.

Splitting
---------
Rows are trials; several trials come from the same participant (and the same interface).
Random row splits would let a model recognise a participant's idiosyncrasies (or an
interface) in the test fold. We therefore always split with ``GroupKFold``:

* ``group_by="participant_id"`` - generalisation to **new people** (default).
* ``group_by="ui_id"`` - generalisation to **new interfaces** (harder; checks whether a model
  merely memorises which screens are bad).

:func:`assert_no_group_overlap` is called on every fold and tested in the test-suite.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


def grouped_cv_splits(
    groups: Sequence, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold splits (shuffled, deterministic) with a leakage check on every fold."""
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        raise ValueError(f"Need at least 2 groups for grouped CV, got {n_groups}")
    splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(np.zeros(len(groups)), groups=groups):
        assert_no_group_overlap(groups[train_idx], groups[test_idx])
        yield train_idx, test_idx


def assert_no_group_overlap(train_groups: Sequence, test_groups: Sequence) -> None:
    overlap = set(np.asarray(train_groups)) & set(np.asarray(test_groups))
    if overlap:
        raise AssertionError(f"Data leakage: groups in both train and test: {sorted(overlap)[:5]}")


def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence,
    proba: np.ndarray | None = None,
    ordinal: bool = False,
) -> dict:
    """Accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix; optional AUC/ordinal."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    present = [lab for lab in labels if (y_true == lab).any()]
    if proba is not None and len(present) == len(labels) and len(labels) > 1:
        # roc_auc_score needs labels in sorted order; reorder the probability columns to match.
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in order]
        try:
            if len(labels) > 2:
                auc = roc_auc_score(
                    y_true,
                    proba[:, order],
                    multi_class="ovr",
                    average="macro",
                    labels=sorted_labels,
                )
            else:
                auc = roc_auc_score(y_true == labels[1], proba[:, 1])
            out["roc_auc_ovr_macro"] = float(auc)
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
    if ordinal:
        idx = {lab: i for i, lab in enumerate(labels)}
        t = np.array([idx[v] for v in y_true])
        q = np.array([idx[v] for v in y_pred])
        out["mae_ordinal"] = float(mean_absolute_error(t, q))
        out["quadratic_weighted_kappa"] = float(cohen_kappa_score(t, q, weights="quadratic"))
    return out


def summarize_folds(
    fold_metrics: list[dict],
    keys=("accuracy", "macro_f1", "weighted_f1", "quadratic_weighted_kappa"),
) -> dict:
    """Mean and SD across folds (fold-to-fold spread signals instability on small data)."""
    df = pd.DataFrame([{k: m.get(k, np.nan) for k in keys} for m in fold_metrics])
    return (
        {f"{k}_fold_mean": float(df[k].mean()) for k in keys if k in df}
        | {f"{k}_fold_sd": float(df[k].std(ddof=1)) for k in keys if k in df}
        | {"n_folds": len(fold_metrics)}
    )


def align_proba(proba: np.ndarray, fitted_classes: Sequence, labels: Sequence) -> np.ndarray:
    """Reorder/pad predict_proba columns to ``labels`` (a fold may miss a rare class)."""
    out = np.zeros((proba.shape[0], len(labels)))
    pos = {c: i for i, c in enumerate(fitted_classes)}
    for j, lab in enumerate(labels):
        if lab in pos:
            out[:, j] = proba[:, pos[lab]]
    return out
