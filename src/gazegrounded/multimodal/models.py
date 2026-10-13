"""Interpretable baseline classifiers for UX friction (low / medium / high)."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_estimator(name: str, cfg):
    """``logistic_regression`` | ``random_forest`` | ``majority`` (chance-level reference).

    Class weights are balanced so that a model cannot look good by predicting the most
    frequent friction level.
    """
    fm = cfg.friction_model
    if name == "logistic_regression":
        return LogisticRegression(C=fm.logreg_C, max_iter=5000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=fm.rf_n_estimators,
            max_depth=fm.rf_max_depth,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model {name!r}")
