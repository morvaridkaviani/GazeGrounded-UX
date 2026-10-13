"""Explanations for the friction models.

* Logistic regression: coefficients on **standardised** features (comparable across features)
  and per-trial *local contributions* ``coef[class] * x_standardised`` - how much each feature
  pushed this trial towards the predicted class relative to an average trial.
* Random forest: impurity-based importances (fast, biased towards high-cardinality features).
* Any model: permutation importance on held-out folds (drop in macro-F1 when a raw input
  column is shuffled). The text column is permuted as a whole, giving one importance for text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def transformed_feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["pre"].get_feature_names_out())


def standardized_coefficients(pipe: Pipeline) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    names = transformed_feature_names(pipe)
    rows = []
    for i, cls in enumerate(clf.classes_):
        for name, c in zip(names, clf.coef_[i], strict=True):
            rows.append({"feature": name, "class": cls, "coefficient": float(c)})
    return pd.DataFrame(rows)


def forest_importances(pipe: Pipeline) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    return pd.DataFrame(
        {"feature": transformed_feature_names(pipe), "importance": clf.feature_importances_}
    )


def linear_contributions(pipe: Pipeline, X: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Long table: one row per (trial, feature) with its contribution to the predicted class."""
    clf = pipe.named_steps["clf"]
    Z = pipe.named_steps["pre"].transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    names = transformed_feature_names(pipe)
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    rows = []
    for r, pred in enumerate(predicted):
        coefs = clf.coef_[cls_idx[pred]]
        for name, z, c in zip(names, Z[r], coefs, strict=True):
            rows.append(
                {"row": r, "feature": name, "value_std": float(z), "contribution": float(c * z)}
            )
    return pd.DataFrame(rows)


def fold_permutation_importance(
    pipe: Pipeline, X_test: pd.DataFrame, y_test, columns: list[str], n_repeats: int, seed: int
) -> pd.DataFrame:
    """Permutation importance of raw input columns on one held-out fold (macro-F1 drop)."""
    text_cols = {"feedback_text", "feedback_category"}
    groups = [c for c in columns if c not in text_cols]
    has_text = any(c in text_cols for c in columns)
    rng = np.random.default_rng(seed)
    base = _macro_f1(pipe, X_test, y_test)
    rows = []
    for col in groups + (["text"] if has_text else []):
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            cols = [c for c in columns if c in text_cols] if col == "text" else [col]
            Xp[cols] = Xp[cols].to_numpy()[perm]
            drops.append(base - _macro_f1(pipe, Xp, y_test))
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops)),
                "importance_sd": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows)


def _macro_f1(pipe, X, y) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pipe.predict(X), average="macro", zero_division=0))
