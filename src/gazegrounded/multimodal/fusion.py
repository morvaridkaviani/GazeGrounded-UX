"""Early fusion: one sklearn pipeline per feature set (modality or combination).

Numeric features -> median imputation -> standardisation. Text -> either predicted
feedback-category probabilities (when coded categories exist) or TF-IDF -> truncated SVD.
Everything is fitted inside each training fold, so no statistic of a test fold leaks.

Early (feature-level) fusion was chosen over late (decision-level) fusion because, with small
datasets, a single interpretable model over concatenated features is easier to inspect and
less prone to overfitting than stacking several per-modality models.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from gazegrounded.data.schemas import FEEDBACK_CATEGORIES
from gazegrounded.multimodal.features import FEATURE_SETS, available, numeric_columns
from gazegrounded.nlp.features import FeedbackCategoryFeatures, make_tfidf


class _TextColumn:
    """Select the feedback text column as a 1-D series for TF-IDF (picklable)."""

    def __call__(self, X):
        return X["feedback_text"].fillna("")


def text_transformer(df: pd.DataFrame, cfg):
    if "feedback_category" in df and df["feedback_category"].notna().sum() >= 10:
        return FeedbackCategoryFeatures(cfg.nlp, FEEDBACK_CATEGORIES), [
            "feedback_text",
            "feedback_category",
        ]
    from sklearn.preprocessing import FunctionTransformer

    svd = TruncatedSVD(n_components=cfg.nlp.svd_components, random_state=cfg.seed)
    return make_pipeline(FunctionTransformer(_TextColumn()), make_tfidf(cfg.nlp), svd), [
        "feedback_text"
    ]


def make_preprocessor(
    df: pd.DataFrame, feature_set: str | list[str], cfg
) -> tuple[ColumnTransformer, list[str]]:
    """ColumnTransformer for a named feature set (or explicit list of modalities); returns it
    and the raw input columns used."""
    modalities = FEATURE_SETS[feature_set] if isinstance(feature_set, str) else list(feature_set)
    num = available(df, numeric_columns(modalities))
    transformers = []
    if num:
        transformers.append(
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)
        )
    used = list(num)
    if "text" in modalities:
        tt, cols = text_transformer(df, cfg)
        transformers.append(("text", tt, cols))
        used += cols
    if not transformers:
        raise ValueError(f"No usable columns for feature set {feature_set!r}")
    return ColumnTransformer(transformers, verbose_feature_names_out=False), used


def make_pipeline_for(
    df: pd.DataFrame, feature_set: str | list[str], estimator, cfg
) -> tuple[Pipeline, list[str]]:
    pre, used = make_preprocessor(df, feature_set, cfg)
    return Pipeline([("pre", pre), ("clf", estimator)]), used
