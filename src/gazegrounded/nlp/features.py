"""Text representations: TF-IDF, and leakage-safe text features for the friction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from gazegrounded.nlp.preprocessing import TOKEN_PATTERN, clean_text


def make_tfidf(nlp_cfg) -> TfidfVectorizer:
    """Word uni/bi-gram TF-IDF with sublinear term frequency (good for short texts)."""
    return TfidfVectorizer(
        ngram_range=(nlp_cfg.ngram_min, nlp_cfg.ngram_max),
        min_df=nlp_cfg.min_df,
        max_features=nlp_cfg.max_features,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
        preprocessor=clean_text,
    )


class FeedbackCategoryFeatures(BaseEstimator, TransformerMixin):
    """Predicted feedback-category probabilities as features for the friction model.

    Input: a DataFrame with ``feedback_text`` and ``feedback_category`` columns. During ``fit``
    a TF-IDF + logistic-regression classifier learns text -> category **on the training fold
    only**; ``transform`` uses text alone and outputs one probability column per category plus
    a ``has_feedback`` flag. Because it is fitted inside the friction model's pipeline, no
    category label from a test fold is ever seen (no leakage).
    """

    def __init__(self, nlp_cfg=None, categories: tuple[str, ...] = ()):
        self.nlp_cfg = nlp_cfg
        self.categories = categories

    def fit(self, X: pd.DataFrame, y=None):
        texts = X["feedback_text"].map(clean_text)
        cats = X["feedback_category"]
        mask = (texts.str.len() > 0) & cats.notna()
        self.classes_ = sorted(cats[mask].unique())
        self.clf_ = None
        if len(self.classes_) >= 2:
            self.clf_ = make_pipeline(
                make_tfidf(self.nlp_cfg),
                LogisticRegression(C=self.nlp_cfg.C, max_iter=2000, class_weight="balanced"),
            ).fit(texts[mask], cats[mask])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        texts = X["feedback_text"].map(clean_text)
        has = (texts.str.len() > 0).to_numpy()
        cats = list(self.categories) or self.classes_
        out = np.zeros((len(X), len(cats) + 1))
        if self.clf_ is not None and has.any():
            proba = self.clf_.predict_proba(texts[has])
            col = {c: i for i, c in enumerate(self.clf_.classes_)}
            for j, c in enumerate(cats):
                if c in col:
                    out[has, j] = proba[:, col[c]]
        out[:, -1] = has
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cats = list(self.categories) or self.classes_
        return np.array([f"text_p_{c}" for c in cats] + ["has_feedback"])
