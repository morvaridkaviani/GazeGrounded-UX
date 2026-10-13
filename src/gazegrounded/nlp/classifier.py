"""Feedback-category classifier (TF-IDF + logistic regression) and exploratory clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from gazegrounded.analysis.model_evaluation import align_proba, grouped_cv_splits
from gazegrounded.nlp.features import make_tfidf

CLUSTER_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - {
    "no",
    "not",
    "nor",
    "never",
    "nothing",
    "where",
    "too",
    "much",
    "didn't",
    "couldn't",
    "wasn't",
}


def make_feedback_classifier(nlp_cfg) -> Pipeline:
    """TF-IDF -> multinomial logistic regression with balanced class weights.

    Interpretable by design: each coefficient says how much a word/bigram pushes a comment
    towards a category.
    """
    return Pipeline(
        [
            ("tfidf", make_tfidf(nlp_cfg)),
            ("clf", LogisticRegression(C=nlp_cfg.C, max_iter=2000, class_weight="balanced")),
        ]
    )


def cross_validated_predictions(
    texts: pd.Series, labels: pd.Series, groups: pd.Series, nlp_cfg, seed: int = 42
) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-fold predictions with participant-grouped CV.

    Returns a DataFrame (index aligned with ``texts``) with ``pred``, ``confidence``,
    ``fold`` and one probability column per class, plus per-fold label arrays.
    """
    classes = sorted(labels.unique())
    oof = pd.DataFrame(index=texts.index, columns=["pred", "confidence", "fold"], dtype=object)
    proba_all = np.zeros((len(texts), len(classes)))
    folds = []
    for k, (tr, te) in enumerate(grouped_cv_splits(groups.to_numpy(), nlp_cfg.n_splits, seed)):
        model = make_feedback_classifier(nlp_cfg).fit(texts.iloc[tr], labels.iloc[tr])
        proba = align_proba(model.predict_proba(texts.iloc[te]), model.classes_, classes)
        pred = np.array(classes)[proba.argmax(axis=1)]
        oof.iloc[te, 0] = pred
        oof.iloc[te, 1] = proba.max(axis=1)
        oof.iloc[te, 2] = k
        proba_all[te] = proba
        folds.append({"y_true": labels.iloc[te].to_numpy(), "y_pred": pred})
    for j, c in enumerate(classes):
        oof[f"p_{c}"] = proba_all[:, j]
    return oof, folds


def top_terms_per_class(model: Pipeline, n: int = 8) -> pd.DataFrame:
    """Highest-coefficient n-grams per class from a fitted classifier."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    rows = []
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if i else -1)
        for j in np.argsort(coefs)[::-1][:n]:
            rows.append({"class": cls, "term": vocab[j], "coefficient": float(coefs[j])})
    return pd.DataFrame(rows)


def cluster_feedback(
    texts: pd.Series, nlp_cfg, seed: int = 42, n_terms: int = 6
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exploratory KMeans on TF-IDF vectors; returns labels and top terms per cluster.

    Clusters are a lens for reading comments, not categories with ground truth. Unlike the
    classifier, clustering drops stop-words (keeping negations and "where"), otherwise
    clusters form around function words.
    """
    vec = make_tfidf(nlp_cfg)
    vec.set_params(stop_words=sorted(CLUSTER_STOP_WORDS))
    X = vec.fit_transform(texts)
    k = min(nlp_cfg.n_clusters, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    vocab = vec.get_feature_names_out()
    rows = []
    for c in range(k):
        top = np.argsort(km.cluster_centers_[c])[::-1][:n_terms]
        rows.append(
            {"cluster": c, "size": int((km.labels_ == c).sum()), "top_terms": ", ".join(vocab[top])}
        )
    return km.labels_, pd.DataFrame(rows)
