import numpy as np
import pandas as pd
import pytest

from gazegrounded.analysis.model_evaluation import (
    assert_no_group_overlap,
    classification_metrics,
    grouped_cv_splits,
)
from gazegrounded.nlp.classifier import (
    cluster_feedback,
    cross_validated_predictions,
    make_feedback_classifier,
    top_terms_per_class,
)
from gazegrounded.nlp.features import FeedbackCategoryFeatures
from gazegrounded.nlp.preprocessing import clean_text, prepare_feedback


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  I DIDN’T find it!!  ", "i didn't find it"),
        ("Where's the 'Edit' button?", "where's the edit button"),
        ("too    much\ntext", "too much text"),
        (None, ""),
        (float("nan"), ""),
        ("", ""),
    ],
)
def test_clean_text_keeps_negations_and_handles_missing(raw, expected):
    assert clean_text(raw) == expected


def test_prepare_feedback_flags_missing():
    df = pd.DataFrame({"feedback_text": ["Easy.", None, "  "]})
    out = prepare_feedback(df)
    assert out["has_feedback"].tolist() == [True, False, False]


def test_grouped_cv_has_no_participant_overlap():
    groups = np.repeat([f"P{i}" for i in range(7)], 10)
    seen_test = []
    for tr, te in grouped_cv_splits(groups, n_splits=5, seed=0):
        assert set(groups[tr]).isdisjoint(groups[te])
        seen_test.extend(groups[te])
    assert sorted(seen_test) == sorted(groups)  # every row tested exactly once


def test_leakage_assertion_fires():
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_group_overlap(["P1", "P2"], ["P2", "P3"])


def test_classification_metrics_ordinal():
    labels = ["low", "medium", "high"]
    m = classification_metrics(
        ["low", "high", "medium"], ["low", "medium", "medium"], labels, ordinal=True
    )
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["mae_ordinal"] == pytest.approx(1 / 3)
    assert m["confusion_matrix"][2][1] == 1


def _toy():
    texts = ["couldn't find the button", "hard to find it", "where is the option"] * 4 + [
        "easy and quick",
        "no problems at all",
        "obvious where to click",
    ] * 4
    cats = ["findability"] * 12 + ["no_friction"] * 12
    groups = [f"P{i % 6}" for i in range(24)]
    return pd.Series(texts), pd.Series(cats), pd.Series(groups)


def test_classifier_cross_validation_and_terms(fast_cfg):
    texts, cats, groups = _toy()
    oof, folds = cross_validated_predictions(texts, cats, groups, fast_cfg.nlp, seed=0)
    assert oof["pred"].notna().all()
    assert {"p_findability", "p_no_friction"} <= set(oof.columns)
    assert (oof["pred"] == cats).mean() > 0.8
    terms = top_terms_per_class(make_feedback_classifier(fast_cfg.nlp).fit(texts, cats), n=3)
    assert set(terms["class"]) == {"findability", "no_friction"}


def test_clustering_runs(fast_cfg):
    texts, _, _ = _toy()
    labels, clusters = cluster_feedback(texts, fast_cfg.nlp, seed=0)
    assert len(labels) == len(texts)
    assert clusters["size"].sum() == len(texts)


def test_feedback_category_features_use_text_only_at_transform(fast_cfg):
    texts, cats, _ = _toy()
    X = pd.DataFrame({"feedback_text": texts, "feedback_category": cats})
    tf = FeedbackCategoryFeatures(fast_cfg.nlp, ("findability", "no_friction")).fit(X)
    unseen = pd.DataFrame(
        {"feedback_text": ["could not find it", None], "feedback_category": [None, None]}
    )
    out = tf.transform(unseen)
    assert out.shape == (2, 3)
    assert out[0, 0] > out[0, 1]  # leans to findability
    assert out[1].tolist() == [0.0, 0.0, 0.0]  # missing feedback -> zeros, has_feedback=0
