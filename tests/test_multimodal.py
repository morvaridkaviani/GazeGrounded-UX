import numpy as np
import pytest

from gazegrounded.analysis.model_evaluation import grouped_cv_splits
from gazegrounded.multimodal.evaluation import evaluate_feature_set
from gazegrounded.multimodal.explainability import linear_contributions
from gazegrounded.multimodal.features import MODALITY_FEATURES
from gazegrounded.multimodal.fusion import make_pipeline_for
from gazegrounded.multimodal.models import make_estimator


def test_feature_table_one_row_per_trial_with_all_modalities(pipeline_features):
    df = pipeline_features
    assert df["trial_id"].is_unique
    assert len(df) == 6 * 10  # fast_dev: 6 participants x 10 tasks
    for modality, cols in MODALITY_FEATURES.items():
        for c in cols:
            assert c in df.columns, f"{modality}: {c}"
    assert df["attention_divergence"].between(0, 1).all()
    assert df["friction_label"].isin(["low", "medium", "high"]).all()


@pytest.mark.parametrize("group_by", ["participant_id", "ui_id"])
def test_cv_folds_have_zero_group_overlap(pipeline_features, pipeline_cfg, group_by):
    res = evaluate_feature_set(
        pipeline_features, "gaze", "logistic_regression", pipeline_cfg, group_by=group_by
    )
    p = res["predictions"].merge(pipeline_features[["trial_id", group_by]], on="trial_id")
    # every group is tested in exactly one fold -> never in train and test of the same fold
    assert (p.groupby(group_by)["fold"].nunique() == 1).all()
    assert (p["fold"] >= 0).all()


def test_text_features_do_not_read_test_labels(pipeline_features, pipeline_cfg):
    df = pipeline_features
    tr, te = next(grouped_cv_splits(df["participant_id"].to_numpy(), 3, seed=0))
    pipe, cols = make_pipeline_for(
        df, "multimodal", make_estimator("logistic_regression", pipeline_cfg), pipeline_cfg
    )
    pipe.fit(df.iloc[tr][cols], df.iloc[tr]["friction_label"])
    X_te = df.iloc[te][cols].copy()
    before = pipe.predict_proba(X_te)
    X_te["feedback_category"] = "garbage"  # labels of the test fold must be irrelevant
    np.testing.assert_allclose(before, pipe.predict_proba(X_te))


def test_linear_contributions_sum_to_decision_score(pipeline_features, pipeline_cfg):
    df = pipeline_features
    pipe, cols = make_pipeline_for(
        df, "multimodal", make_estimator("logistic_regression", pipeline_cfg), pipeline_cfg
    )
    pipe.fit(df[cols], df["friction_label"])
    X = df[cols].head(4)
    pred = pipe.predict(X)
    contrib = linear_contributions(pipe, X, pred)
    clf = pipe.named_steps["clf"]
    scores = pipe.decision_function(X)
    for r in range(4):
        k = list(clf.classes_).index(pred[r])
        total = contrib[contrib["row"] == r]["contribution"].sum() + clf.intercept_[k]
        assert total == pytest.approx(scores[r, k])


def test_majority_baseline_and_metrics(pipeline_features, pipeline_cfg):
    res = evaluate_feature_set(pipeline_features, "interaction", "majority", pipeline_cfg)
    o = res["overall"]
    assert {
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "quadratic_weighted_kappa",
        "mae_ordinal",
        "confusion_matrix",
    } <= set(o)
    assert res["folds"]["n_folds"] == 3
