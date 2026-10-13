import numpy as np
import pandas as pd
import pytest

from gazegrounded.config import load_config
from gazegrounded.data.loader import StandardizedDataset
from gazegrounded.data.preprocessing import clean_gaze_samples, trial_quality
from gazegrounded.data.schemas import FEEDBACK_CATEGORIES, FRICTION_LABELS
from gazegrounded.data.synthetic import generate_synthetic_dataset
from gazegrounded.data.validation import validate_table


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/fast_dev.yaml", overrides=["seed=7", "gaze.heatmap_sigma_px=33"])
    assert cfg.seed == 7
    assert cfg.gaze.heatmap_sigma_px == 33
    assert cfg.saliency.fast_dev_run is True  # from fast_dev.yaml
    assert cfg.data.data_kind == "synthetic_demo"  # inherited from synthetic.yaml
    assert cfg.gaze.minimum_duration_ms == 100  # inherited from base.yaml


def test_synthetic_dataset_follows_schema(synthetic_ds):
    report = synthetic_ds.validate()
    assert report.ok, report.errors
    assert synthetic_ds.is_synthetic
    assert len(synthetic_ds.uis) == 5
    assert set(synthetic_ds.trials["friction_label"]) <= set(FRICTION_LABELS)
    cats = synthetic_ds.trials["feedback_category"].dropna()
    assert set(cats) <= set(FEEDBACK_CATEGORIES)


def test_synthetic_generation_is_deterministic(tmp_path, fast_cfg):
    a = generate_synthetic_dataset(fast_cfg, out_dir=tmp_path / "a")
    b = generate_synthetic_dataset(fast_cfg, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a.trials, b.trials)
    pd.testing.assert_frame_equal(a.gaze_samples, b.gaze_samples)


def test_synthetic_patterns_are_interpretable(synthetic_ds):
    """Friction variants were generated to be slower; check the generator encodes that."""
    t = synthetic_ds.trials
    means = t.groupby("condition")["completion_time_ms"].mean()
    assert means["friction"] > means["clean"]


def test_save_and_load_roundtrip(tmp_path, synthetic_ds):
    synthetic_ds.save(tmp_path / "processed")
    loaded = StandardizedDataset.load(tmp_path / "processed")
    assert loaded.data_kind == "synthetic_demo"
    assert loaded.validate().ok
    assert len(loaded.gaze_samples) == len(synthetic_ds.gaze_samples)
    assert loaded.trials["participant_id"].iloc[0].startswith("P")


def test_validation_catches_bad_rows(synthetic_ds):
    aois = synthetic_ds.aois.copy()
    aois.loc[0, "x_max"] = 1.7  # outside [0, 1]
    assert not validate_table(aois, "aois").ok

    trials = synthetic_ds.trials.copy()
    trials["email"] = "someone@example.com"  # PII column must be rejected
    report = validate_table(trials, "trials")
    assert any("PII" in e for e in report.errors)


def test_validation_catches_broken_references(synthetic_ds):
    import copy

    ds = copy.copy(synthetic_ds)
    ds.tasks = synthetic_ds.tasks.copy()
    ds.tasks.loc[0, "target_aoi_ids"] = "does_not_exist"
    assert not ds.validate().ok


def test_clean_gaze_invalidates_offscreen(synthetic_ds):
    g = synthetic_ds.gaze_samples.head(50).copy()
    g.loc[g.index[0], ["gaze_x", "validity"]] = [99999.0, 1]
    cleaned = clean_gaze_samples(g, synthetic_ds.uis)
    assert cleaned.loc[0, "validity"] == 0
    assert np.isnan(cleaned.loc[0, "gaze_x"])


def test_trial_quality_flags_low_validity(synthetic_ds):
    q = trial_quality(synthetic_ds.gaze_samples, min_valid_ratio=0.6)
    assert {"valid_ratio", "sampling_rate_hz", "low_quality"} <= set(q.columns)
    assert q["sampling_rate_hz"].median() == pytest.approx(60, rel=0.05)
