import numpy as np
import pandas as pd
import pytest

from gazegrounded.gaze.fixation import detect_fixations, detect_fixations_idt
from gazegrounded.gaze.heatmap import generate_heatmap, grid_shape, normalize_map
from gazegrounded.gaze.metrics import aoi_level_metrics, trial_gaze_metrics
from gazegrounded.gaze.scanpath import (
    collapse_visits,
    format_scanpath,
    levenshtein,
    normalized_scanpath_similarity,
    scanpath_length_px,
    transition_matrix,
)

DT = 1000 / 60


def _cluster(x, y, n, rng, t0, jitter=3.0):
    t = t0 + np.arange(n) * DT
    return t, x + rng.normal(0, jitter, n), y + rng.normal(0, jitter, n)


def test_idt_finds_two_fixations_with_correct_centroids():
    rng = np.random.default_rng(0)
    t1, x1, y1 = _cluster(200, 300, 18, rng, 0)  # 300 ms
    t2, x2, y2 = _cluster(900, 500, 24, rng, t1[-1] + 3 * DT)  # 400 ms after a saccade
    sx = np.linspace(200, 900, 4)[1:-1]
    sy = np.linspace(300, 500, 4)[1:-1]
    st = t1[-1] + DT * np.arange(1, 3)
    t = np.concatenate([t1, st, t2])
    x = np.concatenate([x1, sx, x2])
    y = np.concatenate([y1, sy, y2])
    fx = detect_fixations_idt(t, x, y, dispersion_threshold_px=40, minimum_duration_ms=100)
    assert len(fx) == 2
    assert fx.loc[0, "centroid_x"] == pytest.approx(200, abs=3)
    assert fx.loc[1, "centroid_y"] == pytest.approx(500, abs=3)
    assert fx.loc[0, "duration_ms"] == pytest.approx(300, abs=DT + 1e-6)
    assert fx.loc[1, "duration_ms"] == pytest.approx(400, abs=DT + 1e-6)


def test_idt_ignores_too_short_fixations():
    rng = np.random.default_rng(1)
    t, x, y = _cluster(100, 100, 4, rng, 0)  # ~67 ms
    fx = detect_fixations_idt(t, x, y, dispersion_threshold_px=40, minimum_duration_ms=100)
    assert fx.empty


def test_idt_splits_on_long_signal_loss_but_bridges_short_gaps():
    rng = np.random.default_rng(2)
    t, x, y = _cluster(400, 400, 30, rng, 0)  # 500 ms at one location
    valid = np.ones(30, dtype=bool)
    valid[14] = False  # single dropped sample -> bridged
    assert len(detect_fixations_idt(t, x, y, valid, 40, 100, max_gap_ms=75)) == 1
    valid[10:20] = False  # ~167 ms of signal loss -> split
    assert len(detect_fixations_idt(t, x, y, valid, 40, 100, max_gap_ms=75)) == 2


def test_detect_fixations_on_dataframe(synthetic_ds):
    trial = synthetic_ds.gaze_samples["trial_id"].iloc[0]
    g = synthetic_ds.gaze_samples[synthetic_ds.gaze_samples["trial_id"] == trial]
    fx = detect_fixations(g)
    assert {"participant_id", "trial_id", "fixation_id", "duration_ms", "aoi_id"} <= set(fx.columns)
    assert (fx["duration_ms"] >= 100).all()
    assert (np.diff(fx["start_ms"]) > 0).all()


def _fix(rows):
    """rows: (start_ms, duration_ms, x, y, aoi)"""
    return pd.DataFrame(
        [
            {
                "fixation_id": i,
                "start_ms": s,
                "end_ms": s + d,
                "duration_ms": d,
                "centroid_x": x,
                "centroid_y": y,
                "aoi_id": a,
            }
            for i, (s, d, x, y, a) in enumerate(rows)
        ]
    )


AOIS = pd.DataFrame(
    {
        "aoi_id": ["A", "B", "T"],
        "x_min": [0.0, 0.5, 0.0],
        "y_min": [0.0, 0.0, 0.5],
        "x_max": [0.5, 1.0, 0.5],
        "y_max": [0.5, 0.5, 1.0],
    }
)


def test_ttff_revisits_transitions_and_attention():
    f = _fix(
        [
            (0, 200, 100, 100, "A"),
            (250, 200, 110, 110, "A"),
            (500, 300, 700, 100, "B"),
            (850, 250, 100, 700, "T"),  # first target fixation at 850 ms
            (1150, 200, 120, 120, "A"),  # revisit A
            (1400, 300, 100, 700, "T"),  # revisit T
            (1750, 100, 999, 999, None),  # outside all AOIs
        ]
    )
    m = trial_gaze_metrics(f, AOIS, ["T"], 1000, 1000, trial_duration_ms=2000)
    assert m["ttff_ms"] == 850
    assert m["target_fixated"] is True
    assert m["fixation_count"] == 7
    assert m["total_fixation_duration_ms"] == 1550
    assert m["revisit_count"] == 2  # A visited twice, T visited twice
    assert m["target_revisit_count"] == 1
    assert m["aoi_transition_count"] == 4  # A -> B -> T -> A -> T
    assert m["task_relevant_attention"] == pytest.approx(550 / 1550)
    assert m["task_relevant_aoi_coverage"] == 1.0
    assert m["ttff_norm"] == pytest.approx(850 / 2000)


def test_ttff_is_nan_and_censored_when_target_never_fixated():
    f = _fix([(0, 200, 100, 100, "A"), (300, 200, 700, 100, "B")])
    m = trial_gaze_metrics(f, AOIS, ["T"], 1000, 1000, trial_duration_ms=3000)
    assert np.isnan(m["ttff_ms"])
    assert m["target_fixated"] is False
    assert m["ttff_censored_ms"] == 3000
    assert m["task_relevant_aoi_coverage"] == 0.0


def test_aoi_level_metrics():
    f = _fix([(0, 200, 0, 0, "A"), (300, 100, 0, 0, "B"), (500, 300, 0, 0, "A")])
    am = aoi_level_metrics(f, AOIS, ["T"]).set_index("aoi_id")
    assert am.loc["A", "dwell_ms"] == 500
    assert am.loc["A", "revisit_count"] == 1
    assert am.loc["A", "time_to_first_fixation_ms"] == 0
    assert am.loc["T", "fixation_count"] == 0
    assert am["dwell_share"].sum() == pytest.approx(1.0)


def test_scanpath_helpers():
    seq = ["A", "A", "B", "A", "C"]
    assert collapse_visits(seq) == [("A", 2), ("B", 1), ("A", 1), ("C", 1)]
    mat = transition_matrix(seq)
    assert mat.loc["A", "B"] == 1 and mat.loc["B", "A"] == 1 and mat.loc["A", "C"] == 1
    assert mat.to_numpy().sum() == 3
    assert format_scanpath(seq, {"A": "Header"}) == "Header → B → Header → C"
    assert levenshtein(list("ABC"), list("ABD")) == 1
    assert normalized_scanpath_similarity(seq, seq) == 1.0
    f = _fix([(0, 100, 0, 0, "A"), (200, 100, 3, 4, "B"), (400, 100, 3, 0, "C")])
    assert scanpath_length_px(f) == pytest.approx(9.0)


def test_heatmap_dimensions_normalisation_and_peak():
    pts = np.array([[300.0, 200.0]] * 5 + [[1000.0, 700.0]])
    h = generate_heatmap(pts, 1440, 900, sigma=40, scale=0.125, normalize="max")
    assert h.shape == grid_shape(1440, 900, 0.125) == (112, 180)
    assert h.max() == pytest.approx(1.0)
    r, c = np.unravel_index(h.argmax(), h.shape)
    assert abs(c - 300 * 0.125) <= 1 and abs(r - 200 * 0.125) <= 1
    p = generate_heatmap(pts, 1440, 900, sigma=40, scale=0.125, normalize="sum")
    assert p.sum() == pytest.approx(1.0)


def test_heatmap_ignores_nan_and_offscreen_points():
    pts = np.array([[np.nan, 10.0], [-50.0, 10.0], [5000.0, 10.0]])
    h = generate_heatmap(pts, 100, 100, sigma=5, normalize="max")
    assert h.sum() == 0
    assert normalize_map(h, "sum").sum() == pytest.approx(1.0)  # uniform fallback
