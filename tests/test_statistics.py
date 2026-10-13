import numpy as np
import pandas as pd
import pytest

from gazegrounded.analysis.correlations import (
    participant_level_correlation,
    repeated_measures_correlation,
    spearman_correlation,
)
from gazegrounded.analysis.reporting import experiment_block, fmt_p, format_test, markdown_table
from gazegrounded.analysis.statistics import (
    cluster_bootstrap_ci,
    holm_correction,
    paired_comparison,
    rank_biserial_paired,
)


def test_holm_correction_matches_hand_computation():
    adj = holm_correction([0.01, 0.04, 0.03, np.nan])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max -> 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_rank_biserial_extremes():
    assert rank_biserial_paired(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial_paired(np.array([-1.0, -2.0])) == -1.0
    assert rank_biserial_paired(np.array([0.0, 0.0])) == 0.0


def _paired_df(shift, n=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.normal(5, 1)
        for _rep in range(3):  # three trials per participant and condition
            rows.append(
                {"participant_id": f"P{i}", "condition": "clean", "v": base + rng.normal(0, 0.1)}
            )
            rows.append(
                {
                    "participant_id": f"P{i}",
                    "condition": "friction",
                    "v": base + shift + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("test", ["wilcoxon", "ttest"])
def test_paired_comparison_uses_participant_means(test):
    res = paired_comparison(_paired_df(1.0), "v", test=test)
    assert res["n"] == 10  # participants, not 60 trials
    assert res["mean_diff"] == pytest.approx(1.0, abs=0.1)
    assert res["p_value"] < 0.01
    assert res["effect_size"] > 0.9 if test == "wilcoxon" else res["effect_size"] > 3


def test_paired_comparison_reports_too_few_pairs():
    res = paired_comparison(_paired_df(1.0, n=2), "v")
    assert np.isnan(res["p_value"]) and "fewer than 3" in res["note"]


def test_rmcorr_recovers_within_subject_effect_despite_simpsons_paradox():
    """Within each participant y rises with x, but participants with higher x have lower y.
    Pooled correlation is negative; the repeated-measures correlation is correctly positive."""
    rows = []
    for i in range(8):
        x = np.arange(5) + 4 * i
        y = 0.5 * np.arange(5) - 3 * i
        rows += [{"participant_id": f"P{i}", "x": a, "y": b} for a, b in zip(x, y, strict=True)]
    df = pd.DataFrame(rows)
    assert spearman_correlation(df["x"], df["y"])["statistic"] < -0.5
    rm = repeated_measures_correlation(df, "x", "y")
    assert rm["statistic"] == pytest.approx(1.0)
    assert rm["dof"] == len(df) - 8 - 1
    assert participant_level_correlation(df, "x", "y")["n"] == 8


def test_spearman_ci_contains_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    res = spearman_correlation(x, x + rng.normal(size=50))
    assert res["ci95"][0] < res["statistic"] < res["ci95"][1]


def test_cluster_bootstrap_ci_brackets_mean():
    df = _paired_df(0.0, n=12)
    lo, hi = cluster_bootstrap_ci(df, lambda d: d["v"].mean(), n_boot=300, seed=0)
    assert lo < df["v"].mean() < hi


def test_reporting_helpers():
    assert fmt_p(0.0004) == "< .001"
    assert fmt_p(0.0312) == "= .031"
    line = format_test(
        {
            "method": "rmcorr",
            "statistic": 0.41,
            "ci95": [0.2, 0.6],
            "p_value": 0.002,
            "p_holm": float("nan"),
            "n": 120,
        }
    )
    assert line == "rmcorr, r = 0.41 [0.20, 0.60], p = .002, N = 120"
    table = markdown_table(pd.DataFrame({"a": ["x"], "b": [0.123]}))
    assert "| x | 0.12 |" in table
    block = experiment_block("E", {"Hypothesis": "H"}, synthetic=True)
    assert "Not interpretable" in block
    assert "To be written by the researcher" in experiment_block("E", {}, synthetic=False)
