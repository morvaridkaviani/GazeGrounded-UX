"""Paired comparisons, effect sizes, multiple-comparison correction, cluster bootstrap.

Principles
----------
* The test is **pre-specified** (``stats.paired_test`` in the config), not chosen after
  looking at the data; the Shapiro-Wilk p-value of the paired differences is reported as a
  diagnostic only.
* Repeated trials are first averaged within participant, so each participant contributes one
  paired observation (the unit of analysis is the participant, not the trial).
* Every result reports N, the statistic, p, and an effect size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def participant_condition_means(
    df: pd.DataFrame, subject: str, condition: str, value: str, cond_a: str, cond_b: str
) -> pd.DataFrame:
    """Wide table ``subject x {cond_a, cond_b}`` of within-participant means (complete pairs)."""
    d = df[df[condition].isin([cond_a, cond_b])]
    wide = d.groupby([subject, condition])[value].mean().unstack(condition)
    return wide.dropna(subset=[cond_a, cond_b])


def rank_biserial_paired(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size for the Wilcoxon signed-rank)."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def paired_comparison(
    df: pd.DataFrame,
    value: str,
    subject: str = "participant_id",
    condition: str = "condition",
    cond_a: str = "clean",
    cond_b: str = "friction",
    test: str = "wilcoxon",
) -> dict:
    """Compare ``cond_b`` vs ``cond_a`` within participants. Differences are ``b - a``."""
    wide = participant_condition_means(df, subject, condition, value, cond_a, cond_b)
    a, b = wide[cond_a].to_numpy(float), wide[cond_b].to_numpy(float)
    diff = b - a
    n = len(diff)
    out = {
        "variable": value,
        "comparison": f"{cond_b} - {cond_a}",
        "unit": subject,
        "n": n,
        "test": test,
        "mean_a": float(a.mean()) if n else np.nan,
        "mean_b": float(b.mean()) if n else np.nan,
        "mean_diff": float(diff.mean()) if n else np.nan,
        "median_diff": float(np.median(diff)) if n else np.nan,
    }
    out["shapiro_p_diff"] = (
        float(stats.shapiro(diff).pvalue) if n >= 3 and np.ptp(diff) > 0 else np.nan
    )
    if n < 3:
        return out | {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": None,
            "note": "fewer than 3 complete pairs",
        }
    if test == "ttest":
        res = stats.ttest_rel(b, a)
        sd = diff.std(ddof=1)
        dz = float(diff.mean() / sd) if sd > 0 else np.nan
        se = sd / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        return out | {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "effect_size": dz,
            "effect_size_name": "cohens_dz",
            "ci95_mean_diff": [float(diff.mean() - h), float(diff.mean() + h)],
        }
    res = stats.wilcoxon(b, a, zero_method="wilcox")
    return out | {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "effect_size": rank_biserial_paired(diff),
        "effect_size_name": "rank_biserial",
    }


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN and not counted)."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    adjusted = np.full_like(p, np.nan)
    order = np.argsort(np.where(valid, p, np.inf))[:m]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    cluster: str = "participant_id",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI resampling whole clusters (participants) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    values = []
    for _ in range(n_boot):
        sample = pd.concat([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        v = statistic(sample)
        if not np.isnan(v):
            values.append(v)
    if not values:
        return (np.nan, np.nan)
    return (float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2)))
