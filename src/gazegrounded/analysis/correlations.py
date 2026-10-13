"""Correlation analyses for RQ2 (attention divergence vs behavioural friction indicators).

* :func:`repeated_measures_correlation` (primary): the common within-participant association,
  removing between-participant differences (Bakdash & Marusich, 2017, *Frontiers in
  Psychology*). Appropriate because each participant contributes many trials.
* :func:`spearman_correlation` on all trials: reported as *exploratory* only, because trials of
  one participant are not independent (p-values are too optimistic).
* :func:`participant_level_correlation`: one point per participant (means), very conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _fisher_ci(r: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    if np.isnan(r) or abs(r) >= 1 or not np.isfinite(se):
        return (np.nan, np.nan)
    z = np.arctanh(r)
    q = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))


def spearman_correlation(x, y) -> dict:
    """Spearman rho with a Fisher-z CI (SE = sqrt(1.06 / (n - 3)), Fieller et al., 1957)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    if n < 4 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return {
            "method": "spearman",
            "n": n,
            "statistic": np.nan,
            "p_value": np.nan,
            "ci95": [np.nan, np.nan],
        }
    res = stats.spearmanr(d["x"], d["y"])
    rho = float(res.statistic)
    return {
        "method": "spearman",
        "n": n,
        "statistic": rho,
        "p_value": float(res.pvalue),
        "ci95": list(_fisher_ci(rho, np.sqrt(1.06 / (n - 3)))),
    }


def repeated_measures_correlation(
    df: pd.DataFrame, x: str, y: str, subject: str = "participant_id"
) -> dict:
    """rmcorr: Pearson r of within-subject-centred x and y, with df = N - k - 1."""
    d = df[[subject, x, y]].dropna()
    k = d[subject].nunique()
    n = len(d)
    dof = n - k - 1
    if k < 2 or dof < 3:
        return {
            "method": "rmcorr",
            "n": n,
            "n_subjects": k,
            "statistic": np.nan,
            "p_value": np.nan,
            "dof": dof,
            "ci95": [np.nan, np.nan],
        }
    xc = d[x] - d.groupby(subject)[x].transform("mean")
    yc = d[y] - d.groupby(subject)[y].transform("mean")
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    r = float((xc * yc).sum() / denom) if denom > 0 else np.nan
    t = r * np.sqrt(dof / (1 - r**2)) if abs(r) < 1 else np.inf
    p = float(2 * stats.t.sf(abs(t), dof))
    return {
        "method": "rmcorr",
        "n": n,
        "n_subjects": k,
        "statistic": r,
        "p_value": p,
        "dof": dof,
        "ci95": list(_fisher_ci(r, 1 / np.sqrt(dof - 1))),
    }


def participant_level_correlation(
    df: pd.DataFrame, x: str, y: str, subject: str = "participant_id"
) -> dict:
    means = df.groupby(subject)[[x, y]].mean()
    out = spearman_correlation(means[x], means[y])
    out["method"] = "spearman_participant_means"
    return out
