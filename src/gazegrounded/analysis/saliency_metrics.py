"""Map-comparison metrics used for human-vs-VLM alignment and saliency-model evaluation.

Following the saliency benchmark literature (Bylinskii et al., 2019, "What do different
evaluation metrics tell us about saliency models?"), the *reference* is always the human map
and the *prediction* is the model / VLM / baseline map.

* ``cc``        Pearson correlation of the two maps (pixel-wise). [-1, 1], higher = better.
* ``spearman``  rank correlation (robust to monotone rescaling). [-1, 1].
* ``cosine``    cosine similarity of the flattened maps. [0, 1] for non-negative maps.
* ``sim``       histogram intersection of the sum-normalised maps, sum(min(P, Q)). [0, 1].
* ``kl``        KL(P_human || Q_pred) with an epsilon floor. [0, inf), lower = better.
                Penalises predictions that put ~zero mass where humans looked.
* ``jsd``       Jensen-Shannon divergence, base 2. Symmetric, bounded in [0, 1].
* ``nss``       mean of the *standardised* prediction at human fixation locations.
                0 = chance, higher = better. Needs discrete fixation points.

``attention_divergence`` (the project's divergence measure) is defined as ``jsd`` between the
sum-normalised human gaze heatmap and the sum-normalised VLM Task-Relevance Map on the same
grid. It is 0 when the distributions are identical and 1 when they do not overlap at all.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

EPS = 1e-12


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def to_distribution(m: np.ndarray, eps: float = EPS) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype=float), 0, None)
    total = m.sum()
    return m / total if total > eps else np.full_like(m, 1.0 / m.size)


def cc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _flat(a), _flat(b)
    if x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _flat(a), _flat(b)
    if x.std() < EPS or y.std() < EPS:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _flat(a), _flat(b)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > EPS else float("nan")


def sim(human: np.ndarray, pred: np.ndarray) -> float:
    return float(np.minimum(to_distribution(human), to_distribution(pred)).sum())


def kl(human: np.ndarray, pred: np.ndarray, eps: float = EPS) -> float:
    p, q = to_distribution(human), to_distribution(pred)
    return float(np.sum(p * np.log(eps + p / (q + eps))))


def jsd(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    p, q = to_distribution(a), to_distribution(b)
    m = 0.5 * (p + q)

    def _kl2(x, y):
        mask = x > eps
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return float(np.clip(0.5 * _kl2(p, m) + 0.5 * _kl2(q, m), 0.0, 1.0))


def nss(pred: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Normalised Scanpath Saliency at integer grid locations ``(rows, cols)``."""
    p = np.asarray(pred, dtype=float)
    if p.std() < EPS or len(rows) == 0:
        return float("nan")
    z = (p - p.mean()) / p.std()
    r = np.clip(np.asarray(rows, dtype=int), 0, p.shape[0] - 1)
    c = np.clip(np.asarray(cols, dtype=int), 0, p.shape[1] - 1)
    return float(z[r, c].mean())


def fixations_to_grid(
    x_px: np.ndarray, y_px: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates -> integer (row, col) on a map computed at ``scale``."""
    return (
        np.floor(np.asarray(y_px) * scale).astype(int),
        np.floor(np.asarray(x_px) * scale).astype(int),
    )


def compare_maps(
    human: np.ndarray,
    pred: np.ndarray,
    fixation_rows: np.ndarray | None = None,
    fixation_cols: np.ndarray | None = None,
) -> dict[str, float]:
    """All map metrics at once (``nss`` only if fixation locations are given)."""
    out = {
        "cc": cc(human, pred),
        "spearman": spearman(human, pred),
        "cosine": cosine(human, pred),
        "sim": sim(human, pred),
        "kl": kl(human, pred),
        "jsd": jsd(human, pred),
    }
    if fixation_rows is not None and fixation_cols is not None:
        out["nss"] = nss(pred, fixation_rows, fixation_cols)
    return out


def distribution_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """JSD (base 2) between two discrete distributions given as ``{category: mass}``."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    return jsd(a, b)
