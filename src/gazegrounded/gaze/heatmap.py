"""Gaussian-smoothed gaze heatmaps.

A heatmap is built by accumulating (optionally duration-weighted) gaze points on a grid and
convolving with an isotropic Gaussian. ``sigma`` is specified in *screen pixels* and is scaled
with the grid so maps at different resolutions describe the same spatial spread. A sigma of
~1 degree of visual angle approximates foveal extent plus tracker inaccuracy.

Normalisation
-------------
* ``"max"``  -> values in [0, 1] (for display and for MSE-style comparison)
* ``"sum"``  -> a probability distribution summing to 1 (for KL / similarity metrics)
* ``None``   -> raw smoothed counts
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

Normalization = Literal["max", "sum"] | None


def grid_shape(screen_width: int, screen_height: int, scale: float = 1.0) -> tuple[int, int]:
    """``(rows, cols)`` of a map at ``scale`` x the screen resolution."""
    return max(1, round(screen_height * scale)), max(1, round(screen_width * scale))


def normalize_map(m: np.ndarray, method: Normalization = "max", eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    m = np.clip(m, 0.0, None)
    if method == "max":
        peak = m.max()
        return m / peak if peak > eps else np.zeros_like(m)
    if method == "sum":
        total = m.sum()
        return m / total if total > eps else np.full_like(m, 1.0 / m.size)
    return m


def generate_heatmap(
    gaze_points: np.ndarray,
    screen_width: int,
    screen_height: int,
    sigma: float = 40.0,
    weights: np.ndarray | None = None,
    scale: float = 1.0,
    normalize: Normalization = "max",
) -> np.ndarray:
    """Gaussian-smoothed heatmap of ``gaze_points`` (``(n, 2)`` pixel x, y).

    Points outside the screen and NaNs are ignored. Returns an array of shape
    :func:`grid_shape` ``(screen_width, screen_height, scale)``.
    """
    rows, cols = grid_shape(screen_width, screen_height, scale)
    acc = np.zeros((rows, cols), dtype=float)
    pts = np.asarray(gaze_points, dtype=float).reshape(-1, 2)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float)
    keep = ~np.isnan(pts).any(axis=1)
    pts, w = pts[keep], w[keep]
    c = np.floor(pts[:, 0] * scale).astype(int)
    r = np.floor(pts[:, 1] * scale).astype(int)
    inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
    np.add.at(acc, (r[inside], c[inside]), w[inside])
    smoothed = gaussian_filter(acc, sigma=sigma * scale, mode="constant")
    return normalize_map(smoothed, normalize)


def fixation_heatmap(
    fixations: pd.DataFrame,
    screen_width: int,
    screen_height: int,
    sigma: float = 40.0,
    scale: float = 1.0,
    duration_weighted: bool = True,
    normalize: Normalization = "max",
) -> np.ndarray:
    """Heatmap from fixation centroids, weighted by fixation duration by default."""
    pts = fixations[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    weights = fixations["duration_ms"].to_numpy(dtype=float) if duration_weighted else None
    return generate_heatmap(pts, screen_width, screen_height, sigma, weights, scale, normalize)


def sample_heatmap(
    samples: pd.DataFrame,
    screen_width: int,
    screen_height: int,
    sigma: float = 40.0,
    scale: float = 1.0,
    normalize: Normalization = "max",
) -> np.ndarray:
    """Heatmap from raw valid gaze samples (each sample = equal time weight)."""
    valid = samples[samples["validity"] == 1]
    pts = valid[["gaze_x", "gaze_y"]].to_numpy(dtype=float)
    return generate_heatmap(pts, screen_width, screen_height, sigma, None, scale, normalize)
