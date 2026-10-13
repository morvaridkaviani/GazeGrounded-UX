"""VLM Task-Relevance Map: turn grounded elements into a continuous spatial map.

This map is the VLM's *estimate* of where task-relevant information is. It is NOT a measured
or predicted gaze map and must not be labelled as one.

Construction
------------
For each element ``k`` with box ``B_k`` and weight ``w_k = task_relevance x
expected_attention_priority``:

1. ``U_k`` = indicator of ``B_k`` on the analysis grid, softened at the edges by a Gaussian
   with sigma = ``softness x min(box width, box height)``.
2. ``mode="mass"`` (default): ``U_k`` is normalised to unit sum, so every element contributes
   attention *mass* ``w_k`` regardless of its size (a small button and a large card with equal
   weight receive equal mass). ``mode="density"``: ``U_k`` keeps unit height, so larger boxes
   contribute more mass.
3. ``M = sum_k w_k U_k``, then optionally blurred with the same sigma as human heatmaps so both
   maps share the same spatial resolution, and finally normalised to [0, 1] by its maximum.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

from gazegrounded.gaze.heatmap import grid_shape, normalize_map
from gazegrounded.vlm.schemas import VLMAnalysis


def element_distribution(
    box: tuple[float, float, float, float],
    shape: tuple[int, int],
    softness: float = 0.25,
    mode: Literal["mass", "density"] = "mass",
) -> np.ndarray:
    rows, cols = shape
    x0, y0, x1, y1 = box
    c0, c1 = int(np.floor(x0 * cols)), max(int(np.ceil(x1 * cols)), int(np.floor(x0 * cols)) + 1)
    r0, r1 = int(np.floor(y0 * rows)), max(int(np.ceil(y1 * rows)), int(np.floor(y0 * rows)) + 1)
    u = np.zeros(shape, dtype=float)
    u[max(r0, 0) : min(r1, rows), max(c0, 0) : min(c1, cols)] = 1.0
    sigma = softness * min(c1 - c0, r1 - r0)
    if sigma > 0:
        u = gaussian_filter(u, sigma=sigma, mode="constant")
    if mode == "mass":
        total = u.sum()
        return u / total if total > 0 else u
    peak = u.max()
    return u / peak if peak > 0 else u


def vlm_relevance_map(
    analysis: VLMAnalysis,
    screen_width: int,
    screen_height: int,
    scale: float = 0.125,
    softness: float = 0.25,
    blur_sigma_px: float | None = 40.0,
    mode: Literal["mass", "density"] = "mass",
    normalize: Literal["max", "sum"] | None = "max",
) -> np.ndarray:
    """VLM Task-Relevance Map on the same grid as :func:`gazegrounded.gaze.heatmap.generate_heatmap`."""
    shape = grid_shape(screen_width, screen_height, scale)
    m = np.zeros(shape, dtype=float)
    for el in analysis.elements:
        m += el.weight * element_distribution(el.bbox.as_tuple(), shape, softness, mode)
    if blur_sigma_px:
        m = gaussian_filter(m, sigma=blur_sigma_px * scale, mode="constant")
    return normalize_map(m, normalize)
