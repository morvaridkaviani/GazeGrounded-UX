"""Task-agnostic baseline attention maps, to test whether the VLM adds information.

* Baseline A - **centre bias**: an anisotropic Gaussian at the screen centre. Observers fixate
  near the centre of a display regardless of content; any model must beat this.
* Baseline B - **spectral residual saliency** (Hou & Zhang, CVPR 2007): a classic bottom-up
  image saliency model using only image statistics (no learning, no task).
* Baseline C - the trained deep saliency model (:mod:`gazegrounded.vision.saliency_model`).
* Baseline D - the task-conditioned VLM Task-Relevance Map (:mod:`gazegrounded.vlm.attention_map`).

All maps are returned on the analysis grid and normalised to [0, 1].
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from gazegrounded.gaze.heatmap import normalize_map


def center_bias_map(
    shape: tuple[int, int],
    sigma_frac: tuple[float, float] = (0.25, 0.25),
    center: tuple[float, float] = (0.5, 0.45),
) -> np.ndarray:
    """Gaussian centred at ``center`` (x, y fractions); sigma as fractions of width/height."""
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols]
    gx = ((x + 0.5) / cols - center[0]) / sigma_frac[0]
    gy = ((y + 0.5) / rows - center[1]) / sigma_frac[1]
    return normalize_map(np.exp(-0.5 * (gx**2 + gy**2)), "max")


def spectral_residual_saliency(
    image: Image.Image | np.ndarray,
    shape: tuple[int, int],
    work_width: int = 128,
    blur_sigma: float = 2.5,
) -> np.ndarray:
    """Spectral residual saliency (Hou & Zhang 2007) resized to ``shape`` (rows, cols)."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    gray = img.convert("L")
    w, h = gray.size
    work = gray.resize((work_width, max(1, round(h * work_width / w))), Image.Resampling.BILINEAR)
    a = np.asarray(work, dtype=float) / 255.0
    spectrum = np.fft.fft2(a)
    log_amp = np.log(np.abs(spectrum) + 1e-9)
    phase = np.angle(spectrum)
    residual = log_amp - uniform_filter(log_amp, size=3, mode="nearest")
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = gaussian_filter(sal, sigma=blur_sigma)
    out = Image.fromarray(normalize_map(sal, "max").astype(np.float32))
    out = out.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return normalize_map(np.asarray(out, dtype=float), "max")
