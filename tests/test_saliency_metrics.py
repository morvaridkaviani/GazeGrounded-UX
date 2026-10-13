import numpy as np
import pytest

from gazegrounded.analysis.saliency_metrics import cc, compare_maps, jsd, kl, nss, sim
from gazegrounded.vision.baselines import center_bias_map, spectral_residual_saliency


def _blob(shape, r, c, s=3.0):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((y - r) ** 2 + (x - c) ** 2) / (2 * s**2))


def test_identical_maps_are_perfectly_aligned():
    a = _blob((40, 60), 10, 20)
    m = compare_maps(a, a)
    assert m["cc"] == pytest.approx(1.0)
    assert m["sim"] == pytest.approx(1.0)
    assert m["kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["jsd"] == pytest.approx(0.0, abs=1e-6)


def test_disjoint_maps_have_maximal_divergence():
    a, b = _blob((40, 60), 5, 5, 1), _blob((40, 60), 35, 55, 1)
    assert jsd(a, b) == pytest.approx(1.0, abs=1e-3)
    assert sim(a, b) == pytest.approx(0.0, abs=1e-3)
    assert kl(a, b) > 5
    assert cc(a, b) < 0.05


def test_scale_invariance_of_distribution_metrics():
    a, b = _blob((40, 60), 10, 20), _blob((40, 60), 12, 25)
    assert sim(a, b) == pytest.approx(sim(a * 7, b * 0.1))
    assert jsd(a, b) == pytest.approx(jsd(a * 7, b))


def test_nss_positive_at_predicted_peak_and_nan_for_flat_map():
    pred = _blob((40, 60), 10, 20)
    assert nss(pred, np.array([10]), np.array([20])) > 3
    assert nss(pred, np.array([39]), np.array([59])) < 0
    assert np.isnan(nss(np.ones((5, 5)), np.array([1]), np.array([1])))


def test_baseline_maps():
    cb = center_bias_map((45, 72))
    assert cb.shape == (45, 72) and cb.max() == pytest.approx(1.0)
    r, c = np.unravel_index(cb.argmax(), cb.shape)
    assert abs(c - 36) <= 1
    img = np.full((90, 144, 3), 255, dtype=np.uint8)
    img[30:50, 60:90] = 0  # one high-contrast object
    sr = spectral_residual_saliency(img, (45, 72))
    assert sr.shape == (45, 72) and sr.min() >= 0 and sr.max() == pytest.approx(1.0)
    r, c = np.unravel_index(sr.argmax(), sr.shape)
    assert 10 <= r <= 30 and 25 <= c <= 50
