import numpy as np
import pandas as pd
import pytest

from gazegrounded.gaze.aoi import (
    aoi_area_fractions,
    assign_fixations_to_aois,
    assign_points_to_aois,
    box_iou,
)

AOIS = pd.DataFrame(
    {
        "ui_id": ["u"] * 3,
        "aoi_id": ["card", "button", "header"],
        "x_min": [0.1, 0.3, 0.0],
        "y_min": [0.2, 0.4, 0.0],
        "x_max": [0.6, 0.4, 1.0],
        "y_max": [0.8, 0.5, 0.1],
    }
)


def test_point_assignment_prefers_smallest_containing_aoi():
    x = np.array([350.0, 150.0, 500.0, 900.0])  # button-in-card, card, header, nothing
    y = np.array([450.0, 300.0, 50.0, 900.0])
    out = assign_points_to_aois(x, y, AOIS, 1000, 1000)
    assert list(out) == ["button", "card", "header", None]


def test_boundaries_are_inclusive():
    out = assign_points_to_aois(np.array([100.0]), np.array([200.0]), AOIS, 1000, 1000)
    assert out[0] == "card"


def test_assign_fixations_uses_each_ui_size():
    uis = pd.DataFrame({"ui_id": ["u"], "width": [2000], "height": [1000]})
    fx = pd.DataFrame(
        {"ui_id": ["u", "u"], "centroid_x": [700.0, 1500.0], "centroid_y": [450.0, 500.0]}
    )
    out = assign_fixations_to_aois(fx, AOIS, uis)
    assert out["aoi_id"].tolist() == ["button", None]


def test_area_and_iou():
    areas = aoi_area_fractions(AOIS)
    assert areas["button"] == pytest.approx(0.01)
    assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert box_iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_margin_captures_near_misses_but_keeps_smallest_rule():
    x, y = np.array([350.0, 395.0]), np.array([505.0, 450.0])  # 5 px below button; inside card
    assert assign_points_to_aois(x, y, AOIS, 1000, 1000)[0] == "card"
    out = assign_points_to_aois(x, y, AOIS, 1000, 1000, margin_px=10)
    assert list(out) == ["button", "button"]
