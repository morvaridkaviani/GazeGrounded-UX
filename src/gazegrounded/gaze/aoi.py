"""Areas of Interest: point-in-box assignment and AOI geometry helpers.

AOIs are stored with normalised boxes; gaze and fixations are in pixels. When AOIs overlap
(e.g. a button inside a card), a point is assigned to the *smallest* containing AOI, i.e. the
most specific element.

``margin_px`` pads every AOI when testing containment. Eye trackers have an accuracy error
(lab trackers ~0.5 deg, webcams several degrees), so fixations on small targets often land just
outside the drawn box. Padding by roughly the tracker's accuracy is standard practice
(Holmqvist et al., 2011); the smallest-box rule still uses the unpadded areas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aoi_boxes_px(aois: pd.DataFrame, width: int, height: int) -> np.ndarray:
    """``(n_aoi, 4)`` array of pixel boxes ``x_min, y_min, x_max, y_max``."""
    return aois[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float) * np.array(
        [width, height, width, height], dtype=float
    )


def assign_points_to_aois(
    x: np.ndarray,
    y: np.ndarray,
    aois: pd.DataFrame,
    width: int,
    height: int,
    margin_px: float = 0.0,
) -> np.ndarray:
    """Return an object array with the AOI id containing each point (``None`` if outside all)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.full(len(x), None, dtype=object)
    if aois.empty or len(x) == 0:
        return result
    boxes = aoi_boxes_px(aois, width, height)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    m = float(margin_px)
    inside = (
        (x[:, None] >= boxes[None, :, 0] - m)
        & (x[:, None] <= boxes[None, :, 2] + m)
        & (y[:, None] >= boxes[None, :, 1] - m)
        & (y[:, None] <= boxes[None, :, 3] + m)
    )
    masked_area = np.where(inside, areas[None, :], np.inf)
    best = masked_area.argmin(axis=1)
    hit = np.isfinite(masked_area[np.arange(len(x)), best])
    ids = aois["aoi_id"].to_numpy(dtype=object)
    result[hit] = ids[best[hit]]
    return result


def assign_fixations_to_aois(
    fixations: pd.DataFrame, aois: pd.DataFrame, uis: pd.DataFrame, margin_px: float = 0.0
) -> pd.DataFrame:
    """Fill ``aoi_id`` for every fixation using the AOIs of the fixation's interface."""
    out = fixations.copy()
    out["aoi_id"] = None
    sizes = uis.set_index("ui_id")[["width", "height"]]
    for ui_id, idx in out.groupby("ui_id").groups.items():
        w, h = int(sizes.loc[ui_id, "width"]), int(sizes.loc[ui_id, "height"])
        ui_aois = aois[aois["ui_id"] == ui_id]
        out.loc[idx, "aoi_id"] = assign_points_to_aois(
            out.loc[idx, "centroid_x"].to_numpy(),
            out.loc[idx, "centroid_y"].to_numpy(),
            ui_aois,
            w,
            h,
            margin_px,
        )
    return out


def aoi_area_fractions(aois: pd.DataFrame) -> pd.Series:
    """Fraction of the screen covered by each AOI (useful to normalise dwell by size)."""
    area = (aois["x_max"] - aois["x_min"]) * (aois["y_max"] - aois["y_min"])
    return pd.Series(area.to_numpy(), index=aois["aoi_id"].to_numpy(), name="area_fraction")


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-union of two ``(x_min, y_min, x_max, y_max)`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
