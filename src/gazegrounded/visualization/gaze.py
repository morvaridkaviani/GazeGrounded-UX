"""Gaze figures: heatmap overlays, fixations, scanpaths, AOI boxes and AOI dwell bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from gazegrounded.visualization import style
from gazegrounded.visualization.style import plt


def load_image(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def show_screenshot(ax, image: np.ndarray, dim: float = 0.0) -> None:
    """Draw the screenshot; ``dim`` in [0, 1] washes it out so overlays read clearly."""
    ax.imshow(image)
    if dim > 0:
        ax.imshow(
            np.ones_like(image, dtype=float),
            alpha=dim,
            extent=(0, image.shape[1], image.shape[0], 0),
        )
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_aois(ax, aois: pd.DataFrame, width: int, height: int, targets=(), labels=False) -> None:
    """Thin grey AOI outlines; task-relevant (target) AOIs dashed in ink."""
    for row in aois.itertuples():
        x0, y0 = row.x_min * width, row.y_min * height
        is_target = row.aoi_id in targets
        ax.add_patch(
            Rectangle(
                (x0, y0),
                (row.x_max - row.x_min) * width,
                (row.y_max - row.y_min) * height,
                fill=False,
                lw=1.8 if is_target else 0.8,
                ls="--" if is_target else "-",
                ec=style.INK if is_target else style.MUTED,
                alpha=0.9,
            )
        )
        if labels:
            ax.text(
                x0 + 4,
                y0 + 4,
                row.aoi_id,
                fontsize=6.5,
                color=style.INK_2,
                va="top",
                bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1},
            )


def draw_heatmap(
    ax, heat: np.ndarray, width: int, height: int, cmap=style.HUMAN_OVERLAY, gamma: float = 0.6
) -> None:
    """Overlay a [0, 1] map. ``gamma < 1`` brightens weaker regions for *display only*; all
    metrics are computed on the unscaled maps."""
    ax.imshow(
        np.clip(heat, 0, 1) ** gamma,
        cmap=cmap,
        extent=(0, width, height, 0),
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )


def draw_fixations(ax, fixations: pd.DataFrame, color: str = style.HUMAN, numbered=False) -> None:
    """Circles sized by fixation duration (area proportional to duration)."""
    f = fixations.sort_values("start_ms")
    sizes = 20 + f["duration_ms"].to_numpy() / 3.0
    ax.scatter(
        f["centroid_x"],
        f["centroid_y"],
        s=sizes,
        c=color,
        alpha=0.55,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    if numbered:
        for i, row in enumerate(f.itertuples(), start=1):
            ax.text(
                row.centroid_x,
                row.centroid_y,
                str(i),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                zorder=4,
            )


def draw_scanpath(ax, fixations: pd.DataFrame, color: str = style.HUMAN) -> None:
    """Numbered fixation trajectory."""
    f = fixations.sort_values("start_ms")
    ax.plot(f["centroid_x"], f["centroid_y"], color=color, lw=1.4, alpha=0.7, zorder=2)
    draw_fixations(ax, f, color=color, numbered=True)


def trial_figure(
    image: np.ndarray,
    fixations: pd.DataFrame,
    aois: pd.DataFrame,
    targets: list[str],
    heat: np.ndarray,
    title: str,
):
    """Three panels: duration-weighted heatmap | fixations + AOIs | numbered scanpath."""
    h, w = image.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    show_screenshot(axes[0], image, dim=0.35)
    draw_heatmap(axes[0], heat, w, h)
    axes[0].set_title("Gaze heatmap (duration-weighted)")
    show_screenshot(axes[1], image, dim=0.35)
    draw_aois(axes[1], aois, w, h, targets)
    draw_fixations(axes[1], fixations)
    axes[1].set_title("Fixations and AOIs (dashed = task-relevant)")
    show_screenshot(axes[2], image, dim=0.45)
    draw_scanpath(axes[2], fixations)
    axes[2].set_title("Scanpath (numbered)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig


def aoi_dwell_figure(aoi_summary: pd.DataFrame, title: str):
    """Horizontal bars of mean dwell time per AOI; task-relevant AOIs highlighted.

    ``aoi_summary`` needs columns ``aoi_id, mean_dwell_ms, is_target``.
    """
    d = aoi_summary.sort_values("mean_dwell_ms")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1.2))
    colors = [style.HUMAN if t else style.NEUTRAL_BAR for t in d["is_target"]]
    ax.barh(d["aoi_id"], d["mean_dwell_ms"], color=colors, height=0.68)
    ax.set_xlabel("Mean dwell time per trial (ms)")
    ax.set_title(title, loc="left")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelcolor=style.INK_2)
    for y, v in enumerate(d["mean_dwell_ms"]):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=style.INK_2)
    ax.text(
        1.0,
        -0.12 - 0.3 / max(len(d), 1),
        "orange = task-relevant AOI",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=style.INK_2,
    )
    fig.tight_layout()
    return fig
