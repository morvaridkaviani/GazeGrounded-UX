"""Shared figure style: one palette, recessive chrome, and the synthetic-data banner.

Colour roles are fixed across the project so a reader learns them once:
human gaze = warm (orange), VLM task relevance = blue, divergence = blue <-> red with a
neutral grey midpoint. Categorical hues are assigned in a fixed order, never cycled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # headless: scripts, CI and Streamlit render off-screen; notebooks inline

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from gazegrounded.utils import SYNTHETIC_BANNER, ensure_dir  # noqa: E402

# Fixed categorical order, chosen so neighbouring hues stay distinguishable with colour-vision deficiencies.
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
HUMAN = "#eb6834"
VLM = "#2a78d6"
NEUTRAL_BAR = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
DIVERGING_MID = "#f0efec"
FRICTION_COLORS = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#0d366b"}  # ordinal blue

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#f9bfa3", "#f39a70", "#eb6834", "#c94f1f", "#9c3a14", "#6e280c"]
RED = "#e34948"


def sequential_cmap(ramp: list[str], name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ramp)


def overlay_cmap(color: str, name: str) -> LinearSegmentedColormap:
    """Transparent -> opaque single-hue colormap for heatmaps drawn over screenshots."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        name, [(r, g, b, 0.0), (r, g, b, 0.35), (r, g, b, 0.75)], N=256
    )


HUMAN_OVERLAY = overlay_cmap(HUMAN, "human_overlay")
VLM_OVERLAY = overlay_cmap(VLM, "vlm_overlay")
BLUES = sequential_cmap(BLUE_RAMP, "gg_blues")
DIVERGING = LinearSegmentedColormap.from_list("gg_diverging", [VLM, DIVERGING_MID, RED])
FRICTION_CMAP = ListedColormap(list(FRICTION_COLORS.values()))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def add_banner(fig, synthetic: bool) -> None:
    """Stamp the synthetic-data notice at the bottom of a figure (no-op for real data)."""
    if synthetic:
        fig.text(
            0.5,
            -0.01,
            SYNTHETIC_BANNER,
            ha="center",
            va="top",
            fontsize=8.5,
            color=RED,
            fontweight="bold",
        )


def save_figure(fig, path: str | Path, synthetic: bool) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    add_banner(fig, synthetic)
    fig.savefig(p)
    plt.close(fig)
    return p


apply_style()
