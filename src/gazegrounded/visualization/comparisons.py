"""Human-vs-VLM comparison figures and baseline / modality comparison charts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.gaze.heatmap import normalize_map
from gazegrounded.visualization import style
from gazegrounded.visualization.gaze import draw_heatmap, show_screenshot
from gazegrounded.visualization.style import plt


def human_vs_vlm_figure(
    image: np.ndarray,
    human_map: np.ndarray,
    vlm_map: np.ndarray,
    title: str,
    metrics: dict[str, float] | None = None,
):
    """Human heatmap | VLM relevance map | difference (red = humans looked more than the VLM
    expected, blue = the VLM expected more attention than humans gave)."""
    h, w = image.shape[:2]
    hn, vn = normalize_map(human_map, "sum"), normalize_map(vlm_map, "sum")
    diff = hn - vn
    lim = float(np.abs(diff).max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    show_screenshot(axes[0], image, dim=0.35)
    draw_heatmap(axes[0], normalize_map(human_map, "max"), w, h, cmap=style.HUMAN_OVERLAY)
    axes[0].set_title("Human gaze (all participants)")
    show_screenshot(axes[1], image, dim=0.35)
    draw_heatmap(axes[1], normalize_map(vlm_map, "max"), w, h, cmap=style.VLM_OVERLAY)
    axes[1].set_title("VLM Task-Relevance Map")
    show_screenshot(axes[2], image, dim=0.55)
    axes[2].imshow(
        diff,
        cmap=style.DIVERGING,
        vmin=-lim,
        vmax=lim,
        alpha=0.75,
        extent=(0, w, h, 0),
        interpolation="bilinear",
    )
    axes[2].set_title("Divergence: red = human > VLM, blue = VLM > human")
    if metrics:
        text = "   ".join(f"{k} {v:.2f}" for k, v in metrics.items())
        fig.text(0.01, 0.0, text, fontsize=9, color=style.INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig


def baseline_comparison_figure(
    summary: pd.DataFrame, metric: str, title: str, higher_is_better: bool = True
):
    """Dot + interval chart: mean metric per baseline across tasks, with min-max range.

    ``summary`` columns: ``baseline, mean, min, max, n``.
    """
    d = summary.sort_values("mean", ascending=not higher_is_better).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(d) + 1.3))
    for i, row in d.iterrows():
        is_vlm = row["baseline"].startswith("D")
        color = style.VLM if is_vlm else style.INK_2
        ax.plot(
            [row["min"], row["max"]],
            [i, i],
            color=style.NEUTRAL_BAR,
            lw=3,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [row["mean"]], [i], s=70, color=color, zorder=2, edgecolors="white", linewidths=1.5
        )
        ax.text(
            row["max"],
            i,
            f"  {row['mean']:.2f}  (n={int(row['n'])})",
            va="center",
            fontsize=8.5,
            color=style.INK_2,
        )
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["baseline"])
    ax.tick_params(axis="y", length=0, labelcolor=style.INK_2)
    ax.invert_yaxis()
    arrow = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel(f"{metric.upper()} vs human gaze ({arrow}); dot = mean, bar = range over tasks")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def divergence_scatter(
    df: pd.DataFrame, xs: list[str], y: str, title: str, hue: str = "friction_label"
):
    """One scatter panel per divergence measure vs a behavioural outcome, coloured by label."""
    fig, axes = plt.subplots(1, len(xs), figsize=(5.4 * len(xs), 4.2), sharey=True, squeeze=False)
    for ax, x in zip(axes[0], xs, strict=True):
        for label, color in style.FRICTION_COLORS.items():
            d = df[df[hue] == label]
            ax.scatter(
                d[x],
                d[y],
                s=26,
                color=color,
                alpha=0.85,
                label=f"{label} friction",
                edgecolors="white",
                linewidths=0.6,
            )
        ax.set_xlabel(x.replace("_", " "))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(x.replace("_", " "), loc="left", fontsize=11)
    axes[0][0].set_ylabel(y.replace("_", " "))
    axes[0][0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig
