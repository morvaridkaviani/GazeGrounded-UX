"""VLM figures: grounded boxes with relevance scores and the Task-Relevance Map overlay."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from gazegrounded.visualization import style
from gazegrounded.visualization.gaze import draw_heatmap, show_screenshot
from gazegrounded.visualization.style import plt
from gazegrounded.vlm.schemas import VLMAnalysis


def draw_vlm_boxes(
    ax, analysis: VLMAnalysis, width: int, height: int, threshold: float = 0.5, max_labels: int = 6
) -> None:
    """Boxes in VLM blue; line weight encodes relevance; top elements labelled with scores."""
    ranked = sorted(analysis.elements, key=lambda e: -e.weight)
    for i, el in enumerate(ranked):
        b = el.bbox
        strong = el.task_relevance >= threshold
        ax.add_patch(
            Rectangle(
                (b.x_min * width, b.y_min * height),
                (b.x_max - b.x_min) * width,
                (b.y_max - b.y_min) * height,
                fill=False,
                ec=style.VLM,
                lw=2.2 if strong else 1.0,
                ls="-" if strong else ":",
                alpha=0.95,
            )
        )
        if i < max_labels:
            ax.text(
                b.x_min * width,
                b.y_min * height - 6,
                f"{el.label[:28]}  rel {el.task_relevance:.2f}",
                fontsize=7,
                color="white",
                va="bottom",
                bbox={"fc": style.VLM, "ec": "none", "pad": 1.5, "alpha": 0.9},
            )


def vlm_figure(
    image: np.ndarray,
    analysis: VLMAnalysis,
    vlm_map: np.ndarray,
    title: str,
    threshold: float = 0.5,
):
    h, w = image.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    show_screenshot(axes[0], image, dim=0.25)
    draw_vlm_boxes(axes[0], analysis, w, h, threshold)
    axes[0].set_title("VLM grounding (solid = task-relevant)")
    show_screenshot(axes[1], image, dim=0.4)
    draw_heatmap(axes[1], vlm_map, w, h, cmap=style.VLM_OVERLAY)
    axes[1].set_title("VLM Task-Relevance Map (an estimate, not gaze)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig
