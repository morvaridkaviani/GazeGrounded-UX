"""Model figures: confusion matrices, modality comparison, feature importance, top terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gazegrounded.visualization import style
from gazegrounded.visualization.style import plt


def confusion_matrix_figure(cm: list[list[int]], labels: list[str], title: str):
    """Row-normalised colour (recall per true class) with raw counts written in each cell."""
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    size = 1.0 + 0.62 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    ax.imshow(norm, cmap=style.BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if norm[i, j] > 0.55 else style.INK,
            )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([lab.replace("_", " ") for lab in labels], rotation=35, ha="right")
    ax.set_yticklabels([lab.replace("_", " ") for lab in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def modality_comparison_figure(
    results: pd.DataFrame, metric: str, title: str, chance: float | None = None
):
    """Horizontal bars per feature set, mean +- fold SD; multimodal highlighted.

    ``results`` columns: ``modality, model, <metric>_fold_mean, <metric>_fold_sd``.
    One panel per model type; one colour for all single modalities, blue for the fusion.
    """
    models = list(dict.fromkeys(results["model"]))
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(5.2 * len(models), 0.45 * results["modality"].nunique() + 1.4),
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes[0], models, strict=True):
        d = results[results["model"] == model].reset_index(drop=True)
        mean, sd = d[f"{metric}_fold_mean"], d[f"{metric}_fold_sd"].fillna(0)
        colors = [style.VLM if m == "multimodal" else style.NEUTRAL_BAR for m in d["modality"]]
        y = np.arange(len(d))
        ax.barh(
            y,
            mean,
            color=colors,
            height=0.62,
            xerr=sd,
            error_kw={"ecolor": style.INK_2, "lw": 1, "capsize": 2},
        )
        for i, v in enumerate(mean):
            ax.text(
                min(v + sd[i], 1.0) + 0.015,
                i,
                f"{v:.2f}",
                va="center",
                fontsize=8.5,
                color=style.INK_2,
            )
        if chance is not None:
            ax.axvline(chance, color=style.MUTED, lw=1, ls="--")
            ax.text(chance, len(d) - 0.4, " majority-class", fontsize=7.5, color=style.MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(d["modality"].str.replace("_", " "))
        ax.tick_params(axis="y", length=0, labelcolor=style.INK_2)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel(f"{metric.replace('_', ' ')} (mean ± SD over folds)")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(model.replace("_", " "), loc="left", fontsize=11)
    axes[0][0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig


def feature_importance_figure(
    imp: pd.DataFrame, value_col: str, title: str, top: int = 15, signed: bool = True
):
    """Top features by |importance|; signed values use the blue/red diverging poles."""
    d = imp.reindex(imp[value_col].abs().sort_values(ascending=False).index).head(top)
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(d) + 1.3))
    if signed:
        colors = [style.RED if v > 0 else style.VLM for v in d[value_col]]
    else:
        colors = [style.VLM] * len(d)
    ax.barh(d["feature"], d[value_col], color=colors, height=0.66)
    ax.axvline(0, color=style.MUTED, lw=0.8)
    ax.tick_params(axis="y", length=0, labelcolor=style.INK_2)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def top_terms_figure(terms: pd.DataFrame, title: str, per_class: int = 5):
    classes = list(dict.fromkeys(terms["class"]))
    ncols = min(4, len(classes))
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 1.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, cls in zip(axes.ravel(), classes, strict=False):
        d = terms[terms["class"] == cls].head(per_class).iloc[::-1]
        ax.axis("on")
        ax.barh(d["term"], d["coefficient"], color=style.VLM, height=0.6)
        ax.set_title(cls.replace("_", " "), loc="left", fontsize=10)
        ax.tick_params(axis="y", length=0, labelsize=8, labelcolor=style.INK_2)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=style.INK)
    fig.tight_layout()
    return fig
