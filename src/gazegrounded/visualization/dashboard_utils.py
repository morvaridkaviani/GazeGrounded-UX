"""Data loading and Plotly figure builders for the Streamlit dashboard.

Kept free of Streamlit imports so the builders can be unit-tested; the app adds caching.
Colour roles match the static figures: human gaze = orange, VLM relevance = blue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from gazegrounded.data.loader import StandardizedDataset, read_table
from gazegrounded.utils import latest_run_dir
from gazegrounded.visualization import style

HUMAN_SCALE = [
    [0.0, "rgba(235,104,52,0)"],
    [0.25, "rgba(235,104,52,0.25)"],
    [1.0, "rgba(201,79,31,0.85)"],
]
VLM_SCALE = [
    [0.0, "rgba(42,120,214,0)"],
    [0.25, "rgba(42,120,214,0.25)"],
    [1.0, "rgba(37,106,191,0.85)"],
]
DIVERGING_SCALE = [[0.0, style.VLM], [0.5, "rgba(240,239,236,0.0)"], [1.0, style.RED]]


@dataclass
class DashboardData:
    """Everything the dashboard reads, loaded once per config."""

    cfg: object
    ds: StandardizedDataset | None = None
    fixations: pd.DataFrame | None = None
    gaze_metrics: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    vlm_maps: dict = field(default_factory=dict)
    vlm_analyses: dict = field(default_factory=dict)
    vlm_tasks: pd.DataFrame | None = None
    vlm_elements: pd.DataFrame | None = None
    aoi_metrics: pd.DataFrame | None = None
    nlp_predictions: pd.DataFrame | None = None
    friction_run: Path | None = None
    friction_predictions: pd.DataFrame | None = None
    contributions: pd.DataFrame | None = None
    report_dir: Path | None = None
    missing: list[str] = field(default_factory=list)


def _maybe_csv(path: Path, missing: list[str], label: str) -> pd.DataFrame | None:
    if path.exists():
        return read_table(path)
    missing.append(label)
    return None


def load_dashboard_data(cfg) -> DashboardData:
    from gazegrounded.pipeline import load_vlm_analyses, load_vlm_maps

    d = DashboardData(cfg=cfg)
    try:
        d.ds = StandardizedDataset.load(cfg.processed_dir())
    except FileNotFoundError:
        d.missing.append("preprocess")
        return d
    interim = cfg.interim_dir()
    d.fixations = _maybe_csv(interim / "fixations.csv", d.missing, "gaze-analysis")
    d.gaze_metrics = _maybe_csv(interim / "gaze_metrics.csv", d.missing, "gaze-analysis")
    d.aoi_metrics = _maybe_csv(interim / "aoi_metrics.csv", d.missing, "gaze-analysis")
    d.features = _maybe_csv(cfg.processed_dir() / "features.csv", d.missing, "train-friction")
    if (interim / "vlm" / "vlm_maps.npz").exists():
        d.vlm_maps = load_vlm_maps(cfg)
        d.vlm_analyses = load_vlm_analyses(cfg)
        d.vlm_tasks = pd.read_csv(interim / "vlm" / "vlm_task_features.csv")
        el = interim / "vlm" / "vlm_elements.csv"
        d.vlm_elements = read_table(el) if el.exists() else None
    else:
        d.missing.append("vlm-analysis")
    p = interim / "nlp_predictions.csv"
    d.nlp_predictions = read_table(p) if p.exists() else None
    d.friction_run = latest_run_dir(cfg.experiments_dir(), "friction")
    if d.friction_run is not None:
        d.friction_predictions = pd.read_csv(d.friction_run / "predictions.csv")
        c = d.friction_run / "contributions.csv"
        d.contributions = pd.read_csv(c) if c.exists() else None
    d.report_dir = cfg.reports_dir() / cfg.data.source
    return d


# ---- figures ----------------------------------------------------------------------------------


def _base_image_figure(image: Image.Image, height_px: int = 520) -> go.Figure:
    w, h = image.size
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            layer="below",
            opacity=0.9,
        )
    )
    fig.update_xaxes(range=[0, w], visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x")
    fig.update_layout(
        height=height_px,
        margin=dict(l=0, r=0, t=4, b=0),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="white", bordercolor=style.GRID, font_size=12, font_color=style.INK
        ),
    )
    return fig


def _add_map(fig: go.Figure, m: np.ndarray, w: int, h: int, scale, name: str, gamma=0.6):
    rows, cols = m.shape
    fig.add_trace(
        go.Heatmap(
            z=np.clip(m, 0, 1) ** gamma,
            x0=w / cols / 2,
            dx=w / cols,
            y0=h / rows / 2,
            dy=h / rows,
            colorscale=scale,
            zmin=0,
            zmax=1,
            showscale=False,
            name=name,
            hovertemplate=f"{name}: %{{z:.2f}}<extra></extra>",
        )
    )


def overlay_figure(
    image: Image.Image, m: np.ndarray, kind: str = "human", height_px: int = 520
) -> go.Figure:
    """Screenshot with a [0, 1] map overlaid (human gaze or VLM relevance)."""
    fig = _base_image_figure(image, height_px)
    w, h = image.size
    _add_map(
        fig,
        m,
        w,
        h,
        HUMAN_SCALE if kind == "human" else VLM_SCALE,
        "gaze density" if kind == "human" else "VLM relevance",
    )
    return fig


def divergence_figure(
    image: Image.Image, human: np.ndarray, vlm: np.ndarray, height_px: int = 520
) -> go.Figure:
    """Difference of the two normalised distributions: red = human > VLM, blue = VLM > human."""
    hn = human / human.sum() if human.sum() > 0 else human
    vn = vlm / vlm.sum() if vlm.sum() > 0 else vlm
    diff = hn - vn
    lim = float(np.abs(diff).max()) or 1.0
    fig = _base_image_figure(image, height_px)
    w, h = image.size
    rows, cols = diff.shape
    fig.add_trace(
        go.Heatmap(
            z=diff,
            x0=w / cols / 2,
            dx=w / cols,
            y0=h / rows / 2,
            dy=h / rows,
            colorscale=DIVERGING_SCALE,
            zmin=-lim,
            zmax=lim,
            showscale=False,
            hovertemplate="human - VLM: %{z:.4f}<extra></extra>",
        )
    )
    return fig


def fixation_figure(
    image: Image.Image,
    fixations: pd.DataFrame,
    aois: pd.DataFrame,
    targets: list[str],
    scanpath: bool = True,
) -> go.Figure:
    """Fixations sized by duration (hover: order, duration, AOI) and AOI outlines."""
    fig = _base_image_figure(image)
    w, h = image.size
    for a in aois.itertuples():
        is_t = a.aoi_id in targets
        fig.add_shape(
            type="rect",
            x0=a.x_min * w,
            y0=a.y_min * h,
            x1=a.x_max * w,
            y1=a.y_max * h,
            line=dict(
                color=style.INK if is_t else style.MUTED,
                width=2 if is_t else 1,
                dash="dash" if is_t else "solid",
            ),
        )
    f = fixations.sort_values("start_ms").reset_index(drop=True)
    if len(f):
        if scanpath:
            fig.add_trace(
                go.Scatter(
                    x=f["centroid_x"],
                    y=f["centroid_y"],
                    mode="lines",
                    line=dict(color=style.HUMAN, width=1.5),
                    opacity=0.6,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=f["centroid_x"],
                y=f["centroid_y"],
                mode="markers+text" if scanpath else "markers",
                text=[str(i + 1) for i in range(len(f))] if scanpath else None,
                textfont=dict(color="white", size=9),
                marker=dict(
                    size=8 + np.sqrt(f["duration_ms"]) * 0.9,
                    color=style.HUMAN,
                    opacity=0.7,
                    line=dict(color="white", width=1.2),
                ),
                customdata=np.stack(
                    [
                        np.arange(1, len(f) + 1),
                        f["duration_ms"],
                        f["aoi_id"].fillna("outside AOIs"),
                        f["start_ms"],
                    ],
                    axis=1,
                ),
                hovertemplate="fixation %{customdata[0]} at %{customdata[3]:.0f} ms<br>"
                "duration %{customdata[1]:.0f} ms<br>AOI: %{customdata[2]}<extra></extra>",
            )
        )
    return fig


def vlm_boxes_figure(
    image: Image.Image, analysis, threshold: float = 0.5, height_px: int = 520
) -> go.Figure:
    fig = _base_image_figure(image, height_px)
    w, h = image.size
    for el in sorted(analysis.elements, key=lambda e: -e.weight):
        b = el.bbox
        strong = el.task_relevance >= threshold
        fig.add_shape(
            type="rect",
            x0=b.x_min * w,
            y0=b.y_min * h,
            x1=b.x_max * w,
            y1=b.y_max * h,
            line=dict(
                color=style.VLM, width=3 if strong else 1.2, dash="solid" if strong else "dot"
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[(b.x_min + b.x_max) / 2 * w],
                y=[(b.y_min + b.y_max) / 2 * h],
                mode="markers",
                marker=dict(size=14, color="rgba(42,120,214,0.01)"),
                hovertemplate=(
                    f"<b>{el.label[:50]}</b> ({el.semantic_role})<br>relevance "
                    f"{el.task_relevance:.2f} · priority {el.expected_attention_priority:.2f}"
                    f"<br>{el.reason[:90]}<extra></extra>"
                ),
            )
        )
    return fig


def contributions_figure(contrib: pd.DataFrame, top: int = 8) -> go.Figure:
    """Top signed contributions of one trial's prediction (red pushes towards the class)."""
    d = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).head(top)
    d = d.iloc[::-1]
    names = (
        d["feature"]
        .str.replace("_ms", " (ms)", regex=False)
        .str.replace("text_p_", "text: ", regex=False)
        .str.replace("_", " ")
    )
    fig = go.Figure(
        go.Bar(
            x=d["contribution"],
            y=names,
            orientation="h",
            marker_color=[style.RED if v > 0 else style.VLM for v in d["contribution"]],
            customdata=d["value_std"],
            hovertemplate="%{y}<br>contribution %{x:.2f}<br>value %{customdata:.2f} SD from mean"
            "<extra></extra>",
        )
    )
    fig.update_layout(
        height=40 + 30 * len(d),
        margin=dict(l=0, r=10, t=10, b=30),
        xaxis_title="contribution to predicted class (log-odds)",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=style.GRID, zerolinecolor=style.MUTED)
    return fig


def modality_figure(results: pd.DataFrame, metric: str = "macro_f1") -> go.Figure:
    d = results.copy()
    d["label"] = d["modality"].str.replace("_", " ")
    fig = go.Figure()
    for model, color in (("logistic_regression", style.VLM), ("random_forest", style.HUMAN)):
        g = d[d["model"] == model]
        if len(g):
            fig.add_trace(
                go.Bar(
                    y=g["label"],
                    x=g[f"{metric}_fold_mean"],
                    orientation="h",
                    name=model.replace("_", " "),
                    marker_color=color,
                    error_x=dict(type="data", array=g[f"{metric}_fold_sd"], color=style.INK_2),
                    hovertemplate="%{y}: %{x:.2f}<extra>" + model.replace("_", " ") + "</extra>",
                )
            )
    chance = d[d["model"] == "majority"][f"{metric}_fold_mean"]
    if len(chance):
        fig.add_vline(
            x=float(chance.iloc[0]),
            line_dash="dash",
            line_color=style.MUTED,
            annotation_text="majority class",
            annotation_position="bottom right",
        )
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=0, r=10, t=30, b=30),
        xaxis=dict(
            range=[0, 1],
            title=f"{metric.replace('_', ' ')} " "(mean ± SD over folds)",
            gridcolor=style.GRID,
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


TARGET_COLOR = "#0ca30c"


def failure_case_figure(
    image: Image.Image,
    human_map: np.ndarray | None,
    elements: pd.DataFrame,
    targets: pd.DataFrame,
    layers: set[str],
    height_px: int = 520,
) -> go.Figure:
    """Screenshot with toggleable layers: human gaze, ranked VLM elements, annotated target.

    ``elements`` columns: ``label, task_relevance, reason, x_min..y_max`` (ranked, best first).
    ``targets`` columns: ``aoi_id, label, x_min..y_max``. Labels live in hover tooltips and in
    small numbered markers, so nothing overlaps the screenshot text.
    """
    fig = _base_image_figure(image, height_px)
    w, h = image.size
    if "gaze" in layers and human_map is not None and human_map.max() > 0:
        _add_map(fig, human_map / human_map.max(), w, h, HUMAN_SCALE, "gaze density")
    if "target" in layers:
        for t in targets.itertuples():
            fig.add_shape(
                type="rect",
                x0=t.x_min * w,
                y0=t.y_min * h,
                x1=t.x_max * w,
                y1=t.y_max * h,
                line=dict(color=TARGET_COLOR, width=3, dash="dash"),
                fillcolor="rgba(12,163,12,0.08)",
            )
            fig.add_trace(
                go.Scatter(
                    x=[t.x_max * w],
                    y=[t.y_min * h],
                    mode="markers+text",
                    text=["T"],
                    textfont=dict(color="white", size=11),
                    marker=dict(size=20, color=TARGET_COLOR, line=dict(color="white", width=2)),
                    hovertemplate=f"<b>Annotated target</b><br>{t.label or t.aoi_id}<extra></extra>",
                )
            )
    if "vlm" in layers:
        for rank, e in enumerate(elements.itertuples(), start=1):
            strong = rank == 1
            fig.add_shape(
                type="rect",
                x0=e.x_min * w,
                y0=e.y_min * h,
                x1=e.x_max * w,
                y1=e.y_max * h,
                line=dict(
                    color=style.VLM, width=3.5 if strong else 1.6, dash="solid" if strong else "dot"
                ),
                fillcolor="rgba(42,120,214,0.10)" if strong else None,
            )
            reason = (e.reason or "")[:110]
            fig.add_trace(
                go.Scatter(
                    x=[e.x_min * w],
                    y=[e.y_min * h],
                    mode="markers+text",
                    text=[str(rank)],
                    textfont=dict(color="white", size=11),
                    marker=dict(
                        size=20 if strong else 17,
                        color=style.VLM,
                        line=dict(color="white", width=2),
                    ),
                    hovertemplate=(
                        f"<b>VLM #{rank}: {str(e.label)[:60]}</b><br>"
                        f"score {e.weight:.2f} · relevance {e.task_relevance:.2f}<br>{reason}<extra></extra>"
                    ),
                )
            )
    return fig


def share_bars(rows: list[tuple[str, float, str]], height_px: int = 120) -> go.Figure:
    """Compact horizontal bars of shares in [0, 1]: ``[(label, value, color), ...]``."""
    labels = [r[0] for r in rows][::-1]
    values = [0 if r[1] != r[1] else r[1] for r in rows][::-1]  # NaN -> 0
    colors = [r[2] for r in rows][::-1]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            width=0.55,
            text=[f"{100 * v:.0f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height_px,
        margin=dict(l=0, r=36, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], visible=False),
        bargap=0.35,
        yaxis=dict(tickfont=dict(size=12, color=style.INK_2)),
    )
    return fig
