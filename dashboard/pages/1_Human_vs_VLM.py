"""RQ1: pooled human gaze vs the VLM Task-Relevance Map, and the baseline comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

import ui  # noqa: E402
from common import needs, report_path, sidebar_config  # noqa: E402

from gazegrounded.analysis.saliency_metrics import compare_maps  # noqa: E402
from gazegrounded.pipeline import task_human_map  # noqa: E402
from gazegrounded.visualization import dashboard_utils as du  # noqa: E402
from gazegrounded.visualization import style  # noqa: E402

data = sidebar_config()
ui.page_header(
    "Human gaze vs model relevance",
    "For each task: where all participants looked, where the task-aware model expected the "
    "relevant information to be, and how well the model compares with task-agnostic baselines.",
    data,
)
needs(data, "fixations", "vlm_maps")
ds, cfg = data.ds, data.cfg
tasks = ds.tasks.set_index("task_id")

task_id = st.selectbox(
    "Task",
    tasks.index.tolist(),
    format_func=lambda t: f"{ui.ui_name(tasks.loc[t, 'ui_id'])} "
    f"({ui.label(ds.uis.set_index('ui_id').loc[tasks.loc[t, 'ui_id'], 'condition'])})"
    f" — {tasks.loc[t, 'task_description']}",
)
ui_id = tasks.loc[task_id, "ui_id"]
image = Image.open(ds.image_path(ui_id)).convert("RGB")
human, f = task_human_map(cfg, ds, data.fixations, task_id)
vlm = data.vlm_maps[task_id]
m = compare_maps(human, vlm)

c = st.columns(5)
c[0].metric("Participants", f["trial_id"].nunique(), border=True)
c[1].metric(
    "Correlation (CC)",
    f"{m['cc']:.2f}",
    border=True,
    help="Pearson correlation of the two maps. Higher = closer",
)
c[2].metric(
    "Overlap (SIM)",
    f"{m['sim']:.2f}",
    border=True,
    help="Histogram intersection of the two distributions. 1 = identical",
)
c[3].metric(
    "KL (human ‖ model)",
    f"{m['kl']:.2f}",
    border=True,
    help="Lower = the model covers where people looked. Penalises missed regions",
)
c[4].metric(
    "Divergence (JSD)",
    f"{m['jsd']:.2f}",
    border=True,
    help="Symmetric divergence. 0 = same, 1 = no overlap",
)

left, right = st.columns(2, gap="medium")
with left, st.container(border=True):
    ui.section("Human gaze", ":material/visibility:", "Pooled over all participants")
    st.plotly_chart(
        du.overlay_figure(image, human / max(human.max(), 1e-12), "human", height_px=320),
        width="stretch",
        config=ui.PLOTLY_CONFIG,
        key="hv_h",
    )
with right, st.container(border=True):
    ui.section("Model relevance", ":material/smart_toy:", "The model's estimate, not gaze")
    st.plotly_chart(
        du.overlay_figure(image, vlm, "vlm", height_px=320),
        width="stretch",
        config=ui.PLOTLY_CONFIG,
        key="hv_v",
    )
with st.container(border=True):
    ui.section(
        "Where they disagree",
        ":material/difference:",
        "Red: people looked more than the model expected. Blue: the model expected "
        "attention people did not give.",
    )
    st.plotly_chart(
        du.divergence_figure(image, human, vlm),
        width="stretch",
        config=ui.PLOTLY_CONFIG,
        key="hv_d",
    )

st.space("medium")
ui.section("Does the model beat task-agnostic baselines? (RQ1)", ":material/leaderboard:")
p = report_path(data, "rq1_attention_baselines_per_task.csv")
if p is None:
    st.info(
        "Run `python -m gazegrounded.cli evaluate` to compute the baselines.",
        icon=":material/info:",
    )
else:
    per_task = pd.read_csv(p)
    metric = (
        st.segmented_control(
            "Metric",
            ["cc", "sim", "nss", "kl"],
            default="cc",
            key="rq1_metric",
            format_func=lambda x: {
                "cc": "Correlation",
                "sim": "Overlap",
                "nss": "NSS",
                "kl": "KL (lower is better)",
            }[x],
        )
        or "cc"
    )
    s = per_task.groupby("baseline")[metric].agg(["mean", "min", "max", "count"]).reset_index()
    s = s.sort_values("mean", ascending=metric == "kl")
    fig = go.Figure()
    for _, r in s.iterrows():
        is_vlm = r["baseline"].startswith("D")
        fig.add_trace(
            go.Scatter(
                x=[r["min"], r["max"]],
                y=[r["baseline"]] * 2,
                mode="lines",
                line=dict(color=style.NEUTRAL_BAR, width=8),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["mean"]],
                y=[r["baseline"]],
                mode="markers",
                marker=dict(
                    size=15,
                    color=style.VLM if is_vlm else style.INK_2,
                    line=dict(color="white", width=2),
                ),
                hovertemplate=f"mean %{{x:.2f}} · range {r['min']:.2f}–{r['max']:.2f} · "
                f"n={int(r['count'])} tasks<extra></extra>",
            )
        )
    fig.update_layout(
        showlegend=False,
        height=250,
        margin=dict(l=0, r=10, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor=style.GRID, title=f"{metric.upper()} vs human gaze"),
    )
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="hv_rq1")
        st.caption(
            "Dot: mean over tasks. Bar: range. Baseline C is scored only on screenshots "
            "held out from its training run, so its n is smaller."
        )
    with st.expander("Per-task values", icon=":material/table:"):
        st.dataframe(
            per_task.drop(columns=["baseline_key"]).round(3), hide_index=True, width="stretch"
        )
