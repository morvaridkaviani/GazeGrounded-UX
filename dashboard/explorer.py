"""Trial explorer: one participant, one task — gaze, VLM relevance, divergence, explanation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import ui  # noqa: E402
from common import needs, sidebar_config  # noqa: E402

from gazegrounded.gaze.heatmap import fixation_heatmap  # noqa: E402
from gazegrounded.visualization import dashboard_utils as du  # noqa: E402

data = sidebar_config()
ui.page_header(
    "Trial explorer",
    "Where did this person look, where did a task-aware vision-language model expect the "
    "task-relevant information to be, and does the gap reveal UX friction?",
    data,
)
needs(data, "fixations", "gaze_metrics")
ds, cfg = data.ds, data.cfg
uis = ds.uis.set_index("ui_id")

# ---- selection ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Trial**")
    ui_id = st.selectbox(
        "Interface",
        ds.uis["ui_id"].tolist(),
        format_func=lambda u: f"{ui.ui_name(u)} ({ui.label(uis.loc[u, 'condition'])})",
    )
    tasks = ds.tasks[ds.tasks["ui_id"] == ui_id].set_index("task_id")
    task_id = st.selectbox(
        "Task", tasks.index.tolist(), format_func=lambda t: tasks.loc[t, "task_description"]
    )
    trials = ds.trials[ds.trials["task_id"] == task_id]
    participant = st.selectbox("Participant", sorted(trials["participant_id"]))
trial = trials[trials["participant_id"] == participant].iloc[0]
tid = trial["trial_id"]

image = Image.open(ds.image_path(ui_id)).convert("RGB")
w, h = image.size
fix = data.fixations[data.fixations["trial_id"] == tid]
aois = ds.aois_for(ui_id)
targets = ds.targets_for_task(task_id)
gm = data.gaze_metrics.set_index("trial_id").loc[tid]
feat = data.features.set_index("trial_id").loc[tid] if data.features is not None else None

pred = None
if data.friction_predictions is not None:
    p = data.friction_predictions
    sel = p[
        (p["trial_id"] == tid)
        & (p["modality"] == "multimodal")
        & (p["model"] == "logistic_regression")
        & (p["group_by"] == "participant_id")
    ]
    pred = sel.iloc[0] if len(sel) else None

# ---- trial header -----------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(f"### “{trial['task_description']}”")
    with st.container(horizontal=True, gap="small"):
        st.badge(ui.ui_name(ui_id), icon=":material/web:", color="gray")
        ui.condition_badge(trial["condition"])
        st.badge(participant, icon=":material/person:", color="blue")
        if trial["task_success"]:
            st.badge("task completed", icon=":material/check_circle:", color="green")
        else:
            st.badge("task failed", icon=":material/cancel:", color="red")
        st.badge(f"trial {tid}", color="gray")

# ---- metrics ------------------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="medium")
with left:
    ui.section("Behaviour", ":material/directions_run:")
    c = st.columns(2)
    ttff = "never" if np.isnan(gm["ttff_ms"]) else f"{gm['ttff_ms'] / 1000:.1f} s"
    c[0].metric(
        "First look at target",
        ttff,
        border=True,
        help="Time to first fixation on the task-relevant AOI",
    )
    c[1].metric("Completion time", f"{trial['completion_time_ms'] / 1000:.1f} s", border=True)
    c = st.columns(2)
    c[0].metric("Fixations", int(gm["fixation_count"]), border=True)
    c[1].metric(
        "Revisits",
        int(gm["revisit_count"]),
        border=True,
        help="Returns to an AOI after looking elsewhere",
    )
with right:
    ui.section("Alignment & friction", ":material/compare:")
    c = st.columns(2)
    if feat is not None:
        c[0].metric(
            "Gaze–model similarity",
            f"{feat['vlm_human_similarity']:.2f}",
            border=True,
            help="Overlap of the gaze map and the VLM map (1 = identical)",
        )
        c[1].metric(
            "Early divergence",
            f"{feat['attention_divergence_early']:.2f}",
            border=True,
            help="Jensen–Shannon divergence between the first fixations and the VLM map "
            "(0 = same, 1 = no overlap)",
        )
    with st.container(border=True):
        st.caption("Friction level")
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            ui.friction_badge(trial["friction_label"], "label: ")
            ui.friction_badge(pred["pred"] if pred is not None else None, "predicted: ")
            if pred is not None:
                if pred["pred"] == trial["friction_label"]:
                    st.badge("match", icon=":material/check:", color="green")
                else:
                    st.badge("mismatch", icon=":material/close:", color="red")
        st.caption(
            f"Label source: {trial.get('label_source', 'unknown')} · prediction from a "
            "model trained without this participant"
        )

# ---- visual story -----------------------------------------------------------------------------
human = fixation_heatmap(fix, w, h, cfg.gaze.heatmap_sigma_px, cfg.gaze.map_scale)
vlm_map = data.vlm_maps.get(task_id)
cfg_plot = ui.PLOTLY_CONFIG
tabs = st.tabs(
    [
        ":material/visibility: Human gaze",
        ":material/scatter_plot: Fixations & AOIs",
        ":material/timeline: Scanpath",
        ":material/smart_toy: Model relevance",
        ":material/compare: Gaze vs model",
        ":material/image: Screenshot",
    ]
)
with tabs[0]:
    st.plotly_chart(
        du.overlay_figure(image, human, "human"), width="stretch", config=cfg_plot, key="ex_gaze"
    )
    st.caption("Duration-weighted fixation heatmap for this participant (display gamma 0.6).")
with tabs[1]:
    st.plotly_chart(
        du.fixation_figure(image, fix, aois, targets, scanpath=False),
        width="stretch",
        config=cfg_plot,
        key="ex_fix",
    )
    st.caption("Circles are sized by fixation duration. Dashed boxes mark task-relevant AOIs.")
with tabs[2]:
    st.plotly_chart(
        du.fixation_figure(image, fix, aois, targets, scanpath=True),
        width="stretch",
        config=cfg_plot,
        key="ex_scan",
    )
    st.caption("Fixations in order. Hover for timing and AOI.")
with tabs[3]:
    if task_id in data.vlm_analyses:
        a = data.vlm_analyses[task_id]
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            du.vlm_boxes_figure(image, a, cfg.comparison.relevance_threshold, height_px=320),
            width="stretch",
            config=cfg_plot,
            key="ex_vlm_boxes",
        )
        c2.plotly_chart(
            du.overlay_figure(image, vlm_map, "vlm", height_px=320),
            width="stretch",
            config=cfg_plot,
            key="ex_vlm_map",
        )
        st.caption(
            f"Provider `{a.provider}` · model `{a.model}` · prompt {a.prompt_version} · "
            f"parse {a.parse_status}. The map is the model's *estimate* of task-relevant "
            "regions, not measured or predicted gaze."
        )
with tabs[4]:
    if vlm_map is not None:
        c1, c2 = st.columns(2)
        c1.markdown("**Human gaze**")
        c1.plotly_chart(
            du.overlay_figure(image, human, "human", height_px=320),
            width="stretch",
            config=cfg_plot,
            key="ex_cmp_h",
        )
        c2.markdown("**Model relevance**")
        c2.plotly_chart(
            du.overlay_figure(image, vlm_map, "vlm", height_px=320),
            width="stretch",
            config=cfg_plot,
            key="ex_cmp_v",
        )
        st.markdown("**Difference**")
        st.plotly_chart(
            du.divergence_figure(image, human, vlm_map),
            width="stretch",
            config=cfg_plot,
            key="ex_cmp_d",
        )
        st.caption(
            "Red: this participant looked more than the model expected. Blue: the model "
            "expected attention that this participant did not give."
        )
with tabs[5]:
    st.image(image, width="stretch")

# ---- feedback and explanation -----------------------------------------------------------------
left, right = st.columns(2, gap="medium")
with left, st.container(border=True, height="stretch"):
    ui.section("Participant feedback", ":material/chat:")
    text = trial["feedback_text"]
    st.markdown(f"> {text}" if isinstance(text, str) and text.strip() else "_No feedback given._")
    if data.nlp_predictions is not None:
        nlp = data.nlp_predictions[data.nlp_predictions["trial_id"] == tid]
        if len(nlp):
            r = nlp.iloc[0]
            with st.container(horizontal=True, gap="small"):
                st.badge(
                    f"NLP: {ui.label(r['pred'])}",
                    icon=":material/label:",
                    color="blue",
                    help="Out-of-fold prediction from the TF-IDF + logistic regression model",
                )
                st.badge(f"confidence {float(r['confidence']):.2f}", color="gray")
                if isinstance(r["feedback_category"], str):
                    st.badge(f"coded: {ui.label(r['feedback_category'])}", color="gray")
with right, st.container(border=True, height="stretch"):
    ui.section("Why this prediction?", ":material/insights:")
    if pred is not None and data.contributions is not None:
        c = data.contributions[
            (data.contributions["trial_id"] == tid)
            & (data.contributions["model"] == "logistic_regression")
        ]
        st.caption(
            "Multimodal logistic regression, trained without this participant. Red bars "
            "push towards the predicted level, blue bars away from it."
        )
        if len(c):
            st.plotly_chart(
                du.contributions_figure(c),
                width="stretch",
                config={"displayModeBar": False},
                key="ex_contrib",
            )
    else:
        st.info("Run `train-friction` to see model explanations.", icon=":material/info:")
