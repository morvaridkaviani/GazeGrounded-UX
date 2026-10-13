"""Failure cases: tasks where the VLM's top element is not the annotated target."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

import ui  # noqa: E402
from common import needs, report_path, sidebar_config  # noqa: E402

from gazegrounded.pipeline import task_human_map  # noqa: E402
from gazegrounded.visualization import dashboard_utils as du  # noqa: E402
from gazegrounded.visualization import style  # noqa: E402

data = sidebar_config()
ui.page_header(
    "Failure cases",
    "Tasks where the model's highest-ranked element is not the element the task needs. "
    "Compare it with where people actually looked, then record your interpretation.",
    data,
)
needs(data, "fixations", "vlm_elements")
ds, cfg = data.ds, data.cfg

p = report_path(data, "failure_cases.csv")
if p is None:
    st.info(
        "Run `python -m gazegrounded.cli evaluate` to find failure cases.", icon=":material/info:"
    )
    st.stop()
failures = pd.read_csv(p)
notes = ui.load_annotations(data)
annotated = set(notes.loc[notes["category"] != "", "task_id"])
provider = data.vlm_tasks["provider"].iloc[0] if data.vlm_tasks is not None else "unknown"

# ---- summary ----------------------------------------------------------------------------------
cols = st.columns(4)
cols[0].metric("Tasks analysed", len(ds.tasks), border=True)
cols[1].metric(
    "Failure cases",
    len(failures),
    border=True,
    help="Top-ranked VLM element does not match an annotated target AOI",
)
cols[2].metric(
    "Annotated", f"{len(annotated & set(failures['task_id']))} / {len(failures)}", border=True
)
cols[3].metric("VLM provider", provider, border=True)
if provider == "mock":
    st.caption(
        ":material/info: The **mock** provider is a lexical stand-in, so these cases "
        "illustrate the analysis, not real VLM behaviour. For real cases run "
        "`vlm-analysis --set vlm.provider=local`."
    )

if failures.empty:
    st.success(
        "No failure cases: the model's top element matched a target for every task.",
        icon=":material/check_circle:",
    )
    st.stop()

view = st.segmented_control(
    "Show", ["All cases", "Not yet annotated"], default="All cases", key="failure_filter"
)
if view == "Not yet annotated":
    failures = failures[~failures["task_id"].isin(annotated)]
    if failures.empty:
        st.success("All failure cases are annotated.", icon=":material/task_alt:")

tasks = ds.tasks.set_index("task_id")

for f in failures.itertuples():
    task = tasks.loc[f.task_id]
    targets = f.target_aois.split(";")
    aois = ds.aois_for(f.ui_id)
    target_rows = aois[aois["aoi_id"].isin(targets)]
    target_label = ", ".join(
        (r.label if isinstance(r.label, str) and r.label else ui.label(r.aoi_id))
        for r in target_rows.itertuples()
    )
    elements = (
        data.vlm_elements[data.vlm_elements["task_id"] == f.task_id]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    elements["reason"] = elements["reason"].fillna("")

    with st.container(border=True):
        st.markdown(f"### “{f.task}”")
        with st.container(horizontal=True, gap="small"):
            st.badge(ui.ui_name(f.ui_id), icon=":material/web:", color="gray")
            ui.condition_badge(ds.uis.set_index("ui_id").loc[f.ui_id, "condition"])
            designed = task.get("designed_friction_type")
            if isinstance(designed, str) and designed != "no_friction":
                st.badge(
                    f"designed: {ui.label(designed)}",
                    color="violet",
                    help="The friction manipulation built into this interface variant",
                )
            if f.task_id in annotated:
                st.badge("annotated", icon=":material/check_circle:", color="green")

        left, right = st.columns([3, 2], gap="large")
        with left:
            layers = st.segmented_control(
                "Layers",
                ["Gaze", "VLM", "Target"],
                default=["Gaze", "VLM", "Target"],
                selection_mode="multi",
                key=f"layers_{f.task_id}",
                label_visibility="collapsed",
            )
            image = Image.open(ds.image_path(f.ui_id)).convert("RGB")
            human, _ = task_human_map(cfg, ds, data.fixations, f.task_id)
            fig = du.failure_case_figure(
                image,
                human,
                elements.head(3),
                target_rows,
                {s.lower() for s in (layers or [])},
                height_px=360,
            )
            st.plotly_chart(fig, width="stretch", config=ui.PLOTLY_CONFIG, key=f"fig_{f.task_id}")
            st.caption(
                "Orange: where people looked (all participants) · numbered blue boxes: "
                "the model's top 3 elements · green **T**: annotated target. "
                "Hover for details."
            )

        with right:
            ui.section("What went wrong", ":material/troubleshoot:")
            top = elements.iloc[0] if len(elements) else None
            if top is not None:
                st.markdown(
                    f"The model ranked **{ui.short(top['label'])}** first "
                    f"(score {top['weight']:.2f}), but the task needs "
                    f"**{target_label}**."
                )
            st.markdown("**Where people actually looked** · share of dwell time")
            st.plotly_chart(
                du.share_bars(
                    [
                        ("Annotated target", f.human_dwell_share_on_target, du.TARGET_COLOR),
                        ("Model's top choice", f.human_dwell_share_on_vlm_top, style.VLM),
                    ]
                ),
                width="stretch",
                config={"displayModeBar": False},
                key=f"bars_{f.task_id}",
            )
            t_share = f.human_dwell_share_on_target
            v_share = f.human_dwell_share_on_vlm_top
            if pd.notna(v_share) and t_share >= v_share:
                st.caption(
                    ":material/lightbulb: People attended to the target more than to "
                    "the model's choice. This points to a model error rather than a "
                    "design problem."
                )
            elif pd.notna(v_share):
                st.caption(
                    ":material/lightbulb: People also looked at the model's choice. It "
                    "may be a genuine distractor in the design."
                )

            st.markdown("**Model ranking** · by score = relevance × attention priority")
            table = pd.DataFrame(
                {
                    "Element": [
                        f"{i}. {ui.short(x, 26)}" for i, x in enumerate(elements["label"], start=1)
                    ],
                    "Score": elements["weight"],
                    "Target": elements["matched_aoi_id"].isin(targets),
                }
            ).head(5)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Element": st.column_config.TextColumn(width="medium"),
                    "Score": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="%.2f",
                        width="small",
                        help="Task relevance × expected attention priority, as estimated by "
                        "the model. Elements are ranked by this score.",
                    ),
                    "Target": st.column_config.CheckboxColumn(width=60),
                },
            )

        prev = notes[notes["task_id"] == f.task_id]
        prev_cat = prev["category"].iloc[0] if len(prev) else ""
        prev_note = prev["note"].iloc[0] if len(prev) else ""
        with (
            st.expander(
                "Your interpretation",
                icon=":material/edit_note:",
                expanded=f.task_id not in annotated,
            ),
            st.form(key=f"form_{f.task_id}", border=False),
        ):
            category = st.pills(
                "Failure category",
                ui.FAILURE_CATEGORIES,
                selection_mode="single",
                default=prev_cat if prev_cat in ui.FAILURE_CATEGORIES else None,
                key=f"cat_{f.task_id}",
            )
            note = st.text_area(
                "Notes",
                value=prev_note,
                key=f"note_{f.task_id}",
                height=80,
                placeholder="What did the model miss, and what does the gaze suggest?",
            )
            if st.form_submit_button("Save", icon=":material/save:", type="primary"):
                ui.save_annotation(data, f.task_id, category or "", note)
                st.toast("Interpretation saved", icon=":material/check_circle:")
                st.rerun()

st.caption(
    f":material/folder: Interpretations are saved to "
    f"`{ui.annotations_path(data).relative_to(cfg.root())}`."
)
