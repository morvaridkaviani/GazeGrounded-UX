"""RQ3/RQ4: which signals predict friction, and why."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))

import ui  # noqa: E402
from common import sidebar_config  # noqa: E402

from gazegrounded.visualization import dashboard_utils as du  # noqa: E402

data = sidebar_config()
ui.page_header(
    "Predicting UX friction",
    "Do gaze, interaction, model relevance and written feedback predict the friction level, "
    "together better than alone? Every score is from grouped cross-validation.",
    data,
)
run = data.friction_run
if run is None:
    st.info("Run `python -m gazegrounded.cli train-friction` first.", icon=":material/info:")
    st.stop()

metrics = json.loads((run / "metrics.json").read_text())
results = pd.read_csv(run / "modality_comparison.csv")

c = st.columns(4)
c[0].metric("Trials", metrics["n_trials"], border=True)
c[1].metric("Participants", metrics["n_participants"], border=True)
c[2].metric("Interfaces", metrics["n_interfaces"], border=True)
best = results[(results["group_by"] == "participant_id") & (results["model"] != "majority")]
top = best.sort_values("macro_f1", ascending=False).iloc[0]
c[3].metric(
    "Best macro-F1",
    f"{top['macro_f1']:.2f}",
    border=True,
    help=f"{ui.label(top['modality'])} · {ui.label(top['model'])}, new participants",
)
with st.container(horizontal=True, gap="small"):
    st.badge(f"run {run.name}", icon=":material/folder:", color="gray")
    for src in metrics["label_source"]:
        st.badge(f"labels: {ui.label(src)}", icon=":material/label:", color="gray")

with st.container(border=True):
    ui.section("Performance by modality", ":material/bar_chart:")
    split = (
        st.segmented_control(
            "Generalise to",
            ["participant_id", "ui_id"],
            default="participant_id",
            key="split",
            format_func=lambda g: "New participants" if g == "participant_id" else "New interfaces",
        )
        or "participant_id"
    )
    st.plotly_chart(
        du.modality_figure(results[results["group_by"] == split]),
        width="stretch",
        config={"displayModeBar": False},
        key="fm_modality",
    )
    st.caption(
        "No participant (or interface) appears in both training and test folds. The "
        "dashed line is the majority-class reference."
    )
    with st.expander("All metrics", icon=":material/table:"):
        r = results[results["group_by"] == split]
        st.dataframe(
            r[
                [
                    "model",
                    "modality",
                    "macro_f1",
                    "weighted_f1",
                    "accuracy",
                    "quadratic_weighted_kappa",
                    "roc_auc_ovr_macro",
                    "macro_f1_fold_sd",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                k: st.column_config.NumberColumn(format="%.2f")
                for k in (
                    "macro_f1",
                    "weighted_f1",
                    "accuracy",
                    "quadratic_weighted_kappa",
                    "roc_auc_ovr_macro",
                    "macro_f1_fold_sd",
                )
            },
        )

left, right = st.columns(2, gap="medium")
with left, st.container(border=True, height="stretch"):
    ui.section(
        "What each modality adds (RQ4)",
        ":material/layers_clear:",
        "Macro-F1 of the multimodal model with one modality removed. A larger drop "
        "means that modality carried information the others did not.",
    )
    abl = pd.read_csv(run / "ablation.csv")
    full = results[
        (results["group_by"] == "participant_id")
        & (results["model"] == "logistic_regression")
        & (results["modality"] == "multimodal")
    ]
    if len(full):
        abl["change"] = abl["macro_f1"] - float(full["macro_f1"].iloc[0])
    st.dataframe(
        abl[["dropped_modality", "macro_f1", "change"]].rename(
            columns={
                "dropped_modality": "Without",
                "macro_f1": "Macro-F1",
                "change": "Change vs all",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.2f"),
            "Change vs all": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
with right, st.container(border=True, height="stretch"):
    ui.section(
        "Confusion matrix",
        ":material/grid_on:",
        "Multimodal logistic regression, out-of-fold, new participants",
    )
    cm = run / "confusion_matrix_multimodal.png"
    if cm.exists():
        st.image(str(cm), width="stretch")

with st.container(border=True):
    ui.section(
        "What drives the predictions?",
        ":material/insights:",
        "Left: drop in macro-F1 on held-out folds when a signal is shuffled. Right: "
        "standardised logistic-regression weights for 'high' friction.",
    )
    c1, c2 = st.columns(2)
    for col, name in (
        (c1, "feature_importance_permutation.png"),
        (c2, "coefficients_high_friction.png"),
    ):
        if (run / name).exists():
            col.image(str(run / name), width="stretch")
