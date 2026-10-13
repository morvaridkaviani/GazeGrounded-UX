"""How to read the dashboard, and what it can and cannot tell you."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))

import ui  # noqa: E402

ui.page_header(
    "Methodology and limits",
    "How the numbers are produced, how to read them, and what they cannot tell you.",
)

with st.container(border=True):
    ui.section("Pipeline", ":material/account_tree:")
    st.markdown("""
Gaze samples → I-DT fixations → AOIs, time to first fixation, dwell, revisits, scanpaths →
duration-weighted heatmaps. In parallel, a vision-language model receives the screenshot **and
the task** and returns task-relevant elements with boxes and scores, which become a
**model relevance map**. Comparing the two gives alignment and divergence features. These are
combined with interaction logs and feedback text to predict friction.
""")

left, right = st.columns(2, gap="medium")
with left, st.container(border=True, height="stretch"):
    ui.section("Key definitions", ":material/menu_book:")
    st.markdown("""
- **First look at target (TTFF)**: time from stimulus onset to the first fixation on a
  task-relevant AOI.
- **Revisit**: returning to an AOI after looking elsewhere.
- **Early divergence**: Jensen–Shannon divergence (0–1) between the first fixations and the
  model map. It is the primary measure because whole-trial divergence is confounded by the
  number of fixations.
- **Macro-F1**: F1 averaged over friction levels, so rare levels count equally.
""")
with right, st.container(border=True, height="stretch"):
    ui.section("How it is evaluated", ":material/rule:")
    st.markdown("""
- Grouped cross-validation by participant and by interface, never random row splits.
- Majority-class reference, macro-F1 and quadratic weighted kappa.
- Trials are nested in participants: repeated-measures correlation, paired tests on
  participant means, Holm correction.
- Every learned step (including the text classifier) is refit inside each training fold.
""")

with st.container(border=True):
    ui.section("What this is not", ":material/report:")
    st.markdown("""
- The model map is an *estimate of task-relevant regions*, not a measurement of attention or
  cognition.
- Gaze is a proxy for overt visual attention. Looking is not understanding.
- Webcam gaze (the study prototype) is far less precise than lab eye trackers.
- Synthetic demo numbers only show that the software works.
""")
    st.caption(
        "Details: `reports/limitations.md`, `reports/model_card.md`, `reports/dataset_card.md`."
    )
