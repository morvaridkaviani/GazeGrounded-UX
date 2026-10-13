"""GazeGrounded UX dashboard.

Run: ``streamlit run dashboard/app.py`` (after ``python -m gazegrounded.cli run-all``).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import ui  # noqa: E402

st.set_page_config(page_title="GazeGrounded UX", page_icon=":material/visibility:", layout="wide")

nav = st.navigation(
    [
        st.Page(
            "explorer.py", title="Trial explorer", icon=":material/travel_explore:", default=True
        ),
        st.Page("pages/1_Human_vs_VLM.py", title="Human vs VLM", icon=":material/compare:"),
        st.Page("pages/2_Friction_models.py", title="Friction models", icon=":material/bar_chart:"),
        st.Page("pages/3_Failure_cases.py", title="Failure cases", icon=":material/troubleshoot:"),
        st.Page(
            "pages/4_Methodology.py", title="Methodology & limits", icon=":material/menu_book:"
        ),
    ]
)
try:
    nav.run()
finally:  # also shown when a page stops early (e.g. missing pipeline outputs)
    ui.footer()
