"""Shared UI building blocks so every page looks and behaves the same."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from gazegrounded.data.ui_specs import all_specs
from gazegrounded.utils import SYNTHETIC_BANNER

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
}

CONDITION_COLORS = {
    "clean": "green",
    "friction": "orange",
    "single": "gray",
    "free_viewing": "blue",
}
FRICTION_COLORS = {"low": "green", "medium": "orange", "high": "red"}
FRICTION_ICONS = {
    "low": ":material/sentiment_satisfied:",
    "medium": ":material/sentiment_neutral:",
    "high": ":material/sentiment_dissatisfied:",
}

FAILURE_CATEGORIES = [
    "Icon meaning misunderstood",
    "Text too small / low contrast",
    "Several similar elements",
    "Depends on hidden content",
    "Requires scrolling",
    "Semantically right, visually ignored",
    "Plausible alternative (target too narrow)",
    "Other",
]

_SPECS = all_specs()


def ui_name(ui_id: str) -> str:
    """Human-readable interface name, e.g. 'E-commerce · Checkout'."""
    spec = _SPECS.get(ui_id)
    if spec is None:
        return ui_id.replace("_", " ").title()
    title = spec.title.replace(" (clean)", "").replace(" (friction)", "")
    return f"{spec.domain.capitalize()} · {title}"


def label(text: str) -> str:
    return str(text).replace("_", " ")


def page_header(title: str, subtitle: str, data=None, icon: str | None = None) -> None:
    st.title(f"{icon} {title}" if icon else title)
    st.caption(subtitle)
    if data is not None and data.cfg.is_synthetic:
        st.badge(
            SYNTHETIC_BANNER.capitalize().replace("—", "·"),
            icon=":material/science:",
            color="orange",
            help="Every number on this page comes from simulated data whose patterns were "
            "built into the generator. It demonstrates the pipeline, not findings.",
        )


def condition_badge(condition: str) -> None:
    st.badge(f"{label(condition)} variant", color=CONDITION_COLORS.get(condition, "gray"))


def friction_badge(level: str | None, prefix: str = "") -> None:
    if not isinstance(level, str):
        st.badge(f"{prefix}n/a", color="gray")
        return
    st.badge(
        f"{prefix}{level}", icon=FRICTION_ICONS.get(level), color=FRICTION_COLORS.get(level, "gray")
    )


def section(title: str, icon: str, caption: str | None = None) -> None:
    st.markdown(f"#### {icon} {title}")
    if caption:
        st.caption(caption)


# ---- failure-case annotations (persisted next to the generated report) ----------------------


def annotations_path(data) -> Path:
    return data.report_dir / "failure_annotations.csv"


def load_annotations(data) -> pd.DataFrame:
    p = annotations_path(data)
    if p.exists():
        return pd.read_csv(p, dtype=str).fillna("")
    return pd.DataFrame(columns=["task_id", "category", "note", "updated"])


def save_annotation(data, task_id: str, category: str, note: str) -> None:
    df = load_annotations(data)
    df = df[df["task_id"] != task_id]
    row = {
        "task_id": task_id,
        "category": category or "",
        "note": note or "",
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    annotations_path(data).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(annotations_path(data), index=False)


# ---- footer -----------------------------------------------------------------------------------

AUTHOR = "Morvarid Kaviani"
AUTHOR_URL = "https://www.morvaridkaviani.com"
AUTHOR_SITE = "www.morvaridkaviani.com"
REPO_URL = "https://github.com/morvaridkaviani/GazeGrounded-UX"


PROJECT_NAME = "GazeGrounded UX"


def footer() -> None:
    """Copyright and authorship line, centred at the bottom of every page."""
    st.divider()
    st.caption(
        f"© {datetime.now().year} {PROJECT_NAME} · All rights reserved · "
        f"Designed and developed by **{AUTHOR}** · [{AUTHOR_SITE}]({AUTHOR_URL})",
        text_alignment="center",
        width="stretch",
    )


def short(text, n: int = 45) -> str:
    """Truncate long element labels for prose and tables."""
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"
