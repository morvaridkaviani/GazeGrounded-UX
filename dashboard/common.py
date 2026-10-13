"""Shared Streamlit helpers: config picker, cached data loading, missing-output hints."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import ui

from gazegrounded.config import load_config
from gazegrounded.utils import project_root
from gazegrounded.visualization.dashboard_utils import DashboardData, load_dashboard_data

ROOT = project_root()
CONFIGS = sorted(p.name for p in (ROOT / "configs").glob("*.yaml") if p.name != "base.yaml")


@st.cache_resource(show_spinner="Loading pipeline outputs…")
def get_data(config_name: str) -> DashboardData:
    return load_dashboard_data(load_config(ROOT / "configs" / config_name))


def sidebar_config() -> DashboardData:
    with st.sidebar:
        st.markdown("**Data source**")
        default = CONFIGS.index("synthetic.yaml") if "synthetic.yaml" in CONFIGS else 0
        name = st.selectbox(
            "Configuration",
            CONFIGS,
            index=default,
            key="config_name",
            label_visibility="collapsed",
            help="Which pipeline configuration (and data source) to explore",
        )
        data = get_data(name)
        kind = "Synthetic demo" if data.cfg.is_synthetic else "Real data"
        st.caption(f":material/database: {kind} · source `{data.cfg.data.source}`")
        if st.button(
            "Reload outputs",
            icon=":material/refresh:",
            width="stretch",
            help="Clear the cache after re-running the pipeline",
        ):
            st.cache_resource.clear()
            st.rerun()
        st.link_button(
            "Source code on GitHub", ui.REPO_URL, icon=":material/code:", width="stretch"
        )
        st.divider()
    if data.missing:
        stages = ", ".join(sorted(set(data.missing)))
        st.info(
            f"Some pipeline outputs are missing ({stages}). Run "
            f"`python -m gazegrounded.cli run-all --config configs/{name}`, then reload.",
            icon=":material/info:",
        )
    return data


def needs(data: DashboardData, *attrs: str) -> bool:
    """Stop the page if required outputs are missing (a hint is already shown)."""
    if data.ds is None or any(
        getattr(data, a) is None or (isinstance(getattr(data, a), dict) and not getattr(data, a))
        for a in attrs
    ):
        st.stop()
    return True


def report_path(data: DashboardData, name: str) -> Path | None:
    p = data.report_dir / name if data.report_dir else None
    return p if p is not None and p.exists() else None
