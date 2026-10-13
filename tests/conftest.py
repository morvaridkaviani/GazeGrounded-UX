"""Shared fixtures. Tests write only to pytest temp dirs, never to data/ or experiments/."""

from __future__ import annotations

import pytest

from gazegrounded.config import load_config
from gazegrounded.data.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="session")
def fast_cfg():
    return load_config("configs/fast_dev.yaml")


@pytest.fixture(scope="session")
def synthetic_ds(tmp_path_factory, fast_cfg):
    out = tmp_path_factory.mktemp("synthetic")
    return generate_synthetic_dataset(fast_cfg, out_dir=out)


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory):
    """fast_dev config whose every output path points into a temp directory."""
    root = tmp_path_factory.mktemp("pipeline")
    return load_config(
        "configs/fast_dev.yaml",
        [
            f"data.raw_dir={root / 'synthetic'}",
            f"paths.data_root={root / 'data'}",
            f"paths.experiments={root / 'experiments'}",
            f"paths.figures={root / 'figures'}",
            f"paths.reports={root / 'reports'}",
        ],
    )


@pytest.fixture(scope="session")
def pipeline_features(pipeline_cfg):
    """Run generate -> preprocess -> gaze -> vlm -> features (figures off) once per session."""
    from gazegrounded import pipeline

    pipeline.stage_generate_demo(pipeline_cfg)
    pipeline.stage_preprocess(pipeline_cfg)
    pipeline.stage_gaze_analysis(pipeline_cfg, make_figures=False)
    pipeline.stage_vlm_analysis(pipeline_cfg, make_figures=False)
    return pipeline.stage_build_features(pipeline_cfg)
