"""End-to-end smoke test: the documented CLI runs the whole synthetic pipeline."""

import json

import pandas as pd
import pytest

from gazegrounded.cli import COMMANDS, build_parser, main


@pytest.mark.slow
def test_run_all_cli_end_to_end(tmp_path):
    overrides = []
    for key, value in {
        "data.raw_dir": tmp_path / "synthetic",
        "paths.data_root": tmp_path / "data",
        "paths.experiments": tmp_path / "experiments",
        "paths.figures": tmp_path / "figures",
        "paths.reports": tmp_path / "reports",
    }.items():
        overrides += ["--set", f"{key}={value}"]
    assert main(["run-all", "--config", "configs/fast_dev.yaml", *overrides]) == 0

    processed = tmp_path / "data" / "processed" / "synthetic"
    interim = tmp_path / "data" / "interim" / "synthetic"
    assert json.loads((processed / "metadata.json").read_text())["data_kind"] == "synthetic_demo"
    assert len(pd.read_csv(processed / "features.csv")) == 60
    for f in (
        "fixations.csv",
        "gaze_metrics.csv",
        "aoi_metrics.csv",
        "scanpaths.csv",
        "nlp_predictions.csv",
        "vlm/vlm_alignment.csv",
        "vlm/vlm_maps.npz",
    ):
        assert (interim / f).exists(), f
    runs = {p.name.split("-", 2)[2] for p in (tmp_path / "experiments").iterdir()}
    assert {"saliency", "nlp-feedback", "friction"} <= runs
    for run in (tmp_path / "experiments").iterdir():
        assert (run / "config.yaml").exists() and (run / "metrics.json").exists()
    friction = next((tmp_path / "experiments").glob("*-friction"))
    for f in ("feature_importance.csv", "predictions.csv", "modality_comparison.csv"):
        assert (friction / f).exists(), f
    report = (tmp_path / "reports" / "synthetic" / "report.md").read_text()
    assert "SYNTHETIC DEMO DATA" in report and "## RQ1" in report and "## RQ3" in report
    assert any((tmp_path / "figures" / "synthetic" / "evaluation").glob("*.png"))


def test_cli_lists_every_documented_command():
    expected = {
        "generate-demo",
        "preprocess",
        "gaze-analysis",
        "vlm-analysis",
        "train-saliency",
        "nlp-analysis",
        "build-features",
        "train-friction",
        "evaluate",
        "build-study",
        "export-site",
        "run-all",
    }
    assert expected == set(COMMANDS)
    parser = build_parser()
    args = parser.parse_args(["train-saliency", "--device", "cpu", "--fast-dev-run"])
    assert args.device == "cpu" and args.fast_dev_run
