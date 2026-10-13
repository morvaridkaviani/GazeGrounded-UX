"""Command-line interface: ``python -m gazegrounded.cli <command> [--config ...] [--set k=v]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from dotenv import load_dotenv

from gazegrounded import pipeline
from gazegrounded.config import load_config
from gazegrounded.utils import SYNTHETIC_BANNER, get_logger, set_seed

log = get_logger("gazegrounded.cli")

COMMANDS: dict[str, tuple[str, Callable]] = {
    "generate-demo": (
        "Generate the synthetic demo dataset (zero cost)",
        pipeline.stage_generate_demo,
    ),
    "preprocess": (
        "Run the dataset adapter, validate and save standardised tables",
        pipeline.stage_preprocess,
    ),
    "gaze-analysis": (
        "Fixations, AOIs, gaze metrics, scanpaths and heatmaps",
        pipeline.stage_gaze_analysis,
    ),
    "vlm-analysis": (
        "Task-conditioned VLM grounding, relevance maps and human-vs-VLM alignment",
        pipeline.stage_vlm_analysis,
    ),
    "train-saliency": (
        "Train the ResNet18 saliency model on screenshot -> gaze heatmap pairs",
        pipeline.stage_train_saliency,
    ),
    "nlp-analysis": (
        "Feedback text: TF-IDF + logistic regression, top terms, exploratory clusters",
        pipeline.stage_nlp_analysis,
    ),
    "build-features": (
        "Build the trial-level multimodal feature table",
        pipeline.stage_build_features,
    ),
    "train-friction": (
        "Build features and compare friction models per modality with grouped CV",
        pipeline.stage_train_friction,
    ),
    "evaluate": (
        "RQ1 baselines, RQ2 statistics, RQ3/RQ4 tables, failure cases and the report",
        pipeline.stage_evaluate,
    ),
    "build-study": (
        "Render the WebGazer study stimuli and tasks.json from the UI specs",
        pipeline.stage_build_study,
    ),
    "run-all": (
        "Run every stage in order (synthetic config: generates the demo data first)",
        pipeline.stage_run_all,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gazegrounded",
        description="GazeGrounded UX: task-aware multimodal detection of UX friction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", default="configs/synthetic.yaml", help="YAML config file")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a config value, e.g. seed=7",
        )
        if name in ("train-saliency", "run-all"):
            p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
            p.add_argument(
                "--fast-dev-run",
                action="store_true",
                help="Tiny images, 2 epochs: checks the training code only",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    overrides = list(args.overrides)
    if getattr(args, "device", None):
        overrides.append(f"saliency.device={args.device}")
    if getattr(args, "fast_dev_run", False):
        overrides += [
            "saliency.fast_dev_run=true",
            "saliency.epochs=2",
            "saliency.image_width=128",
            "saliency.image_height=80",
            "saliency.pretrained=false",
        ]
    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    if cfg.is_synthetic:
        log.warning(SYNTHETIC_BANNER)
    COMMANDS[args.command][1](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
