"""Typed configuration loaded from YAML.

Config files may inherit from another file with ``extends: base.yaml``; values in the child
override the parent (deep merge). Every tunable number used by the pipeline lives here so that
it is recorded with each experiment run instead of being hidden in scripts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    source: str = "synthetic"
    adapter: Literal["synthetic", "ueyes", "webgazer_study"] = "synthetic"
    raw_dir: str = "data/synthetic"
    screen_width: int = 1440
    screen_height: int = 900
    # "synthetic_demo" or "real". Controls the banner shown on every report and figure.
    data_kind: Literal["synthetic_demo", "real"] = "synthetic_demo"


class PathsConfig(_Section):
    data_root: str = "data"
    experiments: str = "experiments"
    figures: str = "assets/figures"
    reports: str = "reports/generated"


class SyntheticConfig(_Section):
    n_participants: int = 12
    sampling_rate_hz: int = 60
    # Accuracy: systematic offset per fixation. Precision: sample-to-sample jitter.
    gaze_noise_px: float = 18.0
    gaze_precision_px: float = 6.0
    invalid_sample_rate: float = 0.03
    missing_feedback_rate: float = 0.08


class GazeConfig(_Section):
    fixation_algorithm: Literal["idt"] = "idt"
    dispersion_threshold_px: float = 80.0
    minimum_duration_ms: float = 100.0
    max_gap_ms: float = 75.0
    # ~1 degree of visual angle at ~60 cm on a typical laptop display is roughly 35-45 px.
    heatmap_sigma_px: float = 40.0
    # AOIs are padded by roughly the tracker's accuracy when assigning fixations.
    aoi_margin_px: float = 20.0
    # Maps are compared on a downsampled grid (map_scale * screen size) for speed.
    map_scale: float = 0.125
    min_valid_sample_ratio: float = 0.6


class VLMConfig(_Section):
    provider: Literal["mock", "local", "http"] = "mock"
    model: str | None = None
    cache: bool = True
    prompt_version: str = "v1"
    max_new_tokens: int = 1024
    temperature: float = 0.0
    device: str = "auto"
    # Mock provider only: how imperfect the stand-in "VLM" should be (bbox jitter, relevance noise).
    mock_bbox_jitter: float = 0.01
    mock_relevance_noise: float = 0.05


class ComparisonConfig(_Section):
    eps: float = 1e-12
    relevance_threshold: float = 0.5
    # Soft edge of each VLM bounding box, as a fraction of the box's shorter side.
    vlm_box_softness: float = 0.25
    # Early-window divergence: first K fixations after dropping the onset fixation. Whole-trial
    # divergence is confounded by fixation count (sparse maps look divergent), see docs.
    early_fixations: int = 8
    drop_first_fixation: bool = True


class SaliencyConfig(_Section):
    image_width: int = 256
    image_height: int = 160
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 5
    loss_alpha_mse: float = 1.0
    loss_beta_kl: float = 1.0
    pretrained: bool = True
    freeze_encoder: bool = False
    device: str = "auto"
    val_fraction: float = 0.2
    num_workers: int = 0
    fast_dev_run: bool = False


class NLPConfig(_Section):
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 1
    max_features: int = 2000
    C: float = 1.0
    n_clusters: int = 6
    n_splits: int = 5
    svd_components: int = 8


class CVConfig(_Section):
    strategy: Literal["group_kfold"] = "group_kfold"
    group_by: str = "participant_id"
    n_splits: int = 5


class FrictionModelConfig(_Section):
    type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest"])
    modalities: list[str] = Field(
        default_factory=lambda: ["interaction", "gaze", "vlm", "text", "multimodal"]
    )
    cv: CVConfig = Field(default_factory=CVConfig)
    # If true, also evaluate generalisation to unseen interfaces (GroupKFold by ui_id).
    also_group_by_ui: bool = True
    logreg_C: float = 1.0
    rf_n_estimators: int = 300
    rf_max_depth: int | None = 6
    permutation_repeats: int = 10
    # Labels: "human" (preferred), "synthetic_latent" (demo generator) or "heuristic_demo".
    label_source: Literal["human", "synthetic_latent", "heuristic_demo"] = "synthetic_latent"


class StatsConfig(_Section):
    paired_test: Literal["wilcoxon", "ttest"] = "wilcoxon"
    alpha: float = 0.05
    correction: Literal["holm", "none"] = "holm"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = "gazegrounded-ux"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    gaze: GazeConfig = Field(default_factory=GazeConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    saliency: SaliencyConfig = Field(default_factory=SaliencyConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    friction_model: FrictionModelConfig = Field(default_factory=FrictionModelConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)

    # ---- derived paths -------------------------------------------------------------------
    def root(self) -> Path:
        from gazegrounded.utils import project_root

        return project_root()

    def processed_dir(self) -> Path:
        return self.root() / self.paths.data_root / "processed" / self.data.source

    def interim_dir(self) -> Path:
        return self.root() / self.paths.data_root / "interim" / self.data.source

    def raw_dir(self) -> Path:
        return self.root() / self.data.raw_dir

    def experiments_dir(self) -> Path:
        return self.root() / self.paths.experiments

    def figures_dir(self) -> Path:
        return self.root() / self.paths.figures / self.data.source

    def reports_dir(self) -> Path:
        return self.root() / self.paths.reports

    @property
    def is_synthetic(self) -> bool:
        return self.data.data_kind == "synthetic_demo"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml_with_extends(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_data = _read_yaml_with_extends((path.parent / parent).resolve())
        raw = _deep_merge(parent_data, raw)
    return raw


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, value = item.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (with ``extends`` inheritance) and apply ``a.b=value`` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            from gazegrounded.utils import project_root

            p = project_root() / p
        data = _read_yaml_with_extends(p)
    for item in overrides or []:
        keys, value = _parse_override(item)
        cursor = data
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = value
    return Config.model_validate(data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return cfg.model_dump(mode="json")
