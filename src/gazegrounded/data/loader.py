"""Standardised dataset container and the dataset-adapter interface.

Adapters translate a source-specific layout (synthetic generator, UEyes, WebGazer study exports)
into a :class:`StandardizedDataset`. Nothing downstream of an adapter should know which source
the data came from.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from gazegrounded.data.validation import ValidationReport, validate_dataset
from gazegrounded.utils import SYNTHETIC_BANNER, ensure_dir, project_root

_TABLE_FILES = {
    "uis": "uis.csv",
    "aois": "aois.csv",
    "tasks": "tasks.csv",
    "trials": "trials.csv",
    "gaze_samples": "gaze_samples.csv.gz",
    "fixations": "fixations.csv",
}
_STRING_COLS = {
    "participant_id",
    "session_id",
    "trial_id",
    "ui_id",
    "task_id",
    "aoi_id",
    "task_key",
    "pair_id",
    "target_aoi_ids",
}


@dataclass
class StandardizedDataset:
    """All tables of one data source in the project's standard schema."""

    source: str
    data_kind: str  # "synthetic_demo" | "real"
    root: Path  # directory that ``uis.image_path`` is relative to
    uis: pd.DataFrame
    aois: pd.DataFrame
    tasks: pd.DataFrame
    trials: pd.DataFrame
    gaze_samples: pd.DataFrame | None = None
    fixations: pd.DataFrame | None = None  # only for sources that ship fixations, not samples

    @property
    def is_synthetic(self) -> bool:
        return self.data_kind == "synthetic_demo"

    def image_path(self, ui_id: str) -> Path:
        rel = self.uis.set_index("ui_id").loc[ui_id, "image_path"]
        return self.root / rel

    def ui_size(self, ui_id: str) -> tuple[int, int]:
        row = self.uis.set_index("ui_id").loc[ui_id]
        return int(row["width"]), int(row["height"])

    def aois_for(self, ui_id: str) -> pd.DataFrame:
        return self.aois[self.aois["ui_id"] == ui_id].reset_index(drop=True)

    def targets_for_task(self, task_id: str) -> list[str]:
        raw = self.tasks.set_index("task_id").loc[task_id, "target_aoi_ids"]
        return [t for t in str(raw).split(";") if t and t != "nan"]

    def validate(self) -> ValidationReport:
        return validate_dataset(self)

    # ---- persistence ---------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        out = ensure_dir(directory)
        for name, filename in _TABLE_FILES.items():
            df = getattr(self, name)
            if df is not None:
                df.to_csv(out / filename, index=False)
        try:
            image_root = str(self.root.resolve().relative_to(project_root()))
        except ValueError:
            image_root = str(self.root.resolve())
        meta = {
            "source": self.source,
            "data_kind": self.data_kind,
            "image_root": image_root,
            "created": datetime.now().isoformat(timespec="seconds"),
            "counts": {
                n: (len(getattr(self, n)) if getattr(self, n) is not None else 0)
                for n in _TABLE_FILES
            },
        }
        if self.is_synthetic:
            meta["notice"] = SYNTHETIC_BANNER
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        return out

    @classmethod
    def load(cls, directory: str | Path) -> StandardizedDataset:
        d = Path(directory)
        if not (d / "metadata.json").exists():
            raise FileNotFoundError(
                f"No standardised dataset at {d}. Run `python -m gazegrounded.cli preprocess` first."
            )
        meta = json.loads((d / "metadata.json").read_text())
        tables: dict[str, pd.DataFrame | None] = {}
        for name, filename in _TABLE_FILES.items():
            p = d / filename
            tables[name] = read_table(p) if p.exists() else None
        root = Path(meta["image_root"])
        if not root.is_absolute():
            root = project_root() / root
        return cls(source=meta["source"], data_kind=meta["data_kind"], root=root, **tables)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a standard CSV, keeping identifier columns as strings and NaN ids as None."""
    df = pd.read_csv(path, dtype={c: "string" for c in _STRING_COLS}, low_memory=False)
    for col in df.columns:
        if col in _STRING_COLS:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


class EyeTrackingDatasetAdapter(abc.ABC):
    """Interface every data source implements.

    ``load_raw`` reads source files without changing them, ``standardize`` maps them to the
    standard schema, and ``validate`` runs schema + integrity checks.
    """

    source: str = "unknown"
    data_kind: str = "real"

    def __init__(self, raw_dir: str | Path, screen_width: int = 1440, screen_height: int = 900):
        self.raw_dir = Path(raw_dir)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._raw: dict | None = None

    @abc.abstractmethod
    def load_raw(self) -> dict:
        """Read raw files into memory (dict of DataFrames / objects)."""

    @abc.abstractmethod
    def standardize(self) -> StandardizedDataset:
        """Convert the raw data to a :class:`StandardizedDataset`."""

    def validate(self, ds: StandardizedDataset | None = None) -> ValidationReport:
        return (ds or self.standardize()).validate()

    def run(self) -> tuple[StandardizedDataset, ValidationReport]:
        self.load_raw()
        ds = self.standardize()
        return ds, ds.validate()


class SyntheticAdapter(EyeTrackingDatasetAdapter):
    """The synthetic generator already writes standard tables; this adapter just reads them."""

    source = "synthetic"
    data_kind = "synthetic_demo"

    def load_raw(self) -> dict:
        if not (self.raw_dir / "trials.csv").exists():
            raise FileNotFoundError(
                f"No synthetic data in {self.raw_dir}. Run `python -m gazegrounded.cli "
                "generate-demo` first."
            )
        self._raw = {
            name: read_table(self.raw_dir / filename)
            for name, filename in _TABLE_FILES.items()
            if (self.raw_dir / filename).exists()
        }
        return self._raw

    def standardize(self) -> StandardizedDataset:
        raw = self._raw or self.load_raw()
        return StandardizedDataset(
            source=self.source,
            data_kind=self.data_kind,
            root=self.raw_dir,
            uis=raw["uis"],
            aois=raw["aois"],
            tasks=raw["tasks"],
            trials=raw["trials"],
            gaze_samples=raw.get("gaze_samples"),
            fixations=raw.get("fixations"),
        )


def get_adapter(name: str, raw_dir: str | Path, screen_width: int, screen_height: int):
    """Factory for the adapters named in ``configs/*.yaml`` (``data.adapter``)."""
    names = ("synthetic", "ueyes", "webgazer_study")
    if name not in names:
        raise KeyError(f"Unknown adapter {name!r}; choose from {names}")
    if name == "synthetic":
        cls: type[EyeTrackingDatasetAdapter] = SyntheticAdapter
    else:
        from gazegrounded.data import adapters  # local import: avoids a cycle

        cls = adapters.UEyesAdapter if name == "ueyes" else adapters.WebGazerStudyAdapter
    return cls(raw_dir, screen_width=screen_width, screen_height=screen_height)


def load_processed(cfg) -> StandardizedDataset:
    """Load the standardised dataset for ``cfg.data.source`` from ``data/processed/``."""
    return StandardizedDataset.load(cfg.processed_dir())
