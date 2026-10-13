"""Small shared helpers: logging, seeding, paths, hashing, experiment run directories."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SYNTHETIC_BANNER = "SYNTHETIC DEMO DATA — NOT RESEARCH EVIDENCE"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "gazegrounded") -> logging.Logger:
    """Return a module logger; configures a single stream handler on first use."""
    root = logging.getLogger("gazegrounded")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("GAZEGROUNDED_LOG_LEVEL", "INFO"))
        root.propagate = False
    return logging.getLogger(name)


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy and (if installed) torch. Returns a numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def project_root() -> Path:
    """Walk up from the CWD to the directory containing ``pyproject.toml``.

    Falls back to the repository that contains this file, so imports from notebooks work too.
    """
    env = os.environ.get("GAZEGROUNDED_ROOT")
    if env:
        return Path(env).resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "gazegrounded"
        ).exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_sha256(path: str | Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")  # unit separator so ("ab","c") != ("a","bc")
    return h.hexdigest()


def stable_int_hash(*parts: str) -> int:
    """Deterministic (process-independent) integer hash, useful for per-item seeds."""
    return int(text_sha256(*parts)[:12], 16)


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, cls=_NumpyJSONEncoder, allow_nan=False, default=str))
    return p


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json_safe(obj: Any, path: str | Path) -> Path:
    """Like :func:`save_json` but converts NaN to ``null`` instead of failing."""
    return save_json(_nan_to_none(json.loads(json.dumps(obj, cls=_NumpyJSONEncoder))), path)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save_yaml(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(yaml.safe_dump(obj, sort_keys=False))
    return p


def make_run_dir(experiments_dir: str | Path, name: str) -> Path:
    """Create ``experiments/YYYYMMDD-HHMMSS-<name>/`` (suffixing if it already exists)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(experiments_dir) / f"{stamp}-{name}"
    run_dir = base
    i = 1
    while run_dir.exists():
        run_dir = Path(f"{base}-{i}")
        i += 1
    return ensure_dir(run_dir)


def latest_run_dir(experiments_dir: str | Path, name: str) -> Path | None:
    """Most recent run directory whose name ends with ``-<name>`` (or ``-<name>-N``)."""
    d = Path(experiments_dir)
    if not d.exists():
        return None
    pattern = re.compile(rf"\d{{8}}-\d{{6}}-{re.escape(name)}(-\d+)?")
    runs = sorted(p for p in d.iterdir() if p.is_dir() and pattern.fullmatch(p.name))
    return runs[-1] if runs else None


def resolve_device(requested: str = "auto") -> str:
    """Map ``auto|cpu|mps|cuda`` to an available torch device string."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return requested
