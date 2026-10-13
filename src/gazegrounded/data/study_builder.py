"""Build the WebGazer study materials (stimuli + ``tasks.json``) from the UI specs.

Counterbalancing: every participant sees each *paired* interface once - the clean variant in
one group and the friction variant in the other - so nobody performs the same task twice
(no learning effect), yet every participant contributes clean and friction trials for paired
within-participant comparisons. Unpaired interfaces are shown to everyone.
"""

from __future__ import annotations

import json
from pathlib import Path

from gazegrounded.data.ui_render import save_ui
from gazegrounded.data.ui_specs import STUDY_UIS, UISpec
from gazegrounded.utils import ensure_dir, get_logger

log = get_logger(__name__)

STUDY_VERSION = "1.0"


def _group_assignment(specs: tuple[UISpec, ...]) -> dict[str, list[str]]:
    pairs = sorted({s.pair_id for s in specs if s.pair_id})
    groups: dict[str, list[str]] = {"A": [], "B": []}
    for i, pair in enumerate(pairs):
        clean = next(s.ui_id for s in specs if s.pair_id == pair and s.condition == "clean")
        friction = next(s.ui_id for s in specs if s.pair_id == pair and s.condition == "friction")
        groups["A"].append(clean if i % 2 == 0 else friction)
        groups["B"].append(friction if i % 2 == 0 else clean)
    for s in specs:
        if not s.pair_id:
            groups["A"].append(s.ui_id)
            groups["B"].append(s.ui_id)
    return groups


def build_study(out_dir: str | Path, specs: tuple[UISpec, ...] = STUDY_UIS) -> Path:
    """Render stimuli to ``out_dir/stimuli`` and write ``out_dir/tasks.json``."""
    out = ensure_dir(out_dir)
    interfaces, tasks = [], []
    for spec in specs:
        rel = f"stimuli/{spec.ui_id}.png"
        save_ui(spec, out / rel)
        interfaces.append(
            {
                "ui_id": spec.ui_id,
                "image": rel,
                "width": spec.width,
                "height": spec.height,
                "domain": spec.domain,
                "condition": spec.condition,
                "pair_id": spec.pair_id,
                "aois": [
                    {
                        "aoi_id": e.id,
                        "label": e.label.split("\n")[0][:60],
                        "x_min": e.box[0],
                        "y_min": e.box[1],
                        "x_max": e.box[2],
                        "y_max": e.box[3],
                        "semantic_role": e.role,
                        "description": e.description,
                    }
                    for e in spec.aoi_elements()
                ],
            }
        )
        for t in spec.tasks:
            tasks.append(
                {
                    "task_id": f"{spec.ui_id}__{t.task_key}",
                    "ui_id": spec.ui_id,
                    "task_key": t.task_key,
                    "description": t.description,
                    "target_aoi_ids": list(t.target_aoi_ids),
                    "designed_friction_type": t.friction_type,
                }
            )
    groups = _group_assignment(specs)
    doc = {
        "format": "gazegrounded-study-tasks",
        "study_version": STUDY_VERSION,
        "settings": {
            "task_timeout_s": 90,
            "max_calibration_attempts": 3,
            "max_validation_error_px": 200,
            "min_valid_sample_ratio": 0.6,
        },
        "groups": groups,
        "interfaces": interfaces,
        "tasks": tasks,
    }
    (out / "tasks.json").write_text(json.dumps(doc, indent=2))
    n_per = {g: sum(1 for t in tasks if t["ui_id"] in uis) for g, uis in groups.items()}
    log.info(
        "Study: %d interfaces, %d tasks (%s per participant) -> %s",
        len(interfaces),
        len(tasks),
        n_per,
        out,
    )
    return out / "tasks.json"
