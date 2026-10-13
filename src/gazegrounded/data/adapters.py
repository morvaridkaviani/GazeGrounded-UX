"""Adapters for external data: the project's own WebGazer study and the UEyes dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gazegrounded.data.loader import EyeTrackingDatasetAdapter, StandardizedDataset
from gazegrounded.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# WebGazer study exports
# ---------------------------------------------------------------------------------------------


def seq_to_friction(ease: float | None) -> str | None:
    """Map the Single Ease Question (1 = very difficult ... 7 = very easy) to a label.

    Used only when no researcher-coded labels exist. Self-report is a different construct from
    observed friction; ``label_source`` records which one was used.
    """
    if ease is None or (isinstance(ease, float) and np.isnan(ease)):
        return None
    if ease <= 3:
        return "high"
    if ease <= 5:
        return "medium"
    return "low"


class WebGazerStudyAdapter(EyeTrackingDatasetAdapter):
    """Reads ``<raw_dir>/sessions/*.json`` exported by ``study/webgazer``.

    Expected layout::

        data/raw/study/
            tasks.json            # copy of study/webgazer/tasks.json used in the sessions
            stimuli/*.png         # copy of study/webgazer/stimuli
            sessions/*.json       # one file per participant session
            labels.csv            # optional: trial_id, friction_label[, feedback_category]

    If ``tasks.json``/``stimuli`` are absent from ``raw_dir``, ``study/webgazer`` is used.
    Labels: researcher-coded ``labels.csv`` (``label_source="human_coded"``) take precedence;
    otherwise the participant's ease rating is mapped with :func:`seq_to_friction`
    (``label_source="self_report_seq"``).
    """

    source = "study"
    data_kind = "real"

    def _study_dir(self) -> Path:
        if (self.raw_dir / "tasks.json").exists():
            return self.raw_dir
        from gazegrounded.utils import project_root

        return project_root() / "study" / "webgazer"

    def load_raw(self) -> dict:
        study_dir = self._study_dir()
        sessions = []
        for p in sorted((self.raw_dir / "sessions").glob("*.json")):
            doc = json.loads(p.read_text())
            if doc.get("format") != "gazegrounded-study-session":
                log.warning("Skipping %s: not a study session export", p.name)
                continue
            sessions.append(doc)
        if not sessions:
            raise FileNotFoundError(f"No session exports in {self.raw_dir / 'sessions'}")
        labels_path = self.raw_dir / "labels.csv"
        self._raw = {
            "study_dir": study_dir,
            "config": json.loads((study_dir / "tasks.json").read_text()),
            "sessions": sessions,
            "labels": pd.read_csv(labels_path) if labels_path.exists() else None,
        }
        return self._raw

    def standardize(self) -> StandardizedDataset:
        raw = self._raw or self.load_raw()
        cfg = raw["config"]
        uis = pd.DataFrame(
            [
                {
                    "ui_id": u["ui_id"],
                    "image_path": u["image"],
                    "width": u["width"],
                    "height": u["height"],
                    "domain": u["domain"],
                    "condition": u["condition"],
                    "pair_id": u["pair_id"],
                }
                for u in cfg["interfaces"]
            ]
        )
        aois = pd.DataFrame(
            [
                {
                    "ui_id": u["ui_id"],
                    **{
                        k: a[k]
                        for k in (
                            "aoi_id",
                            "label",
                            "x_min",
                            "y_min",
                            "x_max",
                            "y_max",
                            "semantic_role",
                            "description",
                        )
                    },
                }
                for u in cfg["interfaces"]
                for a in u["aois"]
            ]
        )
        tasks = pd.DataFrame(
            [
                {
                    "task_id": t["task_id"],
                    "ui_id": t["ui_id"],
                    "task_key": t["task_key"],
                    "task_description": t["description"],
                    "target_aoi_ids": ";".join(t["target_aoi_ids"]),
                    "designed_friction_type": t.get("designed_friction_type"),
                }
                for t in cfg["tasks"]
            ]
        )
        trial_rows, samples = [], []
        for s in raw["sessions"]:
            pid, sid = s["participant_code"], s["session_id"]
            for t in s["trials"]:
                trial_rows.append(
                    {
                        "participant_id": pid,
                        "session_id": sid,
                        "trial_id": t["trial_id"],
                        "ui_id": t["ui_id"],
                        "task_id": t["task_id"],
                        "task_description": t["task_description"],
                        "condition": t["condition"],
                        "trial_order": t["order"],
                        "task_success": bool(t["task_success"]),
                        "completion_time_ms": float(t["completion_time_ms"]),
                        "error_count": int(t["error_count"]),
                        "click_count": int(t["click_count"]),
                        "feedback_text": t.get("feedback_text"),
                        "ease_rating": t.get("ease_rating"),
                    }
                )
                g = pd.DataFrame(t["gaze"])
                if len(g):
                    samples.append(
                        pd.DataFrame(
                            {
                                "participant_id": pid,
                                "session_id": sid,
                                "trial_id": t["trial_id"],
                                "ui_id": t["ui_id"],
                                "timestamp_ms": g["t"].astype(float),
                                "gaze_x": pd.to_numeric(g["x"], errors="coerce"),
                                "gaze_y": pd.to_numeric(g["y"], errors="coerce"),
                                "validity": g["valid"].astype(int),
                            }
                        )
                    )
        trials = pd.DataFrame(trial_rows)
        labels = raw["labels"]
        if labels is not None:
            trials = trials.merge(labels, on="trial_id", how="left")
            trials["label_source"] = np.where(trials["friction_label"].notna(), "human_coded", None)
        else:
            trials["friction_label"] = trials["ease_rating"].map(seq_to_friction)
            trials["label_source"] = np.where(
                trials["friction_label"].notna(), "self_report_seq", None
            )
        gaze = pd.concat(samples, ignore_index=True) if samples else None
        if gaze is not None:
            gaze = gaze.sort_values(["trial_id", "timestamp_ms"]).reset_index(drop=True)
            bad = gaze["validity"] == 0
            gaze.loc[bad, ["gaze_x", "gaze_y"]] = np.nan
        return StandardizedDataset(
            source=self.source,
            data_kind=self.data_kind,
            root=raw["study_dir"],
            uis=uis,
            aois=aois,
            tasks=tasks,
            trials=trials,
            gaze_samples=gaze,
        )


# ---------------------------------------------------------------------------------------------
# UEyes (Jiang et al., CHI 2023)
# ---------------------------------------------------------------------------------------------

_LOG_NAME = re.compile(r"^(\d+)_(?:kh|KH)(\d+)_fixations\.csv$")


class UEyesAdapter(EyeTrackingDatasetAdapter):
    """Free-viewing UI eye tracking from UEyes (Zenodo record 8010312, CC BY 4.0).

    Layout (inside ``UEyes_dataset/``)::

        image_types.csv         semicolon-separated: Image Name;Category;Block;Train/Test
        images/<name>           1,980 screenshots (file extension may not match the format)
        eyetracker_logs/{block}_{kh|KH}{participant}_fixations.csv   Gazepoint GP3 exports

    Coordinates: ``FPOGX/FPOGY`` are fractions of the 1920 x 1200 *screen*, on which each image
    was scaled to fit with its aspect ratio kept and centred (letterboxed). This adapter maps
    them to image pixels the same way the dataset's own heatmap script does. (The repository's
    log parser instead multiplies by the image size; set ``coordinate_frame="image"`` for that
    interpretation.) Fixation start times are re-based to the first fixation on each image.

    UEyes has no tasks, interaction logs, feedback or friction labels: it supports gaze
    analysis and the saliency model (RQ1 baselines), not the friction model.
    """

    source = "ueyes"
    data_kind = "real"
    SCREEN = (1920, 1200)

    def __init__(
        self,
        raw_dir,
        screen_width: int = 1920,
        screen_height: int = 1200,
        coordinate_frame: str = "screen_letterbox",
    ):
        super().__init__(raw_dir, screen_width, screen_height)
        self.coordinate_frame = coordinate_frame

    def _base(self) -> Path:
        nested = self.raw_dir / "UEyes_dataset"
        return nested if nested.exists() else self.raw_dir

    def load_raw(self) -> dict:
        base = self._base()
        info_path = base / "image_types.csv"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found. See data/README.md for UEyes setup.")
        info = pd.read_csv(info_path, sep=";", dtype=str)
        info.columns = [c.strip() for c in info.columns]
        logs = []
        for p in sorted((base / "eyetracker_logs").glob("*_fixations.csv")):
            m = _LOG_NAME.match(p.name)
            if not m:  # skips .~lock files, __MACOSX copies, etc.
                continue
            df = pd.read_csv(p, low_memory=False)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # trailing comma column
            df["participant_id"] = f"U{int(m.group(2)):03d}"
            df["block"] = int(m.group(1))
            logs.append(df)
        if not logs:
            raise FileNotFoundError(f"No eyetracker logs in {base / 'eyetracker_logs'}")
        self._raw = {"base": base, "info": info, "logs": pd.concat(logs, ignore_index=True)}
        return self._raw

    def _to_image_px(self, fx: np.ndarray, fy: np.ndarray, w: int, h: int):
        if self.coordinate_frame == "image":
            return fx * w, fy * h
        sw, sh = self.SCREEN
        scale = np.minimum(sw / w, sh / h)
        padx, pady = (sw - w * scale) / 2, (sh - h * scale) / 2
        return (fx * sw - padx) / scale, (fy * sh - pady) / scale

    def standardize(self) -> StandardizedDataset:
        raw = self._raw or self.load_raw()
        base, info, logs = raw["base"], raw["info"], raw["logs"]
        ui_rows = []
        for r in info.itertuples(index=False):
            name = r[0]
            path = base / "images" / name
            if not path.exists():
                continue
            with Image.open(path) as im:  # format detected from content, not extension
                w, h = im.size
            ui_rows.append(
                {
                    "ui_id": Path(name).stem,
                    "image_path": f"images/{name}",
                    "width": w,
                    "height": h,
                    "domain": str(r[1]).strip(),
                    "condition": "free_viewing",
                    "pair_id": None,
                }
            )
        uis = pd.DataFrame(ui_rows)
        sizes = uis.set_index("ui_id")[["width", "height"]]
        aois = pd.DataFrame(
            {
                "ui_id": uis["ui_id"],
                "aoi_id": "full_image",
                "label": "image",
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 1.0,
                "y_max": 1.0,
                "semantic_role": "image",
                "description": "Whole screenshot",
            }
        )
        tasks = pd.DataFrame(
            {
                "task_id": uis["ui_id"] + "__free_viewing",
                "ui_id": uis["ui_id"],
                "task_key": "free_viewing",
                "task_description": "Free viewing (no task)",
                "target_aoi_ids": "full_image",
                "designed_friction_type": None,
            }
        )

        f = logs.copy()
        f["ui_id"] = f["MEDIA_NAME"].astype(str).map(lambda n: Path(n).stem)
        f = f[f["ui_id"].isin(sizes.index) & (pd.to_numeric(f["FPOGV"], errors="coerce") == 1)]
        f = f.sort_values("FPOGD").drop_duplicates(
            ["participant_id", "ui_id", "FPOGID"], keep="last"
        )
        w = f["ui_id"].map(sizes["width"]).to_numpy(float)
        h = f["ui_id"].map(sizes["height"]).to_numpy(float)
        x, y = self._to_image_px(
            f["FPOGX"].astype(float).to_numpy(), f["FPOGY"].astype(float).to_numpy(), w, h
        )
        f = f.assign(centroid_x=x, centroid_y=y)
        f = f[
            (f["centroid_x"] >= 0)
            & (f["centroid_x"] <= w)
            & (f["centroid_y"] >= 0)
            & (f["centroid_y"] <= h)
        ]
        f["trial_id"] = f["participant_id"] + "_" + f["ui_id"]
        start = f["FPOGS"].astype(float) * 1000
        f["start_ms"] = start - start.groupby(f["trial_id"]).transform("min")
        f["duration_ms"] = f["FPOGD"].astype(float) * 1000
        f = f[f["duration_ms"] > 0].sort_values(["trial_id", "start_ms"])
        f["end_ms"] = f["start_ms"] + f["duration_ms"]
        f["fixation_id"] = f.groupby("trial_id").cumcount()
        fixations = f[
            [
                "participant_id",
                "trial_id",
                "ui_id",
                "fixation_id",
                "start_ms",
                "end_ms",
                "duration_ms",
                "centroid_x",
                "centroid_y",
            ]
        ].assign(aoi_id=None)
        fixations = fixations.reset_index(drop=True)

        trials = fixations.groupby(["trial_id", "participant_id", "ui_id"], as_index=False).agg(
            completion_time_ms=("end_ms", "max")
        )
        trials["session_id"] = trials["participant_id"] + "_ueyes"
        trials["task_id"] = trials["ui_id"] + "__free_viewing"
        trials["task_description"] = "Free viewing (no task)"
        trials["condition"] = "free_viewing"
        log.info(
            "UEyes: %d images, %d participants, %d trials, %d fixations",
            len(uis),
            trials["participant_id"].nunique(),
            len(trials),
            len(fixations),
        )
        return StandardizedDataset(
            source=self.source,
            data_kind=self.data_kind,
            root=base,
            uis=uis,
            aois=aois,
            tasks=tasks,
            trials=trials,
            fixations=fixations,
        )
