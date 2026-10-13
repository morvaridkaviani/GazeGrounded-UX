import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gazegrounded.data.adapters import UEyesAdapter, WebGazerStudyAdapter, seq_to_friction
from gazegrounded.data.study_builder import build_study

# ---- WebGazer study --------------------------------------------------------------------------


def _fake_session(cfg, code="P01", group="A"):
    """A session document shaped exactly like study.js's sessionDocument()."""
    uis = set(cfg["groups"][group])
    tasks = [t for t in cfg["tasks"] if t["ui_id"] in uis][:3]
    rng = np.random.default_rng(0)
    trials = []
    for i, t in enumerate(tasks, start=1):
        ui = next(u for u in cfg["interfaces"] if u["ui_id"] == t["ui_id"])
        times = np.arange(0, 3000, 33.3)
        gaze = [
            {
                "t": float(tt),
                "x": float(rng.uniform(0, ui["width"])),
                "y": float(rng.uniform(0, ui["height"])),
                "valid": 1,
            }
            for tt in times
        ]
        gaze[5] = {"t": gaze[5]["t"], "x": None, "y": None, "valid": 0}
        trials.append(
            {
                "trial_id": f"{code}_T{i:02d}",
                "order": i,
                "task_id": t["task_id"],
                "ui_id": t["ui_id"],
                "condition": ui["condition"],
                "task_description": t["description"],
                "image_rect": {"left": 0, "top": 0, "width": 1440, "height": 900},
                "natural_size": {"width": ui["width"], "height": ui["height"]},
                "completion_time_ms": 3000,
                "task_success": i != 2,
                "end_reason": "target_clicked",
                "error_count": 1,
                "click_count": 2,
                "clicks": [],
                "ease_rating": [7, 2, 4][i - 1],
                "feedback_text": "It was hidden" if i == 2 else None,
                "quality": {"n_samples": len(gaze), "valid_ratio": 0.99},
                "gaze": gaze,
            }
        )
    return {
        "format": "gazegrounded-study-session",
        "format_version": 1,
        "study_version": cfg["study_version"],
        "participant_code": code,
        "session_id": "abc123",
        "group": group,
        "webgazer_version": "3.5.3",
        "environment": {},
        "calibration": {"attempts": 1, "validations": []},
        "trials": trials,
    }


@pytest.fixture()
def study_raw(tmp_path):
    raw = tmp_path / "study"
    build_study(raw)
    cfg = json.loads((raw / "tasks.json").read_text())
    (raw / "sessions").mkdir()
    for code, group in (("P01", "A"), ("P02", "B")):
        (raw / "sessions" / f"{code}.json").write_text(json.dumps(_fake_session(cfg, code, group)))
    return raw


def test_study_build_counterbalances_pairs(tmp_path):
    cfg = json.loads(build_study(tmp_path).read_text())
    a, b = set(cfg["groups"]["A"]), set(cfg["groups"]["B"])
    paired = {u["ui_id"] for u in cfg["interfaces"] if u["pair_id"]}
    assert (a & paired).isdisjoint(b & paired)  # each paired variant seen by one group only
    assert {u["condition"] for u in cfg["interfaces"] if u["ui_id"] in a} >= {"clean", "friction"}
    for t in cfg["tasks"]:
        ui = next(u for u in cfg["interfaces"] if u["ui_id"] == t["ui_id"])
        assert set(t["target_aoi_ids"]) <= {x["aoi_id"] for x in ui["aois"]}
    assert all((tmp_path / u["image"]).exists() for u in cfg["interfaces"])


def test_webgazer_adapter_standardises_and_validates(study_raw):
    ds, report = WebGazerStudyAdapter(study_raw).run()
    assert report.ok, report.errors
    assert ds.data_kind == "real"
    assert len(ds.trials) == 6 and ds.trials["participant_id"].nunique() == 2
    assert set(ds.trials["label_source"]) == {"self_report_seq"}
    assert ds.trials.set_index("trial_id").loc["P01_T02", "friction_label"] == "high"
    invalid = ds.gaze_samples[ds.gaze_samples["validity"] == 0]
    assert invalid["gaze_x"].isna().all()


def test_webgazer_adapter_prefers_human_coded_labels(study_raw):
    pd.DataFrame(
        {
            "trial_id": ["P01_T01", "P01_T02"],
            "friction_label": ["low", "high"],
            "feedback_category": ["no_friction", "findability"],
        }
    ).to_csv(study_raw / "labels.csv", index=False)
    ds, report = WebGazerStudyAdapter(study_raw).run()
    assert report.ok, report.errors
    coded = ds.trials[ds.trials["label_source"] == "human_coded"]
    assert set(coded["trial_id"]) == {"P01_T01", "P01_T02"}


def test_seq_mapping():
    assert [seq_to_friction(v) for v in (1, 3, 4, 5, 6, 7, None)] == [
        "high",
        "high",
        "medium",
        "medium",
        "low",
        "low",
        None,
    ]


# ---- UEyes -----------------------------------------------------------------------------------

UEYES_HEADER = (
    "MEDIA_ID,MEDIA_NAME,CNT,TIME(2022/06/08 14:27:04.215),TIMETICK(f=10000000),FPOGX,FPOGY,"
    "FPOGS,FPOGD,FPOGID,FPOGV,BPOGX,BPOGY,BPOGV,CX,CY,CS,USER,LPCX,LPCY,LPD,LPS,LPV,RPCX,RPCY,"
    "RPD,RPS,RPV,BKID,BKDUR,BKPMIN,LPMM,LPMMV,RPMM,RPMMV,DIAL,DIALV,GSR,GSRV,HR,HRV,HRP,TTL0,"
    "TTL1,TTLV,PIXS,PIXV,AOI,SACCADE_MAG,SACCADE_DIR,VID_FRAME,"
)


def _ueyes_row(media, fid, fx, fy, fs, fd, valid=1):
    vals = ["0", media, "1", "0.5", "0", fx, fy, fs, fd, fid, valid] + ["0"] * 40
    return ",".join(str(v) for v in vals) + ","


@pytest.fixture()
def ueyes_raw(tmp_path):
    base = tmp_path / "ueyes" / "UEyes_dataset"
    (base / "images").mkdir(parents=True)
    (base / "eyetracker_logs").mkdir()
    Image.new("RGB", (1000, 500), "white").save(base / "images" / "aaa111.png")
    Image.new("RGB", (600, 1200), "white").save(base / "images" / "bbb222.png", format="JPEG")
    (base / "image_types.csv").write_text(
        "Image Name;Category;Block;Train/Test\naaa111.png;web;1,10E+01;Train\n"
        "bbb222.png;mobile;3;Test\n"
    )
    rows = [
        UEYES_HEADER,
        _ueyes_row("aaa111.png", 1, 0.5, 0.5, 10.0, 0.30),  # centre of the image
        _ueyes_row("aaa111.png", 2, 0.5, 0.05, 10.4, 0.20),  # in the letterbox -> dropped
        _ueyes_row("aaa111.png", 3, 0.25, 0.5, 10.7, 0.25, valid=0),  # invalid -> dropped
        _ueyes_row("bbb222.png", 1, 0.5, 0.5, 20.0, 0.40),
    ]
    (base / "eyetracker_logs" / "03_kh012_fixations.csv").write_text("\n".join(rows) + "\n")
    (base / "eyetracker_logs" / "04_KH013_fixations.csv").write_text(
        "\n".join([UEYES_HEADER, _ueyes_row("bbb222.png", 1, 0.5, 0.25, 5.0, 0.5)]) + "\n"
    )
    (base / "eyetracker_logs" / ".~lock.05_kh014_fixations.csv#").write_text("junk")
    return tmp_path / "ueyes"


def test_ueyes_adapter_maps_letterboxed_screen_coordinates(ueyes_raw):
    ds, report = UEyesAdapter(ueyes_raw).run()
    assert report.ok, report.errors
    assert set(ds.uis["ui_id"]) == {"aaa111", "bbb222"}
    assert ds.ui_size("bbb222") == (600, 1200)  # JPEG content behind a .png name
    f = ds.fixations.set_index("trial_id")
    a = f.loc["U012_aaa111"]
    assert float(a["centroid_x"]) == pytest.approx(500) and float(a["centroid_y"]) == pytest.approx(
        250
    )
    assert float(a["start_ms"]) == 0 and float(a["duration_ms"]) == pytest.approx(300)
    assert set(ds.trials["participant_id"]) == {"U012", "U013"}  # kh and KH both parsed
    b = f.loc["U013_bbb222"]  # 600x1200 image: scale 1.0, padx 660 -> x = 960 - 660 = 300
    assert float(b["centroid_x"]) == pytest.approx(300) and float(b["centroid_y"]) == pytest.approx(
        300
    )


def test_ueyes_runs_through_preprocess_and_gaze_analysis(ueyes_raw, tmp_path):
    from gazegrounded import pipeline
    from gazegrounded.config import load_config

    cfg = load_config(
        "configs/ueyes.yaml", [f"data.raw_dir={ueyes_raw}", f"paths.data_root={tmp_path / 'data'}"]
    )
    pipeline.stage_preprocess(cfg)
    out = pipeline.stage_gaze_analysis(cfg, make_figures=False)
    assert len(out["trial_metrics"]) == 3
    assert out["trial_metrics"]["target_fixated"].all()  # full-image AOI
