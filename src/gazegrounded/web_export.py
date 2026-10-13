"""Export pipeline outputs for the static project website in ``docs/``.

The website (plain HTML/CSS/JS, hostable on GitHub Pages) reads ``docs/data/site.json`` and
small greyscale PNG maps. Everything shown on the site is produced by this exporter from the
pipeline's outputs, so the site never contains hand-entered numbers.

Maps (gaze heatmaps, model relevance maps) are stored as 8-bit greyscale PNGs on the analysis
grid; the browser colours them. Run after ``run-all``::

    python -m gazegrounded.cli export-site
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gazegrounded.data.loader import load_processed, read_table
from gazegrounded.data.ui_specs import all_specs
from gazegrounded.gaze.heatmap import fixation_heatmap
from gazegrounded.utils import SYNTHETIC_BANNER, ensure_dir, get_logger, latest_run_dir

log = get_logger(__name__)


def _r(v, n: int = 3):
    """Round for JSON; NaN/None -> None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if np.isnan(f) else round(f, n)


def _png(m: np.ndarray, path: Path) -> str:
    m = np.asarray(m, dtype=float)
    peak = m.max()
    img = (
        (np.clip(m / peak, 0, 1) * 255).astype(np.uint8) if peak > 0 else np.zeros_like(m, np.uint8)
    )
    ensure_dir(path.parent)
    Image.fromarray(img).save(path, optimize=True)
    return path.as_posix()


def _ui_name(ui_id: str, specs) -> tuple[str, str]:
    spec = specs.get(ui_id)
    if spec is None:
        return ui_id.replace("_", " ").title(), ""
    return spec.title.replace(" (clean)", "").replace(" (friction)", ""), spec.domain


def export_site(cfg, out_dir: str | Path | None = None) -> Path:
    from gazegrounded.pipeline import load_fixations, load_vlm_analyses, load_vlm_maps

    out = ensure_dir(out_dir or cfg.root() / "docs")
    data_dir = ensure_dir(out / "data")
    for sub in ("maps", "ui", "figures"):
        shutil.rmtree(data_dir / sub, ignore_errors=True)
    ds = load_processed(cfg)
    fixations = load_fixations(cfg)
    interim = cfg.interim_dir()
    gaze = pd.read_csv(interim / "gaze_metrics.csv").set_index("trial_id")
    features = pd.read_csv(cfg.processed_dir() / "features.csv").set_index("trial_id")
    vlm_maps = load_vlm_maps(cfg)
    analyses = load_vlm_analyses(cfg)
    elements = read_table(interim / "vlm" / "vlm_elements.csv")
    vlm_tasks = pd.read_csv(interim / "vlm" / "vlm_task_features.csv").set_index("task_id")
    nlp_path = interim / "nlp_predictions.csv"
    nlp = read_table(nlp_path).set_index("trial_id") if nlp_path.exists() else pd.DataFrame()
    friction_run = latest_run_dir(cfg.experiments_dir(), "friction")
    report_dir = cfg.reports_dir() / cfg.data.source
    specs = all_specs()
    g = cfg.gaze

    # ---- interfaces, AOIs, tasks --------------------------------------------------------------
    uis = []
    for u in ds.uis.itertuples():
        name, domain = _ui_name(u.ui_id, specs)
        rel = Path("data/ui") / f"{u.ui_id}.png"
        ensure_dir((out / rel).parent)
        shutil.copy(ds.image_path(u.ui_id), out / rel)
        aois = [
            {
                "id": a.aoi_id,
                "label": a.label if isinstance(a.label, str) else None,
                "box": [_r(a.x_min, 4), _r(a.y_min, 4), _r(a.x_max, 4), _r(a.y_max, 4)],
            }
            for a in ds.aois_for(u.ui_id).itertuples()
        ]
        uis.append(
            {
                "id": u.ui_id,
                "name": name,
                "domain": domain or u.domain,
                "condition": u.condition,
                "image": rel.as_posix(),
                "width": int(u.width),
                "height": int(u.height),
                "aois": aois,
            }
        )

    tasks = []
    for t in ds.tasks.itertuples():
        w, h = ds.ui_size(t.ui_id)
        trial_ids = ds.trials.loc[ds.trials["task_id"] == t.task_id, "trial_id"]
        f = fixations[fixations["trial_id"].isin(trial_ids)]
        human = fixation_heatmap(f, w, h, g.heatmap_sigma_px, g.map_scale, normalize="max")
        human_png = _png(human, data_dir / "maps" / "human" / f"{t.task_id}.png")
        vlm_png = _png(vlm_maps[t.task_id], data_dir / "maps" / "vlm" / f"{t.task_id}.png")
        a = analyses.get(t.task_id)
        el = elements[elements["task_id"] == t.task_id].sort_values("weight", ascending=False)
        from gazegrounded.analysis.saliency_metrics import compare_maps

        m = compare_maps(human, vlm_maps[t.task_id])
        vt = vlm_tasks.loc[t.task_id] if t.task_id in vlm_tasks.index else None
        tasks.append(
            {
                "id": t.task_id,
                "ui": t.ui_id,
                "description": t.task_description,
                "targets": [x for x in str(t.target_aoi_ids).split(";") if x],
                "designed": t.designed_friction_type,
                "humanMap": Path(human_png).relative_to(out).as_posix(),
                "modelMap": Path(vlm_png).relative_to(out).as_posix(),
                "nParticipants": int(f["trial_id"].nunique()),
                "metrics": {k: _r(v) for k, v in m.items()},
                "model": {
                    "provider": a.provider if a else None,
                    "name": a.model if a else None,
                    "parseStatus": a.parse_status if a else None,
                    "topIsTarget": bool(vt["vlm_top_is_target"]) if vt is not None else None,
                    "elements": [
                        {
                            "label": e.label,
                            "role": e.semantic_role,
                            "relevance": _r(e.task_relevance),
                            "priority": _r(e.expected_attention_priority),
                            "score": _r(e.weight),
                            "box": [_r(e.x_min, 4), _r(e.y_min, 4), _r(e.x_max, 4), _r(e.y_max, 4)],
                            "matched": e.matched_aoi_id,
                            "reason": e.reason or "",
                        }
                        for e in el.itertuples()
                    ],
                },
            }
        )

    # ---- trials ---------------------------------------------------------------------------------
    preds, contrib = {}, {}
    if friction_run is not None:
        p = pd.read_csv(friction_run / "predictions.csv")
        p = p[
            (p["modality"] == "multimodal")
            & (p["model"] == "logistic_regression")
            & (p["group_by"] == "participant_id")
        ]
        preds = dict(zip(p["trial_id"], p["pred"], strict=True))
        cpath = friction_run / "contributions.csv"
        if cpath.exists():
            c = pd.read_csv(cpath)
            c = c[c["model"] == "logistic_regression"]
            for tid, grp in c.groupby("trial_id"):
                top = grp.reindex(grp["contribution"].abs().sort_values(ascending=False).index)
                contrib[tid] = [[r.feature, _r(r.contribution)] for r in top.head(8).itertuples()]

    fix_by_trial = dict(tuple(fixations.groupby("trial_id")))
    trials = []
    for t in ds.trials.itertuples():
        w, h = ds.ui_size(t.ui_id)
        f = fix_by_trial.get(t.trial_id, fixations.iloc[0:0]).sort_values("start_ms")
        heat = fixation_heatmap(f, w, h, g.heatmap_sigma_px, g.map_scale, normalize="max")
        heat_png = _png(heat, data_dir / "maps" / "trial" / f"{t.trial_id}.png")
        gm = gaze.loc[t.trial_id]
        ft = features.loc[t.trial_id]
        n = nlp.loc[t.trial_id] if t.trial_id in nlp.index else None
        trials.append(
            {
                "id": t.trial_id,
                "participant": t.participant_id,
                "ui": t.ui_id,
                "task": t.task_id,
                "condition": t.condition,
                "success": bool(t.task_success),
                "completionS": _r(t.completion_time_ms / 1000, 2),
                "errors": int(t.error_count),
                "ttffS": _r(gm["ttff_ms"] / 1000, 2),
                "fixationCount": int(gm["fixation_count"]),
                "revisits": int(gm["revisit_count"]),
                "similarity": _r(ft["vlm_human_similarity"]),
                "divergenceEarly": _r(ft["attention_divergence_early"]),
                "label": t.friction_label,
                "predicted": preds.get(t.trial_id),
                "feedback": t.feedback_text if isinstance(t.feedback_text, str) else None,
                "nlp": (
                    None
                    if n is None
                    else {
                        "category": n["pred"],
                        "confidence": _r(n["confidence"], 2),
                        "coded": n["feedback_category"],
                    }
                ),
                "fixations": [
                    [
                        _r(r.centroid_x, 1),
                        _r(r.centroid_y, 1),
                        _r(r.duration_ms, 0),
                        _r(r.start_ms, 0),
                        r.aoi_id if isinstance(r.aoi_id, str) else None,
                    ]
                    for r in f.itertuples()
                ],
                "heatmap": Path(heat_png).relative_to(out).as_posix(),
                "contributions": contrib.get(t.trial_id, []),
            }
        )

    # ---- results --------------------------------------------------------------------------------
    results: dict = {}
    rq1 = report_dir / "rq1_attention_baselines_per_task.csv"
    if rq1.exists():
        per_task = pd.read_csv(rq1)
        results["baselines"] = [
            {
                "baseline": b,
                **{
                    f"{m}_{s}": _r(getattr(grp[m], s)())
                    for m in ("cc", "sim", "kl", "nss")
                    for s in ("mean", "min", "max")
                },
                "n": int(grp["task_id"].nunique()),
            }
            for b, grp in per_task.groupby("baseline")
        ]
    if friction_run is not None:
        comp = pd.read_csv(friction_run / "modality_comparison.csv")
        results["modalities"] = [
            {
                "split": r.group_by,
                "model": r.model,
                "modality": r.modality,
                "macroF1": _r(r.macro_f1),
                "macroF1Mean": _r(r.macro_f1_fold_mean),
                "macroF1Sd": _r(r.macro_f1_fold_sd),
                "kappa": _r(r.quadratic_weighted_kappa),
            }
            for r in comp.itertuples()
        ]
        abl = pd.read_csv(friction_run / "ablation.csv")
        results["ablation"] = [
            {"without": r.dropped_modality, "macroF1": _r(r.macro_f1)} for r in abl.itertuples()
        ]
        p = pd.read_csv(friction_run / "predictions.csv")
        p = p[
            (p["modality"] == "multimodal")
            & (p["model"] == "logistic_regression")
            & (p["group_by"] == "participant_id")
        ]
        labels = [x for x in ("low", "medium", "high") if (p["true"] == x).any()]
        results["confusion"] = {
            "labels": labels,
            "matrix": [
                [int(((p["true"] == a) & (p["pred"] == b)).sum()) for b in labels] for a in labels
            ],
        }
        imp = pd.read_csv(friction_run / "feature_importance.csv")
        perm = imp[
            (imp["kind"] == "permutation_macro_f1_drop") & (imp["model"] == "logistic_regression")
        ]
        results["importance"] = [
            [r.feature, _r(r.importance_mean)]
            for r in perm.sort_values("importance_mean", ascending=False).head(12).itertuples()
        ]
    fc = report_dir / "failure_cases.csv"
    if fc.exists():
        fails = pd.read_csv(fc)
        results["failures"] = [
            {
                "task": r.task_id,
                "targetShare": _r(r.human_dwell_share_on_target),
                "modelTopShare": _r(r.human_dwell_share_on_vlm_top),
            }
            for r in fails.itertuples()
        ]

    site = {
        "meta": {
            "project": "GazeGrounded UX",
            "dataKind": cfg.data.data_kind,
            "notice": SYNTHETIC_BANNER if cfg.is_synthetic else None,
            "exported": date.today().isoformat(),
            "participants": int(ds.trials["participant_id"].nunique()),
            "trials": len(ds.trials),
            "interfaces": len(ds.uis),
            "tasks": len(ds.tasks),
            "mapGrid": [
                int(np.round(ds.uis["height"].iloc[0] * g.map_scale)),
                int(np.round(ds.uis["width"].iloc[0] * g.map_scale)),
            ],
            "provider": tasks[0]["model"]["provider"] if tasks else None,
        },
        "uis": uis,
        "tasks": tasks,
        "trials": trials,
        "results": results,
    }
    path = data_dir / "site.json"
    path.write_text(json.dumps(site, separators=(",", ":"), allow_nan=False))
    size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    log.info("Website data exported to %s (%.1f MB)", data_dir, size / 1e6)
    return path
