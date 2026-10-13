"""Pipeline stages. The CLI and the ``scripts/`` wrappers call these functions.

Each stage reads the previous stage's files from ``data/{processed,interim}/<source>/`` and
writes its own outputs, so stages can be re-run independently.
"""

from __future__ import annotations

import pandas as pd

from gazegrounded.data.loader import StandardizedDataset, load_processed, read_table
from gazegrounded.utils import ensure_dir, get_logger, save_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def stage_generate_demo(cfg) -> StandardizedDataset:
    from gazegrounded.data.synthetic import generate_synthetic_dataset

    if cfg.data.adapter != "synthetic":
        raise ValueError("generate-demo only makes sense with a synthetic config")
    return generate_synthetic_dataset(cfg)


def stage_preprocess(cfg) -> StandardizedDataset:
    from gazegrounded.data.preprocessing import run_preprocessing

    return run_preprocessing(cfg)


# ---------------------------------------------------------------------------------------------
# Gaze
# ---------------------------------------------------------------------------------------------


def load_fixations(cfg) -> pd.DataFrame:
    return read_table(cfg.interim_dir() / "fixations.csv")


def stage_gaze_analysis(cfg, make_figures: bool = True) -> dict[str, pd.DataFrame]:
    """Fixations (I-DT) -> AOI assignment -> trial/AOI metrics -> scanpaths -> figures."""
    from gazegrounded.gaze.aoi import assign_fixations_to_aois
    from gazegrounded.gaze.fixation import detect_fixations
    from gazegrounded.gaze.metrics import compute_gaze_metrics
    from gazegrounded.gaze.scanpath import aoi_sequence, format_scanpath

    ds = load_processed(cfg)
    out = ensure_dir(cfg.interim_dir())
    g = cfg.gaze
    if ds.gaze_samples is not None:
        fixations = detect_fixations(
            ds.gaze_samples, g.dispersion_threshold_px, g.minimum_duration_ms, g.max_gap_ms
        )
        log.info(
            "I-DT detected %d fixations in %d trials",
            len(fixations),
            fixations["trial_id"].nunique(),
        )
    elif ds.fixations is not None:
        fixations = ds.fixations.copy()
        log.info("Source provides fixations directly (%d rows); skipping I-DT", len(fixations))
    else:
        raise ValueError("Dataset has neither gaze samples nor fixations")
    if "ui_id" not in fixations.columns:
        fixations = fixations.merge(ds.trials[["trial_id", "ui_id"]], on="trial_id")
    fixations = assign_fixations_to_aois(fixations, ds.aois, ds.uis, g.aoi_margin_px)
    trial_metrics, aoi_metrics = compute_gaze_metrics(ds, fixations, g.aoi_margin_px)

    labels = dict(zip(ds.aois["aoi_id"], ds.aois["aoi_id"], strict=True))
    scanpaths = [
        {
            "trial_id": tid,
            "scanpath": format_scanpath(aoi_sequence(f), labels, max_len=30),
            "aoi_sequence": ";".join(a for a in aoi_sequence(f)),
        }
        for tid, f in fixations.groupby("trial_id")
    ]
    fixations.to_csv(out / "fixations.csv", index=False)
    trial_metrics.to_csv(out / "gaze_metrics.csv", index=False)
    aoi_metrics.to_csv(out / "aoi_metrics.csv", index=False)
    pd.DataFrame(scanpaths).to_csv(out / "scanpaths.csv", index=False)
    save_json(
        {
            "n_fixations": len(fixations),
            "n_trials": int(trial_metrics.shape[0]),
            "fixation_algorithm": g.fixation_algorithm,
            "dispersion_threshold_px": g.dispersion_threshold_px,
            "minimum_duration_ms": g.minimum_duration_ms,
        },
        out / "gaze_summary.json",
    )
    log.info("Gaze metrics written to %s", out)
    if make_figures:
        gaze_figures(cfg, ds, fixations, aoi_metrics)
    return {"fixations": fixations, "trial_metrics": trial_metrics, "aoi_metrics": aoi_metrics}


def gaze_figures(cfg, ds: StandardizedDataset, fixations, aoi_metrics, max_trials: int = 3):
    """Example trial figures (one per condition) and an AOI dwell plot per interface."""
    from gazegrounded.gaze.heatmap import fixation_heatmap
    from gazegrounded.visualization.gaze import aoi_dwell_figure, load_image, trial_figure
    from gazegrounded.visualization.style import save_figure

    fig_dir = ensure_dir(cfg.figures_dir() / "gaze")
    trials = ds.trials
    examples = trials.groupby("condition", sort=False).head(1).head(max_trials)
    for t in examples.itertuples():
        f = fixations[fixations["trial_id"] == t.trial_id]
        w, h = ds.ui_size(t.ui_id)
        heat = fixation_heatmap(f, w, h, cfg.gaze.heatmap_sigma_px, scale=0.5)
        fig = trial_figure(
            load_image(ds.image_path(t.ui_id)),
            f,
            ds.aois_for(t.ui_id),
            ds.targets_for_task(t.task_id),
            heat,
            f"{t.participant_id} · {t.ui_id} · “{t.task_description}”",
        )
        save_figure(fig, fig_dir / f"trial_{t.trial_id}.png", ds.is_synthetic)

    merged = aoi_metrics.merge(trials[["trial_id", "task_id"]], on="trial_id")
    for task_id, d in merged.groupby("task_id"):
        summary = d.groupby(["aoi_id", "is_target"], as_index=False)["dwell_ms"].mean()
        summary = summary.rename(columns={"dwell_ms": "mean_dwell_ms"})
        task = ds.tasks.set_index("task_id").loc[task_id]
        fig = aoi_dwell_figure(summary, f"{task['ui_id']}: “{task['task_description']}”")
        save_figure(fig, fig_dir / f"aoi_dwell_{task_id}.png", ds.is_synthetic)
    log.info("Gaze figures written to %s", fig_dir)


# ---------------------------------------------------------------------------------------------
# VLM grounding + human/VLM comparison
# ---------------------------------------------------------------------------------------------


def load_vlm_maps(cfg) -> dict:
    import numpy as np

    with np.load(cfg.interim_dir() / "vlm" / "vlm_maps.npz") as z:
        return {k: z[k] for k in z.files}


def load_vlm_analyses(cfg) -> dict:
    from gazegrounded.vlm.schemas import VLMAnalysis

    parsed = cfg.interim_dir() / "vlm" / "parsed"
    return {p.stem: VLMAnalysis.model_validate_json(p.read_text()) for p in parsed.glob("*.json")}


def stage_vlm_analysis(cfg, make_figures: bool = True) -> dict[str, pd.DataFrame]:
    """Run the VLM on every (screenshot, task), build relevance maps, compare with gaze."""
    import numpy as np

    from gazegrounded.analysis.attention_alignment import (
        aoi_label_grid,
        map_mass_by_aoi,
        trial_alignment,
        vlm_task_features,
    )
    from gazegrounded.vlm.attention_map import vlm_relevance_map
    from gazegrounded.vlm.base import make_provider
    from gazegrounded.vlm.grounding import match_elements_to_aois
    from gazegrounded.vlm.schemas import VLMAnalysis

    ds = load_processed(cfg)
    fixations = load_fixations(cfg)
    provider = make_provider(cfg, ds)
    out = ensure_dir(cfg.interim_dir() / "vlm")
    raw_dir, parsed_dir = ensure_dir(out / "raw"), ensure_dir(out / "parsed")
    g, c = cfg.gaze, cfg.comparison
    log.info(
        "VLM provider: %s (model=%s, prompt %s, cache=%s)",
        provider.name,
        provider.model,
        provider.prompt_version,
        provider.use_cache,
    )

    maps, analyses, element_rows, task_rows = {}, {}, [], []
    for task in ds.tasks.itertuples():
        w, h = ds.ui_size(task.ui_id)
        image = ds.image_path(task.ui_id)
        try:
            analysis = provider.analyze_ui(image, task.task_description)
        except Exception as exc:  # a failed call must not stop the batch
            log.error("VLM call failed for %s: %s", task.task_id, exc)
            analysis = VLMAnalysis(
                task=task.task_description,
                provider=provider.name,
                model=provider.model,
                parse_status="failed",
                warnings=[f"generation error: {exc}"],
            )
        raw = provider.raw_response(image, task.task_description)
        if raw is not None:
            (raw_dir / f"{task.task_id}.txt").write_text(raw)
        (parsed_dir / f"{task.task_id}.json").write_text(analysis.model_dump_json(indent=2))
        aois = ds.aois_for(task.ui_id)
        elements = match_elements_to_aois(analysis, aois)
        if len(elements):
            elements.insert(0, "task_id", task.task_id)
            element_rows.append(elements)
        targets = ds.targets_for_task(task.task_id)
        task_rows.append(
            {
                "task_id": task.task_id,
                "ui_id": task.ui_id,
                "provider": provider.name,
                "model": provider.model,
                **vlm_task_features(analysis, elements, targets, c.relevance_threshold),
            }
        )
        maps[task.task_id] = vlm_relevance_map(
            analysis, w, h, g.map_scale, c.vlm_box_softness, g.heatmap_sigma_px
        )
        analyses[task.task_id] = analysis

    # Trial-level alignment between each participant's gaze and the VLM map for that task.
    fix_by_trial = dict(tuple(fixations.groupby("trial_id")))
    grids, align_rows = {}, []
    for trial in ds.trials.itertuples():
        w, h = ds.ui_size(trial.ui_id)
        if trial.ui_id not in grids:
            grids[trial.ui_id] = aoi_label_grid(ds.aois_for(trial.ui_id), w, h, g.map_scale)
        vmap = maps[trial.task_id]
        metrics = trial_alignment(
            fix_by_trial.get(trial.trial_id, fixations.iloc[0:0]),
            vmap,
            analyses[trial.task_id],
            grids[trial.ui_id],
            map_mass_by_aoi(vmap, grids[trial.ui_id]),
            w,
            h,
            g.heatmap_sigma_px,
            g.map_scale,
            c.relevance_threshold,
        )
        align_rows.append(
            {
                "trial_id": trial.trial_id,
                "participant_id": trial.participant_id,
                "task_id": trial.task_id,
                **metrics,
            }
        )

    elements_df = pd.concat(element_rows, ignore_index=True) if element_rows else pd.DataFrame()
    task_df, align_df = pd.DataFrame(task_rows), pd.DataFrame(align_rows)
    elements_df.to_csv(out / "vlm_elements.csv", index=False)
    task_df.to_csv(out / "vlm_task_features.csv", index=False)
    align_df.to_csv(out / "vlm_alignment.csv", index=False)
    np.savez_compressed(out / "vlm_maps.npz", **maps)
    n_ok = int((task_df["vlm_parse_status"] == "ok").sum())
    log.info(
        "VLM analysis: %d/%d tasks parsed ok; alignment for %d trials -> %s",
        n_ok,
        len(task_df),
        len(align_df),
        out,
    )
    if make_figures:
        vlm_figures(cfg, ds, fixations, maps, analyses)
    return {"elements": elements_df, "tasks": task_df, "alignment": align_df}


def task_human_map(cfg, ds, fixations, task_id: str, normalize="sum"):
    """Gaze heatmap pooled over all participants who did ``task_id``."""
    from gazegrounded.gaze.heatmap import fixation_heatmap

    trial_ids = ds.trials.loc[ds.trials["task_id"] == task_id, "trial_id"]
    f = fixations[fixations["trial_id"].isin(trial_ids)]
    ui_id = ds.tasks.set_index("task_id").loc[task_id, "ui_id"]
    w, h = ds.ui_size(ui_id)
    return (
        fixation_heatmap(
            f, w, h, cfg.gaze.heatmap_sigma_px, cfg.gaze.map_scale, normalize=normalize
        ),
        f,
    )


def vlm_figures(cfg, ds, fixations, maps, analyses) -> None:
    from gazegrounded.analysis.saliency_metrics import compare_maps
    from gazegrounded.visualization.comparisons import human_vs_vlm_figure
    from gazegrounded.visualization.gaze import load_image
    from gazegrounded.visualization.style import save_figure
    from gazegrounded.visualization.vlm import vlm_figure

    fig_dir = ensure_dir(cfg.figures_dir() / "vlm")
    for task in ds.tasks.itertuples():
        image = load_image(ds.image_path(task.ui_id))
        title = f"{task.ui_id} · “{task.task_description}”"
        fig = vlm_figure(
            image,
            analyses[task.task_id],
            maps[task.task_id],
            title,
            cfg.comparison.relevance_threshold,
        )
        save_figure(fig, fig_dir / f"vlm_{task.task_id}.png", ds.is_synthetic)
        human, _ = task_human_map(cfg, ds, fixations, task.task_id)
        m = compare_maps(human, maps[task.task_id])
        fig = human_vs_vlm_figure(
            image,
            human,
            maps[task.task_id],
            title,
            {"CC": m["cc"], "SIM": m["sim"], "KL": m["kl"], "attention divergence (JSD)": m["jsd"]},
        )
        save_figure(fig, fig_dir / f"human_vs_vlm_{task.task_id}.png", ds.is_synthetic)
    log.info("VLM figures written to %s", fig_dir)


# ---------------------------------------------------------------------------------------------
# Deep-learning saliency model (baseline C)
# ---------------------------------------------------------------------------------------------


def stage_train_saliency(cfg):
    import shutil

    from gazegrounded.vision.train_saliency import train_saliency

    if cfg.is_synthetic:
        log.warning(
            "Training on synthetic screenshots only tests the code path; "
            "do not interpret the resulting metrics."
        )
    ds = load_processed(cfg)
    run_dir = train_saliency(cfg, ds, load_fixations(cfg))
    fig_dir = ensure_dir(cfg.figures_dir() / "saliency")
    for name in ("loss_curve.png", "val_predictions.png"):
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, fig_dir / name)
    log.info("Saliency run saved to %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------------------------


def stage_nlp_analysis(cfg):
    from gazegrounded.nlp.evaluation import run_nlp_analysis

    ds = load_processed(cfg)
    if "feedback_category" not in ds.trials or ds.trials["feedback_category"].notna().sum() == 0:
        log.warning("No coded feedback categories: skipping the classifier (exploratory only).")
        return None
    run_dir, preds = run_nlp_analysis(cfg, ds.trials)
    # Out-of-fold predictions for the dashboard (never used as model inputs).
    preds.to_csv(ensure_dir(cfg.interim_dir()) / "nlp_predictions.csv", index=False)
    return run_dir


# ---------------------------------------------------------------------------------------------
# Multimodal friction model
# ---------------------------------------------------------------------------------------------


def stage_build_features(cfg) -> pd.DataFrame:
    from gazegrounded.multimodal.features import build_feature_table

    ds = load_processed(cfg)
    df = build_feature_table(cfg, ds)
    out = ensure_dir(cfg.processed_dir())
    df.to_csv(out / "features.csv", index=False)
    log.info(
        "Feature table: %d trials x %d columns -> %s", len(df), df.shape[1], out / "features.csv"
    )
    return df


def stage_train_friction(cfg):
    from gazegrounded.multimodal.evaluation import run_friction_experiment

    df = stage_build_features(cfg)
    if df["friction_label"].notna().sum() == 0:
        raise ValueError("No friction labels available; the friction model needs labelled trials")
    run_dir = run_friction_experiment(cfg, df)
    log.info("Friction experiment saved to %s", run_dir)
    return run_dir


# ---------------------------------------------------------------------------------------------
# Research analyses + report
# ---------------------------------------------------------------------------------------------


def stage_evaluate(cfg):
    """RQ1 baselines, RQ2 statistics, RQ3/RQ4 tables, failure cases, Markdown report."""
    from gazegrounded.analysis import research
    from gazegrounded.analysis.reporting import (
        experiment_block,
        format_test,
        markdown_table,
        write_report,
    )
    from gazegrounded.utils import save_json_safe

    ds = load_processed(cfg)
    fixations = load_fixations(cfg)
    features_path = cfg.processed_dir() / "features.csv"
    features = pd.read_csv(features_path) if features_path.exists() else stage_build_features(cfg)
    vlm_dir = cfg.interim_dir() / "vlm"
    vlm_tasks = pd.read_csv(vlm_dir / "vlm_task_features.csv")
    elements = pd.read_csv(vlm_dir / "vlm_elements.csv")
    aoi_metrics = pd.read_csv(cfg.interim_dir() / "aoi_metrics.csv")
    out = ensure_dir(cfg.reports_dir() / cfg.data.source)
    syn = cfg.is_synthetic
    provider = vlm_tasks["provider"].iloc[0] if len(vlm_tasks) else "unknown"

    per_task, summary, rq1_meta = research.attention_baselines(
        cfg, ds, fixations, load_vlm_maps(cfg)
    )
    per_task.to_csv(out / "rq1_attention_baselines_per_task.csv", index=False)
    summary.to_csv(out / "rq1_attention_baselines_summary.csv", index=False)
    rq2 = research.rq2_statistics(cfg, features)
    rq2.to_csv(out / "rq2_statistics.csv", index=False)
    comp, ablation, friction_run = research.friction_results(cfg)
    failures = research.failure_cases(ds, features, vlm_tasks, elements, aoi_metrics)
    failures.to_csv(out / "failure_cases.csv", index=False)
    save_json_safe(
        {
            "rq1": rq1_meta,
            "friction_run": friction_run,
            "vlm_provider": provider,
            "data_kind": cfg.data.data_kind,
        },
        out / "evaluate_meta.json",
    )
    _evaluate_figures(cfg, ds, fixations, per_task, features, failures)

    # ---- report ------------------------------------------------------------------------
    n_trials, n_part = len(features), features["participant_id"].nunique()
    dataset = (
        f"{cfg.data.source} ({cfg.data.data_kind}); {n_part} participants, "
        f"{n_trials} trials, {ds.uis.shape[0]} interfaces, {ds.tasks.shape[0]} tasks. "
        f"VLM provider: `{provider}`."
    )
    if provider == "mock":
        dataset += (
            " The mock provider is a lexical stand-in that reads AOI annotations; it "
            "is not a vision-language model."
        )
    s = summary.copy()
    s = s[["baseline", "n", "cc_mean", "sim_mean", "kl_mean", "nss_mean", "jsd_mean"]]
    paired = rq1_meta.get("paired_D_vs_A_cc") or {}
    rq1_result = markdown_table(s) + (
        f"\n\nVLM (D) vs centre bias (A), CC across tasks: {format_test(paired)} " "(exploratory)."
        if paired.get("n")
        else ""
    )
    sections = [
        "## RQ1 — Human / VLM attention alignment",
        "",
        experiment_block(
            "Task-aware VLM relevance vs task-agnostic baselines",
            {
                "Hypothesis": "The task-conditioned VLM Task-Relevance Map (D) aligns better with "
                "human gaze than centre bias (A), spectral-residual saliency (B) and a "
                "task-agnostic deep saliency model (C).",
                "Independent variables": "Attention map source (A, B, C, D).",
                "Dependent variables": "CC, SIM, KL, NSS, JSD between each map and the pooled "
                "human gaze heatmap of each task.",
                "Dataset": dataset,
                "Split strategy": "No training for A, B, D. C is evaluated only on screenshots "
                "held out from its training run (pair-aware split).",
                "Model": "See baselines; D from the configured VLM provider.",
                "Metrics": "CC and SIM (higher = better), KL and JSD (lower = better), NSS "
                "(0 = chance).",
                "Result": rq1_result,
                "Limitation": "Pooled task maps hide individual differences; small number of "
                "tasks; NSS/CC depend on heatmap sigma.",
            },
            syn,
        ),
    ]

    shown = rq2[
        rq2["role"].isin(["primary", "secondary"])
        & ~((rq2["family"] == "rq2_correlation") & (rq2["method"] == "spearman_participant_means"))
    ]
    corr_lines = [
        f"- [{r.role}] `{r.x}` vs `{r.y}`: {format_test(r._asdict())}"
        for r in shown.itertuples()
        if r.family == "rq2_correlation"
    ]
    pair_lines = [
        f"- [{r.role}] `{r.y}` (friction − clean, participant means): "
        f"{format_test(r._asdict())}"
        for r in shown.itertuples()
        if r.family == "rq2_paired"
    ]
    sections += [
        "## RQ2 — Attention divergence and UX friction",
        "",
        experiment_block(
            "Divergence vs behavioural friction indicators",
            {
                "Hypothesis": "Trials with larger human/VLM attention divergence show more "
                "behavioural friction (longer completion time, more revisits, later first "
                "fixation on the target); friction variants show larger divergence than clean "
                "variants.",
                "Independent variables": "Primary: attention_divergence_early (map JSD over the "
                "first K fixations). Secondary: whole-trial attention_divergence and "
                "aoi_attention_divergence (whole-trial divergence is confounded by fixation "
                "count). Interface condition (clean vs friction).",
                "Dependent variables": ", ".join(research.RQ2_OUTCOMES) + ".",
                "Dataset": dataset,
                "Split strategy": "Not applicable (no model fitting). Repeated-measures "
                "correlation handles trials nested in participants; paired tests use one mean per "
                "participant and condition.",
                "Model": f"rmcorr; paired {cfg.stats.paired_test} test on participant means; Holm "
                "correction over the primary tests of each family. Trial-level Spearman is "
                "exploratory only (in rq2_statistics.csv).",
                "Metrics": "r / rho with 95% CI, p, Holm-adjusted p, effect sizes.",
                "Result": "\n".join(corr_lines + pair_lines) or "n/a",
                "Limitation": "Correlational; divergence depends on the VLM, the heatmap sigma and "
                "AOI definitions; completion time and gaze measures share variance by design.",
            },
            syn,
        ),
    ]

    if comp is not None:
        table = comp[
            [
                "group_by",
                "model",
                "modality",
                "macro_f1",
                "macro_f1_fold_sd",
                "quadratic_weighted_kappa",
                "roc_auc_ovr_macro",
            ]
        ]
        abl_text = (
            markdown_table(ablation[["dropped_modality", "macro_f1", "quadratic_weighted_kappa"]])
            if ablation is not None and len(ablation)
            else "n/a"
        )
        sections += [
            "## RQ3 / RQ4 — Multimodal friction prediction",
            "",
            experiment_block(
                "Single modalities vs early fusion",
                {
                    "Hypothesis": "Combining gaze, interaction, VLM and text features predicts "
                    "friction level better than any single modality.",
                    "Independent variables": "Feature set; model type.",
                    "Dependent variables": "Friction label (low / medium / high).",
                    "Dataset": dataset + f" Label source: "
                    f"{', '.join(map(str, features['label_source'].dropna().unique()))}.",
                    "Split strategy": "GroupKFold by participant (new people) and by interface "
                    "(new screens); no participant/interface in train and test of the same fold.",
                    "Model": "Logistic regression and random forest (balanced class weights); "
                    "majority-class reference.",
                    "Metrics": "Macro-F1 (primary), quadratic weighted kappa, ROC-AUC (OvR).",
                    "Result": markdown_table(table) + "\n\nLeave-one-modality-out ablation "
                    "(multimodal LR, participant split):\n\n"
                    + abl_text
                    + f"\n\nRun: `experiments/{friction_run}`.",
                    "Limitation": "Small N; friction labels are a single rating per trial; "
                    "UI-level features can act as interface identifiers.",
                },
                syn,
            ),
        ]

    fail_text = (
        "No failure cases: the VLM's top element matched a target AOI for every task."
        if failures.empty
        else markdown_table(
            failures[
                [
                    "task_id",
                    "target_aois",
                    "vlm_top_element",
                    "vlm_top_relevance",
                    "human_dwell_share_on_target",
                    "human_dwell_share_on_vlm_top",
                ]
            ]
        )
    )
    sections += [
        "## Failure cases",
        "",
        "Tasks where the VLM's highest-weighted element is not an annotated target "
        "AOI. Figures: `assets/figures/<source>/failures/`.",
        "",
        fail_text,
        "",
    ]
    report = write_report(
        out / "report.md", f"GazeGrounded UX — {cfg.data.source} report", sections, syn
    )
    log.info("Report written to %s", report)
    return report


def _evaluate_figures(cfg, ds, fixations, per_task, features, failures) -> None:
    from gazegrounded.visualization.comparisons import (
        baseline_comparison_figure,
        divergence_scatter,
    )
    from gazegrounded.visualization.style import save_figure

    fig_dir = ensure_dir(cfg.figures_dir() / "evaluation")
    syn = cfg.is_synthetic
    for metric, higher in (("cc", True), ("kl", False)):
        s = per_task.groupby("baseline")[metric].agg(["mean", "min", "max", "count"]).reset_index()
        s = s.rename(columns={"count": "n"})
        fig = baseline_comparison_figure(
            s, metric, "RQ1 · Which map best matches human gaze?", higher
        )
        save_figure(fig, fig_dir / f"rq1_baselines_{metric}.png", syn)
    for y in ("completion_time_ms", "revisit_count"):
        fig = divergence_scatter(
            features,
            ["attention_divergence_early", "attention_divergence"],
            y,
            f"RQ2 · Divergence vs {y.replace('_', ' ')} — early window (primary) vs whole "
            "trial (secondary, confounded by fixation count)",
        )
        save_figure(fig, fig_dir / f"rq2_divergence_vs_{y}.png", syn)
    if not failures.empty:
        _failure_figures(cfg, ds, fixations, failures)


def _failure_figures(cfg, ds, fixations, failures) -> None:
    from gazegrounded.visualization.gaze import draw_aois, draw_heatmap, load_image, show_screenshot
    from gazegrounded.visualization.style import plt, save_figure
    from gazegrounded.visualization.vlm import draw_vlm_boxes

    analyses = load_vlm_analyses(cfg)
    fig_dir = ensure_dir(cfg.figures_dir() / "failures")
    for f in failures.itertuples():
        image = load_image(ds.image_path(f.ui_id))
        h, w = image.shape[:2]
        human, _ = task_human_map(cfg, ds, fixations, f.task_id, normalize="max")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
        show_screenshot(axes[0], image, dim=0.3)
        draw_vlm_boxes(
            axes[0], analyses[f.task_id], w, h, cfg.comparison.relevance_threshold, max_labels=3
        )
        targets = f.target_aois.split(";")
        ui_aois = ds.aois_for(f.ui_id)
        draw_aois(axes[0], ui_aois[ui_aois["aoi_id"].isin(targets)], w, h, targets=targets)
        axes[0].set_title("VLM elements (blue) vs annotated target (dashed)")
        show_screenshot(axes[1], image, dim=0.35)
        draw_heatmap(axes[1], human, w, h)
        axes[1].set_title(
            f"Human gaze — {100 * f.human_dwell_share_on_target:.0f}% of dwell " "on target"
        )
        fig.suptitle(
            f"Failure case · {f.ui_id} · “{f.task}” — VLM top: “{f.vlm_top_element}”",
            x=0.01,
            ha="left",
            fontsize=12,
            color="#0b0b0b",
        )
        fig.tight_layout()
        save_figure(fig, fig_dir / f"failure_{f.task_id}.png", cfg.is_synthetic)


def stage_run_all(cfg):
    """Full pipeline. For the synthetic demo this starts by generating the data."""
    if cfg.data.adapter == "synthetic":
        stage_generate_demo(cfg)
    stage_preprocess(cfg)
    stage_gaze_analysis(cfg)
    stage_vlm_analysis(cfg)
    stage_train_saliency(cfg)
    stage_nlp_analysis(cfg)
    stage_train_friction(cfg)
    return stage_evaluate(cfg)


def stage_build_study(cfg):
    """Render study stimuli and tasks.json into study/webgazer/."""
    from gazegrounded.data.study_builder import build_study

    return build_study(cfg.root() / "study" / "webgazer")


def stage_export_site(cfg):
    """Export data and images for the static website in docs/.

    The website contains trial-level gaze data. Exporting real participant data is refused
    unless GAZEGROUNDED_ALLOW_REAL_EXPORT=1 is set, because publishing it must be covered by
    the participants' consent.
    """
    import os

    from gazegrounded.web_export import export_site

    if not cfg.is_synthetic and os.environ.get("GAZEGROUNDED_ALLOW_REAL_EXPORT") != "1":
        raise PermissionError(
            "export-site would publish participant-level gaze data. Only do this if your "
            "consent form allows it, then set GAZEGROUNDED_ALLOW_REAL_EXPORT=1."
        )
    return export_site(cfg)
