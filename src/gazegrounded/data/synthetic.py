"""Synthetic demo dataset generator.

SYNTHETIC DEMO DATA — NOT RESEARCH EVIDENCE.

Purpose: make the full pipeline runnable and testable at zero cost. The generator encodes
*assumed* patterns (friction -> longer search, more revisits, slower completion, more negative
feedback) so that the outputs are interpretable. Any "finding" computed on this data merely
recovers what was put in here and must never be reported as a result.

Generative model (per trial)
----------------------------
1. A latent friction severity ``s`` in [0, 1] = designed task friction (from the UI spec)
   + participant susceptibility + trial noise.
2. Gaze: a sequence of fixations — visual search over non-target AOIs (count grows with ``s``,
   AOIs chosen in proportion to visual salience), dwell on the target AOI(s), and
   "confusion" revisits that leave and return to the target. Raw samples are produced at
   ``sampling_rate_hz`` with per-fixation accuracy offsets, per-sample precision noise,
   saccade interpolation, blinks and random drop-outs.
3. Interaction: completion time follows the gaze timeline; errors and failure grow with ``s``.
4. Feedback: a templated sentence whose category usually (not always) matches the designed
   friction type; some feedback is missing.
5. Label: ``friction_label`` = ``s`` + annotator noise binned to low/medium/high
   (``label_source = "synthetic_latent"``). This stands in for a human rating; it is not a
   function of any single observed modality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gazegrounded.data.loader import StandardizedDataset
from gazegrounded.data.schemas import FEEDBACK_CATEGORIES
from gazegrounded.data.ui_render import save_ui
from gazegrounded.data.ui_specs import SYNTHETIC_UIS, Element, TaskSpec, UISpec
from gazegrounded.utils import SYNTHETIC_BANNER, ensure_dir, get_logger, set_seed

log = get_logger(__name__)

# Thresholds used to bin the latent severity into the demo label.
LABEL_THRESHOLDS = (0.35, 0.6)

_TASK_PHRASES: dict[str, dict[str, str]] = {
    "change_address": {
        "action": "change the address",
        "thing": "address option",
        "label": "Shipping preferences",
    },
    "apply_discount": {
        "action": "enter the code",
        "thing": "discount field",
        "label": "Have a promo?",
    },
    "freeze_card": {
        "action": "freeze the card",
        "thing": "freeze button",
        "label": "the padlock icon",
    },
    "find_limit": {"action": "see the limit", "thing": "spending limit", "label": "card details"},
    "change_date": {
        "action": "change the date",
        "thing": "date change option",
        "label": "Manage booking",
    },
    "find_baggage": {
        "action": "check my baggage",
        "thing": "baggage allowance",
        "label": "fare conditions",
    },
}

_FEEDBACK_TEMPLATES: dict[str, list[str]] = {
    "findability": [
        "I couldn't find where to {action} at first.",
        "It took me ages to find the {thing}, it was hidden at the bottom.",
        "Hard to find. I looked all over the page before spotting it.",
        "Where is the option to {action}? I had to search the whole page.",
        "The {thing} was not where I expected it to be.",
    ],
    "visual_hierarchy": [
        "The important information was in tiny grey text.",
        "Everything looked equally important so I missed the {thing}.",
        "The text was too small and faint to notice.",
        "I only noticed the {thing} after reading everything else.",
    ],
    "ambiguous_wording": [
        "The label didn't make it clear what it does.",
        "I wasn't sure '{label}' meant I could {action}.",
        "The wording was confusing, I didn't know which option to pick.",
        "Unclear label, I had to guess what it would do.",
    ],
    "information_overload": [
        "Too much text on the page, the {thing} was buried in it.",
        "Way too much information, I had to read a wall of small print.",
        "So many details at once that I got lost.",
    ],
    "feedback_uncertainty": [
        "I'm not sure whether it actually worked.",
        "There was no clear confirmation after I clicked.",
    ],
    "navigation": [
        "I had to dig through a menu to {action}.",
        "The option to {action} was hidden inside another menu.",
        "I expected a direct button, not a dropdown.",
    ],
    "no_friction": [
        "Easy, the {thing} was right there.",
        "Straightforward, no problems.",
        "It was obvious where to click.",
        "Quick and clear.",
        "No issues, I found it immediately.",
    ],
    "other": [
        "The banner was distracting.",
        "Fine, but the page looked busy.",
        "I was a bit tired, otherwise ok.",
        "Not sure, it was okay I guess.",
    ],
}


@dataclass
class Participant:
    participant_id: str
    susceptibility: float  # shifts latent friction
    duration_mult: float  # individual fixation-duration tendency
    precision_mult: float  # tracking precision (higher = noisier)
    invalid_rate: float


@dataclass
class _Fixation:
    aoi_id: str | None
    x: float
    y: float
    duration_ms: float


def _make_participants(n: int, base_invalid: float, rng: np.random.Generator) -> list[Participant]:
    people = []
    for i in range(n):
        people.append(
            Participant(
                participant_id=f"P{i + 1:02d}",
                susceptibility=float(rng.normal(0, 0.12)),
                duration_mult=float(np.clip(rng.normal(1.0, 0.1), 0.75, 1.3)),
                precision_mult=float(np.clip(rng.lognormal(0, 0.25), 0.6, 2.0)),
                invalid_rate=base_invalid,
            )
        )
    if n >= 7:
        # One participant with poor tracking quality, to exercise the data-quality checks.
        people[6].invalid_rate = 0.45
        people[6].precision_mult = 2.2
    return people


def _point_in(el: Element, w: int, h: int, rng: np.random.Generator, spread: float = 0.35):
    x0, y0, x1, y1 = el.box
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    return (
        float(cx + rng.uniform(-spread, spread) * (x1 - x0) * w),
        float(cy + rng.uniform(-spread, spread) * (y1 - y0) * h),
    )


def _fix_duration(mean: float, rng: np.random.Generator) -> float:
    shape = 3.5
    return float(np.clip(rng.gamma(shape, mean / shape), 80, 1500))


def _simulate_fixations(
    spec: UISpec, task: TaskSpec, s: float, p: Participant, rng: np.random.Generator
) -> tuple[list[_Fixation], bool]:
    """Return the fixation plan for one trial and whether the task succeeded."""
    w, h = spec.width, spec.height
    aois = {e.id: e for e in spec.aoi_elements()}
    targets = [aois[t] for t in task.target_aoi_ids]
    distractors = [e for e in aois.values() if e.id not in task.target_aoi_ids]
    weights = np.array([e.salience**1.5 + 0.02 for e in distractors])
    weights = weights / weights.sum()
    base_dur = p.duration_mult * (200 + 110 * s)
    reading_bonus = (
        150 if task.friction_type in ("information_overload", "visual_hierarchy") else 60
    )

    fixations: list[_Fixation] = []

    def search_fixation() -> None:
        if rng.random() < 0.12:  # glance at non-AOI background
            fixations.append(
                _Fixation(
                    None,
                    rng.uniform(0.05, 0.95) * w,
                    rng.uniform(0.1, 0.9) * h,
                    _fix_duration(base_dur, rng),
                )
            )
        else:
            el = distractors[rng.choice(len(distractors), p=weights)]
            x, y = _point_in(el, w, h, rng)
            fixations.append(_Fixation(el.id, x, y, _fix_duration(base_dur, rng)))

    def target_fixation(el: Element) -> None:
        x, y = _point_in(el, w, h, rng)
        fixations.append(_Fixation(el.id, x, y, _fix_duration(base_dur + reading_bonus * s, rng)))

    # Initial central fixation (stimulus onset / centre bias).
    fixations.append(
        _Fixation(
            None,
            w * rng.normal(0.5, 0.03),
            h * rng.normal(0.45, 0.03),
            _fix_duration(base_dur, rng),
        )
    )
    for _ in range(rng.poisson(3 + 14 * s**1.3)):
        search_fixation()

    failed = rng.random() < 0.02 + 0.45 * s**3
    if failed:
        if rng.random() < 0.5:
            target_fixation(targets[0])  # looked at it but did not recognise it
        for _ in range(rng.poisson(5)):
            search_fixation()
        return fixations, False

    for el in targets:
        target_fixation(el)
        for _ in range(rng.poisson(0.7 + 1.5 * s)):
            target_fixation(el)
    for _ in range(rng.poisson(4 * s**2)):
        search_fixation()  # leave the target ...
        target_fixation(targets[-1])  # ... and come back (revisit)
    return fixations, True


def _samples_for_trial(
    fixations: list[_Fixation],
    p: Participant,
    rate_hz: int,
    accuracy_px: float,
    precision_px: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Turn a fixation plan into raw gaze samples (t, x, y, validity) and the trial end time."""
    dt = 1000.0 / rate_hz
    segments: list[tuple[float, float, float, float, float, float]] = []  # t0,t1,x0,y0,x1,y1
    t = 0.0
    prev = None
    for f in fixations:
        ox, oy = rng.normal(0, accuracy_px, size=2)
        fx, fy = f.x + ox, f.y + oy
        if prev is not None:
            dist = float(np.hypot(fx - prev[0], fy - prev[1]))
            sacc = 25 + dist / 40
            segments.append((t, t + sacc, prev[0], prev[1], fx, fy))
            t += sacc
        segments.append((t, t + f.duration_ms, fx, fy, fx, fy))
        t += f.duration_ms
        prev = (fx, fy)
    end = t
    times = np.arange(0.0, end, dt)
    xs = np.empty_like(times)
    ys = np.empty_like(times)
    seg_idx = 0
    for i, ti in enumerate(times):
        while seg_idx < len(segments) - 1 and ti >= segments[seg_idx][1]:
            seg_idx += 1
        t0, t1, x0, y0, x1, y1 = segments[seg_idx]
        a = 0.0 if t1 == t0 else min(max((ti - t0) / (t1 - t0), 0.0), 1.0)
        xs[i] = x0 + a * (x1 - x0)
        ys[i] = y0 + a * (y1 - y0)
    noise = precision_px * p.precision_mult
    xs += rng.normal(0, noise, size=len(times))
    ys += rng.normal(0, noise, size=len(times))
    valid = rng.random(len(times)) >= p.invalid_rate
    for _ in range(rng.poisson(end / 1000 * 0.25)):  # blinks ~ every 4 s
        start = rng.uniform(0, end)
        valid &= ~((times >= start) & (times < start + rng.uniform(100, 250)))
    xs[~valid] = np.nan
    ys[~valid] = np.nan
    return times, np.round(xs, 1), np.round(ys, 1), valid.astype(int), end


def _feedback(task: TaskSpec, s: float, rng: np.random.Generator) -> tuple[str, str]:
    designed = task.friction_type
    friction_cats = [c for c in FEEDBACK_CATEGORIES if c not in ("no_friction", "other")]
    if designed == "no_friction":
        designed = "findability" if s > 0.45 else "no_friction"
    if designed == "no_friction":
        category = "no_friction" if rng.random() < 0.85 else "other"
    else:
        r = rng.random()
        if s < LABEL_THRESHOLDS[0]:
            category = "no_friction" if r < 0.6 else designed
        elif r < 0.72:
            category = designed
        elif r < 0.87:
            category = "other"
        else:
            category = str(rng.choice(friction_cats))
    template = str(rng.choice(_FEEDBACK_TEMPLATES[category]))
    phrases = _TASK_PHRASES.get(
        task.task_key, {"action": "do it", "thing": "option", "label": "the label"}
    )
    return template.format(**phrases), category


def _label(s: float, rng: np.random.Generator) -> str:
    noisy = s + rng.normal(0, 0.05)
    if noisy < LABEL_THRESHOLDS[0]:
        return "low"
    if noisy < LABEL_THRESHOLDS[1]:
        return "medium"
    return "high"


def _static_tables(specs: tuple[UISpec, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uis, aois, tasks = [], [], []
    for spec in specs:
        uis.append(
            {
                "ui_id": spec.ui_id,
                "image_path": f"screenshots/{spec.ui_id}.png",
                "width": spec.width,
                "height": spec.height,
                "domain": spec.domain,
                "condition": spec.condition,
                "pair_id": spec.pair_id,
            }
        )
        for e in spec.aoi_elements():
            aois.append(
                {
                    "ui_id": spec.ui_id,
                    "aoi_id": e.id,
                    "label": e.label.split("\n")[0][:60],
                    "x_min": e.box[0],
                    "y_min": e.box[1],
                    "x_max": e.box[2],
                    "y_max": e.box[3],
                    "semantic_role": e.role,
                    "description": e.description,
                }
            )
        for t in spec.tasks:
            tasks.append(
                {
                    "task_id": f"{spec.ui_id}__{t.task_key}",
                    "ui_id": spec.ui_id,
                    "task_key": t.task_key,
                    "task_description": t.description,
                    "target_aoi_ids": ";".join(t.target_aoi_ids),
                    "designed_friction_type": t.friction_type,
                }
            )
    return pd.DataFrame(uis), pd.DataFrame(aois), pd.DataFrame(tasks)


def generate_synthetic_dataset(cfg, out_dir: str | Path | None = None) -> StandardizedDataset:
    """Generate the synthetic demo dataset and write it (standard schema) to ``out_dir``."""
    sc = cfg.synthetic
    out = ensure_dir(out_dir or cfg.raw_dir())
    rng = set_seed(cfg.seed)
    specs = SYNTHETIC_UIS
    for spec in specs:
        save_ui(spec, out / "screenshots" / f"{spec.ui_id}.png", watermark="SYNTHETIC DEMO")
    uis, aois, tasks = _static_tables(specs)
    participants = _make_participants(sc.n_participants, sc.invalid_sample_rate, rng)

    trials, samples, latent = [], [], []
    pairs = [(spec, task) for spec in specs for task in spec.tasks]
    for p in participants:
        order = rng.permutation(len(pairs))
        for k, idx in enumerate(order, start=1):
            spec, task = pairs[idx]
            trial_id = f"{p.participant_id}_T{k:02d}"
            s = float(
                np.clip(task.design_friction + p.susceptibility + rng.normal(0, 0.15), 0.02, 0.98)
            )
            fixations, success = _simulate_fixations(spec, task, s, p, rng)
            t, x, y, v, end = _samples_for_trial(
                fixations, p, sc.sampling_rate_hz, sc.gaze_noise_px, sc.gaze_precision_px, rng
            )
            errors = int(rng.poisson(2.2 * s**1.5)) + (0 if success else 1)
            completion = end + 250.0 * (errors + int(success))  # motor time per click
            text, category = _feedback(task, s, rng)
            if rng.random() < sc.missing_feedback_rate:
                text, category = None, None
            trials.append(
                {
                    "participant_id": p.participant_id,
                    "session_id": f"{p.participant_id}_S1",
                    "trial_id": trial_id,
                    "ui_id": spec.ui_id,
                    "task_id": f"{spec.ui_id}__{task.task_key}",
                    "task_description": task.description,
                    "condition": spec.condition,
                    "trial_order": k,
                    "task_success": bool(success),
                    "completion_time_ms": round(completion, 1),
                    "error_count": errors,
                    "click_count": errors + int(success),
                    "feedback_text": text,
                    "feedback_category": category,
                    "friction_label": _label(s, rng),
                    "label_source": "synthetic_latent",
                    "ease_rating": int(np.clip(round(7 - 5.5 * s + rng.normal(0, 0.6)), 1, 7)),
                }
            )
            samples.append(
                pd.DataFrame(
                    {
                        "participant_id": p.participant_id,
                        "session_id": f"{p.participant_id}_S1",
                        "trial_id": trial_id,
                        "ui_id": spec.ui_id,
                        "timestamp_ms": np.round(t, 2),
                        "gaze_x": x,
                        "gaze_y": y,
                        "validity": v,
                    }
                )
            )
            latent.append(
                {
                    "trial_id": trial_id,
                    "latent_severity": round(s, 4),
                    "n_planned_fixations": len(fixations),
                }
            )

    ds = StandardizedDataset(
        source="synthetic",
        data_kind="synthetic_demo",
        root=out,
        uis=uis,
        aois=aois,
        tasks=tasks,
        trials=pd.DataFrame(trials),
        gaze_samples=pd.concat(samples, ignore_index=True),
    )
    ds.save(out)
    # Generator internals, kept only for debugging/tests. Never used as a model input.
    pd.DataFrame(latent).to_csv(out / "_generator_latent.csv", index=False)
    (out / "README.txt").write_text(
        f"{SYNTHETIC_BANNER}\n\nGenerated by gazegrounded.data.synthetic with seed={cfg.seed}.\n"
        "Regenerate with: python -m gazegrounded.cli generate-demo\n"
    )
    log.info(
        "Synthetic dataset: %d participants, %d trials, %d gaze samples -> %s",
        len(participants),
        len(trials),
        sum(len(s) for s in samples),
        out,
    )
    return ds
