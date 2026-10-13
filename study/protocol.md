# Study protocol — GazeGrounded UX

*Version 1.0 · status: draft for ethics review · fill in the bracketed fields.*

## 1. Aim

To test whether the divergence between where people look and where a task-aware
vision-language model (VLM) estimates task-relevant information to be is associated with
behavioural indicators of UX friction (RQ2), and whether combining gaze, interaction, VLM and
text signals predicts friction better than single modalities (RQ3).

## 2. Design

* **Within-participants factor:** interface condition (clean vs friction), varied *across*
  interface pairs. Each participant sees one variant of each pair (counterbalanced groups A/B),
  so no task is performed twice and every participant contributes both conditions.
* **Interfaces:** 4 clean/friction pairs (e-commerce checkout, banking cards, patient portal,
  subscription settings) plus 1 single-version travel booking page — 9 screens, 16 tasks;
  9 tasks per participant. See [task_templates.md](task_templates.md).
* **Manipulations** (one design factor per pair, ethically harmless): hidden or low-contrast
  controls, ambiguous or icon-only labels, competing promotional content, dense small print,
  confirmation shown far from the action.
* **Task order:** randomised per participant (seeded by participant code).

## 3. Hypotheses (confirmatory)

* **H1 (RQ1):** pooled human gaze maps correlate more strongly (CC) with the VLM
  Task-Relevance Map than with a centre-bias baseline.
* **H2a (RQ2):** within participants, early attention divergence
  (`attention_divergence_early`) correlates positively with completion time, revisit count
  and time to first fixation on the target (repeated-measures correlation).
* **H2b (RQ2):** early attention divergence is higher on friction than on clean variants
  (paired test on participant means).
* **H3 (RQ3):** the multimodal model reaches higher macro-F1 than the best single-modality
  model under participant-grouped cross-validation.

All other analyses are exploratory and will be reported as such.

## 4. Participants

* **Target N:** ≥ 24 after exclusions (12 per counterbalancing group). Rationale: a
  within-participant correlation of r = 0.3 across ~9 trials per person yields > 200 trials,
  which gives adequate power for rmcorr. The paired test with N = 24 detects dz ≈ 0.6 at
  80 % power (α = .05, two-sided). [Run a formal power analysis for your final design.]
* **Inclusion:** 18+, normal or corrected-to-normal vision (glasses may reduce webcam accuracy;
  record but do not exclude), laptop/desktop with webcam, Chrome/Edge.
* **Recruitment:** [channel]; compensation: [amount / none].

## 5. Procedure (~15 minutes)

1. Information sheet and consent ([consent_template.md](consent_template.md)).
2. Participant code (issued by the researcher; never a name or e-mail).
3. Set-up instructions; full screen; camera permission.
4. 9-point click calibration (5 clicks per point) → 5-point validation. Recalibrate if mean
   error > 200 px (max. 3 attempts).
5. For each task: instruction → interface → click where you would perform the task, or
   “I can't find it” (timeout 90 s) → ease rating (SEQ, 1–7) + optional comment.
6. End screen: data quality summary; participant downloads the JSON export and sends it via
   [secure channel]; webcam is switched off.

## 6. Measures

* **Gaze:** samples (image pixel coordinates, ms since stimulus onset), fixations (I-DT),
  TTFF on target, dwell, revisits, transitions, early attention divergence.
* **Behaviour:** completion time, success, errors (clicks outside the target), clicks.
* **Self-report:** Single Ease Question (Sauro & Dumas, 2009), free-text feedback.
* **Friction label:** coded by [two] researchers blind to condition, from the recording
  summary (success, time, errors) *and* the comment, on low / medium / high. Inter-rater
  agreement reported as weighted kappa; disagreements resolved by discussion.

## 7. Exclusion rules (fixed in advance)

* Session: final validation accuracy > 300 px, or < 50 % valid samples overall.
* Trial: < 60 % valid gaze samples, or stimulus displayed smaller than 800 px wide.
* Report the number excluded at each step.

## 8. Analysis plan

Implemented in `src/gazegrounded/analysis/research.py` (do not change after data collection):

* RQ1: CC, SIM, KL, NSS of maps A–D vs pooled human maps per task; exploratory Wilcoxon D vs A.
* RQ2: rmcorr (primary, Holm-corrected family of 3 outcomes); paired Wilcoxon on participant
  means for early divergence and completion time (manipulation check). Whole-trial and AOI
  divergence are secondary. Fixed parameters: I-DT 120 px / 100 ms, AOI margin 40 px, heatmap
  σ = 60 px, early window K = 8 fixations after the onset fixation.
* RQ3/RQ4: logistic regression and random forest, GroupKFold by participant (5 folds) and by
  interface; macro-F1 (primary), quadratic weighted kappa; leave-one-modality-out ablation.

## 9. Data management

Pseudonymous codes only; key linking codes to identities (if any) kept separately by the
researcher and deleted after [date]. No video or images are recorded. Data stored on
[encrypted institutional storage], retained for [N years], and never uploaded to third-party
services (including externally hosted models — the VLM only ever sees the fictitious stimulus images).

## 10. Risks

Minimal: brief screen work. Participants may stop at any time. Camera use is explained up
front; the video stream is never recorded.
