# Model card — UX friction model

## Model details

* **What:** classifiers predicting a trial's UX friction level (low / medium / high) from
  gaze, interaction, VLM-alignment and feedback-text features.
* **Variants:** logistic regression (multinomial, balanced class weights, standardised
  features) — the primary, interpretable model; random forest (300 trees, depth ≤ 6,
  balanced subsamples); a majority-class reference.
* **Implementation:** `src/gazegrounded/multimodal/` (scikit-learn pipelines; early fusion).
* **Version:** 0.1.0. **Status:** research prototype; **not validated on real data.**

## Intended use

* Research on whether multimodal signals, including human/VLM attention divergence, help
  identify interface friction (RQ3/RQ4).
* Triage in UX research: pointing researchers to trials/screens worth inspecting, together
  with the evidence (gaze, divergence map, comments).

## Not intended for

* Judging, ranking or monitoring **people** (participants, employees, users). The unit is an
  interface-task interaction, not a person.
* Automated design decisions without human review.
* Inferring cognitive states, emotions, abilities or health from gaze.
* Production use on live users' webcams.

## Inputs

| Modality | Features |
|---|---|
| Gaze | TTFF on target (censored), target fixated, fixation count/duration/rate, revisits, AOI transitions, scanpath length, task-relevant attention and coverage |
| Interaction | completion time, clicks, errors, success |
| VLM | relevance statistics of the grounded elements; human/VLM similarity; whole-trial, early-window and AOI-level divergence; dwell share in VLM-relevant regions |
| Text | predicted feedback-category probabilities (learned inside each training fold) |

## Output

A friction level with class probabilities; for logistic regression also per-trial feature
contributions (standardised value × coefficient).

## Training data

* **Currently:** synthetic demo data only (12 simulated participants × 10 tasks, labels from a
  latent severity). Performance numbers on it are meaningless as evidence.
* **Planned:** the WebGazer study (≥ 24 participants × 9 tasks), labels coded by two
  researchers blind to condition.

## Evaluation

GroupKFold by participant (new people) and by interface (new screens); macro-F1 (primary),
weighted F1, quadratic weighted kappa, ROC-AUC (one-vs-rest), fold SDs; majority-class reference;
leave-one-modality-out ablation; permutation importance on held-out folds.
**Real-data results: pending.**

## Limitations and failure modes

* **Small data:** high variance between folds; random forests overfit easily.
* **Interface identity:** with few interfaces, UI-level features (visual, VLM relevance
  statistics) can identify the screen rather than the friction; check the interface-grouped
  split.
* **Label validity:** friction labels are human judgements (or self-report); different coders or
  scales change the target.
* **Shared variance:** completion time, fixation count and TTFF partly measure the same thing;
  coefficients of correlated features are unstable — read permutation importance alongside.
* **VLM dependence:** VLM features inherit grounding errors and prompt/model choices; a model
  trained with one VLM should not be applied to features from another.
* **Webcam gaze:** noisy gaze weakens gaze and divergence features and can bias results
  against low-quality participants (lighting, glasses, skin tone and camera quality affect
  tracking).

## Ethical considerations

Gaze is sensitive behavioural data; the model should only ever be applied to consenting study
participants on prepared stimuli. Explanations are associations in the training data, not
causal claims about why an interface is hard.
