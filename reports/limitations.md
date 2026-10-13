# Limitations, privacy and ethics

## Scientific limitations

**Gaze is not attention, and attention is not understanding.** Eye tracking measures overt
visual attention. People attend covertly, look without processing and process without
fixating. A long dwell can mean interest, confusion or reading.

**Webcam gaze is coarse.** WebGazer errors are typically 100–200 px (several degrees), with
lower and irregular sampling, drift and sensitivity to lighting, glasses and head movement.
Lab trackers reach ~0.5°. Consequences here: larger I-DT thresholds (merged fixations), padded
AOIs (neighbouring elements blur together), and weaker gaze features. Small AOIs such as icons
cannot be analysed reliably with webcam data.

**The VLM map is an estimate, not ground truth.** VLMs can misread icons, miss small or
low-contrast text, confuse similar buttons, cannot see content hidden behind menus or below
the fold, and are often confident when wrong. Results depend on the model, its version, the
prompt and coordinate conventions. Disagreement between gaze and the VLM can therefore mean
(a) UX friction, (b) a VLM error, or (c) noisy gaze. The design can only separate these with
failure-case analysis and controlled clean/friction pairs.

**Divergence measures have their own biases.** Whole-trial JSD between a sparse gaze map and
a smooth VLM map is inflated for short trials (few fixations). This was found while
validating the pipeline, before any real data existed, which is why the early-window
divergence is the primary measure. All map metrics depend on the heatmap σ and grid
resolution; KL heavily penalises predictions that put near-zero mass where people looked.

**Small samples.** Tens of participants and a handful of interfaces mean wide confidence
intervals, unstable feature importances and limited generalisation to other interfaces. The
interface-grouped split is the honest test and will be weak.

**Labels.** Friction is not directly observable. Coded labels depend on coders and the
coding scheme; self-reported ease (SEQ) measures perceived difficulty, a different construct.

**Ecological validity.** Static screenshots with a single click replace multi-step,
scrollable, live interfaces; participants know they are in a study; manipulations are
deliberately clear.

**Synthetic data.** The demo generator encodes the hypotheses. Anything “found” in it was put
there. It only validates the software.

## Privacy

* Gaze is behavioural data that can reveal attention, reading ability, fatigue and possibly
  health-related traits. Treat it as sensitive.
* The study never records webcam video or images; WebGazer runs in the browser.
  `saveDataAcrossSessions` is disabled so no calibration data persists.
* Participants are identified by codes only; names, e-mails and IP addresses are not collected.
  The validation layer rejects tables with PII-looking columns.
* Data stays in the participant's browser until they download it and send it. Nothing is
  uploaded automatically.
* Never send participant data or confidential screenshots to externally hosted models. In this design
  the VLM only sees the fictitious stimuli.
* Retention, access and deletion follow the consent form; withdrawal is possible at any time
  before analysis.

## Bias and fairness

Webcam eye tracking works less well with poor lighting, reflective glasses, some eye shapes
and darker skin tones under poor lighting, and on low-quality cameras. Quality-based
exclusions can therefore systematically remove some groups. Report exclusions by group where
possible, and do not generalise beyond the sample. VLMs carry the biases of their training
data (e.g. Western, English UI conventions).

## Responsible framing

Allowed: “The model estimates task-relevant interface regions.” “Gaze diverged from the
estimated task-relevant regions early in the trial.”
Not allowed: “The model knows what the user is thinking.” “The user was confused because the
model says so.” “The VLM predicts where people will look.”
