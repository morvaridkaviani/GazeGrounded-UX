# Study kit

Everything needed to collect real data for GazeGrounded UX with a webcam-based eye tracker.

| File | Purpose |
|---|---|
| [protocol.md](protocol.md) | Design, hypotheses, procedure, sample size, exclusion rules, analysis plan |
| [consent_template.md](consent_template.md) | Information sheet and consent form (adapt to your institution) |
| [participant_instructions.md](participant_instructions.md) | What participants read before starting |
| [task_templates.md](task_templates.md) | The 16 tasks, interfaces and friction manipulations |
| [questionnaire.md](questionnaire.md) | Post-task and post-session questions |
| [data_dictionary.md](data_dictionary.md) | Every exported and standardised field |
| `webgazer/` | Browser study (WebGazer.js 3.5.3), stimuli and `tasks.json` |

## Running the browser study locally

```bash
python -m gazegrounded.cli build-study        # (re)renders stimuli + tasks.json
cd study/webgazer && python -m http.server 8000 --bind 127.0.0.1
# open http://localhost:8000 in Chrome or Edge (camera access requires localhost or https)
```

Optional URL parameter `?group=A` or `?group=B` forces the counterbalancing group; otherwise
it is derived from the participant code.

For remote sessions, host the folder on any static HTTPS host. No server-side code exists:
data never leaves the participant's browser until they download the export and send it to you.

## Data quality

The study measures and exports, per session and per trial:

* **Validation accuracy** — mean distance (px) between the gaze estimate and 5 known targets
  after calibration. Threshold 200 px (configurable in `tasks.json`); up to 3 attempts.
* **Precision** — RMS spread of gaze samples around their mean at each validation target.
* **Valid-sample ratio** — share of gaze callbacks with a face/gaze estimate.

The analysis pipeline flags trials with < 60 % valid samples. Exclusion rules are in the
protocol and must be applied as written.

## Important limitations of webcam gaze

WebGazer estimates gaze from a consumer webcam with a regression model trained during
calibration. Expect accuracy errors of roughly 100–200 px (several degrees of visual angle),
lower and irregular sampling rates (often 15–30 Hz), drift over time, and sensitivity to head
movement and lighting. It is suitable for *coarse* questions (which region of the screen? did
attention reach the target area early?), not for fine-grained reading or small-AOI analyses.
Lab eye trackers are typically accurate to about 0.5°. Interpret results accordingly and
consider a lab replication.

## Licence note

WebGazer.js is GPL-3.0-or-later (with an LGPL option for small companies, see its README). The
study loads it from the jsDelivr CDN; it is not redistributed in this repository.
