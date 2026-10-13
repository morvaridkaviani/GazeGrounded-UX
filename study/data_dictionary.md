# Data dictionary

## A. Browser export (`gazegrounded-study-session`, format v1)

One JSON file per session, produced by `study/webgazer/study.js`.

| Field | Type | Description |
|---|---|---|
| `format`, `format_version` | str, int | `"gazegrounded-study-session"`, `1` |
| `study_version` | str | version of `tasks.json` used |
| `participant_code` | str | pseudonymous code, `[A-Za-z0-9_-]{2,12}` |
| `session_id` | str | random 16-hex id generated in the browser |
| `group` | `A`/`B` | counterbalancing group |
| `consented_at` | ISO datetime | when consent was given |
| `webgazer_version` | str | `3.5.3` |
| `environment.viewport/screen` | {width, height} | CSS pixels |
| `environment.device_pixel_ratio` | float | |
| `calibration.attempts` | int | number of calibrations |
| `calibration.validations[]` | list | per attempt: `mean_accuracy_px`, `mean_precision_px`, `valid_ratio`, per-point results |
| `trials[]` | list | see below |

**Trial**

| Field | Description |
|---|---|
| `trial_id` | `<code>_T<order>` |
| `order` | 1-based position in this session |
| `task_id`, `ui_id`, `condition`, `task_description` | from `tasks.json` |
| `image_rect` | on-screen position/size of the stimulus (CSS px) |
| `natural_size` | stimulus pixel size (1440 × 900) |
| `completion_time_ms` | stimulus onset → end of trial |
| `task_success` | true if a target AOI was clicked |
| `end_reason` | `target_clicked` · `gave_up` · `timeout` |
| `error_count` | clicks not on a target AOI |
| `click_count` | all clicks |
| `clicks[]` | `t` (ms), `x`, `y` (image px), `aoi_id` (smallest containing AOI or null), `hit_target` |
| `ease_rating` | SEQ 1–7 |
| `feedback_text` | optional comment or null |
| `quality` | `n_samples`, `valid_ratio` |
| `gaze[]` | `t` (ms since onset), `x`, `y` (image px; null if no estimate), `valid` (0/1) |

## B. Standardised tables (`data/processed/<source>/`)

Defined and validated in `src/gazegrounded/data/schemas.py`.

**uis** — `ui_id`, `image_path` (relative to the dataset root), `width`, `height`, `domain`,
`condition` (clean · friction · single · free_viewing), `pair_id`.

**aois** — `ui_id`, `aoi_id`, `label` (visible text; null for icons), `x_min`, `y_min`,
`x_max`, `y_max` (normalised 0–1), `semantic_role`, `description`.

**tasks** — `task_id`, `ui_id`, `task_key` (shared by both variants), `task_description`,
`target_aoi_ids` (`;`-separated), `designed_friction_type`.

**trials** — `participant_id`, `session_id`, `trial_id`, `ui_id`, `task_id`,
`task_description`, `condition`, `trial_order`, `task_success`, `completion_time_ms`,
`error_count`, `click_count`, `feedback_text`, `feedback_category`, `friction_label`
(low · medium · high), `label_source` (human_coded · self_report_seq · synthetic_latent),
`ease_rating`.

**gaze_samples** — `participant_id`, `session_id`, `trial_id`, `ui_id`, `timestamp_ms`,
`gaze_x`, `gaze_y` (screenshot px, NaN when invalid), `validity` (0/1).

**fixations** (`data/interim/<source>/fixations.csv`) — `participant_id`, `trial_id`, `ui_id`,
`fixation_id`, `start_ms`, `end_ms`, `duration_ms`, `centroid_x`, `centroid_y`, `n_samples`,
`aoi_id`.

## C. Derived features (`features.csv`, one row per trial)

| Feature | Modality | Definition |
|---|---|---|
| `ttff_ms` / `ttff_censored_ms` | gaze | start of first fixation on a target AOI; censored at trial duration if never |
| `target_fixated` | gaze | target AOI fixated at least once |
| `fixation_count`, `mean_fixation_duration_ms`, `fixation_rate_per_s` | gaze | |
| `revisit_count` | gaze | Σ over AOIs of max(visits − 1, 0) |
| `aoi_transition_count` | gaze | changes between consecutive AOI visits |
| `scanpath_length_norm` | gaze | summed saccade amplitude / screen diagonal |
| `task_relevant_attention` | gaze | share of fixation time on target AOIs |
| `task_relevant_aoi_coverage` | gaze | share of target AOIs fixated |
| `completion_time_ms`, `click_count`, `error_count`, `task_success` | interaction | |
| `vlm_human_similarity` | VLM×gaze | SIM (histogram intersection) of gaze map and VLM map |
| `attention_divergence` | VLM×gaze | JSD (base 2) of whole-trial gaze map vs VLM map |
| `attention_divergence_early` | VLM×gaze | JSD using the first 8 fixations after the onset fixation (primary RQ2 measure) |
| `aoi_attention_divergence` | VLM×gaze | JSD of AOI dwell distribution vs VLM mass per AOI |
| `vlm_relevant_dwell_share` | VLM×gaze | share of fixation time inside boxes the VLM rated relevant |
| `task_relevant_region_count`, `max_task_relevance`, `mean_task_relevance` | VLM | interface-level, no human data |
| `text_p_<category>`, `has_feedback` | text | out-of-fold category probabilities (computed inside CV) |
| `ui_aoi_count`, `ui_edge_density`, `ui_saliency_entropy` | visual | screenshot descriptors |
