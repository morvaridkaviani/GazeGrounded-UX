# Experiment template

Every analysis in this project is reported in this structure (the generated report fills it in
automatically; see `analysis/reporting.py::experiment_block`). Keep confirmatory and
exploratory analyses in separate sections.

```markdown
### <Experiment title>  ·  confirmatory | exploratory

**Hypothesis.** Stated before looking at the data (for confirmatory analyses: as in study/protocol.md).

**Independent variables.** What varies (conditions, feature sets, model types).

**Dependent variables.** What is measured, with units.

**Dataset.** Source, data kind (synthetic_demo | real), N participants / trials / interfaces,
exclusions applied, label source, VLM provider + model + prompt version.

**Split strategy.** GroupKFold by participant / interface; number of folds; or "no fitting".

**Model.** Exact model and settings (link the run folder: experiments/<timestamp>-<name>/).

**Metrics.** Primary metric first; why these metrics fit the target.

**Result.** Numbers with N, test statistic, p (and Holm-adjusted p), effect size and CI.
No adjectives without numbers.

**Interpretation.** Written by the researcher. What the result does and does not show.
For synthetic data: "Not interpretable — synthetic demo data."

**Limitation.** Threats to validity specific to this experiment.
```
