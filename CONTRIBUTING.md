# Contributing

Thanks for your interest. This is a research codebase, so correctness and honesty about
results matter more than features.

## Set-up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check .
```

## Ground rules

1. **Never commit data** — no participant data, no third-party datasets, no API responses
   that licences or privacy forbid sharing. `data/` is git-ignored except its README.
2. **Never commit secrets** — use `.env` (git-ignored); document new variables in
   `.env.example`.
3. **No fabricated results** — figures and tables must come from code in this repository.
   Anything computed on synthetic data must carry the synthetic banner.
4. **Leakage** — any new model must be evaluated with grouped CV
   (`analysis.model_evaluation.grouped_cv_splits`), with every fitted transform inside the
   pipeline.
5. **Pre-specified analyses** — do not change `analysis/research.py` (the RQ2 test families)
   after looking at real data; add exploratory analyses separately and label them.

## Code style

* `black` (line length 100) and `ruff` (`make format`, `make lint`).
* Type hints on public functions; docstrings where the *why* is not obvious.
* Core logic lives in `src/gazegrounded/`; notebooks and scripts only call it.
* Every new metric or transformation gets a test with a hand-checkable expected value.

## Commits

Conventional prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
