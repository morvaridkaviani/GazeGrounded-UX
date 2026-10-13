# Convenience targets. Every target calls the same CLI that is documented in the README.
PY ?= python
CONFIG ?= configs/synthetic.yaml
CLI = $(PY) -m gazegrounded.cli --config $(CONFIG)

.PHONY: install lint format test demo generate preprocess gaze vlm saliency nlp friction evaluate dashboard site serve-site clean-demo

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	black --check src tests scripts dashboard

format:
	ruff check --fix .
	black src tests scripts dashboard

test:
	pytest

generate:
	$(CLI) generate-demo

preprocess:
	$(CLI) preprocess

gaze:
	$(CLI) gaze-analysis

vlm:
	$(CLI) vlm-analysis

saliency:
	$(CLI) train-saliency

nlp:
	$(CLI) nlp-analysis

friction:
	$(CLI) train-friction

evaluate:
	$(CLI) evaluate

# Full synthetic pipeline, zero cost, CPU-friendly (~1-3 minutes on a laptop).
demo:
	$(CLI) run-all

dashboard:
	streamlit run dashboard/app.py

# Static website (GitHub Pages) in docs/: refresh its data, then preview it locally.
site:
	$(CLI) export-site

serve-site:
	$(PY) -m http.server 8000 --directory docs

clean-demo:
	rm -rf data/synthetic/* data/interim/synthetic data/processed/synthetic assets/figures/synthetic
