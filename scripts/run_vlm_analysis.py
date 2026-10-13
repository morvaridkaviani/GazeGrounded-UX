"""Task-conditioned VLM grounding and human/VLM alignment.

Thin wrapper around `python -m gazegrounded.cli vlm-analysis`; accepts the same options, e.g.
`python scripts/run_vlm_analysis.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["vlm-analysis", *sys.argv[1:]]))
