"""Fixations, AOIs, gaze metrics, scanpaths and heatmaps.

Thin wrapper around `python -m gazegrounded.cli gaze-analysis`; accepts the same options, e.g.
`python scripts/run_gaze_analysis.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["gaze-analysis", *sys.argv[1:]]))
