"""Feedback text classification and topics.

Thin wrapper around `python -m gazegrounded.cli nlp-analysis`; accepts the same options, e.g.
`python scripts/run_nlp_analysis.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["nlp-analysis", *sys.argv[1:]]))
