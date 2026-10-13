"""Validate and standardise raw data.

Thin wrapper around `python -m gazegrounded.cli preprocess`; accepts the same options, e.g.
`python scripts/preprocess_data.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["preprocess", *sys.argv[1:]]))
