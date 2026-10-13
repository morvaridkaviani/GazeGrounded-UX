"""Build features and compare friction models.

Thin wrapper around `python -m gazegrounded.cli train-friction`; accepts the same options, e.g.
`python scripts/train_friction_model.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["train-friction", *sys.argv[1:]]))
