"""Train the deep saliency model.

Thin wrapper around `python -m gazegrounded.cli train-saliency`; accepts the same options, e.g.
`python scripts/train_saliency.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["train-saliency", *sys.argv[1:]]))
