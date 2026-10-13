"""Generate the synthetic demo dataset.

Thin wrapper around `python -m gazegrounded.cli generate-demo`; accepts the same options, e.g.
`python scripts/generate_demo_data.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["generate-demo", *sys.argv[1:]]))
