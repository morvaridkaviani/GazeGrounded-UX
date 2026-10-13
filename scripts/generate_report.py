"""Research analyses (RQ1-RQ4) and the Markdown report.

Thin wrapper around `python -m gazegrounded.cli evaluate`; accepts the same options, e.g.
`python scripts/generate_report.py --config configs/synthetic.yaml --set seed=7`.
"""

import sys

from gazegrounded.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
