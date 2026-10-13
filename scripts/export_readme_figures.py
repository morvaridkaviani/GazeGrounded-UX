"""Copy a curated set of synthetic-demo figures into assets/figures/readme/ for the README.

Run after `python -m gazegrounded.cli run-all`. Every copied figure carries the synthetic-data
banner that was stamped on it when it was generated.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "figures" / "synthetic"
DST = ROOT / "assets" / "figures" / "readme"

FIGURES = {
    "vlm/human_vs_vlm_shop_checkout_friction__change_address.png": "human_vs_vlm.png",
    "vlm/vlm_bank_cards_friction__freeze_card.png": "vlm_grounding.png",
    "gaze/aoi_dwell_shop_checkout_friction__change_address.png": "aoi_dwell.png",
    "friction/modality_comparison_participant_id.png": "modality_comparison.png",
    "friction/feature_importance_permutation.png": "feature_importance.png",
    "friction/confusion_matrix_multimodal.png": "confusion_matrix.png",
    "evaluation/rq1_baselines_cc.png": "rq1_baselines_cc.png",
    "evaluation/rq2_divergence_vs_completion_time_ms.png": "rq2_divergence.png",
    "nlp/confusion_matrix.png": "nlp_confusion_matrix.png",
    "saliency/val_predictions.png": "saliency_predictions.png",
}


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    missing = []
    for src, dst in FIGURES.items():
        if (SRC / src).exists():
            shutil.copy(SRC / src, DST / dst)
        else:
            missing.append(src)
    gaze = sorted((SRC / "gaze").glob("trial_*.png"))
    if gaze:
        shutil.copy(gaze[0], DST / "trial_gaze.png")
    failures = sorted((SRC / "failures").glob("failure_*.png"))
    if failures:
        shutil.copy(failures[-1], DST / "failure_case.png")
    for m in missing:
        print(f"missing (run the pipeline first): {m}")
    print(f"Copied figures to {DST}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
