"""Evaluation of saliency predictions with the shared map metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gazegrounded.analysis.saliency_metrics import compare_maps
from gazegrounded.vision.saliency_model import SaliencyNet, logits_to_distribution


@torch.no_grad()
def predict(model: SaliencyNet, loader: DataLoader, device: str) -> dict[int, np.ndarray]:
    """Predicted saliency distributions (sum to 1) keyed by dataset index."""
    model.eval()
    out = {}
    for images, _, _, idx in loader:
        probs = logits_to_distribution(model(images.to(device))).cpu().numpy()
        for i, p in zip(idx.tolist(), probs, strict=True):
            out[i] = p[0]
    return out


def evaluate(model: SaliencyNet, loader: DataLoader, device: str) -> pd.DataFrame:
    """Per-item CC, SIM, KL, JSD and NSS of predictions against the human target maps."""
    preds = predict(model, loader, device)
    ds = loader.dataset
    rows = []
    for i, pred in preds.items():
        target, mask = ds.target(i)
        r, c = np.nonzero(mask[0].numpy())
        m = compare_maps(target[0].numpy(), pred, r, c)
        rows.append({"item_id": ds.items[i].item_id, **m})
    return pd.DataFrame(rows)


def summarize(per_item: pd.DataFrame) -> dict[str, float]:
    metrics = ["cc", "sim", "kl", "jsd", "nss"]
    return {f"{m}_mean": float(per_item[m].mean()) for m in metrics if m in per_item} | {
        "n_items": int(len(per_item))
    }
