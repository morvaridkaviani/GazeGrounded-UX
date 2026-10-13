"""Dataset of (UI screenshot -> human gaze heatmap) pairs for saliency prediction.

One item per screenshot: the target is the duration-weighted fixation heatmap pooled over all
viewers of that screenshot (for task-based studies, pooled over its tasks too). The model is
therefore *task-agnostic*: it predicts where people tend to look on an interface, which is
exactly what makes it a useful baseline against the task-conditioned VLM.

Splitting is by interface group (``pair_id`` if available, else ``ui_id``) so near-duplicate
screenshots (clean vs friction variant of the same page) never straddle train and validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gazegrounded.gaze.heatmap import generate_heatmap, normalize_map
from gazegrounded.vision.image_utils import image_to_tensor, load_rgb


@dataclass
class SaliencyItem:
    item_id: str
    image_path: str
    width: int
    height: int
    fix_x: np.ndarray  # pixel coordinates in the original screenshot
    fix_y: np.ndarray
    fix_w: np.ndarray  # weights (fixation durations)
    group: str


def build_items(ds, fixations: pd.DataFrame) -> list[SaliencyItem]:
    """Pool fixations per screenshot into :class:`SaliencyItem` objects."""
    f = fixations.merge(
        ds.trials[["trial_id", "ui_id"]].rename(columns={"ui_id": "_ui"}), on="trial_id", how="left"
    )
    ui_col = "ui_id" if "ui_id" in f.columns else "_ui"
    items = []
    for ui in ds.uis.itertuples():
        g = f[f[ui_col] == ui.ui_id]
        if g.empty:
            continue
        pair = getattr(ui, "pair_id", None)
        group = pair if isinstance(pair, str) and pair else ui.ui_id
        items.append(
            SaliencyItem(
                item_id=ui.ui_id,
                image_path=str(ds.root / ui.image_path),
                width=int(ui.width),
                height=int(ui.height),
                fix_x=g["centroid_x"].to_numpy(float),
                fix_y=g["centroid_y"].to_numpy(float),
                fix_w=g["duration_ms"].to_numpy(float),
                group=group,
            )
        )
    return items


def split_items(
    items: list[SaliencyItem], val_fraction: float, seed: int
) -> tuple[list[SaliencyItem], list[SaliencyItem]]:
    """Group-aware random split; at least one group in each side when possible."""
    groups = sorted({it.group for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction)) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    train = [it for it in items if it.group not in val_groups]
    val = [it for it in items if it.group in val_groups]
    return train, val


class SaliencyDataset(Dataset):
    """Returns ``(image, target_map, fixation_mask, index)`` tensors at ``image_size``.

    ``target_map`` is max-normalised to [0, 1]; ``fixation_mask`` marks fixated cells (for NSS).
    """

    def __init__(self, items: list[SaliencyItem], image_size: tuple[int, int], sigma_px: float):
        self.items = items
        self.size = image_size  # (width, height)
        self.sigma_px = sigma_px
        self._targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def target(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx not in self._targets:
            it = self.items[idx]
            w, h = self.size
            sx, sy = w / it.width, h / it.height
            pts = np.stack([it.fix_x * sx, it.fix_y * sy], axis=1)
            heat = generate_heatmap(
                pts,
                w,
                h,
                sigma=self.sigma_px * (sx + sy) / 2,
                weights=it.fix_w,
                scale=1.0,
                normalize=None,
            )
            mask = np.zeros((h, w), dtype=np.float32)
            c = np.clip((pts[:, 0]).astype(int), 0, w - 1)
            r = np.clip((pts[:, 1]).astype(int), 0, h - 1)
            mask[r, c] = 1.0
            self._targets[idx] = (
                torch.from_numpy(normalize_map(heat, "max").astype(np.float32))[None],
                torch.from_numpy(mask)[None],
            )
        return self._targets[idx]

    def __getitem__(self, idx: int):
        it = self.items[idx]
        image = image_to_tensor(load_rgb(it.image_path), self.size)
        target, mask = self.target(idx)
        return image, target, mask, idx
