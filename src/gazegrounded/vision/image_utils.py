"""Image loading and tensor conversion for the saliency model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image as RGB (format detected from content, not the file extension)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def image_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize to ``size`` = (width, height) and normalise with ImageNet statistics -> (3, H, W)."""
    resized = image.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`image_to_tensor` for display -> (H, W, 3) uint8."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (t.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)
