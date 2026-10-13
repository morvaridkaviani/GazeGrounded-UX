"""Local Hugging Face vision-language model provider (default: Qwen2.5-VL-3B-Instruct).

Requires ``pip install -e ".[vlm]"`` and enough memory (~8 GB for the 3B model in bf16).
Written against transformers 5.x: ``AutoModelForImageTextToText`` + ``AutoProcessor`` with
``processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")``.

Qwen2.5-VL is trained to ground objects with absolute pixel boxes of the image it receives.
We therefore resize the screenshot ourselves to dimensions that are multiples of the model's
28-pixel patch grid (so the processor does not resize it again), ask for pixel coordinates, and
normalise by the resized size in the parser.

Other models use other conventions (e.g. Qwen3-VL emits 0-1000 relative coordinates). The
parser's ``coordinate_space="auto"`` heuristic handles common cases, but always inspect a few
raw responses (saved under ``data/interim/<source>/vlm/raw``) when switching models.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from gazegrounded.utils import get_logger, resolve_device
from gazegrounded.vlm.base import VLMProvider

log = get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def smart_resize(width: int, height: int, factor: int = 28, max_pixels: int = 1280 * 28 * 28):
    """Largest size <= ``max_pixels`` with both sides multiples of ``factor`` (aspect kept)."""
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    w = max(factor, round(width * scale / factor) * factor)
    h = max(factor, round(height * scale / factor) * factor)
    return w, h


class LocalHFProvider(VLMProvider):
    name = "local"
    prompt_coordinates = "pixels"

    def __init__(self, device: str = "auto", max_pixels: int = 1280 * 28 * 28, **kwargs):
        kwargs["model"] = kwargs.get("model") or DEFAULT_MODEL
        super().__init__(**kwargs)
        self.device_request = device
        self.max_pixels = max_pixels
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('Local VLM needs: pip install -e ".[vlm]"') from exc
        device = resolve_device(self.device_request)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        log.info(
            "Loading %s on %s (%s) - first run downloads the weights", self.model, device, dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.model)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model, dtype=dtype).to(
            device
        )
        self._model.eval()
        self._device = device

    def _image_size_for_prompt(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as im:
            return smart_resize(*im.size, max_pixels=self.max_pixels)

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        import torch

        self._load()
        size = self._image_size_for_prompt(image_path)
        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
