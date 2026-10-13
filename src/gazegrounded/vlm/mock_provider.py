"""Deterministic mock "VLM" for tests and the zero-cost demo.

THIS IS NOT A VISION-LANGUAGE MODEL. It never looks at pixels. It reads the AOI annotations of
the screenshot (label + semantic description) and scores each element by IDF-weighted word
overlap with the task, plus small deterministic noise and box jitter. It emits raw JSON text
(sometimes wrapped in markdown fences) so the real parsing path is exercised.

Because it uses human-written AOI descriptions, it behaves like an *idealised* grounding model:
it can find a target even when the visible label is ambiguous. Results obtained with it say
nothing about how real VLMs perform.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gazegrounded.utils import stable_int_hash
from gazegrounded.vlm.base import VLMProvider

_STOP = set(
    [
        "a",
        "an",
        "the",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "and",
        "or",
        "is",
        "it",
        "its",
        "this",
        "that",
        "your",
        "you",
        "have",
        "has",
        "be",
        "can",
        "with",
        "as",
        "by",
        "from",
        "out",
        "how",
        "much",
        "what",
        "which",
        "where",
        "when",
        "there",
        "their",
        "our",
        "my",
        "me",
        "i",
        "do",
        "does",
        "are",
        "was",
        "were",
    ]
)
_SYNONYMS = {
    "edit": "change",
    "modify": "change",
    "update": "change",
    "changed": "change",
    "block": "freeze",
    "lock": "freeze",
    "freezes": "freeze",
    "padlock": "freeze",
    "promo": "discount",
    "coupon": "discount",
    "code": "discount",
    "luggage": "baggage",
    "bag": "baggage",
    "cancel": "cancel",
    "cancellation": "cancel",
    "see": "find",
    "show": "find",
    "view": "find",
    "dates": "date",
    "spending": "limit",
}
_ROLE_PRIOR = {"button": 0.15, "link": 0.1, "input": 0.12, "icon": 0.1, "menu": 0.1}


def _stem(w: str) -> str:
    """Tiny suffix stripper: 'freezes'/'freeze' -> 'freez', 'changed'/'change' -> 'chang'."""
    for suffix in ("ing", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(_SYNONYMS.get(w, w)) for w in words if w not in _STOP and len(w) >= 3}


def _text(value) -> str:
    return value if isinstance(value, str) else ""


class MockVLMProvider(VLMProvider):
    name = "mock"
    prompt_coordinates = "normalized"

    def __init__(
        self,
        uis: pd.DataFrame,
        aois: pd.DataFrame,
        seed: int = 42,
        bbox_jitter: float = 0.01,
        relevance_noise: float = 0.05,
        min_relevance: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault("model", "mock-lexical-v1")
        kwargs["model"] = kwargs["model"] or "mock-lexical-v1"
        super().__init__(**kwargs)
        self.uis = uis
        self.aois = aois
        self.seed = seed
        self.bbox_jitter = bbox_jitter
        self.relevance_noise = relevance_noise
        self.min_relevance = min_relevance

    def _ui_for_image(self, image_path: Path) -> str:
        stems = {
            Path(p).stem: u for u, p in zip(self.uis["ui_id"], self.uis["image_path"], strict=True)
        }
        if image_path.stem not in stems:
            raise KeyError(f"Mock provider has no AOI annotations for {image_path.name}")
        return stems[image_path.stem]

    def score_elements(self, ui_id: str, task: str) -> pd.DataFrame:
        aois = self.aois[self.aois["ui_id"] == ui_id].copy()
        descriptions = aois["description"] if "description" in aois else aois["label"]
        texts = [
            f"{_text(lbl)} {_text(desc)}"
            for lbl, desc in zip(aois["label"], descriptions, strict=True)
        ]
        docs = [_tokens(t) for t in texts]
        n = len(docs)
        idf = {}
        for d in docs:
            for w in d:
                idf[w] = idf.get(w, 0) + 1
        idf = {w: math.log((1 + n) / (1 + df)) + 1 for w, df in idf.items()}
        task_tokens = _tokens(task)
        raw = np.array([sum(idf[w] for w in task_tokens & d) for d in docs], dtype=float)
        # Relative to the best match on this screen: a grounding model ranks candidates.
        aois["lexical_score"] = raw / raw.max() if raw.max() > 0 else raw
        return aois

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        task = re.search(r'The user\'s task is: "(.*)"', user_prompt).group(1)
        ui_id = self._ui_for_image(image_path)
        scored = self.score_elements(ui_id, task)
        elements = []
        for row in scored.itertuples():
            rng = np.random.default_rng(stable_int_hash(str(self.seed), task, row.aoi_id))
            role_prior = _ROLE_PRIOR.get(row.semantic_role, 0.0)
            relevance = float(
                np.clip(
                    0.85 * row.lexical_score**0.8
                    + role_prior * row.lexical_score
                    + rng.normal(0, self.relevance_noise),
                    0,
                    1,
                )
            )
            if relevance < self.min_relevance:
                continue
            priority = float(
                np.clip(
                    0.75 * relevance
                    + 0.25 * (0.5 + role_prior)
                    + rng.normal(0, self.relevance_noise),
                    0,
                    1,
                )
            )
            j = rng.normal(0, self.bbox_jitter, size=4)
            box = np.clip(
                [row.x_min + j[0], row.y_min + j[1], row.x_max + j[2], row.y_max + j[3]], 0, 1
            )
            elements.append(
                {
                    "label": _text(row.label) or row.aoi_id,
                    "semantic_role": row.semantic_role,
                    "task_relevance": round(relevance, 3),
                    "expected_attention_priority": round(priority, 3),
                    "bbox": {
                        "x_min": round(box[0], 4),
                        "y_min": round(box[1], 4),
                        "x_max": round(box[2], 4),
                        "y_max": round(box[3], 4),
                    },
                    "reason": f"Annotated as: {_text(row.description)}"[:200],
                }
            )
        elements.sort(key=lambda e: -e["task_relevance"])
        body = json.dumps({"task": task, "elements": elements[:8]}, indent=2)
        if stable_int_hash(task, ui_id) % 3 == 0:  # exercise the fence-stripping parser path
            return f"Here is the analysis:\n```json\n{body}\n```"
        return body
