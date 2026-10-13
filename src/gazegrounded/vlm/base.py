"""Provider interface for vision-language models, with response caching.

Subclasses implement :meth:`VLMProvider._generate` (image + prompt -> raw text). The base class
handles prompt construction, caching of raw responses and robust parsing, so every provider
(local Hugging Face model, HTTP-served model, mock) produces the same validated
:class:`~gazegrounded.vlm.schemas.VLMAnalysis`.

Cache key = sha256(image checksum, task, provider, model, prompt version). A cached response is
reused only if all five match, so changing the prompt or model never returns stale output.
Raw responses are cached (not parsed ones) so parser improvements apply retroactively at no cost.
"""

from __future__ import annotations

import abc
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from gazegrounded.utils import ensure_dir, file_sha256, get_logger, text_sha256
from gazegrounded.vlm.grounding import parse_vlm_response
from gazegrounded.vlm.prompts import PROMPT_VERSION, build_prompt
from gazegrounded.vlm.schemas import VLMAnalysis

log = get_logger(__name__)


class VLMProvider(abc.ABC):
    """Base class: ``analyze_ui(image_path, task) -> VLMAnalysis``."""

    name: str = "base"
    #: Coordinate convention requested in the prompt ("normalized" or "pixels").
    prompt_coordinates: str = "normalized"

    def __init__(
        self,
        model: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"Prompt version {prompt_version!r} requested but prompts.py defines "
                f"{PROMPT_VERSION!r}; add the old template before switching versions."
            )

    # ---- to implement ----------------------------------------------------------------------
    @abc.abstractmethod
    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        """Run the model and return its raw text output."""

    def _image_size_for_prompt(self, image_path: Path) -> tuple[int, int]:
        """Image size the model will see (override if the provider resizes the image)."""
        with Image.open(image_path) as im:
            return im.size

    # ---- public API ------------------------------------------------------------------------
    def cache_key(self, image_sha: str, task: str) -> str:
        return text_sha256(image_sha, task.strip(), self.name, str(self.model), self.prompt_version)

    def analyze_ui(self, image_path: str | Path, task: str) -> VLMAnalysis:
        image_path = Path(image_path)
        image_sha = file_sha256(image_path)
        key = self.cache_key(image_sha, task)
        size = self._image_size_for_prompt(image_path)
        raw, from_cache = self._cached(key), True
        if raw is None:
            from_cache = False
            system, user = build_prompt(task, self.prompt_coordinates, *size)
            raw = self._generate(image_path, system, user)
            self._store(key, raw, task, image_sha)
        analysis = parse_vlm_response(
            raw,
            task,
            image_size=size,
            coordinate_space="auto",
            provider=self.name,
            model=self.model,
            prompt_version=self.prompt_version,
        )
        analysis.image_sha256 = image_sha
        analysis.cache_key = key
        analysis.from_cache = from_cache
        if analysis.parse_status != "ok":
            log.warning(
                "VLM parse %s for %r: %s",
                analysis.parse_status,
                task,
                "; ".join(analysis.warnings[:3]),
            )
        return analysis

    def raw_response(self, image_path: str | Path, task: str) -> str | None:
        """Cached raw text for an (image, task) pair, if any."""
        return self._cached(self.cache_key(file_sha256(image_path), task))

    # ---- cache -----------------------------------------------------------------------------
    def _cache_file(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    def _cached(self, key: str) -> str | None:
        if not self.use_cache:
            return None
        p = self._cache_file(key)
        if p.exists():
            return json.loads(p.read_text())["raw"]
        return None

    def _store(self, key: str, raw: str, task: str, image_sha: str) -> None:
        if self.cache_dir is None:
            return
        ensure_dir(self.cache_dir)
        record = {
            "raw": raw,
            "provider": self.name,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "task": task,
            "image_sha256": image_sha,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self._cache_file(key).write_text(json.dumps(record, indent=2))


def make_provider(cfg, ds=None) -> VLMProvider:
    """Instantiate the provider named in ``cfg.vlm.provider``.

    ``ds`` (a StandardizedDataset) is only needed by the mock provider, which reads AOI
    annotations instead of looking at pixels.
    """
    v = cfg.vlm
    common = {
        "model": v.model,
        "prompt_version": v.prompt_version,
        "cache_dir": cfg.root() / cfg.paths.data_root / "interim" / "vlm_cache",
        "use_cache": v.cache,
        "max_new_tokens": v.max_new_tokens,
        "temperature": v.temperature,
    }
    if v.provider == "mock":
        from gazegrounded.vlm.mock_provider import MockVLMProvider

        if ds is None:
            raise ValueError("The mock provider needs the dataset's AOI annotations")
        return MockVLMProvider(
            ds.uis,
            ds.aois,
            seed=cfg.seed,
            bbox_jitter=v.mock_bbox_jitter,
            relevance_noise=v.mock_relevance_noise,
            **common,
        )
    if v.provider == "local":
        from gazegrounded.vlm.local_provider import LocalHFProvider

        return LocalHFProvider(device=v.device, **common)
    if v.provider == "http":
        from gazegrounded.vlm.api_provider import HTTPChatProvider

        return HTTPChatProvider(**common)
    raise ValueError(f"Unknown VLM provider {v.provider!r}")
