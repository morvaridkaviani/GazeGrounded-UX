"""Optional HTTP provider for a VLM served behind a chat-completions API.

Typical use: a *local* Ollama, vLLM or LM Studio server running an open vision-language model
(e.g. ``qwen2.5vl``). Configure ``VLM_API_BASE_URL`` (and ``VLM_API_KEY`` only if the server
needs one) in ``.env``. Uses only the Python standard library.

Privacy: the screenshot and task text are sent to the configured endpoint. Only use endpoints
you control for anything other than the fictitious study stimuli.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path

from gazegrounded.vlm.base import VLMProvider


def _b64(image_path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return media_type, base64.standard_b64encode(image_path.read_bytes()).decode("ascii")


class HTTPChatProvider(VLMProvider):
    """``POST {base_url}/chat/completions`` with the screenshot as a base64 data URL."""

    name = "http"
    prompt_coordinates = "normalized"

    def __init__(self, base_url: str | None = None, timeout_s: float = 300.0, **kwargs):
        super().__init__(**kwargs)
        if not self.model:
            raise ValueError("Set vlm.model to the model name served by your endpoint")
        self.base_url = (
            base_url or os.environ.get("VLM_API_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self.api_key = os.environ.get("VLM_API_KEY", "")
        self.timeout_s = timeout_s

    def _generate(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        media_type, data = _b64(image_path)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
