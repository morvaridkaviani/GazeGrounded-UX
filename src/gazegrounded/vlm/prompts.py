"""Versioned prompt templates for task-conditioned UI grounding.

The prompt asks about *task relevance* (which elements matter for completing this task and how
much), not merely "where would a user look". Bump ``PROMPT_VERSION`` whenever the wording
changes: the version is part of the cache key, so old cached responses are never reused for a
new prompt.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a careful UX analyst. You inspect screenshots of user interfaces and reason about "
    "which visible interface elements are relevant to a specific user task. You answer only "
    "with valid JSON."
)

_EXAMPLE = {
    "task": "<the task>",
    "elements": [
        {
            "label": "Edit address",
            "semantic_role": "button",
            "task_relevance": 0.94,
            "expected_attention_priority": 0.9,
            "bbox": {"x_min": 0.61, "y_min": 0.29, "x_max": 0.79, "y_max": 0.35},
            "reason": "This control directly allows the user to modify the delivery address.",
        }
    ],
}

_COORDS = {
    "normalized": (
        "Give each bbox as fractions of the image width and height between 0 and 1 "
        "(x_min, y_min, x_max, y_max), with (0, 0) at the top-left corner."
    ),
    "pixels": (
        "Give each bbox in absolute pixel coordinates of the image, which is {width} x {height} "
        "pixels (x_min, y_min, x_max, y_max), with (0, 0) at the top-left corner."
    ),
}

USER_TEMPLATE = """The user's task is: "{task}"

Analyze the user interface with respect to this task. Identify the visible interface elements
that are relevant to completing the task (at most {max_elements}). For each element estimate:
- task_relevance: how relevant it is for completing the task (0 = irrelevant, 1 = essential)
- expected_attention_priority: how much of a user's attention it should receive while pursuing
  this task (0 = none, 1 = the main focus)
and give its bounding box.

{coordinate_instruction}

Return ONLY a JSON object with exactly this structure (no markdown, no commentary):
{example}

Rules:
- Scores are numbers between 0 and 1.
- Include relevant elements even if they are small, faint or poorly labelled.
- Leave out elements that are irrelevant to the task.
- "semantic_role" is one of: button, link, input, text, heading, image, navigation, banner,
  region, icon, menu.
- "reason" is one short sentence."""


def build_prompt(
    task: str,
    coordinate_space: str = "normalized",
    width: int | None = None,
    height: int | None = None,
    max_elements: int = 8,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for a task."""
    coords = _COORDS[coordinate_space].format(width=width, height=height)
    user = USER_TEMPLATE.format(
        task=task.strip(),
        max_elements=max_elements,
        coordinate_instruction=coords,
        example=json.dumps(_EXAMPLE, indent=2),
    )
    return SYSTEM_PROMPT, user
