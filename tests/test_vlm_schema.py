import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gazegrounded.vlm.attention_map import element_distribution, vlm_relevance_map
from gazegrounded.vlm.base import VLMProvider
from gazegrounded.vlm.grounding import extract_json, match_elements_to_aois, parse_vlm_response
from gazegrounded.vlm.prompts import PROMPT_VERSION, build_prompt
from gazegrounded.vlm.schemas import BBox, VLMAnalysis, VLMElement

GOOD = {
    "task": "Change the delivery address",
    "elements": [
        {
            "label": "Edit address",
            "semantic_role": "button",
            "task_relevance": 0.94,
            "expected_attention_priority": 0.9,
            "bbox": {"x_min": 0.61, "y_min": 0.29, "x_max": 0.79, "y_max": 0.35},
            "reason": "Directly edits the address.",
        },
        {
            "label": "Address card",
            "semantic_role": "region",
            "task_relevance": 0.6,
            "expected_attention_priority": 0.5,
            "bbox": {"x_min": 0.05, "y_min": 0.2, "x_max": 0.6, "y_max": 0.45},
        },
    ],
}


def test_bbox_validation():
    assert BBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.3).area == pytest.approx(0.02)
    with pytest.raises(ValidationError):
        BBox(x_min=0.5, y_min=0.1, x_max=0.2, y_max=0.3)  # min > max
    with pytest.raises(ValidationError):
        BBox(x_min=0.1, y_min=0.1, x_max=1.2, y_max=0.3)  # not normalised


def test_element_weight():
    el = VLMElement.model_validate(GOOD["elements"][0])
    assert el.weight == pytest.approx(0.94 * 0.9)


def test_parse_clean_json():
    a = parse_vlm_response(json.dumps(GOOD), "Change the delivery address")
    assert a.parse_status == "ok"
    assert len(a.elements) == 2
    assert a.relevant(0.9)[0].label == "Edit address"


def test_parse_fenced_json_with_prose_and_trailing_comma():
    raw = "Sure! Here you go:\n```json\n" + json.dumps(GOOD)[:-2] + ",]}\n```\nHope it helps."
    a = parse_vlm_response(raw, "t")
    assert a.parse_status == "ok" and len(a.elements) == 2


def test_parse_qwen_style_pixel_bbox_list_and_percentages():
    raw = json.dumps(
        [{"label": "Freeze", "bbox_2d": [720, 296, 950, 351], "relevance": 90, "priority": 80}]
    )
    a = parse_vlm_response(raw, "t", image_size=(1440, 900))
    assert a.parse_status == "ok"
    b = a.elements[0].bbox
    assert b.x_min == pytest.approx(0.5) and b.y_max == pytest.approx(0.39)
    assert a.elements[0].task_relevance == pytest.approx(0.9)


def test_parse_per_mille_and_swapped_corners():
    raw = json.dumps(
        {"elements": [{"label": "x", "task_relevance": 0.5, "bbox": [500, 600, 100, 200]}]}
    )
    a = parse_vlm_response(raw, "t")  # no image size -> values >1 read as 0-1000
    b = a.elements[0].bbox
    assert (b.x_min, b.y_min, b.x_max, b.y_max) == pytest.approx((0.1, 0.2, 0.5, 0.6))


def test_parse_python_literal_output():
    raw = "{'elements': [{'label': 'A', 'task_relevance': 0.7, 'bbox': [0.1, 0.1, 0.2, 0.2]}]}"
    assert parse_vlm_response(raw, "t").parse_status == "ok"


def test_malformed_output_never_raises():
    for raw in ["", "I cannot help with that.", "{not json", "[1, 2, 3]", '{"foo": 1}', None]:
        a = parse_vlm_response(raw, "t")
        assert a.parse_status == "failed"
        assert a.elements == []


def test_partial_when_some_elements_invalid():
    bad = {
        "elements": [
            GOOD["elements"][0],
            {"label": "no box", "task_relevance": 0.5},
            {"label": "zero area", "task_relevance": 0.5, "bbox": [0.2, 0.2, 0.2, 0.4]},
        ]
    }
    a = parse_vlm_response(json.dumps(bad), "t")
    assert a.parse_status == "partial"
    assert len(a.elements) == 1
    assert len(a.warnings) == 2


def test_extract_json_prefers_fenced_block():
    assert extract_json('noise {"a": 1} ```json\n{"b": 2}\n```') == {"b": 2}


def test_prompt_is_task_conditioned_and_versioned():
    system, user = build_prompt("Freeze the card", "pixels", 1428, 896)
    assert "Freeze the card" in user and "1428 x 896" in user
    assert "task_relevance" in user and "JSON" in system
    assert PROMPT_VERSION == "v1"


def test_relevance_map_is_normalised_and_peaks_in_relevant_box():
    a = parse_vlm_response(json.dumps(GOOD), "t")
    m = vlm_relevance_map(a, 1440, 900, scale=0.125, blur_sigma_px=None)
    assert m.shape == (112, 180)
    assert m.max() == pytest.approx(1.0) and m.min() >= 0
    r, c = np.unravel_index(m.argmax(), m.shape)
    assert 0.61 <= (c + 0.5) / 180 <= 0.79 and 0.29 <= (r + 0.5) / 112 <= 0.35


def test_element_distribution_modes():
    mass = element_distribution((0.1, 0.1, 0.5, 0.5), (50, 50), mode="mass")
    dens = element_distribution((0.1, 0.1, 0.5, 0.5), (50, 50), mode="density")
    assert mass.sum() == pytest.approx(1.0)
    assert dens.max() == pytest.approx(1.0)


def test_empty_analysis_gives_zero_map():
    m = vlm_relevance_map(VLMAnalysis(task="t", parse_status="failed"), 100, 100, scale=0.5)
    assert m.sum() == 0


def test_match_elements_to_aois():
    aois = pd.DataFrame(
        {
            "aoi_id": ["edit", "card"],
            "x_min": [0.6, 0.05],
            "y_min": [0.28, 0.2],
            "x_max": [0.8, 0.6],
            "y_max": [0.36, 0.45],
        }
    )
    a = parse_vlm_response(json.dumps(GOOD), "t")
    matched = match_elements_to_aois(a, aois)
    assert matched["matched_aoi_id"].tolist() == ["edit", "card"]


class _CountingProvider(VLMProvider):
    name = "counting"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def _generate(self, image_path, system_prompt, user_prompt):
        self.calls += 1
        return json.dumps(GOOD)


def test_cache_prevents_repeated_calls_and_keys_on_prompt_inputs(tmp_path, synthetic_ds):
    image = synthetic_ds.image_path(synthetic_ds.uis["ui_id"].iloc[0])
    p = _CountingProvider(model="m1", cache_dir=tmp_path)
    first = p.analyze_ui(image, "Change the delivery address")
    second = p.analyze_ui(image, "Change the delivery address")
    assert p.calls == 1 and not first.from_cache and second.from_cache
    p.analyze_ui(image, "A different task")
    assert p.calls == 2
    other_model = _CountingProvider(model="m2", cache_dir=tmp_path)
    other_model.analyze_ui(image, "Change the delivery address")
    assert other_model.calls == 1  # model is part of the cache key
    nocache = _CountingProvider(model="m1", cache_dir=tmp_path, use_cache=False)
    nocache.analyze_ui(image, "Change the delivery address")
    assert nocache.calls == 1


def test_mock_provider_end_to_end(tmp_path, synthetic_ds):
    from gazegrounded.vlm.mock_provider import MockVLMProvider

    p = MockVLMProvider(synthetic_ds.uis, synthetic_ds.aois, cache_dir=tmp_path)
    task = synthetic_ds.tasks.set_index("task_id").loc["shop_checkout_clean__change_address"]
    a = p.analyze_ui(synthetic_ds.image_path("shop_checkout_clean"), task["task_description"])
    assert a.parse_status == "ok"
    assert a.elements[0].label == "Edit address"  # deterministic, top-ranked
