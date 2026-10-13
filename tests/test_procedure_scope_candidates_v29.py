"""Procedure scopes must not depend on an assessment-subject branch."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, _llm_select


def unit(text, code="HJ 1167-2021"):
    return {
        "unit_id": "scope-regression-unseen", "unit_type": "procedure_clause",
        "schema_version": "source-unit-v2.0", "clause_text": text,
        "provenance": {
            "standard_code": code, "document_title": "观测技术规范",
            "source_file": "fixture.pdf", "source_sha256": "a" * 64,
            "section": "9.26", "heading_chain": ["9 野外观测", "9.26 叶面积指数"],
            "pages": [10], "printed_pages": ["7"],
            "evidence_spans": [{"span_id": "scope-proof", "page": 10,
                                "bbox": [1, 2, 3, 4], "line_ids": ["line-1"], "text": text}],
        },
    }


def edges(candidate):
    return {(r["head_name"], r["relation_type"], r["tail_name"])
            for r in candidate["relations"]}


@pytest.mark.parametrize("code", ["HJ 1167-2021", "HJ 9999-2099"])
def test_scope_candidate_before_observation_return_and_without_assessment_subject(code):
    source = unit("采用叶面积指数仪器进行测量，然后计算样方平均叶面积指数。", code)
    before = deepcopy(source)
    candidate = RuleCandidateExtractorV2().predict_unit(source)
    assert ("叶面积指数仪器测量", "applies_to_space", "样方") in edges(candidate)
    assert source == before
    relation = next(r for r in candidate["relations"] if r["relation_type"] == "applies_to_space")
    assert relation["evidence_span_ids"] == ["scope-proof"]


def test_actual_training_source_recovers_existing_gold_scope_without_reading_gold():
    data = Path(__file__).resolve().parents[1] / "deliveries/human_confirmed_20261002"
    load = lambda name: [json.loads(s) for s in (data / name).read_text(encoding="utf-8").splitlines()]
    source = next(u for u in load("blind/train_units.jsonl") if u["unit_id"] == "0081047e007c3fed")
    gold = next(g for g in load("gold/train_annotations.jsonl") if g["unit_id"] == source["unit_id"])
    target = ("叶面积指数仪器测量", "applies_to_space", "样地")
    assert target in edges(gold)
    assert target in edges(RuleCandidateExtractorV2().predict_unit(source))


def test_added_candidates_are_not_forced_into_model_selection():
    source = unit("采用叶面积指数仪器进行测量，然后计算样方平均叶面积指数。")
    candidate = RuleCandidateExtractorV2().predict_unit(source)
    assert any(r["relation_type"] == "applies_to_space" for r in candidate["relations"])
    provider = MagicMock()
    provider.complete.return_value = json.dumps({"selected_entity_ids": [], "selected_relation_ids": []})
    selected, _, _ = _llm_select(provider, source, candidate)
    assert selected["entities"] == selected["relations"] == []


@pytest.mark.parametrize("text", ["采用仪器测量并计算平均值。", "采用仪器测量。"])
def test_no_spatial_endpoint_without_source_scope(text):
    candidate = RuleCandidateExtractorV2().predict_unit(unit(text))
    assert not any(e["entity_type"] == "spatial_scope" for e in candidate["entities"])
