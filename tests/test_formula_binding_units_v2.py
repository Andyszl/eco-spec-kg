"""Output bindings may remove an explicitly recorded unit, never a qualifier."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from test_candidate_selection_repairs_v2 import output_binding_unit, predict, prompt
from ecospec_kg.extractor_v2 import _llm_select


@pytest.mark.parametrize("definition,unit_text", [
    ("冠层覆盖率，%；", "%"),
    ("冠层覆盖率, %;", "%"),
    ("冠层覆盖率，m3/a；", "m3/a"),
])
def test_explicit_unit_suffix_preserves_exact_output_binding(definition, unit_text):
    unit = output_binding_unit()
    unit["variable_definitions"][0].update(definition=definition, unit=unit_text)
    candidate = predict(unit)
    before_unit, before_candidate = deepcopy(unit), deepcopy(candidate)
    bindings = prompt(unit, candidate)["formula_output_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["definition"] == definition
    assert unit == before_unit and candidate == before_candidate


@pytest.mark.parametrize("definition,unit_text", [
    ("冠层覆盖率，%；", ""),
    ("冠层覆盖率，%；", "m3/a"),
    ("像元冠层覆盖率，%；", "%"),
    ("冠层覆盖率，根据观测数据计算；", ""),
    ("冠层覆盖率，单位，%，根据观测数据计算；", "根据观测数据计算"),
])
def test_binding_does_not_remove_qualifiers_or_unverified_suffixes(definition, unit_text):
    unit = output_binding_unit()
    unit["variable_definitions"][0].update(definition=definition, unit=unit_text)
    assert prompt(unit)["formula_output_bindings"] == []


def test_unit_suffix_binding_still_requires_output_and_both_candidate_edges():
    unit = output_binding_unit()
    unit["variable_definitions"][0].update(definition="冠层覆盖率，%；", unit="%")
    candidate = predict(unit)
    assert len(prompt(unit, candidate)["formula_output_bindings"]) == 1
    for relation_type in ("has_output", "calculated_by"):
        missing = deepcopy(candidate)
        missing["relations"] = [r for r in missing["relations"]
                                if r["relation_type"] != relation_type]
        assert prompt(unit, missing)["formula_output_bindings"] == []
    unit["variable_definitions"][0]["definition_origin"] = "equation_structure_transcription"
    assert prompt(unit, candidate)["formula_output_bindings"] == []


def test_unit_suffix_binding_does_not_select_an_unreturned_edge():
    unit = output_binding_unit()
    unit["variable_definitions"][0].update(definition="冠层覆盖率，%；", unit="%")
    candidate = predict(unit)
    assert prompt(unit, candidate)["formula_output_bindings"]
    relation = next(r for r in candidate["relations"]
                    if r["head_name"] == "Y" and r["relation_type"] == "calculated_by")
    provider = MagicMock()
    provider.complete.return_value = json.dumps({"selected_entity_ids": [],
                                                "selected_relation_ids": [relation["relation_id"]]})
    selected, _, _ = _llm_select(provider, unit, candidate)
    assert selected["relations"] == [relation]


@pytest.mark.parametrize("unit_id", ["e10dfb07c04c0214", "f8f74534ea13a463"])
def test_real_train_binding_pairs_are_already_positive_in_frozen_gold(unit_id):
    delivery = Path(__file__).resolve().parents[1] / "deliveries/human_confirmed_20261002"
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    unit = next(row for row in load(delivery / "blind/train_units.jsonl") if row["unit_id"] == unit_id)
    gold = next(row for row in load(delivery / "gold/train_annotations.jsonl") if row["unit_id"] == unit_id)
    candidate = predict(unit)
    bindings = prompt(unit, candidate)["formula_output_bindings"]
    assert len(bindings) == 1
    gold_edges = {(r["head_name"], r["relation_type"], r["tail_name"]) for r in gold["relations"]}
    by_id = {r["relation_id"]: r for r in candidate["relations"]}
    for key in ("variable_relation_id", "indicator_relation_id"):
        edge = by_id[bindings[0][key]]
        assert (edge["head_name"], edge["relation_type"], edge["tail_name"]) in gold_edges
