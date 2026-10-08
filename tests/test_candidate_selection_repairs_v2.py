"""Source bindings and selector evidence without changing frozen labels or choices."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from test_experiment_chain_v2 import source_unit
from ecospec_kg.extractor_v2 import (
    RuleCandidateExtractorV2,
    _llm_select,
    build_llm_selection_messages,
)


def predict(unit):
    return RuleCandidateExtractorV2().predict_unit(unit)


def edges(candidate):
    return {
        (r["head_name"], r["relation_type"], r["tail_name"])
        for r in candidate["relations"]
    }


def prompt(unit, candidate=None):
    return json.loads(build_llm_selection_messages(unit, candidate or predict(unit))[1])


def table_unit(cells):
    unit = source_unit()
    unit.update(unit_type="table_record", cells=cells, table_title="表 1 评估指标体系")
    unit["provenance"].update(
        standard_code="HJ 1173-2021", document_title="生态系统服务功能评估"
    )
    return unit


@pytest.mark.parametrize("subject,indicator", [
    ("养分循环", "养分转化量"),
    ("河流调节", "洪峰削减率"),
])
def test_indicator_uses_explicit_row_subject_instead_of_document_title(subject, indicator):
    unit = table_unit({"评估科目": subject, "评估指标": indicator})
    candidate = predict(unit)
    assert (subject, "has_indicator", indicator) in edges(candidate)
    assert ("生态系统服务功能", "has_indicator", indicator) not in edges(candidate)
    assert next(e for e in candidate["entities"] if e["name"] == subject)["evidence_span_ids"]


def test_inherited_row_subject_is_used_only_from_current_structured_cells():
    unit = table_unit({"评估科目": "养分循环", "调查评估指标": "养分转化量"})
    unit.update(raw_cells={"评估科目": None, "调查评估指标": "养分转化量"},
                inherited_columns=["评估科目"], inherited_from={"评估科目": "previous-row"})
    assert ("养分循环", "has_indicator", "养分转化量") in edges(predict(unit))


@pytest.mark.parametrize("subject", [None, "", "—", "-", "/"])
def test_missing_row_subject_falls_back_without_creating_placeholder(subject):
    unit = table_unit({"评估科目": subject, "评估指标": "养分转化量"})
    assert ("生态系统服务功能", "has_indicator", "养分转化量") in edges(predict(unit))


def reviewed_formula_unit():
    path = Path(__file__).resolve().parents[1] / "deliveries/human_confirmed_20261002/blind/dev_units.jsonl"
    return next(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["unit_id"] == "3194374504e7f58a")


def test_source_reviewed_formula_alias_recovers_all_inputs_without_mutating_unit():
    unit = reviewed_formula_unit()
    before = deepcopy(unit)
    assert {("公式（B.2）", "has_input", s) for s in ("A", "B", "C", "x")} <= edges(predict(unit))
    payload = prompt(unit)
    formula = next(f for f in payload["source_unit"]["formulas"] if f["formula_number"] == "B.2")
    alias = next(a for a in formula["symbol_aliases"] if a["symbol"] == "c")
    assert alias["canonical_symbol"] == "C"
    assert {"b64ed9b7ba5c122f", "152371b40e9ebfa9"} <= set(alias["evidence_span_ids"])
    assert alias["reason"]
    assert unit == before


@pytest.mark.parametrize("change", ["source", "formula", "expression", "span"])
def test_reviewed_case_alias_cannot_spread_to_other_sources_or_formulas(change):
    unit = reviewed_formula_unit()
    formula = next(f for f in unit["formulas"] if f["formula_number"] == "B.2")
    if change == "source":
        unit["provenance"]["source_sha256"] = "0" * 64
    elif change == "formula":
        formula["formula_number"] = "B.9"
    elif change == "expression":
        formula["expression_text"] = "L = A + Bxc + D"
    else:
        formula["evidence_span"]["span_id"] = "different-span"
    number = formula["formula_number"]
    assert (f"公式（{number}）", "has_input", "C") not in edges(predict(unit))
    assert not next(f for f in prompt(unit)["source_unit"]["formulas"]
                    if f["formula_number"] == number).get("symbol_aliases")


def test_reviewed_alias_requires_both_local_source_spans_and_rejects_conflicts():
    unit = reviewed_formula_unit()
    unit["variable_definitions"] = [v for v in unit["variable_definitions"]
                                    if v["evidence_span"]["span_id"] != "152371b40e9ebfa9"]
    assert ("公式（B.2）", "has_input", "C") not in edges(predict(unit))
    unit = reviewed_formula_unit()
    formula = next(f for f in unit["formulas"] if f["formula_number"] == "B.2")
    formula["symbol_aliases"] = [{"symbol": "c", "canonical_symbol": "D",
                                  "evidence_span_ids": ["b64ed9b7ba5c122f"]}]
    with pytest.raises(ValueError, match="conflicts"):
        predict(unit)


def test_normalized_table_classification_has_structured_body_support():
    unit = table_unit({"级别": "生态系统质量", "优": "EQI≥80", "良": "50≤EQI＜80"})
    unit["table_title"] = "表 1 生态系统质量分级"
    unit["provenance"].update(standard_code="HJ 1172-2021", document_title="生态系统质量评估")
    payload = prompt(unit)
    rule = next(e for e in payload["candidate_entities"] if e["name"] == "优：EQI≥80；良：50≤EQI＜80")
    assert rule["support_scope"] == "body"
    assert "cells" in rule["support_fields"]
    assert rule["id"] not in payload["selection_focus"]["context_only_entity_ids"]


def constraint_unit():
    unit = source_unit()
    unit.update(unit_type="procedure_clause", clause_text=(
        "基于植被模型估算覆盖率，可根据调查范围和所具备的实际条件选择合适的模型和方法。"
    ))
    unit["provenance"].update(standard_code="HJ 1172-2021", section="B.1")
    return unit


def test_normalized_constraint_is_supported_by_its_own_clause():
    payload = prompt(constraint_unit())
    rule = next(e for e in payload["candidate_entities"]
                if e["name"] == "根据调查范围和实际条件选择合适的模型和方法")
    assert rule["support_scope"] == "body"
    assert "clause_text" in rule["support_fields"]
    assert rule["id"] not in payload["selection_focus"]["context_only_entity_ids"]


def test_evidence_span_or_adjacent_text_alone_cannot_upgrade_an_unrelated_rule():
    unit = constraint_unit()
    candidate = predict(unit)
    unit["adjacent_source_text"] = unit.pop("clause_text")
    unit["clause_text"] = "使用植被模型估算覆盖率。"
    payload = prompt(unit, candidate)
    rule = next(e for e in payload["candidate_entities"]
                if e["type"] == "quality_rule")
    assert rule["support_scope"] == "context_only"
    assert "adjacent_source_text" not in payload["source_unit"]


def output_binding_unit():
    unit = source_unit()
    unit["provenance"].update(standard_code="HJ 1172-2021", section="B.8",
                              document_title="植被覆盖评估", heading_chain=["B.8 冠层覆盖率"])
    unit["formulas"][0]["formula_number"] = "B.8"
    unit["formulas"][0]["expression_text"] = "Y = K * X"
    span = unit["variable_definitions"][0]["evidence_span"]
    unit["variable_definitions"] = [
        {"symbol": "Y", "definition": "冠层覆盖率；", "evidence_span": span},
        {"symbol": "K", "definition": "校正系数", "evidence_span": span},
        {"symbol": "X", "definition": "输入比例", "evidence_span": span},
    ]
    return unit


def test_formula_prompt_links_defined_output_symbol_and_indicator_without_merging():
    unit = output_binding_unit()
    candidate = predict(unit)
    payload = prompt(unit, candidate)
    entities = {e["name"]: e for e in payload["candidate_entities"]}
    link = next(b for b in payload["formula_output_bindings"] if b["variable_id"] == entities["Y"]["id"])
    assert link["indicator_id"] == entities["冠层覆盖率"]["id"]
    assert link["definition"] == "冠层覆盖率；"
    assert entities["Y"]["type"] == "model_variable"
    assert entities["冠层覆盖率"]["type"] == "assessment_indicator"
    assert {link["variable_relation_id"], link["indicator_relation_id"]} <= {
        r["id"] for r in payload["candidate_relations"]
    }


def test_output_binding_does_not_use_unrelated_definitions_or_force_model_choices():
    unit = output_binding_unit()
    unit["variable_definitions"][0]["definition"] = "另一项指标；"
    payload = prompt(unit)
    assert not payload.get("formula_output_bindings")
    unit = output_binding_unit()
    candidate = predict(unit)
    relation = next(r for r in candidate["relations"]
                    if r["head_name"] == "Y" and r["relation_type"] == "calculated_by")
    provider = MagicMock()
    provider.complete.return_value = json.dumps({"selected_entity_ids": [],
                                                "selected_relation_ids": [relation["relation_id"]]})
    selected, _, _ = _llm_select(provider, unit, candidate)
    assert selected["relations"] == [relation]
