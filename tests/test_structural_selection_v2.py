import copy

import pytest

from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
from ecospec_kg.structural_selection_v2 import merge_structural_relations, structural_relation_proofs, formula_path_metrics
from test_candidate_selection_repairs_v2 import output_binding_unit


def fixture(expression="Y = K * X"):
    unit = output_binding_unit()
    unit["formulas"][0]["expression_text"] = expression
    unit["formulas"][0]["evidence_span"]["text"] = expression
    return unit, RuleCandidateExtractorV2().predict_unit(unit)


def keys(candidate, proofs):
    ids = {p["relation_id"] for p in proofs}
    return {(r["head_name"], r["relation_type"], r["tail_name"]) for r in candidate["relations"] if r["relation_id"] in ids}


def test_source_roles_binding_and_endpoint_closure_are_generic_and_idempotent():
    u, c = fixture()
    empty = {**c, "entities": [], "relations": []}
    original = copy.deepcopy((u, c, empty))
    result, audit = merge_structural_relations(u, c, empty)
    assert keys(c, audit["proofs"]) == {("公式（B.8）", "has_input", "K"), ("公式（B.8）", "has_input", "X"),
        ("公式（B.8）", "has_output", "Y"), ("Y", "calculated_by", "公式（B.8）"), ("冠层覆盖率", "calculated_by", "公式（B.8）")}
    assert len(result["entities"]) == 5
    again, second = merge_structural_relations(u, c, result)
    assert again == result and second["added_relation_ids"] == []
    assert (u, c, empty) == original
    u["unit_id"] = "unseen-source"
    u["formulas"][0]["formula_number"] = "Z.97"
    c2 = RuleCandidateExtractorV2().predict_unit(u)
    assert len(structural_relation_proofs(u, c2)) == 5


def test_auxiliary_output_and_definition_qualifier_never_bind_heading_indicator():
    u, c = fixture()
    u["variable_definitions"][0]["definition"] = "像元冠层覆盖率；"
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert ("冠层覆盖率", "calculated_by", "公式（B.8）") not in keys(c, structural_relation_proofs(u, c))
    u["variable_definitions"][0]["definition"] = "校正因子；"
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert all(p["rule"] != "explicit_output_definition" for p in structural_relation_proofs(u, c))


def test_explicit_unit_suffix_binds_but_unverified_transcription_does_not():
    u, _ = fixture()
    u["variable_definitions"][0].update(definition="冠层覆盖率，%", unit="%")
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert any(p["rule"] == "explicit_output_definition" for p in structural_relation_proofs(u, c))
    u["variable_definitions"][0]["definition_origin"] = "equation_structure_transcription"
    assert all(p["rule"] != "explicit_output_definition" for p in structural_relation_proofs(u, c))


def test_invalid_evidence_or_incompatible_baseline_is_rejected():
    u, c = fixture()
    c["relations"][0]["evidence_span_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="evidence"):
        structural_relation_proofs(u, c)
    u, c = fixture()
    p = copy.deepcopy(c)
    p["relations"][0]["tail_name"] = "forged"
    with pytest.raises(ValueError, match="candidate"):
        merge_structural_relations(u, c, p)


def test_no_case_folding_or_compound_lhs_guessing():
    u, c = fixture("Y = k * X")
    assert ("公式（B.8）", "has_input", "K") not in keys(c, structural_relation_proofs(u, c))
    u, c = fixture("K + Y = X")
    assert not structural_relation_proofs(u, c)


def test_bound_indices_cardinalities_and_transcribed_limits_abstain():
    u, _ = fixture("Y = sum(X, i, 1, K)")
    u["variable_definitions"][1]["definition"] = "样本总数；"
    u["variable_definitions"].append({"symbol": "i", "definition": "样本序号", "evidence_span": u["variable_definitions"][0]["evidence_span"]})
    c = RuleCandidateExtractorV2().predict_unit(u)
    retained = keys(c, structural_relation_proofs(u, c))
    assert ("公式（B.8）", "has_input", "K") not in retained
    assert ("公式（B.8）", "has_input", "i") not in retained
    u["variable_definitions"][1].update(definition="上限", definition_origin="equation_structure_transcription")
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert ("公式（B.8）", "has_input", "K") not in keys(c, structural_relation_proofs(u, c))


def test_semantic_model_choices_survive_and_no_source_specific_alias_is_injected():
    u, c = fixture()
    extra = next(r for r in c["relations"] if r["head_type"] == "assessment_indicator")
    p = {**c, "relations": [extra]}
    r, _ = merge_structural_relations(u, c, p)
    assert extra in r["relations"]
    # A candidate's manual role cannot override the equation in the retention module.
    u["formulas"][0]["manual_variable_roles"] = {"K": "output", "Y": "input"}
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert ("公式（B.8）", "has_output", "K") not in keys(c, structural_relation_proofs(u, c))


def test_formula_path_requires_every_gold_role_edge_and_evidenced_endpoints():
    u, c = fixture()
    p, _ = merge_structural_relations(u, c, {**c, "entities": [], "relations": []})
    gold = copy.deepcopy(p)
    assert formula_path_metrics([u], [gold], [p])["complete_evidenced_paths"] == 1
    missing = copy.deepcopy(p)
    missing["relations"] = [r for r in missing["relations"] if r["tail_name"] != "K"]
    assert formula_path_metrics([u], [gold], [missing])["complete_evidenced_paths"] == 0
    missing = copy.deepcopy(p)
    missing["entities"][0]["evidence_span_ids"] = ["unknown"]
    assert formula_path_metrics([u], [gold], [missing])["complete_evidenced_paths"] == 0
    assert formula_path_metrics([u], [{**gold, "relations": []}], [p])["completion_rate"] is None
