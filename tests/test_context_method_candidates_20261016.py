from copy import deepcopy
from test_formula_semantic_evidence_v210 import fixture, span
from ecospec_kg.candidate_profile_v214 import candidates_for_unit


def test_explicit_definition_class_has_named_method_candidate_without_auto_selection():
    u = fixture()
    u["variable_definitions"][2]["definition"] = "拟合参数"
    u["introduction"] = "拟合参数常用加权估计法计算得到。"
    u["provenance"]["evidence_spans"].append(span("local-method", u["introduction"]))
    before = deepcopy(u)
    c = candidates_for_unit(u)
    edge = next(r for r in c["relations"] if (r["head_name"],r["relation_type"],r["tail_name"]) == ("b","obtained_by","加权估计法"))
    assert {"local-method","bias"} <= set(edge["evidence_span_ids"])
    assert "selected_relation_ids" not in c
    assert u == before


def test_separate_clause_cooccurrence_does_not_create_input_method_binding():
    u = fixture()
    u["variable_definitions"][2]["definition"] = "拟合参数"
    u["introduction"] = "拟合参数须记录。采用加权估计法计算其他量。"
    c = candidates_for_unit(u)
    assert not any(r["head_name"] == "b" and r["tail_name"] == "加权估计法" for r in c["relations"])


def test_established_named_model_boundary_is_not_whole_sentence():
    u = fixture()
    u["introduction"] = "使用地面观测建立光谱回归模型用于估算指数。"
    u["provenance"]["evidence_spans"].append(span("regression", u["introduction"]))
    c = candidates_for_unit(u)
    assert any(e["entity_type"] == "method" and e["name"] == "光谱回归模型" for e in c["entities"])


def test_latin_model_in_parenthesized_list_is_a_bounded_candidate():
    u = fixture()
    u["introduction"] = "可使用过程模型（如：ALPHA模型、BETA模型）和其他模型进行计算。"
    u["provenance"]["evidence_spans"].append(span("models", u["introduction"]))
    c = candidates_for_unit(u)
    assert {"ALPHA模型","BETA模型"} <= {e["name"] for e in c["entities"]}
