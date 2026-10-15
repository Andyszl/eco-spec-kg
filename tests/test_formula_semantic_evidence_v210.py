"""Semantic candidate evidence must include source mentions and endpoint definitions."""
from copy import deepcopy
import json
from pathlib import Path

from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
from ecospec_kg.structural_selection_v2 import structural_relation_proofs
from test_experiment_chain_v2 import source_unit


def span(ident, text):
    return {"span_id": ident, "text": text, "page": 1, "bbox": [1, 2, 3, 4]}


def fixture():
    u = source_unit()
    u["unit_id"] = "unseen-semantic-source"
    u["provenance"].update(source_sha256="a" * 64, standard_code="HJ 1173-2021", heading_chain=[])
    u["formulas"] = [{"formula_number": "Q.87", "expression_text": "Y = Z + b",
                       "evidence_span": span("equation", "Y = Z + b") }]
    u["variable_definitions"] = [
        {"symbol": "Y", "definition": "采用光谱拟合模型计算的指数；", "unit": "",
         "evidence_span": span("output-definition", "Y——采用光谱拟合模型计算的指数；")},
        {"symbol": "Z", "definition": "从气象数据获取的观测量；", "unit": "",
         "evidence_span": span("input-definition", "Z——从气象数据获取的观测量；")},
        {"symbol": "b", "definition": "偏置；", "unit": "", "evidence_span": span("bias", "b——偏置；")},
    ]
    u["provenance"]["evidence_spans"] = [span("unrelated", "不相关背景")]
    u.update(introduction="", interstitial_text="", adjacent_source_text="")
    return u


def relation(c, head, kind, tail):
    return next(r for r in c["relations"] if (r["head_name"], r["relation_type"], r["tail_name"]) == (head, kind, tail))


def test_real_training_method_definition_is_cited_in_candidate():
    data = Path(__file__).resolve().parents[1] / "deliveries/human_confirmed_20261002/blind/train_units.jsonl"
    u = next(x for x in map(json.loads, data.read_text(encoding="utf-8").splitlines()) if x["unit_id"] == "113680dd91edf495")
    before = deepcopy(u)
    c = RuleCandidateExtractorV2().predict_unit(u)
    r = relation(c, "K_EPIC", "obtained_by", "侵蚀-生产力评价模型")
    method = next(e for e in c["entities"] if e["entity_id"] == r["tail_id"])
    assert "0cb53d9db64b10df" in r["evidence_span_ids"]
    assert "0cb53d9db64b10df" in method["evidence_span_ids"]
    assert u == before


def test_unseen_method_and_source_definitions_are_local_and_not_auto_selected():
    u = fixture()
    before = deepcopy(u)
    c = RuleCandidateExtractorV2().predict_unit(u)
    method = relation(c, "Y", "obtained_by", "光谱拟合模型")
    source = relation(c, "Z", "sourced_from", "气象数据")
    assert {"equation", "output-definition"} <= set(method["evidence_span_ids"])
    assert {"equation", "input-definition"} <= set(source["evidence_span_ids"])
    assert "unrelated" not in method["evidence_span_ids"] + source["evidence_span_ids"]
    for r, expected in ((method, "output-definition"), (source, "input-definition")):
        endpoint = next(e for e in c["entities"] if e["entity_id"] == r["tail_id"])
        assert expected in endpoint["evidence_span_ids"]
    assert {method["relation_id"], source["relation_id"]}.isdisjoint(p["relation_id"] for p in structural_relation_proofs(u, c))
    assert u == before


def test_method_mention_ignores_layout_whitespace():
    u = fixture()
    u["variable_definitions"][0]["evidence_span"]["text"] = "Y——采用光谱拟合\n模型计算的指数；"
    c = RuleCandidateExtractorV2().predict_unit(u)
    assert "output-definition" in relation(c, "Y", "obtained_by", "光谱拟合模型")["evidence_span_ids"]


def test_context_mention_does_not_attach_unrelated_variable_span():
    u = fixture()
    u["variable_definitions"][0]["definition"] = "指数；"
    u["variable_definitions"][0]["evidence_span"]["text"] = "Y——指数；"
    u["introduction"] = "采用光谱拟合模型计算。"
    u["provenance"]["evidence_spans"].append(span("method-context", "采用光谱拟合模型计算。"))
    c = RuleCandidateExtractorV2().predict_unit(u)
    r = relation(c, "Y", "obtained_by", "光谱拟合模型")
    assert {"method-context", "output-definition"} <= set(r["evidence_span_ids"])
    assert "input-definition" not in r["evidence_span_ids"]
