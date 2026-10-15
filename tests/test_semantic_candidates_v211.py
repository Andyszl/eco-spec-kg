"""Training-source motivated candidates; semantic selection remains model-owned."""
from copy import deepcopy
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
from ecospec_kg.structural_selection_v2 import structural_relation_proofs
from test_formula_semantic_evidence_v210 import fixture, span


def test_canonical_named_method_drops_trigger_prefix_and_preserves_original():
    u = fixture()
    u["variable_definitions"][0]["definition"] = "指数；"
    u["variable_definitions"][0]["evidence_span"]["text"] = "Y——指数；"
    u["introduction"] = "基于光谱拟合模型（ABC）计算。"
    u["provenance"]["evidence_spans"].append(span("method", u["introduction"]))
    c = RuleCandidateExtractorV2().predict_unit(u)
    methods = {e["name"] for e in c["entities"] if e["entity_type"] == "method"}
    assert "基于光谱拟合模型（ABC）" in methods
    assert "光谱拟合模型（ABC）" in methods


def test_declared_output_has_source_candidate_from_current_package_context():
    u = fixture()
    u["introduction"] = "使用气象数据，按以下模型计算指数。"
    u["provenance"]["evidence_spans"].append(span("source-context", u["introduction"]))
    before = deepcopy(u)
    c = RuleCandidateExtractorV2().predict_unit(u)
    edge = next(r for r in c["relations"] if (r["head_name"], r["relation_type"], r["tail_name"]) == ("Y", "sourced_from", "气象数据"))
    assert {"source-context", "output-definition"} <= set(edge["evidence_span_ids"])
    assert edge["relation_id"] not in {p["relation_id"] for p in structural_relation_proofs(u, c)}
    assert not any(r["head_name"] == "b" and r["relation_type"] == "sourced_from" for r in c["relations"])
    assert u == before
