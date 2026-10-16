from copy import deepcopy
import json
import pytest
from ecospec_kg.extractor_v2 import PredictionBuilder
from ecospec_kg.method_supervision_scope_v2 import canonicalize_methods, training_selection_view


def graph(names, text):
    unit = {"unit_id": "synthetic", "unit_type": "formula_package", "introduction": text,
            "provenance": {"standard_code": "synthetic", "evidence_spans": [
                {"span_id": "s", "page": 1, "bbox": [0, 0, 1, 1], "text": text}]}}
    b = PredictionBuilder(unit, "rule")
    v = b.add_entity("Y", "model_variable")
    for name in names:
        b.add_relation(v, "obtained_by", b.add_entity(name, "method"))
    return unit, b.result()


def test_source_named_method_aliases_become_one_edge_without_selection():
    names = ["光谱拟合模型", "光谱拟合模型（ABC）", "基于光谱拟合模型", "基于光谱拟合模型（ABC）"]
    unit, original = graph(names, "基于光谱拟合模型（ABC）计算。")
    before = deepcopy(original)
    c, audit = canonicalize_methods(unit, original)
    assert [e["name"] for e in c["entities"] if e["entity_type"] == "method"] == ["光谱拟合模型（ABC）"]
    assert len(c["relations"]) == 1
    assert len(set(audit["relation_id_map"].values())) == 1
    assert "selected_relation_ids" not in c
    assert original == before
    assert canonicalize_methods(unit, c)[0] == c


def test_different_models_and_ambiguous_abbreviations_do_not_collapse():
    names = ["光谱拟合模型", "光谱拟合模型（ABC）", "光谱拟合模型（DEF）", "基于冠层模型估算指数"]
    unit, original = graph(names, "光谱拟合模型（ABC）和光谱拟合模型（DEF）；基于冠层模型估算指数。")
    assert canonicalize_methods(unit, original)[0] == original


def test_full_name_absent_from_source_cannot_merge_short_name():
    unit, original = graph(["光谱拟合模型", "光谱拟合模型（ABC）"], "采用光谱拟合模型。")
    assert canonicalize_methods(unit, original)[0] == original


def test_workflow_and_malformed_sentence_boundaries_are_not_aliases():
    names = ["基于遥感的植被覆盖度估算方法", "遥感的植被覆盖度估算方法",
             "坡度可使用数字高程模型", "坡度可使用数字高程模型（DEM）",
             "基于辐射传输模型的方法首先由辐射传输模型", "辐射传输模型的方法首先由辐射传输模型"]
    unit, original = graph(names, "；".join(names))
    assert canonicalize_methods(unit, original)[0] == original


def test_unknown_is_absent_from_training_prompt_but_retained_for_inference():
    unit, original = graph(["光谱拟合模型"], "采用光谱拟合模型计算。")
    rid = original["relations"][0]["relation_id"]
    selected = {"selected_entity_ids": [], "selected_relation_ids": []}
    view, target = training_selection_view(original, selected, [rid])
    assert view["relations"] == [] and target == selected
    assert len(original["relations"]) == 1
    assert rid not in json.dumps(view)


def test_unknown_positive_conflicts_and_unknown_ids_fail_closed():
    unit, original = graph(["光谱拟合模型"], "采用光谱拟合模型计算。")
    rid = original["relations"][0]["relation_id"]
    with pytest.raises(ValueError, match="positive"):
        training_selection_view(original, {"selected_entity_ids": [], "selected_relation_ids": [rid]}, [rid])
    with pytest.raises(ValueError, match="absent"):
        training_selection_view(original, {"selected_entity_ids": [], "selected_relation_ids": []}, ["missing"])


def test_alias_merge_preserves_both_evidence_sets_and_positive_projection():
    from verify_method_scope_20261016 import project
    unit, original = graph(["光谱拟合模型", "光谱拟合模型（ABC）"], "采用光谱拟合模型（ABC）计算。")
    for e in original["entities"]:
        if e["name"] == "光谱拟合模型":
            e["evidence_span_ids"] = ["alias-proof"]
    for r in original["relations"]:
        if r["tail_name"] == "光谱拟合模型":
            r["evidence_span_ids"] = ["alias-proof"]
    annotation = {"entities": original["entities"], "relations": [original["relations"][0]]}
    c, mapping = canonicalize_methods(unit, original)
    assert set(c["relations"][0]["evidence_span_ids"]) == {"s", "alias-proof"}
    assert project(original, mapping, annotation)["selected_relation_ids"] == [c["relations"][0]["relation_id"]]
    assert canonicalize_methods(unit, c)[0] == c
