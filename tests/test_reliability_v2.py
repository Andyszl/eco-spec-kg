"""Regression cases for symbol scope, candidate provenance and actual request settings."""
import json
from unittest.mock import MagicMock, patch

import pytest

from test_experiment_chain_v2 import source_unit, empty_annotation
from ecospec_kg.analysis_v2 import collect_error_details_v2
from ecospec_kg.experiment_io_v2 import sha256_json, write_json, write_jsonl
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, _symbol_occurs, extract_v2


@pytest.mark.parametrize("symbol,expression,expected", [
    ("C", "L=A+B*x^c", False), ("n", "L=ln(x)", False),
    ("a", "L=alpha", False), ("S", "P=TS", False),
    ("x_i", "L=x_j", False), ("x", "L=x_i", False),
    ("x_i", "L=x_{i}", True), ("TS", "P=S_ij/TS", True),
    ("n", "L=ln(x)+n", True), ("C", "L=C*x", True),
])
def test_exact_formula_symbols(symbol, expression, expected):
    assert _symbol_occurs(symbol, expression) is expected


def formula_unit(expression="L=A*x^c"):
    unit = source_unit()
    span = unit["variable_definitions"][0]["evidence_span"]
    unit["variable_definitions"] = [
        {"symbol": s, "definition": "parameter", "evidence_span": span}
        for s in ("L", "A", "B", "C", "x", "n", "x_i", "x_j")
    ]
    unit["formulas"][0]["expression_text"] = expression
    return unit


def variable_relations(unit):
    return {(r["relation_type"], r["tail_name"])
            for r in RuleCandidateExtractorV2().predict_unit(unit)["relations"]
            if r["relation_type"] in {"has_input", "has_output"}}


def test_formula_alias_is_explicit_evidenced_and_local():
    unit = formula_unit("L=A+Bx^c")
    assert ("has_input", "C") not in variable_relations(unit)
    unit["formulas"][0]["symbol_aliases"] = [{
        "symbol": "c", "canonical_symbol": "C", "reason": "fixture local symbol definition",
        "evidence_span_ids": ["span-variable"],
    }]
    assert ("has_input", "C") in variable_relations(unit)
    assert ("has_input", "C") not in variable_relations(formula_unit("L=A+Bx^c"))
    unit["formulas"][0]["symbol_aliases"][0]["evidence_span_ids"] = ["missing-span"]
    with pytest.raises(ValueError, match="alias"):
        variable_relations(unit)


def test_implicit_product_and_function_are_disambiguated_by_declared_symbols():
    edges = variable_relations(formula_unit("L=Ax^3+Bx^2+Cx+Aln(x)"))
    assert {("has_input", s) for s in ("A", "B", "C", "x")} <= edges
    assert ("has_input", "n") not in edges


def test_subscript_inputs_are_not_mistaken_for_outputs():
    edges = variable_relations(formula_unit("x_i=x_j+A"))
    assert ("has_output", "x_i") in edges
    assert ("has_input", "x_j") in edges
    assert ("has_output", "x_j") not in edges


def test_no_match_does_not_fall_back_to_all_defined_variables():
    assert variable_relations(formula_unit("Q=unknown(z)")) == set()


def analysis_inputs(tmp_path, candidate_hash):
    unit = source_unit()
    candidate = RuleCandidateExtractorV2().predict_unit(unit)
    annotation = {**empty_annotation(unit["unit_id"]), "split": "dev", "entities": candidate["entities"]}
    prediction = {"unit_id": unit["unit_id"], "entities": [], "relations": []}
    if candidate_hash == "correct":
        prediction["candidate_hash"] = sha256_json({k: candidate[k] for k in ("entities", "relations")})
    elif candidate_hash is not None:
        prediction["candidate_hash"] = candidate_hash
    paths = [tmp_path / name for name in ("units.jsonl", "gold.jsonl", "predictions.jsonl")]
    for path, row in zip(paths, (unit, annotation, prediction)):
        write_jsonl(path, [row])
    return paths


def test_candidate_hash_mismatch_stops_diagnosis(tmp_path):
    paths = analysis_inputs(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="candidate.*hash"):
        collect_error_details_v2(*paths)


def test_missing_candidate_hash_does_not_claim_candidate_presence(tmp_path):
    rows = collect_error_details_v2(*analysis_inputs(tmp_path, None))
    assert rows
    assert all(r["candidate_present"] is None for r in rows)
    assert all(r["candidate_verification"] == "missing_hash" for r in rows)
    assert all(r["selection_result"] == "candidate_unverified" for r in rows)


def test_verified_candidate_hash_preserves_diagnosis(tmp_path):
    rows = collect_error_details_v2(*analysis_inputs(tmp_path, "correct"))
    assert rows and all(r["candidate_present"] is True for r in rows)
    assert all(r["candidate_verification"] == "hash_verified" for r in rows)


@pytest.mark.parametrize("flag", ["use_layout", "use_schema", "use_evidence"])
@pytest.mark.parametrize("value", [False, "false", 0])
def test_unsupported_or_mistyped_ablation_is_rejected_before_output(tmp_path, flag, value):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {flag: value})
    with pytest.raises(ValueError, match=flag):
        extract_v2(tmp_path / "units.jsonl", tmp_path / "out", config_path=tmp_path / "config.json")
    assert not (tmp_path / "out").exists()


def test_extraction_config_controls_real_request(tmp_path):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "seed": 43, "temperature": 0.25, "max_tokens": 321})
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({"choices": [{"message": {"content":
        '{"selected_entity_ids":[],"selected_relation_ids":[]}'}}]}).encode()
    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        result = extract_v2(tmp_path / "units.jsonl", tmp_path / "out", config_path=tmp_path / "config.json")
    assert result["status"] == "complete"
    payload = json.loads(urlopen.call_args.args[0].data)
    assert (payload["seed"], payload["temperature"], payload["max_tokens"]) == (43, 0.25, 321)


@pytest.mark.parametrize("setting", [{"seed": True}, {"seed": "42"}, {"temperature": float("nan")}, {"temperature": -1}])
def test_invalid_sampling_config_is_rejected(tmp_path, setting):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", setting)
    with pytest.raises(ValueError):
        extract_v2(tmp_path / "units.jsonl", tmp_path / "out", config_path=tmp_path / "config.json")
