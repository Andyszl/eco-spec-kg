from pathlib import Path
import json
from ecospec_kg import extractor_v2 as ex
ROOT=Path(__file__).resolve().parents[1]


def test_named_latin_equation_after_yi_jiayi_is_detected():
    names=ex._method_candidates("土垄糙度，以 Smith-Carson 方程加以计算，cm；")
    assert "Smith-Carson方程" in names


def test_generic_rule_is_not_specific_to_one_training_name():
    assert "Alpha-Beta模型" in ex._method_candidates("采用 Alpha-Beta 模型加以估算。")
    assert "Smith-Carson方程" not in ex._method_candidates("相关资料列举 Smith-Carson，并未说明计算方法。")


def test_training_variable_definition_evidence_reaches_named_candidate():
    units=[json.loads(s) for s in (ROOT/"deliveries/human_confirmed_20261002/blind/train_units.jsonl").read_text(encoding="utf-8").splitlines()]
    unit=next(u for u in units if any("Smith-Carson" in v.get("definition","") for v in u.get("variable_definitions",[])))
    candidate=ex.RuleCandidateExtractorV2().predict_unit(unit)
    edge=next(r for r in candidate["relations"] if r["head_name"]=="K_r" and r["tail_name"]=="Smith-Carson方程" and r["relation_type"]=="obtained_by")
    definition=next(v for v in unit["variable_definitions"] if v["symbol"]=="K_r")
    assert definition["evidence_span"]["span_id"] in edge["evidence_span_ids"]
