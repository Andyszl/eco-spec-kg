"""Verify v2.8 source repairs and prepare train-only data, without calling a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_candidate_completion_20261002 import compare_train
from verify_human_confirmed_20261002 import check_delivery
from ecospec_kg.analysis_v2 import _entity_key, _relation_key
from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.experiment_io_v2 import read_jsonl, sha256_path, write_json, write_jsonl
from ecospec_kg.extractor_v2 import (
    CANDIDATE_GENERATOR_VERSION, SELECTION_POLICY_VERSION, RuleCandidateExtractorV2, extract_v2,
)
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2
from ecospec_kg.training_v2 import prepare_lora_training_v2

DATA = REPO / "deliveries/human_confirmed_20261002"
BASELINE = "fba99b7f79656474f98e9d54024e87e375d023e5"
BASELINE_VERSION = "structure-aware-rule-v2.7"
EXPECTED_DEV_COVERAGE = {
    "baseline": {"gold_entity_count": 57, "covered_entity_count": 48, "entity_candidate_count": 84,
                 "gold_relation_count": 33, "covered_relation_count": 24, "relation_candidate_count": 52},
    "current": {"gold_entity_count": 57, "covered_entity_count": 54, "entity_candidate_count": 90,
                "gold_relation_count": 33, "covered_relation_count": 33, "relation_candidate_count": 55},
    "lost_coverage": {"entity": 0, "relation": 0},
}
EXPECTED_TRAINING_SHA256 = "cf44bc680753b76765335844476cdaa2abd7fb4bd1567f4230afba5719a12baa"


def load_baseline():
    modules = []
    for filename in ("formula_symbols", "extractor_v2"):
        source = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{BASELINE}:src/ecospec_kg/{filename}.py"],
            encoding="utf-8",
        )
        module = types.ModuleType("ecospec_kg._repair_baseline_" + filename)
        module.__file__ = str(REPO / "src/ecospec_kg" / f"{filename}.py")
        module.__package__ = "ecospec_kg"
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        modules.append(module)
    if modules[1].CANDIDATE_GENERATOR_VERSION != BASELINE_VERSION:
        raise ValueError("baseline version does not match pinned commit")
    modules[1].formula_symbols = modules[0].formula_symbols
    modules[1].normalize_symbol = modules[0].normalize_symbol
    return modules[1].RuleCandidateExtractorV2()


def compare_dev(units, annotations, baseline, current):
    """Dev labels are used only for coverage measurement, never for projection."""
    gold = {row["unit_id"]: row for row in annotations}
    counts = {v: Counter() for v in ("baseline", "current")}
    lost, changes = Counter(), []
    for unit in units:
        old, new = baseline.predict_unit(unit), current.predict_unit(unit)
        detail = {"unit_id": unit["unit_id"]}
        for kind, singular, key in (("entities", "entity", _entity_key),
                                    ("relations", "relation", _relation_key)):
            reference = {key(row) for row in gold[unit["unit_id"]][kind]}
            before, after = ({key(row) for row in p[kind]} for p in (old, new))
            for version, candidates in (("baseline", before), ("current", after)):
                counts[version][f"gold_{singular}_count"] += len(reference)
                counts[version][f"covered_{singular}_count"] += len(reference & candidates)
                counts[version][f"{singular}_candidate_count"] += len(candidates)
            lost[singular] += len((before & reference) - after)
            if before != after:
                detail[kind] = {"added": sorted(after - before), "removed": sorted(before - after),
                                "gained_coverage": sorted((after - before) & reference),
                                "lost_coverage": sorted((before - after) & reference)}
        if len(detail) > 1:
            changes.append(detail)
    return {**{v: dict(c) for v, c in counts.items()}, "lost_coverage": dict(lost)}, changes


def run(out: Path):
    if out.exists():
        raise ValueError("output already exists; refusing to overwrite")
    if (CANDIDATE_GENERATOR_VERSION != "structure-aware-rule-v2.8"
            or SELECTION_POLICY_VERSION != "ecospec-selection-v2.5"):
        raise ValueError("audit requires candidate v2.8 and selection policy v2.5")
    # The frozen package records its original v2.6 baseline; never rewrite it.
    manifest = check_delivery(DATA, expected_candidate_generator="structure-aware-rule-v2.6")
    hashes = {p: sha256_path(DATA / p) for p in
              ["manifest.json", *(entry["path"] for entry in manifest["files"])]}
    baseline, current = load_baseline(), RuleCandidateExtractorV2()
    train, changes, gaps = compare_train(
        read_jsonl(DATA / "blind/train_units.jsonl"), read_jsonl(DATA / "gold/train_annotations.jsonl"),
        baseline, current,
    )
    dev, dev_changes = compare_dev(
        read_jsonl(DATA / "blind/dev_units.jsonl"), read_jsonl(DATA / "gold/dev_annotations.jsonl"),
        baseline, current,
    )
    out.mkdir(parents=True)
    write_json(out / "candidate_coverage.json", {"train": train, "dev": dev})
    write_jsonl(out / "train_candidate_changes.jsonl", changes)
    write_jsonl(out / "candidate_gaps_train.jsonl", gaps)
    write_jsonl(out / "dev_candidate_changes.jsonl", dev_changes)
    training_path = out / "training/qwen35_train.jsonl"
    training = prepare_lora_training_v2(
        DATA / "blind/train_units.jsonl", DATA / "gold/train_annotations.jsonl", training_path,
    )
    validated = {}
    for split in ("train", "dev"):
        folder = out / f"{split}_rule"
        units = DATA / "blind" / f"{split}_units.jsonl"
        extraction = extract_v2(units, folder)
        validation = validate_predictions_v2(
            units, folder / "predictions.jsonl", DATA / "schema_v2.json", folder / "validation",
        )
        if extraction["status"] != "complete" or not validation["passed"]:
            raise ValueError(f"rule extraction or validation failed: {split}")
        validated[split] = validation["valid_prediction_unit_count"]
    folder = out / "dev_rule"
    metrics = evaluate_v2(
        DATA / "gold/dev_annotations.jsonl", folder / "validation/validated_predictions.jsonl",
        DATA / "blind/dev_units.jsonl", folder / "validation/validation_report.json", folder / "evaluation",
    )
    unchanged = hashes == {p: sha256_path(DATA / p) for p in hashes}
    training_hash = sha256_path(training_path)
    passed = (unchanged and not train["lost_coverage_by_type"] and dev == EXPECTED_DEV_COVERAGE
              and training_hash == EXPECTED_TRAINING_SHA256
              and training["training_records"] == manifest["unit_split_counts"]["train"]
              and validated == manifest["unit_split_counts"])
    result = {
        "passed": passed, "frozen_inputs_unchanged": unchanged, "input_sha256": hashes,
        "baseline_commit": BASELINE, "candidate_generator": CANDIDATE_GENERATOR_VERSION,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "implementation_sha256": sha256_path(REPO / "src/ecospec_kg/extractor_v2.py"),
        "package_id": manifest["package_id"], "dataset_version": manifest["dataset_version"],
        "train_coverage": train, "dev_coverage": dev, "expected_dev_coverage": EXPECTED_DEV_COVERAGE,
        "rule_validation": validated,
        "rule_dev_metrics": {
            "entity_f1": metrics["entities"]["micro"]["f1"],
            "relation_f1": metrics["relations"]["strict_micro"]["f1"],
            "macro_relation_f1": metrics["relations"]["macro_relation_f1"],
        },
        "training": training, "training_sha256": training_hash,
        "expected_training_sha256": EXPECTED_TRAINING_SHA256,
        "model_training_started": False,
    }
    write_json(out / "verification.json", result)
    if not passed:
        raise ValueError("repair audit differs from expected coverage/training or changes inputs; see verification.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({
        "passed": result["passed"], "candidate_generator": result["candidate_generator"],
        "selection_policy": result["selection_policy_version"], "dev_coverage": result["dev_coverage"],
        "training_records": result["training"]["training_records"],
        "has_indicator_training": result["training"]["selection_focus_coverage"]["has_indicator"],
        "training_sha256": result["training_sha256"], "frozen_inputs_unchanged": True,
        "rule_validation": result["rule_validation"], "model_training_started": False, "out": str(args.out),
    }, ensure_ascii=False, indent=2))
