"""Replay v2.6/v2.7 candidates on the SAME confirmed train data, then check rule dev."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import types

from verify_human_confirmed_20261002 import check_delivery
from ecospec_kg.analysis_v2 import _entity_key, _relation_key
from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.experiment_io_v2 import read_jsonl, sha256_path, write_json, write_jsonl
from ecospec_kg.extractor_v2 import CANDIDATE_GENERATOR_VERSION, RuleCandidateExtractorV2, extract_v2
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2
from ecospec_kg.training_v2 import prepare_lora_training_v2


BASELINE = "738f1d66bad904cb270ed389fbad83b0478d3694"
BASELINE_VERSION = "structure-aware-rule-v2.6"
CURRENT_VERSION = "structure-aware-rule-v2.7"
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "deliveries/human_confirmed_20261002"


def load_baseline():
    modules = []
    for filename in ("formula_symbols", "extractor_v2"):
        source = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{BASELINE}:src/ecospec_kg/{filename}.py"],
            encoding="utf-8",
        )
        module = types.ModuleType("ecospec_kg._completion_baseline_" + filename)
        module.__file__ = str(REPO / "src/ecospec_kg" / f"{filename}.py")
        module.__package__ = "ecospec_kg"
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        modules.append(module)
    modules[1].formula_symbols = modules[0].formula_symbols
    modules[1].normalize_symbol = modules[0].normalize_symbol
    if modules[1].CANDIDATE_GENERATOR_VERSION != BASELINE_VERSION:
        raise ValueError("baseline version does not match pinned commit")
    return modules[1].RuleCandidateExtractorV2()


def compare_train(units, annotations, baseline, current):
    if any(row.get("split") != "train" for row in annotations):
        raise ValueError("candidate audit accepts train annotations only")
    gold = {row["unit_id"]: row for row in annotations}
    if (len(gold) != len(annotations) or len(units) != len(gold)
            or {u["unit_id"] for u in units} != set(gold)):
        raise ValueError("train source/gold unit inventory mismatch")
    totals = {v: Counter() for v in ("baseline", "current")}
    types_by_version = {v: defaultdict(Counter) for v in totals}
    gained, lost, changes, gaps = Counter(), Counter(), [], []
    for unit in units:
        old, new = baseline.predict_unit(unit), current.predict_unit(unit)
        annotation = gold[unit["unit_id"]]
        difference = {"unit_id": unit["unit_id"]}
        for kind, key, type_index in (("entities", _entity_key, 1), ("relations", _relation_key, 2)):
            reference = {key(row) for row in annotation[kind]}
            before, after = ({key(row) for row in prediction[kind]} for prediction in (old, new))
            singular = "entity" if kind == "entities" else "relation"
            for version, candidates in (("baseline", before), ("current", after)):
                totals[version][f"gold_{singular}_count"] += len(reference)
                totals[version][f"covered_{singular}_count"] += len(reference & candidates)
                totals[version][f"{singular}_candidate_count"] += len(candidates)
                totals[version][f"{singular}_unannotated_candidate_count"] += len(candidates - reference)
                for k in reference:
                    types_by_version[version][k[type_index]]["gold"] += 1
                    types_by_version[version][k[type_index]]["covered"] += int(k in candidates)
                for k in candidates:
                    types_by_version[version][k[type_index]]["candidates"] += 1
                    types_by_version[version][k[type_index]]["unannotated"] += int(k not in reference)
            added, removed = after - before, before - after
            gained.update(k[type_index] for k in added & reference)
            lost.update(k[type_index] for k in removed & reference)
            if added or removed:
                difference[kind] = {"added": sorted(added), "removed": sorted(removed),
                                    "gained_coverage": sorted(added & reference),
                                    "lost_coverage": sorted(removed & reference)}
            for row in annotation[kind]:
                if key(row) not in after:
                    gaps.append({"unit_id": unit["unit_id"], "kind": singular, "gold": row})
        if len(difference) > 1:
            changes.append(difference)
    for version, counts in totals.items():
        for singular in ("entity", "relation"):
            counts[f"{singular}_recall_upper_bound"] = round(
                counts[f"covered_{singular}_count"] / counts[f"gold_{singular}_count"], 6
            ) if counts[f"gold_{singular}_count"] else 1.0
    summary = {**{k: dict(v) for k, v in totals.items()},
               "gained_coverage_by_type": dict(sorted(gained.items())),
               "lost_coverage_by_type": dict(sorted(lost.items())),
               "changed_unit_count": len(changes),
               "by_type": {k: {t: dict(v) for t, v in sorted(types_by_version[k].items())} for k in totals}}
    return summary, changes, gaps


def run(delivery: Path, out: Path):
    if out.exists():
        raise ValueError("output already exists; refusing to overwrite")
    if CANDIDATE_GENERATOR_VERSION != CURRENT_VERSION:
        raise ValueError("audit requires candidate generator v2.7")
    metadata = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))
    if (metadata["schema_version"] != "ecospec-candidate-completion-delivery-v1.0"
            or metadata["candidate_generator"] != CURRENT_VERSION or metadata["baseline_commit"] != BASELINE
            or metadata.get("test_included") is not False
            or metadata["source_delivery_manifest_sha256"] != sha256_path(DATA / "manifest.json")):
        raise ValueError("candidate delivery provenance mismatch")
    if len(metadata["files"]) != 1 or metadata["files"][0]["path"] != "expected.json":
        raise ValueError("candidate delivery must contain only the expected statistics payload")
    for entry in metadata["files"]:
        if sha256_path(delivery / entry["path"]) != entry["sha256"]:
            raise ValueError("candidate delivery hash mismatch")
    manifest = check_delivery(DATA, expected_candidate_generator=BASELINE_VERSION)
    input_hashes = {path: sha256_path(DATA / path) for path in (
        "blind/train_units.jsonl", "gold/train_annotations.jsonl", "blind/dev_units.jsonl", "gold/dev_annotations.jsonl")}
    summary, changes, gaps = compare_train(read_jsonl(DATA / "blind/train_units.jsonl"),
                                           read_jsonl(DATA / "gold/train_annotations.jsonl"),
                                           load_baseline(), RuleCandidateExtractorV2())
    out.mkdir(parents=True)
    write_json(out / "comparison.json", summary)
    write_jsonl(out / "train_candidate_changes.jsonl", changes)
    write_jsonl(out / "candidate_gaps_train.jsonl", gaps)
    training = prepare_lora_training_v2(DATA / "blind/train_units.jsonl", DATA / "gold/train_annotations.jsonl",
                                        out / "training/qwen35_train.jsonl")
    for split in ("train", "dev"):
        folder = out / f"{split}_rule"
        extraction = extract_v2(DATA / "blind" / f"{split}_units.jsonl", folder)
        validation = validate_predictions_v2(DATA / "blind" / f"{split}_units.jsonl", folder / "predictions.jsonl",
                                             DATA / "schema_v2.json", folder / "validation")
        if extraction["status"] != "complete" or not validation["passed"]:
            raise ValueError(f"candidate extraction/validation failed: {split}")
    dev = out / "dev_rule"
    metrics = evaluate_v2(DATA / "gold/dev_annotations.jsonl", dev / "validation/validated_predictions.jsonl",
                          DATA / "blind/dev_units.jsonl", dev / "validation/validation_report.json", dev / "evaluation")
    observed = {"train": summary, "training_records": training["training_records"],
                "dev": {"validated_units": validation["valid_prediction_unit_count"],
                        "entity_f1": metrics["entities"]["micro"]["f1"],
                        "relation_f1": metrics["relations"]["strict_micro"]["f1"],
                        "macro_relation_f1": metrics["relations"]["macro_relation_f1"]}}
    expected = json.loads((delivery / "expected.json").read_text(encoding="utf-8"))
    passed = observed == expected and not summary["lost_coverage_by_type"]
    if input_hashes != {p: sha256_path(DATA / p) for p in input_hashes}:
        raise ValueError("source or labels changed during audit")
    result = {"passed": passed, "observed": observed, "expected": expected,
              "baseline_commit": BASELINE, "candidate_generator": CURRENT_VERSION,
              "source_delivery_manifest_sha256": sha256_path(DATA / "manifest.json"),
              "delivery_manifest_sha256": sha256_path(delivery / "manifest.json"),
              "package_id": manifest["package_id"], "dataset_version": manifest["dataset_version"],
              "implementation_sha256": sha256_path(REPO / "src/ecospec_kg/extractor_v2.py"),
              "test_included": False, "model_training_started": False}
    write_json(out / "verification.json", result)
    if not passed:
        raise ValueError("replay differs from expected or loses coverage; see verification.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=REPO / "deliveries/candidate_completion_20261002")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.delivery, args.out)
    print(json.dumps({"passed": result["passed"], "train": result["observed"]["train"]["current"],
                      "dev": result["observed"]["dev"], "out": str(args.out),
                      "model_training_started": False}, ensure_ascii=False, indent=2))
