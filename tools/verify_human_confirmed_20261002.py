"""Verify the human-confirmed train/dev delivery and reproduce its local baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_human_review_v2 import _validate_review_rows, read_rows, write_json
from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.experiment_io_v2 import assert_blind_records, sha256_path
from ecospec_kg.extractor_v2 import CANDIDATE_GENERATOR_VERSION, extract_v2
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2
from ecospec_kg.training_v2 import prepare_lora_training_v2


FILES = {
    "blind/train_units.jsonl", "gold/train_annotations.jsonl",
    "blind/dev_units.jsonl", "gold/dev_annotations.jsonl",
    "schema_v2.json", "expected.json",
}


def check_delivery(folder: Path, *, expected_candidate_generator: str = CANDIDATE_GENERATOR_VERSION) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    actual_data_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*.jsonl")}
    if ({entry["path"] for entry in files} != FILES or len(files) != len(FILES)
            or actual_data_files != {name for name in FILES if name.endswith(".jsonl")}):
        raise ValueError("delivery file inventory must contain only the fixed train/dev files")
    if manifest.get("candidate_generator") != expected_candidate_generator:
        raise ValueError("candidate generator version differs from this delivery")
    if (manifest.get("gold_nature") != "human_expert_gold"
            or manifest.get("review_method") != "ai_assisted_human_confirmed"):
        raise ValueError("delivery review metadata does not match the confirmed dataset")
    for entry in files:
        if sha256_path(folder / entry["path"]) != entry["sha256"]:
            raise ValueError(f"delivery hash mismatch: {entry['path']}")
    seen = set()
    for split in ("train", "dev"):
        units = read_rows(folder / "blind" / f"{split}_units.jsonl")
        labels = read_rows(folder / "gold" / f"{split}_annotations.jsonl")
        ids = [row["unit_id"] for row in units]
        if (len(units) != manifest["unit_split_counts"][split]
                or len(ids) != len(set(ids)) or set(ids) & seen
                or ids != [row["unit_id"] for row in labels]
                or any(row.get("split") != split for row in labels)):
            raise ValueError(f"delivery split/unit inventory mismatch: {split}")
        seen.update(ids)
        assert_blind_records(units)
        _validate_review_rows(units, labels, "human_confirmed_review")
    return manifest


def run(folder: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("output already exists; refusing to overwrite")
    manifest = check_delivery(folder)
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    train = prepare_lora_training_v2(folder / "blind/train_units.jsonl",
                                     folder / "gold/train_annotations.jsonl",
                                     out / "training/qwen35_train.jsonl")
    dev = out / "dev_rule"
    extraction = extract_v2(folder / "blind/dev_units.jsonl", dev,
                            config_path=Path(__file__).resolve().parents[1]
                            / "config/experiments_v2/rule_baseline.json")
    if extraction["status"] != "complete":
        raise ValueError("rule extraction did not complete")
    validation = validate_predictions_v2(folder / "blind/dev_units.jsonl",
                                         dev / "predictions.jsonl", folder / "schema_v2.json",
                                         dev / "validation")
    if not validation["passed"]:
        raise ValueError("rule predictions failed validation")
    metrics = evaluate_v2(folder / "gold/dev_annotations.jsonl",
                          dev / "validation/validated_predictions.jsonl",
                          folder / "blind/dev_units.jsonl",
                          dev / "validation/validation_report.json", dev / "evaluation")
    observed = {
        "training_records": train["training_records"],
        "candidate_coverage": {key: round(value, 6) if isinstance(value, float) else value
                               for key, value in train["candidate_coverage"].items()},
        "dev": {"validated_units": validation["valid_prediction_unit_count"],
                "entity_f1": metrics["entities"]["micro"]["f1"],
                "relation_f1": metrics["relations"]["strict_micro"]["f1"],
                "macro_relation_f1": metrics["relations"]["macro_relation_f1"]},
    }
    result = {"passed": observed == expected, "observed": observed, "expected": expected,
              "parent_package_id": manifest["package_id"],
              "dataset_version": manifest["dataset_version"],
              "delivery_manifest_sha256": sha256_path(folder / "manifest.json"),
              "model_training_started": False}
    write_json(out / "verification.json", result)
    if not result["passed"]:
        raise ValueError("local/server results differ; see verification.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.delivery, args.out), ensure_ascii=False, indent=2))
