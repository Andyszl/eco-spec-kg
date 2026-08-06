from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.experiment_data_v2 import prepare_experiment_package_v2
from ecospec_kg.extractor_v2 import extract_v2
from ecospec_kg.io_utils import read_json, read_jsonl, stable_id, write_jsonl
from ecospec_kg.ontology_v2 import ONTOLOGY_VERSION
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2
from ecospec_kg.training_v2 import prepare_lora_training_v2


def source_unit() -> dict:
    formula_span = {
        "span_id": "span-formula",
        "page": 6,
        "bbox": [10.0, 20.0, 200.0, 40.0],
        "line_ids": ["line-formula"],
        "text": "P_ij = S_ij / TS （1）",
    }
    variable_span = {
        "span_id": "span-variable",
        "page": 6,
        "bbox": [10.0, 45.0, 200.0, 80.0],
        "line_ids": ["line-variable"],
        "text": "P_ij为构成比例；S_ij为面积；TS为总面积。",
    }
    return {
        "schema_version": "source-unit-v2.0",
        "unit_id": "unit-test-formula",
        "unit_type": "formula_package",
        "provenance": {
            "standard_code": "HJ 1171-2021",
            "document_title": "生态系统格局评估",
            "source_file": "fixture.pdf",
            "source_sha256": "f" * 64,
            "pages": [6],
            "printed_pages": ["3"],
            "section": "6.1",
            "heading_chain": ["6.1 生态系统类型构成比例"],
            "evidence_spans": [formula_span, variable_span],
        },
        "formulas": [
            {
                "formula_number": "1",
                "expression_text": "P_ij = S_ij / TS",
                "evidence_span": formula_span,
            }
        ],
        "variable_definitions": [
            {
                "symbol": "P_ij",
                "definition": "构成比例",
                "unit": "量纲一",
                "evidence_span": variable_span,
            },
            {
                "symbol": "S_ij",
                "definition": "面积",
                "unit": "km2",
                "evidence_span": variable_span,
            },
            {
                "symbol": "TS",
                "definition": "总面积",
                "unit": "km2",
                "evidence_span": variable_span,
            },
        ],
        "introduction": "生态系统类型构成比例按公式（1）计算。",
        "interstitial_text": "",
        "adjacent_source_text": "评估区",
    }


def empty_annotation(unit_id: str) -> dict:
    return {
        "annotation_version": "ecospec-annotation-v2.1",
        "ontology_version": ONTOLOGY_VERSION,
        "unit_id": unit_id,
        "standard_code": "HJ 1171-2021",
        "entities": [],
        "relations": [],
        "review_status": "human_expert_adjudicated",
        "no_relation_reason": "fixture",
        "notes": "",
    }


class ExperimentChainV2Tests(unittest.TestCase):
    def test_prepare_lora_v2_matches_selector_task_and_rejects_test_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units_path = root / "train_units.jsonl"
            write_jsonl(units_path, [source_unit()])
            run = root / "rule"
            extract_v2(units_path, run)
            prediction = read_jsonl(run / "predictions.jsonl")[0]
            annotation = {
                "unit_id": prediction["unit_id"],
                "split": "train",
                "entities": prediction["entities"],
                "relations": prediction["relations"],
            }
            annotations_path = root / "train_annotations.jsonl"
            write_jsonl(annotations_path, [annotation])
            output_path = root / "lora.jsonl"

            manifest = prepare_lora_training_v2(
                units_path, annotations_path, output_path
            )
            row = read_jsonl(output_path)[0]
            completion = json.loads(row["messages"][-1]["content"])
            self.assertEqual(
                set(completion["selected_relation_ids"]),
                {item["relation_id"] for item in prediction["relations"]},
            )
            self.assertEqual(
                manifest["candidate_coverage"]["relation_recall_upper_bound"],
                1.0,
            )

            annotation["split"] = "test"
            write_jsonl(annotations_path, [annotation])
            with self.assertRaisesRegex(ValueError, "split=train"):
                prepare_lora_training_v2(
                    units_path, annotations_path, output_path
                )

    def test_extractor_has_no_external_test_specific_constants(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "src" / "ecospec_kg" / "extractor_v2.py").read_text(
            encoding="utf-8"
        )
        for code in ("HJ 1171-2021", "HJ 1174-2021", "HJ 1175-2021"):
            self.assertNotIn(code, source)
        self.assertNotIn("annotation_v2", source)
        validator = (
            repo / "src" / "ecospec_kg" / "prediction_validation_v2.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            repo / "src" / "ecospec_kg" / "evaluation_v2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("extractor_v2", validator)
        self.assertNotIn("annotation_v2", evaluator)

    def test_prepare_rejects_answer_payload_in_source_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = {**source_unit(), "gold_annotation": {"relations": []}}
            source_path = root / "source.jsonl"
            annotations_path = root / "annotations.jsonl"
            write_jsonl(source_path, [unit])
            write_jsonl(annotations_path, [empty_annotation(unit["unit_id"])])
            with self.assertRaisesRegex(ValueError, "answer payload"):
                prepare_experiment_package_v2(
                    source_path, annotations_path, root / "package"
                )

    def test_rule_extraction_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = root / "units.jsonl"
            write_jsonl(units, [source_unit()])
            extract_v2(units, root / "run-a")
            extract_v2(units, root / "run-b")
            self.assertEqual(
                (root / "run-a" / "predictions.jsonl").read_bytes(),
                (root / "run-b" / "predictions.jsonl").read_bytes(),
            )

    def test_validation_rejects_invalid_evidence_and_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = root / "units.jsonl"
            write_jsonl(units, [source_unit()])
            run = root / "run"
            extract_v2(units, run)
            prediction = read_jsonl(run / "predictions.jsonl")[0]
            prediction["review_status"] = "leaked"
            prediction["entities"][0]["evidence_span_ids"] = ["not-a-span"]
            broken = root / "broken.jsonl"
            write_jsonl(broken, [prediction])

            schema = root / "schema.json"
            schema.write_text(
                json.dumps({"ontology_version": ONTOLOGY_VERSION}),
                encoding="utf-8",
            )
            report = validate_predictions_v2(
                units, broken, schema, root / "validation"
            )
            self.assertFalse(report["passed"])
            self.assertIn("gold_field_leakage", report["failure_code_counts"])
            self.assertIn("invalid_entity_evidence", report["failure_code_counts"])

    def test_full_chain_and_perfect_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            annotations_path = root / "annotations.jsonl"
            unit = source_unit()
            write_jsonl(source_path, [unit])
            write_jsonl(annotations_path, [empty_annotation(unit["unit_id"])])
            package_dir = root / "package"
            prepare_experiment_package_v2(
                source_path,
                annotations_path,
                package_dir,
                gold_nature="human_expert_gold",
            )
            blind = package_dir / "blind" / "test_units.jsonl"
            run = root / "run"
            extract_v2(blind, run)
            prediction = read_jsonl(run / "predictions.jsonl")[0]

            # Build an independent fixture gold file equal to the known prediction.
            gold_record = {
                "annotation_version": "fixture-v2",
                "ontology_version": ONTOLOGY_VERSION,
                "unit_id": unit["unit_id"],
                "standard_code": "HJ 1171-2021",
                "entities": [
                    {
                        "entity_id": stable_id("gold", entity["entity_id"]),
                        "name": entity["name"],
                        "entity_type": entity["entity_type"],
                        "evidence_span_ids": entity["evidence_span_ids"],
                    }
                    for entity in prediction["entities"]
                ],
                "relations": [
                    {
                        **relation,
                        "head_id": stable_id("gold", relation["head_id"]),
                        "tail_id": stable_id("gold", relation["tail_id"]),
                    }
                    for relation in prediction["relations"]
                ],
                "review_status": "human_expert_adjudicated",
                "no_relation_reason": "",
                "notes": "fixture",
                "split": "test",
            }
            gold_path = package_dir / "gold" / "test_annotations.jsonl"
            write_jsonl(gold_path, [gold_record])
            manifest = read_json(package_dir / "manifest.json")
            for item in manifest["files"]:
                if item["path"] == "gold/test_annotations.jsonl":
                    from ecospec_kg.experiment_io_v2 import sha256_path

                    item["bytes"] = gold_path.stat().st_size
                    item["sha256"] = sha256_path(gold_path)
            (package_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            validation_dir = run / "validation"
            validation = validate_predictions_v2(
                blind,
                run / "predictions.jsonl",
                package_dir / "schema_v2.json",
                validation_dir,
            )
            self.assertTrue(validation["passed"])
            report = evaluate_v2(
                gold_path,
                validation_dir / "validated_predictions.jsonl",
                blind,
                validation_dir / "validation_report.json",
                run / "evaluation",
            )
            self.assertEqual(report["entities"]["micro"]["f1"], 1.0)
            self.assertEqual(report["relations"]["strict_micro"]["f1"], 1.0)
            self.assertFalse(
                report["dataset"]["human_expert_review_required_for_publication"]
            )

            empty_prediction = {
                **prediction,
                "entities": [],
                "relations": [],
                "no_relation_reason": "model_predicted_no_relation",
            }
            empty_path = root / "empty_predictions.jsonl"
            write_jsonl(empty_path, [empty_prediction])
            empty_validation_dir = root / "empty_validation"
            empty_validation = validate_predictions_v2(
                blind,
                empty_path,
                package_dir / "schema_v2.json",
                empty_validation_dir,
            )
            self.assertTrue(empty_validation["passed"])
            empty_report = evaluate_v2(
                gold_path,
                empty_validation_dir / "validated_predictions.jsonl",
                blind,
                empty_validation_dir / "validation_report.json",
                root / "empty_evaluation",
            )
            self.assertEqual(
                empty_report["relations"]["strict_micro"]["recall"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
