from __future__ import annotations

import json
import unittest
from pathlib import Path

from ecospec_kg.annotation_v2 import load_enriched_source_units
from ecospec_kg.dataset_quality_v2 import (
    run_full_parser_invariants,
    run_representative_acceptance,
    validate_gold_paths,
    validate_splits,
)
from ecospec_kg.pilot_quality_v2 import (
    compare_experts,
    read_jsonl,
    validate_expert_annotations,
)


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.root = cls.repo / "results" / "dataset_v2_20260724"
        cls.gold = cls.root / "gold"
        required = [
            cls.root / "layouts" / "manifest.json",
            cls.gold / "source_units_enriched.jsonl",
            cls.gold / "entities_final_gold.jsonl",
            cls.gold / "triples_final_gold.jsonl",
        ]
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("generated Schema V2 dataset is unavailable")

    def test_enriched_formula_inventory_is_unique(self) -> None:
        units = load_enriched_source_units(
            self.root / "source_units.jsonl" / "source_units.jsonl",
            self.root / "layouts" / "manifest.json",
            self.repo / "config" / "formula_overrides_v2.json",
        )
        formulas = [
            (
                unit["provenance"]["standard_code"],
                unit["provenance"].get("section", ""),
                formula["formula_number"],
            )
            for unit in units
            if unit["unit_type"] == "formula_package"
            for formula in unit["formulas"]
        ]
        self.assertEqual(len(units), 882)
        self.assertEqual(len(formulas), 65)
        self.assertEqual(len(set(formulas)), 65)
        self.assertEqual(
            sum(bool(unit.get("manual_transcription")) for unit in units),
            4,
        )

    def test_all_11_standards_and_30_representative_pages_pass(self) -> None:
        report = run_representative_acceptance(
            self.root / "layouts",
            self.gold / "source_units_enriched.jsonl",
            self.repo / "config" / "parse_pilot_v2.json",
            self.repo / "config" / "parse_acceptance_11_additions_v2.json",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["case_count"], 30)
        self.assertEqual(
            report["passed_check_count"], report["check_count"]
        )

    def test_all_118_pages_pass_invariants(self) -> None:
        report = run_full_parser_invariants(
            self.root / "layouts",
            self.gold / "source_units_enriched.jsonl",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["scope"]["pages"], 118)
        self.assertEqual(report["scope"]["formulas"], 65)

    def test_gold_formula_source_unit_and_quality_paths_pass(self) -> None:
        report = validate_gold_paths(
            self.gold / "source_units_enriched.jsonl",
            self.gold / "annotations_all_units.jsonl",
            self.gold / "entities_final_gold.jsonl",
            self.gold / "triples_final_gold.jsonl",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["formula_count"], 65)
        self.assertEqual(
            sum(
                not row["output_applicable"]
                for row in report["formula_paths"]
            ),
            2,
        )
        self.assertGreater(report["quality_constraint_count"], 0)

    def test_train_dev_test_split_has_no_group_leakage(self) -> None:
        report = validate_splits(
            self.gold / "source_units_enriched.jsonl",
            self.gold / "splits",
        )
        self.assertTrue(report["passed"])
        self.assertTrue(
            all(
                report["unit_split_counts"][split] > 0
                for split in ("train", "dev", "test")
            )
        )

    def test_dual_expert_files_are_valid_and_independent(self) -> None:
        pilot = self.root / "pilot"
        source_units = read_jsonl(pilot / "pilot_source_units_60.jsonl")
        expert_a = read_jsonl(pilot / "expert_A_annotations.jsonl")
        expert_b = read_jsonl(pilot / "expert_B_annotations.jsonl")
        self.assertTrue(
            validate_expert_annotations(
                source_units, expert_a, "expert_A"
            )["passed"]
        )
        self.assertTrue(
            validate_expert_annotations(
                source_units, expert_b, "expert_B"
            )["passed"]
        )
        agreement, disagreements = compare_experts(
            source_units, expert_a, expert_b
        )
        self.assertEqual(agreement["unit_count"], 60)
        self.assertEqual(
            agreement["disagreement_unit_count"], len(disagreements)
        )
        self.assertGreater(agreement["entity_agreement"]["f1"], 0)
        self.assertGreater(agreement["relation_agreement"]["f1"], 0)

    def test_adjudication_is_complete_and_used_by_final_dataset(self) -> None:
        pilot = self.root / "pilot"
        adjudicated = read_jsonl(
            pilot / "adjudicated_annotations.jsonl"
        )
        adjudication_log = read_jsonl(
            pilot / "adjudication_log.jsonl"
        )
        validation = json.loads(
            (
                pilot / "adjudication_validation_report.json"
            ).read_text(encoding="utf-8")
        )
        summary = json.loads(
            (self.gold / "dataset_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(validation["passed"])
        self.assertEqual(len(adjudicated), 60)
        self.assertEqual(len(adjudication_log), 41)
        self.assertEqual(
            summary["dual_expert_adjudicated_unit_count"],
            len(adjudicated),
        )

    def test_dataset_summary_is_explicit_about_pre_gold_status(self) -> None:
        summary = json.loads(
            (self.gold / "dataset_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["gold_nature"], "ai_expert_pre_gold")
        self.assertTrue(
            summary["human_expert_review_required_for_publication"]
        )


if __name__ == "__main__":
    unittest.main()
