from __future__ import annotations

import unittest
from pathlib import Path

from ecospec_kg.corpus import discover_corpus
from ecospec_kg.foundation_quality import (
    default_pilot_path,
    evaluate_parser_quality,
)
from ecospec_kg.io_utils import read_json
from ecospec_kg.layout_parser import parse_pdf_layout
from ecospec_kg.ontology_v2 import (
    DocumentObjectType,
    EntityTypeV2,
    ProvenanceV2,
    RelationTypeV2,
    legacy_migration_status,
    schema_quality_report,
    validate_provenance_v2,
    validate_relation_v2,
)
from ecospec_kg.source_units import (
    FormulaPackage,
    TableRecord,
    build_source_units,
)


class OntologyV2Tests(unittest.TestCase):
    def test_schema_quality_gate(self) -> None:
        report = schema_quality_report()
        self.assertTrue(report["passed"])
        self.assertEqual(report["domain_entity_count"], 15)
        self.assertEqual(report["relation_count"], 14)

    def test_relation_directions_are_strict(self) -> None:
        self.assertTrue(
            validate_relation_v2(
                EntityTypeV2.ASSESSMENT_INDICATOR,
                RelationTypeV2.CALCULATED_BY,
                EntityTypeV2.FORMULA,
            )[0]
        )
        self.assertTrue(
            validate_relation_v2(
                EntityTypeV2.FORMULA,
                RelationTypeV2.HAS_INPUT,
                EntityTypeV2.MODEL_VARIABLE,
            )[0]
        )
        self.assertFalse(
            validate_relation_v2(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.CALCULATED_BY,
                EntityTypeV2.FORMULA,
            )[0]
        )
        self.assertFalse(
            validate_relation_v2(
                DocumentObjectType.STANDARD,
                RelationTypeV2.HAS_INDICATOR,
                EntityTypeV2.ASSESSMENT_INDICATOR,
            )[0]
        )

    def test_legacy_ambiguity_is_not_auto_migrated(self) -> None:
        status = legacy_migration_status("indicator")
        self.assertEqual(status["status"], "manual_review_required")
        self.assertGreaterEqual(len(status["candidates"]), 2)
        self.assertEqual(
            legacy_migration_status("defined_in")["candidates"], []
        )

    def test_coordinate_level_provenance_is_required(self) -> None:
        valid = ProvenanceV2(
            standard_code="HJ 1173-2021",
            source_sha256="sha",
            source_unit_id="unit",
            evidence_span_ids=["span"],
            pages=[7],
            sections=["A.1"],
            bboxes={7: [[100.0, 200.0, 300.0, 220.0]]},
        )
        self.assertTrue(validate_provenance_v2(valid)[0])
        valid.bboxes = {}
        self.assertFalse(validate_provenance_v2(valid)[0])


class ParserPilotV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workspace = Path(__file__).resolve().parents[2]
        cls.source_dir = workspace / "全国生态状况调查评估技术规范"
        if not cls.source_dir.exists():
            raise unittest.SkipTest("local HJ standard corpus is unavailable")

        cls.config = read_json(default_pilot_path())
        required_codes = {
            case["standard_code"] for case in cls.config["cases"]
        }
        corpus = {
            item.standard_code: item
            for item in discover_corpus(cls.source_dir)
            if item.standard_code in required_codes
        }
        cls.documents = {}
        cls.units = {}
        for code in sorted(required_codes):
            item = corpus[code]
            document = parse_pdf_layout(
                item.path,
                item.standard_code,
                item.title,
                item.sha256,
            )
            cls.documents[code] = document
            cls.units[code] = build_source_units(document)

    def test_all_15_manual_pilot_pages_pass(self) -> None:
        report = evaluate_parser_quality(
            self.documents, self.units, self.config
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["case_count"], 15)
        self.assertEqual(
            report["passed_check_count"], report["check_count"]
        )

    def test_formula_package_restores_subscripted_variables(self) -> None:
        packages = [
            unit
            for unit in self.units["HJ 1173-2021"]
            if isinstance(unit, FormulaPackage)
            and any(
                formula.formula_number == "A.1"
                for formula in unit.formulas
            )
        ]
        self.assertEqual(len(packages), 1)
        symbols = {
            variable.symbol
            for variable in packages[0].variable_definitions
        }
        self.assertTrue(
            {"Q_wr", "A_i", "P_i", "R_i", "ET_i"}.issubset(symbols)
        )

    def test_cross_page_table_drops_repeated_header(self) -> None:
        records = [
            unit
            for unit in self.units["HJ 1167-2021"]
            if isinstance(unit, TableRecord)
            and unit.provenance.pages == [7]
        ]
        self.assertFalse(
            any(
                record.cells.get("观测指标") == "观测指标"
                for record in records
            )
        )
        tree_height = next(
            record
            for record in records
            if record.cells.get("观测指标") == "树高"
        )
        self.assertEqual(tree_height.cells["观测时间"], "7—9月")
        self.assertEqual(tree_height.cells["观测频度"], "一年一次")
        bbox = tree_height.provenance.evidence_spans[0].bbox
        self.assertLess(bbox[3] - bbox[1], 40)

    def test_merged_cell_inheritance_keeps_source_coordinates(self) -> None:
        record = next(
            unit
            for unit in self.units["HJ 1173-2021"]
            if isinstance(unit, TableRecord)
            and unit.cells.get("评估指标") == "物种丰富度"
        )
        self.assertEqual(record.cells["评估科目"], "生物多样性维护")
        self.assertIn("评估科目", record.inherited_columns)
        source = record.inherited_from["评估科目"]
        self.assertEqual(source["page"], 6)
        self.assertEqual(len(source["bbox"]), 4)


if __name__ == "__main__":
    unittest.main()
