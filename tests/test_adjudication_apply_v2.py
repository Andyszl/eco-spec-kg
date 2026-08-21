from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecospec_kg.adjudication_apply_v2 import apply_adjudication_v2
from ecospec_kg.io_utils import read_json, read_jsonl, write_json, write_jsonl


def source_unit(unit_id: str, span_id: str) -> dict:
    return {
        "schema_version": "source-unit-v2.0",
        "unit_id": unit_id,
        "unit_type": "procedure_clause",
        "provenance": {
            "standard_code": "HJ 1172-2021",
            "evidence_spans": [
                {
                    "span_id": span_id,
                    "page": 1,
                    "bbox": [10.0, 20.0, 200.0, 40.0],
                    "text": f"evidence for {unit_id}",
                }
            ],
        },
        "clause_text": f"source text for {unit_id}",
    }


def annotation(unit_id: str, *, entities: list[dict] | None = None) -> dict:
    return {
        "annotation_version": "ecospec-annotation-v2.1",
        "ontology_version": "ecospec-ontology-v2.0",
        "unit_id": unit_id,
        "standard_code": "HJ 1172-2021",
        "entities": entities or [],
        "relations": [],
        "review_status": "ai_expert_pre_gold",
        "no_relation_reason": "fixture",
        "notes": "",
    }


def decision_payload() -> dict:
    return {
        "schema_version": "ecospec-review-provenance-v2.0",
        "gold_nature": "ai_expert_adjudicated_gold",
        "claims_human_expert_review": False,
        "reviewers": [
            {
                "reviewer_id": "fixture-ai-c",
                "reviewer_type": "ai_simulated_domain_reviewer",
                "human_expert": False,
            }
        ],
        "adjudication": {
            "adjudicator_id": "fixture-ai-c",
            "decision_count": 4,
            "unresolved_count": 0,
            "status": "ai_adjudicated",
        },
        "decisions": [
            {
                "source_row": 1,
                "unit_id": "unit-add",
                "action": "add_gold_entity",
                "status": "adjudicated",
                "entity": {
                    "name": "森林",
                    "entity_type": "ecosystem_type",
                    "evidence_span_ids": ["span-add"],
                },
            },
            {
                "source_row": 2,
                "unit_id": "unit-remove",
                "action": "remove_gold_entity",
                "status": "adjudicated",
                "entity": {
                    "name": "观测数据拟合",
                    "entity_type": "method",
                    "evidence_span_ids": ["span-remove"],
                },
            },
            {
                "source_row": 3,
                "unit_id": "unit-relation",
                "action": "no_gold_change",
                "status": "adjudicated",
                "relation": {
                    "head_name": "叶面积指数",
                    "head_type": "assessment_indicator",
                    "relation_type": "calculated_by",
                    "tail_name": "公式（B.2）",
                    "tail_type": "formula",
                    "evidence_span_ids": ["span-relation"],
                },
            },
            {
                "source_row": 4,
                "unit_id": "unit-alias",
                "action": "register_alias_only",
                "status": "adjudicated",
                "alias": {
                    "name": "生态系统质量指数（EQI）",
                    "entity_type": "assessment_indicator",
                    "canonical_name": "EQI",
                    "evidence_span_ids": ["span-alias"],
                },
            },
        ],
    }


class ApplyAdjudicationV2Tests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        units = [
            source_unit("unit-add", "span-add"),
            source_unit("unit-remove", "span-remove"),
            source_unit("unit-relation", "span-relation"),
            source_unit("unit-alias", "span-alias"),
        ]
        remove_entity = {
            "entity_id": "entity-remove",
            "name": "观测数据拟合",
            "entity_type": "method",
            "evidence_span_ids": ["span-remove"],
        }
        relation_head = {
            "entity_id": "entity-head",
            "name": "叶面积指数",
            "entity_type": "assessment_indicator",
            "evidence_span_ids": ["span-relation"],
        }
        relation_tail = {
            "entity_id": "entity-tail",
            "name": "公式（B.2）",
            "entity_type": "formula",
            "evidence_span_ids": ["span-relation"],
        }
        annotations = [
            annotation("unit-add"),
            annotation("unit-remove", entities=[remove_entity]),
            {
                **annotation(
                    "unit-relation", entities=[relation_head, relation_tail]
                ),
                "relations": [
                    {
                        "relation_id": "relation-existing",
                        "head_id": "entity-head",
                        "head_name": "叶面积指数",
                        "head_type": "assessment_indicator",
                        "relation_type": "calculated_by",
                        "tail_id": "entity-tail",
                        "tail_name": "公式（B.2）",
                        "tail_type": "formula",
                        "evidence_span_ids": ["span-relation"],
                    }
                ],
            },
            annotation(
                "unit-alias",
                entities=[
                    {
                        "entity_id": "entity-eqi",
                        "name": "EQI",
                        "entity_type": "assessment_indicator",
                        "evidence_span_ids": ["span-alias"],
                    }
                ],
            ),
        ]
        units_path = root / "source_units.jsonl"
        annotations_path = root / "annotations.jsonl"
        decisions_path = root / "decisions.json"
        write_jsonl(units_path, units)
        write_jsonl(annotations_path, annotations)
        write_json(decisions_path, decision_payload())
        return units_path, annotations_path, decisions_path

    def test_applies_only_explicit_changes_and_writes_audit_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units_path, annotations_path, decisions_path = self._write_fixture(root)
            output_path = root / "annotations_v22.jsonl"

            manifest = apply_adjudication_v2(
                units_path, annotations_path, decisions_path, output_path
            )

            rows = {row["unit_id"]: row for row in read_jsonl(output_path)}
            self.assertEqual(manifest["decision_count"], 4)
            self.assertEqual(manifest["modified_unit_count"], 2)
            self.assertEqual(manifest["delta"]["entity_count"], 0)
            self.assertEqual(manifest["delta"]["relation_count"], 0)
            self.assertEqual(
                rows["unit-add"]["annotation_version"],
                "ecospec-annotation-v2.2",
            )
            self.assertEqual(
                rows["unit-remove"]["entities"],
                [],
            )
            self.assertEqual(
                rows["unit-relation"]["annotation_version"],
                "ecospec-annotation-v2.1",
            )
            self.assertEqual(
                rows["unit-alias"]["annotation_version"],
                "ecospec-annotation-v2.1",
            )
            aliases = read_json(root / "annotations_v22.aliases.json")
            self.assertTrue(aliases["strict_gold_unchanged"])
            self.assertEqual(aliases["aliases"][0]["canonical_name"], "EQI")
            self.assertTrue(
                (root / "annotations_v22.adjudication_manifest.json").is_file()
            )

    def test_rejects_unresolved_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units_path, annotations_path, decisions_path = self._write_fixture(root)
            payload = read_json(decisions_path)
            payload["adjudication"]["unresolved_count"] = 1
            write_json(decisions_path, payload)

            with self.assertRaisesRegex(ValueError, "unresolved"):
                apply_adjudication_v2(
                    units_path,
                    annotations_path,
                    decisions_path,
                    root / "annotations_v22.jsonl",
                )

    def test_rejects_unknown_evidence_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units_path, annotations_path, decisions_path = self._write_fixture(root)
            payload = read_json(decisions_path)
            payload["decisions"][0]["entity"]["evidence_span_ids"] = [
                "missing-span"
            ]
            write_json(decisions_path, payload)

            with self.assertRaisesRegex(ValueError, "unknown evidence spans"):
                apply_adjudication_v2(
                    units_path,
                    annotations_path,
                    decisions_path,
                    root / "annotations_v22.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
