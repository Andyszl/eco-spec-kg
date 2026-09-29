from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ecospec_kg.analysis_v2 import (
    analyze_errors_v2,
    analyze_training_v2,
    collect_error_details_v2,
)
from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.experiment_data_v2 import prepare_experiment_package_v2
from ecospec_kg.extractor_v2 import (
    RuleCandidateExtractorV2,
    build_llm_selection_messages,
    extract_v2,
)
from ecospec_kg.io_utils import read_json, read_jsonl, stable_id, write_json, write_jsonl
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


def human_review_provenance() -> dict:
    return {
        "schema_version": "ecospec-review-provenance-v2.0",
        "gold_nature": "human_expert_gold",
        "reviewers": [
            {
                "reviewer_id": "fixture-human-a",
                "reviewer_type": "human_domain_expert",
                "human_expert": True,
                "qualification_summary": "test fixture ecology expert",
                "identity_verification_reference": "fixture://human-a",
                "signed_at": "2026-08-21T00:00:00+00:00",
            },
            {
                "reviewer_id": "fixture-human-b",
                "reviewer_type": "human_domain_expert",
                "human_expert": True,
                "qualification_summary": "test fixture ecology expert",
                "identity_verification_reference": "fixture://human-b",
                "signed_at": "2026-08-21T00:00:00+00:00",
            },
        ],
        "adjudication": {"adjudicator_id": "fixture-human-a"},
    }


def ai_review_provenance() -> dict:
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
        "adjudication": {"adjudicator_id": "fixture-ai-c"},
    }


class ExperimentChainV2Tests(unittest.TestCase):
    def test_error_analysis_reconstructs_details_and_rejects_test_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            units_path = root / "dev_units.jsonl"
            gold_path = root / "dev_annotations.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(units_path, [unit])
            annotation = {
                **empty_annotation(unit["unit_id"]),
                "split": "dev",
                "entities": [
                    {
                        "name": "P_ij",
                        "entity_type": "model_variable",
                        "evidence_span_ids": ["span-formula"],
                    }
                ],
            }
            write_jsonl(gold_path, [annotation])
            write_jsonl(
                predictions_path,
                [
                    {
                        "unit_id": unit["unit_id"],
                        "entities": [],
                        "relations": [],
                        "candidate_hash": sha256_json({
                            k: RuleCandidateExtractorV2().predict_unit(unit)[k]
                            for k in ("entities", "relations")
                        }),
                    }
                ],
            )

            details = collect_error_details_v2(
                units_path, gold_path, predictions_path
            )
            self.assertEqual(len(details), 1)
            self.assertEqual(details[0]["error_type"], "entity_false_negative")
            self.assertEqual(details[0]["entity_name"], "P_ij")
            self.assertIn("P_ij = S_ij / TS", details[0]["source_text"])
            self.assertEqual(details[0]["review_reason"], "")
            self.assertTrue(details[0]["candidate_present"])
            self.assertTrue(details[0]["candidate_id"])
            self.assertEqual(
                details[0]["selection_result"], "candidate_not_selected"
            )

            annotation["split"] = "test"
            write_jsonl(gold_path, [annotation])
            with self.assertRaisesRegex(ValueError, "allowed gold splits"):
                collect_error_details_v2(units_path, gold_path, predictions_path)

    def test_training_analysis_reports_candidate_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            units_path = root / "train_units.jsonl"
            annotations_path = root / "train_annotations.jsonl"
            write_jsonl(units_path, [unit])
            annotation = {**empty_annotation(unit["unit_id"]), "split": "train"}
            write_jsonl(annotations_path, [annotation])

            with patch("ecospec_kg.analysis_v2._write_workbook") as writer:
                report = analyze_training_v2(
                    units_path, annotations_path, root / "analysis"
                )

            self.assertEqual(report["unit_count"], 1)
            self.assertEqual(report["no_relation_unit_count"], 1)
            self.assertEqual(report["candidate_generator"], "structure-aware-rule-v2.4")
            self.assertGreater(report["entity_candidate_negative_count"], 0)
            self.assertTrue((root / "analysis" / "training_distribution.json").exists())
            writer.assert_called_once()

    def test_error_analysis_splits_relation_false_positive_and_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            units_path = root / "dev_units.jsonl"
            gold_path = root / "dev_annotations.jsonl"
            predictions_path = root / "predictions.jsonl"
            write_jsonl(units_path, [unit])
            annotation = {
                **empty_annotation(unit["unit_id"]),
                "split": "dev",
                "relations": [
                    {
                        "head_name": "构成比例",
                        "head_type": "assessment_indicator",
                        "relation_type": "constrained_by",
                        "tail_name": "质量要求A",
                        "tail_type": "quality_rule",
                    }
                ],
            }
            prediction = {
                "unit_id": unit["unit_id"],
                "entities": [],
                "relations": [
                    {
                        "head_name": "构成比例",
                        "head_type": "assessment_indicator",
                        "relation_type": "constrained_by",
                        "tail_name": "质量要求B",
                        "tail_type": "quality_rule",
                    }
                ],
            }
            write_jsonl(gold_path, [annotation])
            write_jsonl(predictions_path, [prediction])

            with patch("ecospec_kg.analysis_v2._write_workbook"):
                report = analyze_errors_v2(
                    units_path, gold_path, predictions_path, root / "analysis"
                )

            self.assertEqual(
                report["relation_errors_by_type_and_direction"]["constrained_by"],
                {"false_positive": 1, "false_negative": 1, "total": 2},
            )

    def test_rule_candidates_cover_assessment_methods_and_nested_context(self) -> None:
        unit = {
            "schema_version": "source-unit-v2.0",
            "unit_id": "unit-assessment-methods",
            "unit_type": "procedure_clause",
            "provenance": {
                "standard_code": "HJ 1172-2021",
                "document_title": "生态系统质量评估",
                "pages": [10],
                "section": "B.1",
                "heading_chain": ["B.1 叶面积指数"],
                "evidence_spans": [
                    {
                        "span_id": "span-methods",
                        "page": 10,
                        "bbox": [10.0, 20.0, 200.0, 80.0],
                    }
                ],
            },
            "clause_text": (
                "叶面积指数反映陆地生态系统。目前基于光学数据获取叶面积指数的方法"
                "主要包括两类，一类是统计方法，常用的是建立叶面积指数与植被指数之间"
                "经验或半经验关系；一类是基于辐射传输模型的遥感反演方法。"
            ),
        }

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        entity_keys = {
            (entity["name"], entity["entity_type"])
            for entity in prediction["entities"]
        }
        for method in (
            "基于光学数据获取叶面积指数的方法",
            "统计方法",
            "建立叶面积指数与植被指数之间经验或半经验关系",
            "基于辐射传输模型的遥感反演方法",
        ):
            self.assertIn((method, "method"), entity_keys)
        self.assertIn(("光学数据", "data_source"), entity_keys)
        self.assertIn(("陆地生态系统", "ecosystem_type"), entity_keys)
        self.assertIn(("叶面积指数", "assessment_indicator"), entity_keys)
        relation_keys = {
            (
                relation["head_name"],
                relation["relation_type"],
                relation["tail_name"],
            )
            for relation in prediction["relations"]
        }
        self.assertIn(
            ("叶面积指数", "applies_to_ecosystem", "陆地生态系统"),
            relation_keys,
        )

        system, prompt = build_llm_selection_messages(unit, prediction)
        self.assertIn("只输出单行紧凑JSON", system)
        payload = json.loads(prompt)
        self.assertEqual(
            payload["selection_policy"]["version"], "ecospec-selection-v2.4"
        )
        self.assertEqual(
            set(payload["selection_policy"]["rules"]),
            {
                "entity_boundary",
                "formula_relation_recall",
                "independent_candidate_decision",
                "method_recall_precision",
                "relation_precision",
                "same_source_unit",
            },
        )
        self.assertNotIn("provenance", payload["source_unit"])
        self.assertNotIn("adjacent_source_text", payload["source_unit"])
        self.assertNotIn("evidence_span_ids", payload["candidate_entities"][0])
        self.assertEqual(
            set(payload["candidate_relations"][0]), {"id", "head", "type", "tail"}
        )
        entity_by_name = {
            item["name"]: item for item in payload["candidate_entities"]
        }
        self.assertEqual(entity_by_name["统计方法"]["support_scope"], "body")
        self.assertEqual(
            entity_by_name["B.1 叶面积指数"]["boundary"],
            "structure_prefixed",
        )
        self.assertEqual(
            entity_by_name["生态系统质量"]["support_scope"],
            "context_only",
        )
        focus = payload["selection_focus"]
        self.assertIn(
            entity_by_name["统计方法"]["id"], focus["method_entity_ids"]
        )
        self.assertIn(
            entity_by_name["B.1 叶面积指数"]["id"],
            focus["boundary_risk_entity_ids"],
        )
        self.assertTrue(focus["has_indicator_relation_ids"])
        self.assertEqual(
            focus["formula_relation_ids"],
            [
                item["id"]
                for item in payload["candidate_relations"]
                if item["type"]
                in {"has_input", "has_output", "calculated_by", "constrained_by"}
            ],
        )
        self.assertTrue(focus["precision_relation_ids"])

    def test_formula_number_is_not_treated_as_an_input_variable(self) -> None:
        unit = source_unit()
        unit["unit_id"] = "unit-formula-number"
        unit["provenance"]["standard_code"] = "HJ 1172-2021"
        unit["provenance"]["document_title"] = "生态系统质量评估"
        unit["provenance"]["section"] = "B.1"
        unit["provenance"]["heading_chain"] = ["附录 B", "B.1 叶面积指数（LAI）"]
        unit["formulas"] = [
            {
                "formula_number": "B.3",
                "expression_text": "L = -1/2Aln(1-x) （B.3）",
                "evidence_span": unit["provenance"]["evidence_spans"][0],
            }
        ]
        variable_span = unit["provenance"]["evidence_spans"][1]
        unit["variable_definitions"] = [
            {"symbol": "L", "definition": "叶面积指数", "unit": "", "evidence_span": variable_span},
            {"symbol": "x", "definition": "植被指数", "unit": "", "evidence_span": variable_span},
            {"symbol": "A、B", "definition": "经验参数", "unit": "", "evidence_span": variable_span},
        ]

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        relation_keys = {
            (item["head_name"], item["relation_type"], item["tail_name"])
            for item in prediction["relations"]
        }
        self.assertIn(("公式（B.3）", "has_input", "x"), relation_keys)
        self.assertIn(("公式（B.3）", "has_input", "A"), relation_keys)
        self.assertNotIn(("公式（B.3）", "has_input", "B"), relation_keys)

        _, prompt = build_llm_selection_messages(unit, prediction)
        focus = json.loads(prompt)["selection_focus"]
        expected_formula_ids = {
            item["relation_id"]
            for item in prediction["relations"]
            if item["relation_type"]
            in {"has_input", "has_output", "calculated_by", "constrained_by"}
        }
        self.assertEqual(set(focus["formula_relation_ids"]), expected_formula_ids)

    def test_formula_source_keywords_link_only_related_variables(self) -> None:
        unit = source_unit()
        unit["unit_id"] = "unit-formula-source"
        unit["provenance"]["standard_code"] = "HJ 1173-2021"
        unit["provenance"]["document_title"] = "生态系统服务功能评估"
        unit["provenance"]["section"] = "A.2"
        unit["provenance"]["heading_chain"] = ["附录 A", "A.2 土壤保持量"]
        unit["formulas"][0]["formula_number"] = "A.6"
        unit["formulas"][0]["expression_text"] = "R = P + alpha （A.6）"
        variable_span = unit["provenance"]["evidence_spans"][1]
        unit["variable_definitions"] = [
            {"symbol": "R", "definition": "多年平均年降雨侵蚀力", "unit": "", "evidence_span": variable_span},
            {"symbol": "P", "definition": "侵蚀性日降雨量", "unit": "mm", "evidence_span": variable_span},
            {"symbol": "alpha", "definition": "经验参数", "unit": "", "evidence_span": variable_span},
        ]
        unit["adjacent_source_text"] = "降雨侵蚀力空间数据根据逐日降雨量资料获得。"

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        relation_keys = {
            (item["head_name"], item["relation_type"], item["tail_name"])
            for item in prediction["relations"]
        }
        self.assertIn(("R", "sourced_from", "降雨量资料"), relation_keys)
        self.assertIn(("P", "sourced_from", "降雨量资料"), relation_keys)
        self.assertNotIn(("alpha", "sourced_from", "降雨量资料"), relation_keys)

        _, prompt = build_llm_selection_messages(unit, prediction)
        payload = json.loads(prompt)
        self.assertNotIn("adjacent_source_text", payload["source_unit"])

    def test_dynamic_operational_spatial_scopes_are_candidates(self) -> None:
        unit = {
            "schema_version": "source-unit-v2.0",
            "unit_id": "unit-spatial-scopes",
            "unit_type": "procedure_clause",
            "provenance": {
                "standard_code": "HJ 1172-2021",
                "document_title": "生态系统质量评估",
                "pages": [8],
                "section": "B.1",
                "heading_chain": ["B.1 叶面积指数"],
                "evidence_spans": [
                    {"span_id": "span-space", "page": 8, "bbox": [10, 20, 200, 80]}
                ],
            },
            "clause_text": "叶面积指数应在乔木样方、第j分区和林内进行计算。",
        }
        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        names = {
            item["name"]
            for item in prediction["entities"]
            if item["entity_type"] == "spatial_scope"
        }
        self.assertTrue({"乔木样方", "第j分区", "林内"} <= names)

    def test_constraints_and_space_apply_only_to_overall_method(self) -> None:
        unit = {
            "schema_version": "source-unit-v2.0",
            "unit_id": "unit-method-scope",
            "unit_type": "procedure_clause",
            "provenance": {
                "standard_code": "HJ 1172-2021",
                "document_title": "生态系统质量评估",
                "pages": [8],
                "section": "B.1",
                "heading_chain": ["附录 B", "B.1 叶面积指数（LAI）"],
                "evidence_spans": [
                    {"span_id": "span-scope", "page": 8, "bbox": [10, 20, 200, 80]}
                ],
            },
            "clause_text": (
                "（2）冠层模型\n冠层模型通常可划分为四类：参数模型、几何光学模型、"
                "混合介质模型和计算机模拟模型。这些模型已得到广泛应用，目前基于冠层"
                "模型估算叶面积指数常采用反演优化算法、神经网络技术、遗传算法、"
                "贝叶斯网络算法和查找表方法等，可根据评估区域和所具备的实际条件"
                "选择合适的模型和方法估算叶面积指数。"
            ),
        }

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        entity_names = {item["name"] for item in prediction["entities"]}
        relation_keys = {
            (item["head_name"], item["relation_type"], item["tail_name"])
            for item in prediction["relations"]
        }
        rule = "根据评估区域和实际条件选择合适的模型和方法"
        overall = "基于冠层模型估算叶面积指数"
        self.assertIn((overall, "constrained_by", rule), relation_keys)
        self.assertIn((overall, "applies_to_space", "评估区域"), relation_keys)
        for child in ("冠层模型", "参数模型", "神经网络技术", "查找表方法"):
            self.assertNotIn((child, "constrained_by", rule), relation_keys)
            self.assertNotIn((child, "applies_to_space", "评估区域"), relation_keys)
        self.assertNotIn("这些模型", entity_names)
        self.assertNotIn("（2）冠层模型冠层模型", entity_names)

    def test_method_candidate_rejects_discourse_prefix(self) -> None:
        unit = {
            "schema_version": "source-unit-v2.0",
            "unit_id": "unit-method-boundary",
            "unit_type": "procedure_clause",
            "provenance": {
                "standard_code": "HJ 1172-2021",
                "document_title": "生态系统质量评估",
                "pages": [15],
                "section": "B.2",
                "heading_chain": ["附录 B", "B.2 植被覆盖度（FVC）"],
                "evidence_spans": [
                    {"span_id": "span-method", "page": 15, "bbox": [10, 20, 200, 80]}
                ],
            },
            "clause_text": (
                "（4）其他方法除了上述常用植被覆盖度遥感估算方法，"
                "主要还有物理模型法、光谱梯度差法等。"
            ),
        }

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        names = {
            item["name"]
            for item in prediction["entities"]
            if item["entity_type"] == "method"
        }
        self.assertNotIn("其他方法除了上述常用植被覆盖度遥感估算方法", names)
        self.assertIn("物理模型法", names)

    def test_rule_candidates_read_ecosystem_classification_columns(self) -> None:
        unit = {
            "schema_version": "source-unit-v2.0",
            "unit_id": "unit-ecosystem-table",
            "unit_type": "table_record",
            "provenance": {
                "standard_code": "HJ 1166-2021",
                "document_title": "生态系统分类",
                "pages": [8],
                "section": "附录 A",
                "heading_chain": ["附录 A"],
                "evidence_spans": [
                    {
                        "span_id": "span-table",
                        "page": 8,
                        "bbox": [10.0, 20.0, 200.0, 80.0],
                    }
                ],
            },
            "cells": {
                "Ⅰ级分类": "森林生态系统",
                "Ⅱ级分类": "阔叶林",
                "分类依据": "H=3～30 m，C≥0.2，阔叶",
            },
        }

        prediction = RuleCandidateExtractorV2().predict_unit(unit)
        names = {
            entity["name"]
            for entity in prediction["entities"]
            if entity["entity_type"] == "ecosystem_type"
        }
        self.assertTrue(
            {"森林生态系统", "阔叶林", "H=3～30 m，C≥0.2，阔叶"}
            <= names
        )

    def test_prepare_lora_v2_matches_selector_task_and_rejects_test_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units_path = root / "train_units.jsonl"
            write_jsonl(units_path, [source_unit()])
            run = root / "rule"
            extract_v2(units_path, run)
            resolved_config = json.loads((run / "resolved_config.json").read_text())
            self.assertEqual(
                resolved_config["selection_policy"], "ecospec-selection-v2.4"
            )
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
            self.assertEqual(
                manifest["candidate_generator"], "structure-aware-rule-v2.4"
            )
            self.assertEqual(
                manifest["selection_policy_version"], "ecospec-selection-v2.4"
            )
            self.assertEqual(
                manifest["schema_version"], "ecospec-lora-training-v2.2"
            )
            focus = manifest["selection_focus_coverage"]
            self.assertGreater(focus["has_indicator"]["candidate_count"], 0)
            self.assertGreater(focus["context_only_entities"]["candidate_count"], 0)
            self.assertGreater(focus["formula_relations"]["candidate_count"], 0)
            self.assertGreater(focus["precision_relations"]["candidate_count"], 0)
            self.assertEqual(
                focus["has_indicator"]["candidate_count"],
                focus["has_indicator"]["selected_count"],
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

    def test_human_expert_gold_requires_review_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            source_path = root / "source.jsonl"
            annotations_path = root / "annotations.jsonl"
            write_jsonl(source_path, [unit])
            write_jsonl(annotations_path, [empty_annotation(unit["unit_id"])])

            with self.assertRaisesRegex(ValueError, "requires --review-provenance"):
                prepare_experiment_package_v2(
                    source_path,
                    annotations_path,
                    root / "package",
                    gold_nature="human_expert_gold",
                )

    def test_prepare_rejects_ai_provenance_for_human_expert_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            source_path = root / "source.jsonl"
            annotations_path = root / "annotations.jsonl"
            provenance_path = root / "ai-review-provenance.json"
            write_jsonl(source_path, [unit])
            write_jsonl(annotations_path, [empty_annotation(unit["unit_id"])])
            provenance = ai_review_provenance()
            provenance["gold_nature"] = "human_expert_gold"
            write_json(provenance_path, provenance)

            with self.assertRaisesRegex(
                ValueError, "at least two distinct human domain experts"
            ):
                prepare_experiment_package_v2(
                    source_path,
                    annotations_path,
                    root / "package",
                    gold_nature="human_expert_gold",
                    review_provenance_path=provenance_path,
                )

    def test_prepare_records_ai_adjudication_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = source_unit()
            source_path = root / "source.jsonl"
            annotations_path = root / "annotations.jsonl"
            provenance_path = root / "ai-review-provenance.json"
            write_jsonl(source_path, [unit])
            write_jsonl(annotations_path, [empty_annotation(unit["unit_id"])])
            write_json(provenance_path, ai_review_provenance())

            manifest = prepare_experiment_package_v2(
                source_path,
                annotations_path,
                root / "package",
                gold_nature="ai_expert_adjudicated_gold",
                review_provenance_path=provenance_path,
            )

            self.assertEqual(
                manifest["review_provenance"]["reviewer_types"],
                ["ai_simulated_domain_reviewer"],
            )
            self.assertTrue(
                manifest["human_expert_review_required_for_publication"]
            )
            self.assertTrue((root / "package" / "review_provenance.json").is_file())

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
            provenance_path = root / "human-review-provenance.json"
            write_json(provenance_path, human_review_provenance())
            package_dir = root / "package"
            prepare_experiment_package_v2(
                source_path,
                annotations_path,
                package_dir,
                gold_nature="human_expert_gold",
                review_provenance_path=provenance_path,
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
