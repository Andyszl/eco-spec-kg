from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ecospec_kg.corpus import extract_standard_code
from ecospec_kg.evidence import validate_evidence
from ecospec_kg.experiments import extraction_metrics, ranking_metrics
from ecospec_kg.extraction import LLMExtractor, RuleExtractor
from ecospec_kg.graph import GraphIndex
from ecospec_kg.io_utils import read_jsonl
from ecospec_kg.models import DocumentChunk, Relation
from ecospec_kg.providers import MockProvider
from ecospec_kg.schema import EntityType, RelationType, validate_relation
from ecospec_kg.split import group_split


FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "mini_chunks.jsonl"


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [DocumentChunk.from_dict(row) for row in read_jsonl(FIXTURE)]
        self.relations = RuleExtractor().extract(self.chunks).accepted

    def test_standard_code(self) -> None:
        self.assertEqual(extract_standard_code("HJ 1173—2021"), "HJ 1173-2021")

    def test_schema(self) -> None:
        self.assertTrue(
            validate_relation(
                EntityType.INDICATOR,
                RelationType.USES_FORMULA,
                EntityType.FORMULA,
            )[0]
        )
        self.assertFalse(
            validate_relation(
                EntityType.UNIT,
                RelationType.USES_FORMULA,
                EntityType.INDICATOR,
            )[0]
        )

    def test_rule_extraction_and_evidence(self) -> None:
        self.assertGreaterEqual(len(self.relations), 4)
        chunk_map = {item.chunk_id: item for item in self.chunks}
        for relation in self.relations:
            self.assertTrue(
                validate_evidence(relation, chunk_map[relation.evidence.chunk_id])[0]
            )

    def test_pdf_compact_evidence_match(self) -> None:
        chunk = DocumentChunk(
            chunk_id="chunk_pdf_space",
            standard_code="HJ 1172-2021",
            document_title="生态系统质量评估",
            page=4,
            section="3",
            block_type="paragraph",
            text="3.4 总初级生产力 gross primary productivity 主要表征植被光合作用能 力强弱。",
            source_file="fixture.pdf",
            source_sha256="fixture",
            path_id="gpp",
        )
        relation = RuleExtractor._relation(
            chunk,
            head_name="总初级生产力",
            head_type=EntityType.INDICATOR,
            relation_type=RelationType.DEFINED_IN,
            tail_name="3.4",
            tail_type=EntityType.STANDARD_CLAUSE,
            evidence_text="3.4 总初级生产力 gross primary productivity 主要表征植被光合作用能力强弱。",
        )
        self.assertTrue(validate_evidence(relation, chunk)[0])

    def test_llm_extraction_and_evidence(self) -> None:
        provider = MockProvider(
            json.dumps(
                {
                    "relations": [
                        {
                            "head_name": "水源涵养量",
                            "head_type": "indicator",
                            "relation_type": "uses_formula",
                            "tail_name": "水量平衡方程",
                            "tail_type": "formula",
                            "evidence_text": "运用水量平衡方程计算水源涵养量",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        result = LLMExtractor(provider).extract([self.chunks[0]])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].extraction_method, "llm_schema")
        self.assertFalse(result.rejected)

    def test_llm_parse_error_records_response(self) -> None:
        result = LLMExtractor(MockProvider("not-json"), retries=0).extract(
            [self.chunks[0]]
        )
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("provider_or_parse_error", result.rejected[0]["reason"])
        self.assertEqual(result.rejected[0]["response_preview"], "not-json")

    def test_group_split_keeps_paths_together(self) -> None:
        split = group_split(self.relations)
        locations: dict[str, str] = {}
        for name, rows in split.items():
            for row in rows:
                previous = locations.setdefault(row.path_id, name)
                self.assertEqual(previous, name)

    def test_metrics(self) -> None:
        metrics = extraction_metrics(self.relations, self.relations)
        self.assertEqual(metrics["f1"], 1.0)
        ranking = ranking_metrics(
            [{"rank": 1, "is_correct": True}, {"rank": 4, "is_correct": True}]
        )
        self.assertEqual(ranking["hits@1"], 0.5)

    def test_graph_round_trip_and_search(self) -> None:
        graph = GraphIndex.build(self.chunks, self.relations)
        self.assertTrue(graph.search("水源涵养"))
        with tempfile.TemporaryDirectory() as directory:
            graph.save(Path(directory))
            loaded = GraphIndex.load(Path(directory) / "index.json")
            self.assertEqual(len(loaded.relations), len(graph.relations))


if __name__ == "__main__":
    unittest.main()
