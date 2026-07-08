from __future__ import annotations

import csv
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from .extraction import RuleExtractor
from .graph import GraphIndex
from .io_utils import read_jsonl, write_json, write_jsonl
from .models import DocumentChunk, Relation


class GraphRAGAdapter(ABC):
    @abstractmethod
    def index(
        self,
        chunks: list[DocumentChunk],
        relations: list[Relation],
        output_dir: Path,
        run: bool = False,
    ) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def query(self, index_dir: Path, query: str, limit: int = 10) -> object:
        raise NotImplementedError


class NativeGraphRAGAdapter(GraphRAGAdapter):
    def index(
        self,
        chunks: list[DocumentChunk],
        relations: list[Relation],
        output_dir: Path,
        run: bool = False,
    ) -> dict[str, object]:
        del run
        if not relations:
            relations = RuleExtractor().extract(chunks).accepted
        graph = GraphIndex.build(chunks, relations)
        graph.save(output_dir)
        summary = {
            "adapter": "native",
            "chunk_count": len(chunks),
            "relation_count": len(relations),
            "status": "complete",
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    def query(self, index_dir: Path, query: str, limit: int = 10) -> object:
        graph = GraphIndex.load(index_dir / "index.json")
        return graph.search(query, limit=limit)


class MicrosoftGraphRAGAdapter(GraphRAGAdapter):
    """Isolates the versioned Microsoft GraphRAG CLI behind a stable interface."""

    def index(
        self,
        chunks: list[DocumentChunk],
        relations: list[Relation],
        output_dir: Path,
        run: bool = False,
    ) -> dict[str, object]:
        del relations
        input_dir = output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        with (input_dir / "standards.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "text", "standard_code", "page", "section"],
            )
            writer.writeheader()
            for chunk in chunks:
                writer.writerow(
                    {
                        "id": chunk.chunk_id,
                        "text": chunk.text,
                        "standard_code": chunk.standard_code,
                        "page": chunk.page,
                        "section": chunk.section,
                    }
                )
        (output_dir / "settings.yaml").write_text(
            self._settings_yaml(), encoding="utf-8"
        )
        prompts_dir = output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        (prompts_dir / "extract_graph.txt").write_text(
            self._extract_prompt(), encoding="utf-8"
        )

        status = "prepared"
        if run:
            executable = shutil.which("graphrag")
            if not executable:
                raise RuntimeError(
                    "graphrag CLI is not installed; install the ms-graphrag extra"
                )
            subprocess.run(
                [executable, "index", "--root", str(output_dir)],
                check=True,
            )
            status = "complete"
            self.map_outputs(output_dir)
        summary = {
            "adapter": "microsoft_graphrag",
            "version": "3.1.0",
            "chunk_count": len(chunks),
            "status": status,
        }
        write_json(output_dir / "adapter_summary.json", summary)
        return summary

    def query(self, index_dir: Path, query: str, limit: int = 10) -> object:
        del limit
        executable = shutil.which("graphrag")
        if not executable:
            raise RuntimeError("graphrag CLI is not installed")
        result = subprocess.run(
            [
                executable,
                "query",
                "--root",
                str(index_dir),
                "--method",
                "local",
                "--query",
                query,
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout

    @staticmethod
    def map_outputs(root: Path) -> dict[str, int]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and pyarrow are required to map outputs") from exc
        output_dir = root / "output"
        mapped: dict[str, int] = {}
        for name in ("entities", "relationships", "text_units", "communities"):
            matches = list(output_dir.rglob(f"{name}.parquet"))
            if not matches:
                continue
            frame = pd.read_parquet(matches[0])
            rows = frame.to_dict(orient="records")
            write_jsonl(root / "mapped" / f"{name}.jsonl", rows)
            mapped[name] = len(rows)
        write_json(root / "mapped" / "summary.json", mapped)
        return mapped

    @staticmethod
    def _settings_yaml() -> str:
        return """completion_models:
  default_completion_model:
    model_provider: openai
    model: ${ECOSPEC_COMPLETION_MODEL}
    api_base: ${ECOSPEC_LLM_BASE_URL}
    api_key: ${ECOSPEC_LLM_API_KEY}
embedding_models:
  default_embedding_model:
    model_provider: openai
    model: ${ECOSPEC_EMBEDDING_MODEL}
    api_base: ${ECOSPEC_LLM_BASE_URL}
    api_key: ${ECOSPEC_LLM_API_KEY}
input:
  type: file
  file_type: csv
  base_dir: input
  file_pattern: ".*\\\\.csv$"
  text_column: text
  title_column: id
chunks:
  size: 300
  overlap: 40
extract_graph:
  prompt: prompts/extract_graph.txt
  entity_types:
    - indicator
    - formula
    - parameter
    - unit
    - data_source
    - method
    - ecosystem_type
    - spatial_scope
    - temporal_scope
    - quality_requirement
    - standard_clause
  max_gleanings: 1
"""

    @staticmethod
    def _extract_prompt() -> str:
        return """从生态评估技术规范文本中提取实体和关系。
仅使用给定文本，不补充外部知识。所有关系必须保留原文证据。
实体类型和关系类型必须服从EcoSpec-KG Schema。
输出格式遵循GraphRAG实体和关系抽取约定。
"""


def load_chunks(path: Path) -> list[DocumentChunk]:
    return [DocumentChunk.from_dict(row) for row in read_jsonl(path)]


def load_relations(path: Path | None) -> list[Relation]:
    if path is None or not path.exists():
        return []
    return [Relation.from_dict(row) for row in read_jsonl(path)]

