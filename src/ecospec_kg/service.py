from __future__ import annotations

from pathlib import Path

from .extraction import RuleExtractor
from .graph import GraphIndex
from .io_utils import read_jsonl
from .models import DocumentChunk


class KnowledgeService:
    def __init__(self, graph: GraphIndex) -> None:
        self.graph = graph

    @classmethod
    def from_data_root(cls, data_root: Path) -> "KnowledgeService":
        index_path = data_root / "index" / "native" / "index.json"
        if index_path.exists():
            return cls(GraphIndex.load(index_path))
        chunks_path = data_root / "processed" / "chunks.jsonl"
        if not chunks_path.exists():
            chunks_path = data_root / "fixtures" / "mini_chunks.jsonl"
        chunks = [DocumentChunk.from_dict(row) for row in read_jsonl(chunks_path)]
        relations = RuleExtractor().extract(chunks).accepted
        return cls(GraphIndex.build(chunks, relations))

    @property
    def standards(self) -> list[str]:
        return sorted({chunk.standard_code for chunk in self.graph.chunks.values()})

    def search(self, query: str, standard_code: str = "", limit: int = 10) -> list[dict[str, object]]:
        return self.graph.search(query, limit=limit, standard_code=standard_code)

    def path(self, indicator: str) -> list[dict[str, str]]:
        return self.graph.entity_path(indicator)

    def missing(self, indicator: str) -> dict[str, object]:
        return {
            "indicator": indicator,
            "missing": self.graph.missing_requirements(indicator),
        }

    def evidence(self, query: str, standard_code: str = "") -> list[dict[str, object]]:
        rows = self.search(query, standard_code=standard_code, limit=5)
        return [
            {
                "standard_code": row["standard_code"],
                "page": row["page"],
                "section": row["section"],
                "text": row["text"],
                "source_sha256": row["source_sha256"],
                "score": row["score"],
            }
            for row in rows
        ]

