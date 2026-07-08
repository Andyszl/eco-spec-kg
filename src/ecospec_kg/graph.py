from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import read_json, write_json
from .models import DocumentChunk, Relation
from .schema import RelationType


def _terms(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.lower())
    chinese = [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]
    latin = re.findall(r"[a-z0-9_@.-]+", text.lower())
    return chinese + latin


@dataclass(slots=True)
class GraphIndex:
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    inverted: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls, chunks: list[DocumentChunk], relations: list[Relation]
    ) -> "GraphIndex":
        inverted: dict[str, dict[str, int]] = defaultdict(dict)
        for chunk in chunks:
            counts = Counter(_terms(chunk.text))
            for term, count in counts.items():
                inverted[term][chunk.chunk_id] = count
        return cls(
            chunks={item.chunk_id: item for item in chunks},
            relations={item.relation_id: item for item in relations},
            inverted=dict(inverted),
        )

    def search(
        self, query: str, limit: int = 10, standard_code: str = ""
    ) -> list[dict[str, object]]:
        query_counts = Counter(_terms(query))
        scores: dict[str, float] = defaultdict(float)
        corpus_size = max(1, len(self.chunks))
        for term, query_count in query_counts.items():
            postings = self.inverted.get(term, {})
            idf = math.log((corpus_size + 1) / (len(postings) + 1)) + 1
            for chunk_id, count in postings.items():
                chunk = self.chunks[chunk_id]
                if standard_code and chunk.standard_code != standard_code:
                    continue
                scores[chunk_id] += query_count * count * idf
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            {
                "score": round(score, 6),
                **self.chunks[chunk_id].to_dict(),
            }
            for chunk_id, score in ranked
        ]

    def entity_path(self, name: str, max_depth: int = 4) -> list[dict[str, str]]:
        starts = {
            relation.head_id
            for relation in self.relations.values()
            if relation.head_name == name
        } | {
            relation.tail_id
            for relation in self.relations.values()
            if relation.tail_name == name
        }
        if not starts:
            return []
        adjacency: dict[str, list[Relation]] = defaultdict(list)
        for relation in self.relations.values():
            adjacency[relation.head_id].append(relation)
        output: list[dict[str, str]] = []
        queue = deque((start, 0) for start in starts)
        visited: set[tuple[str, str]] = set()
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for relation in adjacency.get(node, []):
                key = (relation.relation_id, relation.tail_id)
                if key in visited:
                    continue
                visited.add(key)
                output.append(
                    {
                        "head": relation.head_name,
                        "relation": relation.relation_type,
                        "tail": relation.tail_name,
                        "standard_code": relation.evidence.standard_code,
                        "page": str(relation.evidence.page),
                        "section": relation.evidence.section,
                    }
                )
                queue.append((relation.tail_id, depth + 1))
        return output

    def missing_requirements(self, indicator: str) -> list[str]:
        outgoing = [
            relation
            for relation in self.relations.values()
            if relation.head_name == indicator
        ]
        relation_types = {relation.relation_type for relation in outgoing}
        required = {
            RelationType.USES_FORMULA.value: "计算公式",
            RelationType.DEFINED_IN.value: "标准条款溯源",
        }
        missing = [label for relation, label in required.items() if relation not in relation_types]
        if not any(
            relation.relation_type
            in {RelationType.DEPENDS_ON.value, RelationType.DERIVED_FROM.value}
            for relation in outgoing
        ):
            missing.append("输入参数或依赖指标")
        return missing

    def save(self, output_dir: Path) -> None:
        payload = {
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
            "relations": [
                relation.to_dict() for relation in self.relations.values()
            ],
        }
        write_json(output_dir / "index.json", payload)

    @classmethod
    def load(cls, path: Path) -> "GraphIndex":
        payload = read_json(path)
        chunks = [DocumentChunk.from_dict(item) for item in payload["chunks"]]
        relations = [Relation.from_dict(item) for item in payload["relations"]]
        return cls.build(chunks, relations)

