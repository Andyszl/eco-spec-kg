from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceRef:
    standard_code: str
    page: int
    section: str
    chunk_id: str
    evidence_text: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return cls(**data)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    standard_code: str
    document_title: str
    page: int
    section: str
    block_type: str
    text: str
    source_file: str
    source_sha256: str
    path_id: str
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


@dataclass(slots=True)
class Entity:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_refs"] = [item.to_dict() for item in self.source_refs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        refs = [SourceRef.from_dict(item) for item in data.get("source_refs", [])]
        return cls(
            entity_id=data["entity_id"],
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            aliases=list(data.get("aliases", [])),
            source_refs=refs,
        )


@dataclass(slots=True)
class Relation:
    relation_id: str
    path_id: str
    head_id: str
    head_name: str
    head_type: str
    relation_type: str
    tail_id: str
    tail_name: str
    tail_type: str
    evidence: SourceRef
    extraction_method: str
    confidence: float
    review_status: str = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = self.evidence.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        payload = dict(data)
        payload["evidence"] = SourceRef.from_dict(payload["evidence"])
        return cls(**payload)


@dataclass(slots=True)
class AnnotationRecord:
    record_id: str
    relation: Relation
    reviewer_1_decision: str = ""
    reviewer_1_comment: str = ""
    reviewer_2_decision: str = ""
    reviewer_2_comment: str = ""
    consensus_decision: str = ""
    review_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relation"] = self.relation.to_dict()
        return data

