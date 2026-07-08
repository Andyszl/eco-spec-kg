from __future__ import annotations

import re
from dataclasses import dataclass

from .evidence import validate_evidence
from .io_utils import normalize_space, stable_id
from .models import DocumentChunk, Relation, SourceRef
from .schema import EntityType, RelationType


@dataclass(slots=True)
class ExtractionResult:
    accepted: list[Relation]
    rejected: list[dict[str, str]]


class RuleExtractor:
    """Deterministic baseline. It is intentionally narrow and auditable."""

    _uses_formula = re.compile(
        r"运用(?P<formula>[^，。；]{2,45}?)(?:计算|估算)"
        r"(?P<indicator>[^，。；]{2,32}?)(?:，|。|；|具体)"
    )
    _quality_formula = re.compile(
        r"(?P<indicator>生态系统质量).*?(?:计算公式为|构建.*?公式).*?(?P<formula>EQI)"
    )
    _quality_dependencies = re.compile(
        r"生态系统质量由(?P<parameters>[^。；]{4,90}?)的?相对密度"
    )

    def extract(self, chunks: list[DocumentChunk]) -> ExtractionResult:
        accepted: list[Relation] = []
        rejected: list[dict[str, str]] = []
        for chunk in chunks:
            candidates = self._extract_chunk(chunk)
            for relation in candidates:
                valid, reason = validate_evidence(relation, chunk)
                if valid:
                    accepted.append(relation)
                else:
                    rejected.append(
                        {"relation_id": relation.relation_id, "reason": reason}
                    )
        return ExtractionResult(accepted=accepted, rejected=rejected)

    def _extract_chunk(self, chunk: DocumentChunk) -> list[Relation]:
        text = normalize_space(chunk.text)
        relations: list[Relation] = []
        for match in self._uses_formula.finditer(text):
            formula = self._clean_formula(match.group("formula"))
            indicator = self._clean_indicator(match.group("indicator"))
            evidence = match.group(0)
            relations.append(
                self._relation(
                    chunk,
                    head_name=indicator,
                    head_type=EntityType.INDICATOR,
                    relation_type=RelationType.USES_FORMULA,
                    tail_name=formula,
                    tail_type=EntityType.FORMULA,
                    evidence_text=evidence,
                )
            )

        quality_match = self._quality_formula.search(text)
        if quality_match:
            relations.append(
                self._relation(
                    chunk,
                    head_name=quality_match.group("indicator"),
                    head_type=EntityType.INDICATOR,
                    relation_type=RelationType.USES_FORMULA,
                    tail_name=quality_match.group("formula"),
                    tail_type=EntityType.FORMULA,
                    evidence_text=quality_match.group(0),
                )
            )

        dependency_match = self._quality_dependencies.search(text)
        if dependency_match:
            raw = dependency_match.group("parameters")
            parts = re.split(r"[、，和及与]", raw)
            for parameter in (part.strip() for part in parts):
                if len(parameter) < 2:
                    continue
                relations.append(
                    self._relation(
                        chunk,
                        head_name="生态系统质量",
                        head_type=EntityType.INDICATOR,
                        relation_type=RelationType.DEPENDS_ON,
                        tail_name=parameter,
                        tail_type=EntityType.PARAMETER,
                        evidence_text=dependency_match.group(0),
                    )
                )
        return relations

    @staticmethod
    def _clean_formula(value: str) -> str:
        value = value.strip(" ：:")
        return re.sub(r"^(?:修正|通用)(?=水量平衡方程$)", "", value)

    @staticmethod
    def _clean_indicator(value: str) -> str:
        return re.sub(r"^(?:生态系统的?)", "", value.strip(" ：:"))

    @staticmethod
    def _relation(
        chunk: DocumentChunk,
        head_name: str,
        head_type: EntityType,
        relation_type: RelationType,
        tail_name: str,
        tail_type: EntityType,
        evidence_text: str,
    ) -> Relation:
        head_id = stable_id(head_type.value, head_name)
        tail_id = stable_id(tail_type.value, tail_name)
        relation_id = stable_id(
            chunk.standard_code,
            chunk.page,
            head_id,
            relation_type.value,
            tail_id,
        )
        return Relation(
            relation_id=relation_id,
            path_id=chunk.path_id,
            head_id=head_id,
            head_name=head_name,
            head_type=head_type.value,
            relation_type=relation_type.value,
            tail_id=tail_id,
            tail_name=tail_name,
            tail_type=tail_type.value,
            evidence=SourceRef(
                standard_code=chunk.standard_code,
                page=chunk.page,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                evidence_text=evidence_text,
                source_sha256=chunk.source_sha256,
            ),
            extraction_method="rule",
            confidence=1.0,
        )

