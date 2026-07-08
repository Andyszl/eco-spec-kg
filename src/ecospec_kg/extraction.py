from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .evidence import validate_evidence
from .io_utils import normalize_space, stable_id
from .models import DocumentChunk, Relation, SourceRef
from .providers import CompletionProvider
from .schema import (
    ENTITY_LABELS_ZH,
    RELATION_LABELS_ZH,
    EntityType,
    RelationType,
    schema_rows,
)


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


LLM_SYSTEM_PROMPT = (
    "你是生态评估技术规范知识抽取助手。只能依据用户提供的规范文本抽取结构化关系，"
    "不得使用外部知识，不得推断文本中没有明示的关系。输出必须是JSON。"
)


class LLMExtractor:
    """Schema-constrained candidate extractor backed by an LLM provider."""

    def __init__(
        self,
        provider: CompletionProvider,
        max_relations_per_chunk: int = 20,
    ) -> None:
        self._provider = provider
        self._max_relations_per_chunk = max_relations_per_chunk

    def extract(self, chunks: list[DocumentChunk]) -> ExtractionResult:
        accepted: list[Relation] = []
        rejected: list[dict[str, str]] = []
        for chunk in chunks:
            prompt = self._build_prompt(chunk)
            try:
                response = self._provider.complete(LLM_SYSTEM_PROMPT, prompt)
                candidates = self._parse_response(response)
            except Exception as exc:  # pragma: no cover - defensive runtime path
                rejected.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "reason": f"provider_or_parse_error: {exc}",
                    }
                )
                continue

            for index, candidate in enumerate(
                candidates[: self._max_relations_per_chunk], start=1
            ):
                try:
                    relation = self._candidate_to_relation(chunk, candidate, index)
                except Exception as exc:
                    rejected.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "reason": f"candidate_error: {exc}",
                            "candidate": json.dumps(candidate, ensure_ascii=False),
                        }
                    )
                    continue
                valid, reason = validate_evidence(relation, chunk)
                if valid:
                    accepted.append(relation)
                else:
                    rejected.append(
                        {
                            "relation_id": relation.relation_id,
                            "chunk_id": chunk.chunk_id,
                            "reason": reason,
                            "candidate": json.dumps(candidate, ensure_ascii=False),
                        }
                    )
        return ExtractionResult(accepted=accepted, rejected=rejected)

    @staticmethod
    def _build_prompt(chunk: DocumentChunk) -> str:
        entity_lines = [
            f"- {entity.value}: {label}"
            for entity, label in ENTITY_LABELS_ZH.items()
        ]
        relation_lines = [
            f"- {relation.value}: {label}"
            for relation, label in RELATION_LABELS_ZH.items()
        ]
        schema_lines = [
            f"- {row['head_type']} --{row['relation_type']}--> {row['tail_type']}"
            for row in schema_rows()
        ]
        return (
            "请从以下生态评估技术规范文本中抽取候选知识图谱关系。\n"
            "要求：\n"
            "1. 只抽取文本中明确出现或直接定义的关系，不要补充常识。\n"
            "2. evidence_text 必须逐字复制原文中的连续片段，不能改写。\n"
            "3. 每条关系必须符合允许的 Schema 组合。\n"
            "4. 若没有可抽取关系，返回 {\"relations\":[]}。\n"
            "5. 不要输出解释、Markdown 或代码块，只输出 JSON。\n\n"
            "实体类型：\n"
            + "\n".join(entity_lines)
            + "\n\n关系类型：\n"
            + "\n".join(relation_lines)
            + "\n\n允许的 Schema：\n"
            + "\n".join(schema_lines)
            + "\n\n输出格式：\n"
            '{"relations":[{"head_name":"","head_type":"","relation_type":"",'
            '"tail_name":"","tail_type":"","evidence_text":"","confidence":0.0}]}\n\n'
            f"标准编号：{chunk.standard_code}\n"
            f"页码：{chunk.page}\n"
            f"章节：{chunk.section}\n"
            f"chunk_id：{chunk.chunk_id}\n"
            "规范文本：\n"
            f"{chunk.text}"
        )

    @staticmethod
    def _parse_response(response: str) -> list[dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(text[start : end + 1])
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        relations = payload.get("relations", [])
        if not isinstance(relations, list):
            raise ValueError("relations must be a list")
        return [item for item in relations if isinstance(item, dict)]

    @staticmethod
    def _candidate_to_relation(
        chunk: DocumentChunk, candidate: dict[str, Any], index: int
    ) -> Relation:
        head_name = str(candidate["head_name"]).strip()
        head_type = str(candidate["head_type"]).strip()
        relation_type = str(candidate["relation_type"]).strip()
        tail_name = str(candidate["tail_name"]).strip()
        tail_type = str(candidate["tail_type"]).strip()
        evidence_text = str(candidate["evidence_text"]).strip()
        confidence = float(candidate.get("confidence", 0.8))
        if not all(
            [head_name, head_type, relation_type, tail_name, tail_type, evidence_text]
        ):
            raise ValueError("empty required field")

        head_id = stable_id(head_type, head_name)
        tail_id = stable_id(tail_type, tail_name)
        relation_id = stable_id(
            chunk.standard_code,
            chunk.page,
            chunk.chunk_id,
            index,
            head_id,
            relation_type,
            tail_id,
        )
        return Relation(
            relation_id=relation_id,
            path_id=chunk.path_id,
            head_id=head_id,
            head_name=head_name,
            head_type=head_type,
            relation_type=relation_type,
            tail_id=tail_id,
            tail_name=tail_name,
            tail_type=tail_type,
            evidence=SourceRef(
                standard_code=chunk.standard_code,
                page=chunk.page,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                evidence_text=evidence_text,
                source_sha256=chunk.source_sha256,
            ),
            extraction_method="llm_schema",
            confidence=max(0.0, min(confidence, 1.0)),
        )
