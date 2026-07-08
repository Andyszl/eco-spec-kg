from __future__ import annotations

import re

from .io_utils import normalize_space
from .models import DocumentChunk, Relation
from .schema import validate_relation


def _compact_for_pdf_match(text: str) -> str:
    return re.sub(r"\s+", "", normalize_space(text))


def validate_evidence(relation: Relation, chunk: DocumentChunk) -> tuple[bool, str]:
    valid_schema, reason = validate_relation(
        relation.head_type, relation.relation_type, relation.tail_type
    )
    if not valid_schema:
        return False, reason
    if relation.evidence.chunk_id != chunk.chunk_id:
        return False, "evidence chunk id does not match"
    if relation.evidence.standard_code != chunk.standard_code:
        return False, "evidence standard code does not match"
    evidence = normalize_space(relation.evidence.evidence_text)
    source = normalize_space(chunk.text)
    if not evidence:
        return False, "evidence text is empty"
    if evidence not in source:
        compact_evidence = _compact_for_pdf_match(evidence)
        compact_source = _compact_for_pdf_match(source)
        if compact_evidence not in compact_source:
            return (
                False,
                "evidence is not an exact or PDF-normalized substring of the source chunk",
            )
    return True, ""
