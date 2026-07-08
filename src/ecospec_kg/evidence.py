from __future__ import annotations

from .io_utils import normalize_space
from .models import DocumentChunk, Relation
from .schema import validate_relation


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
        return False, "evidence is not an exact normalized substring of the source chunk"
    return True, ""

