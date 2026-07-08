"""EcoSpec-KG public package."""

from .models import AnnotationRecord, DocumentChunk, Entity, Relation, SourceRef
from .schema import EntityType, RelationType, validate_relation

__all__ = [
    "AnnotationRecord",
    "DocumentChunk",
    "Entity",
    "Relation",
    "SourceRef",
    "EntityType",
    "RelationType",
    "validate_relation",
]

__version__ = "0.1.0"

