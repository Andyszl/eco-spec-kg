from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    INDICATOR = "indicator"
    FORMULA = "formula"
    PARAMETER = "parameter"
    UNIT = "unit"
    DATA_SOURCE = "data_source"
    METHOD = "method"
    ECOSYSTEM_TYPE = "ecosystem_type"
    SPATIAL_SCOPE = "spatial_scope"
    TEMPORAL_SCOPE = "temporal_scope"
    QUALITY_REQUIREMENT = "quality_requirement"
    STANDARD_CLAUSE = "standard_clause"


class RelationType(StrEnum):
    USES_FORMULA = "uses_formula"
    DEPENDS_ON = "depends_on"
    SOURCED_FROM = "sourced_from"
    HAS_UNIT = "has_unit"
    APPLIES_TO = "applies_to"
    DEFINED_IN = "defined_in"
    USES_METHOD = "uses_method"
    CONSTRAINED_BY = "constrained_by"
    DERIVED_FROM = "derived_from"


ENTITY_LABELS_ZH = {
    EntityType.INDICATOR: "评估指标",
    EntityType.FORMULA: "计算公式",
    EntityType.PARAMETER: "模型参数",
    EntityType.UNIT: "单位",
    EntityType.DATA_SOURCE: "数据来源",
    EntityType.METHOD: "处理方法",
    EntityType.ECOSYSTEM_TYPE: "生态系统类型",
    EntityType.SPATIAL_SCOPE: "空间范围",
    EntityType.TEMPORAL_SCOPE: "时间范围",
    EntityType.QUALITY_REQUIREMENT: "质量控制要求",
    EntityType.STANDARD_CLAUSE: "标准条款",
}

RELATION_LABELS_ZH = {
    RelationType.USES_FORMULA: "采用公式",
    RelationType.DEPENDS_ON: "依赖参数",
    RelationType.SOURCED_FROM: "数据来源于",
    RelationType.HAS_UNIT: "单位为",
    RelationType.APPLIES_TO: "适用于",
    RelationType.DEFINED_IN: "定义于条款",
    RelationType.USES_METHOD: "采用处理方法",
    RelationType.CONSTRAINED_BY: "受质量要求约束",
    RelationType.DERIVED_FROM: "由参数派生",
}

_ALLOWED: dict[RelationType, set[tuple[EntityType, EntityType]]] = {
    RelationType.USES_FORMULA: {(EntityType.INDICATOR, EntityType.FORMULA)},
    RelationType.DEPENDS_ON: {
        (EntityType.FORMULA, EntityType.PARAMETER),
        (EntityType.INDICATOR, EntityType.PARAMETER),
        (EntityType.INDICATOR, EntityType.INDICATOR),
    },
    RelationType.SOURCED_FROM: {(EntityType.PARAMETER, EntityType.DATA_SOURCE)},
    RelationType.HAS_UNIT: {
        (EntityType.PARAMETER, EntityType.UNIT),
        (EntityType.INDICATOR, EntityType.UNIT),
    },
    RelationType.APPLIES_TO: {
        (EntityType.INDICATOR, EntityType.ECOSYSTEM_TYPE),
        (EntityType.INDICATOR, EntityType.SPATIAL_SCOPE),
        (EntityType.INDICATOR, EntityType.TEMPORAL_SCOPE),
        (EntityType.FORMULA, EntityType.ECOSYSTEM_TYPE),
        (EntityType.METHOD, EntityType.ECOSYSTEM_TYPE),
    },
    RelationType.DEFINED_IN: {
        (source, EntityType.STANDARD_CLAUSE)
        for source in EntityType
        if source != EntityType.STANDARD_CLAUSE
    },
    RelationType.USES_METHOD: {
        (EntityType.INDICATOR, EntityType.METHOD),
        (EntityType.PARAMETER, EntityType.METHOD),
    },
    RelationType.CONSTRAINED_BY: {
        (source, EntityType.QUALITY_REQUIREMENT)
        for source in EntityType
        if source not in {EntityType.QUALITY_REQUIREMENT, EntityType.STANDARD_CLAUSE}
    },
    RelationType.DERIVED_FROM: {
        (EntityType.PARAMETER, EntityType.PARAMETER),
        (EntityType.INDICATOR, EntityType.PARAMETER),
        (EntityType.INDICATOR, EntityType.INDICATOR),
    },
}


def validate_relation(head_type: str, relation_type: str, tail_type: str) -> tuple[bool, str]:
    try:
        head = EntityType(head_type)
        relation = RelationType(relation_type)
        tail = EntityType(tail_type)
    except ValueError as exc:
        return False, f"unknown schema value: {exc}"
    if (head, tail) not in _ALLOWED[relation]:
        return False, f"{head.value} -[{relation.value}]-> {tail.value} is not allowed"
    return True, ""


def schema_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relation, pairs in _ALLOWED.items():
        for head, tail in sorted(pairs, key=lambda item: (item[0].value, item[1].value)):
            rows.append(
                {
                    "head_type": head.value,
                    "relation_type": relation.value,
                    "tail_type": tail.value,
                    "relation_label_zh": RELATION_LABELS_ZH[relation],
                }
            )
    return rows

