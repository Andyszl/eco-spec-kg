from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable

ONTOLOGY_VERSION = "ecospec-ontology-v2.0"


class EntityTypeV2(StrEnum):
    ASSESSMENT_SUBJECT = "assessment_subject"
    ASSESSMENT_INDICATOR = "assessment_indicator"
    OBSERVATION_VARIABLE = "observation_variable"
    MODEL_VARIABLE = "model_variable"
    FORMULA = "formula"
    METHOD = "method"
    INSTRUMENT = "instrument"
    DATA_SOURCE = "data_source"
    UNIT = "unit"
    ECOSYSTEM_TYPE = "ecosystem_type"
    SPATIAL_SCOPE = "spatial_scope"
    TEMPORAL_SCOPE = "temporal_scope"
    FREQUENCY = "frequency"
    QUALITY_RULE = "quality_rule"
    CLASSIFICATION_RULE = "classification_rule"


class DocumentObjectType(StrEnum):
    STANDARD = "standard"
    CLAUSE = "clause"
    TABLE = "table"
    FORMULA_BLOCK = "formula_block"
    EVIDENCE_SPAN = "evidence_span"
    SOURCE_UNIT = "source_unit"


class RelationTypeV2(StrEnum):
    HAS_INDICATOR = "has_indicator"
    CALCULATED_BY = "calculated_by"
    HAS_INPUT = "has_input"
    HAS_OUTPUT = "has_output"
    OBTAINED_BY = "obtained_by"
    MEASURED_WITH = "measured_with"
    SOURCED_FROM = "sourced_from"
    HAS_UNIT = "has_unit"
    OBSERVED_DURING = "observed_during"
    OBSERVED_EVERY = "observed_every"
    APPLIES_TO_ECOSYSTEM = "applies_to_ecosystem"
    APPLIES_TO_SPACE = "applies_to_space"
    CONSTRAINED_BY = "constrained_by"
    CLASSIFIED_BY = "classified_by"


@dataclass(frozen=True, slots=True)
class EntityDefinition:
    label_zh: str
    definition: str
    inclusion_rule: str
    exclusion_rule: str
    examples: tuple[str, ...] = ()


ENTITY_DEFINITIONS: dict[EntityTypeV2, EntityDefinition] = {
    EntityTypeV2.ASSESSMENT_SUBJECT: EntityDefinition(
        "评估对象",
        "被整体评价的生态属性或服务功能体系。",
        "标准明确以其为评估目标或总体结果。",
        "不得用于单项观测字段、公式符号或具体计算指标。",
        ("生态系统质量", "生态系统服务功能"),
    ),
    EntityTypeV2.ASSESSMENT_INDICATOR: EntityDefinition(
        "评估指标",
        "用于表征或判定评估对象状态的评价量。",
        "出现在指标体系、评估结果或明确的指标定义中。",
        "直接观测项目和仅在公式中出现的输入/中间符号不属于该类。",
        ("水源涵养量", "土壤保持量", "EQI"),
    ),
    EntityTypeV2.OBSERVATION_VARIABLE: EntityDefinition(
        "观测变量",
        "通过现场观测、仪器测量或调查表直接记录的项目。",
        "有明确观测方法、仪器、时间、频率或表格字段语义。",
        "只在计算公式中出现而无直接观测语义的符号不属于该类。",
        ("树高", "坡度", "地表径流量"),
    ),
    EntityTypeV2.MODEL_VARIABLE: EntityDefinition(
        "模型变量",
        "规范公式中的输入、输出或中间变量。",
        "必须能定位到公式表达式中的真实符号。",
        "不得用泛化概念或未出现于公式的名称替代真实公式符号。",
        ("EQI_i,j", "P_i", "R_i", "ET_i"),
    ),
    EntityTypeV2.FORMULA: EntityDefinition(
        "规范公式",
        "具有规范内编号及可定位表达式的计算公式。",
        "公式编号和表达式必须共同存在并可回到原页核验。",
        "一般方法名称、无编号的概念定义或模型简称不单独作为公式。",
        ("公式（3）", "公式（A.1）"),
    ),
    EntityTypeV2.METHOD: EntityDefinition(
        "获取或处理方法",
        "用于观测、调查、计算或数据处理的明确方法。",
        "文本存在采用、运用、测定、计算等方法触发语义。",
        "仪器设备、数据来源和公式节点不属于该类。",
        ("样方调查方法", "插值方法"),
    ),
    EntityTypeV2.INSTRUMENT: EntityDefinition(
        "仪器设备",
        "用于直接观测或测量的仪器、传感器或设备。",
        "与观测变量在同一条款或同一表格逻辑行明确关联。",
        "软件模型、数据集和一般方法不属于该类。",
        ("罗盘", "测高仪"),
    ),
    EntityTypeV2.DATA_SOURCE: EntityDefinition(
        "数据来源",
        "模型或观测变量明确取自的数据集、数据库、资料或监测来源。",
        "存在来源、获取、根据某数据等显式语义。",
        "仅被标准引用但未提供变量数据的文献或标准不属于该类。",
        ("逐日降雨量资料", "遥感数据"),
    ),
    EntityTypeV2.UNIT: EntityDefinition(
        "计量单位",
        "与指标或变量在同一变量说明或表格逻辑行给出的计量单位。",
        "单位原文边界可定位，并保留量纲一等特殊单位表述。",
        "范围、阈值和数值本身不属于该类。",
        ("m3/a", "mm/a", "量纲一"),
    ),
    EntityTypeV2.ECOSYSTEM_TYPE: EntityDefinition(
        "生态系统类型",
        "规范明确列出的生态系统分类或适用生态类型。",
        "作为指标、公式或方法的明确适用对象出现。",
        "行政区、空间分区和一般评估对象不属于该类。",
        ("森林生态系统", "草地生态系统"),
    ),
    EntityTypeV2.SPATIAL_SCOPE: EntityDefinition(
        "空间范围",
        "指标、公式或方法适用的地理区域、分区或空间尺度。",
        "文本明确表达空间范围或分区条件。",
        "生态系统类型和数据来源不属于该类。",
        ("评估区", "生态功能区"),
    ),
    EntityTypeV2.TEMPORAL_SCOPE: EntityDefinition(
        "观测时段",
        "观测变量的具体时间窗口或季节时段。",
        "与观测变量在同一条款或表格逻辑行出现。",
        "重复周期和频率不属于该类。",
        ("7—9月", "生长季"),
    ),
    EntityTypeV2.FREQUENCY: EntityDefinition(
        "观测频率",
        "观测行为的重复周期或次数。",
        "与观测变量在同一条款或表格逻辑行出现。",
        "单次日期、年份范围和持续时间不属于该类。",
        ("一年一次", "每月1次"),
    ),
    EntityTypeV2.QUALITY_RULE: EntityDefinition(
        "质量规则",
        "对变量、数据来源或方法施加的精度、完整性或处理约束。",
        "包含约束对象及阈值、条件或处理动作。",
        "生态系统质量这一评估对象不因名称含质量而归入该类。",
        ("缺失值处理规则", "精度不低于90%"),
    ),
    EntityTypeV2.CLASSIFICATION_RULE: EntityDefinition(
        "分级规则",
        "将评估指标映射为等级或类别的阈值规则。",
        "规则必须包含指标与至少一个等级边界或分类条件。",
        "一般质量控制阈值和生态类型分类表不属于该类。",
        ("EQI≥75为优",),
    ),
}


@dataclass(frozen=True, slots=True)
class RelationSpec:
    relation_type: RelationTypeV2
    label_zh: str
    definition: str
    allowed_pairs: frozenset[tuple[EntityTypeV2, EntityTypeV2]]
    evidence_rule: str


def _pairs(
    heads: Iterable[EntityTypeV2], tails: Iterable[EntityTypeV2]
) -> frozenset[tuple[EntityTypeV2, EntityTypeV2]]:
    return frozenset((head, tail) for head in heads for tail in tails)


RELATION_SPECS: dict[RelationTypeV2, RelationSpec] = {
    RelationTypeV2.HAS_INDICATOR: RelationSpec(
        RelationTypeV2.HAS_INDICATOR,
        "包含评估指标",
        "评估对象的指标体系包含某评估指标。",
        _pairs(
            [EntityTypeV2.ASSESSMENT_SUBJECT],
            [EntityTypeV2.ASSESSMENT_INDICATOR],
        ),
        "指标体系表或正文明确列出组成关系。",
    ),
    RelationTypeV2.CALCULATED_BY: RelationSpec(
        RelationTypeV2.CALCULATED_BY,
        "由公式计算",
        "评估指标或模型变量由规范内真实公式计算。",
        _pairs(
            [
                EntityTypeV2.ASSESSMENT_INDICATOR,
                EntityTypeV2.MODEL_VARIABLE,
            ],
            [EntityTypeV2.FORMULA],
        ),
        "指标/变量与公式编号在公式包中明确关联。",
    ),
    RelationTypeV2.HAS_INPUT: RelationSpec(
        RelationTypeV2.HAS_INPUT,
        "包含输入变量",
        "公式表达式使用某模型变量作为输入。",
        _pairs([EntityTypeV2.FORMULA], [EntityTypeV2.MODEL_VARIABLE]),
        "变量符号真实出现在公式右侧并有坐标证据。",
    ),
    RelationTypeV2.HAS_OUTPUT: RelationSpec(
        RelationTypeV2.HAS_OUTPUT,
        "产生输出变量",
        "公式表达式产生某模型变量。",
        _pairs([EntityTypeV2.FORMULA], [EntityTypeV2.MODEL_VARIABLE]),
        "变量符号位于公式左侧并有坐标证据。",
    ),
    RelationTypeV2.OBTAINED_BY: RelationSpec(
        RelationTypeV2.OBTAINED_BY,
        "通过方法获得",
        "变量通过明确的方法观测、调查或计算得到。",
        _pairs(
            [
                EntityTypeV2.MODEL_VARIABLE,
                EntityTypeV2.OBSERVATION_VARIABLE,
            ],
            [EntityTypeV2.METHOD],
        ),
        "同一来源单元存在获取/计算方法触发语义。",
    ),
    RelationTypeV2.MEASURED_WITH: RelationSpec(
        RelationTypeV2.MEASURED_WITH,
        "使用仪器测量",
        "观测变量由明确仪器或设备测量。",
        _pairs(
            [EntityTypeV2.OBSERVATION_VARIABLE],
            [EntityTypeV2.INSTRUMENT],
        ),
        "变量与仪器位于同一条款或表格逻辑行。",
    ),
    RelationTypeV2.SOURCED_FROM: RelationSpec(
        RelationTypeV2.SOURCED_FROM,
        "来源于数据",
        "变量取自明确的数据来源。",
        _pairs(
            [
                EntityTypeV2.MODEL_VARIABLE,
                EntityTypeV2.OBSERVATION_VARIABLE,
            ],
            [EntityTypeV2.DATA_SOURCE],
        ),
        "来源触发词与变量在同一来源单元明确关联。",
    ),
    RelationTypeV2.HAS_UNIT: RelationSpec(
        RelationTypeV2.HAS_UNIT,
        "计量单位",
        "指标或变量使用某计量单位。",
        _pairs(
            [
                EntityTypeV2.MODEL_VARIABLE,
                EntityTypeV2.OBSERVATION_VARIABLE,
                EntityTypeV2.ASSESSMENT_INDICATOR,
            ],
            [EntityTypeV2.UNIT],
        ),
        "单位与对象位于同一变量说明或表格逻辑行。",
    ),
    RelationTypeV2.OBSERVED_DURING: RelationSpec(
        RelationTypeV2.OBSERVED_DURING,
        "观测时段",
        "观测变量在某时段观测。",
        _pairs(
            [EntityTypeV2.OBSERVATION_VARIABLE],
            [EntityTypeV2.TEMPORAL_SCOPE],
        ),
        "变量与时段位于同一条款或表格逻辑行。",
    ),
    RelationTypeV2.OBSERVED_EVERY: RelationSpec(
        RelationTypeV2.OBSERVED_EVERY,
        "观测频率",
        "观测变量按某频率重复观测。",
        _pairs(
            [EntityTypeV2.OBSERVATION_VARIABLE],
            [EntityTypeV2.FREQUENCY],
        ),
        "变量与频率位于同一条款或表格逻辑行。",
    ),
    RelationTypeV2.APPLIES_TO_ECOSYSTEM: RelationSpec(
        RelationTypeV2.APPLIES_TO_ECOSYSTEM,
        "适用生态类型",
        "指标、公式或方法适用于某生态系统类型。",
        _pairs(
            [
                EntityTypeV2.ASSESSMENT_INDICATOR,
                EntityTypeV2.FORMULA,
                EntityTypeV2.METHOD,
            ],
            [EntityTypeV2.ECOSYSTEM_TYPE],
        ),
        "适用对象必须由正文、表头或注释明确表达。",
    ),
    RelationTypeV2.APPLIES_TO_SPACE: RelationSpec(
        RelationTypeV2.APPLIES_TO_SPACE,
        "适用空间范围",
        "指标、公式或方法适用于某空间范围。",
        _pairs(
            [
                EntityTypeV2.ASSESSMENT_INDICATOR,
                EntityTypeV2.FORMULA,
                EntityTypeV2.METHOD,
            ],
            [EntityTypeV2.SPATIAL_SCOPE],
        ),
        "空间条件必须由正文、表头或注释明确表达。",
    ),
    RelationTypeV2.CONSTRAINED_BY: RelationSpec(
        RelationTypeV2.CONSTRAINED_BY,
        "受质量规则约束",
        "变量、数据来源或方法受某质量规则约束。",
        _pairs(
            [
                EntityTypeV2.MODEL_VARIABLE,
                EntityTypeV2.OBSERVATION_VARIABLE,
                EntityTypeV2.DATA_SOURCE,
                EntityTypeV2.METHOD,
            ],
            [EntityTypeV2.QUALITY_RULE],
        ),
        "约束对象与阈值、条件或动作必须在质量条款中共现。",
    ),
    RelationTypeV2.CLASSIFIED_BY: RelationSpec(
        RelationTypeV2.CLASSIFIED_BY,
        "依据规则分级",
        "评估指标依据某阈值规则划分等级。",
        _pairs(
            [EntityTypeV2.ASSESSMENT_INDICATOR],
            [EntityTypeV2.CLASSIFICATION_RULE],
        ),
        "指标与等级边界必须来自同一分级表或条款。",
    ),
}


@dataclass(frozen=True, slots=True)
class PathEdge:
    head_type: EntityTypeV2
    relation_type: RelationTypeV2
    tail_type: EntityTypeV2


@dataclass(frozen=True, slots=True)
class PathTemplate:
    template_id: str
    label_zh: str
    required_edges: tuple[PathEdge, ...]
    optional_edges: tuple[PathEdge, ...] = ()


PATH_TEMPLATES: tuple[PathTemplate, ...] = (
    PathTemplate(
        "assessment_formula_path",
        "评估对象—指标—公式—变量",
        (
            PathEdge(
                EntityTypeV2.ASSESSMENT_SUBJECT,
                RelationTypeV2.HAS_INDICATOR,
                EntityTypeV2.ASSESSMENT_INDICATOR,
            ),
            PathEdge(
                EntityTypeV2.ASSESSMENT_INDICATOR,
                RelationTypeV2.CALCULATED_BY,
                EntityTypeV2.FORMULA,
            ),
            PathEdge(
                EntityTypeV2.FORMULA,
                RelationTypeV2.HAS_OUTPUT,
                EntityTypeV2.MODEL_VARIABLE,
            ),
            PathEdge(
                EntityTypeV2.FORMULA,
                RelationTypeV2.HAS_INPUT,
                EntityTypeV2.MODEL_VARIABLE,
            ),
        ),
        (
            PathEdge(
                EntityTypeV2.MODEL_VARIABLE,
                RelationTypeV2.HAS_UNIT,
                EntityTypeV2.UNIT,
            ),
            PathEdge(
                EntityTypeV2.MODEL_VARIABLE,
                RelationTypeV2.SOURCED_FROM,
                EntityTypeV2.DATA_SOURCE,
            ),
        ),
    ),
    PathTemplate(
        "observation_path",
        "观测变量—方法/仪器—时间频率",
        (
            PathEdge(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.OBTAINED_BY,
                EntityTypeV2.METHOD,
            ),
        ),
        (
            PathEdge(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.MEASURED_WITH,
                EntityTypeV2.INSTRUMENT,
            ),
            PathEdge(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.OBSERVED_DURING,
                EntityTypeV2.TEMPORAL_SCOPE,
            ),
            PathEdge(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.OBSERVED_EVERY,
                EntityTypeV2.FREQUENCY,
            ),
        ),
    ),
    PathTemplate(
        "quality_path",
        "变量/来源/方法—质量规则",
        (
            PathEdge(
                EntityTypeV2.OBSERVATION_VARIABLE,
                RelationTypeV2.CONSTRAINED_BY,
                EntityTypeV2.QUALITY_RULE,
            ),
        ),
    ),
)


@dataclass(slots=True)
class ProvenanceV2:
    standard_code: str
    source_sha256: str
    source_unit_id: str
    evidence_span_ids: list[str]
    pages: list[int]
    sections: list[str]
    bboxes: dict[int, list[list[float]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allowed_relation_pairs(
    relation_type: RelationTypeV2,
) -> frozenset[tuple[EntityTypeV2, EntityTypeV2]]:
    return RELATION_SPECS[relation_type].allowed_pairs


def validate_relation_v2(
    head_type: str, relation_type: str, tail_type: str
) -> tuple[bool, str]:
    try:
        head = EntityTypeV2(head_type)
        relation = RelationTypeV2(relation_type)
        tail = EntityTypeV2(tail_type)
    except ValueError as exc:
        return False, f"unknown ontology v2 value: {exc}"
    if (head, tail) not in allowed_relation_pairs(relation):
        return (
            False,
            f"{head.value} -[{relation.value}]-> {tail.value} is not allowed",
        )
    return True, ""


def validate_provenance_v2(
    provenance: ProvenanceV2,
) -> tuple[bool, str]:
    if not provenance.standard_code or not provenance.source_sha256:
        return False, "standard_code and source_sha256 are required"
    if not provenance.source_unit_id or not provenance.evidence_span_ids:
        return False, "source_unit_id and evidence_span_ids are required"
    if not provenance.pages or any(page <= 0 for page in provenance.pages):
        return False, "positive PDF page numbers are required"
    for page in provenance.pages:
        boxes = provenance.bboxes.get(page, [])
        if not boxes:
            return False, f"missing evidence bbox for PDF page {page}"
        if any(len(box) != 4 for box in boxes):
            return False, f"invalid bbox for PDF page {page}"
    return True, ""


def schema_rows_v2() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relation, spec in RELATION_SPECS.items():
        for head, tail in sorted(
            spec.allowed_pairs,
            key=lambda pair: (pair[0].value, pair[1].value),
        ):
            rows.append(
                {
                    "ontology_version": ONTOLOGY_VERSION,
                    "head_type": head.value,
                    "relation_type": relation.value,
                    "tail_type": tail.value,
                    "relation_label_zh": spec.label_zh,
                    "evidence_rule": spec.evidence_rule,
                }
            )
    return rows


LEGACY_AMBIGUITIES: dict[str, tuple[str, ...]] = {
    "indicator": (
        EntityTypeV2.ASSESSMENT_INDICATOR.value,
        EntityTypeV2.OBSERVATION_VARIABLE.value,
    ),
    "parameter": (
        EntityTypeV2.MODEL_VARIABLE.value,
        EntityTypeV2.OBSERVATION_VARIABLE.value,
    ),
    "uses_formula": (RelationTypeV2.CALCULATED_BY.value,),
    "depends_on": (
        RelationTypeV2.HAS_INPUT.value,
        RelationTypeV2.HAS_OUTPUT.value,
    ),
    "applies_to": (
        RelationTypeV2.APPLIES_TO_ECOSYSTEM.value,
        RelationTypeV2.APPLIES_TO_SPACE.value,
    ),
    "defined_in": (),
}


def legacy_migration_status(value: str) -> dict[str, Any]:
    candidates = LEGACY_AMBIGUITIES.get(value)
    if candidates is None:
        return {"status": "not_legacy", "value": value, "candidates": []}
    return {
        "status": "manual_review_required",
        "value": value,
        "candidates": list(candidates),
        "reason": (
            "legacy value is semantically ambiguous"
            if candidates
            else "document provenance is no longer a domain relation"
        ),
    }


def schema_quality_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    missing_definitions = sorted(
        entity.value
        for entity in EntityTypeV2
        if entity not in ENTITY_DEFINITIONS
    )
    checks.append(
        {
            "check": "all_domain_entities_defined",
            "passed": not missing_definitions,
            "details": missing_definitions,
        }
    )

    missing_relations = sorted(
        relation.value
        for relation in RelationTypeV2
        if relation not in RELATION_SPECS
    )
    checks.append(
        {
            "check": "all_relations_constrained",
            "passed": not missing_relations,
            "details": missing_relations,
        }
    )

    empty_specs = sorted(
        relation.value
        for relation, spec in RELATION_SPECS.items()
        if not spec.allowed_pairs or not spec.evidence_rule
    )
    checks.append(
        {
            "check": "no_unbounded_relation",
            "passed": not empty_specs,
            "details": empty_specs,
        }
    )

    legacy_values = {
        "indicator",
        "parameter",
        "defined_in",
        "applies_to",
    }
    active_values = {
        item.value for item in EntityTypeV2
    } | {item.value for item in RelationTypeV2}
    active_legacy = sorted(legacy_values & active_values)
    checks.append(
        {
            "check": "legacy_ambiguous_values_excluded",
            "passed": not active_legacy,
            "details": active_legacy,
        }
    )

    invalid_template_edges: list[str] = []
    for template in PATH_TEMPLATES:
        for edge in (*template.required_edges, *template.optional_edges):
            valid, reason = validate_relation_v2(
                edge.head_type.value,
                edge.relation_type.value,
                edge.tail_type.value,
            )
            if not valid:
                invalid_template_edges.append(
                    f"{template.template_id}: {reason}"
                )
    checks.append(
        {
            "check": "path_templates_use_allowed_edges",
            "passed": not invalid_template_edges,
            "details": invalid_template_edges,
        }
    )

    return {
        "ontology_version": ONTOLOGY_VERSION,
        "passed": all(check["passed"] for check in checks),
        "domain_entity_count": len(EntityTypeV2),
        "document_object_count": len(DocumentObjectType),
        "relation_count": len(RelationTypeV2),
        "allowed_triple_pattern_count": len(schema_rows_v2()),
        "path_template_count": len(PATH_TEMPLATES),
        "checks": checks,
    }
