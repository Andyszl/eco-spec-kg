from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import normalize_space, stable_id
from .ontology_v2 import (
    EntityTypeV2,
    ONTOLOGY_VERSION,
    RelationTypeV2,
    validate_relation_v2,
)

ANNOTATION_VERSION = "ecospec-annotation-v2.1"

ASSESSMENT_SUBJECTS = {
    "HJ 1171-2021": "生态系统格局",
    "HJ 1172-2021": "生态系统质量",
    "HJ 1173-2021": "生态系统服务功能",
    "HJ 1174-2021": "生态问题",
    "HJ 1175-2021": "项目尺度生态影响",
}

OBSERVATION_CODES = {
    "HJ 1166-2021",
    "HJ 1167-2021",
    "HJ 1168-2021",
    "HJ 1169-2021",
    "HJ 1170-2021",
}

EXTERNAL_TEST_CODES = {
    "HJ 1171-2021",
    "HJ 1174-2021",
    "HJ 1175-2021",
}

INSTRUMENT_TERMS = (
    "GPS",
    "罗盘",
    "流速仪",
    "测径尺",
    "测高仪",
    "生长锥",
    "叶面积指数仪",
    "蒸渗仪",
    "涡度相关仪",
    "时域反射仪",
    "土壤水分传感器",
    "红外相机",
    "光合有效辐射计",
    "相机",
    "记录本",
    "定位工具",
)

DATA_SOURCE_TERMS = (
    "遥感数据",
    "遥感影像",
    "遥感图像",
    "地面调查数据",
    "野外调查数据",
    "长期监测数据",
    "气象数据",
    "降雨量资料",
    "土地利用数据",
    "土地覆盖数据",
    "数字高程模型（DEM）",
    "数字高程模型",
    "DEM",
)

ECOSYSTEM_TERMS = (
    "森林生态系统",
    "灌丛生态系统",
    "草地生态系统",
    "湿地生态系统",
    "农田生态系统",
    "城镇生态系统",
    "荒漠生态系统",
    "针叶林",
    "阔叶林",
    "针阔混交林",
    "稀疏林",
    "草甸",
    "草原",
    "草丛",
    "稀疏草地",
    "沼泽",
    "湖泊",
    "河流",
    "沙漠",
    "沙地",
    "盐碱地",
)

SPATIAL_TERMS = (
    "全国",
    "省级行政区",
    "省市县级行政区域",
    "评估区",
    "生态功能区",
    "项目区",
)

METHOD_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9（）()·—\-]{2,24}"
    r"(?:调查法|观测法|测量法|测定法|计算法|估算法|分析法|"
    r"插值法|抽样法|样方法|观察法|照相法|烘干法|"
    r"模型法|机器学习法|回归法|方程|模型|算法))"
)

GENERIC_METHODS = {
    "具体方法",
    "计算方法",
    "评估方法",
    "调查方法",
    "观测方法",
    "方法",
}

FORMULA_INDICATOR_OVERRIDES = {
    ("HJ 1172-2021", "1"): "生态系统参数相对密度",
    ("HJ 1172-2021", "2"): "归一化指数",
    ("HJ 1172-2021", "3"): "生态系统质量指数（EQI）",
    ("HJ 1172-2021", "B.1"): "叶面积指数",
    ("HJ 1172-2021", "B.2"): "叶面积指数",
    ("HJ 1172-2021", "B.3"): "叶面积指数",
    ("HJ 1172-2021", "B.4"): "植被覆盖度",
}

for _number in ("A.1",):
    FORMULA_INDICATOR_OVERRIDES[("HJ 1173-2021", _number)] = "水源涵养量"
for _number in (
    "A.2",
    "A.3",
    "A.4",
    "A.5",
    "A.6",
    "A.7",
    "A.8",
    "A.9",
    "A.10",
    "A.11",
    "A.12",
    "A.13",
):
    FORMULA_INDICATOR_OVERRIDES[("HJ 1173-2021", _number)] = "土壤保持量"
for _number in (
    "A.14",
    "A.15",
    "A.16",
    "A.17",
    "A.18",
    "A.19",
    "A.20",
    "A.21",
    "A.22",
):
    FORMULA_INDICATOR_OVERRIDES[("HJ 1173-2021", _number)] = "防风固沙量"
FORMULA_INDICATOR_OVERRIDES[("HJ 1173-2021", "A.23")] = "生境不可替代性指数"
FORMULA_INDICATOR_OVERRIDES[("HJ 1175-2021", "A.1")] = "单一生态系统动态度"
FORMULA_INDICATOR_OVERRIDES[("HJ 1175-2021", "A.2")] = "生态系统转出强度"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _clean_name(value: str) -> str:
    return normalize_space(value).strip("；;：:，,。 ")


def _span_ids(unit: dict[str, Any]) -> list[str]:
    return _unique(
        span["span_id"]
        for span in unit["provenance"].get("evidence_spans", [])
        if span.get("span_id")
    )


def _all_evidence_spans(unit: dict[str, Any]) -> list[dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if all(key in value for key in ("span_id", "page", "bbox")):
                spans[value["span_id"]] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(unit)
    return list(spans.values())


def _pages(unit: dict[str, Any]) -> list[int]:
    return list(unit["provenance"].get("pages", []))


def _heading_title(unit: dict[str, Any]) -> str:
    chain = unit["provenance"].get("heading_chain", [])
    if not chain:
        return ""
    text = chain[-1]
    text = re.sub(
        r"^(?:附录\s*[A-Z]|[A-Z]?\d+(?:\.\d+)*)\s*",
        "",
        text,
    )
    return _clean_name(text)


def _entity_key(
    standard_code: str,
    entity_type: str,
    name: str,
    context: str,
) -> tuple[str, str, str, str]:
    if entity_type not in {
        EntityTypeV2.MODEL_VARIABLE.value,
        EntityTypeV2.FORMULA.value,
        EntityTypeV2.QUALITY_RULE.value,
        EntityTypeV2.CLASSIFICATION_RULE.value,
    }:
        context = ""
    return standard_code, entity_type, _clean_name(name), context


def _symbol_base(symbol: str) -> str:
    return re.split(r"[_\s]", symbol, maxsplit=1)[0].strip("′'")


def _compact_math(value: str) -> str:
    return re.sub(r"[\s_{}]", "", value).replace("（", "(").replace("）", ")")


def _symbol_identity(symbol: str) -> str:
    return (
        _compact_math(symbol)
        .replace("，", ",")
        .replace("’", "′")
        .replace("'", "′")
    )


def _symbol_occurs(symbol: str, expression: str) -> bool:
    compact_symbol = _symbol_identity(symbol)
    compact_expression = _compact_math(expression)
    if compact_symbol and compact_symbol in compact_expression:
        return True
    base = _compact_math(_symbol_base(symbol))
    return bool(base and base in compact_expression)


def _split_compound_variable(variable: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = variable["symbol"]
    if not re.fullmatch(r"[A-Za-z](?:、[A-Za-z])+(?:和[A-Za-z])?", symbol):
        return [variable]
    symbols = re.findall(r"[A-Za-z]", symbol)
    return [{**variable, "symbol": item} for item in symbols]


def _formula_lhs_symbols(expression: str) -> list[str]:
    symbols: list[str] = []
    for match in re.finditer(r"=", expression):
        prefix = expression[: match.start()]
        segment = re.split(r"[;\n]", prefix)[-1].strip()
        segment = re.sub(r"^[^A-Za-z\u0370-\u03ff\u4e00-\u9fff]+", "", segment)
        candidate_match = re.search(
            r"([A-Za-z\u0370-\u03ff\u4e00-\u9fff]+"
            r"[′’']?"
            r"(?:_[A-Za-z0-9,]+)?"
            r"(?:\s*\([^()=]{1,20}\))?)\s*$",
            segment,
        )
        if candidate_match:
            symbols.append(_compact_math(candidate_match.group(1)))
    return _unique(symbols)


def _formula_output_symbols(
    expression: str,
    variables: list[dict[str, Any]],
) -> list[str]:
    lhs_symbols = _formula_lhs_symbols(expression)
    output: list[str] = []
    for lhs_symbol in lhs_symbols:
        exact_matches = [
            variable["symbol"]
            for variable in variables
            if _symbol_identity(variable["symbol"])
            == _symbol_identity(lhs_symbol)
        ]
        base_matches = [
            variable["symbol"]
            for variable in variables
            if _symbol_identity(_symbol_base(variable["symbol"]))
            == _symbol_identity(_symbol_base(lhs_symbol))
        ]
        match = (
            exact_matches[0]
            if exact_matches
            else base_matches[0]
            if len(base_matches) == 1
            else None
        )
        output.append(match or lhs_symbol)
    return _unique(output)


def _formula_indicator(unit: dict[str, Any], number: str) -> str:
    code = unit["provenance"]["standard_code"]
    override = FORMULA_INDICATOR_OVERRIDES.get((code, number))
    if override:
        return override
    title = _heading_title(unit)
    if not title or title in {"评估技术方法", "指标与方法"}:
        return ""
    return title


def _extract_methods(text: str) -> list[str]:
    methods = []
    for match in METHOD_RE.finditer(text):
        name = _clean_name(match.group(1))
        name = re.sub(r"^(?:主要|具体|利用|采用|通过|运用|根据)", "", name)
        if name and name not in GENERIC_METHODS and len(name) <= 28:
            methods.append(name)
    if "采用仪器测量" in text or "仪器直接观测" in text:
        methods.append("仪器测量")
    if "现场调查" in text:
        methods.append("现场调查")
    return _unique(methods)


def _extract_instruments(text: str) -> list[str]:
    return _unique(term for term in INSTRUMENT_TERMS if term in text)


def _extract_sources(text: str) -> list[str]:
    sources = []
    for term in DATA_SOURCE_TERMS:
        if term in text and not any(term in value for value in sources):
            sources.append(term)
    return sources


def _extract_ecosystems(text: str) -> list[str]:
    return _unique(term for term in ECOSYSTEM_TERMS if term in text)


def _extract_spaces(text: str) -> list[str]:
    return _unique(term for term in SPATIAL_TERMS if term in text)


def _formula_override_units(
    override_path: Path,
    source_sha_by_code: dict[str, str],
) -> list[dict[str, Any]]:
    if not override_path.exists():
        return []
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    output = []
    for item in payload.get("overrides", []):
        span_id = stable_id(
            item["standard_code"],
            item["formula_number"],
            item["page"],
            "manual-formula-span",
        )
        unit_id = stable_id(
            item["standard_code"],
            item["section"],
            item["formula_number"],
            item["page"],
            "manual-override",
        )
        evidence_span = {
            "span_id": span_id,
            "page": item["page"],
            "printed_page": item.get("printed_page", ""),
            "bbox": item["bbox"],
            "line_ids": item.get("line_ids", []),
            "text": item["expression_text"],
        }
        variables = []
        roles = {}
        for variable in item["variables"]:
            roles[variable["symbol"]] = variable["role"]
            variables.append(
                {
                    "symbol": variable["symbol"],
                    "definition": variable["definition"],
                    "unit": variable.get("unit", ""),
                    "evidence_span": evidence_span,
                }
            )
        output.append(
            {
                "schema_version": "source-unit-v2.1",
                "unit_id": unit_id,
                "unit_type": "formula_package",
                "introduction": "人工核对原PDF后补录不可直接提取的公式主体。",
                "interstitial_text": "",
                "adjacent_source_text": "",
                "formulas": [
                    {
                        "formula_number": item["formula_number"],
                        "expression_text": item["expression_text"],
                        "expression_lines": [item["expression_text"]],
                        "evidence_span": evidence_span,
                    }
                ],
                "variable_definitions": variables,
                "manual_variable_roles": roles,
                "manual_transcription": True,
                "replace_existing": bool(item.get("replace_existing")),
                "transcription_method": item["transcription_method"],
                "review_status": item["review_status"],
                "provenance": {
                    "standard_code": item["standard_code"],
                    "document_title": item["document_title"],
                    "source_file": "",
                    "source_sha256": source_sha_by_code[item["standard_code"]],
                    "pages": [item["page"]],
                    "printed_pages": [item.get("printed_page", "")],
                    "section": item["section"],
                    "heading_chain": item["heading_chain"],
                    "evidence_spans": [evidence_span],
                },
            }
        )
    return output


def load_enriched_source_units(
    source_units_path: Path,
    manifest_path: Path,
    override_path: Path,
) -> list[dict[str, Any]]:
    units = [
        json.loads(line)
        for line in source_units_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha_by_code = {
        item["standard_code"]: item["source_sha256"] for item in manifest
    }
    existing_numbers = {
        (
            unit["provenance"]["standard_code"],
            unit["provenance"].get("section", ""),
            formula["formula_number"],
        )
        for unit in units
        if unit["unit_type"] == "formula_package"
        for formula in unit["formulas"]
    }
    for override in _formula_override_units(override_path, sha_by_code):
        key = (
            override["provenance"]["standard_code"],
            override["provenance"].get("section", ""),
            override["formulas"][0]["formula_number"],
        )
        if override.get("replace_existing"):
            retained_units = []
            for unit in units:
                if (
                    unit["unit_type"] != "formula_package"
                    or unit["provenance"]["standard_code"] != key[0]
                    or unit["provenance"].get("section", "") != key[1]
                ):
                    retained_units.append(unit)
                    continue
                unit["formulas"] = [
                    formula
                    for formula in unit["formulas"]
                    if formula["formula_number"] != key[2]
                ]
                if unit["formulas"]:
                    retained_units.append(unit)
            units = retained_units
            existing_numbers.discard(key)
        if key not in existing_numbers:
            units.append(override)
            existing_numbers.add(key)
    units.sort(
        key=lambda unit: (
            unit["provenance"]["standard_code"],
            unit["provenance"]["pages"],
            unit["unit_type"],
            unit["unit_id"],
        )
    )
    return units


@dataclass(slots=True)
class AnnotationResult:
    source_units: list[dict[str, Any]]
    unit_annotations: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    rejected_relations: list[dict[str, Any]]


class AnnotationBuilder:
    def __init__(
        self,
        units: list[dict[str, Any]],
        annotation_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.units = units
        self.annotation_overrides = annotation_overrides or {}
        self.annotations = {
            unit["unit_id"]: {
                "annotation_version": ANNOTATION_VERSION,
                "ontology_version": ONTOLOGY_VERSION,
                "unit_id": unit["unit_id"],
                "standard_code": unit["provenance"]["standard_code"],
                "entities": [],
                "relations": [],
                "review_status": (
                    "ai_dual_expert_adjudicated"
                    if unit["unit_id"] in self.annotation_overrides
                    else "ai_expert_verified"
                    if unit.get("manual_transcription")
                    else "ai_rule_pre_gold"
                ),
                "no_relation_reason": "",
                "notes": "",
            }
            for unit in units
        }
        self.entity_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.triple_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.rejected: list[dict[str, Any]] = []

    def annotate_expert_override(
        self,
        unit: dict[str, Any],
        override: dict[str, Any],
    ) -> None:
        section = unit["provenance"].get("section", "")
        formula_contexts: dict[str, str] = {}
        formula_definitions: dict[str, str] = {}
        for formula in unit.get("formulas", []):
            name = f"公式（{formula['formula_number']}）"
            context = f"{section}|{formula['formula_number']}"
            formula_contexts[_clean_name(name)] = context
            formula_definitions[context] = formula.get(
                "expression_text", ""
            )

        def context_for_formula_name(name: str) -> str:
            cleaned = _clean_name(name)
            if cleaned in formula_contexts:
                return formula_contexts[cleaned]
            match = re.search(r"公式[（(]([^）)]+)[）)]", cleaned)
            return f"{section}|{match.group(1)}" if match else ""

        linked_variable_contexts: dict[
            tuple[str, str], set[str]
        ] = defaultdict(set)
        for relation in override.get("relations", []):
            formula_name = ""
            variable_name = ""
            if relation["head_type"] == EntityTypeV2.FORMULA.value:
                formula_name = relation["head_name"]
                if (
                    relation["tail_type"]
                    == EntityTypeV2.MODEL_VARIABLE.value
                ):
                    variable_name = relation["tail_name"]
            elif relation["tail_type"] == EntityTypeV2.FORMULA.value:
                formula_name = relation["tail_name"]
                if (
                    relation["head_type"]
                    == EntityTypeV2.MODEL_VARIABLE.value
                ):
                    variable_name = relation["head_name"]
            context = context_for_formula_name(formula_name)
            if context and variable_name:
                linked_variable_contexts[
                    (
                        _clean_name(variable_name),
                        EntityTypeV2.MODEL_VARIABLE.value,
                    )
                ].add(context)

        local_entities: dict[
            tuple[str, str], list[tuple[str, dict[str, Any]]]
        ] = defaultdict(list)
        variable_definitions = [
            expanded
            for variable in unit.get("variable_definitions", [])
            for expanded in _split_compound_variable(variable)
        ]
        for item in override.get("entities", []):
            name = _clean_name(item["name"])
            entity_type = item["entity_type"]
            key = (name, entity_type)
            contexts: list[str]
            if entity_type == EntityTypeV2.FORMULA.value:
                contexts = [context_for_formula_name(name)]
            elif entity_type == EntityTypeV2.MODEL_VARIABLE.value:
                contexts = sorted(linked_variable_contexts.get(key, set()))
                if not contexts:
                    contexts = [
                        context
                        for formula_name, context in formula_contexts.items()
                        if _symbol_occurs(
                            name, formula_definitions.get(context, "")
                        )
                    ]
                if not contexts:
                    contexts = list(formula_contexts.values())[:1]
                if not contexts:
                    contexts = [f"{section}|{unit['unit_id']}"]
            elif entity_type in {
                EntityTypeV2.QUALITY_RULE.value,
                EntityTypeV2.CLASSIFICATION_RULE.value,
            }:
                contexts = [unit["unit_id"]]
            else:
                contexts = [""]

            definition = ""
            if entity_type == EntityTypeV2.FORMULA.value:
                definition = formula_definitions.get(contexts[0], "")
            elif entity_type == EntityTypeV2.MODEL_VARIABLE.value:
                definition = next(
                    (
                        variable.get("definition", "")
                        for variable in variable_definitions
                        if _symbol_identity(variable["symbol"])
                        == _symbol_identity(name)
                    ),
                    "",
                )
            for context in dict.fromkeys(contexts):
                entity = self.add_entity(
                    unit,
                    name,
                    entity_type,
                    item.get("evidence_span_ids", []),
                    definition=definition,
                    context=context,
                )
                if entity is not None:
                    local_entities[key].append((context, entity))

        def candidates(
            name: str,
            entity_type: str,
            preferred_context: str,
        ) -> list[dict[str, Any]]:
            values = local_entities.get(
                (_clean_name(name), entity_type), []
            )
            if preferred_context:
                preferred = [
                    entity
                    for context, entity in values
                    if context == preferred_context
                ]
                if preferred:
                    return preferred
            return [entity for _context, entity in values]

        for relation in override.get("relations", []):
            preferred_context = ""
            if relation["head_type"] == EntityTypeV2.FORMULA.value:
                preferred_context = context_for_formula_name(
                    relation["head_name"]
                )
            elif relation["tail_type"] == EntityTypeV2.FORMULA.value:
                preferred_context = context_for_formula_name(
                    relation["tail_name"]
                )
            heads = candidates(
                relation["head_name"],
                relation["head_type"],
                preferred_context,
            )
            tails = candidates(
                relation["tail_name"],
                relation["tail_type"],
                preferred_context,
            )
            for head in heads:
                for tail in tails:
                    self.add_relation(
                        unit,
                        head,
                        relation["relation_type"],
                        tail,
                        relation.get("evidence_span_ids", []),
                    )

        annotation = self.annotations[unit["unit_id"]]
        annotation["annotator_id"] = override.get(
            "annotator_id", "adjudicator_C"
        )
        annotation["notes"] = override.get("notes", "")
        annotation["no_relation_reason"] = override.get(
            "no_relation_reason", ""
        )

    def add_entity(
        self,
        unit: dict[str, Any],
        name: str,
        entity_type: EntityTypeV2 | str,
        evidence_span_ids: list[str] | None = None,
        *,
        definition: str = "",
        context: str = "",
    ) -> dict[str, Any] | None:
        name = _clean_name(name)
        if not name or name in {"—", "-", "无", "其他"}:
            return None
        entity_type_value = (
            entity_type.value
            if isinstance(entity_type, EntityTypeV2)
            else entity_type
        )
        code = unit["provenance"]["standard_code"]
        key = _entity_key(code, entity_type_value, name, context)
        entity = self.entity_index.get(key)
        spans = evidence_span_ids or _span_ids(unit)
        if not spans:
            return None
        if entity is None:
            entity_id = stable_id(
                "entity-v2",
                code,
                entity_type_value,
                name,
                key[3],
            )
            entity = {
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type_value,
                "standard_code": code,
                "context": key[3],
                "definitions": [],
                "source_unit_ids": [],
                "evidence_span_ids": [],
                "review_statuses": [],
                "annotation_version": ANNOTATION_VERSION,
                "ontology_version": ONTOLOGY_VERSION,
                "gold_nature": "ai_expert_pre_gold",
            }
            self.entity_index[key] = entity
        if definition and definition not in entity["definitions"]:
            entity["definitions"].append(definition)
        if unit["unit_id"] not in entity["source_unit_ids"]:
            entity["source_unit_ids"].append(unit["unit_id"])
        review_status = self.annotations[unit["unit_id"]]["review_status"]
        if review_status not in entity["review_statuses"]:
            entity["review_statuses"].append(review_status)
        entity["evidence_span_ids"] = _unique(
            entity["evidence_span_ids"] + spans
        )
        local = {
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "entity_type": entity["entity_type"],
            "evidence_span_ids": spans,
        }
        if local not in self.annotations[unit["unit_id"]]["entities"]:
            self.annotations[unit["unit_id"]]["entities"].append(local)
        return entity

    def add_relation(
        self,
        unit: dict[str, Any],
        head: dict[str, Any] | None,
        relation_type: RelationTypeV2 | str,
        tail: dict[str, Any] | None,
        evidence_span_ids: list[str] | None = None,
    ) -> None:
        if head is None or tail is None:
            return
        relation_value = (
            relation_type.value
            if isinstance(relation_type, RelationTypeV2)
            else relation_type
        )
        valid, reason = validate_relation_v2(
            head["entity_type"],
            relation_value,
            tail["entity_type"],
        )
        if not valid:
            self.rejected.append(
                {
                    "unit_id": unit["unit_id"],
                    "head_id": head["entity_id"],
                    "relation_type": relation_value,
                    "tail_id": tail["entity_id"],
                    "reason": reason,
                }
            )
            return
        spans = evidence_span_ids or _span_ids(unit)
        if not spans:
            return
        local = {
            "head_id": head["entity_id"],
            "head_name": head["name"],
            "head_type": head["entity_type"],
            "relation_type": relation_value,
            "tail_id": tail["entity_id"],
            "tail_name": tail["name"],
            "tail_type": tail["entity_type"],
            "evidence_span_ids": spans,
        }
        if local not in self.annotations[unit["unit_id"]]["relations"]:
            self.annotations[unit["unit_id"]]["relations"].append(local)
        key = (head["entity_id"], relation_value, tail["entity_id"])
        triple = self.triple_index.get(key)
        if triple is None:
            triple = {
                "triple_id": stable_id("triple-v2", *key),
                "head_id": head["entity_id"],
                "head_name": head["name"],
                "head_type": head["entity_type"],
                "relation_type": relation_value,
                "tail_id": tail["entity_id"],
                "tail_name": tail["name"],
                "tail_type": tail["entity_type"],
                "evidence": [],
                "annotation_version": ANNOTATION_VERSION,
                "ontology_version": ONTOLOGY_VERSION,
                "schema_valid": True,
                "gold_nature": "ai_expert_pre_gold",
            }
            self.triple_index[key] = triple
        evidence_record = {
            "standard_code": unit["provenance"]["standard_code"],
            "source_sha256": unit["provenance"]["source_sha256"],
            "source_unit_id": unit["unit_id"],
            "evidence_span_ids": spans,
            "pages": _pages(unit),
            "section": unit["provenance"].get("section", ""),
            "review_status": self.annotations[unit["unit_id"]][
                "review_status"
            ],
            "bboxes": {
                str(span["page"]): span["bbox"]
                for span in _all_evidence_spans(unit)
                if span.get("span_id") in spans
            },
        }
        if evidence_record not in triple["evidence"]:
            triple["evidence"].append(evidence_record)

    def _subject(
        self, unit: dict[str, Any]
    ) -> dict[str, Any] | None:
        code = unit["provenance"]["standard_code"]
        name = ASSESSMENT_SUBJECTS.get(code)
        if not name:
            return None
        return self.add_entity(
            unit,
            name,
            EntityTypeV2.ASSESSMENT_SUBJECT,
        )

    def annotate_formula(self, unit: dict[str, Any]) -> None:
        variables = [
            expanded
            for variable in unit.get("variable_definitions", [])
            for expanded in _split_compound_variable(variable)
        ]
        role_overrides = unit.get("manual_variable_roles", {})
        variable_entities: dict[tuple[str, str], dict[str, Any]] = {}
        for formula in unit.get("formulas", []):
            for symbol in _formula_lhs_symbols(
                formula.get("expression_text", "")
            ):
                exact_matches = [
                    variable
                    for variable in variables
                    if _symbol_identity(variable["symbol"])
                    == _symbol_identity(symbol)
                ]
                base_matches = [
                    variable
                    for variable in variables
                    if _symbol_identity(_symbol_base(variable["symbol"]))
                    == _symbol_identity(_symbol_base(symbol))
                ]
                if exact_matches or len(base_matches) == 1:
                    continue
                variables.append(
                    {
                        "symbol": symbol,
                        "definition": "公式左侧输出（原文未单列变量释义）",
                        "unit": "",
                        "evidence_span": formula["evidence_span"],
                        "inferred_from_formula_lhs": True,
                    }
                )
        for formula in unit.get("formulas", []):
            number = formula["formula_number"]
            section = unit["provenance"].get("section", "")
            formula_context = f"{section}|{number}"
            span_ids = [formula["evidence_span"]["span_id"]]
            formula_entity = self.add_entity(
                unit,
                f"公式（{number}）",
                EntityTypeV2.FORMULA,
                span_ids,
                definition=formula.get("expression_text", ""),
                context=formula_context,
            )
            indicator_name = _formula_indicator(unit, number)
            indicator = None
            if indicator_name and unit["provenance"]["standard_code"] in ASSESSMENT_SUBJECTS:
                indicator = self.add_entity(
                    unit,
                    indicator_name,
                    EntityTypeV2.ASSESSMENT_INDICATOR,
                    span_ids,
                )
                self.add_relation(
                    unit,
                    self._subject(unit),
                    RelationTypeV2.HAS_INDICATOR,
                    indicator,
                    span_ids,
                )
                self.add_relation(
                    unit,
                    indicator,
                    RelationTypeV2.CALCULATED_BY,
                    formula_entity,
                    span_ids,
                )

            expression = formula.get("expression_text", "")
            output_symbols = [
                symbol
                for symbol, role in role_overrides.items()
                if role == "output"
            ]
            if not output_symbols:
                output_symbols = _formula_output_symbols(expression, variables)
            matched_variables = [
                variable
                for variable in variables
                if _symbol_occurs(variable["symbol"], expression)
                or variable["symbol"] in output_symbols
            ]
            if "=" not in expression:
                matched_variables = list(variables)
            if not matched_variables:
                non_outputs = [
                    variable
                    for variable in variables
                    if variable["symbol"] not in output_symbols
                ]
                if len(unit.get("formulas", [])) == 1:
                    matched_variables = [
                        *[
                            variable
                            for variable in variables
                            if variable["symbol"] in output_symbols
                        ],
                        *non_outputs,
                    ]
            for variable in matched_variables:
                symbol = variable["symbol"]
                variable_span = variable.get("evidence_span", {})
                variable_spans = [
                    variable_span.get("span_id")
                ] if variable_span.get("span_id") else span_ids
                entity = self.add_entity(
                    unit,
                    symbol,
                    EntityTypeV2.MODEL_VARIABLE,
                    variable_spans,
                    definition=variable.get("definition", ""),
                    context=formula_context,
                )
                if entity is None:
                    continue
                variable_entities[(formula_context, symbol)] = entity
                role = role_overrides.get(symbol)
                if role == "output" or symbol in output_symbols:
                    relation = RelationTypeV2.HAS_OUTPUT
                else:
                    relation = RelationTypeV2.HAS_INPUT
                self.add_relation(
                    unit,
                    formula_entity,
                    relation,
                    entity,
                    span_ids + variable_spans,
                )
                if variable.get("unit"):
                    unit_entity = self.add_entity(
                        unit,
                        variable["unit"],
                        EntityTypeV2.UNIT,
                        variable_spans,
                    )
                    self.add_relation(
                        unit,
                        entity,
                        RelationTypeV2.HAS_UNIT,
                        unit_entity,
                        variable_spans,
                    )

            package_text = " ".join(
                [
                    unit.get("introduction", ""),
                    unit.get("interstitial_text", ""),
                    unit.get("adjacent_source_text", ""),
                    " ".join(
                        variable.get("definition", "")
                        for variable in variables
                    ),
                ]
            )
            for source_name in _extract_sources(package_text):
                source_entity = self.add_entity(
                    unit,
                    source_name,
                    EntityTypeV2.DATA_SOURCE,
                    span_ids,
                )
                for variable in matched_variables:
                    if not (
                        set(source_name.replace("数据", ""))
                        & set(variable.get("definition", ""))
                    ):
                        continue
                    variable_entity = variable_entities.get(
                        (formula_context, variable["symbol"])
                    )
                    self.add_relation(
                        unit,
                        variable_entity,
                        RelationTypeV2.SOURCED_FROM,
                        source_entity,
                        span_ids,
                    )
            for ecosystem_name in _extract_ecosystems(package_text):
                ecosystem = self.add_entity(
                    unit,
                    ecosystem_name,
                    EntityTypeV2.ECOSYSTEM_TYPE,
                    span_ids,
                )
                self.add_relation(
                    unit,
                    formula_entity,
                    RelationTypeV2.APPLIES_TO_ECOSYSTEM,
                    ecosystem,
                    span_ids,
                )
            for space_name in _extract_spaces(package_text):
                space = self.add_entity(
                    unit,
                    space_name,
                    EntityTypeV2.SPATIAL_SCOPE,
                    span_ids,
                )
                self.add_relation(
                    unit,
                    formula_entity,
                    RelationTypeV2.APPLIES_TO_SPACE,
                    space,
                    span_ids,
                )

    def annotate_observation_table(self, unit: dict[str, Any]) -> None:
        cells = unit.get("cells", {})
        name = cells.get("观测指标") or cells.get("核查指标")
        if not name:
            return
        spans = _span_ids(unit)
        variable = self.add_entity(
            unit,
            name,
            EntityTypeV2.OBSERVATION_VARIABLE,
            spans,
            definition=(
                cells.get("指标定义")
                or cells.get("指标含义")
                or ""
            ),
        )
        temporal = cells.get("观测时间", "")
        if temporal and temporal not in {"—", "-", "/"}:
            temporal_entity = self.add_entity(
                unit,
                temporal,
                EntityTypeV2.TEMPORAL_SCOPE,
                spans,
            )
            self.add_relation(
                unit,
                variable,
                RelationTypeV2.OBSERVED_DURING,
                temporal_entity,
                spans,
            )
        frequency = cells.get("观测频度", "")
        if frequency and frequency not in {"—", "-", "/"}:
            frequency_entity = self.add_entity(
                unit,
                frequency,
                EntityTypeV2.FREQUENCY,
                spans,
            )
            self.add_relation(
                unit,
                variable,
                RelationTypeV2.OBSERVED_EVERY,
                frequency_entity,
                spans,
            )
        definition = " ".join(str(value) for value in cells.values() if value)
        for instrument_name in _extract_instruments(definition):
            instrument = self.add_entity(
                unit,
                instrument_name,
                EntityTypeV2.INSTRUMENT,
                spans,
            )
            self.add_relation(
                unit,
                variable,
                RelationTypeV2.MEASURED_WITH,
                instrument,
                spans,
            )

    def annotate_quality_table(self, unit: dict[str, Any]) -> None:
        cells = unit.get("cells", {})
        requirement = cells.get("具体要求", "")
        if not requirement:
            return
        spans = _span_ids(unit)
        content = _clean_name(cells.get("内容", "数据"))
        target_name = re.sub(r"质量$", "", content) or "数据"
        target = self.add_entity(
            unit,
            target_name,
            EntityTypeV2.DATA_SOURCE,
            spans,
        )
        label = _clean_name(cells.get("二级指标", "质量规则"))
        rule_name = f"{label}：{_clean_name(requirement)}"
        rule = self.add_entity(
            unit,
            rule_name[:240],
            EntityTypeV2.QUALITY_RULE,
            spans,
            definition=requirement,
            context=unit["unit_id"],
        )
        self.add_relation(
            unit,
            target,
            RelationTypeV2.CONSTRAINED_BY,
            rule,
            spans,
        )

    def annotate_indicator_table(self, unit: dict[str, Any]) -> None:
        cells = unit.get("cells", {})
        indicator_name = (
            cells.get("评估指标")
            or cells.get("调查评估指标")
            or cells.get("二级指标")
        )
        if not indicator_name:
            return
        spans = _span_ids(unit)
        indicator = self.add_entity(
            unit,
            indicator_name,
            EntityTypeV2.ASSESSMENT_INDICATOR,
            spans,
            definition=cells.get("指标定义") or cells.get("指标含义") or "",
        )
        self.add_relation(
            unit,
            self._subject(unit),
            RelationTypeV2.HAS_INDICATOR,
            indicator,
            spans,
        )
        text = " ".join(str(value) for value in cells.values() if value)
        for ecosystem_name in _extract_ecosystems(text):
            ecosystem = self.add_entity(
                unit,
                ecosystem_name,
                EntityTypeV2.ECOSYSTEM_TYPE,
                spans,
            )
            self.add_relation(
                unit,
                indicator,
                RelationTypeV2.APPLIES_TO_ECOSYSTEM,
                ecosystem,
                spans,
            )

    def annotate_classification_table(self, unit: dict[str, Any]) -> bool:
        title = unit.get("table_title", "")
        cells = unit.get("cells", {})
        if not re.search(r"分级|等级|程度", title):
            return False
        text = "；".join(
            f"{key}={value}"
            for key, value in cells.items()
            if value not in {None, "", "—"}
        )
        if not re.search(r"[<>≤≥％%]|\d", text):
            return False
        code = unit["provenance"]["standard_code"]
        if code not in ASSESSMENT_SUBJECTS:
            return False
        indicator_name = re.sub(
            r"^(?:表\s*[A-Z]?\d+(?:\.\d+)?)\s*",
            "",
            title,
        )
        indicator_name = re.sub(r"(?:分级标准表?|等级划分表?)$", "", indicator_name)
        indicator_name = _clean_name(indicator_name)
        if indicator_name == "生态系统质量":
            indicator_name = "生态系统质量指数（EQI）"
        spans = _span_ids(unit)
        indicator = self.add_entity(
            unit,
            indicator_name,
            EntityTypeV2.ASSESSMENT_INDICATOR,
            spans,
        )
        rule = self.add_entity(
            unit,
            text[:240],
            EntityTypeV2.CLASSIFICATION_RULE,
            spans,
            definition=text,
            context=unit["unit_id"],
        )
        self.add_relation(
            unit,
            self._subject(unit),
            RelationTypeV2.HAS_INDICATOR,
            indicator,
            spans,
        )
        self.add_relation(
            unit,
            indicator,
            RelationTypeV2.CLASSIFIED_BY,
            rule,
            spans,
        )
        return True

    def annotate_ecosystem_table(self, unit: dict[str, Any]) -> bool:
        cells = unit.get("cells", {})
        names = [
            value
            for key, value in cells.items()
            if "分类" in key and value
        ]
        if not names:
            return False
        spans = _span_ids(unit)
        for name in names:
            self.add_entity(
                unit,
                name,
                EntityTypeV2.ECOSYSTEM_TYPE,
                spans,
            )
        return True

    def annotate_table(self, unit: dict[str, Any]) -> None:
        if self.annotate_ecosystem_table(unit):
            return
        if self.annotate_classification_table(unit):
            return
        code = unit["provenance"]["standard_code"]
        cells = unit.get("cells", {})
        if code == "HJ 1176-2021" and "具体要求" in cells:
            self.annotate_quality_table(unit)
            return
        if "观测指标" in cells or "核查指标" in cells:
            self.annotate_observation_table(unit)
            return
        if (
            "评估指标" in cells
            or "调查评估指标" in cells
            or (
                "二级指标" in cells
                and code in ASSESSMENT_SUBJECTS
            )
        ):
            self.annotate_indicator_table(unit)

    def _observation_name_from_clause(
        self, unit: dict[str, Any], text: str
    ) -> str:
        match = re.search(r"观测指标[：:]\s*([^；;，,]+)", text)
        if match:
            return _clean_name(match.group(1))
        section = unit["provenance"].get("section", "")
        if re.match(r"(?:7\.3|9)(?:\.|$)", section):
            return _heading_title(unit)
        return ""

    def annotate_procedure(self, unit: dict[str, Any]) -> None:
        text = unit.get("clause_text", "")
        if (
            not text
            or _pages(unit)[0] < 4
            or "目 次" in text
            or len(text) < 8
        ):
            return
        code = unit["provenance"]["standard_code"]
        spans = _span_ids(unit)
        if code in OBSERVATION_CODES:
            name = self._observation_name_from_clause(unit, text)
            variable = None
            if name and name not in {"野外观测技术方法", "野外核查"}:
                variable = self.add_entity(
                    unit,
                    name,
                    EntityTypeV2.OBSERVATION_VARIABLE,
                    spans,
                )
            methods = _extract_methods(text)
            for method_name in methods:
                method = self.add_entity(
                    unit,
                    method_name,
                    EntityTypeV2.METHOD,
                    spans,
                )
                self.add_relation(
                    unit,
                    variable,
                    RelationTypeV2.OBTAINED_BY,
                    method,
                    spans,
                )
            for instrument_name in _extract_instruments(text):
                instrument = self.add_entity(
                    unit,
                    instrument_name,
                    EntityTypeV2.INSTRUMENT,
                    spans,
                )
                self.add_relation(
                    unit,
                    variable,
                    RelationTypeV2.MEASURED_WITH,
                    instrument,
                    spans,
                )
            for source_name in _extract_sources(text):
                source = self.add_entity(
                    unit,
                    source_name,
                    EntityTypeV2.DATA_SOURCE,
                    spans,
                )
                self.add_relation(
                    unit,
                    variable,
                    RelationTypeV2.SOURCED_FROM,
                    source,
                    spans,
                )
            for temporal_name in unit.get("temporal_mentions", []):
                temporal = self.add_entity(
                    unit,
                    temporal_name,
                    EntityTypeV2.TEMPORAL_SCOPE,
                    spans,
                )
                self.add_relation(
                    unit,
                    variable,
                    RelationTypeV2.OBSERVED_DURING,
                    temporal,
                    spans,
                )
            for frequency_name in unit.get("frequency_mentions", []):
                frequency = self.add_entity(
                    unit,
                    frequency_name,
                    EntityTypeV2.FREQUENCY,
                    spans,
                )
                self.add_relation(
                    unit,
                    variable,
                    RelationTypeV2.OBSERVED_EVERY,
                    frequency,
                    spans,
                )
            return

        if code in ASSESSMENT_SUBJECTS:
            section = unit["provenance"].get("section", "")
            title = _heading_title(unit)
            if (
                title
                and re.match(r"(?:6|7|9|A|B)(?:\.|$)", section)
                and title not in {"评估技术方法", "评估结果"}
            ):
                indicator = self.add_entity(
                    unit,
                    title,
                    EntityTypeV2.ASSESSMENT_INDICATOR,
                    spans,
                )
                self.add_relation(
                    unit,
                    self._subject(unit),
                    RelationTypeV2.HAS_INDICATOR,
                    indicator,
                    spans,
                )
            for source_name in _extract_sources(text):
                self.add_entity(
                    unit,
                    source_name,
                    EntityTypeV2.DATA_SOURCE,
                    spans,
                )
            for ecosystem_name in _extract_ecosystems(text):
                ecosystem = self.add_entity(
                    unit,
                    ecosystem_name,
                    EntityTypeV2.ECOSYSTEM_TYPE,
                    spans,
                )
                if "indicator" in locals() and indicator:
                    self.add_relation(
                        unit,
                        indicator,
                        RelationTypeV2.APPLIES_TO_ECOSYSTEM,
                        ecosystem,
                        spans,
                    )
            for space_name in _extract_spaces(text):
                space = self.add_entity(
                    unit,
                    space_name,
                    EntityTypeV2.SPATIAL_SCOPE,
                    spans,
                )
                if "indicator" in locals() and indicator:
                    self.add_relation(
                        unit,
                        indicator,
                        RelationTypeV2.APPLIES_TO_SPACE,
                        space,
                        spans,
                    )

    def annotate_quality_clause(self, unit: dict[str, Any]) -> None:
        text = unit.get("clause_text", "")
        if (
            unit["provenance"]["standard_code"] != "HJ 1176-2021"
            or _pages(unit)[0] < 5
            or len(text) < 20
            or "目 次" in text
        ):
            return
        if not re.search(
            r"精度|完整|质量|误差|小于|大于|不少于|不低于|应|要求",
            text,
        ):
            return
        spans = _span_ids(unit)
        sources = _extract_sources(text)
        if not sources:
            return
        rule = self.add_entity(
            unit,
            _clean_name(text)[:240],
            EntityTypeV2.QUALITY_RULE,
            spans,
            definition=text,
            context=unit["unit_id"],
        )
        for source_name in sources:
            source = self.add_entity(
                unit,
                source_name,
                EntityTypeV2.DATA_SOURCE,
                spans,
            )
            self.add_relation(
                unit,
                source,
                RelationTypeV2.CONSTRAINED_BY,
                rule,
                spans,
            )

    def run(self) -> AnnotationResult:
        for unit in self.units:
            override = self.annotation_overrides.get(unit["unit_id"])
            if override is not None:
                self.annotate_expert_override(unit, override)
                annotation = self.annotations[unit["unit_id"]]
                if not annotation["relations"] and not annotation[
                    "no_relation_reason"
                ]:
                    annotation["no_relation_reason"] = (
                        "adjudicated_no_explicit_schema_v2_relation"
                    )
                continue
            unit_type = unit["unit_type"]
            if unit_type == "formula_package":
                self.annotate_formula(unit)
            elif unit_type == "table_record":
                self.annotate_table(unit)
            elif unit_type == "procedure_clause":
                self.annotate_procedure(unit)
            elif unit_type == "quality_clause":
                self.annotate_quality_clause(unit)
            annotation = self.annotations[unit["unit_id"]]
            if not annotation["relations"]:
                annotation["no_relation_reason"] = (
                    "no_explicit_schema_v2_relation"
                )
        entities = sorted(
            self.entity_index.values(),
            key=lambda item: (
                item["standard_code"],
                item["entity_type"],
                item["name"],
                item["entity_id"],
            ),
        )
        triples = sorted(
            self.triple_index.values(),
            key=lambda item: (
                item["head_type"],
                item["relation_type"],
                item["tail_type"],
                item["triple_id"],
            ),
        )
        annotations = [
            self.annotations[unit["unit_id"]] for unit in self.units
        ]
        return AnnotationResult(
            source_units=self.units,
            unit_annotations=annotations,
            entities=entities,
            triples=triples,
            rejected_relations=self.rejected,
        )


def split_for_unit(unit: dict[str, Any]) -> str:
    code = unit["provenance"]["standard_code"]
    if code in EXTERNAL_TEST_CODES:
        return "test"
    if code not in {"HJ 1172-2021", "HJ 1173-2021"}:
        return "train"
    if unit["unit_type"] == "table_record":
        group = f"{code}|table|{unit.get('table_id', '')}"
    else:
        group = f"{code}|section|{unit['provenance'].get('section', '')}"
    bucket = int(stable_id("split-v2", group), 16) % 5
    return "dev" if bucket == 0 else "train"


def split_triples(
    units: list[dict[str, Any]],
    triples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    unit_split = {unit["unit_id"]: split_for_unit(unit) for unit in units}
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        evidence_splits = {
            unit_split[evidence["source_unit_id"]]
            for evidence in triple["evidence"]
        }
        split = (
            "test"
            if "test" in evidence_splits
            else "dev"
            if "dev" in evidence_splits
            else "train"
        )
        item = dict(triple)
        item["split"] = split
        output[split].append(item)
    return {
        split: sorted(items, key=lambda item: item["triple_id"])
        for split, items in output.items()
    }
