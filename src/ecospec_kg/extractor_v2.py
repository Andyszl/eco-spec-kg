from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .experiment_io_v2 import (
    assert_blind_records,
    load_json_config,
    read_jsonl,
    runtime_metadata,
    sha256_json,
    sha256_path,
    utc_now,
    write_json,
    write_jsonl,
)
from .io_utils import normalize_space, stable_id
from .ontology_v2 import EntityTypeV2, ONTOLOGY_VERSION, RelationTypeV2
from .prediction_contract_v2 import PREDICTION_SCHEMA_VERSION
from .providers import OpenAICompatibleProvider


RUN_MANIFEST_VERSION = "ecospec-extraction-run-v2.0"

OBSERVATION_CODES = {
    "HJ 1166-2021",
    "HJ 1167-2021",
    "HJ 1168-2021",
    "HJ 1169-2021",
    "HJ 1170-2021",
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
    "定位工具",
)
DATA_SOURCE_TERMS = (
    "遥感影像数据",
    "遥感图像",
    "遥感影像",
    "遥感数据",
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
FORMULA_INDICATORS = {
    ("HJ 1172-2021", "1"): "生态系统参数相对密度",
    ("HJ 1172-2021", "2"): "归一化指数",
    ("HJ 1172-2021", "3"): "生态系统质量指数（EQI）",
    ("HJ 1172-2021", "B.1"): "叶面积指数",
    ("HJ 1172-2021", "B.2"): "叶面积指数",
    ("HJ 1172-2021", "B.3"): "叶面积指数",
    ("HJ 1172-2021", "B.4"): "植被覆盖度",
    ("HJ 1173-2021", "A.1"): "水源涵养量",
}
for _number in range(2, 14):
    FORMULA_INDICATORS[("HJ 1173-2021", f"A.{_number}")] = "土壤保持量"
for _number in range(14, 23):
    FORMULA_INDICATORS[("HJ 1173-2021", f"A.{_number}")] = "防风固沙量"
FORMULA_INDICATORS[("HJ 1173-2021", "A.23")] = "生境不可替代性指数"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _clean(value: str) -> str:
    return normalize_space(str(value)).strip("；;：:，,。 ")


def _all_spans(value: Any) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        if all(key in value for key in ("span_id", "page", "bbox")):
            spans[str(value["span_id"])] = value
        for child in value.values():
            spans.update(_all_spans(child))
    elif isinstance(value, list):
        for child in value:
            spans.update(_all_spans(child))
    return spans


def _span_ids(unit: dict[str, Any]) -> list[str]:
    return list(_all_spans(unit))


def _heading_title(unit: dict[str, Any]) -> str:
    chain = unit.get("provenance", {}).get("heading_chain", [])
    if not chain:
        return ""
    title = re.sub(
        r"^(?:附录\s*[A-Z]|[A-Z]?\d+(?:\.\d+)*)\s*",
        "",
        str(chain[-1]),
    )
    return _clean(title)


def _page(unit: dict[str, Any]) -> int:
    pages = unit.get("provenance", {}).get("pages", [])
    return int(pages[0]) if pages else 0


def _assessment_subject_name(unit: dict[str, Any]) -> str:
    title = _clean(unit.get("provenance", {}).get("document_title", ""))
    if not title.endswith("评估"):
        return ""
    return _clean(re.sub(r"评估$", "", title))


def _terms(text: str, terms: tuple[str, ...]) -> list[str]:
    matches = [term for term in terms if term in text]
    # Keep the most specific form when one candidate contains another.
    return [
        item
        for item in matches
        if not any(item != other and item in other for other in matches)
    ]


def _formula_lhs_symbols(expression: str) -> list[str]:
    symbols: list[str] = []
    for match in re.finditer("=", expression):
        segment = re.split(r"[;\n]", expression[: match.start()])[-1].strip()
        found = re.search(
            r"([A-Za-z\u0370-\u03ff\u4e00-\u9fff]+[′’']?"
            r"(?:_[A-Za-z0-9,]+)?(?:\s*\([^()=]{1,20}\))?)\s*$",
            segment,
        )
        if found:
            symbols.append(re.sub(r"[\s{}]", "", found.group(1)))
    return _unique(symbols)


def _symbol_base(symbol: str) -> str:
    return re.split(r"[_\s]", symbol, maxsplit=1)[0].strip("′'")


def _symbol_occurs(symbol: str, expression: str) -> bool:
    compact_symbol = re.sub(r"[\s_{}]", "", symbol)
    compact_expression = re.sub(r"[\s_{}]", "", expression)
    if compact_symbol and compact_symbol in compact_expression:
        return True
    base = re.sub(r"[\s_{}]", "", _symbol_base(symbol))
    return bool(base and base in compact_expression)


def _split_variable(variable: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = str(variable.get("symbol", ""))
    if not re.fullmatch(r"[A-Za-z](?:、[A-Za-z])+(?:和[A-Za-z])?", symbol):
        return [variable]
    return [{**variable, "symbol": item} for item in re.findall(r"[A-Za-z]", symbol)]


@dataclass(slots=True)
class PredictionBuilder:
    unit: dict[str, Any]
    backend: str
    entities: dict[tuple[str, str], dict[str, Any]] = field(init=False)
    relations: dict[
        tuple[str, str, str, str, str], dict[str, Any]
    ] = field(init=False)

    def __post_init__(self) -> None:
        self.entities: dict[tuple[str, str], dict[str, Any]] = {}
        self.relations: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add_entity(
        self,
        name: str,
        entity_type: EntityTypeV2 | str,
        spans: list[str] | None = None,
    ) -> dict[str, Any] | None:
        cleaned = _clean(name)
        if not cleaned or cleaned in {"—", "-", "/", "无", "其他"}:
            return None
        type_value = entity_type.value if isinstance(entity_type, EntityTypeV2) else entity_type
        key = (cleaned, type_value)
        evidence = _unique(spans or _span_ids(self.unit))
        if not evidence:
            return None
        entity = self.entities.get(key)
        if entity is None:
            entity = {
                "entity_id": stable_id(
                    "prediction-entity-v2",
                    self.unit["unit_id"],
                    type_value,
                    cleaned,
                ),
                "name": cleaned,
                "entity_type": type_value,
                "evidence_span_ids": evidence,
            }
            self.entities[key] = entity
        else:
            entity["evidence_span_ids"] = _unique(
                entity["evidence_span_ids"] + evidence
            )
        return entity

    def add_relation(
        self,
        head: dict[str, Any] | None,
        relation_type: RelationTypeV2 | str,
        tail: dict[str, Any] | None,
        spans: list[str] | None = None,
    ) -> None:
        if head is None or tail is None:
            return
        relation_value = (
            relation_type.value
            if isinstance(relation_type, RelationTypeV2)
            else relation_type
        )
        key = (
            head["name"],
            head["entity_type"],
            relation_value,
            tail["name"],
            tail["entity_type"],
        )
        evidence = _unique(spans or _span_ids(self.unit))
        self.relations[key] = {
            "relation_id": stable_id("prediction-relation-v2", self.unit["unit_id"], *key),
            "head_id": head["entity_id"],
            "head_name": head["name"],
            "head_type": head["entity_type"],
            "relation_type": relation_value,
            "tail_id": tail["entity_id"],
            "tail_name": tail["name"],
            "tail_type": tail["entity_type"],
            "evidence_span_ids": evidence,
        }

    def result(self) -> dict[str, Any]:
        entities = sorted(
            self.entities.values(),
            key=lambda item: (item["entity_type"], item["name"], item["entity_id"]),
        )
        relations = sorted(
            self.relations.values(),
            key=lambda item: (
                item["relation_type"],
                item["head_name"],
                item["tail_name"],
                item["relation_id"],
            ),
        )
        return {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "ontology_version": ONTOLOGY_VERSION,
            "unit_id": self.unit["unit_id"],
            "standard_code": self.unit["provenance"]["standard_code"],
            "unit_type": self.unit["unit_type"],
            "backend": self.backend,
            "status": "success",
            "entities": entities,
            "relations": relations,
            "no_relation_reason": "" if relations else "no_explicit_schema_v2_relation",
        }


class RuleCandidateExtractorV2:
    def predict_unit(self, unit: dict[str, Any]) -> dict[str, Any]:
        builder = PredictionBuilder(unit, "rule")
        unit_type = unit.get("unit_type")
        if unit_type == "formula_package":
            self._formula(builder)
        elif unit_type == "table_record":
            self._table(builder)
        elif unit_type == "procedure_clause":
            self._procedure(builder)
        elif unit_type == "quality_clause":
            self._quality(builder)
        return builder.result()

    @staticmethod
    def _subject(builder: PredictionBuilder) -> dict[str, Any] | None:
        name = _assessment_subject_name(builder.unit)
        return builder.add_entity(name, EntityTypeV2.ASSESSMENT_SUBJECT) if name else None

    def _formula(self, builder: PredictionBuilder) -> None:
        unit = builder.unit
        variables = [
            expanded
            for variable in unit.get("variable_definitions", [])
            for expanded in _split_variable(variable)
        ]
        role_overrides = unit.get("manual_variable_roles", {})
        code = unit["provenance"]["standard_code"]
        package_text = " ".join(
            [
                str(unit.get("introduction", "")),
                str(unit.get("interstitial_text", "")),
                str(unit.get("adjacent_source_text", "")),
                " ".join(str(item.get("definition", "")) for item in variables),
            ]
        )
        for formula in unit.get("formulas", []):
            number = str(formula.get("formula_number", ""))
            formula_span = formula.get("evidence_span", {}).get("span_id")
            formula_spans = [formula_span] if formula_span else _span_ids(unit)
            formula_entity = builder.add_entity(
                f"公式（{number}）", EntityTypeV2.FORMULA, formula_spans
            )
            indicator_name = FORMULA_INDICATORS.get((code, number)) or _heading_title(unit)
            if indicator_name in {"评估技术方法", "指标与方法", "评估结果"}:
                indicator_name = ""
            if indicator_name and _assessment_subject_name(unit):
                indicator = builder.add_entity(
                    indicator_name, EntityTypeV2.ASSESSMENT_INDICATOR, formula_spans
                )
                builder.add_relation(
                    self._subject(builder),
                    RelationTypeV2.HAS_INDICATOR,
                    indicator,
                    formula_spans,
                )
                builder.add_relation(
                    indicator,
                    RelationTypeV2.CALCULATED_BY,
                    formula_entity,
                    formula_spans,
                )

            expression = str(formula.get("expression_text", ""))
            outputs = [
                symbol for symbol, role in role_overrides.items() if role == "output"
            ] or _formula_lhs_symbols(expression)
            matched = [
                variable
                for variable in variables
                if _symbol_occurs(str(variable.get("symbol", "")), expression)
            ]
            if "=" not in expression or not matched:
                matched = list(variables) if len(unit.get("formulas", [])) == 1 else matched
            variable_entities: list[dict[str, Any]] = []
            for variable in matched:
                symbol = str(variable.get("symbol", ""))
                variable_span = variable.get("evidence_span", {}).get("span_id")
                variable_spans = [variable_span] if variable_span else formula_spans
                entity = builder.add_entity(
                    symbol, EntityTypeV2.MODEL_VARIABLE, variable_spans
                )
                if entity is not None:
                    variable_entities.append(entity)
                role = role_overrides.get(symbol)
                relation = (
                    RelationTypeV2.HAS_OUTPUT
                    if role == "output"
                    or symbol in outputs
                    or _symbol_base(symbol) in {_symbol_base(item) for item in outputs}
                    else RelationTypeV2.HAS_INPUT
                )
                builder.add_relation(
                    formula_entity,
                    relation,
                    entity,
                    formula_spans + variable_spans,
                )
                unit_name = _clean(str(variable.get("unit", "")))
                if unit_name:
                    unit_entity = builder.add_entity(
                        unit_name, EntityTypeV2.UNIT, variable_spans
                    )
                    builder.add_relation(
                        entity,
                        RelationTypeV2.HAS_UNIT,
                        unit_entity,
                        variable_spans,
                    )
            for source in _terms(package_text, DATA_SOURCE_TERMS):
                source_entity = builder.add_entity(
                    source, EntityTypeV2.DATA_SOURCE, formula_spans
                )
                for variable_entity in variable_entities:
                    builder.add_relation(
                        variable_entity,
                        RelationTypeV2.SOURCED_FROM,
                        source_entity,
                        formula_spans,
                    )
            for ecosystem in _terms(package_text, ECOSYSTEM_TERMS):
                target = builder.add_entity(
                    ecosystem, EntityTypeV2.ECOSYSTEM_TYPE, formula_spans
                )
                builder.add_relation(
                    formula_entity,
                    RelationTypeV2.APPLIES_TO_ECOSYSTEM,
                    target,
                    formula_spans,
                )
            for space in _terms(package_text, SPATIAL_TERMS):
                target = builder.add_entity(space, EntityTypeV2.SPATIAL_SCOPE, formula_spans)
                builder.add_relation(
                    formula_entity,
                    RelationTypeV2.APPLIES_TO_SPACE,
                    target,
                    formula_spans,
                )

    def _table(self, builder: PredictionBuilder) -> None:
        unit = builder.unit
        cells = unit.get("cells", {})
        spans = _span_ids(unit)
        code = unit["provenance"]["standard_code"]
        if code == "HJ 1176-2021" and cells.get("具体要求"):
            target_name = re.sub(r"质量$", "", _clean(cells.get("内容", "数据"))) or "数据"
            target = builder.add_entity(target_name, EntityTypeV2.DATA_SOURCE, spans)
            label = _clean(cells.get("二级指标", "质量规则"))
            rule = builder.add_entity(
                f"{label}：{_clean(cells['具体要求'])}"[:240],
                EntityTypeV2.QUALITY_RULE,
                spans,
            )
            builder.add_relation(target, RelationTypeV2.CONSTRAINED_BY, rule, spans)
            return

        name = cells.get("观测指标") or cells.get("核查指标")
        if name:
            variable = builder.add_entity(name, EntityTypeV2.OBSERVATION_VARIABLE, spans)
            temporal = _clean(cells.get("观测时间", ""))
            frequency = _clean(cells.get("观测频度", ""))
            if temporal and temporal not in {"—", "-", "/"}:
                target = builder.add_entity(temporal, EntityTypeV2.TEMPORAL_SCOPE, spans)
                builder.add_relation(variable, RelationTypeV2.OBSERVED_DURING, target, spans)
            if frequency and frequency not in {"—", "-", "/"}:
                target = builder.add_entity(frequency, EntityTypeV2.FREQUENCY, spans)
                builder.add_relation(variable, RelationTypeV2.OBSERVED_EVERY, target, spans)
            text = " ".join(str(value) for value in cells.values() if value)
            for instrument in _terms(text, INSTRUMENT_TERMS):
                target = builder.add_entity(instrument, EntityTypeV2.INSTRUMENT, spans)
                builder.add_relation(variable, RelationTypeV2.MEASURED_WITH, target, spans)
            return

        title = str(unit.get("table_title", ""))
        if re.search(r"分级|等级|程度", title):
            text = "；".join(
                f"{key}={value}" for key, value in cells.items() if value not in {None, "", "—"}
            )
            if re.search(r"[<>≤≥％%]|\d", text) and _assessment_subject_name(unit):
                indicator_name = re.sub(
                    r"^(?:表\s*[A-Z]?\d+(?:\.\d+)?)\s*", "", title
                )
                indicator_name = re.sub(
                    r"(?:分级标准表?|等级划分表?)$", "", indicator_name
                )
                indicator_name = _clean(indicator_name)
                if indicator_name == "生态系统质量":
                    indicator_name = "生态系统质量指数（EQI）"
                indicator = builder.add_entity(
                    indicator_name, EntityTypeV2.ASSESSMENT_INDICATOR, spans
                )
                rule = builder.add_entity(
                    text[:240], EntityTypeV2.CLASSIFICATION_RULE, spans
                )
                builder.add_relation(
                    self._subject(builder), RelationTypeV2.HAS_INDICATOR, indicator, spans
                )
                builder.add_relation(
                    indicator, RelationTypeV2.CLASSIFIED_BY, rule, spans
                )
                return

        indicator_name = (
            cells.get("评估指标")
            or cells.get("调查评估指标")
            or (cells.get("二级指标") if _assessment_subject_name(unit) else "")
        )
        if indicator_name:
            indicator = builder.add_entity(
                indicator_name, EntityTypeV2.ASSESSMENT_INDICATOR, spans
            )
            builder.add_relation(
                self._subject(builder), RelationTypeV2.HAS_INDICATOR, indicator, spans
            )
            text = " ".join(str(value) for value in cells.values() if value)
            for ecosystem in _terms(text, ECOSYSTEM_TERMS):
                target = builder.add_entity(ecosystem, EntityTypeV2.ECOSYSTEM_TYPE, spans)
                builder.add_relation(
                    indicator, RelationTypeV2.APPLIES_TO_ECOSYSTEM, target, spans
                )

    def _procedure(self, builder: PredictionBuilder) -> None:
        unit = builder.unit
        text = str(unit.get("clause_text", ""))
        if not text or _page(unit) < 4 or "目 次" in text or len(text) < 8:
            return
        code = unit["provenance"]["standard_code"]
        spans = _span_ids(unit)
        if code in OBSERVATION_CODES:
            match = re.search(r"观测指标[：:]\s*([^；;，,]+)", text)
            section = unit["provenance"].get("section", "")
            name = _clean(match.group(1)) if match else (
                _heading_title(unit) if re.match(r"(?:7\.3|9)(?:\.|$)", section) else ""
            )
            variable = (
                builder.add_entity(name, EntityTypeV2.OBSERVATION_VARIABLE, spans)
                if name and name not in {"野外观测技术方法", "野外核查"}
                else None
            )
            for method_match in METHOD_RE.finditer(text):
                method_name = re.sub(
                    r"^(?:主要|具体|利用|采用|通过|运用|根据)",
                    "",
                    _clean(method_match.group(1)),
                )
                if method_name not in {"方法", "具体方法", "观测方法", "调查方法"}:
                    target = builder.add_entity(method_name, EntityTypeV2.METHOD, spans)
                    builder.add_relation(variable, RelationTypeV2.OBTAINED_BY, target, spans)
            for instrument in _terms(text, INSTRUMENT_TERMS):
                target = builder.add_entity(instrument, EntityTypeV2.INSTRUMENT, spans)
                builder.add_relation(variable, RelationTypeV2.MEASURED_WITH, target, spans)
            for source in _terms(text, DATA_SOURCE_TERMS):
                target = builder.add_entity(source, EntityTypeV2.DATA_SOURCE, spans)
                builder.add_relation(variable, RelationTypeV2.SOURCED_FROM, target, spans)
            for temporal in unit.get("temporal_mentions", []):
                target = builder.add_entity(temporal, EntityTypeV2.TEMPORAL_SCOPE, spans)
                builder.add_relation(variable, RelationTypeV2.OBSERVED_DURING, target, spans)
            for frequency in unit.get("frequency_mentions", []):
                target = builder.add_entity(frequency, EntityTypeV2.FREQUENCY, spans)
                builder.add_relation(variable, RelationTypeV2.OBSERVED_EVERY, target, spans)
            return

        if _assessment_subject_name(unit):
            section = unit["provenance"].get("section", "")
            title = _heading_title(unit)
            indicator = None
            if (
                title
                and re.match(r"(?:6|7|9|A|B)(?:\.|$)", section)
                and title not in {"评估技术方法", "评估结果"}
            ):
                indicator = builder.add_entity(
                    title, EntityTypeV2.ASSESSMENT_INDICATOR, spans
                )
                builder.add_relation(
                    self._subject(builder), RelationTypeV2.HAS_INDICATOR, indicator, spans
                )
            for ecosystem in _terms(text, ECOSYSTEM_TERMS):
                target = builder.add_entity(ecosystem, EntityTypeV2.ECOSYSTEM_TYPE, spans)
                builder.add_relation(
                    indicator, RelationTypeV2.APPLIES_TO_ECOSYSTEM, target, spans
                )
            for space in _terms(text, SPATIAL_TERMS):
                target = builder.add_entity(space, EntityTypeV2.SPATIAL_SCOPE, spans)
                builder.add_relation(indicator, RelationTypeV2.APPLIES_TO_SPACE, target, spans)

    def _quality(self, builder: PredictionBuilder) -> None:
        unit = builder.unit
        text = str(unit.get("clause_text", ""))
        if (
            unit["provenance"]["standard_code"] != "HJ 1176-2021"
            or _page(unit) < 5
            or len(text) < 20
            or "目 次" in text
            or not re.search(r"精度|完整|质量|误差|小于|大于|不少于|不低于|应|要求", text)
        ):
            return
        sources = _terms(text, DATA_SOURCE_TERMS)
        if not sources:
            return
        spans = _span_ids(unit)
        rule = builder.add_entity(_clean(text)[:240], EntityTypeV2.QUALITY_RULE, spans)
        for source in sources:
            target = builder.add_entity(source, EntityTypeV2.DATA_SOURCE, spans)
            builder.add_relation(target, RelationTypeV2.CONSTRAINED_BY, rule, spans)


def _parse_llm_selection(response: str) -> dict[str, list[str]]:
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    payload = json.loads(text)
    return {
        "selected_entity_ids": [str(item) for item in payload.get("selected_entity_ids", [])],
        "selected_relation_ids": [str(item) for item in payload.get("selected_relation_ids", [])],
    }


def build_llm_selection_messages(
    unit: dict[str, Any], candidates: dict[str, Any]
) -> tuple[str, str]:
    system = (
        "你是生态评估技术规范关系抽取器。只能从候选实体和候选关系中选择，"
        "不得改写名称、类型、关系或证据。只输出JSON。"
    )
    prompt = json.dumps(
        {
            "task": (
                "返回 selected_relation_ids 和 selected_entity_ids。仅选择原文明确支持的候选；"
                "被关系使用的实体无需重复放入 selected_entity_ids。"
            ),
            "source_unit": unit,
            "candidate_entities": candidates["entities"],
            "candidate_relations": candidates["relations"],
            "output_schema": {
                "selected_entity_ids": ["entity_id"],
                "selected_relation_ids": ["relation_id"],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return system, prompt


def _llm_select(
    provider: OpenAICompatibleProvider,
    unit: dict[str, Any],
    candidates: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    system, prompt = build_llm_selection_messages(unit, candidates)
    raw = provider.complete(system, prompt)
    selection = _parse_llm_selection(raw)
    entity_by_id = {item["entity_id"]: item for item in candidates["entities"]}
    relation_by_id = {item["relation_id"]: item for item in candidates["relations"]}
    selected_relations = [
        relation_by_id[item]
        for item in selection["selected_relation_ids"]
        if item in relation_by_id
    ]
    required_entities = {
        endpoint
        for relation in selected_relations
        for endpoint in (relation["head_id"], relation["tail_id"])
    }
    required_entities.update(
        item for item in selection["selected_entity_ids"] if item in entity_by_id
    )
    result = {
        **candidates,
        "backend": "llm",
        "entities": [
            item for item in candidates["entities"] if item["entity_id"] in required_entities
        ],
        "relations": selected_relations,
        "no_relation_reason": "" if selected_relations else "llm_selected_no_relation",
    }
    return result, raw, sha256_json({"system": system, "prompt": prompt})


def extract_v2(
    units_path: Path,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    backend: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    units = read_jsonl(units_path)
    if not units:
        raise ValueError("blind source unit file is empty")
    assert_blind_records(units)
    unit_ids = [unit.get("unit_id") for unit in units]
    if None in unit_ids or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("blind source unit ids must be present and unique")

    config = {
        "experiment_id": "rule_baseline_v2",
        "backend": "rule",
        "model": "deterministic-v2-rule",
        "seed": 42,
        "temperature": 0,
        "max_tokens": 1024,
        "candidate_generator": "structure-aware-rule-v2",
        "use_schema": True,
        "use_layout": True,
        "use_evidence": True,
        **load_json_config(config_path),
    }
    if backend is not None:
        config["backend"] = backend
    if model is not None:
        config["model"] = model
    if seed is not None:
        config["seed"] = seed
    if config["backend"] not in {"rule", "llm"}:
        raise ValueError("backend must be 'rule' or 'llm'")

    provider = None
    if config["backend"] == "llm":
        provider = OpenAICompatibleProvider.from_env()
        provider.model = str(config["model"])
        provider.max_tokens = int(config["max_tokens"])
        if provider.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if "enable_thinking" in config:
            value = config["enable_thinking"]
            if not isinstance(value, bool):
                raise ValueError("enable_thinking must be a boolean")
            provider.enable_thinking = value
    extractor = RuleCandidateExtractorV2()
    predictions: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    for unit in units:
        candidates: dict[str, Any] | None = None
        prompt_hash: str | None = None
        try:
            candidates = extractor.predict_unit(unit)
            if provider is None:
                prediction = candidates
            else:
                prediction, raw, prompt_hash = _llm_select(
                    provider, unit, candidates
                )
                raw_responses.append(
                    {
                        "unit_id": unit["unit_id"],
                        "prompt_hash": prompt_hash,
                        "response_sha256": sha256_json(raw),
                        "raw_response": raw,
                    }
                )
            prediction["experiment_id"] = config["experiment_id"]
            prediction["config_hash"] = sha256_json(config)
            prediction["candidate_hash"] = sha256_json(
                {
                    "entities": candidates["entities"],
                    "relations": candidates["relations"],
                }
            )
            prediction["model"] = config["model"]
            prediction["seed"] = config["seed"]
            prediction["prompt_hash"] = prompt_hash
            predictions.append(prediction)
        except Exception as exc:  # Keep every source unit visible to validation.
            candidate_hash = None
            if candidates is not None:
                candidate_hash = sha256_json(
                    {
                        "entities": candidates["entities"],
                        "relations": candidates["relations"],
                    }
                )
                if provider is not None and prompt_hash is None:
                    system, prompt = build_llm_selection_messages(unit, candidates)
                    prompt_hash = sha256_json({"system": system, "prompt": prompt})
            if provider is not None and provider.last_raw_response is not None:
                raw_responses.append(
                    {
                        "unit_id": unit["unit_id"],
                        "prompt_hash": prompt_hash,
                        "status": "error",
                        "provider_response": provider.last_raw_response,
                    }
                )
            predictions.append(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "ontology_version": ONTOLOGY_VERSION,
                    "unit_id": unit["unit_id"],
                    "standard_code": unit["provenance"]["standard_code"],
                    "unit_type": unit["unit_type"],
                    "backend": config["backend"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "entities": [],
                    "relations": [],
                    "no_relation_reason": "extraction_error",
                    "experiment_id": config["experiment_id"],
                    "config_hash": sha256_json(config),
                    "candidate_hash": candidate_hash,
                    "model": config["model"],
                    "seed": config["seed"],
                    "prompt_hash": prompt_hash,
                }
            )

    predictions.sort(key=lambda item: item["unit_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    raw_path = out_dir / "raw_responses.jsonl"
    resolved_config_path = out_dir / "resolved_config.json"
    write_jsonl(predictions_path, predictions)
    write_jsonl(raw_path, raw_responses)
    write_json(resolved_config_path, config)
    summary = {
        "unit_count": len(predictions),
        "success_count": sum(item["status"] == "success" for item in predictions),
        "error_count": sum(item["status"] != "success" for item in predictions),
        "entity_count": sum(len(item["entities"]) for item in predictions),
        "relation_count": sum(len(item["relations"]) for item in predictions),
    }
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": RUN_MANIFEST_VERSION,
        "run_id": stable_id(
            RUN_MANIFEST_VERSION,
            sha256_path(units_path),
            sha256_json(config),
        ),
        "experiment_id": config["experiment_id"],
        "ontology_version": ONTOLOGY_VERSION,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "complete" if summary["error_count"] == 0 else "completed_with_errors",
        "input": {"path": str(units_path), "sha256": sha256_path(units_path)},
        "config": {"path": str(resolved_config_path), "sha256": sha256_path(resolved_config_path)},
        "predictions": {"path": str(predictions_path), "sha256": sha256_path(predictions_path)},
        "raw_responses": {"path": str(raw_path), "sha256": sha256_path(raw_path)},
        "runtime": runtime_metadata(repo_root),
        "implementation": {
            "extractor_v2.py": sha256_path(Path(__file__)),
            "prediction_contract_v2.py": sha256_path(
                Path(__file__).with_name("prediction_contract_v2.py")
            ),
            "ontology_v2.py": sha256_path(
                Path(__file__).with_name("ontology_v2.py")
            ),
        },
        "summary": summary,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    return manifest


__all__ = [
    "PREDICTION_SCHEMA_VERSION",
    "RUN_MANIFEST_VERSION",
    "RuleCandidateExtractorV2",
    "extract_v2",
]
