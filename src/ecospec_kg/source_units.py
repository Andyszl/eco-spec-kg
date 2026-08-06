from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_json, stable_id, write_json, write_jsonl
from .layout_parser import (
    FORMULA_NUMBER_RE,
    LayoutDocument,
    LayoutLine,
    LayoutTable,
    load_layout_document,
    normalize_pdf_text,
)

SOURCE_UNIT_SCHEMA_VERSION = "source-unit-v2.0"

DEFINITION_RE = re.compile(
    r"^(?:式中[：:]?\s*)?(?P<symbol>[^，,；;。]{1,30}?)\s*"
    r"(?:——|--)\s*(?P<definition>.+)$"
)
ALT_DEFINITION_RE = re.compile(
    r"^(?:式中[：:]?\s*)?(?P<symbol>[A-Za-zΑ-ωΔλθρ′_,.]{1,16})\s+"
    r"(?P<definition>为.+)$"
)
SOURCE_TRIGGER_RE = re.compile(
    r"数据|资料|来源|获取|插值|遥感|监测|调查|数据库"
)
PROCEDURE_TRIGGER_RE = re.compile(
    r"采用|运用|通过|调查|观测|监测|测定|计算|模型|方程|仪器|设备|"
    r"时间范围|空间范围|频率|每年|每月|一年一次"
)
QUALITY_TRIGGER_RE = re.compile(
    r"质量控制|数据质量|误差|精度|准确率|完整性|一致性|异常值|缺失值|"
    r"阈值|校核|复核|有效性|剔除"
)


@dataclass(slots=True)
class EvidenceSpan:
    span_id: str
    page: int
    printed_page: str
    text: str
    bbox: list[float]
    line_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnitProvenance:
    standard_code: str
    document_title: str
    source_file: str
    source_sha256: str
    section: str
    heading_chain: list[str]
    pages: list[int]
    printed_pages: list[str]
    evidence_spans: list[EvidenceSpan]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_spans"] = [
            span.to_dict() for span in self.evidence_spans
        ]
        return data


@dataclass(slots=True)
class FormulaExpression:
    formula_number: str
    expression_text: str
    expression_lines: list[str]
    evidence_span: EvidenceSpan

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_span"] = self.evidence_span.to_dict()
        return data


@dataclass(slots=True)
class FormulaVariableDefinition:
    symbol: str
    definition: str
    unit: str
    evidence_span: EvidenceSpan

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_span"] = self.evidence_span.to_dict()
        return data


@dataclass(slots=True)
class FormulaPackage:
    unit_id: str
    provenance: UnitProvenance
    introduction: str
    interstitial_text: str
    formulas: list[FormulaExpression]
    variable_definitions: list[FormulaVariableDefinition]
    adjacent_source_text: str = ""
    unit_type: str = "formula_package"
    schema_version: str = SOURCE_UNIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "provenance": self.provenance.to_dict(),
            "introduction": self.introduction,
            "interstitial_text": self.interstitial_text,
            "formulas": [item.to_dict() for item in self.formulas],
            "variable_definitions": [
                item.to_dict() for item in self.variable_definitions
            ],
            "adjacent_source_text": self.adjacent_source_text,
        }


@dataclass(slots=True)
class TableRecord:
    unit_id: str
    provenance: UnitProvenance
    table_id: str
    table_title: str
    header_rows: list[list[str | None]]
    row_index: int
    raw_cells: dict[str, str | None]
    cells: dict[str, str | None]
    inherited_columns: list[str]
    cell_bboxes: dict[str, list[float] | None]
    inherited_from: dict[str, dict[str, Any]]
    table_notes: list[str] = field(default_factory=list)
    unit_type: str = "table_record"
    schema_version: str = SOURCE_UNIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "provenance": self.provenance.to_dict(),
            "table_id": self.table_id,
            "table_title": self.table_title,
            "header_rows": self.header_rows,
            "row_index": self.row_index,
            "raw_cells": self.raw_cells,
            "cells": self.cells,
            "inherited_columns": self.inherited_columns,
            "cell_bboxes": self.cell_bboxes,
            "inherited_from": self.inherited_from,
            "table_notes": self.table_notes,
        }


@dataclass(slots=True)
class ProcedureClause:
    unit_id: str
    provenance: UnitProvenance
    clause_text: str
    trigger_terms: list[str]
    temporal_mentions: list[str]
    frequency_mentions: list[str]
    instrument_mentions: list[str]
    unit_type: str = "procedure_clause"
    schema_version: str = SOURCE_UNIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "provenance": self.provenance.to_dict(),
            "clause_text": self.clause_text,
            "trigger_terms": self.trigger_terms,
            "temporal_mentions": self.temporal_mentions,
            "frequency_mentions": self.frequency_mentions,
            "instrument_mentions": self.instrument_mentions,
        }


@dataclass(slots=True)
class QualityClause:
    unit_id: str
    provenance: UnitProvenance
    clause_text: str
    trigger_terms: list[str]
    threshold_mentions: list[str]
    action_mentions: list[str]
    unit_type: str = "quality_clause"
    schema_version: str = SOURCE_UNIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "provenance": self.provenance.to_dict(),
            "clause_text": self.clause_text,
            "trigger_terms": self.trigger_terms,
            "threshold_mentions": self.threshold_mentions,
            "action_mentions": self.action_mentions,
        }


SourceUnit = FormulaPackage | TableRecord | ProcedureClause | QualityClause


def _union_bbox(lines: Iterable[LayoutLine]) -> list[float]:
    rows = list(lines)
    if not rows:
        return []
    return [
        round(min(line.bbox[0] for line in rows), 3),
        round(min(line.bbox[1] for line in rows), 3),
        round(max(line.bbox[2] for line in rows), 3),
        round(max(line.bbox[3] for line in rows), 3),
    ]


def _spans_from_lines(
    document: LayoutDocument, lines: Iterable[LayoutLine]
) -> list[EvidenceSpan]:
    page_map = {page.page: page.printed_page for page in document.pages}
    grouped: dict[int, list[LayoutLine]] = {}
    for line in lines:
        grouped.setdefault(line.page, []).append(line)
    spans: list[EvidenceSpan] = []
    for page_no, page_lines in sorted(grouped.items()):
        page_lines.sort(key=lambda line: line.reading_order)
        text = "\n".join(line.text for line in page_lines)
        bbox = _union_bbox(page_lines)
        spans.append(
            EvidenceSpan(
                span_id=stable_id(
                    document.standard_code,
                    page_no,
                    *(line.line_id for line in page_lines),
                ),
                page=page_no,
                printed_page=page_map.get(page_no, ""),
                text=text,
                bbox=bbox,
                line_ids=[line.line_id for line in page_lines],
            )
        )
    return spans


def _provenance(
    document: LayoutDocument,
    lines: list[LayoutLine],
    section: str = "",
    heading_chain: list[str] | None = None,
) -> UnitProvenance:
    spans = _spans_from_lines(document, lines)
    reference = lines[0] if lines else None
    return UnitProvenance(
        standard_code=document.standard_code,
        document_title=document.document_title,
        source_file=document.source_file,
        source_sha256=document.source_sha256,
        section=section or (reference.section if reference else ""),
        heading_chain=(
            list(heading_chain)
            if heading_chain is not None
            else list(reference.heading_chain if reference else [])
        ),
        pages=[span.page for span in spans],
        printed_pages=[span.printed_page for span in spans],
        evidence_spans=spans,
    )


def _content_lines(document: LayoutDocument) -> list[LayoutLine]:
    return [
        line
        for page in document.pages
        for line in page.lines
        if line.kind not in {"header_footer", "table_cell"}
    ]


def _is_definition_line(text: str) -> bool:
    return bool(
        DEFINITION_RE.match(text)
        or ALT_DEFINITION_RE.match(text)
        or text.startswith("式中")
    )


def _definition_end(lines: list[LayoutLine], start: int) -> int:
    index = start
    seen_definition = False
    while index < len(lines):
        line = lines[index]
        if line.kind in {"heading", "formula", "table_caption"}:
            break
        if DEFINITION_RE.match(line.text) or ALT_DEFINITION_RE.match(line.text):
            seen_definition = True
            index += 1
            continue
        if line.text.startswith("式中"):
            index += 1
            continue
        if (
            seen_definition
            and line.kind in {"formula_fragment", "variable_fragment"}
            and len(line.text) <= 20
        ):
            index += 1
            continue
        break
    return index


def _has_nearby_formula(lines: list[LayoutLine], start: int) -> bool:
    for line in lines[start : start + 12]:
        if line.kind == "heading" or line.text.startswith("式中"):
            return False
        if line.kind == "formula":
            return True
    return False


def _bridges_formula(text: str) -> bool:
    return bool(
        text.endswith(("：", ":"))
        or re.search(r"计算公式|公式如下|计算如下|侵蚀量|风力侵蚀", text)
    )


def _formula_clusters(lines: list[LayoutLine]) -> list[tuple[list[int], int, int]]:
    clusters: list[tuple[list[int], int, int]] = []
    index = 0
    while index < len(lines):
        if lines[index].kind != "formula":
            index += 1
            continue
        anchors = [index]
        cursor = index + 1
        definition_start = -1
        definition_end = -1
        while cursor < len(lines):
            line = lines[cursor]
            if line.kind == "heading":
                break
            if line.text.startswith("式中"):
                definition_start = cursor
                definition_end = _definition_end(lines, cursor)
                cursor = definition_end
                break
            if line.kind == "formula":
                anchors.append(cursor)
                cursor += 1
                continue
            if line.kind == "formula_fragment":
                cursor += 1
                continue
            if (
                line.kind == "paragraph"
                and _bridges_formula(line.text)
                and _has_nearby_formula(lines, cursor + 1)
            ):
                cursor += 1
                continue
            if line.kind == "paragraph" and re.search(
                r"[\u3400-\u9fff]", line.text
            ):
                break
            cursor += 1
        clusters.append((anchors, definition_start, definition_end))
        index = max(cursor, anchors[-1] + 1)
    return clusters


def _introduction_lines(
    lines: list[LayoutLine], first_anchor: int
) -> list[LayoutLine]:
    output: list[LayoutLine] = []
    index = first_anchor - 1
    while index >= 0 and len(output) < 5:
        line = lines[index]
        if line.kind in {
            "heading",
            "formula",
            "formula_fragment",
            "table_caption",
            "figure_caption",
        }:
            break
        if _is_definition_line(line.text):
            break
        if line.kind == "paragraph":
            output.append(line)
        index -= 1
    output.reverse()
    return output


def _expression_lines(
    lines: list[LayoutLine],
    anchor: int,
    next_anchor: int,
    definition_start: int,
) -> list[LayoutLine]:
    start = anchor
    while start > 0:
        previous = lines[start - 1]
        if (
            previous.kind != "formula_fragment"
            or previous.page != lines[anchor].page
        ):
            break
        start -= 1
    stop = min(
        value
        for value in (next_anchor, definition_start, len(lines))
        if value >= 0
    )
    output = [lines[start]]
    for index in range(start + 1, stop):
        if index == anchor or lines[index].kind == "formula_fragment":
            output.append(lines[index])
    return output


def _formula_number(text: str) -> str:
    match = FORMULA_NUMBER_RE.search(text)
    return match.group("number") if match else ""


def _extract_unit(definition: str) -> str:
    match = re.search(r"[，,]([^，,；;。]{1,40})[；;。]?$", definition)
    if not match:
        return ""
    candidate = match.group(1).strip()
    if re.search(r"[%％/∙·0-9A-Za-z]|量纲", candidate):
        return candidate
    return ""


def _variable_definitions(
    document: LayoutDocument,
    lines: list[LayoutLine],
    start: int,
    end: int,
) -> list[FormulaVariableDefinition]:
    if start < 0 or end <= start:
        return []
    output: list[FormulaVariableDefinition] = []
    index = start
    while index < end:
        line = lines[index]
        match = DEFINITION_RE.match(line.text) or ALT_DEFINITION_RE.match(
            line.text
        )
        if not match:
            index += 1
            continue
        symbol = re.sub(r"\s+", "", match.group("symbol")).strip("：:")
        definition = match.group("definition").strip()
        evidence_lines = [line]
        if index + 1 < end:
            fragment = lines[index + 1]
            if (
                fragment.kind in {"formula_fragment", "variable_fragment"}
                and len(fragment.text) <= 20
            ):
                suffix = re.sub(r"\s+", "", fragment.text)
                symbol = f"{symbol}_{suffix}"
                evidence_lines.append(fragment)
                index += 1
        span = _spans_from_lines(document, evidence_lines)[0]
        output.append(
            FormulaVariableDefinition(
                symbol=symbol,
                definition=definition,
                unit=_extract_unit(definition),
                evidence_span=span,
            )
        )
        index += 1
    return output


def _adjacent_source_lines(
    lines: list[LayoutLine], start: int
) -> list[LayoutLine]:
    output: list[LayoutLine] = []
    for line in lines[start : start + 4]:
        if line.kind in {"heading", "formula", "table_caption"}:
            break
        if line.kind != "paragraph":
            continue
        output.append(line)
    if not SOURCE_TRIGGER_RE.search("".join(line.text for line in output)):
        return []
    return output


def build_formula_packages(
    document: LayoutDocument,
) -> tuple[list[FormulaPackage], set[str]]:
    lines = _content_lines(document)
    packages: list[FormulaPackage] = []
    used_line_ids: set[str] = set()
    for anchors, definition_start, definition_end in _formula_clusters(lines):
        intro_lines = _introduction_lines(lines, anchors[0])
        expressions: list[FormulaExpression] = []
        expression_lines: list[LayoutLine] = []
        for position, anchor in enumerate(anchors):
            next_anchor = (
                anchors[position + 1]
                if position + 1 < len(anchors)
                else (
                    definition_start
                    if definition_start >= 0
                    else len(lines)
                )
            )
            rows = _expression_lines(
                lines, anchor, next_anchor, definition_start
            )
            expression_lines.extend(rows)
            span = _spans_from_lines(document, rows)[0]
            expressions.append(
                FormulaExpression(
                    formula_number=_formula_number(lines[anchor].text),
                    expression_text="\n".join(row.text for row in rows),
                    expression_lines=[row.text for row in rows],
                    evidence_span=span,
                )
            )

        definition_lines = (
            lines[definition_start:definition_end]
            if definition_start >= 0
            else []
        )
        interstitial_lines = [
            lines[index]
            for index in range(anchors[0] + 1, anchors[-1])
            if lines[index].kind == "paragraph"
            and not _is_definition_line(lines[index].text)
        ]
        source_lines = _adjacent_source_lines(
            lines, definition_end if definition_end >= 0 else anchors[-1] + 1
        )
        all_lines = _deduplicate_lines(
            intro_lines
            + expression_lines
            + interstitial_lines
            + definition_lines
            + source_lines
        )
        used_line_ids.update(line.line_id for line in all_lines)
        reference = lines[anchors[0]]
        numbers = [item.formula_number for item in expressions]
        unit_id = stable_id(
            document.standard_code,
            reference.section,
            *numbers,
            reference.page,
        )
        packages.append(
            FormulaPackage(
                unit_id=unit_id,
                provenance=_provenance(document, all_lines),
                introduction="".join(line.text for line in intro_lines),
                interstitial_text="".join(
                    line.text for line in interstitial_lines
                ),
                formulas=expressions,
                variable_definitions=_variable_definitions(
                    document,
                    lines,
                    definition_start,
                    definition_end,
                ),
                adjacent_source_text="".join(
                    line.text for line in source_lines
                ),
            )
        )
    return packages, used_line_ids


def _deduplicate_lines(lines: list[LayoutLine]) -> list[LayoutLine]:
    seen: set[str] = set()
    output: list[LayoutLine] = []
    for line in lines:
        if line.line_id not in seen:
            seen.add(line.line_id)
            output.append(line)
    output.sort(key=lambda row: (row.page, row.reading_order))
    return output


def _row_bbox(
    table: LayoutTable, row_index: int
) -> list[float]:
    cells = table.cell_bboxes[row_index]
    concrete = [cell for cell in cells if cell]
    if not concrete:
        return list(table.bbox)
    heights = [cell[3] - cell[1] for cell in concrete]
    minimum_height = min(heights)
    local_cells = [
        cell
        for cell in concrete
        if cell[3] - cell[1] <= minimum_height * 1.5
    ]
    return [
        table.bbox[0],
        round(min(cell[1] for cell in local_cells), 3),
        table.bbox[2],
        round(max(cell[3] for cell in local_cells), 3),
    ]


def _line_intersects_bbox(line: LayoutLine, bbox: list[float]) -> bool:
    return (
        min(line.bbox[2], bbox[2]) > max(line.bbox[0], bbox[0])
        and min(line.bbox[3], bbox[3]) > max(line.bbox[1], bbox[1])
    )


def _synthetic_table_span(
    document: LayoutDocument,
    table: LayoutTable,
    row_index: int,
    resolved: list[str | None],
) -> EvidenceSpan:
    page = next(page for page in document.pages if page.page == table.page)
    bbox = _row_bbox(table, row_index)
    line_ids = [
        line.line_id
        for line in page.lines
        if line.in_table and _line_intersects_bbox(line, bbox)
    ]
    text = " | ".join(value or "" for value in resolved)
    return EvidenceSpan(
        span_id=stable_id(table.table_id, row_index, text),
        page=table.page,
        printed_page=page.printed_page,
        text=text,
        bbox=bbox,
        line_ids=line_ids,
    )


def build_table_records(document: LayoutDocument) -> list[TableRecord]:
    records: list[TableRecord] = []
    inherited_state: dict[
        str,
        tuple[list[str | None], list[dict[str, Any] | None]],
    ] = {}
    for page in document.pages:
        for table in page.tables:
            width = len(table.column_names)
            prior_state = inherited_state.get(
                table.continued_from_table_id
            )
            if prior_state:
                previous_values = list(prior_state[0])
                previous_sources = [
                    dict(source) if source else None
                    for source in prior_state[1]
                ]
            else:
                previous_values = [None] * width
                previous_sources = [None] * width
            previous_values.extend(
                [None] * (width - len(previous_values))
            )
            previous_sources.extend(
                [None] * (width - len(previous_sources))
            )

            for row_index in range(table.data_start_row, len(table.rows)):
                raw = list(table.rows[row_index])
                raw.extend([None] * (width - len(raw)))
                raw_bboxes = list(table.cell_bboxes[row_index])
                raw_bboxes.extend([None] * (width - len(raw_bboxes)))
                resolved = list(raw)
                resolved_bboxes = list(raw_bboxes)
                inherited_columns: list[str] = []
                inherited_from: dict[str, dict[str, Any]] = {}
                for column in range(width):
                    column_name = table.column_names[column]
                    if (
                        raw[column] is None
                        and previous_values[column] is not None
                    ):
                        resolved[column] = previous_values[column]
                        source = previous_sources[column]
                        if source:
                            resolved_bboxes[column] = source.get("bbox")
                            inherited_from[column_name] = dict(source)
                        inherited_columns.append(column_name)
                    elif raw[column] is not None:
                        source = {
                            "table_id": table.table_id,
                            "page": table.page,
                            "row_index": row_index + 1,
                            "bbox": raw_bboxes[column],
                        }
                        previous_values[column] = raw[column]
                        previous_sources[column] = source
                if not any(value for value in resolved):
                    continue
                span = _synthetic_table_span(
                    document, table, row_index, resolved
                )
                provenance = UnitProvenance(
                    standard_code=document.standard_code,
                    document_title=document.document_title,
                    source_file=document.source_file,
                    source_sha256=document.source_sha256,
                    section=table.section,
                    heading_chain=list(table.heading_chain),
                    pages=[table.page],
                    printed_pages=[span.printed_page],
                    evidence_spans=[span],
                )
                records.append(
                    TableRecord(
                        unit_id=stable_id(
                            table.table_id, row_index, *resolved
                        ),
                        provenance=provenance,
                        table_id=table.table_id,
                        table_title=table.title,
                        header_rows=table.header_rows,
                        row_index=row_index + 1,
                        raw_cells=dict(zip(table.column_names, raw)),
                        cells=dict(zip(table.column_names, resolved)),
                        inherited_columns=inherited_columns,
                        cell_bboxes=dict(
                            zip(table.column_names, resolved_bboxes)
                        ),
                        inherited_from=inherited_from,
                        table_notes=table.notes,
                    )
                )
            inherited_state[table.table_id] = (
                previous_values,
                previous_sources,
            )
    return records


def _paragraph_groups(document: LayoutDocument) -> list[list[LayoutLine]]:
    groups: list[list[LayoutLine]] = []
    for page in document.pages:
        current: list[LayoutLine] = []
        previous: LayoutLine | None = None
        for line in page.lines:
            if (
                line.kind != "paragraph"
                or line.in_table
                or _is_definition_line(line.text)
            ):
                if current:
                    groups.append(current)
                    current = []
                previous = None
                continue
            starts_new = bool(
                previous
                and (
                    line.section != previous.section
                    or line.bbox[1] - previous.bbox[3] > 9
                    or line.bbox[0] - previous.bbox[0] > 12
                )
            )
            if starts_new and current:
                groups.append(current)
                current = []
            current.append(line)
            previous = line
        if current:
            groups.append(current)
    return groups


def _matched_terms(pattern: re.Pattern[str], text: str) -> list[str]:
    return sorted({match.group(0) for match in pattern.finditer(text)})


def build_clause_units(
    document: LayoutDocument, formula_line_ids: set[str]
) -> list[ProcedureClause | QualityClause]:
    output: list[ProcedureClause | QualityClause] = []
    for lines in _paragraph_groups(document):
        if all(line.line_id in formula_line_ids for line in lines):
            continue
        text = normalize_pdf_text("".join(line.text for line in lines))
        if QUALITY_TRIGGER_RE.search(text):
            triggers = _matched_terms(QUALITY_TRIGGER_RE, text)
            thresholds = re.findall(
                r"(?:不低于|不高于|不少于|不超过|≥|≤|>|<)?\s*"
                r"\d+(?:\.\d+)?\s*(?:%|％|mm|m|cm|次)?",
                text,
            )
            actions = sorted(
                {
                    term
                    for term in ("检查", "校核", "复核", "修正", "剔除", "补测")
                    if term in text
                }
            )
            output.append(
                QualityClause(
                    unit_id=stable_id(
                        document.standard_code,
                        "quality_clause",
                        *(line.line_id for line in lines),
                    ),
                    provenance=_provenance(document, lines),
                    clause_text=text,
                    trigger_terms=triggers,
                    threshold_mentions=[
                        item.strip() for item in thresholds if item.strip()
                    ],
                    action_mentions=actions,
                )
            )
        elif PROCEDURE_TRIGGER_RE.search(text):
            triggers = _matched_terms(PROCEDURE_TRIGGER_RE, text)
            temporal = re.findall(
                r"\d+\s*[—\-～~至]\s*\d+\s*月|"
                r"(?:一年|多年|每年|每月|每季度|生长季|基准年)",
                text,
            )
            frequency = re.findall(
                r"(?:一年|每年|每月|每季度)\s*\d*\s*次|一年一次",
                text,
            )
            instruments = re.findall(
                r"[^，。；]{0,12}(?:仪器|设备|传感器|样方)",
                text,
            )
            output.append(
                ProcedureClause(
                    unit_id=stable_id(
                        document.standard_code,
                        "procedure_clause",
                        *(line.line_id for line in lines),
                    ),
                    provenance=_provenance(document, lines),
                    clause_text=text,
                    trigger_terms=triggers,
                    temporal_mentions=sorted(set(temporal)),
                    frequency_mentions=sorted(set(frequency)),
                    instrument_mentions=sorted(set(instruments)),
                )
            )
    return output


def _table_clause_units(
    records: list[TableRecord],
) -> list[ProcedureClause | QualityClause]:
    output: list[ProcedureClause | QualityClause] = []
    for record in records:
        populated = {
            key: value for key, value in record.cells.items() if value
        }
        text = "；".join(
            f"{key}：{value}" for key, value in populated.items()
        )
        indicator_columns = [
            key
            for key in populated
            if "观测指标" in key or "观测项目" in key
        ]
        time_columns = [key for key in populated if "时间" in key]
        frequency_columns = [
            key
            for key in populated
            if "频度" in key or "频率" in key
        ]
        if indicator_columns and (time_columns or frequency_columns):
            output.append(
                ProcedureClause(
                    unit_id=stable_id(
                        record.unit_id, "procedure_clause"
                    ),
                    provenance=record.provenance,
                    clause_text=text,
                    trigger_terms=["观测表逻辑行"],
                    temporal_mentions=[
                        populated[key] for key in time_columns
                    ],
                    frequency_mentions=[
                        populated[key] for key in frequency_columns
                    ],
                    instrument_mentions=[],
                )
            )

        is_quality_table = bool(
            "数据质量" in record.table_title
            or "质量控制" in record.table_title
            or any("具体要求" in key for key in populated)
        )
        if not is_quality_table:
            continue
        thresholds = re.findall(
            r"(?:不低于|不高于|不少于|不超过|大于等于|小于等于|"
            r"≥|≤|>|<|小于|大于|优于)?\s*"
            r"\d+(?:\.\d+)?\s*(?:%|％|mm|m|km2|次|倍|个像元)?",
            text,
        )
        actions = sorted(
            {
                term
                for term in (
                    "检查",
                    "校核",
                    "复核",
                    "修正",
                    "剔除",
                    "补测",
                    "检验",
                )
                if term in text
            }
        )
        output.append(
            QualityClause(
                unit_id=stable_id(record.unit_id, "quality_clause"),
                provenance=record.provenance,
                clause_text=text,
                trigger_terms=["质量要求表逻辑行"],
                threshold_mentions=[
                    item.strip() for item in thresholds if item.strip()
                ],
                action_mentions=actions,
            )
        )
    return output


def build_source_units(document: LayoutDocument) -> list[SourceUnit]:
    formulas, formula_line_ids = build_formula_packages(document)
    tables = build_table_records(document)
    clauses = [
        *build_clause_units(document, formula_line_ids),
        *_table_clause_units(tables),
    ]
    units: list[SourceUnit] = [*formulas, *tables, *clauses]
    units.sort(
        key=lambda item: (
            item.provenance.pages[0] if item.provenance.pages else 0,
            item.provenance.evidence_spans[0].bbox[1]
            if item.provenance.evidence_spans
            else 0,
            item.unit_type,
        )
    )
    return units


def build_units_from_layout_dir(
    layout_dir: Path, output_dir: Path
) -> dict[str, Any]:
    manifest_path = layout_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        layout_files = [layout_dir / row["layout_file"] for row in manifest]
    else:
        layout_files = sorted(layout_dir.glob("*.layout.json"))
    if not layout_files:
        raise RuntimeError("no layout documents found")

    units: list[SourceUnit] = []
    for path in layout_files:
        units.extend(build_source_units(load_layout_document(path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "source_units.jsonl",
        (unit.to_dict() for unit in units),
    )
    counts = {
        unit_type: sum(unit.unit_type == unit_type for unit in units)
        for unit_type in (
            "formula_package",
            "table_record",
            "procedure_clause",
            "quality_clause",
        )
    }
    coordinate_count = sum(
        bool(span.bbox)
        for unit in units
        for span in unit.provenance.evidence_spans
    )
    span_count = sum(
        len(unit.provenance.evidence_spans) for unit in units
    )
    summary = {
        "schema_version": SOURCE_UNIT_SCHEMA_VERSION,
        "document_count": len(layout_files),
        "unit_count": len(units),
        "unit_counts": counts,
        "evidence_span_count": span_count,
        "coordinate_coverage": (
            round(coordinate_count / span_count, 6) if span_count else 0.0
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
