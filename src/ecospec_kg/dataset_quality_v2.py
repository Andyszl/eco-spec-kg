from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .annotation_v2 import (
    ASSESSMENT_SUBJECTS,
    EXTERNAL_TEST_CODES,
    _extract_sources,
    _formula_output_symbols,
    _split_compound_variable,
    _symbol_identity,
    _symbol_occurs,
    split_for_unit,
)
from .foundation_quality import evaluate_parser_quality
from .layout_parser import load_layout_document
from .ontology_v2 import validate_relation_v2
from .source_units import build_source_units

EXPECTED_CODES = [f"HJ {number}-2021" for number in range(1166, 1177)]
EXPECTED_PAGE_COUNT = 118
EXPECTED_FORMULA_COUNT = 65
EXPECTED_SOURCE_UNIT_COUNT = 882


def _all_evidence_spans(value: Any) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        if all(key in value for key in ("span_id", "page", "bbox")):
            spans[value["span_id"]] = value
        for child in value.values():
            spans.update(_all_evidence_spans(child))
    elif isinstance(value, list):
        for child in value:
            spans.update(_all_evidence_spans(child))
    return spans


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check(
    name: str,
    passed: bool,
    expected: Any,
    actual: Any,
    failures: list[str] | None = None,
) -> dict[str, Any]:
    item = {
        "check": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }
    if failures:
        item["failures"] = failures
    return item


def _valid_bbox(bbox: Any, width: float, height: float) -> bool:
    return bool(
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and 0 <= bbox[0] < bbox[2] <= width + 1
        and 0 <= bbox[1] < bbox[3] <= height + 1
    )


def run_representative_acceptance(
    layout_dir: Path,
    enriched_units_path: Path,
    base_config_path: Path,
    additions_config_path: Path,
) -> dict[str, Any]:
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    additions = json.loads(
        additions_config_path.read_text(encoding="utf-8")
    )
    config = {
        "schema_version": "parse-acceptance-v2.1",
        "cases": [*base.get("cases", []), *additions.get("cases", [])],
    }
    manifest = json.loads(
        (layout_dir / "manifest.json").read_text(encoding="utf-8")
    )
    documents = {
        item["standard_code"]: load_layout_document(
            layout_dir / item["layout_file"]
        )
        for item in manifest
    }
    units_by_standard = {
        code: build_source_units(document)
        for code, document in documents.items()
    }
    report = evaluate_parser_quality(
        documents,
        units_by_standard,
        config,
    )

    enriched_units = read_jsonl(enriched_units_path)
    manual_formulas = {
        (
            unit["provenance"]["standard_code"],
            page,
            formula["formula_number"],
        )
        for unit in enriched_units
        if unit.get("manual_transcription")
        for page in unit["provenance"].get("pages", [])
        for formula in unit.get("formulas", [])
    }
    case_config = {case["case_id"]: case for case in config["cases"]}
    for case in report["cases"]:
        expected_manual = case_config[case["case_id"]].get(
            "expected_manual_formula_numbers", []
        )
        if not expected_manual:
            continue
        actual_manual = sorted(
            number
            for code, page, number in manual_formulas
            if code == case["standard_code"] and page == case["pdf_page"]
        )
        check = _check(
            "manual_formula_override_verified",
            all(number in actual_manual for number in expected_manual),
            expected_manual,
            actual_manual,
        )
        case["checks"].append(check)
        case["passed"] = all(item["passed"] for item in case["checks"])

    checks = [item for case in report["cases"] for item in case["checks"]]
    report.update(
        {
            "passed": bool(report["cases"])
            and all(case["passed"] for case in report["cases"]),
            "case_count": len(report["cases"]),
            "passed_case_count": sum(
                case["passed"] for case in report["cases"]
            ),
            "check_count": len(checks),
            "passed_check_count": sum(item["passed"] for item in checks),
            "covered_standards": sorted(
                {case["standard_code"] for case in report["cases"]}
            ),
            "manual_expectation_basis": (
                "30 representative pages with manually specified headings, "
                "formula numbers, table counts, text fragments and row assertions"
            ),
        }
    )
    return report


def run_full_parser_invariants(
    layout_dir: Path,
    enriched_units_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(
        (layout_dir / "manifest.json").read_text(encoding="utf-8")
    )
    units = read_jsonl(enriched_units_path)
    checks: list[dict[str, Any]] = []

    codes = [item["standard_code"] for item in manifest]
    checks.append(
        _check(
            "canonical_11_standard_inventory",
            sorted(codes) == EXPECTED_CODES and len(set(codes)) == 11,
            EXPECTED_CODES,
            sorted(codes),
        )
    )
    source_hashes = [item["source_sha256"] for item in manifest]
    checks.append(
        _check(
            "canonical_source_hashes_unique",
            len(source_hashes) == len(set(source_hashes)) == 11,
            11,
            len(set(source_hashes)),
        )
    )

    documents: dict[str, dict[str, Any]] = {}
    page_by_code: dict[str, dict[int, dict[str, Any]]] = {}
    line_ids: dict[str, dict[int, set[str]]] = defaultdict(dict)
    layout_failures: dict[str, list[str]] = defaultdict(list)
    total_pages = 0
    total_lines = 0
    total_tables = 0
    private_characters: set[str] = set()

    for item in manifest:
        code = item["standard_code"]
        document = json.loads(
            (layout_dir / item["layout_file"]).read_text(encoding="utf-8")
        )
        documents[code] = document
        pages = document.get("pages", [])
        total_pages += len(pages)
        total_lines += sum(len(page.get("lines", [])) for page in pages)
        total_tables += sum(len(page.get("tables", [])) for page in pages)
        expected_pages = list(range(1, int(item["page_count"]) + 1))
        actual_pages = [int(page["page"]) for page in pages]
        if actual_pages != expected_pages:
            layout_failures["page_sequence"].append(code)
        page_by_code[code] = {
            int(page["page"]): page for page in pages
        }
        document_line_ids: set[str] = set()
        for page in pages:
            page_no = int(page["page"])
            width = float(page["width"])
            height = float(page["height"])
            lines = page.get("lines", [])
            orders = [int(line["reading_order"]) for line in lines]
            if orders != sorted(orders) or len(orders) != len(set(orders)):
                layout_failures["reading_order"].append(
                    f"{code}:p{page_no}"
                )
            ids = {line["line_id"] for line in lines}
            if len(ids) != len(lines) or document_line_ids & ids:
                layout_failures["line_id_uniqueness"].append(
                    f"{code}:p{page_no}"
                )
            document_line_ids.update(ids)
            line_ids[code][page_no] = ids
            for line in lines:
                if not _valid_bbox(line.get("bbox"), width, height):
                    layout_failures["line_bbox"].append(
                        f"{code}:p{page_no}:{line.get('line_id')}"
                    )
                private_characters.update(
                    character
                    for character in line.get("text", "")
                    if 0xE000 <= ord(character) <= 0xF8FF
                )
            for table in page.get("tables", []):
                if not _valid_bbox(table.get("bbox"), width, height):
                    layout_failures["table_bbox"].append(
                        f"{code}:p{page_no}:{table.get('table_id')}"
                    )
                rows = table.get("rows", [])
                boxes = table.get("cell_bboxes", [])
                if len(rows) != len(boxes):
                    layout_failures["table_shape"].append(
                        f"{code}:p{page_no}:{table.get('table_id')}"
                    )
                for row_index, row_boxes in enumerate(boxes):
                    if row_index < len(rows) and len(row_boxes) != len(
                        rows[row_index]
                    ):
                        layout_failures["table_shape"].append(
                            f"{code}:p{page_no}:{table.get('table_id')}:r{row_index}"
                        )
                    for bbox in row_boxes:
                        if bbox is not None and not _valid_bbox(
                            bbox, width, height
                        ):
                            layout_failures["table_cell_bbox"].append(
                                f"{code}:p{page_no}:{table.get('table_id')}:r{row_index}"
                            )

        if len(pages) != int(item["pages"]):
            layout_failures["manifest_page_count"].append(code)
        if sum(len(page.get("lines", [])) for page in pages) != int(
            item["lines"]
        ):
            layout_failures["manifest_line_count"].append(code)
        if sum(len(page.get("tables", [])) for page in pages) != int(
            item["tables"]
        ):
            layout_failures["manifest_table_count"].append(code)

    checks.extend(
        [
            _check(
                "all_118_pages_present",
                total_pages == EXPECTED_PAGE_COUNT
                and not layout_failures["page_sequence"],
                EXPECTED_PAGE_COUNT,
                total_pages,
                layout_failures["page_sequence"],
            ),
            _check(
                "layout_manifest_counts_match",
                not any(
                    layout_failures[name]
                    for name in (
                        "manifest_page_count",
                        "manifest_line_count",
                        "manifest_table_count",
                    )
                ),
                "all per-document counts match",
                {
                    "pages": total_pages,
                    "lines": total_lines,
                    "tables": total_tables,
                },
                [
                    failure
                    for name in (
                        "manifest_page_count",
                        "manifest_line_count",
                        "manifest_table_count",
                    )
                    for failure in layout_failures[name]
                ],
            ),
            _check(
                "all_page_reading_orders_stable",
                not layout_failures["reading_order"],
                "strictly increasing and unique",
                len(layout_failures["reading_order"]),
                layout_failures["reading_order"],
            ),
            _check(
                "all_layout_coordinates_valid",
                not any(
                    layout_failures[name]
                    for name in (
                        "line_bbox",
                        "table_bbox",
                        "table_cell_bbox",
                        "table_shape",
                    )
                ),
                "all line/table/cell boxes within page",
                sum(
                    len(layout_failures[name])
                    for name in (
                        "line_bbox",
                        "table_bbox",
                        "table_cell_bbox",
                        "table_shape",
                    )
                ),
                [
                    failure
                    for name in (
                        "line_bbox",
                        "table_bbox",
                        "table_cell_bbox",
                        "table_shape",
                    )
                    for failure in layout_failures[name]
                ][:100],
            ),
            _check(
                "line_ids_unique_per_document",
                not layout_failures["line_id_uniqueness"],
                0,
                len(layout_failures["line_id_uniqueness"]),
                layout_failures["line_id_uniqueness"],
            ),
            _check(
                "symbol_font_private_use_removed",
                not private_characters,
                [],
                [f"U+{ord(value):04X}" for value in sorted(private_characters)],
            ),
        ]
    )

    unit_failures: dict[str, list[str]] = defaultdict(list)
    unit_ids: set[str] = set()
    formula_keys: list[tuple[str, str, str]] = []
    manual_formula_count = 0
    standard_unit_counts: Counter[str] = Counter()
    for unit in units:
        unit_id = unit["unit_id"]
        code = unit["provenance"]["standard_code"]
        standard_unit_counts[code] += 1
        if unit_id in unit_ids:
            unit_failures["duplicate_unit_id"].append(unit_id)
        unit_ids.add(unit_id)
        if code not in page_by_code:
            unit_failures["unknown_standard"].append(unit_id)
            continue
        spans = list(_all_evidence_spans(unit).values())
        if not spans:
            unit_failures["missing_evidence"].append(unit_id)
        for span in spans:
            page_no = int(span["page"])
            page = page_by_code[code].get(page_no)
            if page is None:
                unit_failures["unknown_page"].append(
                    f"{unit_id}:p{page_no}"
                )
                continue
            if not _valid_bbox(
                span.get("bbox"), float(page["width"]), float(page["height"])
            ):
                unit_failures["evidence_bbox"].append(
                    f"{unit_id}:{span.get('span_id')}"
                )
            unknown_lines = set(span.get("line_ids", [])) - line_ids[code][
                page_no
            ]
            if unknown_lines:
                unit_failures["evidence_line_ids"].append(
                    f"{unit_id}:{span.get('span_id')}:{sorted(unknown_lines)}"
                )
        if unit["unit_type"] == "table_record":
            cells = unit.get("cells", {})
            boxes = unit.get("cell_bboxes", {})
            for column, value in cells.items():
                if value is not None and column not in boxes:
                    unit_failures["table_record_bbox"].append(
                        f"{unit_id}:{column}"
                    )
            for column, source in unit.get("inherited_from", {}).items():
                if not all(
                    key in source
                    for key in ("table_id", "page", "row_index", "bbox")
                ):
                    unit_failures["inheritance_provenance"].append(
                        f"{unit_id}:{column}"
                    )
        if unit["unit_type"] == "formula_package":
            if unit.get("manual_transcription"):
                manual_formula_count += len(unit.get("formulas", []))
            for formula in unit.get("formulas", []):
                formula_keys.append(
                    (
                        code,
                        unit["provenance"].get("section", ""),
                        formula["formula_number"],
                    )
                )

    checks.extend(
        [
            _check(
                "source_unit_count_and_ids",
                len(units) == EXPECTED_SOURCE_UNIT_COUNT
                and not unit_failures["duplicate_unit_id"],
                EXPECTED_SOURCE_UNIT_COUNT,
                len(units),
                unit_failures["duplicate_unit_id"],
            ),
            _check(
                "every_standard_has_source_units",
                set(standard_unit_counts) == set(EXPECTED_CODES)
                and all(standard_unit_counts[code] > 0 for code in EXPECTED_CODES),
                EXPECTED_CODES,
                dict(sorted(standard_unit_counts.items())),
            ),
            _check(
                "all_source_unit_evidence_traceable",
                not any(
                    unit_failures[name]
                    for name in (
                        "missing_evidence",
                        "unknown_standard",
                        "unknown_page",
                        "evidence_bbox",
                        "evidence_line_ids",
                    )
                ),
                "all spans resolve to source page, bbox and line IDs",
                sum(
                    len(unit_failures[name])
                    for name in (
                        "missing_evidence",
                        "unknown_standard",
                        "unknown_page",
                        "evidence_bbox",
                        "evidence_line_ids",
                    )
                ),
                [
                    failure
                    for name in (
                        "missing_evidence",
                        "unknown_standard",
                        "unknown_page",
                        "evidence_bbox",
                        "evidence_line_ids",
                    )
                    for failure in unit_failures[name]
                ][:100],
            ),
            _check(
                "all_table_records_structurally_traceable",
                not unit_failures["table_record_bbox"]
                and not unit_failures["inheritance_provenance"],
                0,
                len(unit_failures["table_record_bbox"])
                + len(unit_failures["inheritance_provenance"]),
                [
                    *unit_failures["table_record_bbox"],
                    *unit_failures["inheritance_provenance"],
                ][:100],
            ),
            _check(
                "formula_inventory_unique_by_section",
                len(formula_keys) == EXPECTED_FORMULA_COUNT
                and len(set(formula_keys)) == EXPECTED_FORMULA_COUNT,
                EXPECTED_FORMULA_COUNT,
                {
                    "formulas": len(formula_keys),
                    "unique_formula_contexts": len(set(formula_keys)),
                },
            ),
            _check(
                "manual_formula_transcriptions_traceable",
                manual_formula_count == 4,
                4,
                manual_formula_count,
            ),
        ]
    )

    formula_numbers = {
        (code, number) for code, _section, number in formula_keys
    }
    unresolved_refs: list[str] = []
    reference_count = 0
    for code, document in documents.items():
        for page in document["pages"]:
            text = "\n".join(line["text"] for line in page["lines"])
            references = re.findall(
                r"按式\s*[（(]([A-Z]?\.?\d+)[）)]",
                text,
            )
            reference_count += len(references)
            for number in references:
                if (code, number) not in formula_numbers:
                    unresolved_refs.append(
                        f"{code}:p{page['page']}:formula:{number}"
                    )
    checks.append(
        _check(
            "explicit_formula_references_resolved",
            reference_count > 0 and not unresolved_refs,
            "all explicit 按式 references resolve",
            {
                "reference_count": reference_count,
                "unresolved_count": len(unresolved_refs),
            },
            unresolved_refs,
        )
    )

    return {
        "schema_version": "parser-invariants-v2.1",
        "passed": all(item["passed"] for item in checks),
        "scope": {
            "standards": len(manifest),
            "pages": total_pages,
            "lines": total_lines,
            "tables": total_tables,
            "source_units": len(units),
            "formulas": len(formula_keys),
        },
        "check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "checks": checks,
    }


def validate_gold_paths(
    source_units_path: Path,
    annotations_path: Path,
    entities_path: Path,
    triples_path: Path,
) -> dict[str, Any]:
    units = read_jsonl(source_units_path)
    annotations = {
        item["unit_id"]: item for item in read_jsonl(annotations_path)
    }
    entities = read_jsonl(entities_path)
    triples = read_jsonl(triples_path)
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    entity_by_id = {entity["entity_id"]: entity for entity in entities}
    entity_by_key = {
        (
            entity["standard_code"],
            entity["entity_type"],
            entity["context"],
            _symbol_identity(entity["name"]),
        ): entity
        for entity in entities
    }
    relation_keys = {
        (
            triple["head_id"],
            triple["relation_type"],
            triple["tail_id"],
        )
        for triple in triples
    }
    outgoing: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        outgoing[(triple["head_id"], triple["relation_type"])].append(triple)
        incoming[(triple["tail_id"], triple["relation_type"])].append(triple)

    schema_failures: list[str] = []
    evidence_failures: list[str] = []
    for triple in triples:
        valid, reason = validate_relation_v2(
            triple["head_type"],
            triple["relation_type"],
            triple["tail_type"],
        )
        if not valid:
            schema_failures.append(f"{triple['triple_id']}:{reason}")
        if triple["head_id"] not in entity_by_id or triple[
            "tail_id"
        ] not in entity_by_id:
            schema_failures.append(
                f"{triple['triple_id']}:missing_entity_endpoint"
            )
        if not triple.get("evidence"):
            evidence_failures.append(
                f"{triple['triple_id']}:missing_evidence"
            )
        for evidence in triple.get("evidence", []):
            unit = unit_by_id.get(evidence["source_unit_id"])
            if unit is None:
                evidence_failures.append(
                    f"{triple['triple_id']}:unknown_unit"
                )
                continue
            valid_spans = set(_all_evidence_spans(unit))
            if not set(evidence.get("evidence_span_ids", [])) <= valid_spans:
                evidence_failures.append(
                    f"{triple['triple_id']}:unknown_span"
                )
            if evidence.get("evidence_span_ids") and not evidence.get(
                "bboxes"
            ):
                evidence_failures.append(
                    f"{triple['triple_id']}:missing_evidence_bbox"
                )

    formula_rows: list[dict[str, Any]] = []
    for unit in units:
        if unit["unit_type"] != "formula_package":
            continue
        code = unit["provenance"]["standard_code"]
        section = unit["provenance"].get("section", "")
        variables = [
            expanded
            for variable in unit.get("variable_definitions", [])
            for expanded in _split_compound_variable(variable)
        ]
        role_overrides = unit.get("manual_variable_roles", {})
        package_text = " ".join(
            [
                unit.get("introduction", ""),
                unit.get("interstitial_text", ""),
                unit.get("adjacent_source_text", ""),
                " ".join(
                    variable.get("definition", "") for variable in variables
                ),
            ]
        )
        explicit_sources = _extract_sources(package_text)
        annotation = annotations[unit["unit_id"]]
        local_source_names = {
            relation["tail_name"]
            for relation in annotation["relations"]
            if relation["relation_type"] == "sourced_from"
        }
        for formula in unit.get("formulas", []):
            number = formula["formula_number"]
            context = f"{section}|{number}"
            formula_entity = next(
                (
                    entity
                    for entity in entities
                    if entity["standard_code"] == code
                    and entity["entity_type"] == "formula"
                    and entity["context"] == context
                    and unit["unit_id"] in entity["source_unit_ids"]
                ),
                None,
            )
            if formula_entity is None:
                formula_rows.append(
                    {
                        "standard_code": code,
                        "section": section,
                        "formula_number": number,
                        "passed": False,
                        "failure": "formula_entity_missing",
                    }
                )
                continue
            formula_id = formula_entity["entity_id"]
            input_relations = outgoing[(formula_id, "has_input")]
            output_relations = outgoing[(formula_id, "has_output")]
            expression = formula.get("expression_text", "")
            outputs = _formula_output_symbols(expression, variables)
            output_applicable = bool("=" in expression or outputs)

            unit_checks: list[dict[str, Any]] = []
            explicit_unit_variables = []
            for variable in variables:
                symbol = variable["symbol"]
                if not variable.get("unit"):
                    continue
                if not (
                    _symbol_occurs(symbol, expression)
                    or symbol in outputs
                    or role_overrides.get(symbol)
                ):
                    continue
                explicit_unit_variables.append(symbol)
                variable_entity = entity_by_key.get(
                    (
                        code,
                        "model_variable",
                        context,
                        _symbol_identity(symbol),
                    )
                )
                unit_ok = bool(
                    variable_entity
                    and outgoing[
                        (variable_entity["entity_id"], "has_unit")
                    ]
                )
                unit_checks.append(
                    {
                        "symbol": symbol,
                        "unit": variable["unit"],
                        "passed": unit_ok,
                    }
                )

            indicator_applicable = code in ASSESSMENT_SUBJECTS
            indicator_ok = bool(
                incoming[(formula_id, "calculated_by")]
            ) if indicator_applicable else True
            source_ok = all(
                source in local_source_names for source in explicit_sources
            )
            row = {
                "standard_code": code,
                "section": section,
                "formula_number": number,
                "formula_entity_id": formula_id,
                "manual_transcription": bool(
                    unit.get("manual_transcription")
                ),
                "input_count": len(input_relations),
                "input_passed": bool(input_relations),
                "output_applicable": output_applicable,
                "output_count": len(output_relations),
                "output_passed": bool(output_relations)
                if output_applicable
                else True,
                "output_not_applicable_reason": (
                    ""
                    if output_applicable
                    else "objective_function_has_no_named_left_hand_output"
                ),
                "explicit_unit_variable_count": len(
                    explicit_unit_variables
                ),
                "unit_checks": unit_checks,
                "unit_passed": all(
                    item["passed"] for item in unit_checks
                ),
                "explicit_sources": explicit_sources,
                "source_passed": source_ok,
                "indicator_path_applicable": indicator_applicable,
                "indicator_path_passed": indicator_ok,
            }
            row["passed"] = all(
                [
                    row["input_passed"],
                    row["output_passed"],
                    row["unit_passed"],
                    row["source_passed"],
                    row["indicator_path_passed"],
                ]
            )
            formula_rows.append(row)

    quality_rows = []
    for unit in units:
        if (
            unit["unit_type"] != "table_record"
            or unit["provenance"]["standard_code"] != "HJ 1176-2021"
            or not unit.get("cells", {}).get("具体要求")
        ):
            continue
        relations = annotations[unit["unit_id"]]["relations"]
        passed = any(
            relation["relation_type"] == "constrained_by"
            and relation["head_type"] == "data_source"
            and relation["tail_type"] == "quality_rule"
            for relation in relations
        )
        quality_rows.append(
            {
                "unit_id": unit["unit_id"],
                "page": unit["provenance"]["pages"][0],
                "secondary_indicator": unit["cells"].get("二级指标", ""),
                "passed": passed,
            }
        )

    checks = [
        _check(
            "all_triples_schema_valid",
            not schema_failures,
            0,
            len(schema_failures),
            schema_failures[:100],
        ),
        _check(
            "all_triples_have_traceable_evidence",
            not evidence_failures,
            0,
            len(evidence_failures),
            evidence_failures[:100],
        ),
        _check(
            "all_formula_inputs_validated",
            len(formula_rows) == EXPECTED_FORMULA_COUNT
            and all(row.get("input_passed") for row in formula_rows),
            EXPECTED_FORMULA_COUNT,
            sum(bool(row.get("input_passed")) for row in formula_rows),
        ),
        _check(
            "all_applicable_formula_outputs_validated",
            all(row.get("output_passed") for row in formula_rows),
            "every formula with a named left-hand output",
            {
                "applicable": sum(
                    bool(row.get("output_applicable")) for row in formula_rows
                ),
                "passed": sum(
                    bool(row.get("output_passed"))
                    for row in formula_rows
                    if row.get("output_applicable")
                ),
                "not_applicable": sum(
                    not bool(row.get("output_applicable"))
                    for row in formula_rows
                ),
            },
        ),
        _check(
            "all_explicit_formula_units_validated",
            all(row.get("unit_passed") for row in formula_rows),
            "all explicit units linked with has_unit",
            sum(
                len(row.get("unit_checks", [])) for row in formula_rows
            ),
        ),
        _check(
            "all_explicit_formula_sources_validated",
            all(row.get("source_passed") for row in formula_rows),
            "all explicit sources linked with sourced_from",
            sum(
                len(row.get("explicit_sources", []))
                for row in formula_rows
            ),
        ),
        _check(
            "all_assessment_formula_indicator_paths_validated",
            all(row.get("indicator_path_passed") for row in formula_rows),
            "assessment_subject -> indicator -> formula",
            {
                "applicable": sum(
                    bool(row.get("indicator_path_applicable"))
                    for row in formula_rows
                ),
                "passed": sum(
                    bool(row.get("indicator_path_passed"))
                    for row in formula_rows
                    if row.get("indicator_path_applicable")
                ),
            },
        ),
        _check(
            "all_quality_table_constraints_validated",
            bool(quality_rows)
            and all(row["passed"] for row in quality_rows),
            "every HJ 1176 quality requirement has constrained_by",
            {
                "applicable": len(quality_rows),
                "passed": sum(row["passed"] for row in quality_rows),
            },
        ),
    ]
    return {
        "schema_version": "gold-path-validation-v2.1",
        "passed": all(item["passed"] for item in checks)
        and all(row.get("passed") for row in formula_rows),
        "gold_nature": "ai_expert_pre_gold",
        "human_expert_review_required_for_publication": True,
        "entity_count": len(entities),
        "triple_count": len(triples),
        "formula_count": len(formula_rows),
        "quality_constraint_count": len(quality_rows),
        "check_count": len(checks),
        "passed_check_count": sum(item["passed"] for item in checks),
        "checks": checks,
        "formula_paths": formula_rows,
        "quality_paths": quality_rows,
    }


def validate_splits(
    source_units_path: Path,
    split_dir: Path,
) -> dict[str, Any]:
    units = read_jsonl(source_units_path)
    unit_splits = {unit["unit_id"]: split_for_unit(unit) for unit in units}
    split_rows = {
        split: read_jsonl(split_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    triple_ids = {
        split: {row["triple_id"] for row in rows}
        for split, rows in split_rows.items()
    }
    overlap = (
        triple_ids["train"] & triple_ids["dev"]
        | triple_ids["train"] & triple_ids["test"]
        | triple_ids["dev"] & triple_ids["test"]
    )
    test_unit_codes = {
        unit["provenance"]["standard_code"]
        for unit in units
        if unit_splits[unit["unit_id"]] == "test"
    }
    non_test_external = [
        unit["unit_id"]
        for unit in units
        if unit["provenance"]["standard_code"] in EXTERNAL_TEST_CODES
        and unit_splits[unit["unit_id"]] != "test"
    ]
    invalid_test_codes = test_unit_codes - EXTERNAL_TEST_CODES

    groups: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        code = unit["provenance"]["standard_code"]
        if code not in {"HJ 1172-2021", "HJ 1173-2021"}:
            continue
        if unit["unit_type"] == "table_record":
            group = f"{code}|table|{unit.get('table_id', '')}"
        else:
            group = (
                f"{code}|section|"
                f"{unit['provenance'].get('section', '')}"
            )
        groups[group].add(unit_splits[unit["unit_id"]])
    split_groups = {
        group: sorted(values)
        for group, values in groups.items()
        if len(values) > 1
    }
    unit_counts = Counter(unit_splits.values())
    checks = [
        _check(
            "triple_splits_disjoint",
            not overlap,
            0,
            len(overlap),
            sorted(overlap)[:100],
        ),
        _check(
            "external_standard_test_policy",
            not non_test_external and not invalid_test_codes,
            sorted(EXTERNAL_TEST_CODES),
            sorted(test_unit_codes),
            [
                *non_test_external[:100],
                *sorted(invalid_test_codes),
            ],
        ),
        _check(
            "core_document_groups_are_atomic",
            not split_groups,
            0,
            len(split_groups),
            [f"{group}:{values}" for group, values in split_groups.items()],
        ),
        _check(
            "all_three_splits_nonempty",
            all(unit_counts[split] > 0 for split in ("train", "dev", "test"))
            and all(split_rows[split] for split in ("train", "dev", "test")),
            "non-empty train/dev/test",
            {
                "unit_counts": dict(sorted(unit_counts.items())),
                "triple_counts": {
                    split: len(rows) for split, rows in split_rows.items()
                },
            },
        ),
    ]
    return {
        "schema_version": "split-validation-v2.1",
        "passed": all(item["passed"] for item in checks),
        "split_policy": {
            "test_standards": sorted(EXTERNAL_TEST_CODES),
            "dev_policy": (
                "HJ 1172 and HJ 1173 grouped by table or section; "
                "deterministic 20 percent bucket"
            ),
            "train_policy": "remaining standards and core groups",
        },
        "unit_split_counts": dict(sorted(unit_counts.items())),
        "triple_split_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "checks": checks,
    }
