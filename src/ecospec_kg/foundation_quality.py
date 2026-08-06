from __future__ import annotations

from pathlib import Path
from typing import Any

from .corpus import discover_corpus
from .io_utils import read_json, write_json, write_jsonl
from .layout_parser import (
    LayoutDocument,
    LayoutPage,
    layout_filename,
    parse_pdf_layout,
    save_layout_document,
)
from .ontology_v2 import schema_quality_report
from .source_units import (
    FormulaPackage,
    SourceUnit,
    TableRecord,
    build_source_units,
)

PILOT_SCHEMA_VERSION = "parse-pilot-v2.0"


def default_pilot_path() -> Path:
    return Path(__file__).parents[2] / "config" / "parse_pilot_v2.json"


def _check(name: str, passed: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }


def _valid_bbox(bbox: list[float], page: LayoutPage) -> bool:
    return bool(
        len(bbox) == 4
        and 0 <= bbox[0] < bbox[2] <= page.width + 1
        and 0 <= bbox[1] < bbox[3] <= page.height + 1
    )


def _page_formula_numbers(page: LayoutPage) -> list[str]:
    from .layout_parser import FORMULA_NUMBER_RE

    numbers: list[str] = []
    for line in page.lines:
        if line.kind != "formula":
            continue
        match = FORMULA_NUMBER_RE.search(line.text)
        if match:
            numbers.append(match.group("number"))
    return numbers


def _units_on_page(units: list[SourceUnit], page_no: int) -> list[SourceUnit]:
    return [unit for unit in units if page_no in unit.provenance.pages]


def _formula_packages_on_page(
    units: list[SourceUnit], page_no: int
) -> list[FormulaPackage]:
    return [
        unit
        for unit in units
        if isinstance(unit, FormulaPackage)
        and any(
            formula.evidence_span.page == page_no
            for formula in unit.formulas
        )
    ]


def _table_records_on_page(
    units: list[SourceUnit], page_no: int
) -> list[TableRecord]:
    return [
        unit
        for unit in units
        if isinstance(unit, TableRecord)
        and unit.provenance.pages == [page_no]
    ]


def _table_assertion_result(
    records: list[TableRecord], assertion: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    matches = [
        record
        for record in records
        if all(
            record.cells.get(column) == value
            for column, value in assertion.get("match", {}).items()
        )
    ]
    if not matches:
        return False, {"matched_records": 0}
    for record in matches:
        exact_ok = all(
            record.cells.get(column) == value
            for column, value in assertion.get("expect", {}).items()
        )
        contains_ok = all(
            value in str(record.cells.get(column) or "")
            for column, value in assertion.get("expect_contains", {}).items()
        )
        if exact_ok and contains_ok:
            return True, {
                "matched_records": len(matches),
                "record": record.cells,
                "inherited_columns": record.inherited_columns,
            }
    return False, {
        "matched_records": len(matches),
        "records": [record.cells for record in matches],
    }


def _evaluate_case(
    document: LayoutDocument,
    units: list[SourceUnit],
    case: dict[str, Any],
) -> dict[str, Any]:
    page_no = int(case["pdf_page"])
    page = next((item for item in document.pages if item.page == page_no), None)
    if page is None:
        return {
            "case_id": case["case_id"],
            "standard_code": document.standard_code,
            "pdf_page": page_no,
            "passed": False,
            "checks": [
                _check("page_exists", False, page_no, "missing")
            ],
        }

    checks: list[dict[str, Any]] = []
    headings = [
        line.section for line in page.lines if line.kind == "heading"
    ]
    expected_headings = list(case.get("expected_heading_sections", []))
    checks.append(
        _check(
            "heading_sections_exact",
            headings == expected_headings,
            expected_headings,
            headings,
        )
    )

    formulas = _page_formula_numbers(page)
    expected_formulas = list(case.get("expected_formula_numbers", []))
    checks.append(
        _check(
            "formula_numbers_exact",
            formulas == expected_formulas,
            expected_formulas,
            formulas,
        )
    )
    checks.append(
        _check(
            "table_count_exact",
            len(page.tables) == int(case.get("expected_table_count", 0)),
            int(case.get("expected_table_count", 0)),
            len(page.tables),
        )
    )

    text = "\n".join(line.text for line in page.lines)
    missing_fragments = [
        fragment
        for fragment in case.get("text_fragments", [])
        if fragment not in text
    ]
    checks.append(
        _check(
            "manual_text_fragments",
            not missing_fragments,
            case.get("text_fragments", []),
            {"missing": missing_fragments},
        )
    )

    private_use = sorted(
        {
            character
            for character in text
            if 0xE000 <= ord(character) <= 0xF8FF
        }
    )
    checks.append(
        _check(
            "symbol_font_normalized",
            not private_use,
            [],
            [f"U+{ord(character):04X}" for character in private_use],
        )
    )

    orders = [line.reading_order for line in page.lines]
    checks.append(
        _check(
            "reading_order_stable",
            orders == sorted(orders) and len(orders) == len(set(orders)),
            "strictly increasing unique order",
            orders,
        )
    )

    invalid_line_boxes = [
        line.line_id for line in page.lines if not _valid_bbox(line.bbox, page)
    ]
    checks.append(
        _check(
            "line_coordinates_valid",
            not invalid_line_boxes,
            [],
            invalid_line_boxes,
        )
    )

    page_units = _units_on_page(units, page_no)
    invalid_span_ids = [
        span.span_id
        for unit in page_units
        for span in unit.provenance.evidence_spans
        if span.page == page_no and not _valid_bbox(span.bbox, page)
    ]
    checks.append(
        _check(
            "source_unit_coordinates_valid",
            not invalid_span_ids,
            [],
            invalid_span_ids,
        )
    )

    packages = _formula_packages_on_page(units, page_no)
    packaged_formula_numbers = [
        formula.formula_number
        for package in packages
        for formula in package.formulas
        if formula.evidence_span.page == page_no
    ]
    checks.append(
        _check(
            "formula_anchors_packaged",
            packaged_formula_numbers == expected_formulas,
            expected_formulas,
            packaged_formula_numbers,
        )
    )

    required_types = list(case.get("required_unit_types", []))
    present_types = sorted({unit.unit_type for unit in page_units})
    checks.append(
        _check(
            "required_source_unit_types",
            all(unit_type in present_types for unit_type in required_types),
            required_types,
            present_types,
        )
    )

    title_fragment = str(case.get("table_title_contains", ""))
    if title_fragment:
        table_titles = [table.title for table in page.tables]
        checks.append(
            _check(
                "table_title_recovered",
                any(title_fragment in title for title in table_titles),
                title_fragment,
                table_titles,
            )
        )

    if case.get("require_continued_table"):
        continuation_ids = [
            table.continued_from_table_id
            for table in page.tables
            if table.continued_from_table_id
        ]
        checks.append(
            _check(
                "cross_page_table_link",
                bool(continuation_ids),
                "non-empty continued_from_table_id",
                continuation_ids,
            )
        )

    records = _table_records_on_page(units, page_no)
    header_like_records = [
        record.unit_id
        for record in records
        if record.cells
        and all(
            record.cells.get(column) == column
            for column in record.cells
        )
    ]
    checks.append(
        _check(
            "table_headers_not_emitted_as_records",
            not header_like_records,
            [],
            header_like_records,
        )
    )

    pages_by_number = {item.page: item for item in document.pages}
    invalid_cell_boxes: list[str] = []
    invalid_inheritance: list[str] = []
    for record in records:
        for column, value in record.cells.items():
            if value is None:
                continue
            source = record.inherited_from.get(column, {})
            source_page_no = int(source.get("page", page_no))
            source_page = pages_by_number.get(source_page_no)
            bbox = record.cell_bboxes.get(column)
            if source_page is None or bbox is None or not _valid_bbox(
                bbox, source_page
            ):
                invalid_cell_boxes.append(f"{record.unit_id}:{column}")
        for column, source in record.inherited_from.items():
            if not all(
                key in source
                for key in ("table_id", "page", "row_index", "bbox")
            ):
                invalid_inheritance.append(f"{record.unit_id}:{column}")
    checks.append(
        _check(
            "table_cell_coordinates_valid",
            not invalid_cell_boxes,
            [],
            invalid_cell_boxes,
        )
    )
    checks.append(
        _check(
            "merged_cell_sources_recorded",
            not invalid_inheritance,
            [],
            invalid_inheritance,
        )
    )

    for index, assertion in enumerate(
        case.get("table_assertions", []), start=1
    ):
        passed, actual = _table_assertion_result(records, assertion)
        checks.append(
            _check(
                f"table_row_alignment_{index}",
                passed,
                assertion,
                actual,
            )
        )

    for index, assertion in enumerate(
        case.get("formula_variable_assertions", []), start=1
    ):
        number = assertion["formula_number"]
        relevant = [
            package
            for package in packages
            if any(
                formula.formula_number == number
                for formula in package.formulas
            )
        ]
        actual_symbols = sorted(
            {
                variable.symbol
                for package in relevant
                for variable in package.variable_definitions
            }
        )
        required_symbols = list(assertion.get("required_symbols", []))
        checks.append(
            _check(
                f"formula_variables_{index}",
                all(symbol in actual_symbols for symbol in required_symbols),
                required_symbols,
                actual_symbols,
            )
        )

    for index, assertion in enumerate(
        case.get("formula_expression_assertions", []), start=1
    ):
        number = assertion["formula_number"]
        expressions = [
            formula.expression_text
            for package in packages
            for formula in package.formulas
            if formula.formula_number == number
            and formula.evidence_span.page == page_no
        ]
        actual_text = "\n".join(expressions)
        required_fragments = list(
            assertion.get("required_fragments", [])
        )
        missing = [
            fragment
            for fragment in required_fragments
            if fragment not in actual_text
        ]
        checks.append(
            _check(
                f"formula_expression_symbols_{index}",
                bool(expressions) and not missing,
                required_fragments,
                {"missing": missing, "expression": actual_text},
            )
        )

    return {
        "case_id": case["case_id"],
        "standard_code": document.standard_code,
        "pdf_page": page_no,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def evaluate_parser_quality(
    documents: dict[str, LayoutDocument],
    units_by_standard: dict[str, list[SourceUnit]],
    pilot_config: dict[str, Any],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in pilot_config.get("cases", []):
        standard_code = str(case["standard_code"])
        document = documents.get(standard_code)
        if document is None:
            cases.append(
                {
                    "case_id": case["case_id"],
                    "standard_code": standard_code,
                    "pdf_page": int(case["pdf_page"]),
                    "passed": False,
                    "checks": [
                        _check(
                            "document_exists",
                            False,
                            standard_code,
                            "missing",
                        )
                    ],
                }
            )
            continue
        cases.append(
            _evaluate_case(
                document,
                units_by_standard.get(standard_code, []),
                case,
            )
        )

    checks = [check for case in cases for check in case["checks"]]
    return {
        "pilot_schema_version": pilot_config.get(
            "schema_version", PILOT_SCHEMA_VERSION
        ),
        "passed": bool(cases) and all(case["passed"] for case in cases),
        "case_count": len(cases),
        "passed_case_count": sum(case["passed"] for case in cases),
        "check_count": len(checks),
        "passed_check_count": sum(check["passed"] for check in checks),
        "cases": cases,
    }


def run_foundation_gate(
    source_dir: Path,
    output_dir: Path,
    pilot_path: Path | None = None,
) -> dict[str, Any]:
    pilot_config = read_json(pilot_path or default_pilot_path())
    standards = {
        str(case["standard_code"]) for case in pilot_config.get("cases", [])
    }
    corpus = {
        item.standard_code: item
        for item in discover_corpus(source_dir)
        if item.standard_code in standards
    }

    documents: dict[str, LayoutDocument] = {}
    units_by_standard: dict[str, list[SourceUnit]] = {}
    layout_dir = output_dir / "layouts"
    layout_manifest: list[dict[str, Any]] = []
    all_units: list[SourceUnit] = []
    for standard_code in sorted(standards):
        item = corpus.get(standard_code)
        if item is None:
            continue
        document = parse_pdf_layout(
            item.path,
            item.standard_code,
            item.title,
            item.sha256,
        )
        documents[standard_code] = document
        filename = layout_filename(standard_code)
        save_layout_document(document, layout_dir / filename)
        units = build_source_units(document)
        units_by_standard[standard_code] = units
        all_units.extend(units)
        layout_manifest.append(
            {
                **item.to_dict(),
                "layout_file": filename,
                "source_unit_count": len(units),
            }
        )

    write_json(layout_dir / "manifest.json", layout_manifest)
    write_jsonl(
        output_dir / "source_units.jsonl",
        (unit.to_dict() for unit in all_units),
    )
    parser_report = evaluate_parser_quality(
        documents, units_by_standard, pilot_config
    )
    ontology_report = schema_quality_report()
    report = {
        "document_parser_quality": (
            "passed" if parser_report["passed"] else "failed"
        ),
        "schema_professionalism": (
            "passed" if ontology_report["passed"] else "failed"
        ),
        "passed": parser_report["passed"] and ontology_report["passed"],
        "parser_report": parser_report,
        "ontology_report": ontology_report,
        "artifacts": {
            "layout_dir": str(layout_dir),
            "source_units": str(output_dir / "source_units.jsonl"),
        },
    }
    write_json(output_dir / "foundation_quality_report.json", report)
    return report
