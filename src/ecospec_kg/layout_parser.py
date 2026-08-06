from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .corpus import CorpusFile, discover_corpus
from .io_utils import read_json, sha256_file, stable_id, write_json

LAYOUT_SCHEMA_VERSION = "layout-v2.0"
COORDINATE_SYSTEM = "pdfplumber_top_left_points"

FORMULA_NUMBER_RE = re.compile(
    r"[（(](?P<number>[A-Z](?:\.\d+)+|\d+(?:\.\d+)?)[）)]"
)
NUMERIC_HEADING_RE = re.compile(
    r"^(?P<section>\d{1,2}(?:\.\d{1,2}){0,3})\s+"
    r"(?P<title>[\u3400-\u9fff].{0,80})$"
)
APPENDIX_SECTION_RE = re.compile(
    r"^(?P<section>[A-Z]\.\d+(?:\.\d+)*)\s+"
    r"(?P<title>[\u3400-\u9fff].{0,80})$"
)
APPENDIX_RE = re.compile(r"^附\s*录\s*(?P<letter>[A-Z])$")
TABLE_CAPTION_RE = re.compile(r"^表\s*[A-Z]?(?:\.\s*)?\d+(?:\.\d+)?\s*")
FIGURE_CAPTION_RE = re.compile(r"^图\s*\d+(?:\.\d+)?\s*")

# The PDFs use an embedded Symbol font without a ToUnicode map. pdfminer exposes
# those glyphs as U+F0xx, where the low byte is the Symbol encoding code.
SYMBOL_FONT_MAP = {
    "\uf028": "(",
    "\uf029": ")",
    "\uf02b": "+",
    "\uf02d": "−",
    "\uf03d": "=",
    "\uf044": "Δ",
    "\uf05b": "[",
    "\uf05d": "]",
    "\uf061": "α",
    "\uf062": "β",
    "\uf06c": "λ",
    "\uf071": "θ",
    "\uf072": "ρ",
    "\uf0a2": "′",
    "\uf0b0": "°",
    "\uf0b4": "×",
    "\uf0d7": "·",
    "\uf0e5": "∑",
    "\uf0e6": "⎛",
    "\uf0e7": "⎝",
    "\uf0e8": "⎞",
    "\uf0e9": "⎡",
    "\uf0eb": "⎧",
    "\uf0ec": "⎤",
    "\uf0ed": "⎨",
    "\uf0ee": "⎩",
    "\uf0ef": "⎪",
    "\uf0f6": "⎜",
    "\uf0f7": "⎟",
    "\uf0f8": "⎠",
    "\uf0f9": "⎣",
    "\uf0fb": "⎬",
}


def normalize_pdf_text(text: str) -> str:
    for source, target in SYMBOL_FONT_MAP.items():
        text = text.replace(source, target)
    text = text.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _round_bbox(bbox: Iterable[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox]


@dataclass(slots=True)
class LayoutLine:
    line_id: str
    page: int
    reading_order: int
    text: str
    bbox: list[float]
    kind: str
    section: str = ""
    heading_chain: list[str] = field(default_factory=list)
    in_table: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutLine":
        return cls(**data)


@dataclass(slots=True)
class LayoutTable:
    table_id: str
    page: int
    title: str
    bbox: list[float]
    section: str
    heading_chain: list[str]
    header_rows: list[list[str | None]]
    column_names: list[str]
    rows: list[list[str | None]]
    cell_bboxes: list[list[list[float] | None]]
    data_start_row: int
    notes: list[str] = field(default_factory=list)
    continued_from_table_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutTable":
        return cls(**data)


@dataclass(slots=True)
class LayoutPage:
    page: int
    printed_page: str
    width: float
    height: float
    lines: list[LayoutLine]
    tables: list[LayoutTable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "printed_page": self.printed_page,
            "width": self.width,
            "height": self.height,
            "lines": [line.to_dict() for line in self.lines],
            "tables": [table.to_dict() for table in self.tables],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutPage":
        return cls(
            page=int(data["page"]),
            printed_page=str(data.get("printed_page", "")),
            width=float(data["width"]),
            height=float(data["height"]),
            lines=[LayoutLine.from_dict(row) for row in data.get("lines", [])],
            tables=[LayoutTable.from_dict(row) for row in data.get("tables", [])],
        )


@dataclass(slots=True)
class LayoutDocument:
    standard_code: str
    document_title: str
    source_file: str
    source_sha256: str
    pages: list[LayoutPage]
    schema_version: str = LAYOUT_SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_system": self.coordinate_system,
            "standard_code": self.standard_code,
            "document_title": self.document_title,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutDocument":
        return cls(
            schema_version=str(data.get("schema_version", LAYOUT_SCHEMA_VERSION)),
            coordinate_system=str(
                data.get("coordinate_system", COORDINATE_SYSTEM)
            ),
            standard_code=str(data["standard_code"]),
            document_title=str(data["document_title"]),
            source_file=str(data["source_file"]),
            source_sha256=str(data["source_sha256"]),
            pages=[LayoutPage.from_dict(row) for row in data.get("pages", [])],
        )


@dataclass(slots=True)
class _RawTable:
    bbox: list[float]
    rows: list[list[str | None]]
    cell_bboxes: list[list[list[float] | None]]


class _HeadingState:
    def __init__(self) -> None:
        self._levels: dict[int, str] = {}
        self.section = ""

    @property
    def chain(self) -> list[str]:
        return [self._levels[level] for level in sorted(self._levels)]

    def update(self, section: str, title: str) -> None:
        if section.startswith("附录 "):
            level = 1
        elif section[:1].isalpha():
            level = section.count(".") + 1
        else:
            level = section.count(".") + 1
        for key in [key for key in self._levels if key >= level]:
            del self._levels[key]
        label = f"{section} {title}".strip()
        self._levels[level] = label
        self.section = section


def _heading_parts(text: str) -> tuple[str, str] | None:
    appendix = APPENDIX_RE.fullmatch(text)
    if appendix:
        return f"附录 {appendix.group('letter')}", ""
    if _formula_number(text) and _has_formula_operator(text):
        return None
    match = APPENDIX_SECTION_RE.fullmatch(text)
    if match:
        return match.group("section"), match.group("title")
    match = NUMERIC_HEADING_RE.fullmatch(text)
    if not match:
        return None
    first = int(match.group("section").split(".", 1)[0])
    if first > 30 or match.group("title").startswith("式中"):
        return None
    return match.group("section"), match.group("title")


def _formula_number(text: str) -> str:
    match = FORMULA_NUMBER_RE.search(text)
    return match.group("number") if match else ""


def _has_formula_operator(text: str) -> bool:
    return bool(
        re.search(r"[=∑×·+\-−/<>≤≥\[\]{}]", text)
        or re.search(r"\b(?:exp|sin|cos|ln|max|min)\b", text, re.IGNORECASE)
    )


def _is_formula_anchor(text: str) -> bool:
    if not _formula_number(text):
        return False
    return _has_formula_operator(text)


def _is_header_footer(text: str, bbox: list[float], page_height: float) -> bool:
    if bbox[1] < 105 and re.fullmatch(r"HJ\s*\d{4}[—-]\d{4}", text):
        return True
    return bool(bbox[3] > page_height - 70 and re.fullmatch(r"\d+", text))


def _overlap_ratio(inner: list[float], outer: list[float]) -> float:
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 0.001)
    return intersection / area


def _extract_raw_tables(page: Any) -> list[_RawTable]:
    tables: list[_RawTable] = []
    for table in page.find_tables():
        matrix = table.extract()
        width = max((len(row) for row in matrix), default=0)
        if len(matrix) < 2 or width < 2:
            continue
        rows: list[list[str | None]] = []
        cell_bboxes: list[list[list[float] | None]] = []
        for row_index, row in enumerate(matrix):
            normalized = [
                normalize_pdf_text(value.replace("\n", " ")) if value else None
                for value in row
            ]
            normalized.extend([None] * (width - len(normalized)))
            rows.append(normalized)
            source_cells = (
                table.rows[row_index].cells
                if row_index < len(table.rows)
                else []
            )
            bboxes = [
                _round_bbox(cell) if cell else None for cell in source_cells
            ]
            bboxes.extend([None] * (width - len(bboxes)))
            cell_bboxes.append(bboxes)
        tables.append(
            _RawTable(
                bbox=_round_bbox(table.bbox),
                rows=rows,
                cell_bboxes=cell_bboxes,
            )
        )
    return tables


def _extract_lines(
    page: Any,
    page_no: int,
    standard_code: str,
    raw_tables: list[_RawTable],
    heading_state: _HeadingState,
) -> tuple[list[LayoutLine], str]:
    raw_lines = page.extract_text_lines(
        x_tolerance=1,
        y_tolerance=3,
        strip=True,
        return_chars=False,
    )
    lines: list[LayoutLine] = []
    printed_page = ""
    for order, row in enumerate(raw_lines, start=1):
        text = normalize_pdf_text(str(row["text"]))
        if not text:
            continue
        bbox = _round_bbox(
            [row["x0"], row["top"], row["x1"], row["bottom"]]
        )
        in_table = any(
            _overlap_ratio(bbox, table.bbox) >= 0.5 for table in raw_tables
        )
        if _is_header_footer(text, bbox, float(page.height)):
            kind = "header_footer"
            if re.fullmatch(r"\d+", text):
                printed_page = text
        elif in_table:
            kind = "table_cell"
        elif _is_formula_anchor(text):
            kind = "formula"
        elif TABLE_CAPTION_RE.match(text):
            kind = "table_caption"
        elif FIGURE_CAPTION_RE.match(text):
            kind = "figure_caption"
        else:
            heading = _heading_parts(text)
            kind = "heading" if heading else "paragraph"
            if heading:
                heading_state.update(*heading)

        line_id = stable_id(standard_code, page_no, order, text, *bbox)
        lines.append(
            LayoutLine(
                line_id=line_id,
                page=page_no,
                reading_order=order,
                text=text,
                bbox=bbox,
                kind=kind,
                section=heading_state.section,
                heading_chain=heading_state.chain,
                in_table=in_table,
            )
        )
    _mark_formula_fragments(lines)
    return lines, printed_page


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _is_formula_fragment_candidate(
    text: str, candidate_height: float, anchor_height: float
) -> bool:
    if text.endswith(("：", ":", "。", "；", ";")):
        return False
    if not _contains_cjk(text):
        return True
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    has_formula_token = bool(
        re.search(r"[A-Za-z0-9=∑×·+\-−/<>≤≥]", text)
    )
    return bool(
        len(text) <= 30
        and (
            candidate_height <= anchor_height * 0.75
            or (cjk_count <= 2 and has_formula_token)
        )
    )


def _mark_formula_fragments(lines: list[LayoutLine]) -> None:
    for index, line in enumerate(lines):
        if line.kind != "formula":
            continue
        anchor_height = line.bbox[3] - line.bbox[1]
        cursor = index - 1
        while cursor >= 0:
            candidate = lines[cursor]
            gap = line.bbox[1] - candidate.bbox[3]
            candidate_height = candidate.bbox[3] - candidate.bbox[1]
            fragment_like = _is_formula_fragment_candidate(
                candidate.text, candidate_height, anchor_height
            )
            if (
                candidate.kind not in {"paragraph", "formula_fragment"}
                or not fragment_like
                or gap > 24
            ):
                break
            candidate.kind = "formula_fragment"
            cursor -= 1
        cursor = index + 1
        previous = line
        while cursor < len(lines):
            candidate = lines[cursor]
            gap = candidate.bbox[1] - previous.bbox[3]
            candidate_height = candidate.bbox[3] - candidate.bbox[1]
            fragment_like = _is_formula_fragment_candidate(
                candidate.text, candidate_height, anchor_height
            )
            if (
                candidate.kind not in {"paragraph", "formula_fragment"}
                or not fragment_like
                or gap > 10
            ):
                break
            candidate.kind = "formula_fragment"
            previous = candidate
            cursor += 1
    _mark_variable_fragments(lines)


def _mark_variable_fragments(lines: list[LayoutLine]) -> None:
    for index, line in enumerate(lines[:-1]):
        if "——" not in line.text and "--" not in line.text:
            continue
        candidate = lines[index + 1]
        line_height = line.bbox[3] - line.bbox[1]
        candidate_height = candidate.bbox[3] - candidate.bbox[1]
        if (
            candidate.kind == "paragraph"
            and len(candidate.text) <= 20
            and "——" not in candidate.text
            and "--" not in candidate.text
            and candidate_height <= line_height * 0.75
            and candidate.bbox[1] - line.bbox[3] <= 8
        ):
            candidate.kind = "variable_fragment"


def _header_row_count(rows: list[list[str | None]]) -> int:
    if not rows:
        return 0
    if len(rows) > 2 and any(value is None for value in rows[0]):
        second = [value for value in rows[1] if value]
        if second and all(not re.search(r"\d", value) for value in second):
            return 2
    return 1


def _column_names(
    header_rows: list[list[str | None]], width: int
) -> list[str]:
    if not header_rows:
        return [f"column_{index}" for index in range(1, width + 1)]
    filled_rows: list[list[str]] = []
    for row in header_rows:
        filled: list[str] = []
        previous = ""
        for value in row:
            if value:
                previous = value
            filled.append(value or previous)
        filled.extend([""] * (width - len(filled)))
        filled_rows.append(filled)

    names: list[str] = []
    used: dict[str, int] = {}
    for column in range(width):
        parts: list[str] = []
        for row in filled_rows:
            value = row[column].strip()
            if value and (not parts or value != parts[-1]):
                parts.append(value)
        base = " / ".join(parts) or f"column_{column + 1}"
        used[base] = used.get(base, 0) + 1
        names.append(base if used[base] == 1 else f"{base}_{used[base]}")
    return names


def _nearest_table_title(lines: list[LayoutLine], bbox: list[float]) -> str:
    candidates = [
        line
        for line in lines
        if line.kind == "table_caption"
        and line.bbox[3] <= bbox[1] + 3
        and bbox[1] - line.bbox[3] <= 90
    ]
    if candidates:
        return min(candidates, key=lambda line: bbox[1] - line.bbox[3]).text
    fallback = [
        line
        for line in lines
        if line.kind == "paragraph"
        and line.bbox[3] <= bbox[1] + 3
        and bbox[1] - line.bbox[3] <= 100
        and line.bbox[0] >= 120
        and len(line.text) <= 40
        and not line.text.startswith(("（", "("))
        and not line.text.endswith(("。", "；", ";"))
    ]
    return (
        min(fallback, key=lambda line: bbox[1] - line.bbox[3]).text
        if fallback
        else ""
    )


def _table_notes(lines: list[LayoutLine], bbox: list[float]) -> list[str]:
    return [
        line.text
        for line in lines
        if line.text.startswith("注")
        and line.bbox[1] >= bbox[3] - 3
        and line.bbox[1] - bbox[3] <= 80
    ]


def _build_tables(
    standard_code: str,
    page_no: int,
    page_height: float,
    lines: list[LayoutLine],
    raw_tables: list[_RawTable],
    previous_table: LayoutTable | None,
    previous_page_height: float,
) -> list[LayoutTable]:
    output: list[LayoutTable] = []
    for index, raw in enumerate(raw_tables, start=1):
        width = max((len(row) for row in raw.rows), default=0)
        continuation = bool(
            previous_table
            and previous_table.bbox[3] >= previous_page_height - 120
            and raw.bbox[1] <= 150
            and len(previous_table.column_names) == width
        )
        if continuation:
            title = previous_table.title
            header_rows = previous_table.header_rows
            names = previous_table.column_names
            repeated_headers = 0
            for row_index, header in enumerate(header_rows):
                if (
                    row_index < len(raw.rows)
                    and raw.rows[row_index] == header
                ):
                    repeated_headers += 1
                else:
                    break
            data_start = repeated_headers
            continued_from = previous_table.table_id
        else:
            title = _nearest_table_title(lines, raw.bbox)
            header_count = _header_row_count(raw.rows)
            header_rows = raw.rows[:header_count]
            names = _column_names(header_rows, width)
            data_start = header_count
            continued_from = ""

        inside = [
            line
            for line in lines
            if line.in_table and _overlap_ratio(line.bbox, raw.bbox) >= 0.5
        ]
        reference = inside[0] if inside else None
        table_id = stable_id(
            standard_code, page_no, index, title, *raw.bbox
        )
        output.append(
            LayoutTable(
                table_id=table_id,
                page=page_no,
                title=title,
                bbox=raw.bbox,
                section=reference.section if reference else "",
                heading_chain=reference.heading_chain if reference else [],
                header_rows=header_rows,
                column_names=names,
                rows=raw.rows,
                cell_bboxes=raw.cell_bboxes,
                data_start_row=data_start,
                notes=_table_notes(lines, raw.bbox),
                continued_from_table_id=continued_from,
            )
        )
        previous_table = output[-1]
    return output


def parse_pdf_layout(
    path: Path,
    standard_code: str,
    document_title: str,
    source_sha256: str = "",
) -> LayoutDocument:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for layout parsing") from exc

    digest = source_sha256 or sha256_file(path)
    heading_state = _HeadingState()
    output_pages: list[LayoutPage] = []
    previous_table: LayoutTable | None = None
    previous_page_height = 0.0
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            raw_tables = _extract_raw_tables(page)
            lines, printed_page = _extract_lines(
                page,
                page_no,
                standard_code,
                raw_tables,
                heading_state,
            )
            tables = _build_tables(
                standard_code,
                page_no,
                float(page.height),
                lines,
                raw_tables,
                previous_table,
                previous_page_height,
            )
            if tables:
                previous_table = tables[-1]
            elif previous_table and previous_table.bbox[3] < previous_page_height - 120:
                previous_table = None
            output_pages.append(
                LayoutPage(
                    page=page_no,
                    printed_page=printed_page,
                    width=round(float(page.width), 3),
                    height=round(float(page.height), 3),
                    lines=lines,
                    tables=tables,
                )
            )
            previous_page_height = float(page.height)

    return LayoutDocument(
        standard_code=standard_code,
        document_title=document_title,
        source_file=str(path),
        source_sha256=digest,
        pages=output_pages,
    )


def layout_filename(standard_code: str) -> str:
    return standard_code.replace(" ", "_").replace("-", "_") + ".layout.json"


def save_layout_document(document: LayoutDocument, path: Path) -> None:
    write_json(path, document.to_dict())


def load_layout_document(path: Path) -> LayoutDocument:
    return LayoutDocument.from_dict(read_json(path))


def parse_layout_corpus(
    source_dir: Path,
    output_dir: Path,
    standards: set[str] | None = None,
) -> dict[str, Any]:
    corpus = discover_corpus(source_dir)
    if standards:
        corpus = [item for item in corpus if item.standard_code in standards]
    if not corpus:
        raise RuntimeError("no matching HJ standards found")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    totals = {"pages": 0, "lines": 0, "tables": 0, "formulas": 0}
    for item in corpus:
        document = parse_pdf_layout(
            item.path,
            item.standard_code,
            item.title,
            item.sha256,
        )
        filename = layout_filename(item.standard_code)
        save_layout_document(document, output_dir / filename)
        counts = _layout_counts(document)
        for key in totals:
            totals[key] += counts[key]
        manifest.append(
            {
                **item.to_dict(),
                "layout_file": filename,
                "layout_schema_version": LAYOUT_SCHEMA_VERSION,
                **counts,
            }
        )
    write_json(output_dir / "manifest.json", manifest)
    summary = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "document_count": len(corpus),
        **totals,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _layout_counts(document: LayoutDocument) -> dict[str, int]:
    lines = [line for page in document.pages for line in page.lines]
    return {
        "pages": len(document.pages),
        "lines": len(lines),
        "tables": sum(len(page.tables) for page in document.pages),
        "formulas": sum(line.kind == "formula" for line in lines),
    }


def corpus_file_for_code(
    source_dir: Path, standard_code: str
) -> CorpusFile:
    for item in discover_corpus(source_dir):
        if item.standard_code == standard_code:
            return item
    raise ValueError(f"standard not found: {standard_code}")
