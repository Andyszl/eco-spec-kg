from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .io_utils import normalize_space, sha256_file, stable_id, write_json, write_jsonl
from .models import DocumentChunk

HJ_PATTERN = re.compile(r"HJ\s*(11(?:6[6-9]|7[0-6]))\s*[—-]\s*(2021)")
SECTION_PATTERN = re.compile(r"(?m)^\s*(\d+(?:\.\d+){0,3})\s+([^\n]{1,60})")
EXCLUDED_NAME_PARTS = ("编制说明",)
EXTERNAL_TEST_CODES = {"HJ 1171-2021", "HJ 1174-2021", "HJ 1175-2021"}


@dataclass(slots=True)
class CorpusFile:
    path: Path
    sha256: str
    standard_code: str
    title: str
    page_count: int
    role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": str(self.path),
            "source_sha256": self.sha256,
            "standard_code": self.standard_code,
            "title": self.title,
            "page_count": self.page_count,
            "role": self.role,
        }


def _pdf_text_head(path: Path, pages: int = 3) -> tuple[str, int]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for corpus extraction") from exc
    with pdfplumber.open(path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages[:pages])
        return text, len(pdf.pages)


def extract_standard_code(text: str, filename: str = "") -> str:
    match = HJ_PATTERN.search(text.replace("\u2014", "—"))
    if not match:
        match = HJ_PATTERN.search(filename.replace("\u2014", "—"))
    if not match:
        return ""
    return f"HJ {match.group(1)}-{match.group(2)}"


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = normalize_space(line)
        if line.startswith("——") and len(line) > 2:
            return line.lstrip("——").strip()
    cleaned = re.sub(r"^\d+\.", "", Path(fallback).stem)
    cleaned = re.sub(r"全国生态状况调查评估技术规范[—（(]?", "", cleaned)
    return cleaned.rstrip("）)")


def _role(code: str) -> str:
    if code in EXTERNAL_TEST_CODES:
        return "external_test"
    if code in {"HJ 1172-2021", "HJ 1173-2021"}:
        return "core"
    return "support"


def discover_corpus(source_dir: Path) -> list[CorpusFile]:
    candidates: list[CorpusFile] = []
    seen_hashes: set[str] = set()
    for path in sorted(source_dir.rglob("*.pdf")):
        if any(part in path.name for part in EXCLUDED_NAME_PARTS):
            continue
        digest = sha256_file(path)
        if digest in seen_hashes:
            continue
        head, page_count = _pdf_text_head(path)
        code = extract_standard_code(head, path.name)
        if not code:
            continue
        seen_hashes.add(digest)
        candidates.append(
            CorpusFile(
                path=path,
                sha256=digest,
                standard_code=code,
                title=_title_from_text(head, path.name),
                page_count=page_count,
                role=_role(code),
            )
        )

    by_code: dict[str, list[CorpusFile]] = defaultdict(list)
    for item in candidates:
        by_code[item.standard_code].append(item)

    canonical: list[CorpusFile] = []
    for code, items in by_code.items():
        # Prefer unnumbered filenames and the smaller official PDF when copies differ.
        items.sort(
            key=lambda item: (
                bool(re.match(r"^\d+\.", item.path.name)),
                item.path.stat().st_size,
                len(item.path.name),
            )
        )
        canonical.append(items[0])
    return sorted(canonical, key=lambda item: item.standard_code)


def classify_block(text: str) -> str:
    if "式中" in text or re.search(r"公式[（(]?[A-Z]?\d+", text):
        return "formula"
    if re.search(r"(^|\s)表\s*[A-Z]?\d", text):
        return "table"
    if re.search(r"(^|\s)图\s*\d", text):
        return "figure_caption"
    return "paragraph"


def _split_page_text(text: str, max_chars: int, overlap: int) -> Iterable[str]:
    clean = normalize_space(text)
    if not clean:
        return
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            cut = max(clean.rfind("。", start, end), clean.rfind("；", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        yield clean[start:end]
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)


def _section_for_text(text: str, previous: str) -> str:
    match = SECTION_PATTERN.search(text.replace("。", "\n"))
    return match.group(1) if match else previous


def extract_chunks(
    corpus_files: list[CorpusFile],
    max_chars: int = 1400,
    overlap: int = 150,
) -> list[DocumentChunk]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for corpus extraction") from exc
    chunks: list[DocumentChunk] = []
    for item in corpus_files:
        current_section = ""
        with pdfplumber.open(item.path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                for local_index, text in enumerate(
                    _split_page_text(page_text, max_chars, overlap), start=1
                ):
                    current_section = _section_for_text(text, current_section)
                    chunk_id = (
                        f"{item.standard_code.replace(' ', '').replace('-', '_')}"
                        f"-p{page_no}-c{local_index}"
                    )
                    path_hint = current_section or f"page-{page_no}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            standard_code=item.standard_code,
                            document_title=item.title,
                            page=page_no,
                            section=current_section,
                            block_type=classify_block(text),
                            text=text,
                            source_file=str(item.path),
                            source_sha256=item.sha256,
                            path_id=stable_id(item.standard_code, path_hint),
                            bbox=None,
                        )
                    )
    return chunks


def build_corpus(source_dir: Path, output_dir: Path) -> dict[str, object]:
    corpus_files = discover_corpus(source_dir)
    if len(corpus_files) != 11:
        raise RuntimeError(
            f"expected 11 unique HJ 1166-HJ 1176 standards, found {len(corpus_files)}"
        )
    chunks = extract_chunks(corpus_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", [item.to_dict() for item in corpus_files])
    write_jsonl(output_dir / "chunks.jsonl", (item.to_dict() for item in chunks))
    summary = {
        "standard_count": len(corpus_files),
        "chunk_count": len(chunks),
        "roles": {
            role: sum(1 for item in corpus_files if item.role == role)
            for role in ("core", "support", "external_test")
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary

