from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_json, read_jsonl, write_json, write_jsonl


FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "gold_annotation",
        "gold_nature",
        "review_status",
        "entities",
        "relations",
        "triple_id",
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def runtime_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
    }


def forbidden_key_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in FORBIDDEN_BLIND_KEYS:
                failures.append(child_path)
            failures.extend(forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return failures


def assert_blind_records(records: Iterable[dict[str, Any]]) -> None:
    failures: list[str] = []
    for index, record in enumerate(records):
        failures.extend(
            f"record[{index}]{path[1:]}" for path in forbidden_key_paths(record)
        )
    if failures:
        preview = ", ".join(failures[:20])
        raise ValueError(f"blind input contains forbidden gold fields: {preview}")


def sanitize_blind_record(
    value: Any,
    *,
    prefix: str = "$",
) -> tuple[Any, list[str]]:
    removed: list[str] = []
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if key in FORBIDDEN_BLIND_KEYS:
                removed.append(path)
                continue
            sanitized, child_removed = sanitize_blind_record(
                child,
                prefix=path,
            )
            output[key] = sanitized
            removed.extend(child_removed)
        return output, removed
    if isinstance(value, list):
        output_list = []
        for index, child in enumerate(value):
            sanitized, child_removed = sanitize_blind_record(
                child,
                prefix=f"{prefix}[{index}]",
            )
            output_list.append(sanitized)
            removed.extend(child_removed)
        return output_list, removed
    return value, removed


def load_json_config(path: Path | None) -> dict[str, Any]:
    return read_json(path) if path else {}


def write_manifested_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    write_jsonl(path, rows)
    return {
        "path": path.as_posix(),
        "records": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


__all__ = [
    "FORBIDDEN_BLIND_KEYS",
    "assert_blind_records",
    "forbidden_key_paths",
    "load_json_config",
    "read_json",
    "read_jsonl",
    "runtime_metadata",
    "sanitize_blind_record",
    "sha256_json",
    "sha256_path",
    "utc_now",
    "write_json",
    "write_jsonl",
    "write_manifested_jsonl",
]
