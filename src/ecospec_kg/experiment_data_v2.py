from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .experiment_io_v2 import (
    assert_blind_records,
    forbidden_key_paths,
    sanitize_blind_record,
    sha256_json,
    sha256_path,
    utc_now,
    write_json,
    write_manifested_jsonl,
)
from .io_utils import read_jsonl, stable_id
from .ontology_v2 import ONTOLOGY_VERSION, schema_quality_report, schema_rows_v2


DATASET_PACKAGE_VERSION = "ecospec-experiment-dataset-v2.1"
SPLIT_POLICY_VERSION = "ecospec-document-group-split-v2.0"
EXTERNAL_TEST_CODES = frozenset(
    {"HJ 1171-2021", "HJ 1174-2021", "HJ 1175-2021"}
)


def split_for_experiment_unit(unit: dict[str, Any]) -> str:
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


def _split_rows(
    source_units: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    unit_by_id = {unit["unit_id"]: unit for unit in source_units}
    annotation_by_id = {row["unit_id"]: row for row in annotations}
    if len(unit_by_id) != len(source_units):
        raise ValueError("source unit ids are not unique")
    if len(annotation_by_id) != len(annotations):
        raise ValueError("annotation unit ids are not unique")
    if set(unit_by_id) != set(annotation_by_id):
        missing_gold = sorted(set(unit_by_id) - set(annotation_by_id))
        extra_gold = sorted(set(annotation_by_id) - set(unit_by_id))
        raise ValueError(
            "source/gold unit mismatch: "
            f"missing_gold={missing_gold[:10]} extra_gold={extra_gold[:10]}"
        )

    blind = {split: [] for split in ("train", "dev", "test")}
    gold = {split: [] for split in ("train", "dev", "test")}
    for unit in source_units:
        split = split_for_experiment_unit(unit)
        blind[split].append(unit)
        gold[split].append(
            {
                **annotation_by_id[unit["unit_id"]],
                "split": split,
            }
        )
    for rows in (*blind.values(), *gold.values()):
        rows.sort(key=lambda row: row["unit_id"])
    return blind, gold


def prepare_experiment_package_v2(
    source_units_path: Path,
    annotations_path: Path,
    out_dir: Path,
    *,
    dataset_version: str = "v2.1",
    gold_nature: str = "ai_expert_pre_gold",
) -> dict[str, Any]:
    source_units = read_jsonl(source_units_path)
    annotations = read_jsonl(annotations_path)
    if not source_units:
        raise ValueError("source unit file is empty")
    if not annotations:
        raise ValueError("annotation file is empty")

    answer_payload_paths = [
        path
        for unit in source_units
        for path in forbidden_key_paths(unit)
        if path.rsplit(".", 1)[-1]
        in {"gold_annotation", "entities", "relations", "triple_id"}
    ]
    if answer_payload_paths:
        raise ValueError(
            "source unit input contains answer payload fields: "
            + ", ".join(answer_payload_paths[:20])
        )

    blind, gold = _split_rows(source_units, annotations)
    removed_blind_metadata: list[str] = []
    for split, rows in blind.items():
        sanitized_rows = []
        for row in rows:
            sanitized, removed = sanitize_blind_record(row)
            sanitized_rows.append(sanitized)
            removed_blind_metadata.extend(
                f"{split}:{row['unit_id']}:{path}" for path in removed
            )
        blind[split] = sanitized_rows
    for rows in blind.values():
        assert_blind_records(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        **schema_quality_report(),
        "ontology_version": ONTOLOGY_VERSION,
        "schema_rows": schema_rows_v2(),
    }
    schema_path = out_dir / "schema_v2.json"
    write_json(schema_path, schema)

    files: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        files.append(
            write_manifested_jsonl(
                out_dir / "blind" / f"{split}_units.jsonl",
                blind[split],
            )
        )
        files.append(
            write_manifested_jsonl(
                out_dir / "gold" / f"{split}_annotations.jsonl",
                gold[split],
            )
        )

    unit_counts = {split: len(rows) for split, rows in blind.items()}
    relation_counts = {
        split: sum(len(row.get("relations", [])) for row in rows)
        for split, rows in gold.items()
    }
    entity_counts = {
        split: sum(len(row.get("entities", [])) for row in rows)
        for split, rows in gold.items()
    }
    package_id = stable_id(
        DATASET_PACKAGE_VERSION,
        dataset_version,
        sha256_json(unit_counts),
        sha256_json(relation_counts),
    )
    manifest = {
        "schema_version": DATASET_PACKAGE_VERSION,
        "package_id": package_id,
        "dataset_version": dataset_version,
        "gold_nature": gold_nature,
        "human_expert_review_required_for_publication": (
            gold_nature != "human_expert_gold"
        ),
        "ontology_version": ONTOLOGY_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "created_at": utc_now(),
        "source_unit_count": len(source_units),
        "standard_counts": dict(
            sorted(
                Counter(
                    unit["provenance"]["standard_code"]
                    for unit in source_units
                ).items()
            )
        ),
        "unit_split_counts": unit_counts,
        "entity_split_counts": entity_counts,
        "relation_split_counts": relation_counts,
        "removed_blind_metadata_count": len(removed_blind_metadata),
        "removed_blind_metadata": removed_blind_metadata,
        "schema": {
            "path": schema_path.relative_to(out_dir).as_posix(),
            "sha256": sha256_path(schema_path),
        },
        "files": [
            {
                **item,
                "path": Path(item["path"]).relative_to(out_dir).as_posix(),
            }
            for item in files
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "DATASET_PACKAGE_VERSION",
    "EXTERNAL_TEST_CODES",
    "SPLIT_POLICY_VERSION",
    "prepare_experiment_package_v2",
    "split_for_experiment_unit",
]
