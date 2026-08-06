from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiment_io_v2 import assert_blind_records, sha256_path, utc_now
from .extractor_v2 import RuleCandidateExtractorV2, build_llm_selection_messages
from .io_utils import read_jsonl, write_json, write_jsonl


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return str(entity["name"]), str(entity["entity_type"])


def _relation_key(
    relation: dict[str, Any], entities: dict[str, dict[str, Any]]
) -> tuple[str, str, str, str, str]:
    head = entities[str(relation["head_id"])]
    tail = entities[str(relation["tail_id"])]
    return (*_entity_key(head), str(relation["relation_type"]), *_entity_key(tail))


def prepare_lora_training_v2(
    units_path: Path,
    annotations_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    units = read_jsonl(units_path)
    annotations = read_jsonl(annotations_path)
    if not units:
        raise ValueError("blind training source unit file is empty")
    assert_blind_records(units)
    non_train_splits = sorted(
        {str(row.get("split", "missing")) for row in annotations}
        - {"train"}
    )
    if non_train_splits:
        raise ValueError(
            "prepare-lora-v2 accepts only annotations explicitly marked split=train; "
            f"found {non_train_splits}"
        )

    annotation_by_unit = {str(row["unit_id"]): row for row in annotations}
    unit_ids = {str(row["unit_id"]) for row in units}
    if len(unit_ids) != len(units) or len(annotation_by_unit) != len(annotations):
        raise ValueError("training unit ids and annotation unit ids must be unique")
    if set(annotation_by_unit) != unit_ids:
        missing = sorted(unit_ids - set(annotation_by_unit))
        extra = sorted(set(annotation_by_unit) - unit_ids)
        raise ValueError(
            f"training unit/gold mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    extractor = RuleCandidateExtractorV2()
    rows: list[dict[str, Any]] = []
    gold_entity_count = 0
    covered_entity_count = 0
    gold_relation_count = 0
    covered_relation_count = 0

    for unit in units:
        unit_id = str(unit["unit_id"])
        gold = annotation_by_unit[unit_id]
        candidates = extractor.predict_unit(unit)

        gold_entities = {
            str(entity["entity_id"]): entity for entity in gold.get("entities", [])
        }
        candidate_entities = {
            str(entity["entity_id"]): entity
            for entity in candidates.get("entities", [])
        }
        gold_entity_keys = {_entity_key(entity) for entity in gold_entities.values()}
        candidate_entity_ids_by_key = {
            _entity_key(entity): entity_id
            for entity_id, entity in candidate_entities.items()
        }

        gold_relation_keys = {
            _relation_key(relation, gold_entities)
            for relation in gold.get("relations", [])
        }
        candidate_relation_ids_by_key = {
            _relation_key(relation, candidate_entities): str(relation["relation_id"])
            for relation in candidates.get("relations", [])
        }
        selected_relation_ids = sorted(
            candidate_relation_ids_by_key[key]
            for key in gold_relation_keys
            if key in candidate_relation_ids_by_key
        )
        selected_relation_set = set(selected_relation_ids)
        relation_by_id = {
            str(relation["relation_id"]): relation
            for relation in candidates.get("relations", [])
        }
        required_entity_ids = {
            str(endpoint)
            for relation_id in selected_relation_set
            for endpoint in (
                relation_by_id[relation_id]["head_id"],
                relation_by_id[relation_id]["tail_id"],
            )
        }
        selected_entity_ids = sorted(
            entity_id
            for key, entity_id in candidate_entity_ids_by_key.items()
            if key in gold_entity_keys and entity_id not in required_entity_ids
        )

        gold_entity_count += len(gold_entity_keys)
        covered_entity_count += len(gold_entity_keys & set(candidate_entity_ids_by_key))
        gold_relation_count += len(gold_relation_keys)
        covered_relation_count += len(
            gold_relation_keys & set(candidate_relation_ids_by_key)
        )

        system, prompt = build_llm_selection_messages(unit, candidates)
        completion = json.dumps(
            {
                "selected_entity_ids": selected_entity_ids,
                "selected_relation_ids": selected_relation_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }
        )

    write_jsonl(output_path, rows)
    manifest = {
        "schema_version": "ecospec-lora-training-v2.0",
        "created_at": utc_now(),
        "source_units": str(units_path),
        "source_units_sha256": sha256_path(units_path),
        "annotations": str(annotations_path),
        "annotations_sha256": sha256_path(annotations_path),
        "output": str(output_path),
        "training_records": len(rows),
        "candidate_coverage": {
            "gold_entities": gold_entity_count,
            "covered_entities": covered_entity_count,
            "entity_recall_upper_bound": (
                covered_entity_count / gold_entity_count if gold_entity_count else 1.0
            ),
            "gold_relations": gold_relation_count,
            "covered_relations": covered_relation_count,
            "relation_recall_upper_bound": (
                covered_relation_count / gold_relation_count
                if gold_relation_count
                else 1.0
            ),
        },
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest
