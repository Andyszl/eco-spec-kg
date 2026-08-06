from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import normalize_space
from .ontology_v2 import EntityTypeV2, validate_relation_v2


def _all_span_ids(value: Any) -> set[str]:
    spans: set[str] = set()
    if isinstance(value, dict):
        if value.get("span_id"):
            spans.add(value["span_id"])
        for child in value.values():
            spans.update(_all_span_ids(child))
    elif isinstance(value, list):
        for child in value:
            spans.update(_all_span_ids(child))
    return spans


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_space(entity["name"]),
        entity["entity_type"],
    )


def _relation_key(
    relation: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        normalize_space(relation["head_name"]),
        relation["head_type"],
        relation["relation_type"],
        normalize_space(relation["tail_name"]),
        relation["tail_type"],
    )


def validate_expert_annotations(
    source_units: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    annotator_id: str,
) -> dict[str, Any]:
    source_by_id = {unit["unit_id"]: unit for unit in source_units}
    expected_ids = [unit["unit_id"] for unit in source_units]
    actual_ids = [item["unit_id"] for item in annotations]
    failures: list[dict[str, Any]] = []
    valid_types = {item.value for item in EntityTypeV2}

    if actual_ids != expected_ids:
        failures.append(
            {
                "type": "unit_order_or_inventory",
                "missing": sorted(set(expected_ids) - set(actual_ids)),
                "extra": sorted(set(actual_ids) - set(expected_ids)),
                "order_matches": actual_ids == expected_ids,
            }
        )
    for item in annotations:
        unit_id = item["unit_id"]
        unit = source_by_id.get(unit_id)
        if unit is None:
            continue
        if item.get("annotator_id") != annotator_id:
            failures.append(
                {
                    "unit_id": unit_id,
                    "type": "annotator_id",
                    "actual": item.get("annotator_id"),
                }
            )
        valid_spans = _all_span_ids(unit)
        entity_keys = {_entity_key(entity) for entity in item["entities"]}
        if len(entity_keys) != len(item["entities"]):
            failures.append(
                {"unit_id": unit_id, "type": "duplicate_entity"}
            )
        for entity in item["entities"]:
            if entity["entity_type"] not in valid_types:
                failures.append(
                    {
                        "unit_id": unit_id,
                        "type": "invalid_entity_type",
                        "entity": entity,
                    }
                )
            if not entity.get("evidence_span_ids") or not set(
                entity["evidence_span_ids"]
            ) <= valid_spans:
                failures.append(
                    {
                        "unit_id": unit_id,
                        "type": "invalid_entity_evidence",
                        "entity": entity,
                    }
                )
        relation_keys = {
            _relation_key(relation) for relation in item["relations"]
        }
        if len(relation_keys) != len(item["relations"]):
            failures.append(
                {"unit_id": unit_id, "type": "duplicate_relation"}
            )
        for relation in item["relations"]:
            valid, reason = validate_relation_v2(
                relation["head_type"],
                relation["relation_type"],
                relation["tail_type"],
            )
            if not valid:
                failures.append(
                    {
                        "unit_id": unit_id,
                        "type": "invalid_relation_schema",
                        "reason": reason,
                        "relation": relation,
                    }
                )
            endpoints = {
                (
                    normalize_space(relation["head_name"]),
                    relation["head_type"],
                ),
                (
                    normalize_space(relation["tail_name"]),
                    relation["tail_type"],
                ),
            }
            if not endpoints <= entity_keys:
                failures.append(
                    {
                        "unit_id": unit_id,
                        "type": "relation_endpoint_missing",
                        "relation": relation,
                    }
                )
            if not relation.get("evidence_span_ids") or not set(
                relation["evidence_span_ids"]
            ) <= valid_spans:
                failures.append(
                    {
                        "unit_id": unit_id,
                        "type": "invalid_relation_evidence",
                        "relation": relation,
                    }
                )
    return {
        "annotator_id": annotator_id,
        "passed": not failures,
        "unit_count": len(annotations),
        "entity_count": sum(len(item["entities"]) for item in annotations),
        "relation_count": sum(
            len(item["relations"]) for item in annotations
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def _set_metrics(
    left_sets: dict[str, set[Any]],
    right_sets: dict[str, set[Any]],
) -> dict[str, Any]:
    intersection = sum(
        len(left_sets[unit_id] & right_sets[unit_id])
        for unit_id in left_sets
    )
    left_count = sum(len(values) for values in left_sets.values())
    right_count = sum(len(values) for values in right_sets.values())
    precision = intersection / left_count if left_count else 1.0
    recall = intersection / right_count if right_count else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact = sum(
        left_sets[unit_id] == right_sets[unit_id]
        for unit_id in left_sets
    )
    return {
        "shared_count": intersection,
        "expert_A_count": left_count,
        "expert_B_count": right_count,
        "precision_A_to_B": round(precision, 6),
        "recall_A_to_B": round(recall, 6),
        "f1": round(f1, 6),
        "exact_unit_count": exact,
        "exact_unit_rate": round(exact / len(left_sets), 6),
    }


def compare_experts(
    source_units: list[dict[str, Any]],
    expert_a: list[dict[str, Any]],
    expert_b: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_by_id = {unit["unit_id"]: unit for unit in source_units}
    a_by_id = {item["unit_id"]: item for item in expert_a}
    b_by_id = {item["unit_id"]: item for item in expert_b}
    entity_a = {
        unit_id: {_entity_key(item) for item in row["entities"]}
        for unit_id, row in a_by_id.items()
    }
    entity_b = {
        unit_id: {_entity_key(item) for item in row["entities"]}
        for unit_id, row in b_by_id.items()
    }
    relation_a = {
        unit_id: {_relation_key(item) for item in row["relations"]}
        for unit_id, row in a_by_id.items()
    }
    relation_b = {
        unit_id: {_relation_key(item) for item in row["relations"]}
        for unit_id, row in b_by_id.items()
    }
    disagreements: list[dict[str, Any]] = []
    exact_both = 0
    by_standard: dict[str, Counter[str]] = {}
    for unit in source_units:
        unit_id = unit["unit_id"]
        code = unit["provenance"]["standard_code"]
        counter = by_standard.setdefault(code, Counter())
        counter["units"] += 1
        entity_exact = entity_a[unit_id] == entity_b[unit_id]
        relation_exact = relation_a[unit_id] == relation_b[unit_id]
        counter["entity_exact"] += entity_exact
        counter["relation_exact"] += relation_exact
        counter["both_exact"] += entity_exact and relation_exact
        exact_both += entity_exact and relation_exact
        if entity_exact and relation_exact:
            continue
        disagreements.append(
            {
                "unit_id": unit_id,
                "standard_code": code,
                "source_unit": unit,
                "expert_A": a_by_id[unit_id],
                "expert_B": b_by_id[unit_id],
                "diff": {
                    "entities_only_A": sorted(
                        entity_a[unit_id] - entity_b[unit_id]
                    ),
                    "entities_only_B": sorted(
                        entity_b[unit_id] - entity_a[unit_id]
                    ),
                    "relations_only_A": sorted(
                        relation_a[unit_id] - relation_b[unit_id]
                    ),
                    "relations_only_B": sorted(
                        relation_b[unit_id] - relation_a[unit_id]
                    ),
                },
            }
        )
    report = {
        "schema_version": "pilot-agreement-v2.1",
        "annotation_nature": "independent_ai_expert_pilot",
        "human_expert_review_required_for_publication": True,
        "unit_count": len(source_units),
        "entity_agreement": _set_metrics(entity_a, entity_b),
        "relation_agreement": _set_metrics(relation_a, relation_b),
        "both_exact_unit_count": exact_both,
        "both_exact_unit_rate": round(exact_both / len(source_units), 6),
        "disagreement_unit_count": len(disagreements),
        "by_standard": {
            code: {
                **dict(counter),
                "both_exact_rate": round(
                    counter["both_exact"] / counter["units"], 6
                ),
            }
            for code, counter in sorted(by_standard.items())
        },
    }
    return report, disagreements
