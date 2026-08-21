from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment_io_v2 import sha256_path, utc_now
from .io_utils import read_jsonl, stable_id, write_json, write_jsonl
from .review_provenance_v2 import (
    AI_ADJUDICATED_GOLD,
    validate_review_provenance_v2,
)


ADJUDICATED_ANNOTATION_VERSION = "ecospec-annotation-v2.2"
SUPPORTED_ACTIONS = frozenset(
    {
        "add_gold_entity",
        "remove_gold_entity",
        "no_gold_change",
        "register_alias_only",
    }
)


def _span_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        span_id = value.get("span_id")
        if isinstance(span_id, str) and span_id:
            found.add(span_id)
        for child in value.values():
            found.update(_span_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_span_ids(child))
    return found


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return str(entity.get("name", "")), str(entity.get("entity_type", ""))


def _relation_key(relation: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(relation.get("head_name", "")),
        str(relation.get("head_type", "")),
        str(relation.get("relation_type", "")),
        str(relation.get("tail_name", "")),
        str(relation.get("tail_type", "")),
    )


def _validate_inventory(
    source_units: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    units_by_id = {row["unit_id"]: row for row in source_units}
    annotations_by_id = {row["unit_id"]: row for row in annotations}
    if len(units_by_id) != len(source_units):
        raise ValueError("source unit ids are not unique")
    if len(annotations_by_id) != len(annotations):
        raise ValueError("annotation unit ids are not unique")
    if set(units_by_id) != set(annotations_by_id):
        raise ValueError("source units and annotations do not have identical unit ids")
    return units_by_id, annotations_by_id


def _validate_decisions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("adjudication decisions are missing")
    adjudication = payload.get("adjudication", {})
    if adjudication.get("unresolved_count") != 0:
        raise ValueError("adjudication contains unresolved decisions")
    if adjudication.get("decision_count") != len(decisions):
        raise ValueError("adjudication decision_count does not match decisions")
    source_rows = [decision.get("source_row") for decision in decisions]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError("adjudication source_row values are not unique")
    for decision in decisions:
        if decision.get("status") != "adjudicated":
            raise ValueError(
                "all adjudication decisions must have status=adjudicated"
            )
        if decision.get("action") not in SUPPORTED_ACTIONS:
            raise ValueError(
                f"unsupported adjudication action: {decision.get('action')}"
            )
        if not decision.get("unit_id"):
            raise ValueError("adjudication decision is missing unit_id")
    return decisions


def _require_evidence(
    unit: dict[str, Any],
    evidence_span_ids: list[str],
    *,
    source_row: Any,
) -> None:
    if not evidence_span_ids:
        raise ValueError(f"decision row {source_row} has no evidence_span_ids")
    unknown = sorted(set(evidence_span_ids) - _span_ids(unit))
    if unknown:
        raise ValueError(
            f"decision row {source_row} references unknown evidence spans: {unknown}"
        )


def _add_entity(
    unit: dict[str, Any],
    annotation: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    entity = decision.get("entity")
    if not isinstance(entity, dict):
        raise ValueError("add_gold_entity requires a structured entity")
    key = _entity_key(entity)
    if not all(key):
        raise ValueError("add_gold_entity requires entity name and entity_type")
    if key in {_entity_key(item) for item in annotation.get("entities", [])}:
        raise ValueError(f"entity already exists for add_gold_entity: {key}")
    evidence_span_ids = list(entity.get("evidence_span_ids", []))
    _require_evidence(
        unit,
        evidence_span_ids,
        source_row=decision.get("source_row"),
    )
    annotation.setdefault("entities", []).append(
        {
            "entity_id": stable_id(
                "entity-v2",
                annotation["standard_code"],
                key[1],
                key[0],
                "",
            ),
            "name": key[0],
            "entity_type": key[1],
            "evidence_span_ids": evidence_span_ids,
        }
    )


def _remove_entity(
    annotation: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    entity = decision.get("entity")
    if not isinstance(entity, dict):
        raise ValueError("remove_gold_entity requires a structured entity")
    key = _entity_key(entity)
    matches = [
        item
        for item in annotation.get("entities", [])
        if _entity_key(item) == key
    ]
    if len(matches) != 1:
        raise ValueError(
            f"remove_gold_entity requires exactly one existing entity: {key}"
        )
    entity_id = matches[0].get("entity_id")
    linked = [
        relation
        for relation in annotation.get("relations", [])
        if relation.get("head_id") == entity_id
        or relation.get("tail_id") == entity_id
    ]
    if linked:
        raise ValueError(
            f"cannot remove entity referenced by relations without an explicit "
            f"relation decision: {key}"
        )
    annotation["entities"] = [
        item for item in annotation.get("entities", []) if _entity_key(item) != key
    ]


def _verify_relation(
    unit: dict[str, Any],
    annotation: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    relation = decision.get("relation")
    if not isinstance(relation, dict):
        raise ValueError("no_gold_change requires a structured relation")
    key = _relation_key(relation)
    if key not in {
        _relation_key(item) for item in annotation.get("relations", [])
    }:
        raise ValueError(f"relation selected for no_gold_change is missing: {key}")
    _require_evidence(
        unit,
        list(relation.get("evidence_span_ids", [])),
        source_row=decision.get("source_row"),
    )


def _register_alias(
    unit: dict[str, Any],
    annotation: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    alias = decision.get("alias")
    if not isinstance(alias, dict):
        raise ValueError("register_alias_only requires a structured alias")
    canonical_key = (
        str(alias.get("canonical_name", "")),
        str(alias.get("entity_type", "")),
    )
    alias_key = (str(alias.get("name", "")), canonical_key[1])
    existing = {_entity_key(item) for item in annotation.get("entities", [])}
    if canonical_key not in existing:
        raise ValueError(f"canonical entity for alias is missing: {canonical_key}")
    if alias_key in existing:
        raise ValueError(f"alias must not duplicate a strict gold entity: {alias_key}")
    evidence_span_ids = list(alias.get("evidence_span_ids", []))
    _require_evidence(
        unit,
        evidence_span_ids,
        source_row=decision.get("source_row"),
    )
    return {
        "unit_id": decision["unit_id"],
        "name": alias_key[0],
        "entity_type": alias_key[1],
        "canonical_name": canonical_key[0],
        "evidence_span_ids": evidence_span_ids,
        "strict_gold_changed": False,
    }


def apply_adjudication_v2(
    source_units_path: Path,
    annotations_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    provenance = validate_review_provenance_v2(
        AI_ADJUDICATED_GOLD,
        decisions_path,
    )
    assert provenance is not None
    decisions = _validate_decisions(provenance)
    source_units = read_jsonl(source_units_path)
    original_annotations = read_jsonl(annotations_path)
    if not source_units or not original_annotations:
        raise ValueError("source units and annotations must not be empty")
    units_by_id, _ = _validate_inventory(source_units, original_annotations)
    annotations = deepcopy(original_annotations)
    annotations_by_id = {row["unit_id"]: row for row in annotations}

    changed_unit_ids: set[str] = set()
    aliases: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    adjudicator_id = str(provenance["adjudication"]["adjudicator_id"])
    for decision in decisions:
        unit_id = str(decision["unit_id"])
        if unit_id not in units_by_id:
            raise ValueError(f"decision references unknown unit_id: {unit_id}")
        unit = units_by_id[unit_id]
        annotation = annotations_by_id[unit_id]
        if annotation.get("standard_code") != unit["provenance"]["standard_code"]:
            raise ValueError(f"standard_code mismatch for unit_id: {unit_id}")
        action = str(decision["action"])
        if action == "add_gold_entity":
            _add_entity(unit, annotation, decision)
            changed_unit_ids.add(unit_id)
        elif action == "remove_gold_entity":
            _remove_entity(annotation, decision)
            changed_unit_ids.add(unit_id)
        elif action == "no_gold_change":
            _verify_relation(unit, annotation, decision)
        elif action == "register_alias_only":
            aliases.append(_register_alias(unit, annotation, decision))
        action_counts[action] += 1

    note = f"AI adjudication applied by {adjudicator_id}"
    for unit_id in changed_unit_ids:
        annotation = annotations_by_id[unit_id]
        annotation["annotation_version"] = ADJUDICATED_ANNOTATION_VERSION
        annotation["review_status"] = "ai_expert_adjudicated"
        annotation["annotator_id"] = adjudicator_id
        existing_note = str(annotation.get("notes", "")).strip()
        annotation["notes"] = f"{existing_note}; {note}".strip("; ")
        annotation["entities"].sort(
            key=lambda item: (
                item.get("entity_type", ""),
                item.get("name", ""),
                item.get("entity_id", ""),
            )
        )

    original_by_id = {row["unit_id"]: row for row in original_annotations}
    actual_changed = {
        row["unit_id"]
        for row in annotations
        if row != original_by_id[row["unit_id"]]
    }
    if actual_changed != changed_unit_ids:
        raise AssertionError(
            "adjudication changed an unexpected unit set: "
            f"expected={sorted(changed_unit_ids)} actual={sorted(actual_changed)}"
        )

    before_entities = sum(len(row.get("entities", [])) for row in original_annotations)
    after_entities = sum(len(row.get("entities", [])) for row in annotations)
    before_relations = sum(
        len(row.get("relations", [])) for row in original_annotations
    )
    after_relations = sum(len(row.get("relations", [])) for row in annotations)
    write_jsonl(output_path, annotations)
    aliases_path = output_path.with_name(f"{output_path.stem}.aliases.json")
    write_json(
        aliases_path,
        {
            "schema_version": "ecospec-alias-registry-v2.0",
            "strict_gold_unchanged": True,
            "aliases": aliases,
        },
    )
    manifest_path = output_path.with_name(
        f"{output_path.stem}.adjudication_manifest.json"
    )
    manifest = {
        "schema_version": "ecospec-adjudication-application-v2.0",
        "status": "complete",
        "created_at": utc_now(),
        "gold_nature": AI_ADJUDICATED_GOLD,
        "adjudicator_id": adjudicator_id,
        "inputs": {
            "source_units": str(source_units_path.resolve()),
            "source_units_sha256": sha256_path(source_units_path),
            "annotations": str(annotations_path.resolve()),
            "annotations_sha256": sha256_path(annotations_path),
            "decisions": str(decisions_path.resolve()),
            "decisions_sha256": sha256_path(decisions_path),
        },
        "outputs": {
            "annotations": str(output_path.resolve()),
            "annotations_sha256": sha256_path(output_path),
            "aliases": str(aliases_path.resolve()),
            "aliases_sha256": sha256_path(aliases_path),
        },
        "decision_count": len(decisions),
        "action_counts": dict(sorted(action_counts.items())),
        "modified_unit_count": len(changed_unit_ids),
        "modified_unit_ids": sorted(changed_unit_ids),
        "alias_count": len(aliases),
        "before": {
            "unit_count": len(original_annotations),
            "entity_count": before_entities,
            "relation_count": before_relations,
        },
        "after": {
            "unit_count": len(annotations),
            "entity_count": after_entities,
            "relation_count": after_relations,
        },
        "delta": {
            "unit_count": len(annotations) - len(original_annotations),
            "entity_count": after_entities - before_entities,
            "relation_count": after_relations - before_relations,
        },
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path.resolve())}


__all__ = [
    "ADJUDICATED_ANNOTATION_VERSION",
    "SUPPORTED_ACTIONS",
    "apply_adjudication_v2",
]
