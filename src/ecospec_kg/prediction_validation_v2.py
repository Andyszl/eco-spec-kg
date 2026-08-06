from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .experiment_io_v2 import (
    assert_blind_records,
    read_json,
    read_jsonl,
    sha256_path,
    utc_now,
    write_json,
    write_jsonl,
)
from .ontology_v2 import (
    EntityTypeV2,
    ONTOLOGY_VERSION,
    RelationTypeV2,
    validate_relation_v2,
)
from .prediction_contract_v2 import PREDICTION_SCHEMA_VERSION


VALIDATION_SCHEMA_VERSION = "ecospec-prediction-validation-v2.0"
PREDICTION_FORBIDDEN_KEYS = frozenset(
    {"gold_annotation", "gold_nature", "review_status", "triple_id"}
)


def _all_spans(value: Any) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        if all(key in value for key in ("span_id", "page", "bbox")):
            spans[str(value["span_id"])] = value
        for child in value.values():
            spans.update(_all_spans(child))
    elif isinstance(value, list):
        for child in value:
            spans.update(_all_spans(child))
    return spans


def _forbidden_prediction_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if key in PREDICTION_FORBIDDEN_KEYS:
                failures.append(path)
            failures.extend(_forbidden_prediction_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(_forbidden_prediction_paths(child, f"{prefix}[{index}]"))
    return failures


def _failure(
    unit_id: str,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "code": code,
        "message": message,
        **details,
    }


def _validate_unit_prediction(
    unit: dict[str, Any],
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    unit_id = unit["unit_id"]
    failures: list[dict[str, Any]] = []
    valid_spans = set(_all_spans(unit))
    valid_entity_types = {item.value for item in EntityTypeV2}
    valid_relation_types = {item.value for item in RelationTypeV2}

    for path in _forbidden_prediction_paths(prediction):
        failures.append(
            _failure(unit_id, "gold_field_leakage", "prediction contains a gold-only field", path=path)
        )
    if prediction.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        failures.append(
            _failure(
                unit_id,
                "prediction_schema_version",
                "prediction schema version does not match",
                actual=prediction.get("schema_version"),
                expected=PREDICTION_SCHEMA_VERSION,
            )
        )
    if prediction.get("ontology_version") != ONTOLOGY_VERSION:
        failures.append(
            _failure(
                unit_id,
                "ontology_version",
                "ontology version does not match",
                actual=prediction.get("ontology_version"),
                expected=ONTOLOGY_VERSION,
            )
        )
    if prediction.get("status") != "success":
        failures.append(
            _failure(
                unit_id,
                "extraction_status",
                "extraction did not complete successfully",
                actual=prediction.get("status"),
            )
        )
    reproducibility_fields = {
        "experiment_id": prediction.get("experiment_id"),
        "config_hash": prediction.get("config_hash"),
        "candidate_hash": prediction.get("candidate_hash"),
        "model": prediction.get("model"),
        "seed": prediction.get("seed"),
    }
    for field, value in reproducibility_fields.items():
        if value is None or value == "":
            failures.append(
                _failure(
                    unit_id,
                    "missing_reproducibility_field",
                    "prediction is missing reproducibility metadata",
                    field=field,
                )
            )
    if prediction.get("backend") == "llm" and not prediction.get(
        "prompt_hash"
    ):
        failures.append(
            _failure(
                unit_id,
                "missing_prompt_hash",
                "LLM prediction is missing its prompt hash",
            )
        )
    if prediction.get("standard_code") != unit["provenance"]["standard_code"]:
        failures.append(
            _failure(unit_id, "standard_code", "prediction standard does not match source unit")
        )
    if prediction.get("unit_type") != unit["unit_type"]:
        failures.append(
            _failure(unit_id, "unit_type", "prediction unit type does not match source unit")
        )

    entities = prediction.get("entities")
    relations = prediction.get("relations")
    if not isinstance(entities, list):
        failures.append(_failure(unit_id, "entities_type", "entities must be a list"))
        entities = []
    if not isinstance(relations, list):
        failures.append(_failure(unit_id, "relations_type", "relations must be a list"))
        relations = []

    entity_by_id: dict[str, dict[str, Any]] = {}
    entity_keys: set[tuple[str, str]] = set()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            failures.append(
                _failure(unit_id, "entity_record", "entity must be an object", index=index)
            )
            continue
        entity_id = str(entity.get("entity_id", ""))
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("entity_type", ""))
        spans = entity.get("evidence_span_ids")
        if not entity_id or entity_id in entity_by_id:
            failures.append(
                _failure(unit_id, "duplicate_entity_id", "entity id is missing or duplicated", index=index)
            )
        else:
            entity_by_id[entity_id] = entity
        key = (name, entity_type)
        if not name:
            failures.append(_failure(unit_id, "empty_entity_name", "entity name is empty", index=index))
        if entity_type not in valid_entity_types:
            failures.append(
                _failure(
                    unit_id,
                    "invalid_entity_type",
                    "entity type is not defined by Schema V2",
                    index=index,
                    actual=entity_type,
                )
            )
        if key in entity_keys:
            failures.append(
                _failure(unit_id, "duplicate_entity", "semantic entity is duplicated", index=index)
            )
        entity_keys.add(key)
        if not isinstance(spans, list) or not spans:
            failures.append(
                _failure(unit_id, "missing_entity_evidence", "entity has no evidence spans", index=index)
            )
        elif not set(map(str, spans)) <= valid_spans:
            failures.append(
                _failure(
                    unit_id,
                    "invalid_entity_evidence",
                    "entity references evidence outside the source unit",
                    index=index,
                    invalid=sorted(set(map(str, spans)) - valid_spans),
                )
            )

    relation_ids: set[str] = set()
    relation_keys: set[tuple[str, str, str, str, str]] = set()
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            failures.append(
                _failure(unit_id, "relation_record", "relation must be an object", index=index)
            )
            continue
        relation_id = str(relation.get("relation_id", ""))
        if not relation_id or relation_id in relation_ids:
            failures.append(
                _failure(unit_id, "duplicate_relation_id", "relation id is missing or duplicated", index=index)
            )
        relation_ids.add(relation_id)
        head_id = str(relation.get("head_id", ""))
        tail_id = str(relation.get("tail_id", ""))
        head = entity_by_id.get(head_id)
        tail = entity_by_id.get(tail_id)
        if head is None or tail is None:
            failures.append(
                _failure(
                    unit_id,
                    "missing_relation_endpoint",
                    "relation endpoint is not present in predicted entities",
                    index=index,
                )
            )
        else:
            expected = (
                head["name"],
                head["entity_type"],
                tail["name"],
                tail["entity_type"],
            )
            actual = (
                relation.get("head_name"),
                relation.get("head_type"),
                relation.get("tail_name"),
                relation.get("tail_type"),
            )
            if expected != actual:
                failures.append(
                    _failure(
                        unit_id,
                        "endpoint_payload_mismatch",
                        "relation endpoint payload differs from entity records",
                        index=index,
                    )
                )
        relation_type = str(relation.get("relation_type", ""))
        if relation_type not in valid_relation_types:
            failures.append(
                _failure(
                    unit_id,
                    "invalid_relation_type",
                    "relation type is not defined by Schema V2",
                    index=index,
                    actual=relation_type,
                )
            )
        else:
            valid, reason = validate_relation_v2(
                str(relation.get("head_type", "")),
                relation_type,
                str(relation.get("tail_type", "")),
            )
            if not valid:
                failures.append(
                    _failure(
                        unit_id,
                        "invalid_relation_direction",
                        "relation endpoint types violate Schema V2",
                        index=index,
                        reason=reason,
                    )
                )
        key = (
            str(relation.get("head_name", "")),
            str(relation.get("head_type", "")),
            relation_type,
            str(relation.get("tail_name", "")),
            str(relation.get("tail_type", "")),
        )
        if key in relation_keys:
            failures.append(
                _failure(unit_id, "duplicate_relation", "semantic relation is duplicated", index=index)
            )
        relation_keys.add(key)
        spans = relation.get("evidence_span_ids")
        if not isinstance(spans, list) or not spans:
            failures.append(
                _failure(unit_id, "missing_relation_evidence", "relation has no evidence spans", index=index)
            )
        elif not set(map(str, spans)) <= valid_spans:
            failures.append(
                _failure(
                    unit_id,
                    "invalid_relation_evidence",
                    "relation references evidence outside the source unit",
                    index=index,
                    invalid=sorted(set(map(str, spans)) - valid_spans),
                )
            )
    return failures


def validate_predictions_v2(
    units_path: Path,
    predictions_path: Path,
    schema_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    units = read_jsonl(units_path)
    predictions = read_jsonl(predictions_path)
    assert_blind_records(units)
    schema = read_json(schema_path)
    global_failures: list[dict[str, Any]] = []
    run_manifest_path = predictions_path.parent / "run_manifest.json"
    run_manifest_verified = False
    if run_manifest_path.exists():
        run_manifest = read_json(run_manifest_path)
        expected_units = run_manifest.get("input", {}).get("sha256")
        expected_predictions = run_manifest.get("predictions", {}).get(
            "sha256"
        )
        if expected_units != sha256_path(units_path):
            global_failures.append(
                {
                    "unit_id": "",
                    "code": "run_manifest_units_hash",
                    "message": "blind unit hash differs from extraction run manifest",
                }
            )
        if expected_predictions != sha256_path(predictions_path):
            global_failures.append(
                {
                    "unit_id": "",
                    "code": "run_manifest_predictions_hash",
                    "message": "prediction hash differs from extraction run manifest",
                }
            )
        run_manifest_verified = not any(
            item["code"].startswith("run_manifest_")
            for item in global_failures
        )
    if schema.get("ontology_version") != ONTOLOGY_VERSION:
        global_failures.append(
            {
                "unit_id": "",
                "code": "schema_ontology_version",
                "message": "schema file ontology version does not match the validator",
                "actual": schema.get("ontology_version"),
                "expected": ONTOLOGY_VERSION,
            }
        )

    unit_by_id = {str(unit.get("unit_id", "")): unit for unit in units}
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        prediction_groups[str(prediction.get("unit_id", ""))].append(prediction)
    if len(unit_by_id) != len(units):
        global_failures.append(
            {"unit_id": "", "code": "duplicate_source_units", "message": "source unit ids are duplicated"}
        )
    missing = sorted(set(unit_by_id) - set(prediction_groups))
    extra = sorted(set(prediction_groups) - set(unit_by_id))
    duplicates = sorted(
        unit_id for unit_id, rows in prediction_groups.items() if len(rows) != 1
    )
    for code, values in (
        ("missing_prediction_units", missing),
        ("extra_prediction_units", extra),
        ("duplicate_prediction_units", duplicates),
    ):
        if values:
            global_failures.append(
                {"unit_id": "", "code": code, "message": code.replace("_", " "), "values": values[:100]}
            )

    failures = list(global_failures)
    valid_predictions: list[dict[str, Any]] = []
    unit_failure_counts: Counter[str] = Counter()
    for unit_id, unit in unit_by_id.items():
        rows = prediction_groups.get(unit_id, [])
        if len(rows) != 1:
            continue
        unit_failures = _validate_unit_prediction(unit, rows[0])
        failures.extend(unit_failures)
        if unit_failures:
            unit_failure_counts[unit_id] += len(unit_failures)
        else:
            valid_predictions.append(rows[0])

    valid_predictions.sort(key=lambda row: row["unit_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    validated_path = out_dir / "validated_predictions.jsonl"
    failures_path = out_dir / "validation_failures.jsonl"
    write_jsonl(validated_path, valid_predictions)
    write_jsonl(failures_path, failures)
    passed = not failures and len(valid_predictions) == len(units)
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "created_at": utc_now(),
        "passed": passed,
        "input": {
            "units_path": str(units_path),
            "units_sha256": sha256_path(units_path),
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_path(predictions_path),
            "schema_path": str(schema_path),
            "schema_sha256": sha256_path(schema_path),
        },
        "output": {
            "validated_predictions_path": str(validated_path),
            "validated_predictions_sha256": sha256_path(validated_path),
            "failures_path": str(failures_path),
            "failures_sha256": sha256_path(failures_path),
        },
        "source_unit_count": len(units),
        "prediction_unit_count": len(predictions),
        "valid_prediction_unit_count": len(valid_predictions),
        "failure_count": len(failures),
        "failed_unit_count": len(unit_failure_counts),
        "run_manifest": {
            "path": str(run_manifest_path) if run_manifest_path.exists() else None,
            "verified": run_manifest_verified,
        },
        "failure_code_counts": dict(
            sorted(Counter(item["code"] for item in failures).items())
        ),
    }
    write_json(out_dir / "validation_report.json", report)
    return report


__all__ = [
    "PREDICTION_FORBIDDEN_KEYS",
    "VALIDATION_SCHEMA_VERSION",
    "validate_predictions_v2",
]
