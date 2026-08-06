from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable

from .experiment_io_v2 import (
    assert_blind_records,
    read_json,
    read_jsonl,
    sha256_path,
    utc_now,
    write_json,
    write_jsonl,
)
from .io_utils import normalize_space
from .ontology_v2 import ONTOLOGY_VERSION


EVALUATION_SCHEMA_VERSION = "ecospec-strict-evaluation-v2.0"


def _name(value: Any) -> str:
    return normalize_space(str(value))


def _entity_key(unit_id: str, entity: dict[str, Any]) -> tuple[str, str, str]:
    return unit_id, _name(entity.get("name", "")), str(entity.get("entity_type", ""))


def _relation_key(
    unit_id: str,
    relation: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    return (
        unit_id,
        _name(relation.get("head_name", "")),
        str(relation.get("head_type", "")),
        str(relation.get("relation_type", "")),
        _name(relation.get("tail_name", "")),
        str(relation.get("tail_type", "")),
    )


def _prf(gold: set[Hashable], predicted: set[Hashable]) -> dict[str, Any]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / len(gold) if gold else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "gold_count": len(gold),
        "prediction_count": len(predicted),
        "true_positive": true_positive,
        "false_positive": len(predicted - gold),
        "false_negative": len(gold - predicted),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _group_metrics(
    values: list[str],
    gold: set[Hashable],
    predicted: set[Hashable],
    selector: Callable[[Hashable, str], bool],
) -> dict[str, dict[str, Any]]:
    return {
        value: _prf(
            {item for item in gold if selector(item, value)},
            {item for item in predicted if selector(item, value)},
        )
        for value in sorted(set(values))
    }


def _evidence_map(
    rows: list[dict[str, Any]],
    key_fn: Callable[[str, dict[str, Any]], Hashable],
    field: str,
) -> dict[Hashable, set[str]]:
    output: dict[Hashable, set[str]] = defaultdict(set)
    for row in rows:
        unit_id = row["unit_id"]
        for item in row.get(field, []):
            output[key_fn(unit_id, item)].update(
                str(value) for value in item.get("evidence_span_ids", [])
            )
    return dict(output)


def _evidence_metrics(
    gold: dict[Hashable, set[str]],
    predicted: dict[Hashable, set[str]],
) -> dict[str, Any]:
    matched = sorted(set(gold) & set(predicted), key=str)
    overlap = sum(bool(gold[key] & predicted[key]) for key in matched)
    exact = sum(gold[key] == predicted[key] for key in matched)
    return {
        "matched_semantic_count": len(matched),
        "overlap_count": overlap,
        "overlap_accuracy": round(overlap / len(matched), 6) if matched else None,
        "exact_count": exact,
        "exact_rate": round(exact / len(matched), 6) if matched else None,
    }


def _complete_groups(
    gold_relations: set[tuple[str, str, str, str, str, str]],
    predicted_relations: set[tuple[str, str, str, str, str, str]],
    relation_type: str,
    group_side: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], set[tuple[str, str, str, str, str, str]]] = defaultdict(set)
    for relation in gold_relations:
        if relation[3] != relation_type:
            continue
        if group_side == "head":
            key = (relation[0], relation[1], relation[2])
        else:
            key = (relation[0], relation[4], relation[5])
        grouped[key].add(relation)
    complete = sum(edges <= predicted_relations for edges in grouped.values())
    return {
        "applicable": len(grouped),
        "complete": complete,
        "rate": round(complete / len(grouped), 6) if grouped else None,
    }


def _edge_recall(
    gold_relations: set[tuple[str, str, str, str, str, str]],
    predicted_relations: set[tuple[str, str, str, str, str, str]],
    relation_type: str,
) -> dict[str, Any]:
    gold = {item for item in gold_relations if item[3] == relation_type}
    predicted = {item for item in predicted_relations if item[3] == relation_type}
    return _prf(gold, predicted)


def _indicator_formula_paths(
    gold_relations: set[tuple[str, str, str, str, str, str]],
    predicted_relations: set[tuple[str, str, str, str, str, str]],
) -> dict[str, Any]:
    paths: list[set[tuple[str, str, str, str, str, str]]] = []
    for calculated_by in gold_relations:
        if calculated_by[3] != "calculated_by":
            continue
        has_indicator = {
            relation
            for relation in gold_relations
            if relation[0] == calculated_by[0]
            and relation[3] == "has_indicator"
            and relation[4] == calculated_by[1]
            and relation[5] == calculated_by[2]
        }
        paths.append({calculated_by, *has_indicator})
    complete = sum(path <= predicted_relations for path in paths)
    return {
        "applicable": len(paths),
        "complete": complete,
        "rate": round(complete / len(paths), 6) if paths else None,
    }


def _find_package_manifest(path: Path) -> tuple[Path, dict[str, Any]] | None:
    resolved = path.resolve()
    for parent in [resolved.parent, *resolved.parents]:
        candidate = parent / "manifest.json"
        if candidate.exists():
            payload = read_json(candidate)
            if payload.get("schema_version") == "ecospec-experiment-dataset-v2.1":
                return candidate, payload
    return None


def _assert_manifest_file(
    manifest_path: Path,
    manifest: dict[str, Any],
    path: Path,
) -> None:
    expected = {
        item["path"]: item["sha256"] for item in manifest.get("files", [])
    }
    relative = path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    if relative not in expected:
        raise ValueError(f"file is not registered in dataset manifest: {relative}")
    actual = sha256_path(path)
    if actual != expected[relative]:
        raise ValueError(f"dataset file hash differs from manifest: {relative}")


def evaluate_v2(
    gold_path: Path,
    predictions_path: Path,
    units_path: Path,
    validation_report_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    units = read_jsonl(units_path)
    gold_rows = read_jsonl(gold_path)
    prediction_rows = read_jsonl(predictions_path)
    assert_blind_records(units)
    validation = read_json(validation_report_path)
    if not validation.get("passed"):
        raise ValueError("prediction validation report is not passed")
    expected_prediction_hash = validation.get("output", {}).get(
        "validated_predictions_sha256"
    )
    if expected_prediction_hash != sha256_path(predictions_path):
        raise ValueError("validated prediction hash does not match validation report")
    if validation.get("input", {}).get("units_sha256") != sha256_path(units_path):
        raise ValueError("blind unit hash does not match validation report")

    package = _find_package_manifest(units_path)
    gold_package = _find_package_manifest(gold_path)
    gold_nature = "unknown"
    dataset_version = "unknown"
    package_id = "unknown"
    if package is not None and gold_package is not None:
        if package[0].resolve() != gold_package[0].resolve():
            raise ValueError("blind units and gold annotations come from different packages")
        _assert_manifest_file(package[0], package[1], units_path)
        _assert_manifest_file(package[0], package[1], gold_path)
        gold_nature = package[1].get("gold_nature", "unknown")
        dataset_version = package[1].get("dataset_version", "unknown")
        package_id = package[1].get("package_id", "unknown")

    unit_by_id = {row["unit_id"]: row for row in units}
    gold_by_id = {row["unit_id"]: row for row in gold_rows}
    prediction_by_id = {row["unit_id"]: row for row in prediction_rows}
    if len(unit_by_id) != len(units):
        raise ValueError("blind unit ids are duplicated")
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("gold unit ids are duplicated")
    if len(prediction_by_id) != len(prediction_rows):
        raise ValueError("prediction unit ids are duplicated")
    if set(unit_by_id) != set(gold_by_id) or set(unit_by_id) != set(prediction_by_id):
        raise ValueError("blind, gold, and prediction unit ids are not identical")
    if any(
        row.get("ontology_version") != ONTOLOGY_VERSION for row in gold_rows
    ):
        raise ValueError("gold annotations use a different ontology version")

    gold_entities = {
        _entity_key(row["unit_id"], entity)
        for row in gold_rows
        for entity in row.get("entities", [])
    }
    predicted_entities = {
        _entity_key(row["unit_id"], entity)
        for row in prediction_rows
        for entity in row.get("entities", [])
    }
    gold_relations = {
        _relation_key(row["unit_id"], relation)
        for row in gold_rows
        for relation in row.get("relations", [])
    }
    predicted_relations = {
        _relation_key(row["unit_id"], relation)
        for row in prediction_rows
        for relation in row.get("relations", [])
    }

    relation_types = [item[3] for item in gold_relations | predicted_relations]
    entity_types = [item[2] for item in gold_entities | predicted_entities]
    by_relation_type = _group_metrics(
        relation_types,
        gold_relations,
        predicted_relations,
        lambda item, value: item[3] == value,
    )
    by_entity_type = _group_metrics(
        entity_types,
        gold_entities,
        predicted_entities,
        lambda item, value: item[2] == value,
    )
    macro_relation_f1 = (
        sum(item["f1"] for item in by_relation_type.values())
        / len(by_relation_type)
        if by_relation_type
        else 0.0
    )

    by_standard: dict[str, dict[str, Any]] = {}
    by_unit_type: dict[str, dict[str, Any]] = {}
    standards = sorted(
        {unit["provenance"]["standard_code"] for unit in units}
    )
    unit_types = sorted({unit["unit_type"] for unit in units})
    for label, values, target in (
        ("standard", standards, by_standard),
        ("unit_type", unit_types, by_unit_type),
    ):
        for value in values:
            ids = {
                unit_id
                for unit_id, unit in unit_by_id.items()
                if (
                    unit["provenance"]["standard_code"]
                    if label == "standard"
                    else unit["unit_type"]
                )
                == value
            }
            target[value] = {
                "entities": _prf(
                    {item for item in gold_entities if item[0] in ids},
                    {item for item in predicted_entities if item[0] in ids},
                ),
                "relations": _prf(
                    {item for item in gold_relations if item[0] in ids},
                    {item for item in predicted_relations if item[0] in ids},
                ),
                "unit_count": len(ids),
            }

    exact_entities = 0
    exact_relations = 0
    exact_both = 0
    for unit_id in unit_by_id:
        gold_unit_entities = {item for item in gold_entities if item[0] == unit_id}
        pred_unit_entities = {item for item in predicted_entities if item[0] == unit_id}
        gold_unit_relations = {item for item in gold_relations if item[0] == unit_id}
        pred_unit_relations = {item for item in predicted_relations if item[0] == unit_id}
        entity_exact = gold_unit_entities == pred_unit_entities
        relation_exact = gold_unit_relations == pred_unit_relations
        exact_entities += entity_exact
        exact_relations += relation_exact
        exact_both += entity_exact and relation_exact

    gold_entity_evidence = _evidence_map(gold_rows, _entity_key, "entities")
    pred_entity_evidence = _evidence_map(prediction_rows, _entity_key, "entities")
    gold_relation_evidence = _evidence_map(gold_rows, _relation_key, "relations")
    pred_relation_evidence = _evidence_map(prediction_rows, _relation_key, "relations")

    errors: list[dict[str, Any]] = []
    for error_type, values in (
        ("entity_false_negative", gold_entities - predicted_entities),
        ("entity_false_positive", predicted_entities - gold_entities),
        ("relation_false_negative", gold_relations - predicted_relations),
        ("relation_false_positive", predicted_relations - gold_relations),
    ):
        for value in sorted(values):
            errors.append({"error_type": error_type, "key": list(value)})

    relation_metrics = _prf(gold_relations, predicted_relations)
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "created_at": utc_now(),
        "dataset": {
            "package_id": package_id,
            "dataset_version": dataset_version,
            "gold_nature": gold_nature,
            "human_expert_review_required_for_publication": (
                gold_nature != "human_expert_gold"
            ),
            "unit_count": len(units),
        },
        "inputs": {
            "units_path": str(units_path),
            "units_sha256": sha256_path(units_path),
            "gold_path": str(gold_path),
            "gold_sha256": sha256_path(gold_path),
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_path(predictions_path),
            "validation_report_path": str(validation_report_path),
            "validation_report_sha256": sha256_path(validation_report_path),
        },
        "entities": {
            "micro": _prf(gold_entities, predicted_entities),
            "by_type": by_entity_type,
        },
        "relations": {
            "strict_micro": relation_metrics,
            "macro_relation_f1": round(macro_relation_f1, 6),
            "by_type": by_relation_type,
            "false_discovery_rate": round(
                relation_metrics["false_positive"]
                / relation_metrics["prediction_count"],
                6,
            )
            if relation_metrics["prediction_count"]
            else 0.0,
        },
        "evidence": {
            "entities": _evidence_metrics(gold_entity_evidence, pred_entity_evidence),
            "relations": _evidence_metrics(gold_relation_evidence, pred_relation_evidence),
            "all_predictions_traceable_to_blind_units": True,
        },
        "unit_exact_match": {
            "entity_exact_count": exact_entities,
            "relation_exact_count": exact_relations,
            "both_exact_count": exact_both,
            "both_exact_rate": round(exact_both / len(units), 6) if units else None,
        },
        "paths": {
            "formula_input_complete": _complete_groups(
                gold_relations, predicted_relations, "has_input", "head"
            ),
            "formula_output_complete": _complete_groups(
                gold_relations, predicted_relations, "has_output", "head"
            ),
            "explicit_unit_edges": _edge_recall(
                gold_relations, predicted_relations, "has_unit"
            ),
            "explicit_source_edges": _edge_recall(
                gold_relations, predicted_relations, "sourced_from"
            ),
            "indicator_formula_complete": _indicator_formula_paths(
                gold_relations, predicted_relations
            ),
            "quality_constraint_edges": _edge_recall(
                gold_relations, predicted_relations, "constrained_by"
            ),
        },
        "by_standard": by_standard,
        "by_unit_type": by_unit_type,
        "error_count": len(errors),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    errors_path = out_dir / "errors.jsonl"
    write_json(metrics_path, report)
    write_jsonl(errors_path, errors)
    markdown = f"""# EcoSpec-KG V2 严格评测报告

## 数据状态

- 数据版本：{dataset_version}
- 金标准性质：{gold_nature}
- 来源单元：{len(units)}
- 真实专家复核仍需完成：{gold_nature != 'human_expert_gold'}

## 核心指标

- 实体严格 F1：{report['entities']['micro']['f1']:.6f}
- 关系严格五元组 F1：{relation_metrics['f1']:.6f}
- 关系宏平均 F1：{report['relations']['macro_relation_f1']:.6f}
- 单元实体与关系完全一致率：{report['unit_exact_match']['both_exact_rate']:.6f}
- 匹配关系证据重叠正确率：{report['evidence']['relations']['overlap_accuracy']}

## 使用边界

本报告只对冻结金标准与已通过无金标验证的预测进行比较。当前金标准若为
`ai_expert_pre_gold`，指标只能用于流程验收和预实验，不能作为论文最终模型性能结论。
"""
    (out_dir / "evaluation_report.md").write_text(markdown, encoding="utf-8")
    return report


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate_v2"]
