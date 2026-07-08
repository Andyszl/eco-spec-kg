from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .evidence import validate_evidence
from .io_utils import read_jsonl, write_json
from .models import DocumentChunk, Relation

BASELINES = [
    "rule",
    "zero_shot",
    "few_shot",
    "vector_rag",
    "microsoft_graphrag",
    "native_graphrag",
    "schema_graphrag",
    "full_method",
]


def relation_key(relation: Relation) -> tuple[str, str, str]:
    return (
        relation.head_name.strip(),
        relation.relation_type,
        relation.tail_name.strip(),
    )


def extraction_metrics(
    gold: list[Relation], predictions: list[Relation]
) -> dict[str, float | int]:
    gold_keys = {relation_key(item) for item in gold}
    pred_keys = {relation_key(item) for item in predictions}
    true_positive = len(gold_keys & pred_keys)
    precision = true_positive / len(pred_keys) if pred_keys else 0.0
    recall = true_positive / len(gold_keys) if gold_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_count": len(gold_keys),
        "prediction_count": len(pred_keys),
        "true_positive": true_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def macro_relation_f1(
    gold: list[Relation], predictions: list[Relation]
) -> float:
    types = sorted({item.relation_type for item in gold + predictions})
    if not types:
        return 0.0
    scores = []
    for relation_type in types:
        subset_gold = [item for item in gold if item.relation_type == relation_type]
        subset_pred = [
            item for item in predictions if item.relation_type == relation_type
        ]
        scores.append(float(extraction_metrics(subset_gold, subset_pred)["f1"]))
    return round(sum(scores) / len(scores), 6)


def ranking_metrics(rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 10)) -> dict[str, float]:
    ranks = [int(row["rank"]) for row in rows if row.get("is_correct") is True]
    if not ranks:
        return {"mrr": 0.0, **{f"hits@{k}": 0.0 for k in ks}}
    metrics = {"mrr": round(sum(1 / rank for rank in ranks) / len(ranks), 6)}
    metrics.update(
        {
            f"hits@{k}": round(sum(rank <= k for rank in ranks) / len(ranks), 6)
            for k in ks
        }
    )
    return metrics


def provenance_metrics(
    predictions: list[Relation], chunks: list[DocumentChunk]
) -> dict[str, float | int]:
    chunk_map = {item.chunk_id: item for item in chunks}
    accepted = 0
    for relation in predictions:
        chunk = chunk_map.get(relation.evidence.chunk_id)
        if chunk and validate_evidence(relation, chunk)[0]:
            accepted += 1
    total = len(predictions)
    return {
        "traceable_count": accepted,
        "prediction_count": total,
        "provenance_accuracy": round(accepted / total, 6) if total else 0.0,
        "unsupported_rate": round((total - accepted) / total, 6) if total else 0.0,
    }


def evaluate_files(
    gold_path: Path, prediction_path: Path, chunk_path: Path | None = None
) -> dict[str, Any]:
    gold = [Relation.from_dict(row) for row in read_jsonl(gold_path)]
    predictions = [Relation.from_dict(row) for row in read_jsonl(prediction_path)]
    result: dict[str, Any] = {
        "micro": extraction_metrics(gold, predictions),
        "macro_relation_f1": macro_relation_f1(gold, predictions),
    }
    if chunk_path:
        chunks = [DocumentChunk.from_dict(row) for row in read_jsonl(chunk_path)]
        result["provenance"] = provenance_metrics(predictions, chunks)
    return result


def ablation_matrix() -> list[dict[str, Any]]:
    fixed = [
        ("full_method", {}),
        ("without_schema", {"schema": False}),
        ("without_graphrag", {"graphrag": False}),
        ("without_lora", {"lora": False}),
        ("without_few_shot", {"few_shot": False}),
        ("without_evidence", {"evidence_validation": False}),
    ]
    rows = [
        {"experiment_id": name, "overrides": overrides, "metrics": None}
        for name, overrides in fixed
    ]
    rows.extend(
        {
            "experiment_id": f"chunk_{size}",
            "overrides": {"chunk_tokens": size},
            "metrics": None,
        }
        for size in (150, 300, 600)
    )
    rows.extend(
        {
            "experiment_id": f"shot_{shots}",
            "overrides": {"shots": shots},
            "metrics": None,
        }
        for shots in (0, 1, 3, 5)
    )
    rows.extend(
        {
            "experiment_id": f"threshold_{threshold}",
            "overrides": {"candidate_threshold": threshold},
            "metrics": None,
        }
        for threshold in (0.65, 0.75, 0.85)
    )
    return rows


def class_distribution(relations: list[Relation]) -> dict[str, int]:
    return dict(sorted(Counter(item.relation_type for item in relations).items()))


def write_ablation_plan(path: Path) -> None:
    write_json(path, {"status": "not_run", "experiments": ablation_matrix()})

