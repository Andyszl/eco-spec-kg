from __future__ import annotations

import hashlib
from collections import defaultdict

from .models import Relation

EXTERNAL_TEST = {"HJ 1171-2021", "HJ 1174-2021", "HJ 1175-2021"}


def _bucket(path_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{path_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def group_split(
    relations: list[Relation], seed: int = 42
) -> dict[str, list[Relation]]:
    groups: dict[str, list[Relation]] = defaultdict(list)
    for relation in relations:
        groups[relation.path_id].append(relation)

    result: dict[str, list[Relation]] = {
        "train": [],
        "validation": [],
        "test": [],
        "external_test": [],
    }
    for path_id, items in groups.items():
        if any(item.evidence.standard_code in EXTERNAL_TEST for item in items):
            split = "external_test"
        else:
            value = _bucket(path_id, seed)
            split = "train" if value < 0.70 else "validation" if value < 0.85 else "test"
        result[split].extend(items)
    return result

