import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_human_confirmed_20261002 import check_delivery
from ecospec_kg.experiment_io_v2 import sha256_path
from ecospec_kg.extractor_v2 import CANDIDATE_GENERATOR_VERSION


def delivery(tmp_path, label_split="train"):
    folder = tmp_path / "delivery"
    source = {"unit_id": "u", "unit_type": "procedure_clause",
              "provenance": {"standard_code": "HJ 1167-2021",
                             "evidence_spans": [{"span_id": "s", "text": "叶面积指数"}]}}
    label = {"unit_id": "u", "split": label_split,
             "annotator_id": "human_confirmed_review", "entities": [], "relations": []}
    payloads = {"blind/train_units.jsonl": [source], "gold/train_annotations.jsonl": [label],
                "blind/dev_units.jsonl": [], "gold/dev_annotations.jsonl": []}
    records = []
    for name, rows in payloads.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        records.append({"path": name, "sha256": sha256_path(path)})
    for name in ("schema_v2.json", "expected.json"):
        path = folder / name
        path.write_text("{}", encoding="utf-8")
        records.append({"path": name, "sha256": sha256_path(path)})
    (folder / "manifest.json").write_text(json.dumps({
        "gold_nature": "human_expert_gold", "review_method": "ai_assisted_human_confirmed",
        "candidate_generator": CANDIDATE_GENERATOR_VERSION,
        "unit_split_counts": {"train": 1, "dev": 0}, "files": records}), encoding="utf-8")
    return folder


def test_delivery_rejects_modified_bytes(tmp_path):
    folder = delivery(tmp_path)
    check_delivery(folder)
    with (folder / "blind/train_units.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        check_delivery(folder)


def test_delivery_rejects_test_annotations_even_with_valid_hash(tmp_path):
    folder = delivery(tmp_path, label_split="test")
    with pytest.raises(ValueError, match="split"):
        check_delivery(folder)


def test_delivery_rejects_extra_test_file(tmp_path):
    folder = delivery(tmp_path)
    (folder / "gold/test_annotations.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        check_delivery(folder)
