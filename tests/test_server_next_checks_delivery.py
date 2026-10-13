"""Check the actual Git-delivered entry point without Swift or model loading."""
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/next_checks_20261013.py"


def entrypoint(source):
    module = ast.parse(source)
    block = module.body[-1]
    assert isinstance(block, ast.If)
    return compile(ast.Module(body=block.body, type_ignores=[]), str(SCRIPT), "exec")


def get_payload(source):
    received = []
    exec(entrypoint(source), {"hashlib": hashlib, "json": json, "main": received.append})
    assert len(received) == 1
    return received[0]


def test_actual_entrypoint_retains_parent_gold_and_valid_selection_ids():
    payload = get_payload(SCRIPT.read_text(encoding="utf-8"))
    assert set(payload) == {"source_units.jsonl", "derived_train_annotations.jsonl",
                            "qwen35_replacements.jsonl", "derivation.json"}
    rows = lambda text: [json.loads(line) for line in text.splitlines() if line.strip()]
    derivation = json.loads(payload["derivation.json"])
    parent = derivation["parent_unit_id"]
    original = next(g for g in rows((ROOT / "deliveries/human_confirmed_20261002/gold/train_annotations.jsonl").read_text(encoding="utf-8")) if g["unit_id"] == parent)
    gold = rows(payload["derived_train_annotations.jsonl"])
    keys = lambda gs: {tuple(r[k] for k in ("head_name", "head_type", "relation_type", "tail_name", "tail_type")) for g in gs for r in g["relations"]}
    assert keys(gold) == keys([original])
    assert len(keys(gold)) == 34
    units = rows(payload["source_units.jsonl"])
    assert len(units) == len(gold) == 3
    ids = {u["unit_id"] for u in units}
    assert len(ids) == 3 and parent not in ids
    assert {g["unit_id"] for g in gold} == ids
    assert all(g["split"] == "train" and g["derived_from_unit_id"] == parent for g in gold)
    for row in rows(payload["qwen35_replacements.jsonl"]):
        messages = row["messages"]
        prompt, target = (json.loads(messages[i]["content"]) for i in (1, 2))
        assert prompt["source_unit"]["unit_id"] in ids
        assert set(target["selected_relation_ids"]) <= {r["id"] for r in prompt["candidate_relations"]}
        assert set(target["selected_entity_ids"]) <= {e["id"] for e in prompt["candidate_entities"]}


def test_payload_checksum_failure_stops_before_main():
    source = SCRIPT.read_text(encoding="utf-8")
    marker = 'assert hashlib.sha256(data).hexdigest() == "'
    start = source.index(marker) + len(marker)
    corrupted = source[:start] + "0" * 64 + source[start + 64:]
    received = []
    with pytest.raises(AssertionError, match="Embedded payload mismatch"):
        exec(entrypoint(corrupted), {"hashlib": hashlib, "json": json, "main": received.append})
    assert not received


def test_import_has_no_subprocess_or_output_side_effects():
    code = compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")
    with patch("subprocess.check_output", side_effect=AssertionError("Unexpected subprocess")), patch("pathlib.Path.write_text", side_effect=AssertionError("Unexpected write")):
        namespace = {"__name__": "delivery_import"}
        exec(code, namespace)
    assert callable(namespace["run"]) and callable(namespace["main"])
