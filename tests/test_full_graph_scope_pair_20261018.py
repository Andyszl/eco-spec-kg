"""Reconstructed targets and actual length gates must fail closed."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
from types import SimpleNamespace
import pytest
import full_graph_scope_pair_20261018 as pair
from ecospec_kg.selection_decoding_v2 import candidate_ids
from check_scope_inputs_20261013 import encode_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    folder = tmp_path_factory.mktemp("scoped") / "inputs"
    manifest = pair.build_inputs(ROOT, folder)
    return folder, manifest


def test_complete_graph_targets_split_and_pair_symmetry(prepared):
    folder, m = prepared
    assert m["counts"] == {"control_train": 626, "repair_train": 626, "validation": 72}
    assert m["unscored_counts"] == {"train": 16, "validation": 5}
    assert m["paired_changed_training_targets"] == 6
    assert not m["training_ready"] and not m["model_training_started"]
    arms = {n: pair.rows(folder / (n + ".jsonl")) for n in m["counts"]}
    for n, records in arms.items():
        old = pair.rows(ROOT / pair.OLD / (n + ".jsonl"))
        assert [r["messages"][-1] for r in records] == [r["messages"][-1] for r in old]
        for row in records:
            q = json.loads(row["messages"][1]["content"])
            assert len(candidate_ids(row["messages"][1]["content"])[1]) == m["graph_sizes"][q["source_unit"]["unit_id"]]["relations"]
    assert all(a["messages"][:2] == b["messages"][:2] for a, b in zip(arms["control_train"], arms["repair_train"]))
    assert set(m["unit_ids"]["repair_train"]).isdisjoint(m["unit_ids"]["validation"])
    pair.verify_inputs(ROOT, folder)


def test_tampered_target_rejected_even_if_file_hash_is_updated(prepared, tmp_path):
    src, _ = prepared
    out = tmp_path / "bad"
    shutil.copytree(src, out)
    p = out / "repair_train.jsonl"
    rows = pair.rows(p)
    # Always alter a real selected target, rather than potentially empty row 0.
    row = next(r for r in rows if json.loads(r["messages"][-1]["content"])["selected_relation_ids"])
    target = json.loads(row["messages"][-1]["content"])
    target["selected_relation_ids"].pop()
    row["messages"][-1]["content"] = json.dumps(target, ensure_ascii=False, sort_keys=True)
    p.write_text(pair.dumps(rows), encoding="utf-8", newline="\n")
    m = pair.read(out / "input_manifest.json")
    m["files"][p.name] = pair.sha(p)
    pair.save(out / "input_manifest.json", m)
    with pytest.raises(AssertionError, match="Projection"):
        pair.verify_inputs(ROOT, out)


def test_real_report_gate_rejects_stale_oversize_empty_and_extra_encoding(prepared):
    folder, m = prepared
    fixture = {
        "passed": True, "real_tokenizer": True, "checked_records": 1324, "failures": [],
        "input_manifest_sha256": pair.sha(folder / "input_manifest.json"), "input_manifest": m,
        "source_sha256_lf": m["source_sha256_lf"], "swift_version": "4.4.2",
        "model_weights_loaded": False, "model_training_started": False,
        "original_inputs_unchanged": True, "training_ready": False,
        "template_contract": pair.TEMPLATE_CONTRACT,
        "encodings": [{"file": n, "unit_id": i, "tokens": 64, "supervised_tokens": 10}
                      for n, ids in m["unit_ids"].items() for i in ids],
    }
    # Only a metadata validator fixture: this does not claim real tokenization.
    pair.verify_preflight(ROOT, folder, fixture)
    changes = (
        lambda r: r.update(input_manifest_sha256="old-report"),
        lambda r: r["encodings"][0].update(tokens=8193),
        lambda r: r["encodings"][0].update(supervised_tokens=0),
        lambda r: r["encodings"].append(dict(r["encodings"][0])),
        lambda r: r.update(model_weights_loaded=True),
        lambda r: r.update(template_contract={"max_length": 16384}),
    )
    for change in changes:
        bad = deepcopy(fixture)
        change(bad)
        with pytest.raises(AssertionError):
            pair.verify_preflight(ROOT, folder, bad)


def test_actual_encoding_failure_is_recorded_without_truncation(prepared):
    folder, _ = prepared
    row = pair.rows(folder / "repair_train.jsonl")[0]
    class OversizedTemplate:
        def encode(self, row):
            return {"input_ids": [1] * 8193, "labels": [-100] * 8192 + [1]}
    report = encode_files(OversizedTemplate(), {"repair_train": [row]})
    assert not report["passed"] and report["checked_records"] == 1
    assert not report["encodings"] and len(report["failures"]) == 1


def test_cpu_preflight_wires_no_weights_and_strict_real_template_options(prepared, tmp_path, monkeypatch):
    folder, m = prepared
    checked = {"manifest_sha256": pair.sha(folder / "input_manifest.json"), "manifest": m}
    actual_rows, actual_sha = pair.rows, pair.sha
    model = tmp_path / "model"
    calls = []
    def processor(path, **kwargs):
        calls.append((path, kwargs))
        assert kwargs == {"load_model": False, "download_model": False}
        return None, "mock-processor"
    class Template:
        template_meta = SimpleNamespace(template_type="qwen3_5")
        def set_mode(self, mode):
            assert mode == "train"
        def encode(self, row):
            return {"input_ids": [1] * 64, "labels": [-100] * 54 + [1] * 10}
    def template(value, **kwargs):
        assert value == "mock-processor"
        assert kwargs == {k: v for k, v in pair.TEMPLATE_CONTRACT.items() if k not in ("template_type", "mode")}
        return Template()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(pair, "verify_inputs", lambda *args: checked)
    monkeypatch.setattr(pair.importlib.metadata, "version", lambda name: "4.4.2")
    monkeypatch.setattr(pair, "sha", lambda p: m["tokenizer_config_sha256"][p.name] if p.parent == model else actual_sha(p))
    monkeypatch.setattr(pair, "rows", lambda p: actual_rows(folder / p.name) if p.parent == ROOT / pair.FOLDER else actual_rows(p))
    monkeypatch.setitem(sys.modules, "swift.model", SimpleNamespace(get_model_processor=processor))
    monkeypatch.setitem(sys.modules, "swift.template", SimpleNamespace(get_template=template))
    report = pair.preflight(ROOT, model, tmp_path / "mock-report")
    assert report["passed"] and len(calls) == 1 and report["checked_records"] == 1324
    assert report["model_weights_loaded"] is False and report["training_ready"] is False
    # This is a mocked wiring test; its temporary output is never an approval report.


def test_preflight_refuses_visible_gpu_before_any_model_import(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(AssertionError, match="hide GPUs"):
        pair.preflight(ROOT, tmp_path, tmp_path / "out")
