"""The CPU check must retain every row and report failed supervision explicitly."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import json
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "tools/check_scope_inputs_20261013.py"
    spec = importlib.util.spec_from_file_location("scope_preflight", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(name):
    return {"messages": [{"role": "user", "content": '{"source_unit":{"unit_id":"' + name + '"}}'}]}


class Encoder:
    def encode(self, value):
        name = module().unit_id(value)
        value.clear()
        if name == "long":
            raise ValueError("too long")
        return {"input_ids": [1, 2, 3], "labels": [-100, -100, -100 if name == "empty" else 3]}


def test_encoding_does_not_mutate_or_drop_input_and_records_all_failures():
    m = module()
    data = [row("ok"), row("long"), row("empty")]
    before = deepcopy(data)
    result = m.encode_files(Encoder(), {"train": data})
    assert data == before
    assert result["checked_records"] == 3 and result["passed"] is False
    assert [r["unit_id"] for r in result["encodings"]] == ["ok"]
    assert {r["unit_id"] for r in result["failures"]} == {"long", "empty"}


def test_each_arm_and_validation_are_checked_and_success_is_explicit():
    m = module()
    result = m.encode_files(Encoder(), {"control_train": [row("c")], "repair_train": [row("r")], "repair_validation": [row("v")]})
    assert result["passed"] and result["checked_records"] == 3
    assert {r["file"] for r in result["encodings"]} == {"control_train", "repair_train", "repair_validation"}


@pytest.mark.parametrize("tamper", [False, True])
def test_full_draft_delivery_validates_hashes_split_and_original_control_before_encoding(tmp_path, monkeypatch, tamper):
    m = module()
    repo = Path(__file__).resolve().parents[1]
    project = repo.parents[2]
    original = project / "output/服务器结果/next_step_results_20261013_113024_034601484/bindings_v26_20261012_110721"
    data_name = "experiments/candidate_scope_v29_20261013"
    data = tmp_path / data_name
    shutil.copytree(repo / data_name, data)
    manifest = m.read(data / "input_manifest.json")
    source = tmp_path / "src/ecospec_kg"
    source.mkdir(parents=True)
    for name in manifest["source_sha256_lf"]:
        shutil.copyfile(repo / "src/ecospec_kg" / name, source / name)
    model = tmp_path / "model"
    model.mkdir()
    (model / "test-tokenizer.json").write_text("fixture", encoding="utf-8")
    manifest["tokenizer_config_sha256"] = {"test-tokenizer.json": m.sha(model / "test-tokenizer.json")}
    (data / "input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    m.__file__ = str(tmp_path / "tools/check_scope_inputs_20261013.py")
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [m.__file__, "--model", str(model), "--original-prep", str(original), "--output", str(output)])
    monkeypatch.setattr(m.importlib.metadata, "version", lambda name: "4.4.2")
    monkeypatch.setattr(m.subprocess, "check_output", lambda *args, **kwargs: "test-fixture-commit")
    class Template(Encoder):
        template_meta = SimpleNamespace(template_type="qwen3_5")
        def set_mode(self, mode):
            assert mode == "train"
    model_module, template_module = ModuleType("swift.model"), ModuleType("swift.template")
    calls = []
    def get_processor(path, **kwargs):
        assert kwargs == {"load_model": False, "download_model": False}
        calls.append(path)
        return None, object()
    model_module.get_model_processor = get_processor
    template_module.get_template = lambda *args, **kwargs: Template()
    monkeypatch.setitem(sys.modules, "swift.model", model_module)
    monkeypatch.setitem(sys.modules, "swift.template", template_module)
    if tamper:
        with (data / "repair_train.jsonl").open("ab") as handle:
            handle.write(b"\n")
        with pytest.raises(AssertionError, match="Input hash mismatch"):
            m.main()
        assert not calls and not output.exists()
    else:
        m.main()
        report = m.read(output / "preflight_report.json")
        assert report["passed"] and report["checked_records"] == 1387
        assert report["ready_for_training"] is report["model_training_started"] is False
        assert report["original_inputs_unchanged"]
