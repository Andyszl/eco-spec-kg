"""A reviewed pilot must preserve partition, budget and failure boundaries."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
from dataclasses import make_dataclass

import pytest


def module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("reviewed_pair", root / "tools/train_reviewed_pair_20261014.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, root


@pytest.fixture
def delivery(tmp_path):
    m, repo = module()
    data = tmp_path / "experiments/candidate_scope_v29_20261013"
    shutil.copytree(repo / "experiments/candidate_scope_v29_20261013", data)
    source = tmp_path / "src/ecospec_kg"
    source.mkdir(parents=True)
    manifest = m.read(data / "input_manifest.json")
    for name in manifest["source_sha256_lf"]:
        shutil.copyfile(repo / "src/ecospec_kg" / name, source / name)
    frozen = tmp_path / "deliveries/human_confirmed_20261002"
    for name in manifest["frozen_sha256"]:
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "deliveries/human_confirmed_20261002" / name, path)
    report = repo.parents[2] / "output/服务器结果/scope-preflight-20261014-092238-1073350/preflight_report.json"
    return m, tmp_path, data, report


def test_real_received_preflight_and_review_accept_exact_inputs(delivery):
    m, root, data, report = delivery
    result = m.verify_delivery(root, report)
    assert result["counts"] == {"control_train": 623, "repair_train": 626, "validation": 69}
    assert result["review"]["pilot_allowed"]


@pytest.mark.parametrize("target", ["repair_train.jsonl", "control_validation.jsonl", "supervision_review_v1_20261014.json"])
def test_changed_input_or_review_blocks_before_launch(delivery, target):
    m, root, data, report = delivery
    with (data / target).open("ab") as f:
        f.write(b"\n")
    with pytest.raises(AssertionError):
        m.verify_delivery(root, report)


def test_both_commands_have_exact_budget_explicit_validation_and_no_resume(tmp_path):
    m, _ = module()
    for arm in ("control", "repair"):
        cmd = m.build_command(tmp_path, tmp_path / "model", tmp_path / arm, arm)
        values = {cmd[i]: cmd[i+1] for i in range(2, len(cmd), 2)}
        assert cmd[:2] == ["swift", "sft"]
        for key, expected in {"--seed": "42", "--data_seed": "42", "--max_steps": "200",
                              "--split_dataset_ratio": "0", "--early_stop_interval": "0",
                              "--eval_steps": "40", "--save_steps": "40", "--strict": "true",
                              "--max_length": "8192", "--truncation_strategy": "delete"}.items():
            assert values[key] == expected
        assert values["--dataset"].endswith(f"{arm}_train.jsonl")
        assert values["--val_dataset"].endswith(f"{arm}_validation.jsonl")
        assert "--resume_from_checkpoint" not in cmd


def checkpoint(tmp_path, steps=200):
    path = tmp_path / f"checkpoint-{steps}"
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"fake test fixture; not real weights")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": steps, "best_model_checkpoint": str(path),
        "best_metric": 0.5, "log_history": [{"step": 40, "eval_loss": 0.8}, {"step": 200, "eval_loss": 0.5}]}), encoding="utf-8")
    (path / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0.05}), encoding="utf-8")
    return path


def test_completion_requires_full_budget_and_nonempty_adapter(tmp_path):
    m, _ = module()
    checkpoint(tmp_path)
    assert m.inspect_completion(tmp_path)["global_step"] == 200
    (tmp_path / "checkpoint-200/adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(AssertionError):
        m.inspect_completion(tmp_path)


def test_early_exit_cannot_be_reported_as_completed_pair(tmp_path):
    m, _ = module()
    checkpoint(tmp_path, 40)
    with pytest.raises(AssertionError):
        m.inspect_completion(tmp_path)


def test_failed_first_arm_never_launches_second(tmp_path):
    m, _ = module()
    calls = []
    def launch(command, log):
        calls.append(command)
        return 7
    with pytest.raises(RuntimeError, match="control"):
        m.execute_pair(tmp_path, tmp_path / "model", tmp_path / "result", {"status": "ready"}, launch=launch)
    assert len(calls) == 1
    assert m.read(tmp_path / "result/pair_manifest.json")["status"] == "failed"


def test_successful_pair_records_two_checked_checkpoints(tmp_path):
    m, _ = module()
    def launch(command, log):
        path = Path(command[command.index("--output_dir") + 1])
        path.mkdir()
        checkpoint(path)
        return 0
    result = m.execute_pair(tmp_path, tmp_path / "model", tmp_path / "result", {"status": "ready"}, launch=launch)
    assert result["status"] == "complete"
    assert set(result["arms"]) == {"control", "repair"}
    assert all(v["completion"]["global_step"] == 200 for v in result["arms"].values())


@pytest.mark.parametrize("drop_row", [False, True])
def test_server_dry_run_checks_installed_flags_and_loaded_counts_without_training(delivery, monkeypatch, drop_row):
    m, root, data, report = delivery
    manifest = m.read(data / "input_manifest.json")
    model = root / "model"
    model.mkdir()
    actual_sha = m.sha
    for name in manifest["tokenizer_config_sha256"]:
        (model / name).write_bytes(b"tokenizer mock; not actual model config")
    monkeypatch.setattr(m, "sha", lambda p: manifest["tokenizer_config_sha256"][p.name] if p.parent == model else actual_sha(p))
    monkeypatch.setattr(m.importlib.metadata, "version", lambda name: "4.4.2")
    monkeypatch.setattr(m.subprocess, "check_output", lambda *args, **kwargs: "test-only-commit")
    m.__file__ = str(root / "tools/train_reviewed_pair_20261014.py")
    output = root / "new-run"
    monkeypatch.setattr(sys, "argv", [m.__file__, "--model", str(model), "--preflight", str(report), "--out", str(output)])
    args_module, data_module = ModuleType("swift.arguments"), ModuleType("swift.dataset")
    names = [s[2:] for s in m.build_command(root, model, output, "repair")[2::2]]
    args_module.SftArguments = make_dataclass("MockSftArguments", [(name, str) for name in names])
    calls = []
    def load(paths, **kwargs):
        assert kwargs == {"split_dataset_ratio": 0, "seed": 42, "shuffle": False, "strict": True}
        values = [json.loads(s) for s in Path(paths[0]).read_text(encoding="utf-8").splitlines()]
        calls.append(paths[0])
        return (values[:-1] if drop_row else values), None
    data_module.load_dataset = load
    monkeypatch.setitem(sys.modules, "swift.arguments", args_module)
    monkeypatch.setitem(sys.modules, "swift.dataset", data_module)
    if drop_row:
        with pytest.raises(AssertionError, match="partition counts"):
            m.main()
    else:
        m.main()
        assert len(calls) == 4
    assert not output.exists()
