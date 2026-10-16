"""New v2.11 evaluation contract: synthetic adapters/HTTP, never GPU inference."""
import copy
import io
import hashlib
import os
import shlex
import shutil
import subprocess
import tarfile
import json
from pathlib import Path
import sys
from unittest.mock import patch, Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_semantic_pair_20261016 as ev
from test_reviewed_pair_evaluation import http


@pytest.fixture
def pair(tmp_path, monkeypatch):
    lock = copy.deepcopy(ev.read(ROOT / ev.PROTOCOL))
    manifest = lock["training_manifest"]
    train = tmp_path / "train"
    for arm in ("control", "repair"):
        out = train / arm
        best = out / "checkpoint-80"
        best.mkdir(parents=True)
        (best / "adapter_model.safetensors").write_bytes(b"synthetic-weight-fixture")
        ev.save(best / "adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05})
        last = out / "checkpoint-200"
        last.mkdir()
        ev.save(last / "trainer_state.json", {"global_step": 200, "best_model_checkpoint": str(best),
            "best_metric": .02, "log_history": [{"eval_loss": .02, "step": 80}]})
        manifest["arms"][arm]["output"] = str(out)
        manifest["arms"][arm]["command"] = ev.semantic.build_command(ROOT, Path("/home/hello/szl/models/Qwen3.5-9B"), out, arm)
        manifest["arms"][arm]["completion"] = ev.inspect_completion(out)
    ev.save(train / "pair_manifest.json", manifest)
    lock["training_manifest_sha256"] = ev.sha(train / "pair_manifest.json")
    protocol = tmp_path / "protocol.json"
    ev.save(protocol, lock)
    monkeypatch.setattr(ev, "PROTOCOL", str(protocol))
    monkeypatch.setattr(ev, "EXPECTED_PAIR_SHA", lock["training_manifest_sha256"])
    return train


def test_new_pair_inputs_and_selected_weights_verified_without_mutation(pair):
    original = {p: p.read_bytes() for p in pair.rglob("*") if p.is_file()}
    info = ev.inspect_pair(ROOT, pair)
    assert info["verification"]["manifest"]["counts"] == {"control_train": 625, "repair_train": 625, "validation": 70}
    assert original == {p: p.read_bytes() for p in original}
    (pair / "repair/checkpoint-80/adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="completion"):
        ev.inspect_pair(ROOT, pair)


def test_changed_training_manifest_rejected(pair):
    with (pair / "pair_manifest.json").open("a") as f: f.write(" ")
    with pytest.raises(AssertionError, match="manifest"):
        ev.inspect_pair(ROOT, pair)


def test_changed_source_lock_rejected_before_service(pair, monkeypatch):
    protocol = Path(ev.PROTOCOL)
    lock = ev.read(protocol)
    lock["source_sha256_lf"]["tools/paired_structural_metrics.py"] = "0" * 64
    ev.save(protocol, lock)
    with pytest.raises(AssertionError, match="source"):
        ev.inspect_pair(ROOT, pair)


def test_real_dev_66_mock_requests_same_v211_prompts_and_frozen_structure(pair, tmp_path, http):
    with patch("urllib.request.urlopen", side_effect=http["respond"]):
        result = ev.evaluate(ROOT, pair, tmp_path / "evaluation")
    assert result["status"] == "complete" and len(http["calls"]) == 66
    assert [p for _,_,p in http["calls"][:22]] == [p for _,_,p in http["calls"][22:44]] == [p for _,_,p in http["calls"][44:]]
    for arm in ("base", "control", "repair"):
        dest = tmp_path / "evaluation" / arm
        assert (dest / "hybrid/structural_proofs.jsonl").is_file()
        assert (dest / "evaluation/metrics.json").is_file()
    cfg = ev.read(tmp_path / "evaluation/evaluation_config.json")
    assert cfg["candidate_generator"] == "structure-aware-rule-v2.11"
    assert cfg["selection_policy"] == "ecospec-selection-v2.6"
    assert result["training_started"] is False


@pytest.mark.parametrize("failure", ["fail", "bad_proof", "models"])
def test_failed_or_invalid_inference_never_emits_gate(pair, tmp_path, http, failure):
    http[failure] = [] if failure == "models" else True
    out = tmp_path / "failed"
    with patch("urllib.request.urlopen", side_effect=http["respond"]), pytest.raises(AssertionError):
        ev.evaluate(ROOT, pair, out)
    assert not (out / "summary.json").exists()


def test_deployment_uses_new_selected_adapters_and_base(pair):
    manifest = ev.inspect_pair(ROOT, pair)
    cmd = ev.deploy_command(ROOT, manifest, 8018)
    assert cmd[:2] == ["swift", "deploy"]
    for arm in ("control", "repair"):
        assert any(x.startswith(ev.MODELS[arm] + "=") and x.endswith("checkpoint-80") for x in cmd)
    assert cmd[cmd.index("--max_length") + 1] == "8192"
    assert cmd[cmd.index("--max_new_tokens") + 1] == "3072"


def test_gate_checks_raw_and_hybrid_regressions_and_paths():
    metric = {"entities": {"f1": .8, "false_positive": 2}, "relations": {"f1": .8, "false_positive": 1}, "macro_relation_f1": .6}
    run = {"before": copy.deepcopy(metric), "after": copy.deepcopy(metric),
           "indicator_formula_edges": [{"after": True} for _ in range(3)],
           "formula_paths_after": {"expanded_complete_evidenced_paths": 1}}
    assert ev.strict_gate(run, copy.deepcopy(run))["passed"]
    for change in ("raw_f1", "macro", "fp", "binding", "path"):
        repair = copy.deepcopy(run)
        if change == "raw_f1": repair["before"]["entities"]["f1"] = .79
        if change == "macro": repair["after"]["macro_relation_f1"] = .59
        if change == "fp": repair["after"]["relations"]["false_positive"] = 2
        if change == "binding": repair["indicator_formula_edges"][0]["after"] = False
        if change == "path": repair["formula_paths_after"]["expanded_complete_evidenced_paths"] = 0
        assert not ev.strict_gate(run, repair)["passed"], change


@pytest.mark.parametrize("fails", [False, True])
def test_owned_service_stops_only_its_child_group(tmp_path, fails):
    child = Mock(pid=54321)
    child.poll.return_value = None
    with patch.object(ev.subprocess, "Popen", return_value=child) as start, \
         patch.object(ev, "check_service", return_value={}), \
         patch.object(ev.os, "killpg", create=True) as kill:
        try:
            with ev.owned_service(["swift", "deploy"], tmp_path / "deploy.log", "http://127.0.0.1:8018/v1"):
                if fails: raise ValueError("evaluation failed")
        except ValueError:
            assert fails
        assert start.call_args.kwargs["start_new_session"] is True
        kill.assert_called_once_with(54321, ev.signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=30)


@pytest.mark.parametrize("status", [0, 7])
def test_exact_packaging_tail_roundtrips_success_and_failure_without_weights(tmp_path, status):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    assert bash
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    out = tmp_path / "results"
    out.mkdir()
    (out / "evaluation").mkdir()
    content = '{"状态":"complete"}\n'.encode()
    (out / "evaluation/summary.json").write_bytes(content)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"must stay outside result archive")
    transfer = tmp_path / "transfer"
    script = (ROOT / "tools/run_semantic_pair_dev_20261016.sh").read_text(encoding="utf-8")
    tail = script[script.index('if [ ! -d "$OUT" ]; then'):]
    def bash_path(p):
        s = p.as_posix()
        return "/" + s[0].lower() + s[2:] if os.name == "nt" else s
    prefix = "set -euo pipefail\n" + "\n".join(f"{k}={shlex.quote(v)}" for k, v in {
        "OUT": bash_path(out), "TRANSFER": bash_path(transfer), "REMOTE": bash_path(remote),
        "BRANCH": "codex/fixture-pair-dev", "STATUS": str(status)}.items()) + "\n"
    result = subprocess.run([bash, "-c", prefix + tail], text=True, capture_output=True, encoding="utf-8")
    assert result.returncode == status, result.stderr
    assert "RESULTS_UPLOAD_OK=codex/fixture-pair-dev" in result.stdout
    payload = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/fixture-pair-dev:results.tar.gz"])
    checksum = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/fixture-pair-dev:reports.sha256"]).decode()[:64]
    assert hashlib.sha256(payload).hexdigest() == checksum
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.extractfile("./evaluation/summary.json").read() == content
        assert not any("safetensors" in n for n in archive.getnames())
    assert (out / "evaluation/summary.json").read_bytes() == content


def test_busy_gpu_blocks_before_deploy_or_output(pair, tmp_path, monkeypatch):
    from dataclasses import make_dataclass
    from types import SimpleNamespace, ModuleType
    manifest = ev.read(pair / "pair_manifest.json")
    command = ev.deploy_command(ROOT, manifest, 8018)
    arguments = ModuleType("swift.arguments")
    arguments.DeployArguments = make_dataclass("DeployArguments", [(n[2:], str) for n in command if n.startswith("--")])
    monkeypatch.setitem(sys.modules, "swift", ModuleType("swift"))
    monkeypatch.setitem(sys.modules, "swift.arguments", arguments)
    monkeypatch.setattr(ev.importlib.metadata, "version", lambda name: "4.4.2")
    original_sha = ev.sha
    config_hashes = manifest["verification"]["manifest"]["tokenizer_config_sha256"]
    monkeypatch.setattr(ev, "sha", lambda p: config_hashes[p.name] if str(p).replace("\\", "/").startswith("/home/hello/szl/models/") else original_sha(p))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(ev.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="12345", stderr=""))
    out = tmp_path / "blocked"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--train", str(pair), "--out", str(out)])
    with patch.object(ev.subprocess, "Popen") as deploy, pytest.raises(AssertionError, match="GPU0 unavailable or occupied"):
        ev.main()
    deploy.assert_not_called()
    assert not out.exists()
