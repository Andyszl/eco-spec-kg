"""Diagnose existing supervision, never train or select a new checkpoint."""
import json
import io
import hashlib
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import probe_semantic_learning_20261016 as probe
import evaluate_semantic_pair_20261016 as ev
from test_semantic_pair_evaluation_20261016 import pair
from test_reviewed_pair_evaluation import http


def test_cases_come_from_positive_training_and_validation_targets_only():
    cases = probe.build_cases(ROOT)
    assert len(cases) == 6
    assert [c["split"] for c in cases].count("train") == 5
    assert [c["split"] for c in cases].count("validation") == 1
    assert sum(len(c["new_positive_ids"]) for c in cases) == 4
    assert sum(len(c["reference_positive_ids"]) for c in cases) == 9
    data = probe.semantic.rows(ROOT / probe.semantic.FOLDER / "repair_train.jsonl")
    data += probe.semantic.rows(ROOT / probe.semantic.FOLDER / "validation.jsonl")
    prompts = {probe.semantic.uid(r): r["messages"][:2] for r in data}
    for c in cases:
        assert c["messages"] == prompts[c["unit"]["unit_id"]]
        assert c["reference_positive_ids"]


def test_full_probe_18_requests_uses_exact_saved_prompts_no_training(pair, tmp_path, http):
    with patch("urllib.request.urlopen", side_effect=http["respond"]):
        result = probe.run_probe(ROOT, pair, tmp_path / "probe")
    cases = probe.build_cases(ROOT)
    expected = [c["messages"][1]["content"] for c in cases]
    assert len(http["calls"]) == 18
    assert [q for _,_,q in http["calls"]] == expected * 3
    assert result["model_requests"] == 18 and result["training_started"] is False
    assert result["new_checkpoint_selected"] is False and result["three_seed_eligible"] is False
    for arm in ("base", "control", "repair"):
        rows = result["runs"][arm]["cases"]
        assert len(rows) == 6 and sum(len(r["selected_new_positive_ids"]) for r in rows) == 4
        assert sum(len(r["missed_reference_ids"]) for r in rows) == 0
        assert sum(len(r["extra_selected_semantic_ids"]) for r in rows) > 0
        assert (tmp_path / "probe" / arm / "raw_responses.jsonl").is_file()


@pytest.mark.parametrize("failure", ["fail", "bad_proof", "models"])
def test_probe_errors_cannot_emit_success_or_change_gate(pair, tmp_path, http, failure):
    http[failure] = [] if failure == "models" else True
    out = tmp_path / "failed"
    with patch("urllib.request.urlopen", side_effect=http["respond"]), pytest.raises(AssertionError):
        probe.run_probe(ROOT, pair, out)
    assert not (out / "summary.json").exists()


def test_probe_deploy_only_reserves_larger_inference_context(pair):
    manifest = ev.inspect_pair(ROOT, pair)
    expected = ev.deploy_command(ROOT, manifest, 8018)
    actual = probe.deploy_command(ROOT, manifest, 8018)
    expected[expected.index("--max_length") + 1] = "16384"
    assert actual == expected
    assert actual[:2] == ["swift", "deploy"]


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
    script = (ROOT / "tools/run_semantic_learning_probe_20261016.sh").read_text(encoding="utf-8")
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
        probe.main()
    deploy.assert_not_called()
    assert not out.exists()
