"""Real dev fixtures, synthetic weights and mocked HTTP; no GPU inference."""
import copy
import importlib.util
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from unittest.mock import patch
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_reviewed_pair_20261014 as ev
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.selection_decoding_v2 import VERSION, implementation_hashes, request_identity


@pytest.fixture
def pair(tmp_path, monkeypatch):
    # The checked-in protocol contains metadata, not model weights or dev predictions.
    manifest = copy.deepcopy(ev.read(ROOT / ev.PROTOCOL)["training_manifest"])
    audit = ROOT / "experiments/candidate_scope_v29_20261013"
    manifest["verification"]["review"] = ev.read(audit / "supervision_review_v1_20261014.json")
    manifest["verification"]["input_manifest"] = ev.read(audit / "input_manifest.json")
    train = tmp_path / "train"
    for arm, step in (("control", 160), ("repair", 80)):
        out = train / arm
        best = out / f"checkpoint-{step}"
        best.mkdir(parents=True)
        (best / "adapter_model.safetensors").write_bytes(b"synthetic-weight-fixture")
        ev.save(best / "adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05})
        last = out / "checkpoint-200"
        last.mkdir()
        ev.save(last / "trainer_state.json", {"global_step": 200, "best_model_checkpoint": str(best),
                "best_metric": 0.02, "log_history": [{"eval_loss": 0.02, "step": step}]})
        manifest["arms"][arm]["output"] = str(out)
        manifest["arms"][arm]["completion"] = ev.inspect_completion(out)
    ev.save(train / "pair_manifest.json", manifest)
    monkeypatch.setattr(ev, "EXPECTED_PAIR_SHA", ev.sha(train / "pair_manifest.json"))
    return train


def test_recorded_best_weights_and_hashes_are_checked(pair):
    before = {p: p.read_bytes() for p in pair.rglob("*") if p.is_file()}
    info = ev.inspect_pair(ROOT, pair)
    assert [Path(info["arms"][a]["completion"]["best_checkpoint"]).name for a in ("control", "repair")] == ["checkpoint-160", "checkpoint-80"]
    assert before == {p: p.read_bytes() for p in before}
    next(pair.glob("repair/checkpoint-80/adapter_model.safetensors")).write_bytes(b"changed")
    with pytest.raises(AssertionError, match="completion"):
        ev.inspect_pair(ROOT, pair)


def test_changed_manifest_is_rejected_before_reading_weights(pair):
    with (pair / "pair_manifest.json").open("a") as f:
        f.write(" ")
    with pytest.raises(AssertionError, match="manifest"):
        ev.inspect_pair(ROOT, pair)


class Reply(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


@pytest.fixture
def http():
    state = {"calls": [], "bad_proof": False, "fail": False, "models": list(ev.MODELS.values())}
    def respond(request, timeout):
        if request.full_url.endswith("/models"):
            value = {"data": [{"id": x} for x in state["models"]]}
        elif request.full_url.endswith("/ecospec-selection-decoding"):
            value = {"version": VERSION, "backend": "transformers", "swift_version": "4.4.2", "implementation": implementation_hashes()}
        else:
            p = json.loads(request.data)
            assert timeout == 600
            assert (p["seed"], p["temperature"], p["max_tokens"]) == (42, 0, 3072)
            assert p["chat_template_kwargs"] == {"enable_thinking": False, "ecospec_selection_decoding": VERSION}
            system, prompt = [m["content"] for m in p["messages"]]
            d = json.loads(prompt)
            state["calls"].append((p["model"], d["source_unit"]["unit_id"], prompt))
            if state["fail"]: raise TimeoutError("synthetic failure")
            # Select all candidates to exercise real schema, structural merge and scoring.
            answer = {"selected_entity_ids": [e["id"] for e in d["candidate_entities"]],
                      "selected_relation_ids": [r["id"] for r in d["candidate_relations"]]}
            proof = {**request_identity(system, prompt), "prefix_calls": 1, "max_completion_tokens": 1073}
            if state["bad_proof"]: proof["prompt_hash"] = "bad"
            value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}], "ecospec_selection_decoding": proof}
        return Reply(json.dumps(value).encode())
    state["respond"] = respond
    return state


def test_full_66_requests_same_prompts_and_structure_all_models(pair, tmp_path, http):
    out = tmp_path / "evaluation"
    with patch("urllib.request.urlopen", side_effect=http["respond"]):
        result = ev.evaluate(ROOT, pair, out)
    assert result["status"] == "complete" and len(http["calls"]) == 66
    assert [p for _, _, p in http["calls"][:22]] == [p for _, _, p in http["calls"][22:44]] == [p for _, _, p in http["calls"][44:]]
    assert set(result["runs"]) == {"base", "control", "repair"}
    for arm in result["runs"]:
        assert (out / arm / "hybrid/structural_proofs.jsonl").is_file()
        assert (out / arm / "evaluation/metrics.json").is_file()
    assert result["gate"]["comparison"] == "repair versus control, both without and with the same structural module"
    assert result["training_started"] is False


@pytest.mark.parametrize("failure", ["fail", "bad_proof", "models"])
def test_incomplete_or_bad_response_never_produces_gate(pair, tmp_path, http, failure):
    http[failure] = [] if failure == "models" else True
    out = tmp_path / "bad"
    with patch("urllib.request.urlopen", side_effect=http["respond"]), pytest.raises(AssertionError):
        ev.evaluate(ROOT, pair, out)
    assert not (out / "summary.json").exists()


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


def test_deploy_uses_explicit_named_adapters_and_no_training(pair):
    info = ev.inspect_pair(ROOT, pair)
    command = ev.deploy_command(ROOT, info, 8018)
    assert command[:2] == ["swift", "deploy"]
    assert any(x.startswith("eco-pair-control=") and x.endswith("checkpoint-160") for x in command)
    assert any(x.startswith("eco-pair-repair=") and x.endswith("checkpoint-80") for x in command)
    assert command[command.index("--served_model_name") + 1] == "eco-pair-base"
    assert command[command.index("--infer_backend") + 1] == "transformers"


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
    script = (ROOT / "tools/run_reviewed_pair_dev_20261014.sh").read_text(encoding="utf-8")
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
