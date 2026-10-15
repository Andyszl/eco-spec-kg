"""New pair must have real role coverage, parent isolation and fail-closed input locks."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from dataclasses import make_dataclass
import hashlib
import io
import os
import shlex
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from semantic_pair_20261015 import build_delivery, verify_delivery, verify_preflight, build_command, FOLDER, HOLDOUT
import semantic_pair_20261015 as m


@pytest.fixture
def delivery(tmp_path):
    result = build_delivery(ROOT, tmp_path / "delivery")
    return tmp_path / "delivery", result


def test_both_arms_share_parent_disjoint_partitions_and_semantic_validation(delivery):
    folder, report = delivery
    ids = report["unit_ids"]
    assert len(ids["control_train"]) == len(ids["repair_train"]) == 625
    assert ids["control_train"] == ids["repair_train"]
    assert len(ids["validation"]) == 70 and HOLDOUT in ids["validation"]
    assert not set(ids["repair_train"]) & set(ids["validation"])
    assert HOLDOUT not in report["training_parent_ids"]
    for role in ("model_variable|obtained_by|method", "model_variable|sourced_from|data_source"):
        assert report["supervision"]["repair_train"][role]["positive_edges"] >= 3
        assert report["supervision"]["validation"][role]["positive_edges"] >= 1
    assert report["validation_policy"]["independent_final_test"] is False


def test_only_reviewed_targets_change_and_all_ids_are_prompt_members(delivery):
    folder, report = delivery
    assert report["changed_training_targets"] == 4
    assert report["added_training_positive_relations"] == 4
    assert len(report["annotation_revisions"]) == 5
    for name in ("control_train", "repair_train", "validation"):
        for line in (folder / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            q, t = json.loads(row["messages"][1]["content"]), json.loads(row["messages"][-1]["content"])
            assert set(t["selected_relation_ids"]) <= {r["id"] for r in q["candidate_relations"]}
            assert set(t["selected_entity_ids"]) <= {e["id"] for e in q["candidate_entities"]}


def test_rebuilding_existing_delivery_refuses_overwrite(delivery):
    folder, _ = delivery
    with pytest.raises(AssertionError, match="exists"):
        build_delivery(ROOT, folder)


def test_delivery_hash_change_fails_before_tokenizer(tmp_path, delivery):
    folder, _ = delivery
    assert verify_delivery(ROOT, folder)["status"] == "input_audit_passed"
    with (folder / "repair_train.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(AssertionError, match="hash"):
        verify_delivery(ROOT, folder)


def test_failed_or_mock_preflight_cannot_authorize_training(delivery):
    folder, report = delivery
    for preflight in ({"passed": False}, {"passed": True, "real_tokenizer": False}):
        with pytest.raises(AssertionError):
            verify_preflight(ROOT, folder, preflight)


def test_pair_commands_match_except_arm_paths_and_keep_fixed_budget():
    commands = [build_command(ROOT, Path("model"), Path(a), a) for a in ("control", "repair")]
    normalize = lambda c: ["PATH" if i and c[i - 1] in {"--dataset", "--output_dir"} else x for i, x in enumerate(c)]
    assert normalize(commands[0]) == normalize(commands[1])
    for cmd in commands:
        options = dict(zip(cmd[2::2], cmd[3::2]))
        assert options["--max_steps"] == "200" and options["--seed"] == "42"
        assert options["--split_dataset_ratio"] == "0" and options["--max_length"] == "8192"
        assert options["--val_dataset"] == str(ROOT / FOLDER / "validation.jsonl")


def runtime(monkeypatch, folder, manifest, tmp_path, fails=False):
    model = tmp_path / "model"
    real_sha = m.sha
    monkeypatch.setattr(m, "sha", lambda p: manifest["tokenizer_config_sha256"][p.name] if p.parent == model else real_sha(p))
    monkeypatch.setattr(m, "FOLDER", str(folder))
    monkeypatch.setattr(m.importlib.metadata, "version", lambda n: "4.4.2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    calls = []
    class Template:
        template_meta = SimpleNamespace(template_type="qwen3_5")
        def set_mode(self, mode): assert mode == "train"
        def encode(self, record):
            if fails and m.uid(record) == HOLDOUT: raise ValueError("synthetic overlength")
            return {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    swift, model_mod, template_mod = ModuleType("swift"), ModuleType("swift.model"), ModuleType("swift.template")
    def processor(path, **kwargs):
        calls.append(kwargs)
        return None, object()
    model_mod.get_model_processor = processor
    template_mod.get_template = lambda processor, **kwargs: Template()
    for n, module in (("swift", swift), ("swift.model", model_mod), ("swift.template", template_mod)):
        monkeypatch.setitem(sys.modules, n, module)
    return model, calls


@pytest.mark.parametrize("fails", [False, True])
def test_preflight_encodes_all_shared_inputs_without_weights_and_reports_failure(delivery, monkeypatch, tmp_path, fails):
    folder, manifest = delivery
    model, calls = runtime(monkeypatch, folder, manifest, tmp_path, fails)
    report = m.preflight(ROOT, model, tmp_path / "report")
    assert calls == [{"load_model": False, "download_model": False}]
    assert report["checked_records"] == 1320 and report["model_weights_loaded"] is False
    assert report["passed"] is (not fails)
    assert len(report["failures"]) == int(fails)
    if fails:
        with pytest.raises(AssertionError): verify_preflight(ROOT, folder, report)
    else:
        assert verify_preflight(ROOT, folder, report)["status"] == "input_audit_passed"


@pytest.mark.parametrize("failure", ["busy", "first_arm", "success"])
def test_launch_guards_and_first_arm_failure_do_not_start_second(delivery, monkeypatch, tmp_path, failure):
    folder, manifest = delivery
    model, _ = runtime(monkeypatch, folder, manifest, tmp_path)
    pf = tmp_path / "pf"
    m.preflight(ROOT, model, pf)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    arguments, dataset = ModuleType("swift.arguments"), ModuleType("swift.dataset")
    options = [n[2:] for n in m.build_command(ROOT, model, tmp_path / "run", "control")[2::2]]
    arguments.SftArguments = make_dataclass("SftArguments", [(n, str) for n in options])
    dataset.load_dataset = lambda paths, **kwargs: (range(70 if Path(paths[0]).stem == "validation" else 625), None)
    monkeypatch.setitem(sys.modules, "swift.arguments", arguments)
    monkeypatch.setitem(sys.modules, "swift.dataset", dataset)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="1234" if failure == "busy" else ""))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)))
    calls = []
    def launch(cmd, log): calls.append(cmd); return 0 if failure == "success" else 7
    monkeypatch.setattr(m.previous, "inspect_completion", lambda p: {"global_step": 200, "best_checkpoint": str(p / "checkpoint-80")})
    out = tmp_path / "train"
    if failure == "success":
        result = m.train(ROOT, model, out, pf / "preflight_report.json", run=True, launch=launch)
        assert result["status"] == "complete" and len(result["arms"]) == 2 and result["seed"] == 42
        assert result["expansion_to_three_seeds"] is False
    else:
        with pytest.raises(AssertionError): m.train(ROOT, model, out, pf / "preflight_report.json", run=True, launch=launch)
    assert len(calls) == (0 if failure == "busy" else 2 if failure == "success" else 1)
    if failure == "busy": assert not out.exists()
    elif failure != "success": assert json.loads((out / "pair_manifest.json").read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("kind", ["preflight", "pair"])
def test_reports_git_roundtrip_preserves_success_and_failure(tmp_path, status, kind):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    remote, out, transfer = tmp_path / "remote.git", tmp_path / "result", tmp_path / "transfer"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    out.mkdir()
    content = json.dumps({"passed": status == 0, "fixture": True}).encode()
    (out / "preflight_report.json").write_bytes(content)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"excluded")
    script = (ROOT / f"tools/run_semantic_{kind}_20261015.sh").read_text(encoding="utf-8")
    marker = 'if [ ! -d "$OUT" ]; then' if kind == "preflight" else 'test ! -e "$TRANSFER"'
    tail = script[script.index(marker):]
    def bp(p):
        s = p.as_posix()
        return "/" + s[0].lower() + s[2:] if os.name == "nt" else s
    prefix = "set -euo pipefail\n" + "\n".join(f"{k}={shlex.quote(v)}" for k, v in {
        "OUT": bp(out), "PAIR": bp(out), "TRANSFER": bp(transfer), "REMOTE": bp(remote), "BRANCH": "codex/semantic-fixture", "STATUS": str(status)}.items()) + "\n"
    result = subprocess.run([bash, "-c", prefix + tail], text=True, capture_output=True, encoding="utf-8")
    assert result.returncode == status, result.stderr
    assert "RESULTS_UPLOAD_OK=codex/semantic-fixture" in result.stdout
    payload = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/semantic-fixture:results.tar.gz"])
    digest = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/semantic-fixture:reports.sha256"]).decode()[:64]
    assert hashlib.sha256(payload).hexdigest() == digest
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.extractfile("./preflight_report.json").read() == content
        assert not any("safetensors" in p for p in archive.getnames())
    assert (out / "preflight_report.json").read_bytes() == content
