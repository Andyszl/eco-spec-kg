"""Local transfer and GPU occupancy guard for the diagnostic entry."""
import io,hashlib,os,shlex,shutil,subprocess,tarfile,json,sys
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import evaluate_cross_source_pair_20261016 as ev
import probe_cross_source_learning_20261017 as probe
from test_cross_source_dev_20261016 import pair
from test_cross_source_learning_probe_20261017 import probe_pair


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
    script = (ROOT / "tools/run_cross_source_probe_20261017.sh").read_text(encoding="utf-8")
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

def test_busy_gpu_blocks_before_deploy_or_output(probe_pair, tmp_path, monkeypatch):
    from dataclasses import make_dataclass
    from types import SimpleNamespace, ModuleType
    manifest = ev.read(probe_pair / "pair_manifest.json")
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
    monkeypatch.setattr(sys, "argv", ["evaluate", "--train", str(probe_pair), "--out", str(out)])
    with patch.object(ev.subprocess, "Popen") as deploy, pytest.raises(AssertionError, match="GPU0 unavailable or occupied"):
        probe.main()
    deploy.assert_not_called()
    assert not out.exists()
