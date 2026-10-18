"""Real local Git roundtrip of the CPU-only reporting tail."""
import io,hashlib,os,shlex,shutil,subprocess,tarfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

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
    script = (ROOT / "tools/run_method_clause_offline_20261018.sh").read_text(encoding="utf-8")
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
