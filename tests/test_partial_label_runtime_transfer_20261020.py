import io,hashlib,os,shlex,shutil,subprocess,tarfile,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import inspect_partial_label_runtime_20261020 as audit

@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("network", ["normal", "first_tls_failure", "all_uploads_fail"])
def test_exact_packaging_tail_roundtrips_success_and_failure_without_weights(tmp_path, status, network):
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
    script = (ROOT / "tools/run_partial_label_runtime_inspection_20261020.sh").read_text(encoding="utf-8")
    tail = script[script.index('if [ ! -d "$OUT" ]; then'):]
    def bash_path(p):
        s = p.as_posix()
        return "/" + s[0].lower() + s[2:] if os.name == "nt" else s
    prefix = "set -euo pipefail\n" + "\n".join(f"{k}={shlex.quote(v)}" for k, v in {
        "OUT": bash_path(out), "TRANSFER": bash_path(transfer), "REMOTE": bash_path(remote),
        "BRANCH": "codex/fixture-pair-dev", "STATUS": str(status)}.items()) + "\n"
    prefix+='sleep(){ :; }\n'
    if network!='normal':
        marker=shlex.quote(bash_path(tmp_path/'first-attempt'))
        condition='true' if network=='all_uploads_fail' else '[ ! -e '+marker+' ]'
        prefix+='git(){ for arg in "$@"; do if [ "$arg" = push ] && '+condition+'; then touch '+marker+'; return 128; fi; done; command git "$@"; }\n'
    result = subprocess.run([bash, "-c", prefix + tail], text=True, capture_output=True, encoding="utf-8")
    if network=='all_uploads_fail':
        assert result.returncode==1 and 'RESULTS_UPLOAD_OK=' not in result.stdout
        assert (transfer/'results.tar.gz').is_file()
        assert (transfer/'inspection_exit_code.txt').read_text().strip()==str(status)
        assert (out/'evaluation/summary.json').read_bytes()==content
        return
    assert result.returncode == status, result.stderr
    assert "RESULTS_UPLOAD_OK=codex/fixture-pair-dev" in result.stdout
    payload = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/fixture-pair-dev:results.tar.gz"])
    checksum = subprocess.check_output(["git", "--git-dir", str(remote), "show", "refs/heads/codex/fixture-pair-dev:reports.sha256"]).decode()[:64]
    assert hashlib.sha256(payload).hexdigest() == checksum
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.extractfile("./evaluation/summary.json").read() == content
        assert not any("safetensors" in n for n in archive.getnames())
    assert (out / "evaluation/summary.json").read_bytes() == content
