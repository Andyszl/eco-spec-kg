"""Exercise new wrapper's exact packaging/retry code against local bare Git."""
import io
import hashlib
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_never_trains_or_uses_network_models():
    script = (ROOT / 'tools/run_source_supplement_preflight_20261021.sh').read_text(encoding='utf8')
    assert 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""' in script
    assert 'source_supplement_preflight_20261021.py" preflight' in script
    assert 'swift sft' not in script and 'deploy' not in script


@pytest.mark.parametrize('status', [0, 7])
@pytest.mark.parametrize('network', ['normal', 'first_failure', 'all_fail'])
def test_reports_keep_original_status_and_retry_only_transfer(tmp_path, status, network):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    remote, out, transfer = [tmp_path / n for n in ('remote.git', 'out', 'transfer')]
    subprocess.run(['git', 'init', '--bare', '-q', str(remote)], check=True)
    out.mkdir(); (out / 'launch.log').write_bytes(b'CPU check; no model weights\n')
    def posix(p):
        s = p.as_posix()
        return '/' + s[0].lower() + s[2:] if os.name == 'nt' else s
    prefix = 'set -euo pipefail\n' + '\n'.join(k + '=' + shlex.quote(v) for k, v in {
        'OUT': posix(out), 'TRANSFER': posix(transfer), 'REMOTE': posix(remote),
        'BRANCH': 'codex/preflight-fixture', 'STATUS': str(status)}.items()) + '\nsleep(){ :; }\n'
    if network != 'normal':
        marker = shlex.quote(posix(tmp_path / 'attempt'))
        condition = 'true' if network == 'all_fail' else '[ ! -e ' + marker + ' ]'
        prefix += 'git(){ for arg in "$@"; do if [ "$arg" = push ] && ' + condition + '; then touch ' + marker + '; return 128; fi; done; command git "$@"; }\n'
    script = (ROOT / 'tools/run_source_supplement_preflight_20261021.sh').read_text(encoding='utf8')
    tail = script[script.index('if [ ! -d "$OUT" ]; then'):]
    result = subprocess.run([bash, '-c', prefix + tail], text=True, encoding='utf8', capture_output=True)
    assert (transfer / 'preflight_exit_code.txt').read_text().strip() == str(status)
    assert (out / 'launch.log').read_bytes() == b'CPU check; no model weights\n'
    if network == 'all_fail':
        assert result.returncode == 1 and 'RESULTS_UPLOAD_OK=' not in result.stdout
    else:
        assert result.returncode == status, result.stderr
        assert 'RESULTS_UPLOAD_OK=codex/preflight-fixture' in result.stdout
        payload = subprocess.check_output(['git', '--git-dir', str(remote), 'show', 'refs/heads/codex/preflight-fixture:results.tar.gz'])
        assert hashlib.sha256(payload).hexdigest() == (transfer / 'reports.sha256').read_text().split()[0]
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            assert archive.extractfile('./launch.log').read() == b'CPU check; no model weights\n'
