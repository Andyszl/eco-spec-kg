"""Real local Git transport with a fake CPU job, never real Swift/model calls."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

BASH = Path('C:/Program Files/Git/bin/bash.exe')
SCRIPT = Path(__file__).resolve().parents[1] / 'tools/run_final_source_preflight_20261021.sh'


@unittest.skipUnless(BASH.exists(), 'local Git Bash integration check')
class TransferTests(unittest.TestCase):
    def exercise(self, status):
        with tempfile.TemporaryDirectory(prefix='eco-final-transfer-') as tmp:
            base = Path(tmp)
            remote = base / 'remote.git'
            subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True)
            fake = base / 'fake_cpu.sh'
            fake.write_text('#!/usr/bin/env bash\nset -eu\n'
                            'while [ "$#" -gt 0 ]; do if [ "$1" = --out ]; then out=$2; break; fi; shift; done\n'
                            'mkdir -p "$out"\nprintf \'{"fixture":true,"model_calls":0}\\n\' > "$out/report.json"\n'
                            f'exit {status}\n', encoding='utf-8', newline='\n')
            env = dict(os.environ, ECOSPEC_FINAL_PREFLIGHT_REMOTE=remote.as_posix(),
                       ECOSPEC_FINAL_PREFLIGHT_PYTHON=fake.as_posix(),
                       ECOSPEC_FINAL_PREFLIGHT_RESULTS_ROOT=subprocess.check_output(
                           [str(BASH), '-c', 'cygpath -u "$1"', '_', str(base / 'results')], text=True).strip())
            run = subprocess.run([str(BASH), str(SCRIPT)], env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, status, run.stdout + run.stderr)
            branch = next(s.split('=', 1)[1] for s in run.stdout.splitlines() if s.startswith('RESULTS_UPLOAD_OK='))
            code = subprocess.check_output(['git', '--git-dir=' + str(remote), 'show', branch + ':preflight_exit_code.txt'], text=True)
            self.assertEqual(int(code.strip()), status)
            archive = subprocess.check_output(['git', '--git-dir=' + str(remote), 'show', branch + ':results.tar.gz'])
            path = base / 'received.tar.gz'; path.write_bytes(archive)
            with tarfile.open(path) as package:
                content = json.load(package.extractfile('./job/report.json'))
                self.assertIs(content['fixture'], True)
                self.assertEqual(content['model_calls'], 0)

    def test_success_is_uploaded_once(self): self.exercise(0)
    def test_job_failure_code_is_preserved_in_git_result(self): self.exercise(7)


if __name__ == '__main__':
    unittest.main()
