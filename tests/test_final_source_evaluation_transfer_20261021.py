"""Actual bare-Git transfer; fake job only, no Swift or model calls."""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

BASH = Path('C:/Program Files/Git/bin/bash.exe')
SCRIPT = Path(__file__).resolve().parents[1] / 'tools/run_final_source_evaluation_20261021.sh'


@unittest.skipUnless(BASH.exists(), 'local Git Bash integration check')
class TransferTests(unittest.TestCase):
    def exercise(self, status):
        with tempfile.TemporaryDirectory(prefix='eco-final-eval-') as tmp:
            base = Path(tmp); remote = base/'remote.git'
            subprocess.run(['git','init','-q','--bare',str(remote)],check=True)
            fake = base/'fake_job.sh'
            fake.write_text('#!/usr/bin/env bash\nset -eu\n'
                'while [ "$#" -gt 0 ]; do if [ "$1" = --out ]; then out=$2; break; fi; shift; done\n'
                'mkdir -p "$out"\nprintf \'{"fixture":true,"not_model_results":true}\\n\' > "$out/report.json"\n'
                f'exit {status}\n',encoding='utf8',newline='\n')
            posix = subprocess.check_output([str(BASH),'-c','cygpath -u "$1"','_',str(base/'results')],text=True).strip()
            env = dict(os.environ,ECOSPEC_FINAL_EVAL_REMOTE=remote.as_posix(),
                ECOSPEC_FINAL_EVAL_PYTHON=fake.as_posix(),ECOSPEC_FINAL_EVAL_RESULTS_ROOT=posix)
            run = subprocess.run([str(BASH),str(SCRIPT)],env=env,capture_output=True,text=True,timeout=45)
            self.assertEqual(run.returncode,status,run.stdout+run.stderr)
            self.assertEqual(run.stdout.count('RESULTS_UPLOAD_OK='),1)
            branch = next(s.split('=',1)[1] for s in run.stdout.splitlines() if s.startswith('RESULTS_UPLOAD_OK='))
            code = subprocess.check_output(['git','--git-dir='+str(remote),'show',branch+':evaluation_exit_code.txt'],text=True)
            self.assertEqual(int(code.strip()),status)
            archive = subprocess.check_output(['git','--git-dir='+str(remote),'show',branch+':results.tar.gz'])
            with tarfile.open(fileobj=io.BytesIO(archive)) as package:
                report = json.load(package.extractfile('./job/report.json'))
                self.assertTrue(report['fixture'] and report['not_model_results'])

    def test_success_is_transferred(self): self.exercise(0)
    def test_failure_is_transferred_without_claiming_success(self): self.exercise(9)


if __name__ == '__main__':
    unittest.main()
