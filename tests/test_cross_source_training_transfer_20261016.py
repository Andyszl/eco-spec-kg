"""Exercise the real transfer tail against a local bare Git fixture."""
from pathlib import Path
import hashlib,io,json,os,shlex,shutil,subprocess,sys,tarfile
import pytest
ROOT=Path(__file__).resolve().parents[1]


def bp(p):
    s=p.as_posix()
    return "/"+s[0].lower()+s[2:] if os.name=="nt" else s


@pytest.mark.parametrize("status",[0,7])
def test_training_report_transfer_preserves_status_and_excludes_weights(tmp_path,status):
    bash="C:/Program Files/Git/bin/bash.exe" if os.name=="nt" else shutil.which("bash")
    remote,pair,out,transfer=(tmp_path/n for n in ("remote.git","pair","result","transfer"))
    subprocess.run(["git","init","--bare","-q",str(remote)],check=True)
    checkpoint=pair/"control/checkpoint-200";checkpoint.mkdir(parents=True);out.mkdir()
    (checkpoint/"trainer_state.json").write_text('{"fixture":true,"global_step":200}',encoding="utf-8")
    (checkpoint/"adapter_model.safetensors").write_bytes(b"excluded-weight-fixture")
    (checkpoint/"optimizer.pt").write_bytes(b"excluded-optimizer-fixture")
    (pair/"control.log").write_text("synthetic log",encoding="utf-8")
    (out/"launch.log").write_text("synthetic launch",encoding="utf-8")
    script=(ROOT/"tools/run_cross_source_train_20261016.sh").read_text(encoding="utf-8")
    tail=script[script.index('python - "$PAIR" "$OUT"'):]
    prefix="set -euo pipefail\n"+"\n".join(k+"="+shlex.quote(v) for k,v in {
        "PAIR":bp(pair),"OUT":bp(out),"TRANSFER":bp(transfer),"REMOTE":bp(remote),
        "BRANCH":"codex/cross-source-fixture","STATUS":str(status)}.items())+"\n"
    prefix+="python(){ "+shlex.quote(bp(Path(sys.executable)))+' "$@"; }\n'
    result=subprocess.run([bash,"-c",prefix+tail],capture_output=True,text=True,encoding="utf-8")
    assert result.returncode==status,result.stderr
    assert "RESULTS_UPLOAD_OK=codex/cross-source-fixture" in result.stdout
    payload=subprocess.check_output(["git","--git-dir",str(remote),"show","refs/heads/codex/cross-source-fixture:results.tar.gz"])
    digest=subprocess.check_output(["git","--git-dir",str(remote),"show","refs/heads/codex/cross-source-fixture:reports.sha256"]).decode()[:64]
    assert hashlib.sha256(payload).hexdigest()==digest
    with tarfile.open(fileobj=io.BytesIO(payload),mode="r:gz") as archive:
        assert json.loads(archive.extractfile("./metadata/control/checkpoint-200/trainer_state.json").read())["fixture"]
        assert not any(n.endswith((".safetensors",".pt")) for n in archive.getnames())
