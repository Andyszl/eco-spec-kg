"""Launch one full-graph scoped seed42 pair after the received real preflight."""
import argparse
from dataclasses import fields
import importlib.metadata
import math,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import full_graph_scope_pair_20261018 as inputs
import train_reviewed_pair_20261014 as baseline

FOLDER="experiments/full_graph_scope_training_20261019"
REPORT=FOLDER+"/reviewed_preflight_report.json"
EXPECTED_PREFLIGHT_SHA="c4e01add9babca6ef1722cf51ccd3fb455d6662e8d704dd353dafdbc0be34e7f"
EXPECTED_INPUT_SHA="0b3c79e533a57fa596a49e4bca0c53f1772c5c31ff270eabf78348395fd93ba5"


def verify_approved(root,report_path):
    assert inputs.sha(report_path)==EXPECTED_PREFLIGHT_SHA,"Reviewed preflight hash changed"
    review=inputs.read(root/FOLDER/"training_review.json")
    assert review["seed42_pair_authorized"] and review["preflight_sha256"]==EXPECTED_PREFLIGHT_SHA
    assert review["input_manifest_sha256"]==EXPECTED_INPUT_SHA
    for n,h in review["launcher_sha256_lf"].items():assert inputs.lfsha(root/n)==h,"Launcher changed: "+n
    report=inputs.read(report_path)
    assert report["code_commit"]==review["preflight_code_commit"]
    assert report["code_commit"]=="f1d6cd6bc9b0fb7fc4235c4d613fee90f433bfb9"
    checked=inputs.verify_preflight(root,root/inputs.FOLDER,report)
    assert checked["manifest_sha256"]==EXPECTED_INPUT_SHA
    assert report["max_tokens_by_file"]==review["verified_max_tokens"]
    assert min(r["supervised_tokens"] for r in report["encodings"])==review["minimum_supervised_tokens"]==15
    return checked


def build_command(root,model,out,arm):
    command=baseline.build_command(root,model,out,arm)
    command[command.index("--dataset")+1]=str(root/inputs.FOLDER/(arm+"_train.jsonl"))
    command[command.index("--val_dataset")+1]=str(root/inputs.FOLDER/"validation.jsonl")
    return command


def runtime_checks(root,model,out,checked):
    assert importlib.metadata.version("ms-swift")=="4.4.2","Unexpected Swift version"
    for n,h in checked["manifest"]["tokenizer_config_sha256"].items():assert inputs.sha(model/n)==h,"Model/tokenizer configuration changed: "+n
    # Retain the successful 200-step CLI recipe; new paths hold the scoped questions.
    # The real template report proves all records fit: no expected length deletion.
    from swift.arguments import SftArguments
    supported={f.name for f in fields(SftArguments)}
    for arm in ("control","repair"):
        cmd=build_command(root,model,out/arm,arm)
        assert all(n[2:] in supported for n in cmd[2::2]),"Swift parser lacks an audited option"
    from swift.dataset import load_dataset
    for name,count in checked["manifest"]["counts"].items():
        dataset,validation=load_dataset([str(root/inputs.FOLDER/(name+".jsonl"))],split_dataset_ratio=0,seed=42,shuffle=False,strict=True)
        assert validation is None and len(dataset)==count,"Dataset count or implicit split changed"
        assert [inputs.uid(r) for r in dataset]==checked["manifest"]["unit_ids"][name],"Dataset loader changed unit order/membership"


def require_idle_gpu():
    gpu=subprocess.run(["nvidia-smi","-i","0","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True)
    assert gpu.returncode==0,"GPU0 occupancy query failed"
    assert not gpu.stdout.strip(),"GPU 0 occupied by processes: "+gpu.stdout.strip()


def inspect_completion(out):
    result=baseline.inspect_completion(out)
    state=next(inputs.read(p) for p in out.glob("checkpoint-*/trainer_state.json") if inputs.read(p).get("global_step")==200)
    evaluations=[v for v in state["log_history"] if "eval_loss" in v]
    assert sorted({int(v["step"]) for v in evaluations})==[40,80,120,160,200],"Fixed validation schedule changed"
    minimum=min(float(v["eval_loss"]) for v in evaluations)
    assert math.isclose(float(state["best_metric"]),minimum,rel_tol=1e-6,abs_tol=1e-8),"Checkpoint is not minimum shared validation loss"
    match=re.search(r"checkpoint-(\d+)$",result["best_checkpoint"])
    assert match and any(int(v["step"])==int(match.group(1)) and math.isclose(float(v["eval_loss"]),minimum,rel_tol=1e-6,abs_tol=1e-8) for v in evaluations),"Selected checkpoint loss differs from shared minimum"
    result["validation_steps"]=sorted({int(v["step"]) for v in evaluations})
    return result


def train(root,model,out,report_path,run=False,launch=baseline.stream_launch):
    checked=verify_approved(root,report_path)
    assert not out.exists(),"Preserve existing training run"
    runtime_checks(root,model,out,checked)
    print("TRAINING_RUNTIME_INPUT_CHECKS_OK=true",flush=True)
    if not run:
        return {"status":"dry_run_verified","training_started":False,"commands":[build_command(root,model,out/a,a) for a in ("control","repair")]}
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0" and os.environ.get("WORLD_SIZE","1")=="1","Expected single GPU0 process"
    require_idle_gpu()
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"GPU0/bfloat16 unavailable"
    out.mkdir(parents=True)
    result={"status":"running","seed":42,"max_steps_per_arm":200,"arms":{},"verification":checked,
            "preflight_sha256":EXPECTED_PREFLIGHT_SHA,"candidate_profile":inputs.profile.VERSION,
            "decision_scope_version":inputs.scope.VERSION,"question_contract_changed":True,
            "token_level_loss_masking":False,"clause_guard_used_in_training":False,
            "training_code_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
            "expansion_to_three_seeds":False,"dev_evaluation_completed":False,
            "control_is_incomplete_supervision_ablation":True,"checkpoint_selection":"minimum shared 72-record validation loss"}
    try:
        for arm in ("control","repair"):
            assert verify_approved(root,report_path)==checked,"Protected inputs changed before arm"
            require_idle_gpu()
            command=build_command(root,model,out/arm,arm)
            result["arms"][arm]={"status":"running","command":command,"output":str(out/arm)}
            inputs.save(out/"pair_manifest.json",result)
            print("STARTING_ARM="+arm+" SEED=42 MAX_STEPS=200",flush=True)
            code=launch(command,out/(arm+".log"))
            assert code==0,f"{arm} failed with exit code {code}; subsequent arm not started"
            result["arms"][arm].update(status="complete",completion=inspect_completion(out/arm))
            assert verify_approved(root,report_path)==checked,"Protected inputs changed during arm"
            for n,h in checked["manifest"]["tokenizer_config_sha256"].items():assert inputs.sha(model/n)==h,"Base model configuration changed"
            inputs.save(out/"pair_manifest.json",result)
        result["status"]="complete"
    except BaseException as exc:
        if arm in result["arms"] and result["arms"][arm]["status"]=="running":
            result["arms"][arm]["status"]="failed"
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        inputs.save(out/"pair_manifest.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,default=ROOT/REPORT);parser.add_argument("--run",action="store_true")
    a=parser.parse_args()
    r=train(ROOT,a.model,a.out,a.preflight,a.run)
    print(inputs.dumps([{k:v for k,v in r.items() if k in {"status","training_started","seed","max_steps_per_arm"}}]),flush=True)
