from pathlib import Path
from types import ModuleType,SimpleNamespace
from dataclasses import make_dataclass
import sys,json
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import train_full_graph_scope_pair_20261019 as train


def test_fixed_pair_commands_are_existing_budget_with_same_shared_validation():
    commands=[train.build_command(ROOT,Path("model"),Path(a),a) for a in ("control","repair")]
    normalize=lambda c:["PATH" if i and c[i-1] in {"--dataset","--output_dir"} else x for i,x in enumerate(c)]
    assert normalize(commands[0])==normalize(commands[1])
    for c in commands:
        o=dict(zip(c[2::2],c[3::2]))
        assert o["--seed"]==o["--data_seed"]=="42" and o["--max_steps"]=="200"
        assert o["--max_length"]=="8192" and o["--split_dataset_ratio"]=="0"
        assert o["--val_dataset"]==str(ROOT/train.inputs.FOLDER/"validation.jsonl")
        assert not {"--resume_from_checkpoint","--adapters"}&set(o)


def test_real_preflight_and_input_locks_authorize_only_this_pair(tmp_path):
    v=train.verify_approved(ROOT,ROOT/train.REPORT)
    assert v["manifest"]["counts"]=={"control_train":626,"repair_train":626,"validation":72}
    p=tmp_path/"modified.json";p.write_bytes((ROOT/train.REPORT).read_bytes()+b" ")
    with pytest.raises(AssertionError,match="preflight hash"):
        train.verify_approved(ROOT,p)


def runtime(monkeypatch,tmp_path):
    checked=train.verify_approved(ROOT,ROOT/train.REPORT)
    model=tmp_path/"model";real_sha=train.inputs.sha
    monkeypatch.setattr(train.inputs,"sha",lambda p:checked["manifest"]["tokenizer_config_sha256"][p.name] if p.parent==model else real_sha(p))
    monkeypatch.setattr(train.importlib.metadata,"version",lambda n:"4.4.2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0");monkeypatch.setenv("WORLD_SIZE","1")
    arguments,dataset=ModuleType("swift.arguments"),ModuleType("swift.dataset")
    options=[n[2:] for n in train.build_command(ROOT,model,tmp_path/"run","control")[2::2]]
    arguments.SftArguments=make_dataclass("SftArguments",[(n,str) for n in options])
    dataset.load_dataset=lambda paths,**kwargs:(train.inputs.rows(Path(paths[0])),None)
    monkeypatch.setitem(sys.modules,"swift.arguments",arguments);monkeypatch.setitem(sys.modules,"swift.dataset",dataset)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True)))
    return model


@pytest.mark.parametrize("scenario",["busy","query_failed","first_arm_failure","success","second_arm_busy"])
def test_runtime_checks_and_sequential_failure_guards(monkeypatch,tmp_path,scenario):
    model=runtime(monkeypatch,tmp_path)
    calls=[]
    monkeypatch.setattr(train.subprocess,"run",lambda *a,**kw:SimpleNamespace(
        returncode=1 if scenario=="query_failed" else 0,
        stdout="1234" if scenario=="busy" or scenario=="second_arm_busy" and calls else ""))
    def launch(command,log):calls.append(command);return 7 if scenario=="first_arm_failure" else 0
    monkeypatch.setattr(train,"inspect_completion",lambda p:{"global_step":200,"best_checkpoint":str(p/"checkpoint-80")})
    dest=tmp_path/"run"
    if scenario=="success":
        result=train.train(ROOT,model,dest,ROOT/train.REPORT,run=True,launch=launch)
        assert result["status"]=="complete" and not result["expansion_to_three_seeds"]
    else:
        with pytest.raises(AssertionError):train.train(ROOT,model,dest,ROOT/train.REPORT,run=True,launch=launch)
    assert len(calls)==(0 if scenario in {"busy","query_failed"} else 1 if scenario in {"first_arm_failure","second_arm_busy"} else 2)
    if scenario in {"busy","query_failed"}:assert not dest.exists()
    elif scenario in {"first_arm_failure","second_arm_busy"}:assert train.inputs.read(dest/"pair_manifest.json")["status"]=="failed"


def test_completion_requires_minimum_shared_validation_loss(tmp_path,monkeypatch):
    out=tmp_path/"arm";best=out/"checkpoint-80";last=out/"checkpoint-200"
    best.mkdir(parents=True);last.mkdir()
    (best/"adapter_model.safetensors").write_bytes(b"synthetic-test-adapter")
    train.inputs.save(best/"adapter_config.json",{"r":8,"lora_alpha":16,"lora_dropout":0.05})
    values=[0.3,0.1,0.2,0.25,0.26]
    state={"global_step":200,"best_model_checkpoint":str(best),"best_metric":0.1,
           "log_history":[{"step":s,"eval_loss":v} for s,v in zip([40,80,120,160,200],values)]}
    train.inputs.save(last/"trainer_state.json",state)
    assert train.inspect_completion(out)["internal_validation_loss"]==0.1
    state["log_history"][2]["eval_loss"]=0.05;train.inputs.save(last/"trainer_state.json",state)
    with pytest.raises(AssertionError,match="minimum"):
        train.inspect_completion(out)


def test_runtime_checks_reject_changed_loader_membership_before_launch(monkeypatch,tmp_path):
    model=runtime(monkeypatch,tmp_path)
    sys.modules['swift.dataset'].load_dataset=lambda paths,**kwargs:([],None)
    calls=[]
    with pytest.raises(AssertionError,match='count'):
        train.train(ROOT,model,tmp_path/'run',ROOT/train.REPORT,run=True,launch=lambda *args:calls.append(args))
    assert not calls and not (tmp_path/'run').exists()


def test_dry_run_never_queries_gpu_or_starts_training(monkeypatch,tmp_path):
    model=runtime(monkeypatch,tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('must not run GPU query or launcher')
    monkeypatch.setattr(train.subprocess,'run',forbidden)
    result=train.train(ROOT,model,tmp_path/'dry',ROOT/train.REPORT,run=False,launch=forbidden)
    assert not result['training_started'] and result['status']=='dry_run_verified'
    assert not (tmp_path/'dry').exists()
