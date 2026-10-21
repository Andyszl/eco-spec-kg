from pathlib import Path
from types import ModuleType,SimpleNamespace
from dataclasses import make_dataclass
import sys,json
import pytest
import train_source_supplement_pair_20261021 as t


def test_new_template_and_same_fixed_budget():
    a,b=[t.build_command(t.ROOT,Path('model'),Path(arm),arm) for arm in ('control','repair')]
    normalize=lambda c:['PATH' if j and c[j-1] in ('--dataset','--output_dir') else v for j,v in enumerate(c)]
    assert normalize(a)==normalize(b)
    o=dict(zip(a[2::2],a[3::2]))
    assert o['--template']==t.inputs.policy.TEMPLATE
    assert o['--external_plugins']==str(t.ROOT/t.PLUGIN)
    assert o['--val_dataset']==str(t.ROOT/t.inputs.FOLDER/'validation.jsonl')
    assert o['--seed']==o['--data_seed']=='42' and o['--max_steps']=='200'
    assert o['--max_length']=='8192' and o['--strict']=='true'
    assert o['--lazy_tokenize']==o['--packing']==o['--padding_free']=='false'
    assert o['--load_from_cache_file']==o['--use_logits_to_keep']=='false'
    assert o['--enable_channel_loss']==o['--enable_dft_loss']==o['--use_liger_kernel']=='false'
    assert not {'--adapters','--resume_from_checkpoint','--cached_dataset'}&set(o)
    captured=t.inputs.read(t.ROOT/t.FOLDER/'reference_sft_arg_names.json')
    assert {k[2:] for k in o}<=set(captured['argument_names'])


def test_actual_new_report_and_all_input_locks(tmp_path):
    v=t.verify_approved(t.ROOT,t.ROOT/t.REPORT)
    assert v['manifest']['counts']=={'control_train':629,'repair_train':629,'validation':74}
    assert v['checked_records']==1332 and v['masked_positions']==86
    p=tmp_path/'changed.json';p.write_bytes((t.ROOT/t.REPORT).read_bytes()+b' ')
    with pytest.raises(AssertionError,match='preflight hash'):t.verify_approved(t.ROOT,p)


def setup_runtime(monkeypatch,tmp_path):
    checked=t.verify_approved(t.ROOT,t.ROOT/t.REPORT);model=tmp_path/'model';real_sha=t.inputs.sha
    monkeypatch.setattr(t.inputs,'sha',lambda p:checked['manifest']['tokenizer_config_sha256'][p.name] if p.parent==model else real_sha(p))
    monkeypatch.setattr(t,'verify_sdk',lambda root:None)
    arguments,dataset=ModuleType('swift.arguments'),ModuleType('swift.dataset')
    names=[v[2:] for v in t.build_command(t.ROOT,model,tmp_path/'arm','control')[2::2]]
    arguments.SftArguments=make_dataclass('SftArguments',[(n,str) for n in names])
    dataset.load_dataset=lambda files,**kw:(t.inputs.rows(Path(files[0])),None)
    monkeypatch.setitem(sys.modules,'swift.arguments',arguments);monkeypatch.setitem(sys.modules,'swift.dataset',dataset)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('WORLD_SIZE','1')
    # Expensive immutable chain was checked above; repeated arm calls use the
    # same audited fixture in these process-control tests, not fake train data.
    monkeypatch.setattr(t,'verify_approved',lambda root,report:checked)
    return model,checked


@pytest.mark.parametrize('scenario',['busy','query_failed','first_failure','success','second_busy'])
def test_sequential_guards(monkeypatch,tmp_path,scenario):
    model,checked=setup_runtime(monkeypatch,tmp_path);calls=[]
    monkeypatch.setattr(t.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1 if scenario=='query_failed' else 0,
        stdout='1234567' if scenario=='busy' or scenario=='second_busy' and calls else ''))
    def launch(cmd,log):calls.append(cmd);return 7 if scenario=='first_failure' else 0
    monkeypatch.setattr(t,'inspect_completion',lambda p:dict(global_step=200,best_checkpoint=str(p/'checkpoint-80')))
    out=tmp_path/'run'
    if scenario=='success':
        result=t.train(t.ROOT,model,out,t.ROOT/t.REPORT,True,launch)
        assert result['status']=='complete' and result['token_level_loss_masking'] and not result['expansion_to_three_seeds']
        assert result['checkpoint_selection']=='minimum shared 74-record internal validation loss'
    else:
        with pytest.raises(AssertionError):t.train(t.ROOT,model,out,t.ROOT/t.REPORT,True,launch)
    assert len(calls)==(0 if scenario in ('busy','query_failed') else 2 if scenario=='success' else 1)
    if scenario in ('busy','query_failed'):assert not out.exists()
    elif scenario!='success':assert t.inputs.read(out/'pair_manifest.json')['status']=='failed'


def test_loader_cannot_change_channel_or_drop_record(monkeypatch,tmp_path):
    model,v=setup_runtime(monkeypatch,tmp_path)
    def changed(files,**kw):
        values=t.inputs.rows(Path(files[0]));values[0]['channel']='changed';return values,None
    sys.modules['swift.dataset'].load_dataset=changed
    with pytest.raises(AssertionError,match='Loader changed'):t.runtime_checks(t.ROOT,model,tmp_path/'run',v)
    sys.modules['swift.dataset'].load_dataset=lambda *a,**kw:([],None)
    with pytest.raises(AssertionError,match='count'):t.runtime_checks(t.ROOT,model,tmp_path/'run',v)


def test_dry_run_and_preserve_existing_output(monkeypatch,tmp_path):
    model,v=setup_runtime(monkeypatch,tmp_path)
    def forbid(*a,**kw):raise AssertionError('No GPU query or training in dry run')
    monkeypatch.setattr(t.subprocess,'run',forbid)
    r=t.train(t.ROOT,model,tmp_path/'dry',t.ROOT/t.REPORT,False,forbid)
    assert not r['training_started'] and not (tmp_path/'dry').exists()
    with pytest.raises(AssertionError,match='Preserve'):t.train(t.ROOT,model,tmp_path,t.ROOT/t.REPORT,True,forbid)


def test_completion_selects_only_shared_validation_minimum(tmp_path):
    arm=tmp_path/'arm';best=arm/'checkpoint-80';last=arm/'checkpoint-200';best.mkdir(parents=True);last.mkdir()
    (best/'adapter_model.safetensors').write_bytes(b'local-test-not-a-model')
    t.inputs.save(best/'adapter_config.json',dict(r=8,lora_alpha=16,lora_dropout=0.05))
    s=dict(global_step=200,best_model_checkpoint=str(best),best_metric=.1,
           log_history=[dict(step=n,eval_loss=l) for n,l in zip([40,80,120,160,200],[.3,.1,.2,.4,.5])])
    t.inputs.save(last/'trainer_state.json',s)
    assert t.inspect_completion(arm)['internal_validation_loss']==.1
    s['log_history'][2]['eval_loss']=.05;t.inputs.save(last/'trainer_state.json',s)
    with pytest.raises(AssertionError,match='minimum'):t.inspect_completion(arm)
