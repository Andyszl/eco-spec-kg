"""One seed42 pair using the actually preflighted full-graph partial labels."""
import argparse
from dataclasses import fields
import importlib.metadata,importlib.util,math,os,subprocess
from pathlib import Path
import source_supplement_preflight_20261021 as inputs
import train_reviewed_pair_20261014 as baseline
import train_full_graph_scope_pair_20261019 as prior_train

ROOT=Path(__file__).resolve().parents[1]
FOLDER='experiments/source_supplement_training_v229_20261021'
REPORT=FOLDER+'/reviewed_preflight_report.json'
PLUGIN='tools/swift_alias_partial_label_plugin_20261020.py'
EXPECTED_REPORT_SHA='98a7c73f920793ef64c8df5b19f201b44963062b82c7a4cdb3f97538bd769265'
EXPECTED_INPUT_SHA='554e18400938f9470594f418c78e4e8d7820d5bf3e588ae6feca1cbb719f9581'
SOURCES=['tools/train_source_supplement_pair_20261021.py','tools/run_source_supplement_train_20261021.sh',
         PLUGIN,'src/ecospec_kg/alias_partial_label_v227.py','src/ecospec_kg/partial_label_v226.py',
         'tools/train_reviewed_pair_20261014.py','tools/train_full_graph_scope_pair_20261019.py']


def verify_approved(root,report_path):
    assert inputs.sha(report_path)==EXPECTED_REPORT_SHA,'Reviewed preflight hash changed'
    review=inputs.read(root/FOLDER/'training_review.json')
    assert review['seed42_pilot_allowed'] and not review['three_seed_eligible']
    assert review['preflight_sha256']==EXPECTED_REPORT_SHA and review['input_manifest_sha256']==EXPECTED_INPUT_SHA
    assert review['shared_validation_count']==74 and review['max_steps_per_arm']==200 and review['seed']==42
    for n,h in review['launcher_sha256_lf'].items():assert inputs.lfsha(root/n)==h,'Launcher changed: '+n
    m=inputs.verify(root);report=inputs.read(report_path)
    assert inputs.sha(root/inputs.FOLDER/'input_manifest.json')==EXPECTED_INPUT_SHA
    assert report['code_commit']==review['preflight_code_commit']=='1d4e0672cabc9e37ca778354e23766b0dde23a84'
    assert report['input_manifest_sha256']==EXPECTED_INPUT_SHA and report['input_manifest']==m
    assert report['runtime_source_lock']==inputs.prior.verify_runtime_lock(root)
    assert report['passed'] and not report['failures'] and report['real_tokenizer'] and report['swift_version']=='4.4.2'
    assert report['original_inputs_unchanged'] and report['inference_checks']==len(report['encodings'])==1332
    assert [(x['file'],x['unit_id']) for x in report['encodings']]==[(n,i) for n,ids in m['unit_ids'].items() for i in ids]
    assert all(0<x['supervised_tokens']<=x['tokens']<=8192 for x in report['encodings'])
    masked={(x['file'],x['unit_id']):x['unknown_bits_masked'] for x in report['encodings'] if x['unknown_bits_masked']}
    assert masked=={(x['file'],x['unit_id']):x['unknown_zero_gradient'] for x in report['loss_checks']}
    assert len(report['loss_checks'])==26 and sum(masked.values())==86
    assert sum(x['unknown_entity_bits_masked'] for x in report['encodings'])==20
    assert sum(x['unknown_relation_bits_masked'] for x in report['encodings'])==66
    assert all(x['known_nonzero_gradient']>0 and abs(x['synthetic_loss']-math.log(3))<1e-5 for x in report['loss_checks'])
    assert not any(report[k] for k in ('model_calls','model_weights_loaded','model_training_started','training_ready'))
    assert report['max_tokens_by_file']==review['verified_max_tokens']=={
        n:max(x['tokens'] for x in report['encodings'] if x['file']==n) for n in m['counts']}
    assert min(x['supervised_tokens'] for x in report['encodings'])==review['minimum_supervised_tokens']==12
    return dict(manifest=m,preflight_sha256=EXPECTED_REPORT_SHA,manifest_sha256=EXPECTED_INPUT_SHA,
                checked_records=1332,masked_positions=86,review=review)


def build_command(root,model,out,arm):
    command=baseline.build_command(root,model,out,arm)
    options=dict(zip(command[2::2],command[3::2]))
    options.update({'--dataset':str(root/inputs.FOLDER/(arm+'_train.jsonl')),
        '--val_dataset':str(root/inputs.FOLDER/'validation.jsonl'),
        '--template':inputs.policy.TEMPLATE,'--external_plugins':str(root/PLUGIN),
        '--template_backend':'swift','--lazy_tokenize':'false','--dataset_num_proc':'1',
        '--load_from_cache_file':'false','--use_logits_to_keep':'false','--is_binary_loss_scale':'true',
        '--enable_channel_loss':'false','--enable_dft_loss':'false','--use_liger_kernel':'false',
        '--remove_unused_columns':'true'})
    # Actual SDK maps CLI 'delete' to template 'raise'. With strict=True the
    # observed RowPreprocessor rethrows encoding errors rather than deleting.
    assert options['--truncation_strategy']=='delete' and options['--strict']=='true'
    return command[:2]+[part for k,v in options.items() for part in (k,v)]


def verify_sdk(root):
    assert importlib.metadata.version('ms-swift')=='4.4.2','Unexpected Swift version'
    lock=inputs.prior.verify_runtime_lock(root)
    sdk=Path(next(iter(importlib.util.find_spec('swift').submodule_search_locations)))
    for name,info in lock['files'].items():assert inputs.sha(sdk.parent/name)==info['sha256'],'SDK source changed: '+name


def runtime_checks(root,model,out,checked):
    verify_sdk(root)
    for n,h in checked['manifest']['tokenizer_config_sha256'].items():
        assert Path(n).name==n and inputs.sha(model/n)==h,'Model/tokenizer configuration changed: '+n
    from swift.arguments import SftArguments
    supported={f.name for f in fields(SftArguments)}
    for arm in ('control','repair'):
        command=build_command(root,model,out/arm,arm)
        assert all(n[2:] in supported for n in command[2::2]),'Swift parser lacks audited option'
    from swift.dataset import load_dataset
    for name,count in checked['manifest']['counts'].items():
        saved=inputs.rows(root/inputs.FOLDER/(name+'.jsonl'))
        dataset,validation=load_dataset([str(root/inputs.FOLDER/(name+'.jsonl'))],split_dataset_ratio=0,seed=42,shuffle=False,strict=True)
        assert validation is None and len(dataset)==len(saved)==count,'Dataset count or implicit split changed'
        assert [inputs.uid(row) for row in dataset]==checked['manifest']['unit_ids'][name],'Dataset order/membership changed'
        for source,row in zip(saved,dataset):inputs.check_loaded_row(source,row)


def require_idle_gpu():
    gpu=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert gpu.returncode==0,'GPU0 occupancy query failed'
    # This parent only queries BF16 capability; it never loads model weights.
    busy=[s.strip() for s in gpu.stdout.splitlines() if s.strip() and s.strip()!=str(os.getpid())]
    assert not busy,'GPU0 occupied by processes: '+', '.join(busy)


def inspect_completion(out):
    return prior_train.inspect_completion(out)


def train(root,model,out,report_path,run=False,launch=baseline.stream_launch):
    checked=verify_approved(root,report_path)
    assert not out.exists(),'Preserve existing training run'
    runtime_checks(root,model,out,checked)
    print('TRAINING_RUNTIME_INPUT_CHECKS_OK=true',flush=True)
    if not run:
        return dict(status='dry_run_verified',training_started=False,commands=[build_command(root,model,out/a,a) for a in ('control','repair')])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('WORLD_SIZE','1')=='1','Expected single GPU0 process'
    require_idle_gpu()
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'GPU0/BF16 unavailable'
    out.mkdir(parents=True)
    result=dict(status='running',seed=42,max_steps_per_arm=200,arms={},verification=checked,
        preflight_sha256=EXPECTED_REPORT_SHA,input_manifest_sha256=EXPECTED_INPUT_SHA,
        candidate_profile=inputs.profile.VERSION,question_format_version='ecospec-partial-label-v226',
        template_type=inputs.policy.TEMPLATE,token_level_loss_masking=True,unknown_entity_bits=20,unknown_relation_bits=66,
        use_logits_to_keep=False,clause_guard_used_in_training=False,
        training_code_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        expansion_to_three_seeds=False,dev_evaluation_completed=False,control_is_incomplete_supervision_ablation=True,
        new_source_examples_shared=True,checkpoint_selection='minimum shared 74-record internal validation loss',
        limitations=checked['manifest']['limitations'])
    arm=None
    try:
        for arm in ('control','repair'):
            assert verify_approved(root,report_path)==checked,'Protected inputs changed before arm'
            verify_sdk(root)
            for n,h in checked['manifest']['tokenizer_config_sha256'].items():assert inputs.sha(model/n)==h,'Model configuration changed'
            require_idle_gpu()
            command=build_command(root,model,out/arm,arm)
            result['arms'][arm]=dict(status='running',command=command,output=str(out/arm))
            inputs.save(out/'pair_manifest.json',result)
            print('STARTING_ARM='+arm+' SEED=42 MAX_STEPS=200 PARTIAL_LABEL_MASK=true',flush=True)
            code=launch(command,out/(arm+'.log'))
            assert code==0,f'{arm} failed with exit code {code}; subsequent arm not started'
            result['arms'][arm].update(status='complete',completion=inspect_completion(out/arm))
            assert verify_approved(root,report_path)==checked,'Protected inputs changed during arm'
            verify_sdk(root)
            for n,h in checked['manifest']['tokenizer_config_sha256'].items():assert inputs.sha(model/n)==h,'Model configuration changed'
            inputs.save(out/'pair_manifest.json',result)
        result['status']='complete'
    except BaseException as exc:
        if arm in result['arms'] and result['arms'][arm]['status']=='running':result['arms'][arm]['status']='failed'
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:inputs.save(out/'pair_manifest.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,default=ROOT/REPORT);parser.add_argument('--run',action='store_true')
    a=parser.parse_args();result=train(ROOT,a.model,a.out,a.preflight,a.run)
    print(inputs.dumps([{k:v for k,v in result.items() if k in ('status','training_started','seed','max_steps_per_arm')}]),flush=True)
