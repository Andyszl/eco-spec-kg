"""Fixed v219-scope adapters: train fit and same-graph scope diagnosis, no training."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json,os,socket,subprocess
from pathlib import Path
import evaluate_full_graph_scope_pair_20261019 as ev
import probe_cross_source_learning_20261017 as reference
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.selection_decoding_v2 import candidate_ids,check_selection

ROOT=Path(__file__).resolve().parents[1]
LOCK='experiments/full_graph_scope_probe_20261019/protocol.json'


def build_cases(root):
    """Reuse the reviewed cohort; every view retains the identical complete graph."""
    ev.training.verify_approved(root,root/ev.training.REPORT)
    source=[c for c in reference.build_cases(root) if c['view']=='full']
    inputs=ev.training.inputs
    saved={inputs.uid(r):r for name in ('repair_train','validation')
           for r in inputs.rows(root/inputs.FOLDER/(name+'.jsonl'))}
    cases=[]
    for old in source:
        unit=old['unit'];graph=old['graph'];row=saved[unit['unit_id']]
        assert graph==ev.profile.candidates_for_unit(unit)
        q=json.loads(row['messages'][1]['content'])
        unscored=q['decision_scope']['unscored_relation_ids']
        assert row['messages'][:2]==[{'role':'system','content':ev.scope.build_messages(unit,graph,unscored)[0]},
                                    {'role':'user','content':ev.scope.build_messages(unit,graph,unscored)[1]}]
        target=json.loads(row['messages'][-1]['content'])
        ev.scope.validate_selection(row['messages'][1]['content'],target)
        assert set(old['positive_ids'])==set(old['semantic_candidate_ids'])&set(target['selected_relation_ids'])
        assert set(old['unknown_ids'])==set(old['semantic_candidate_ids'])&set(unscored)
        assert not set(old['confirmed_negative_ids'])&set(unscored)
        for view,excluded in [('full',[])]+([('as_supervised',unscored)] if unscored else []):
            system,prompt=ev.scope.build_messages(unit,graph,excluded)
            c={**deepcopy(old),'case_id':unit['unit_id']+':'+view,'view':view,
               'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
               'saved_messages':row['messages'][:2], 'unscored_relation_ids':excluded,
               'prompt_hash':sha256_json({'system':system,'prompt':prompt})}
            assert c['completion_bound']<=3072
            cases.append(c)
    assert len(cases)==16 and len({c['unit']['unit_id'] for c in cases})==10
    return cases


def verify_cases(root,cases):
    lock=ev.read(root/LOCK)
    assert lock['training_manifest_sha256']==ev.EXPECTED_PAIR_SHA
    assert lock['input_manifest_sha256']==ev.sha(root/ev.training.inputs.FOLDER/'input_manifest.json')
    for n,h in lock['source_sha256_lf'].items():
        assert ev.training.inputs.lfsha(root/n)==h,'Probe code changed: '+n
    assert {c['case_id']:sha256_json(c) for c in cases}==lock['case_hashes'],'Probe cases changed'
    return lock


def assemble_case(case,raw):
    # Selecting an unscored but valid candidate is a diagnostic observation.
    # Record it rather than silently removing it or failing the entire probe.
    prompt=case['messages'][1]['content']
    selection=check_selection(raw,*candidate_ids(prompt))
    graph=case['graph'];relations={r['relation_id']:r for r in graph['relations']}
    selected=[relations[i] for i in selection['selected_relation_ids']]
    endpoints={r[k] for r in selected for k in ('head_id','tail_id')}
    endpoints.update(selection['selected_entity_ids'])
    return {**graph,'backend':'llm','decision_scope_version':ev.scope.VERSION,
            'entities':[e for e in graph['entities'] if e['entity_id'] in endpoints],
            'relations':selected,'no_relation_reason':'' if selected else 'llm_selected_no_relation'}


def score_case(case,selected):
    value=reference.score_case(case,selected)
    value['selected_unscored_ids']=sorted(set(selected)&set(case['unscored_relation_ids']))
    return value


def deploy_command(root,manifest,port):
    cmd=ev.deploy_command(root,manifest,port)
    cmd[cmd.index('--max_length')+1]='16384'
    return cmd


def run_probe(root,train,out,base_url='http://127.0.0.1:8018/v1'):
    assert not out.exists(),'Preserve existing diagnostic output'
    manifest=ev.inspect_pair(root,train)
    cases=build_cases(root);lock=verify_cases(root,cases);cap=ev.check_service(base_url)
    out.mkdir(parents=True)
    ev.save(out/'cases.json',cases)
    ev.save(out/'protocol.json',{'training_manifest_sha256':ev.EXPECTED_PAIR_SHA,
        'probe_lock_sha256':ev.sha(root/LOCK),'case_hashes':lock['case_hashes'],'service':cap,
        'code_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'context_limit':16384,'formal_dev_context_limit':8192,'not_a_dev_gate_run':True,
        'scope_only_changes_unscored_relation_ids':True,'candidate_graph_changed':False})
    results={};raws=[];predictions=[]
    try:
        for arm,model in ev.MODELS.items():
            provider=ev.Provider(base_url=base_url,api_key='local-token',timeout=600,enable_thinking=False)
            cfg={'experiment_id':'full-graph-scope-fit-probe','backend':'llm','model':model,
                 **ev.read(root/ev.PROTOCOL)['selection_protocol'],'selection_decoding':ev.VERSION}
            for n in ('model','seed','temperature','max_tokens','enable_thinking','selection_decoding'):
                setattr(provider,n,cfg[n])
            values=[]
            for c in cases:
                print('PROBING_MODEL='+arm+' CASE='+c['case_id'],flush=True)
                try:
                    raw=provider.complete(*(m['content'] for m in c['messages']))
                    pred=assemble_case(c,raw)
                    pred.update(experiment_id=cfg['experiment_id'],model=model,seed=42,
                                prompt_hash=c['prompt_hash'],config_hash=sha256_json(cfg),candidate_hash=c['candidate_hash'])
                    assert not _validate_unit_prediction(c['unit'],pred)
                except BaseException as error:
                    raws.append({'model':arm,'case_id':c['case_id'],'status':'error','message':str(error),
                                 'provider_response':provider.last_raw_response})
                    raise
                raws.append({'model':arm,'case_id':c['case_id'],'raw_response':raw,'response_sha256':sha256_json(raw),
                             'prompt_hash':c['prompt_hash'],'provider_response':provider.last_raw_response})
                predictions.append({'model':arm,'case_id':c['case_id'],'prediction':pred})
                value=score_case(c,{r['relation_id'] for r in pred['relations']})
                value['relation_details']=[{'relation_id':r['relation_id'],'head':r['head_name'],
                    'type':r['relation_type'],'tail':r['tail_name']} for r in c['graph']['relations']
                    if r['relation_id'] in c['semantic_candidate_ids']]
                value['usage']=provider.last_raw_response.get('usage')
                values.append(value)
            grouped={}
            for split in ('train','validation'):
                for view in ('full','as_supervised'):
                    subset=[v for v in values if v['split']==split and v['view']==view]
                    grouped[split+':'+view]={'units':len(subset),**{k:sum(len(v[k]) for v in subset)
                        for k in ('positive_ids','selected_positive_ids','missed_positive_ids','selected_new_positive_ids',
                                  'selected_confirmed_negative_ids','selected_unknown_ids',
                                  'selected_supervision_exclusion_ids','selected_unscored_ids')}}
            results[arm]={'cases':values,'by_split_view':grouped}
    finally:
        # LF output is part of the transfer/replay contract.
        for name,data in (('raw_responses.jsonl',raws),('predictions.jsonl',predictions)):
            (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8',newline='\n')
    assert ev.inspect_pair(root,train)==manifest and verify_cases(root,build_cases(root))==lock
    result={'status':'complete','runs':results,'model_requests':len(cases)*3,'training_started':False,
        'dev_gate_replaced':False,'new_checkpoint_selected':False,'three_seed_eligible':False,
        'reference_status':'Reused reviewed supervision; unknown and ordinary supervision exclusions are not verified false positives.',
        'scope':'Train fit and internal validation; paired views overlap; DEM spellings are one logical source.',
        'decision_scope_version':ev.scope.VERSION,'candidate_graph_changed':False}
    ev.save(out/'summary.json',result)
    print(json.dumps({'requests':result['model_requests'],'runs':{a:v['by_split_view'] for a,v in results.items()}},ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    manifest=ev.inspect_pair(ROOT,args.train);verify_cases(ROOT,build_cases(ROOT))
    assert ev.importlib.metadata.version('ms-swift')=='4.4.2'
    for n,h in manifest['verification']['manifest']['tokenizer_config_sha256'].items():
        assert ev.sha(Path(manifest['arms']['control']['command'][3])/n)==h
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('WORLD_SIZE','1')=='1'
    cmd=deploy_command(ROOT,manifest,8018)
    from swift.arguments import DeployArguments
    supported={f.name for f in fields(DeployArguments)}
    assert all(n[2:] in supported for n in cmd if n.startswith('--'))
    gpu=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU0 unavailable or occupied'
    with socket.socket() as sock:sock.bind(('127.0.0.1',8018))
    args.out.mkdir(parents=True);ev.save(args.out/'launch.json',{'command':cmd,'not_a_dev_gate_run':True})
    try:
        with ev.owned_service(cmd,args.out/'deploy.log','http://127.0.0.1:8018/v1'):
            run_probe(ROOT,args.train,args.out/'probe')
    except BaseException as error:
        ev.save(args.out/'failure.json',{'status':'failed','type':type(error).__name__,'message':str(error)})
        raise


if __name__=='__main__':main()
