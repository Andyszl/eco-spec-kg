"""Diagnose fixed adapters on existing train/internal-validation supervision only."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json,os,socket,subprocess
from pathlib import Path
import evaluate_cross_source_pair_20261016 as ev
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.io_utils import write_jsonl

ROOT=Path(__file__).resolve().parents[1]
inputs=ev.training.inputs
LOCK='experiments/cross_source_probe_20261017/protocol.json'


def semantic_ids(graph):
    return {r['relation_id'] for r in graph['relations'] if '|'.join((r['head_type'],r['relation_type'],r['tail_type'])) in inputs.ROLES}


def build_cases(root):
    ev.training.verify_approved(root,root/ev.training.REPORT)
    units={u['unit_id']:u for u in inputs.rows(root/inputs.OLD/'units.jsonl')+inputs.rows(root/inputs.SEED/'source_units.jsonl')}
    control={inputs.uid(r):r for name in ('control_train','validation') for r in inputs.rows(root/inputs.FOLDER/(name+'.jsonl'))}
    reviewed={r['unit_id']:r for r in inputs.read(root/inputs.SEED/'supervision_review.json')['cases']}
    old_review=inputs.read(root/'experiments/method_scope_review_20261016/source_review.json')['cases']
    cases=[]
    for split,name in (('train','repair_train'),('validation','validation')):
        for row in inputs.rows(root/inputs.FOLDER/(name+'.jsonl')):
            uid=inputs.uid(row);unit=units[uid];raw,full,mapping=ev.profile.candidates_with_alias_map(unit)
            role_ids=semantic_ids(full)
            positive=role_ids & set(json.loads(row['messages'][-1]['content'])['selected_relation_ids'])
            if not positive:continue
            q=json.loads(row['messages'][1]['content']);seen={r['id'] for r in q['candidate_relations']}
            full_ids={r['relation_id'] for r in full['relations']}
            assert seen<=full_ids and positive<=seen
            unknown=full_ids-seen
            negative={r['relation_id'] for r in reviewed.get(uid,{}).get('confirmed_negative_relations',[])}
            for r in old_review:
                if r['unit_id']==uid and r['review_status']=='exclude_generic_or_bad_boundary':
                    negative.add(mapping['relation_id_map'][r['candidate_relation_id']])
            assert not (negative&unknown or positive&unknown or negative&positive)
            as_seen=deepcopy(full);as_seen['relations']=[r for r in full['relations'] if r['relation_id'] in seen]
            assert row['messages'][:2]==[{'role':'system','content':ev.ex.build_llm_selection_messages(unit,as_seen)[0]},
                                       {'role':'user','content':ev.ex.build_llm_selection_messages(unit,as_seen)[1]}]
            original=set(json.loads(control[uid]['messages'][-1]['content'])['selected_relation_ids'])
            variants=[('full',full)]+([('as_supervised',as_seen)] if unknown else [])
            for view,graph in variants:
                system,prompt=ev.ex.build_llm_selection_messages(unit,graph)
                # ASCII JSON character count is a conservative completion-token bound.
                bound=len(json.dumps({'selected_entity_ids':[e['entity_id'] for e in graph['entities']],
                                      'selected_relation_ids':[r['relation_id'] for r in graph['relations']]}))+1
                assert bound<=3072,'Diagnostic exceeds fixed completion budget'
                cases.append({'case_id':uid+':'+view,'unit':unit,'split':split,'view':view,'graph':graph,
                    'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
                    'saved_messages':row['messages'][:2], 'positive_ids':sorted(positive),
                    'new_positive_ids':sorted(positive-original) if split=='train' else [],
                    'unknown_ids':sorted(unknown&role_ids),'confirmed_negative_ids':sorted(negative&role_ids),
                    'supervised_exclusion_ids':sorted((seen&role_ids)-positive-negative),
                    'semantic_candidate_ids':sorted(semantic_ids(graph)), 'completion_bound':bound,
                    'prompt_hash':sha256_json({'system':system,'prompt':prompt}),
                    'candidate_hash':sha256_json({k:graph[k] for k in ('entities','relations')})})
    return cases


def verify_cases(root,cases):
    lock=ev.read(root/LOCK)
    assert lock['training_manifest_sha256']==ev.EXPECTED_PAIR_SHA
    for n,h in lock['source_sha256_lf'].items():assert inputs.lfsha(root/n)==h,'Probe code changed: '+n
    assert {c['case_id']:sha256_json(c) for c in cases}==lock['case_hashes'],'Probe cases changed'
    return lock


def score_case(case,selected):
    selected=set(selected)&set(case['semantic_candidate_ids']);positive=set(case['positive_ids'])
    return {'case_id':case['case_id'],'unit_id':case['unit']['unit_id'],'split':case['split'],'view':case['view'],
        'positive_ids':case['positive_ids'],'selected_positive_ids':sorted(selected&positive),
        'missed_positive_ids':sorted(positive-selected),'selected_new_positive_ids':sorted(selected&set(case['new_positive_ids'])),
        'selected_confirmed_negative_ids':sorted(selected&set(case['confirmed_negative_ids'])),
        'selected_unknown_ids':sorted(selected&set(case['unknown_ids'])),
        'selected_supervision_exclusion_ids':sorted(selected&set(case['supervised_exclusion_ids']))}


def deploy_command(root,manifest,port):
    cmd=ev.deploy_command(root,manifest,port)
    cmd[cmd.index('--max_length')+1]='16384'
    return cmd


def run_probe(root,train,out,base_url='http://127.0.0.1:8018/v1'):
    assert not out.exists(),'Preserve existing probe output'
    manifest=ev.inspect_pair(root,train);cap=ev.check_service(base_url)
    cases=build_cases(root);lock=verify_cases(root,cases)
    out.mkdir(parents=True)
    ev.save(out/'cases.json',cases)
    ev.save(out/'protocol.json',{'training_manifest_sha256':ev.EXPECTED_PAIR_SHA,'probe_lock_sha256':ev.sha(root/LOCK),
        'case_hashes':lock['case_hashes'],'service':cap,'code_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'context_limit':16384,'dev_context_unchanged':8192,'not_a_dev_gate_run':True})
    results={};raws=[];predictions=[]
    try:
        for arm,model in ev.MODELS.items():
            provider=ev.Provider(base_url=base_url,api_key='local-token',timeout=600,enable_thinking=False)
            cfg={'experiment_id':'cross-source-supervision-fit-probe','backend':'llm','model':model,
                **ev.read(root/ev.PROTOCOL)['selection_protocol'],'selection_decoding':ev.VERSION}
            for n in ('model','seed','temperature','max_tokens','enable_thinking','selection_decoding'):setattr(provider,n,cfg[n])
            values=[]
            for c in cases:
                print('PROBING_MODEL='+arm+' CASE='+c['case_id'],flush=True)
                try:
                    pred,raw,ph=ev.ex._llm_select(provider,c['unit'],c['graph'])
                except BaseException as error:
                    raws.append({'model':arm,'case_id':c['case_id'],'status':'error','message':str(error),'provider_response':provider.last_raw_response})
                    raise
                assert ph==c['prompt_hash']
                pred.update(experiment_id=cfg['experiment_id'],model=model,seed=42,prompt_hash=ph,
                    config_hash=sha256_json(cfg),candidate_hash=c['candidate_hash'])
                assert not _validate_unit_prediction(c['unit'],pred)
                raws.append({'model':arm,'case_id':c['case_id'],'raw_response':raw,'response_sha256':sha256_json(raw),
                    'prompt_hash':ph,'provider_response':provider.last_raw_response})
                predictions.append({'model':arm,'case_id':c['case_id'],'prediction':pred})
                value=score_case(c,{r['relation_id'] for r in pred['relations']})
                value['relation_details']=[{'relation_id':r['relation_id'],'head':r['head_name'],'type':r['relation_type'],'tail':r['tail_name']}
                    for r in c['graph']['relations'] if r['relation_id'] in c['semantic_candidate_ids']]
                value['usage']=provider.last_raw_response.get('usage')
                values.append(value)
            results[arm]={'cases':values,'by_split_view':{}}
            for split in ('train','validation'):
                for view in ('full','as_supervised'):
                    subset=[v for v in values if v['split']==split and v['view']==view]
                    results[arm]['by_split_view'][split+':'+view]={'units':len(subset),
                        **{k:sum(len(v[k]) for v in subset) for k in ('positive_ids','selected_positive_ids','missed_positive_ids',
                            'selected_new_positive_ids','selected_confirmed_negative_ids','selected_unknown_ids','selected_supervision_exclusion_ids')}}
    finally:
        write_jsonl(out/'raw_responses.jsonl',raws);write_jsonl(out/'predictions.jsonl',predictions)
    assert ev.inspect_pair(root,train)==manifest and verify_cases(root,build_cases(root))==lock
    result={'status':'complete','runs':results,'model_requests':len(cases)*3,'training_started':False,
        'dev_gate_replaced':False,'new_checkpoint_selected':False,'three_seed_eligible':False,
        'reference_status':'Existing reused/assistant-reviewed supervision; unknown and mere supervision exclusions are not verified false positives.',
        'scope':'Train fit and internal validation diagnosis; views overlap and DEM spellings are not independent samples.'}
    ev.save(out/'summary.json',result)
    print(json.dumps({'requests':result['model_requests'],'runs':{a:v['by_split_view'] for a,v in results.items()}},ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--train',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    manifest=ev.inspect_pair(ROOT,args.train);verify_cases(ROOT,build_cases(ROOT))
    assert ev.importlib.metadata.version('ms-swift')=='4.4.2'
    for n,h in manifest['verification']['manifest']['tokenizer_config_sha256'].items():assert ev.sha(Path(manifest['arms']['control']['command'][3])/n)==h
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('WORLD_SIZE','1')=='1'
    cmd=deploy_command(ROOT,manifest,8018)
    from swift.arguments import DeployArguments
    supported={f.name for f in fields(DeployArguments)};assert all(n[2:] in supported for n in cmd if n.startswith('--'))
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
