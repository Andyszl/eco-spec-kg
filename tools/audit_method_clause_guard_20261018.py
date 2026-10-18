"""CPU-only saved-prediction postfilter audit, not a new model evaluation or training."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import probe_cross_source_learning_20261017 as probe
from ecospec_kg.method_clause_guard_v218 import RULES,VERSION,rejection_reason,filter_prediction
from ecospec_kg.evaluation_v2 import _entity_key,_relation_key
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
read=probe.ev.read;save=probe.ev.save;rows=probe.inputs.rows;sha=probe.ev.sha
LOCK='experiments/method_clause_guard_20261018/protocol.json'


def review_training(root):
    probe.ev.training.verify_approved(root,root/probe.ev.training.REPORT)
    records=rows(root/probe.inputs.FOLDER/'repair_train.jsonl')
    positive=[]
    for row in records:
        q=json.loads(row['messages'][1]['content']);t=json.loads(row['messages'][-1]['content']);es={e['id']:e for e in q['candidate_entities']}
        ids=set(t['selected_entity_ids'])|{r[s] for r in q['candidate_relations'] if r['id'] in t['selected_relation_ids'] for s in ('head','tail')}
        positive.extend(es[i]['name'] for i in ids if es[i]['type']=='method')
    negative=[]
    for c in read(root/'experiments/method_scope_review_20261016/source_review.json')['cases']:
        if c['review_status']=='exclude_generic_or_bad_boundary':negative.append({'unit_id':c['unit_id'],'name':c['tail']})
    for c in read(root/probe.inputs.SEED/'supervision_review.json')['cases']:
        if c['split']=='train':negative.extend({'unit_id':c['unit_id'],'name':r['edge'][2]} for r in c['confirmed_negative_relations'] if r['edge'][1]=='obtained_by')
    witnesses={name:[n for n in negative if rejection_reason(n['name'])==name] for name,_ in RULES}
    rejected=sorted({n for n in positive if rejection_reason(n)})
    assert not rejected,'Known positive method name rejected'
    assert all(witnesses.values()),'Rule lacks a reviewed training witness'
    return {'records':len(records),'known_positive_method_occurrences':len(positive),'known_positive_method_names':sorted(set(positive)),
        'rejected_known_positive_names':rejected,'negative_witnesses':witnesses,
        'rule_design_sources':'Reviewed train only; no validation/dev name fitting; no blanket calculation-suffix or method whitelist rule.'}


def keyset(data,field,key):return {key(p['unit_id'],x) for p in data for x in p[field]}


def run(root,probe_dir,dev_dir,out):
    assert not out.exists(),'Preserve previous offline report'
    lock=read(root/LOCK)
    for name,h in lock['source_sha256_lf'].items():assert probe.inputs.lfsha(root/name)==h,'Code changed: '+name
    locations={'probe':probe_dir,'dev':dev_dir}
    for name,h in lock['files'].items():
        group,relative=name.split('/',1)
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert sha(locations[group]/relative)==h,'Saved result changed: '+name
    training=review_training(root)
    cases=probe.build_cases(root);assert cases==read(probe_dir/'cases.json')
    case_map={c['case_id']:c for c in cases};original_probe=read(probe_dir/'summary.json')
    assert original_probe['status']=='complete' and original_probe['model_requests']==48
    original_dev=read(dev_dir/'summary.json')
    assert original_dev['status']=='complete' and original_dev['model_requests']==66
    units=rows(root/'deliveries/human_confirmed_20261002/blind/dev_units.jsonl')
    gold=rows(root/'deliveries/human_confirmed_20261002/gold/dev_annotations.jsonl');unit_map={u['unit_id']:u for u in units}
    out.mkdir(parents=True)
    probe_results={};new_dev={};all_proofs=[]
    for arm in probe.ev.MODELS:
        values=[]
        for item in rows(probe_dir/'predictions.jsonl'):
            if item['model']!=arm:continue
            c=case_map[item['case_id']];pred,proof=filter_prediction(item['prediction'])
            assert not _validate_unit_prediction(c['unit'],pred)
            old_selected={r['relation_id'] for r in item['prediction']['relations']}
            selected={r['relation_id'] for r in pred['relations']}
            assert old_selected&set(c['positive_ids'])==selected&set(c['positive_ids']),'Reference positive selection removed'
            assert old_selected&set(c['unknown_ids'])==selected&set(c['unknown_ids']),'Unknown edge was quarantined at inference'
            values.append(probe.score_case(c,selected))
            all_proofs.append({'scope':'probe','model':arm,'case_id':c['case_id'],**proof})
        probe_results[arm]={}
        for split in ('train','validation'):
            for view in ('full','as_supervised'):
                subset=[v for v in values if v['split']==split and v['view']==view]
                probe_results[arm][split+':'+view]={'units':len(subset),**{k:sum(len(v[k]) for v in subset) for k in
                    ('positive_ids','selected_positive_ids','missed_positive_ids','selected_confirmed_negative_ids','selected_unknown_ids','selected_supervision_exclusion_ids')}}
        raw=rows(dev_dir/arm/'predictions.jsonl');combined=rows(dev_dir/arm/'hybrid/predictions.jsonl')
        filtered=[]
        for stage,data in (('raw',raw),('combined',combined)):
            changed=[]
            for pred in data:
                new,proof=filter_prediction(pred);assert not _validate_unit_prediction(unit_map[pred['unit_id']],new)
                changed.append(new);all_proofs.append({'scope':'dev','model':arm,'stage':stage,'unit_id':pred['unit_id'],**proof})
            for field,key in (('entities',_entity_key),('relations',_relation_key)):
                g=keyset(gold,field,key)
                assert keyset(changed,field,key)<=keyset(data,field,key)
                assert keyset(changed,field,key)&g==keyset(data,field,key)&g,'Frozen dev true positive removed'
            filtered.append(changed)
        cfg=read(dev_dir/arm/'resolved_config.json')
        baseline,_,_=probe.ev.compare_one(units,gold,cfg,raw)
        assert json.loads(json.dumps(baseline))==original_dev['runs'][arm]
        report,merged,_=probe.ev.compare_one(units,gold,{**cfg,'postprocessor':VERSION},filtered[0])
        assert all(a[k]==b[k] for a,b in zip(merged,filtered[1]) for k in ('entities','relations')),'Filter and structure changed ordering semantics'
        new_dev[arm]=report
        dest=out/arm;dest.mkdir()
        write=lambda name,data:probe.write_jsonl(dest/name,data)
        write('filtered_raw_predictions.jsonl',filtered[0]);write('filtered_combined_predictions.jsonl',filtered[1])
    report={'status':'offline_saved_predictions_verified','policy':VERSION,'model_calls':0,'training_started':False,
        'training_ready':False,'rules_review':training,'probe_before':{a:r['by_split_view'] for a,r in original_probe['runs'].items()},
        'probe_after':probe_results,'dev_before':original_dev['runs'],'dev_after':new_dev,
        'original_gate':original_dev['gate'],'offline_gate_after_filter':probe.ev.strict_gate(new_dev['control'],new_dev['repair']),
        'scope':'Reused predictions only, unchanged candidate prompts/models/checkpoints; not a new inference result or independent test.'}
    probe.write_jsonl(out/'filter_proofs.jsonl',all_proofs);save(out/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--dev',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit must hide GPUs'
    result=run(ROOT,args.probe,args.dev,args.out)
    print(json.dumps({'dev_after':{a:{s:r[s] for s in ('before','after')} for a,r in result['dev_after'].items()},
        'probe_after':result['probe_after'],'offline_gate':result['offline_gate_after_filter']},ensure_ascii=False))
    print('CLAUSE_GUARD_CHECK_OK=true\nTRAINING_READY=false',flush=True)
