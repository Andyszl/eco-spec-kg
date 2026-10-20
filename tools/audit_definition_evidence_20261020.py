"""CPU replay of definition evidence on saved dev predictions; no model calls."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import evaluate_full_graph_scope_pair_20261019 as ev
from ecospec_kg.definition_evidence_v223 import VERSION,attach_definition_evidence
from ecospec_kg.structural_selection_v2 import merge_structural_relations,formula_path_metrics
from ecospec_kg.prediction_validation_v2 import _all_spans,_validate_unit_prediction
from ecospec_kg.evaluation_v2 import _entity_key,_relation_key
LOCK='experiments/definition_evidence_20261020/protocol.json'


def review_training(root):
    inputs=ev.training.inputs
    ev.training.verify_approved(root,root/ev.training.REPORT)
    previous=inputs.previous
    units={u['unit_id']:u for u in inputs.rows(root/previous.OLD/'units.jsonl')+inputs.rows(root/previous.SEED/'source_units.jsonl')}
    records=inputs.rows(root/inputs.FOLDER/'repair_train.jsonl');witnesses=[]
    for row in records:
        u=units[inputs.uid(row)];graph=ev.profile.candidates_for_unit(u)
        p,_=merge_structural_relations(u,graph,{**graph,'entities':[],'relations':[]})
        _,proof=attach_definition_evidence(u,graph,p)
        for b in proof['approved_definition_bindings']:
            witnesses.append({'unit_id':u['unit_id'],'binding':b,
                'definition_span':_all_spans(u)[b['definition_span_id']],
                'approved_span_ids':[b['definition_span_id']]})
    assert witnesses,'No training-source definition witness'
    return {'records':len(records),'witnesses':witnesses,
        'validation_dev_test_used_for_rule_derivation':False,
        'policy':'Existing explicit equation-output and exact indicator definition only; no source/entity/formula ID rules.'}


def write_rows(path,data):
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8',newline='\n')


def run(root,dev,out):
    assert not out.exists(),'Preserve existing offline result'
    lock=ev.read(root/LOCK)
    for n,h in lock['source_sha256_lf'].items():
        assert ev.training.inputs.lfsha(root/n)==h,'Source changed: '+n
    for n,h in lock['dev_files_sha256'].items():
        assert not Path(n).is_absolute() and '..' not in Path(n).parts
        assert ev.sha(dev/n)==h,'Saved dev result changed: '+n
    before_hashes={n:ev.sha(dev/n) for n in lock['dev_files_sha256']}
    training=review_training(root)
    units=ev.read_jsonl(root/'deliveries/human_confirmed_20261002/blind/dev_units.jsonl')
    gold=ev.read_jsonl(root/'deliveries/human_confirmed_20261002/gold/dev_annotations.jsonl')
    summary=ev.read(dev/'summary.json');protocol=ev.read(dev/'evaluation_protocol.json')
    assert summary['status']=='complete' and summary['model_requests']==66
    assert protocol['training_manifest_sha256']==ev.EXPECTED_PAIR_SHA
    for n,h in ev.read(root/ev.PROTOCOL)['evaluation_input_sha256'].items():assert ev.sha(root/n)==h
    unit_map={u['unit_id']:u for u in units};out.mkdir(parents=True)
    results={};all_proofs=[]
    for arm in ev.MODELS:
        raw=ev.read_jsonl(dev/arm/'predictions.jsonl')
        cfg=ev.read(dev/arm/'resolved_config.json')
        baseline,merged,proofs=ev.compare_one(units,gold,cfg,raw)
        assert json.loads(json.dumps(baseline))==summary['runs'][arm]
        assert merged==ev.read_jsonl(dev/arm/'hybrid/predictions.jsonl')
        assert proofs==ev.read_jsonl(dev/arm/'hybrid/structural_proofs.jsonl')
        results[arm]={};dest=out/arm;dest.mkdir()
        for stage,data in (('raw',raw),('structure',merged)):
            updated=[];changes=[]
            for pred in data:
                u=unit_map[pred['unit_id']];graph=ev.profile.candidates_for_unit(u)
                new,proof=attach_definition_evidence(u,graph,pred)
                assert new['relations']==pred['relations']
                assert {e['entity_id'] for e in new['entities']}=={e['entity_id'] for e in pred['entities']}
                for field,key in (('entities',_entity_key),('relations',_relation_key)):
                    assert {key(u['unit_id'],i) for i in new[field]}=={key(u['unit_id'],i) for i in pred[field]}
                new['config_hash']=ev.sha256_json({'parent_config_hash':pred['config_hash'],'evidence_postprocessor':VERSION})
                assert not _validate_unit_prediction(u,new)
                updated.append(new);changes.extend(proof['changes'])
                all_proofs.append({'model':arm,'stage':stage,'unit_id':u['unit_id'],**proof})
            old=formula_path_metrics(units,gold,data);new=formula_path_metrics(units,gold,updated)
            assert all(old[k]==new[k] for k in old if k not in {'details','reference_span_overlap_complete_paths'})
            results[arm][stage]={'changed_entities':len(changes),
                'added_entity_spans':sum(len(c['added_span_ids']) for c in changes),
                'label_metrics_unchanged':True,'formula_paths_before':old,'formula_paths_after':new}
            write_rows(dest/(stage+'_predictions.jsonl'),updated)
    assert before_hashes=={n:ev.sha(dev/n) for n in lock['dev_files_sha256']}
    report={'status':'offline_definition_evidence_verified','policy':VERSION,'model_calls':0,
        'training_started':False,'training_ready':False,'three_seed_eligible':False,
        'training_review':training,'results':results,'original_gate':summary['gate'],
        'dev_before':summary['runs'],'received_results_unchanged':True,
        'scope':'Saved dev postprocessing only; raw/structure label metrics and original gate unchanged. Reference-span overlap is not semantic entailment or independent-source validation.'}
    write_rows(out/'evidence_proofs.jsonl',all_proofs);ev.save(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit must hide GPUs'
    report=run(ROOT,args.dev,args.out)
    print(json.dumps({'status':report['status'],'model_calls':0,
        'paths':{a:{s:v['formula_paths_after']['reference_span_overlap_complete_paths'] for s,v in r.items()} for a,r in report['results'].items()}}),flush=True)
    print('DEFINITION_EVIDENCE_CHECK_OK=true\nTRAINING_READY=false',flush=True)


if __name__=='__main__':main()
