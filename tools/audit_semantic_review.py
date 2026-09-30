"""Separate code, source and label effects; compare only training candidates."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import audit_sum_candidates as old
from build_semantic_review import digest, read_rows, write_json, validate_records
from ecospec_kg import analysis_v2
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, CANDIDATE_GENERATOR_VERSION

BASELINE = '9b8e4adb5388a6dccc8176b9dd129bddf6157599'
FIELDS = ('covered_entity_count','gold_entity_count','covered_relation_count',
          'gold_relation_count','entity_recall_upper_bound','relation_recall_upper_bound',
          'entity_candidate_negative_count','relation_candidate_negative_count')


def audit(base, reviewed, out):
    if out.exists(): raise ValueError('output exists; refusing to overwrite')
    if CANDIDATE_GENERATOR_VERSION != 'structure-aware-rule-v2.6':
        raise ValueError('expected v2.6')
    manifest=json.loads((reviewed/'manifest.json').read_text(encoding='utf-8'))
    for key,path in [('units','blind/train_units.jsonl'),('annotations','gold/train_annotations.jsonl')]:
        if digest(base/path)!=manifest['parent_input_hashes'][key]: raise ValueError('parent mismatch')
    for item in manifest['files']:
        if digest(reviewed/item['path'])!=item['sha256']: raise ValueError('reviewed hash mismatch')
    repo=Path(__file__).resolve().parents[1]
    old.BASELINE=BASELINE
    baseline=old.load_baseline(repo)
    packages={'rc1':base,'rc2':reviewed}
    summary={'baseline_commit':BASELINE,'candidate_generator':CANDIDATE_GENERATOR_VERSION,
             'source_hashes':{n:digest(repo/'src/ecospec_kg'/n) for n in ('formula_symbols.py','extractor_v2.py')},
             'matrix':{},'code_differences':{},'remaining_reviewed_false_relations':[]}
    for source,package in packages.items():
        units=read_rows(package/'blind/train_units.jsonl')
        for labels,gold_package in packages.items():
            annotations=read_rows(gold_package/'gold/train_annotations.jsonl')
            # Crossed sources keep original span identities, so evidence remains addressable.
            validate_records(units,annotations)
            for version,extractor in [('v2.5',baseline),('v2.6',RuleCandidateExtractorV2)]:
                key=f'{version}__{source}_source__{labels}_labels'
                with patch.object(analysis_v2,'RuleCandidateExtractorV2',extractor), \
                     patch.object(analysis_v2,'CANDIDATE_GENERATOR_VERSION','structure-aware-rule-'+version), \
                     patch.object(analysis_v2,'_write_workbook',return_value=None):
                    report=analysis_v2.analyze_training_v2(package/'blind/train_units.jsonl',gold_package/'gold/train_annotations.jsonl',out/key)
                summary['matrix'][key]={f:report[f] for f in FIELDS}
        differences=[]
        for unit in units:
            a=old.relation_keys(baseline().predict_unit(unit))
            b=old.relation_keys(RuleCandidateExtractorV2().predict_unit(unit))
            if a!=b: differences.append({'unit_id':unit['unit_id'],'added':sorted(b-a),'removed':sorted(a-b)})
        summary['code_differences'][source]=differences
    revised_units={u['unit_id']:u for u in read_rows(reviewed/'blind/train_units.jsonl')}
    revised_gold={u['unit_id']:u for u in read_rows(reviewed/'gold/train_annotations.jsonl')}
    spec=json.loads((reviewed/'review/decisions.json').read_text(encoding='utf-8'))
    for decision in spec['units']:
        uid=decision['unit_id']
        prediction=RuleCandidateExtractorV2().predict_unit(revised_units[uid])
        triples={(r['head_name'],r['relation_type'],r['tail_name']) for r in prediction['relations']}
        for f in decision.get('formulas',[]):
            name=f"公式（{f['number']}）"
            for relation,role in [('has_input','inputs'),('has_output','outputs')]:
                actual={t for h,r,t in triples if h==name and r==relation}
                if actual!=set(f[role]): raise ValueError(f'reviewed role mismatch: {uid} {name} {relation}: {actual}')
        if uid=='5954b773ec29497b':
            if any(h in ('i','j','k','n','m') and r=='sourced_from' for h,r,t in triples):
                raise ValueError('index or structural bound retains source edge')
            for symbol in ('R','R_半月k','P_i,j,k'):
                if (symbol,'sourced_from','降雨量资料') not in triples:
                    raise ValueError('observed quantity source edge lost: '+symbol)
        negatives=old.relation_keys(prediction)-old.relation_keys(revised_gold[uid])
        summary['remaining_reviewed_false_relations'].append({'unit_id':uid,'unannotated_candidates':sorted(negatives)})
    summary['reviewed_formula_roles_and_bound_index_check']='passed'
    summary['artifact_hashes']={i['path']:i['sha256'] for i in manifest['files']}
    write_json(out/'comparison.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('base','reviewed','out'): parser.add_argument('--'+flag,required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.base,args.reviewed,args.out)
    print(json.dumps(result['matrix'],ensure_ascii=False,indent=2))
