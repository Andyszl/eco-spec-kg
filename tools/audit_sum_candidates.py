"""Compare v2.4 and loaded v2.5 candidates on unchanged training inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

from ecospec_kg import analysis_v2
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, CANDIDATE_GENERATOR_VERSION
from ecospec_kg.experiment_io_v2 import sha256_path

BASELINE = '9bc751ff7281faf41e75193975159d429db76e71'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_baseline(repo):
    modules=[]
    for filename in ('formula_symbols','extractor_v2'):
        name='ecospec_kg._sum_audit_old_'+filename
        source=subprocess.check_output(['git','-C',str(repo),'show',f'{BASELINE}:src/ecospec_kg/{filename}.py'],encoding='utf-8')
        module=types.ModuleType(name)
        module.__file__=str(repo/'src/ecospec_kg'/f'{filename}.py')
        module.__package__='ecospec_kg'
        sys.modules[name]=module
        exec(compile(source, module.__file__, 'exec'),module.__dict__)
        modules.append(module)
    # The old extractor's imports otherwise resolve the currently loaded lexer.
    modules[1].formula_symbols=modules[0].formula_symbols
    modules[1].normalize_symbol=modules[0].normalize_symbol
    return modules[1].RuleCandidateExtractorV2


def relation_keys(prediction):
    return {analysis_v2._relation_key(r) for r in prediction.get('relations',[])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--reviewed',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('output exists; refusing to overwrite')
    if CANDIDATE_GENERATOR_VERSION!='structure-aware-rule-v2.5':
        raise ValueError('expected candidate generator v2.5')
    manifest=json.loads((args.reviewed/'manifest.json').read_text(encoding='utf-8'))
    for key,path in [('units','blind/train_units.jsonl'),('annotations','gold/train_annotations.jsonl')]:
        if sha256_path(args.base/path)!=manifest['parent_input_hashes'][key]:
            raise ValueError('parent hash mismatch')
    for item in manifest['files']:
        if sha256_path(args.reviewed/item['path'])!=item['sha256']:
            raise ValueError('reviewed file hash mismatch: '+item['path'])
    repo=Path(__file__).resolve().parents[1]
    baseline=load_baseline(repo)
    summary={'baseline_commit':BASELINE,'current_commit':subprocess.check_output(
        ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'loaded_source_hashes':{name:sha256_path(repo/'src/ecospec_kg'/name)
                               for name in ('formula_symbols.py','extractor_v2.py')},
        'tracked_src_dirty':bool(subprocess.check_output(
            ['git','-C',str(repo),'diff','--name-only','HEAD','--','src/ecospec_kg/formula_symbols.py','src/ecospec_kg/extractor_v2.py'],encoding='utf-8')),
        'candidate_generator':CANDIDATE_GENERATOR_VERSION,'conditions':{}}
    for label,package in [('original',args.base),('reviewed',args.reviewed)]:
        units=package/'blind/train_units.jsonl'; gold=package/'gold/train_annotations.jsonl'
        hashes=[sha256_path(units),sha256_path(gold)]
        reports={}
        for version,extractor in [('v2.4',baseline),('v2.5',RuleCandidateExtractorV2)]:
            with patch.object(analysis_v2,'RuleCandidateExtractorV2',extractor), \
                 patch.object(analysis_v2,'CANDIDATE_GENERATOR_VERSION','structure-aware-rule-'+version), \
                 patch.object(analysis_v2,'_write_workbook',return_value=None):
                report=analysis_v2.analyze_training_v2(units,gold,args.out/label/version)
            fields=('covered_entity_count','gold_entity_count','covered_relation_count','gold_relation_count',
                    'entity_recall_upper_bound','relation_recall_upper_bound','entity_candidate_negative_count','relation_candidate_negative_count')
            reports[version]={key:report[key] for key in fields}
        annotations={r['unit_id']:r for r in rows(gold)}
        differences=[]
        old,new=baseline(),RuleCandidateExtractorV2()
        for unit in rows(units):
            a,b=relation_keys(old.predict_unit(unit)),relation_keys(new.predict_unit(unit))
            if a==b:
                continue
            reference=relation_keys(annotations[unit['unit_id']])
            differences.append({'unit_id':unit['unit_id'],'added':sorted(b-a),'removed':sorted(a-b),
                                'gained_coverage':sorted((b-a)&reference),'lost_coverage':sorted((a-b)&reference)})
        if hashes!=[sha256_path(units),sha256_path(gold)]:
            raise ValueError('input changed during audit')
        summary['conditions'][label]={'input_hashes':hashes,'coverage':reports,'changed_units':differences}
    (args.out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v['coverage'] for k,v in summary['conditions'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
