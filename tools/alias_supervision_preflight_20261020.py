"""Apply user-confirmed alias quarantine; preserve v226 questions and answers."""
import argparse
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from collections import defaultdict
from ecospec_kg import alias_partial_label_v227 as policy
from ecospec_kg import candidate_profile_v214 as profile
import partial_label_preflight_20261020 as old
from cross_source_pair_20261016 import read, rows, sha, lfsha, save, dumps, uid
from partial_label_preflight_20261020 import check_loaded_row, synthetic_loss_check

ROOT=Path(__file__).resolve().parents[1]
FOLDER='experiments/alias_partial_label_v227_20261020'
OLD_SHA='84792aa90b938b335115a998e134349a185cd9aa809bd9a8439589e5582aafca'
REPORT_SHA='2a9b588ceb2a2e7ae3d6e5ebd03e783d6b138d766aaa022c762044a4434db891'
SOURCES=['src/ecospec_kg/alias_partial_label_v227.py','tools/alias_supervision_preflight_20261020.py',
         'tools/swift_alias_partial_label_plugin_20261020.py','tools/run_alias_supervision_preflight_20261020.sh']


def audit_previous(root):
    path=root/FOLDER/'previous_preflight_report.json'
    assert sha(path)==REPORT_SHA
    r=read(path);m=old.verify(root)
    assert sha(root/old.FOLDER/'input_manifest.json')==OLD_SHA
    assert r['input_manifest_sha256']==OLD_SHA and r['input_manifest']==m
    assert r['runtime_source_lock']==old.verify_runtime_lock(root)
    assert r['code_commit']=='4246b1cb8225fb69102524415d5915059b823110'
    assert r['passed'] and not r['failures'] and r['real_tokenizer'] and r['swift_version']=='4.4.2'
    assert not r['model_weights_loaded'] and not r['model_training_started'] and not r['training_ready'] and r['model_calls']==0
    assert r['inference_checks']==len(r['encodings'])==1324 and r['original_inputs_unchanged']
    assert [(v['file'],v['unit_id']) for v in r['encodings']]==[(n,i) for n,ids in m['unit_ids'].items() for i in ids]
    assert all(0<v['supervised_tokens']<=v['tokens']<=8192 for v in r['encodings'])
    assert {(v['file'],v['unit_id']):v['unknown_bits_masked'] for v in r['encodings'] if v['unknown_bits_masked']}=={
        (v['file'],v['unit_id']):v['unknown_zero_gradient'] for v in r['loss_checks']}
    assert sum(v['unknown_zero_gradient'] for v in r['loss_checks'])==37
    assert all(v['known_nonzero_gradient']>0 and abs(v['synthetic_loss']-1.0986122886681098)<1e-5 for v in r['loss_checks'])
    assert r['max_tokens_by_file']=={n:max(v['tokens'] for v in r['encodings'] if v['file']==n) for n in m['counts']}
    return r


def verify_runtime_lock(root):
    return old.verify_runtime_lock(root)


def confirmed_negatives(root, ident, graph):
    result={c['candidate_relation_id'] for c in read(root/'experiments/method_scope_review_20261016/source_review.json')['cases']
            if c['unit_id']==ident and c['review_status']=='exclude_generic_or_bad_boundary'}
    for case in read(root/'experiments/cross_source_supervision_20261016/supervision_review.json')['cases']:
        if case['unit_id']==ident:
            triples=case.get('confirmed_negative_relations',[])
            result.update(r['relation_id'] for r in graph['relations']
                          if [r['head_name'],r['relation_type'],r['tail_name']] in triples)
    return result


def compose(root):
    prior=audit_previous(root);m=prior['input_manifest']
    policy_review=read(root/FOLDER/'policy_review.json')
    assert policy_review['status']=='user_confirmed_conservative_alias_quarantine'
    units={u['unit_id']:u for u in rows(root/'experiments/semantic_pair_v211_20261015/units.jsonl')+
           rows(root/'experiments/cross_source_supervision_20261016/source_units.jsonl')}
    original={n:rows(root/old.FOLDER/(n+'.jsonl')) for n in m['counts']}
    by_unit=defaultdict(list)
    for values in original.values():
        for row in values:by_unit[uid(row)].append(row)
    graphs={};reviews={};entity_masks={};relation_masks={};cases=[]
    for ident,records in by_unit.items():
        u=units[ident];g=profile.candidates_for_unit(u);graphs[ident]=g
        targets=[policy.selection_for(r['messages'][1]['content'],r['messages'][2]['content']) for r in records]
        reviewed=policy.review_aliases(u,g,targets,confirmed_negatives(root,ident,g));reviews[ident]=reviewed
        entity_masks[ident]=reviewed['unknown_entity_ids']
        relation_masks[ident]=sorted(set(m['unscored_relation_ids'][ident])|set(reviewed['unknown_relation_ids']))
        if reviewed['groups']:
            cases.append({'unit_id':ident,'split':'validation' if ident in m['unit_ids']['validation'] else 'train',
                          'source_sha256':u['provenance']['source_sha256'],
                          'standard_code':u['provenance'].get('standard_code'), 'review':reviewed})
    prepared={};coverage=deepcopy(m['supervision_coverage']);strata={}
    for name,records in original.items():
        values=[];stratum=defaultdict(lambda:{'positive_edges':0,'units':set(),'sources':set()})
        unknown_e=0;unknown_r=0
        for row in records:
            ident=uid(row);q,a=row['messages'][1]['content'],row['messages'][2]['content']
            target=policy.selection_for(q,a);u=units[ident]
            assert not set(entity_masks[ident])&set(target['selected_entity_ids'])
            assert not set(relation_masks[ident])&set(target['selected_relation_ids'])
            revised=deepcopy(row);revised['channel']=policy.supervision_channel(q,a,relation_masks[ident],entity_masks[ident])
            values.append(revised);unknown_e+=len(entity_masks[ident]);unknown_r+=len(relation_masks[ident])
            for rel in graphs[ident]['relations']:
                if rel['relation_id'] in target['selected_relation_ids'] and rel['relation_type'] in ('obtained_by','sourced_from'):
                    key=rel['head_type']+'->'+rel['relation_type'];s=stratum[key]
                    s['positive_edges']+=1;s['units'].add(ident);s['sources'].add(u['provenance']['source_sha256'])
                if rel['relation_id'] in set(relation_masks[ident])-set(m['unscored_relation_ids'][ident]):
                    typ=coverage[name]['by_relation_type'][rel['relation_type']]
                    typ['retained_unselected_relations']-=1;typ['unknown_relations_masked']+=1
        coverage[name]['totals'].update(unknown_entities_masked=unknown_e,unknown_relations_masked=unknown_r)
        coverage[name]['totals']['retained_unselected_relations']-=unknown_r-m['supervision_coverage'][name]['totals']['unknown_relations_masked']
        prepared[name]=values
        strata[name]={k:{'positive_edges':v['positive_edges'],'positive_units':len(v['units']),
                         'source_count':len(v['sources']),'unit_ids':sorted(v['units']),'source_sha256':sorted(v['sources'])}
                      for k,v in stratum.items()}
    assert all(a['messages']==b['messages'] for n in prepared for a,b in zip(prepared[n],original[n]))
    for a,b in zip(prepared['control_train'],prepared['repair_train']):
        ca,cb=json.loads(a['channel']),json.loads(b['channel'])
        assert a['messages'][:2]==b['messages'][:2]
        assert ca['unknown_entity_ids']==cb['unknown_entity_ids'] and ca['unknown_relation_ids']==cb['unknown_relation_ids']
    report={'policy':policy_review,'cases':cases,'semantic_positive_coverage':strata,
            'new_positive_relations':0,'new_supervision_sources':0,'training_ready':False,
            'limitations':['Positive edge counts are not independent cases; shared-source internal validation is not final validation.',
                           'Alias quarantine removes disputed negative supervision but does not expand positive source coverage.']}
    manifest=deepcopy(m)
    manifest.pop('files')
    manifest.update(version=policy.VERSION,question_format_version=old.policy.VERSION,template_type=policy.TEMPLATE,
                    previous_input_manifest_sha256=OLD_SHA,previous_preflight_sha256=REPORT_SHA,
                    unscored_entity_ids=entity_masks,unscored_relation_ids=relation_masks,
                    unscored_counts={s:sum(len(relation_masks[i]) for i in m['unit_ids'][n])
                                     for s,n in [('train','repair_train'),('validation','validation')]},
                    unknown_entity_counts={s:sum(len(entity_masks[i]) for i in m['unit_ids'][n])
                                           for s,n in [('train','repair_train'),('validation','validation')]},
                    source_sha256_lf={n:lfsha(root/n) for n in SOURCES},
                    review_sha256_lf={n:lfsha(root/FOLDER/n) for n in ('policy_review.json','previous_preflight_report.json')},
                    supervision_coverage=coverage,questions_and_answers_unchanged_from_v226=True,
                    new_positive_relations=0,requires_real_tokenizer_preflight=True)
    manifest['limitations']=['New alias entity/relation mask runtime preflight pending.',*report['limitations'],*m['limitations'][1:]]
    return prepared,manifest,report


def build(root):
    folder=root/FOLDER;assert not (folder/'input_manifest.json').exists()
    prepared,m,review=compose(root)
    for n,values in prepared.items():(folder/(n+'.jsonl')).write_text(dumps(values),encoding='utf8',newline='\n')
    save(folder/'alias_review.json',review)
    m['files']={n+'.jsonl':sha(folder/(n+'.jsonl')) for n in prepared}
    m['files']['alias_review.json']=sha(folder/'alias_review.json')
    save(folder/'input_manifest.json',m);return m


def verify(root):
    folder=root/FOLDER;m=read(folder/'input_manifest.json')
    for n,h in m['files'].items():assert Path(n).name==n and sha(folder/n)==h
    prepared,expected,review=compose(root)
    assert {k:v for k,v in m.items() if k!='files'}==expected
    assert read(folder/'alias_review.json')==review
    assert all(rows(folder/(n+'.jsonl'))==v for n,v in prepared.items())
    return m


def preflight(root, model, out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only; hide GPUs'
    assert not out.exists(), 'Preserve existing report'
    out.mkdir(parents=True)
    m = verify(root); lock = verify_runtime_lock(root)
    assert importlib.metadata.version('ms-swift') == '4.4.2'
    sdkroot = Path(next(iter(importlib.util.find_spec('swift').submodule_search_locations)))
    for name, info in lock['files'].items():
        assert sha(sdkroot.parent / name) == info['sha256'], 'SDK source changed: ' + name
    for name, h in m['tokenizer_config_sha256'].items():
        assert Path(name).name == name and sha(model / name) == h, 'Tokenizer changed'
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.dataset import load_dataset
    import swift_alias_partial_label_plugin_20261020
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None
    kwargs = dict(max_length=8192, truncation_strategy='raise', add_non_thinking_prefix=True,
                  loss_scale='default+ignore_empty_think')
    template = get_template(processor, template_type=policy.TEMPLATE, **kwargs)
    plain = get_template(processor, template_type='qwen3_5', **kwargs)
    template.set_mode('train'); plain.set_mode('train')
    report = {'passed': False, 'encodings': [], 'failures': [], 'loss_checks': [], 'inference_checks': 0,
              'real_tokenizer': True, 'swift_version': '4.4.2', 'input_manifest_sha256': sha(root / FOLDER / 'input_manifest.json'),
              'input_manifest': m, 'runtime_source_lock': lock, 'model_weights_loaded': False,
              'model_training_started': False, 'training_ready': False, 'model_calls': 0,
              'code_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()}
    try:
        for name in m['counts']:
            saved = rows(root / FOLDER / (name + '.jsonl'))
            loaded, validation = load_dataset([str(root / FOLDER / (name + '.jsonl'))], split_dataset_ratio=0,
                                               seed=42, shuffle=False, strict=True)
            assert validation is None and len(loaded) == len(saved)
            for source, row in zip(saved, loaded):
                identity = {'file': name, 'unit_id': uid(source)}
                try:
                    check_loaded_row(source, row)
                    prompt, answer = row['messages'][1]['content'], row['messages'][2]['content']
                    baseline = plain.encode(deepcopy(row))
                    expected, proof = policy.mask_encoded(baseline, template.tokenizer, prompt, answer, row['channel'])
                    encoded = template.encode(deepcopy(row))
                    assert encoded['input_ids'] == baseline['input_ids'], 'Mask changed tokenization'
                    assert encoded['labels'] == expected['labels'], 'Plugin lost mask'
                    assert not encoded.get('loss_scale'), 'Expected binary loss labels, not weighted trainer branch'
                    assert 'channel' not in encoded
                    assert len(encoded['input_ids']) <= 8192 and any(x != -100 for x in encoded['labels'])
                    batch = template.data_collator([deepcopy(encoded)])
                    assert batch['labels'][0].tolist()[:len(encoded['labels'])] == encoded['labels']
                    assert 'channel' not in batch
                    if proof['unknown_count']:
                        report['loss_checks'].append({**identity, **synthetic_loss_check(template, encoded, proof)})
                        # Flip only unknown placeholders, keeping question and mask identity.
                        changed = json.loads(answer); ids = policy.ordered_ids(prompt)[1]
                        bits = changed['relation_decisions'].split()
                        unknown_ids = json.loads(row['channel'])['unknown_relation_ids']
                        for j, ident in enumerate(ids):
                            if ident in unknown_ids:
                                bits[j] = '1' if bits[j] == '0' else '0'
                        changed['relation_decisions'] = ' '.join(bits)
                        entity_ids = policy.ordered_ids(prompt)[0]
                        entity_bits = changed['entity_decisions'].split()
                        for j, ident in enumerate(entity_ids):
                            if ident in json.loads(row['channel'])['unknown_entity_ids']:
                                entity_bits[j] = '1' if entity_bits[j] == '0' else '0'
                        changed['entity_decisions'] = ' '.join(entity_bits)
                        other_answer = json.dumps(changed, separators=(',', ':'))
                        other_row = deepcopy(row); other_row['messages'][2]['content'] = other_answer
                        other_row['channel'] = policy.supervision_channel(prompt, other_answer, unknown_ids, json.loads(row['channel'])['unknown_entity_ids'])
                        other = template.encode(other_row)
                        _, other_proof = policy.mask_encoded(plain.encode(deepcopy(other_row)), template.tokenizer, prompt, other_answer, other_row['channel'])
                        assert proof == other_proof and encoded['labels'] == other['labels'], 'Unknown placeholder affects supervised labels'
                    # Identical system/user and tokenizer prefix during formal inference.
                    infer = {'messages': deepcopy(row['messages'][:2])}
                    template.set_mode('transformers'); plain.set_mode('transformers')
                    try:
                        a, b = template.encode(deepcopy(infer)), plain.encode(deepcopy(infer))
                        assert a['input_ids'] == b['input_ids'] and not a.get('labels') and 'channel' not in a
                        generation = a['input_ids']; training = encoded['input_ids']
                        assert training[:len(generation)] == generation, 'Training/inference prompt prefix differs'
                        report['inference_checks'] += 1
                    finally:
                        template.set_mode('train'); plain.set_mode('train')
                    report['encodings'].append({**identity, 'tokens': len(encoded['input_ids']),
                                                'supervised_tokens': sum(x != -100 for x in encoded['labels']),
                                                'unknown_bits_masked': proof['unknown_count'],
                                                'unknown_entity_bits_masked': len(proof['unknown_entity_token_indices']),
                                                'unknown_relation_bits_masked': len(proof['unknown_relation_token_indices']),
                                                'known_bits_supervised': len(proof['known_token_indices'])})
                except Exception as exc:
                    report['failures'].append({**identity, 'error_type': type(exc).__name__, 'error': str(exc)})
                total = len(report['encodings']) + len(report['failures'])
                if total % 100 == 0:
                    print('CHECKED_OR_RECORDED_FAILURE=' + str(total), flush=True)
        assert len(report['encodings']) + len(report['failures']) == 1324
        assert verify(root) == m
        report['original_inputs_unchanged'] = True
        report['max_tokens_by_file'] = {n: max((r['tokens'] for r in report['encodings'] if r['file'] == n), default=0) for n in m['counts']}
        passed = not report['failures'] and len(report['encodings']) == report['inference_checks'] == 1324
        if passed:
            for name in m['counts']:
                observed = sum(r['unknown_bits_masked'] for r in report['encodings'] if r['file'] == name)
                assert observed == sum(m['supervision_coverage'][name]['totals'][k] for k in ('unknown_entities_masked','unknown_relations_masked'))
            assert sum(r['unknown_zero_gradient'] for r in report['loss_checks']) == sum(v['totals']['unknown_entities_masked'] + v['totals']['unknown_relations_masked'] for v in m['supervision_coverage'].values())
        report['passed'] = passed
    finally:
        save(out / 'preflight_report.json', report)
    print('ALIAS_SUPERVISION_PREFLIGHT_OK=' + str(report['passed']).lower())
    print('TRAINING_READY=false')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'verify', 'preflight'))
    p.add_argument('--model', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'build':
        print(json.dumps(build(ROOT)['counts']))
    elif a.action == 'verify':
        print(json.dumps(verify(ROOT)['counts']))
    else:
        assert a.model and a.out
        raise SystemExit(0 if preflight(ROOT, a.model, a.out)['passed'] else 1)
