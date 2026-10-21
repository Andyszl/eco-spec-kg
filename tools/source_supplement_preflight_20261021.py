"""Source-reviewed additions plus real CPU SDK preflight. No training entry point."""
import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.metadata,importlib.util,json,os,re,subprocess
from pathlib import Path
from ecospec_kg import alias_partial_label_v227 as policy
from ecospec_kg import candidate_profile_v214 as baseline
from ecospec_kg import candidate_profile_v228 as profile
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
import alias_supervision_preflight_20261020 as prior
from partial_label_preflight_20261020 import check_loaded_row,synthetic_loss_check
from cross_source_pair_20261016 import read,rows,sha,lfsha,save,dumps,uid

ROOT=Path(__file__).resolve().parents[1]
PACKET='experiments/source_supplement_v228_20261021'
FOLDER='experiments/source_supplement_inputs_v228_20261021'
PRIOR_MANIFEST_SHA='53cc7456c3e60027dafddaf60141c3394121a3dea3455a33028f8dd979bb8e12'
PRIOR_REPORT_SHA='e73dcaad63d32a54658147b32fa32a15d1098e300e5e6b0ea0ec78c5a25f99c0'
SOURCES=['src/ecospec_kg/candidate_profile_v228.py','tools/source_supplement_preflight_20261021.py',
         'tools/run_source_supplement_preflight_20261021.sh']


def audit_previous(root):
    m=prior.verify(root);r=read(root/PACKET/'previous_preflight_report.json')
    assert sha(root/prior.FOLDER/'input_manifest.json')==PRIOR_MANIFEST_SHA
    assert sha(root/PACKET/'previous_preflight_report.json')==PRIOR_REPORT_SHA
    assert r['input_manifest']==m and r['input_manifest_sha256']==PRIOR_MANIFEST_SHA
    assert r['runtime_source_lock']==prior.verify_runtime_lock(root)
    assert r['code_commit']=='771bd0dff6dcc54945dbeb44800f636e5060161b'
    assert r['passed'] and not r['failures'] and r['swift_version']=='4.4.2' and r['real_tokenizer']
    assert r['original_inputs_unchanged'] and r['inference_checks']==len(r['encodings'])==1324
    assert [(v['file'],v['unit_id']) for v in r['encodings']]==[(n,i) for n,ids in m['unit_ids'].items() for i in ids]
    assert {(v['file'],v['unit_id']):v['unknown_bits_masked'] for v in r['encodings'] if v['unknown_bits_masked']}=={
        (v['file'],v['unit_id']):v['unknown_zero_gradient'] for v in r['loss_checks']}
    assert sum(v['unknown_zero_gradient'] for v in r['loss_checks'])==73
    assert all(v['known_nonzero_gradient']>0 and abs(v['synthetic_loss']-1.0986122886681098)<1e-5 for v in r['loss_checks'])
    assert not any(r[k] for k in ('model_calls','model_weights_loaded','model_training_started','training_ready'))
    return m


def validate_cases(units,review,source_manifest,annotations):
    sources={s['sha256']:s for s in source_manifest['sources']}
    assert len(sources)==3 and not source_manifest['independent_final_test']
    assert set(units)==set(annotations)=={c['unit_id'] for c in review['cases']} and len(units)==5
    source_sets=defaultdict(set);counts=defaultdict(int);focus=defaultdict(int)
    for c in review['cases']:
        u=units[c['unit_id']];g=profile.candidates_for_unit(u);a=annotations[u['unit_id']]
        assert c['source_sha256']==u['provenance']['source_sha256']
        assert c['split']==sources[c['source_sha256']]['role']
        source_sets[c['split']].add(c['source_sha256']);counts[c['split']]+=1
        edges={r['relation_id']:r for r in g['relations']};labels={d['relation_id']:d for d in c['decisions']}
        assert set(edges)==set(labels)
        for i,r in edges.items():assert labels[i]['edge']==[r[k] for k in ('head_name','relation_type','tail_name')]
        positives={r['relation_id'] for r in a['relations']}
        assert positives=={i for i,d in labels.items() if d['status']=='positive'}
        assert set(c['unknown_relation_ids'])=={i for i,d in labels.items() if d['status']=='unknown'}
        assert set(c['confirmed_negative_relation_ids'])=={i for i,d in labels.items() if d['status']=='negative'}
        assert {tuple(e) for e in c['positive']}<={tuple(labels[i]['edge']) for i in positives}
        assert a['entities']==[e for e in g['entities'] if e['entity_id'] not in c['confirmed_negative_entity_ids']]
        assert a['relations']==[r for r in g['relations'] if r['relation_id'] in positives]
        failures=[v for v in _validate_unit_prediction(u,{**g,'entities':a['entities'],'relations':a['relations']})
                  if v['code']!='missing_reproducibility_field']
        assert not failures,failures
        focus[c['split']]+=len(c['positive'])
    assert not source_sets['train'] & source_sets['validation']
    assert dict(counts)=={'train':3,'validation':2} and dict(focus)=={'train':8,'validation':2}
    return dict(counts=dict(counts),focus_positive_counts=dict(focus),source_sets={k:sorted(v) for k,v in source_sets.items()},
                train_source_count=len(source_sets['train']),validation_source_count=len(source_sets['validation']))


def check_packet(root):
    f=root/PACKET;s=read(f/'source_manifest.json');review=read(f/'supervision_review.json')
    assert s['candidate_rule_sha256']==lfsha(root/'src/ecospec_kg/candidate_profile_v228.py')
    for source in s['sources']:
        assert Path(source['file']).name==source['file'] and sha(f/source['file'])==source['sha256']
    units={u['unit_id']:u for u in rows(f/'source_units.jsonl')}
    annotations={u['unit_id']:u for u in rows(f/'annotations.jsonl')}
    return dict(units=units,review=review,source_manifest=s,annotations=annotations,
                **validate_cases(units,review,s,annotations))


def coverage(records,units,graphs):
    strata=defaultdict(lambda:dict(edges=0,units=set(),sources=set()))
    for row in records:
        i=uid(row);target=policy.selection_for(row['messages'][1]['content'],row['messages'][2]['content'])
        for r in graphs[i]['relations']:
            if r['relation_id'] in target['selected_relation_ids'] and r['relation_type'] in ('obtained_by','sourced_from'):
                v=strata[r['head_type']+'->'+r['relation_type']];v['edges']+=1;v['units'].add(i);v['sources'].add(units[i]['provenance']['source_sha256'])
    return {k:dict(positive_edges=v['edges'],positive_units=len(v['units']),source_count=len(v['sources']),
        unit_ids=sorted(v['units']),source_sha256=sorted(v['sources'])) for k,v in strata.items()}


def compose(root):
    previous=audit_previous(root);packet=check_packet(root)
    units={u['unit_id']:u for u in rows(root/'experiments/semantic_pair_v211_20261015/units.jsonl')+
           rows(root/'experiments/cross_source_supervision_20261016/source_units.jsonl')}
    old_sources={u['provenance']['source_sha256'] for u in units.values()}
    assert not old_sources & {s['sha256'] for s in packet['source_manifest']['sources']},'Not a new source PDF'
    units.update(packet['units']);cases={c['unit_id']:c for c in packet['review']['cases']}
    prepared={};graphs={};expansions={};all_masks={};totals={}
    for name in previous['counts']:
        original=rows(root/prior.FOLDER/(name+'.jsonl'));result=[]
        for row in original:
            i=uid(row);u=units[i];g=graphs.setdefault(i,profile.candidates_for_unit(u))
            oldq=json.loads(row['messages'][1]['content']);sys,q=policy.build_messages(u,g);newq=json.loads(q)
            assert oldq['source_unit']==newq['source_unit'] and oldq.get('formula_output_bindings')==newq.get('formula_output_bindings')
            old_e={e['id'] for e in oldq['candidate_entities']};old_r={r['id'] for r in oldq['candidate_relations']}
            es={e['entity_id'] for e in g['entities']};rs={r['relation_id'] for r in g['relations']}
            assert old_e<=es and old_r<=rs,'Frozen candidate removed'
            # Existing labels remain exactly the previous effective graph. Any
            # new proposal on old text is unknown until separately reviewed.
            target=policy.selection_for(row['messages'][1]['content'],row['messages'][2]['content'])
            meta=json.loads(row['channel']);ue=sorted(set(meta['unknown_entity_ids'])|(es-old_e));ur=sorted(set(meta['unknown_relation_ids'])|(rs-old_r))
            a=policy.answer_for(q,target)
            assert policy.selection_for(q,a)=={k:[j for j in policy.ordered_ids(q)[n] if j in target[k]] for n,k in enumerate(('selected_entity_ids','selected_relation_ids'))}
            result.append(dict(messages=[dict(role='system',content=sys),dict(role='user',content=q),dict(role='assistant',content=a)],
                               channel=policy.supervision_channel(q,a,ur,ue)))
            expansions[i]=dict(new_entity_ids=sorted(es-old_e),new_relation_ids=sorted(rs-old_r))
            all_masks[i]=dict(entity=ue,relation=ur)
        split='validation' if name=='validation' else 'train'
        for i,c in cases.items():
            if c['split']!=split:continue
            u=units[i];g=graphs.setdefault(i,profile.candidates_for_unit(u));annotation=packet['annotations'][i]
            target=dict(selected_entity_ids=[e['entity_id'] for e in annotation['entities']],selected_relation_ids=[r['relation_id'] for r in annotation['relations']])
            sys,q=policy.build_messages(u,g);a=policy.answer_for(q,target)
            ur=c['unknown_relation_ids'];ue=c['unknown_entity_ids'];all_masks[i]=dict(entity=ue,relation=ur)
            result.append(dict(messages=[dict(role='system',content=sys),dict(role='user',content=q),dict(role='assistant',content=a)],
                               channel=policy.supervision_channel(q,a,ur,ue)))
        prepared[name]=result
        totals[name]=dict(totals=dict(unknown_entities_masked=sum(len(json.loads(r['channel'])['unknown_entity_ids']) for r in result),
            unknown_relations_masked=sum(len(json.loads(r['channel'])['unknown_relation_ids']) for r in result)))
    assert prepared['control_train'][-3:]==prepared['repair_train'][-3:]
    assert all(a['messages'][:2]==b['messages'][:2] for a,b in zip(prepared['control_train'],prepared['repair_train']))
    differences=sum(a['messages'][2]!=b['messages'][2] for a,b in zip(prepared['control_train'],prepared['repair_train']))
    assert differences==6
    # Provenance for the generic name(alias) pattern is frozen TRAIN prose,
    # not the new validation equation name. This is audit data, not ID logic.
    pattern=re.compile(r'(?:的|采用|使用|运用)((?:(?!的)[\u4e00-\u9fff]){2,24}?(?:方程|模型)[（(][A-Za-z][A-Za-z0-9\-]{1,20}[）)])')
    derivation=[]
    for i in previous['unit_ids']['repair_train']:
        for s in units[i]['provenance']['evidence_spans']:
            for match in pattern.finditer(re.sub(r'\s+','',s['text'])):
                if match.group(1) not in ex.DATA_SOURCE_TERMS:
                    derivation.append(dict(unit_id=i,source_sha256=units[i]['provenance']['source_sha256'],span_id=s['span_id'],name=match.group(1)))
    assert derivation,'No train-source derivation of parenthetical named methods'
    cov={n:coverage(v,units,graphs) for n,v in prepared.items()}
    report=dict(counts={n:len(v) for n,v in prepared.items()},semantic_positive_coverage=cov,
        new_train_units=3,new_validation_units=2,new_train_sources=2,new_validation_sources=1,
        new_shared_train_focus_positive_edges=8,new_validation_focus_positive_edges=2,
        old_graph_proposals_expanded_units=sum(bool(v['new_entity_ids'] or v['new_relation_ids']) for v in expansions.values()),
        expansion_masks=expansions,named_alias_rule_train_evidence=derivation,
        source_partition=packet['source_sets'],same_family_validation_case='RUSLE is a known formula family; no claim of new-algorithm or final unseen generalization.',
        source_negative_gap=packet['review']['source_negative_gap'],candidate_positive_coverage_complete_for_reviewed_focus=True,
        validation_label_status='assistant internal source review; consulted during design; not independent human or final blind gold',
        model_calls=0,training_ready=False)
    assert report['counts']=={'control_train':629,'repair_train':629,'validation':74}
    files=[p for p in (root/PACKET).iterdir() if p.is_file()]
    manifest=dict(version='source-supplement-v228',question_format_version='ecospec-partial-label-v226',template_type=policy.TEMPLATE,
        candidate_profile_version=profile.VERSION,counts=report['counts'],unit_ids={n:[uid(r) for r in v] for n,v in prepared.items()},
        previous_input_manifest_sha256=PRIOR_MANIFEST_SHA,previous_preflight_sha256=PRIOR_REPORT_SHA,
        tokenizer_config_sha256=previous['tokenizer_config_sha256'],source_sha256_lf={n:lfsha(root/n) for n in SOURCES},
        packet_sha256={p.name:sha(p) for p in sorted(files)},paired_changed_training_targets=differences,
        supervision_coverage=totals,semantic_positive_coverage=cov,new_shared_train_examples_identical=True,
        independent_final_test=False,training_ready=False,
        limitations=['Real tokenizer/collator/loss preflight pending for these newly rebuilt inputs.',
                    'Old-text candidate additions are masked unknown; existing source annotations are not relabelled.',
                    'No new verified negative sourced_from example; ambiguous provenance remains unknown.',
                    'New validation source is internally reviewed and has a known RUSLE formula family; final unseen source still pending.',
                    'Unknown placeholders still influence teacher-forcing context; only direct loss is masked.'])
    return prepared,manifest,report


def build(root):
    f=root/FOLDER;f.mkdir(exist_ok=True);assert not (f/'input_manifest.json').exists()
    prepared,m,r=compose(root)
    for n,v in prepared.items():(f/(n+'.jsonl')).write_text(dumps(v),encoding='utf8',newline='\n')
    save(f/'supervision_coverage.json',r)
    m['files']={n+'.jsonl':sha(f/(n+'.jsonl')) for n in prepared};m['files']['supervision_coverage.json']=sha(f/'supervision_coverage.json')
    save(f/'input_manifest.json',m);return m


def verify(root):
    f=root/FOLDER;m=read(f/'input_manifest.json')
    for n,h in m['files'].items():assert Path(n).name==n and sha(f/n)==h
    prepared,expected,report=compose(root)
    assert {k:v for k,v in m.items() if k!='files'}==expected
    assert read(f/'supervision_coverage.json')==report
    assert all(rows(f/(n+'.jsonl'))==v for n,v in prepared.items())
    return m


def preflight(root,model,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only; hide GPUs'
    assert not out.exists(),'Preserve old reports'
    out.mkdir(parents=True)
    m=verify(root);lock=prior.verify_runtime_lock(root)
    assert importlib.metadata.version('ms-swift')=='4.4.2'
    sdk=Path(next(iter(importlib.util.find_spec('swift').submodule_search_locations)))
    for n,v in lock['files'].items():assert sha(sdk.parent/n)==v['sha256'],'SDK changed: '+n
    for n,h in m['tokenizer_config_sha256'].items():assert Path(n).name==n and sha(model/n)==h
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.dataset import load_dataset
    import swift_alias_partial_label_plugin_20261020
    unloaded,processor=get_model_processor(str(model),load_model=False,download_model=False)
    assert unloaded is None
    kwargs=dict(max_length=8192,truncation_strategy='raise',add_non_thinking_prefix=True,loss_scale='default+ignore_empty_think')
    template=get_template(processor,template_type=policy.TEMPLATE,**kwargs)
    plain=get_template(processor,template_type='qwen3_5',**kwargs)
    template.set_mode('train');plain.set_mode('train')
    report=dict(passed=False,encodings=[],failures=[],loss_checks=[],inference_checks=0,
        real_tokenizer=True,swift_version='4.4.2',input_manifest_sha256=sha(root/FOLDER/'input_manifest.json'),
        input_manifest=m,runtime_source_lock=lock,model_weights_loaded=False,model_training_started=False,
        training_ready=False,model_calls=0,code_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())
    try:
        for name in m['counts']:
            saved=rows(root/FOLDER/(name+'.jsonl'))
            loaded,validation=load_dataset([str(root/FOLDER/(name+'.jsonl'))],split_dataset_ratio=0,seed=42,shuffle=False,strict=True)
            assert validation is None and len(loaded)==len(saved)
            for source,row in zip(saved,loaded):
                identity=dict(file=name,unit_id=uid(source))
                try:
                    check_loaded_row(source,row)
                    q,a=row['messages'][1]['content'],row['messages'][2]['content']
                    baseline=plain.encode(deepcopy(row))
                    expected,proof=policy.mask_encoded(baseline,template.tokenizer,q,a,row['channel'])
                    encoded=template.encode(deepcopy(row))
                    assert encoded['input_ids']==baseline['input_ids'] and encoded['labels']==expected['labels']
                    assert not encoded.get('loss_scale') and 'channel' not in encoded
                    assert len(encoded['input_ids'])<=8192 and any(v!=-100 for v in encoded['labels'])
                    batch=template.data_collator([deepcopy(encoded)])
                    assert batch['labels'][0].tolist()[:len(encoded['labels'])]==encoded['labels'] and 'channel' not in batch
                    if proof['unknown_count']:
                        report['loss_checks'].append({**identity,**synthetic_loss_check(template,encoded,proof)})
                        changed=json.loads(a);meta=json.loads(row['channel'])
                        for field,ids,unknown in zip(('entity_decisions','relation_decisions'),policy.ordered_ids(q),
                                (meta['unknown_entity_ids'],meta['unknown_relation_ids'])):
                            bits=changed[field].split()
                            for j,i in enumerate(ids):
                                if i in unknown:bits[j]='1' if bits[j]=='0' else '0'
                            changed[field]=' '.join(bits)
                        a2=json.dumps(changed,separators=(',',':'));other=deepcopy(row);other['messages'][2]['content']=a2
                        other['channel']=policy.supervision_channel(q,a2,meta['unknown_relation_ids'],meta['unknown_entity_ids'])
                        after=template.encode(deepcopy(other))
                        _,proof2=policy.mask_encoded(plain.encode(other),template.tokenizer,q,a2,other['channel'])
                        assert proof==proof2 and encoded['labels']==after['labels']
                    infer=dict(messages=deepcopy(row['messages'][:2]))
                    template.set_mode('transformers');plain.set_mode('transformers')
                    try:
                        x,y=template.encode(deepcopy(infer)),plain.encode(deepcopy(infer))
                        assert x['input_ids']==y['input_ids'] and not x.get('labels') and 'channel' not in x
                        assert encoded['input_ids'][:len(x['input_ids'])]==x['input_ids']
                        report['inference_checks']+=1
                    finally:template.set_mode('train');plain.set_mode('train')
                    report['encodings'].append({**identity,'tokens':len(encoded['input_ids']),
                        'supervised_tokens':sum(v!=-100 for v in encoded['labels']), 'unknown_bits_masked':proof['unknown_count'],
                        'unknown_entity_bits_masked':len(proof['unknown_entity_token_indices']),
                        'unknown_relation_bits_masked':len(proof['unknown_relation_token_indices'])})
                except Exception as exc:
                    report['failures'].append({**identity,'error_type':type(exc).__name__,'error':str(exc)})
                total=len(report['encodings'])+len(report['failures'])
                if total%100==0:print('CHECKED_OR_RECORDED_FAILURE='+str(total),flush=True)
        expected_count=sum(m['counts'].values())
        assert len(report['encodings'])+len(report['failures'])==expected_count
        assert verify(root)==m;report['original_inputs_unchanged']=True
        report['max_tokens_by_file']={n:max((v['tokens'] for v in report['encodings'] if v['file']==n),default=0) for n in m['counts']}
        passed=not report['failures'] and len(report['encodings'])==report['inference_checks']==expected_count
        if passed:
            for n,v in m['supervision_coverage'].items():
                assert sum(x['unknown_bits_masked'] for x in report['encodings'] if x['file']==n)==sum(v['totals'].values())
            assert sum(x['unknown_zero_gradient'] for x in report['loss_checks'])==sum(sum(v['totals'].values()) for v in m['supervision_coverage'].values())
        report['passed']=passed
    finally:save(out/'preflight_report.json',report)
    print('SOURCE_SUPPLEMENT_PREFLIGHT_OK='+str(report['passed']).lower())
    print('TRAINING_READY=false')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('build','verify','preflight'))
    parser.add_argument('--model',type=Path);parser.add_argument('--out',type=Path);a=parser.parse_args()
    if a.action=='build':print(json.dumps(build(ROOT)['counts']))
    elif a.action=='verify':print(json.dumps(verify(ROOT)['counts']))
    else:
        assert a.model and a.out
        raise SystemExit(0 if preflight(ROOT,a.model,a.out)['passed'] else 1)
