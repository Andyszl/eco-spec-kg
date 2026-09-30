"""Replay an explicit source review into a train-only candidate; never use predictions as labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8',newline='\n')


def spans(value):
    found=set()
    if isinstance(value,dict):
        if value.get('span_id'): found.add(value['span_id'])
        for v in value.values(): found.update(spans(v))
    elif isinstance(value,list):
        for v in value: found.update(spans(v))
    return found


def relation_key(r):
    return tuple(r[k] for k in ('head_name','head_type','relation_type','tail_name','tail_type'))


def validate_records(units, annotations):
    um={u['unit_id']:u for u in units}; am={a['unit_id']:a for a in annotations}
    if len(um)!=len(units) or len(am)!=len(annotations) or set(um)!=set(am):
        raise ValueError('unit identity mismatch or duplicate')
    for uid,a in am.items():
        if a.get('split')!='train': raise ValueError('only train annotations allowed')
        available=spans(um[uid]); entities={}
        for e in a['entities']:
            identity=(e['name'],e['entity_type'])
            if e['entity_id'] in entities and entities[e['entity_id']]!=identity:
                raise ValueError('conflicting entity ID')
            entities[e['entity_id']]=identity
            if not set(e.get('evidence_span_ids',[])) <= available:
                raise ValueError(f'entity evidence missing: {uid}')
        for r in a['relations']:
            for side in ('head','tail'):
                if entities.get(r[side+'_id']) != (r[side+'_name'],r[side+'_type']):
                    raise ValueError(f'dangling relation endpoint: {uid}')
            if not r.get('evidence_span_ids') or not set(r['evidence_span_ids']) <= available:
                raise ValueError(f'relation evidence missing: {uid}')
    return {'unit_count':len(units),'split':'train','endpoint_and_evidence_checks':'passed'}


def apply_review(units, annotations, decisions):
    validate_records(units,annotations)
    new_units=copy.deepcopy(units); new_annotations=copy.deepcopy(annotations)
    um={u['unit_id']:u for u in new_units}; am={a['unit_id']:a for a in new_annotations}
    audit=[]
    for decision in decisions['units']:
        if decision['unit_id'] not in um: raise ValueError('review unit missing')
        uid=decision['unit_id']; unit=um[uid]; annotation=am[uid]
        before_u=copy.deepcopy(unit); before_a=copy.deepcopy(annotation)
        formulas={f['formula_number']:f for f in unit['formulas']}
        available={}
        def collect(value):
            if isinstance(value,dict):
                if value.get('span_id'): available[value['span_id']]=value
                for v in value.values(): collect(v)
            elif isinstance(value,list):
                for v in value: collect(v)
        collect(unit)
        for addition in decision.get('definitions',[]):
            sid=addition['evidence_span_id']
            if sid not in available: raise ValueError('definition evidence missing')
            existing=next((v for v in unit['variable_definitions'] if v['symbol']==addition['symbol']),None)
            if existing is None:
                existing={'symbol':addition['symbol'],'unit':''}
                unit['variable_definitions'].append(existing)
            existing.update({k:v for k,v in addition.items() if k!='evidence_span_id'})
            existing['evidence_span']=copy.deepcopy(available[sid])
        drops={tuple(r) for r in decision.get('drop_relations',[])}
        actual={(r['head_name'],r['relation_type'],r['tail_name']) for r in annotation['relations']}
        if not drops <= actual: raise ValueError('reviewed removal absent from parent')
        annotation['relations']=[r for r in annotation['relations'] if
            (r['head_name'],r['relation_type'],r['tail_name']) not in drops]

        def entity(name,kind,evidence):
            for e in annotation['entities']:
                if (e['name'],e['entity_type'])==(name,kind):
                    e['evidence_span_ids']=list(evidence)
                    return e
            e={'entity_id':hashlib.sha256(f'{uid}|{kind}|{name}'.encode()).hexdigest()[:16],
               'name':name,'entity_type':kind,'evidence_span_ids':list(evidence)}
            annotation['entities'].append(e); return e

        def edge(head,relation,tail,evidence):
            return {'head_id':head['entity_id'],'head_name':head['name'],'head_type':head['entity_type'],
                    'relation_type':relation,'tail_id':tail['entity_id'],'tail_name':tail['name'],
                    'tail_type':tail['entity_type'],'evidence_span_ids':list(dict.fromkeys(evidence))}

        for review in decision.get('formulas',[]):
            inputs=set(review['inputs']); outputs=set(review['outputs']); indices=set(review.get('indices',[]))
            if inputs & outputs or inputs & indices or outputs & indices:
                raise ValueError('overlapping formula roles')
            definitions={v['symbol']:v for v in unit['variable_definitions']}
            if not inputs|outputs|indices <= set(definitions):
                raise ValueError(f'role refers to undefined source symbol: {uid}')
            formula=formulas[review['number']]
            formula.setdefault('raw_expression_text',formula['expression_text'])
            formula.setdefault('raw_expression_lines',formula.get('expression_lines',[]))
            if 'symbol_aliases' in review: formula['symbol_aliases']=copy.deepcopy(review['symbol_aliases'])
            formula['expression_text']=review['expression']; formula['expression_lines']=review['expression'].splitlines()
            name=f"公式（{review['number']}）"; fs=formula['evidence_span']['span_id']
            fe=entity(name,'formula',[fs])
            annotation['relations']=[r for r in annotation['relations'] if not (
                (r['head_name']==name and r['head_type']=='formula' and r['relation_type'] in ('has_input','has_output'))
                or (r['tail_name']==name and r['tail_type']=='formula' and r['head_type']=='model_variable' and r['relation_type']=='calculated_by'))]
            for symbol in review['inputs']+review['outputs']:
                vs=definitions[symbol]['evidence_span']['span_id']; ve=entity(symbol,'model_variable',[vs])
                relation='has_output' if symbol in outputs else 'has_input'
                annotation['relations'].append(edge(fe,relation,ve,[fs,vs]))
                if symbol in outputs: annotation['relations'].append(edge(ve,'calculated_by',fe,[fs,vs]))
                unit_name=definitions[symbol].get('unit','')
                if unit_name and not any(r['head_id']==ve['entity_id'] and r['relation_type']=='has_unit' and r['tail_name']==unit_name for r in annotation['relations']):
                    annotation['relations'].append(edge(ve,'has_unit',entity(unit_name,'unit',[vs]),[vs]))
            # Roles stay in the external review ledger, never in blind source fields.
        annotation['review_status']='assistant_source_review_candidate_requires_human_expert_review'
        annotation['notes']=(annotation.get('notes','')+'\n20260930: deferred semantics and index sources reviewed; partial review, not expert gold.').strip()
        before_keys={relation_key(r) for r in before_a['relations']}; after_keys={relation_key(r) for r in annotation['relations']}
        audit.append({'unit_id':uid,'standard':unit['provenance'].get('standard_code'),
                      'source_sha256':unit['provenance'].get('source_sha256'),
                      'pages':unit['provenance'].get('pages'), 'reason':decision['reason'],
                      'source_before':before_u,'source_after':unit,
                      'annotation_before':before_a,'annotation_after':annotation,
                      'removed_relation_keys':sorted(before_keys-after_keys),
                      'added_relation_keys':sorted(after_keys-before_keys),
                      'formula_reviews':decision.get('formulas',[])})
    validate_records(new_units,new_annotations)
    return new_units,new_annotations,audit


def build(base, destination, decisions_path):
    if destination.exists(): raise ValueError('output exists; refusing to overwrite')
    spec=json.loads(decisions_path.read_text(encoding='utf-8'))
    source=base/'blind/train_units.jsonl'; labels=base/'gold/train_annotations.jsonl'
    if digest(source)!=spec['units_sha256'] or digest(labels)!=spec['annotations_sha256']:
        raise ValueError('parent SHA256 mismatch')
    units=read_rows(source); annotations=read_rows(labels)
    revised, gold, audit=apply_review(units,annotations,spec)
    from ecospec_kg.experiment_io_v2 import assert_blind_records
    assert_blind_records(revised)
    destination.mkdir(parents=True)
    write_rows(destination/'blind/train_units.jsonl',revised)
    write_rows(destination/'gold/train_annotations.jsonl',gold)
    write_rows(destination/'review/revision_ledger.jsonl',audit)
    write_json(destination/'review/decisions.json',spec)
    manifest={'dataset_version':'v2.2-train-source-review-20260930-rc2',
              'status':'assistant_review_candidate_not_frozen', 'parent_dataset':'v2.2-train-source-review-20260929-rc1',
              'parent_input_hashes':{'units':digest(source),'annotations':digest(labels)},
              'scope':'train only; dev/test not copied or modified',
              'human_expert_review_completed':False, 'gold_nature':'partial_assistant_source_review_of_existing_AI_labels',
              'unit_count':len(revised),'reviewed_units':len(audit),'source_changed_units':sum(a['source_before']!=a['source_after'] for a in audit), 'label_changed_units':sum(a['annotation_before']!=a['annotation_after'] for a in audit),
              'modified_formula_count':sum(len(a['formula_reviews']) for a in audit),
              'deferred_units':[],
              'ids_order_preserved': [u['unit_id'] for u in units]==[u['unit_id'] for u in revised],
              'validation':validate_records(revised,gold),
              'relation_keys_added':sum(len(a['added_relation_keys']) for a in audit),
              'relation_keys_removed':sum(len(a['removed_relation_keys']) for a in audit),
              'files':[{'path':str(p.relative_to(destination)).replace('\\','/'),'sha256':digest(p)}
                       for p in sorted(destination.rglob('*')) if p.is_file()]}
    if digest(source)!=spec['units_sha256'] or digest(labels)!=spec['annotations_sha256']:
        raise ValueError('parent files changed during replay')
    write_json(destination/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--decisions',type=Path,default=Path(__file__).resolve().parents[1]/'deliveries/semantic_review_20260930/review_decisions.json')
    args=parser.parse_args()
    print(json.dumps(build(args.base,args.out,args.decisions),ensure_ascii=False,indent=2))
