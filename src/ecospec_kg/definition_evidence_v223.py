"""Append original definition spans to already selected explicit output indicators.

An independent postprocessor, not a model selector or a change to frozen gold.
No names/IDs, entity types, edges, or candidate prompts are added or rewritten.
"""
from copy import deepcopy
from .extractor_v2 import _clean,_split_variable,_formula_output_bindings
from .structural_selection_v2 import _index,structural_relation_proofs

VERSION='explicit-output-definition-evidence-v223'


def attach_definition_evidence(unit,candidates,prediction):
    entities,relations,spans=_index(unit,candidates)
    if prediction.get('unit_id')!=unit['unit_id'] or prediction.get('status')!='success':
        raise ValueError('Incompatible prediction')
    selected={r['relation_id']:r for r in prediction['relations']}
    pe={e['entity_id']:e for e in prediction['entities']}
    if len(selected)!=len(prediction['relations']) or len(pe)!=len(prediction['entities']):
        raise ValueError('Duplicate prediction member')
    if any(r!=relations.get(i) for i,r in selected.items()):
        raise ValueError('Changed relation candidate')
    for r in selected.values():
        if any(r[s+'_id'] not in pe for s in ('head','tail')):
            raise ValueError('Missing selected endpoint')
    proven={p['relation_id'] for p in structural_relation_proofs(unit,candidates)
            if p['rule']=='explicit_output_definition'}
    allowed={};bindings=[]
    lookup={(r['head_id'],r['relation_type'],r['tail_id']):r['relation_id'] for r in selected.values()}
    for b in _formula_output_bindings(unit,candidates):
        root=b['indicator_relation_id'];indicator=b['indicator_id'];variable=b['variable_id']
        output=lookup.get((b['formula_id'],'has_output',variable))
        if root not in proven or root not in selected or b['variable_relation_id'] not in selected or not output:
            continue
        for raw in unit.get('variable_definitions',[]):
            if raw.get('definition_origin')=='equation_structure_transcription' or raw['definition']!=b['definition']:
                continue
            if not any(v.get('symbol')==entities[variable]['name'] for v in _split_variable(raw)):
                continue
            span=raw.get('evidence_span',{});sid=span.get('span_id')
            definition=_clean(raw['definition']).rstrip('；;，,。.')
            text=_clean(span.get('text',''))
            if sid not in spans or not definition or definition not in text or _clean(entities[indicator]['name']) not in text:
                continue
            allowed.setdefault(indicator,set()).add(sid)
            bindings.append({'indicator_relation_id':root,'indicator_id':indicator,'variable_id':variable,
                             'definition_span_id':sid,'policy':VERSION})
    # Permit a previous application, but never accept arbitrary span additions.
    for i,e in pe.items():
        source=entities.get(i)
        if source is None or {k:v for k,v in e.items() if k!='evidence_span_ids'}!={k:v for k,v in source.items() if k!='evidence_span_ids'}:
            raise ValueError('Changed entity candidate')
        original=set(source['evidence_span_ids']);actual=set(e.get('evidence_span_ids',[]))
        if not original<=actual or not actual<=original|allowed.get(i,set()):
            raise ValueError('Unsupported prediction evidence')
    result=deepcopy(prediction);changes=[]
    for e in result['entities']:
        added=sorted(allowed.get(e['entity_id'],set())-set(e['evidence_span_ids']))
        if added:
            e['evidence_span_ids'].extend(added)
            changes.append({'entity_id':e['entity_id'],'added_span_ids':added})
    result['evidence_postprocessor_version']=VERSION
    return result,{'policy':VERSION,'changes':changes,'approved_definition_bindings':bindings}
