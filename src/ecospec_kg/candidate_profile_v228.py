"""Evidence-local candidate expansion. These edges are proposals, never labels.

Method patterns inherit the training-source extractor. Input definitions may
name their own acquisition method; section data fields apply to the quantities
in that section. Review must still separate direct inputs from derived outputs.
No source, entity, variable symbol or formula number is special-cased.
"""
from copy import deepcopy
import re
from . import candidate_profile_v214 as previous
from . import extractor_v2 as ex

VERSION='definition-local-candidates-v228-20261021'


def augment(unit, original):
    if unit.get('unit_type')!='formula_package':
        return deepcopy(original)
    b=ex.PredictionBuilder(unit, original['backend'])
    b.entities={(e['name'],e['entity_type']):deepcopy(e) for e in original['entities']}
    b.relations={tuple(r[k] for k in ('head_name','head_type','relation_type','tail_name','tail_type')):deepcopy(r)
                 for r in original['relations']}
    spans=ex._all_spans(unit)
    text=re.sub(r'\s+','',unit.get('introduction',''))
    outputs={r['tail_id'] for r in original['relations'] if r['relation_type']=='has_output'}
    quantities={r['tail_id'] for r in original['relations'] if r['relation_type'] in ('has_input','has_output')}
    for m in re.finditer(r'(?:的|采用|使用|运用)((?:(?!的)[\u4e00-\u9fff]){2,24}?(?:方程|模型)[（(][A-Za-z][A-Za-z0-9\-]{1,20}[）)])',text):
        name=m.group(1)
        if name in ex.DATA_SOURCE_TERMS:continue
        evidence=[sid for sid,s in spans.items() if name in re.sub(r'\s+','',s.get('text',''))]
        if evidence:
            method=b.add_entity(name,'method',evidence)
            for v in list(b.entities.values()):
                if v['entity_id'] in outputs:
                    b.add_relation(v,'obtained_by',method,list(dict.fromkeys(v['evidence_span_ids']+evidence)))
    for definition in unit.get('variable_definitions',[]):
        sid=definition.get('evidence_span',{}).get('span_id')
        if sid not in spans or definition.get('definition_origin')=='equation_structure_transcription':
            continue
        text=re.sub(r'\s+','',definition.get('definition',''))
        evidence=re.sub(r'\s+','',spans[sid].get('text',''))
        # Do not use an annotation-only description absent from source evidence.
        if text not in evidence:
            continue
        variables=[b.entities.get((v['symbol'],'model_variable')) for v in ex._split_variable(definition)]
        variables=[v for v in variables if v]
        names=ex._method_candidates(text)
        names += [m.group(1) for m in re.finditer(
            r'(?:采用|选择|使用|运用|或)([\u4e00-\u9fffA-Za-z0-9\-]{2,20}?法)(?=或|和|计算|确定|估算|[，。；;）)]|$)',text)]
        names += [m.group(1) for m in re.finditer(
            r'(?:通过|采用|或)((?:(?!或)[\u4e00-\u9fff]){2,14}?(?:调查|观测|检尺))(?=得到|获取|计算|$)',text)]
        for name in dict.fromkeys(names):
            if re.sub(r'\s+','',name) not in evidence:
                continue
            method=b.add_entity(name,'method',[sid])
            for variable in variables:
                b.add_relation(variable,'obtained_by',method,[sid])
        # Materials names use the same noun class as historical rainfall资料.
        # The boundary is explicit acquisition wording, not a document keyword.
        for m in re.finditer(r'(?:通过|依据|根据|来自)([\u4e00-\u9fff]{2,12}?资料)(?=查阅|获取|得到|[，。；;])',text):
            source=b.add_entity(m.group(1),'data_source',[sid])
            for variable in variables:
                b.add_relation(variable,'sourced_from',source,[sid])
    # Explicit method ... quantity(symbol) syntax, already seen in survey
    # training prose. A shared paragraph alone never selects a target variable.
    intro=re.sub(r'\s+','',unit.get('introduction',''))
    for name in ex._method_candidates(intro):
        for variable in list(b.entities.values()):
            if variable['entity_type']!='model_variable':continue
            pattern=re.escape(name)+r'[^，。；;]{0,60}?[（(]'+re.escape(variable['name'])+r'[）)]'
            matches=list(re.finditer(pattern,intro))
            evidence=[sid for sid,s in spans.items() if any(m.group(0) in re.sub(r'\s+','',s.get('text','')) for m in matches)]
            if evidence:
                method=b.add_entity(name,'method',evidence)
                b.add_relation(variable,'obtained_by',method,list(dict.fromkeys(variable['evidence_span_ids']+evidence)))
    adjacent=re.sub(r'\s+','',unit.get('adjacent_source_text',''))
    for m in re.finditer(r'数据来源[：:]([^。；;]+(?:数据|资料))',adjacent):
        name=m.group(1).strip('，,。 ')
        evidence=[sid for sid,s in spans.items() if name in re.sub(r'\s+','',s.get('text',''))]
        if not evidence:
            continue
        source=b.add_entity(name,'data_source',evidence)
        for variable in list(b.entities.values()):
            if variable['entity_type']=='model_variable' and variable['entity_id'] in quantities:
                b.add_relation(variable,'sourced_from',source,list(dict.fromkeys(variable['evidence_span_ids']+evidence)))
    return b.result()


def candidates_for_unit(unit):
    return augment(unit,previous.candidates_for_unit(unit))
