from copy import deepcopy
import pytest
from ecospec_kg import definition_evidence_v223 as evidence
from ecospec_kg.structural_selection_v2 import merge_structural_relations
from test_structural_selection_v2 import fixture


def prepared():
    unit,graph=fixture()
    unit['variable_definitions'][0]['evidence_span']['text']='Y——冠层覆盖率；'
    sid=unit['formulas'][0]['evidence_span']['span_id']
    for e in graph['entities']:
        if e['entity_type']=='assessment_indicator':e['evidence_span_ids']=[sid]
    pred,_=merge_structural_relations(unit,graph,{**graph,'entities':[],'relations':[]})
    return unit,graph,pred


def test_only_explicit_definition_evidence_changes_and_is_idempotent():
    u,c,p=prepared();original=deepcopy((u,c,p))
    out,proof=evidence.attach_definition_evidence(u,c,p)
    assert len(proof['changes'])==1
    changed=proof['changes'][0];sid=u['variable_definitions'][0]['evidence_span']['span_id']
    assert changed['added_span_ids']==[sid]
    assert out['relations']==p['relations'] and (u,c,p)==original
    assert [{k:v for k,v in e.items() if k!='evidence_span_ids'} for e in out['entities']]==[{k:v for k,v in e.items() if k!='evidence_span_ids'} for e in p['entities']]
    again,proof2=evidence.attach_definition_evidence(u,c,out)
    assert again==out and proof2['changes']==[]


@pytest.mark.parametrize('kind',['missing_root','missing_output','missing_inverse','transcription','qualifier','source_text_mismatch','compound'])
def test_no_definition_claim_without_complete_explicit_support(kind):
    u,c,p=prepared()
    if kind.startswith('missing_'):
        remove={'missing_root':lambda r:r['head_type']=='assessment_indicator',
                'missing_output':lambda r:r['relation_type']=='has_output',
                'missing_inverse':lambda r:r['head_type']=='model_variable' and r['relation_type']=='calculated_by'}[kind]
        p['relations']=[r for r in p['relations'] if not remove(r)]
    elif kind=='transcription':u['variable_definitions'][0]['definition_origin']='equation_structure_transcription'
    elif kind=='qualifier':u['variable_definitions'][0]['definition']='像元冠层覆盖率；'
    elif kind=='source_text_mismatch':u['variable_definitions'][0]['evidence_span']['text']='Y——别的指标；'
    elif kind=='compound':u['formulas'][0]['expression_text']='Y + K = X'
    out,proof=evidence.attach_definition_evidence(u,c,p)
    assert not proof['changes'] and out['entities']==p['entities']


def test_renamed_source_formula_and_indicator_remain_generic():
    from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
    u,_,_=prepared();u['unit_id']='another-source';u['formulas'][0]['formula_number']='C.97'
    u['variable_definitions'][0]['definition']='冠层温度；'
    u['variable_definitions'][0]['evidence_span']['text']='Y——冠层温度；'
    u['provenance']['heading_chain']=['冠层温度']
    c=RuleCandidateExtractorV2().predict_unit(u)
    # Heading recognition is not essential to the evidence rule; supply the
    # same generic candidate rename if the fixture heading is constructed elsewhere.
    for e in c['entities']:
        if e['name']=='冠层覆盖率':e['name']='冠层温度'
    for r in c['relations']:
        for side in ('head','tail'):
            if r[side+'_name']=='冠层覆盖率':r[side+'_name']='冠层温度'
    for e in c['entities']:
        if e['entity_type']=='assessment_indicator':e['evidence_span_ids']=[u['formulas'][0]['evidence_span']['span_id']]
    p,_=merge_structural_relations(u,c,{**c,'entities':[],'relations':[]})
    out,proof=evidence.attach_definition_evidence(u,c,p)
    assert len(proof['changes'])==1


@pytest.mark.parametrize('kind',['entity','relation','duplicate','outside_span'])
def test_forged_predictions_rejected(kind):
    u,c,p=prepared()
    if kind=='entity':p['entities'][0]['name']='invented'
    elif kind=='relation':p['relations'][0]['tail_name']='invented'
    elif kind=='duplicate':p['entities'].append(p['entities'][0])
    else:p['entities'][0]['evidence_span_ids'].append('outside-source')
    with pytest.raises(ValueError):evidence.attach_definition_evidence(u,c,p)
