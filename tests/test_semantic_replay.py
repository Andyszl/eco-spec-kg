"""Review replay must be local, evidence-backed and independent of predictions."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_semantic_review import apply_review, validate_records


def case():
    span={'span_id':'e','page':1,'text':'x=y'}
    unit={'unit_id':'u','provenance':{'evidence_spans':[span]},'formulas':[
        {'formula_number':'1','expression_text':'x y','expression_lines':['x y'],'evidence_span':span}],
        'variable_definitions':[{'symbol':s,'definition':s,'evidence_span':span} for s in ('x','y','i')]}
    annotation={'unit_id':'u','split':'train','entities':[], 'relations':[]}
    spec={'units':[{'unit_id':'u','reason':'source page','formulas':[
        {'number':'1','expression':'x=y','inputs':['y'],'outputs':['x']}]}]}
    return [unit],[annotation],spec


def test_replay_preserves_parent_and_raw_evidence():
    units,labels,spec=case(); before=copy.deepcopy((units,labels))
    new,gold,ledger=apply_review(units,labels,spec)
    assert (units,labels)==before
    assert new[0]['formulas'][0]['raw_expression_text']=='x y'
    assert new[0]['formulas'][0]['evidence_span']['text']=='x=y'
    assert {(r['head_name'],r['relation_type'],r['tail_name']) for r in gold[0]['relations']}=={
        ('公式（1）','has_input','y'),('公式（1）','has_output','x'),('x','calculated_by','公式（1）')}
    assert len(ledger)==1
    validate_records(new,gold)


def test_missing_source_definition_is_rejected():
    units,labels,spec=case(); spec['units'][0]['formulas'][0]['inputs']=['absent']
    with pytest.raises(ValueError,match='defined'):
        apply_review(units,labels,spec)


def test_nontrain_review_is_rejected():
    units,labels,spec=case(); labels[0]['split']='test'
    with pytest.raises(ValueError,match='train'):
        apply_review(units,labels,spec)


def test_new_definition_requires_existing_evidence():
    units,labels,spec=case(); spec['units'][0]['definitions']=[{'symbol':'z','definition':'derived','evidence_span_id':'missing'}]
    with pytest.raises(ValueError,match='evidence'):
        apply_review(units,labels,spec)


def test_input_output_overlap_is_rejected():
    units,labels,spec=case(); spec['units'][0]['formulas'][0]['inputs']=['x']
    with pytest.raises(ValueError,match='overlap'):
        apply_review(units,labels,spec)


def test_reviewed_entity_evidence_follows_correct_definition():
    units,labels,spec=case()
    other={'span_id':'old','text':'unrelated','page':1}
    units[0]['provenance']['evidence_spans'].append(other)
    labels[0]['entities']=[{'entity_id':'x','name':'x','entity_type':'model_variable','evidence_span_ids':['old']}]
    _,gold,_=apply_review(units,labels,spec)
    entity=next(e for e in gold[0]['entities'] if e['name']=='x')
    assert entity['entity_id']=='x'
    assert entity['evidence_span_ids']==['e']
