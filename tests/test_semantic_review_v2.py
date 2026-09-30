"""Source-defined composite terms and bound-index source attribution."""
import pytest
from test_experiment_chain_v2 import source_unit
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
from ecospec_kg.formula_symbols import formula_symbols


def unit_with(expression, symbols):
    unit=source_unit()
    span=unit['variable_definitions'][0]['evidence_span']
    unit['formulas'][0]['expression_text']=expression
    unit['variable_definitions']=[{'symbol':s,'definition':s,'evidence_span':span} for s in symbols]
    return unit


def relations(unit):
    return {(r['head_name'],r['relation_type'],r['tail_name'])
            for r in RuleCandidateExtractorV2().predict_unit(unit)['relations']}


def test_explicit_composite_quantities_are_atomic():
    terms=['∑Cost_PUs','BLM∑Boundary_PUs','∑SPF_ConValue','CostThersholdPenalty(t)']
    unit=unit_with(' + '.join(terms), terms+['Cost','BLM','Boundary','SPF','t'])
    edges=relations(unit)
    assert {t for _,r,t in edges if r=='has_input'}==set(terms)
    assert not any(r=='has_output' for _,r,_ in edges)


def test_composite_match_requires_full_identity_and_declaration():
    assert 'CostThersholdPenalty(t)' not in formula_symbols('OtherCostThersholdPenalty(t)', ['CostThersholdPenalty(t)'])
    assert 'CostThersholdPenalty(t)' not in formula_symbols('CostThersholdPenalty(t)', ['t'])
    assert formula_symbols('CostThersholdPenalty(t)+t', ['CostThersholdPenalty(t)','t']) == {'CostThersholdPenalty(t)','t'}


@pytest.mark.parametrize('free_index', [False,True])
def test_only_bound_index_loses_source_inference(free_index):
    expr='R=sum(i*P_i, i, 1, n)' + ('+i' if free_index else '')
    unit=unit_with(expr,['R','i','P_i','n'])
    unit['adjacent_source_text']='降雨侵蚀力根据降雨量资料获得。'
    for v in unit['variable_definitions']:
        if v['symbol']=='i': v['definition']='所用降雨资料的年份序号'
        if v['symbol']=='P_i': v['definition']='第i年的降雨量'
    edges=relations(unit)
    assert (('i','sourced_from','降雨量资料') in edges) is free_index
    assert ('P_i','sourced_from','降雨量资料') in edges


def test_potential_and_actual_equation_groups_keep_distinct_outputs():
    variables=['S_L潜','S_L','S潜','S','Q_MAX潜','Q_MAX','Z','WF','EF','SCF','K′','C']
    unit=unit_with('S_L潜=2*Z/S潜^2*Q_MAX潜*exp(-(Z/S潜)^2)\nQ_MAX潜=109.8*WF*EF*SCF*K′\nS潜=150.71*(WF*EF*SCF*K′)^(-0.3711)',variables)
    edges=relations(unit)
    assert {t for _,r,t in edges if r=='has_output'} == {'S_L潜','Q_MAX潜','S潜'}
    assert {t for _,r,t in edges if r=='has_input'} == {'Z','WF','EF','SCF','K′'}


def test_carbonate_alias_stays_local():
    unit=unit_with('EF=(sa-0.95*Caco_3)/100',['EF','sa','CaCo_3'])
    unit['formulas'][0]['symbol_aliases']=[{'symbol':'Caco_3','canonical_symbol':'CaCo_3',
        'evidence_span_ids':['span-formula','span-variable'],'reason':'Same-page formula and definition spelling correspondence'}]
    assert ('公式（1）','has_input','CaCo_3') in relations(unit)
    unit['formulas'][0].pop('symbol_aliases')
    assert ('公式（1）','has_input','CaCo_3') not in relations(unit)


def test_structural_notes_are_not_source_text():
    unit=unit_with('R=sum(P_i, i, 1, m)',['R','i','P_i','m'])
    unit['adjacent_source_text']='降雨侵蚀力根据降雨量资料获得。'
    for v in unit['variable_definitions']:
        if v['symbol']=='m':
            v['definition']='降雨计数由局部方程补录'
            v['definition_origin']='equation_structure_transcription'
    edges=relations(unit)
    assert ('m','sourced_from','降雨量资料') not in edges
    assert ('公式（1）','has_input','m') in edges
    assert not any(r=='obtained_by' and '局部方程' in t for h,r,t in edges)
