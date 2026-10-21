from copy import deepcopy
from ecospec_kg import candidate_profile_v228 as p


def unit():
    def span(i, text):
        return dict(span_id=i, text=text, page=1, printed_page='1', bbox=[0,0,100,20], line_ids=[i])
    f=span('f','Z=(X-Y)/Y'); x=span('x','X——采用样带积分法计算的面积'); y=span('y','Y——基期面积')
    s=span('s','数据来源：某部门调查数据。')
    return dict(unit_id='synthetic-unseen',unit_type='formula_package',schema_version='source-unit-v2.0',
        introduction='面积变化率按照公式（9）计算。',adjacent_source_text=s['text'],interstitial_text='',
        formulas=[dict(formula_number='9',expression_text='Z=(X-Y)/Y',expression_lines=['Z=(X-Y)/Y'],evidence_span=f)],
        variable_definitions=[dict(symbol='Z',definition='面积变化率',unit='',evidence_span=f),
            dict(symbol='X',definition='采用样带积分法计算的面积',unit='',evidence_span=x),
            dict(symbol='Y',definition='基期面积',unit='',evidence_span=y)],
        provenance=dict(standard_code='SYNTHETIC',document_title='调查评价',source_file='x.pdf',source_sha256='a'*64,
            section='A',heading_chain=['面积变化率'],pages=[1],printed_pages=['1'],evidence_spans=[f,x,y,s]))


def test_local_definition_adds_input_candidate_without_selecting_or_moving_method():
    u=unit(); before=deepcopy(u); g=p.candidates_for_unit(u)
    assert u==before
    r=next(r for r in g['relations'] if (r['head_name'],r['relation_type'],r['tail_name'])==('X','obtained_by','样带积分法'))
    assert 'x' in r['evidence_span_ids']
    assert not any(r['head_name']=='Y' and r['relation_type']=='obtained_by' for r in g['relations'])
    assert 'selected_relation_ids' not in g


def test_source_heading_offers_all_variables_as_candidates_not_labels():
    g=p.candidates_for_unit(unit())
    assert {r['head_name'] for r in g['relations'] if r['relation_type']=='sourced_from' and r['tail_name']=='某部门调查数据'}=={'Z','X','Y'}


def test_no_evidence_no_new_candidates():
    u=unit(); u['provenance']['evidence_spans']=u['provenance']['evidence_spans'][:1]
    u['variable_definitions'][1]['evidence_span']['text']='X——面积'
    g=p.candidates_for_unit(u)
    assert not any(r['head_name']=='X' and r['relation_type']=='obtained_by' for r in g['relations'])


def test_materials_name_is_bounded_and_not_a_global_source_label():
    u=unit(); u['variable_definitions'][1]['definition']='通过历史调查资料查阅得到的面积'
    u['variable_definitions'][1]['evidence_span']['text']='X——通过历史调查资料查阅得到的面积'
    u['provenance']['evidence_spans'][1]=deepcopy(u['variable_definitions'][1]['evidence_span'])
    g=p.candidates_for_unit(u)
    assert any((r['head_name'],r['relation_type'],r['tail_name'])==('X','sourced_from','历史调查资料') for r in g['relations'])
    assert not any(r['head_name']=='Y' and r['tail_name']=='历史调查资料' for r in g['relations'])


def test_optional_method_list_has_bounded_names():
    u=unit();d=u['variable_definitions'][1]
    d['definition']='现期面积，可选择分层积分法或面积交叠法计算'
    d['evidence_span']['text']='X——'+d['definition']
    u['provenance']['evidence_spans'][1]=deepcopy(d['evidence_span'])
    g=p.candidates_for_unit(u)
    assert {('X','obtained_by',n) for n in ('分层积分法','面积交叠法')} <= {
        (r['head_name'],r['relation_type'],r['tail_name']) for r in g['relations']}


def test_local_acquisition_and_explicit_symbol_binding():
    u=unit();d=u['variable_definitions'][1]
    d['definition']='通过历史调查资料查阅或现场抽样调查得到的面积'
    d['evidence_span']['text']='X——'+d['definition']
    u['provenance']['evidence_spans'][1]=deepcopy(d['evidence_span'])
    u['introduction']='采用每木检尺得到的胸高断面积（Y）作为计算输入。'
    u['provenance']['evidence_spans'].append(dict(span_id='intro',text=u['introduction'],page=1,printed_page='1',bbox=[0,0,100,20],line_ids=['intro']))
    g=p.candidates_for_unit(u)
    keys={(r['head_name'],r['relation_type'],r['tail_name']) for r in g['relations']}
    assert ('X','obtained_by','现场抽样调查') in keys
    assert ('Y','obtained_by','每木检尺') in keys
    assert ('X','obtained_by','每木检尺') not in keys


def test_named_parenthetical_equation_does_not_include_leading_prose():
    u=unit();u['introduction']='评价采用规范中推荐的空间离散方程（SDX）计算。'
    u['provenance']['evidence_spans'].append(dict(span_id='intro',text=u['introduction'],page=1,printed_page='1',bbox=[0,0,100,20],line_ids=['intro']))
    g=p.candidates_for_unit(u)
    assert any((r['head_name'],r['relation_type'],r['tail_name'])==('Z','obtained_by','空间离散方程（SDX）') for r in g['relations'])
