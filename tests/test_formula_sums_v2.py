"""Sum notation keeps full variable identity and binds only local indices."""
import pytest

from test_experiment_chain_v2 import source_unit
from ecospec_kg.formula_symbols import formula_symbols
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2


def edges(expression, symbols):
    unit = source_unit()
    span = unit['variable_definitions'][0]['evidence_span']
    unit['variable_definitions'] = [
        {'symbol':s, 'definition':s, 'evidence_span':span} for s in symbols
    ]
    unit['formulas'][0]['expression_text'] = expression
    candidate = RuleCandidateExtractorV2().predict_unit(unit)
    return {(r['relation_type'],r['tail_name']) for r in candidate['relations']
            if r['relation_type'] in {'has_input','has_output'}}


@pytest.mark.parametrize('expression', ['R = sum(R_半月k, k, 1, 24)', 'R=sum(R_半月k,k,1,24)'])
def test_a5_subscript_argument_and_bound_index(expression):
    assert edges(expression, ['R','R_半月k','k']) == {
        ('has_output','R'), ('has_input','R_半月k')}


def test_trailing_punctuation_is_not_part_of_subscript():
    assert formula_symbols('x_i, y_j,', ['x_i','y_j']) == {'x_i','y_j'}


@pytest.mark.parametrize('expression', [
    'R_半月k = (1/n) * sum(sum(α * P_i,j,k^1.7265, j, 0, m), i, 1, n)',
    'R_半月k=(1/n)*sum(sum(α*P_i,j,k^1.7265,j,0,m),i,1,n)',
])
def test_a6_nested_sum_retains_bounds_and_indexed_quantity(expression):
    assert edges(expression,['R_半月k','α','P_i,j,k','i','j','k','m','n']) == {
        ('has_output','R_半月k'), *{('has_input',s) for s in ('α','P_i,j,k','m','n')}}


@pytest.mark.parametrize('expression,inputs', [
    ('Q=sum(i*x_i, i, 1, n)', {'x_i','n'}),
    ('Q=sum(i*x_i, i, 1, n)+i', {'x_i','n','i'}),
    ('Q=sum(x_i, i, 1, i)', {'x_i','i'}),
    ('Q=sum(sum(i*j*x_i,j,j,1,m),i,1,n)', {'x_i,j','m','n'}),
    ('Q=sum(x_i,i,1,n)+sum(i*x_j,j,1,m)', {'x_i','x_j','i','m','n'}),
    ('Q=sum(x_i)', {'x_i'}),
])
def test_index_scope_and_nonbinding_sum(expression,inputs):
    symbols=['Q','i','j','n','m','x_i','x_j','x_i,j']
    assert edges(expression,symbols) == {('has_output','Q'), *{('has_input',s) for s in inputs}}


def test_nested_index_assignment_is_not_a_formula_output():
    result=edges('Q=sum(x_i, i=1, n)', ['Q','x_i','i','n'])
    assert {name for role,name in result if role=='has_output'} == {'Q'}


def test_piecewise_comparison_does_not_create_output():
    assert edges('S=piecewise(x if θ <= a; y if θ >= b)', ['S','x','y','θ','a','b']) == {
        ('has_output','S'), *{('has_input',s) for s in ('x','y','θ','a','b')}}


def test_unbound_index_in_separate_formula_remains_input():
    assert edges('i=n\nQ=i*x_i', ['Q','i','n','x_i']) == {
        ('has_output','i'), ('has_output','Q'), ('has_input','n'), ('has_input','x_i')}


def test_subscript_identity_is_not_replaced_by_bare_index():
    assert edges('Q=sum(x_i,j, k, 1, n)', ['Q','x_i,j','x_i','i','j','k','n']) == {
        ('has_output','Q'), ('has_input','x_i,j'), ('has_input','n')}


def test_braced_subscript_and_free_bound_in_outer_sum():
    assert edges('Q=sum(sum(x_{i,j},j,1,i),i,1,n)', ['Q','x_i,j','i','j','n']) == {
        ('has_output','Q'), ('has_input','x_i,j'), ('has_input','n')}


def test_bare_comparison_is_not_an_output_assignment():
    assert edges('x <= a', ['x','a']) == {('has_input','x'), ('has_input','a')}
