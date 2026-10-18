"""Full context must remain identical; scope expresses missing decisions, not negatives."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ecospec_kg import full_graph_scope_v219 as scope
from ecospec_kg import candidate_profile_v214 as profile
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.io_utils import read_jsonl
from ecospec_kg.selection_decoding_v2 import candidate_ids
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def case():
    unit=read_jsonl(ROOT/'experiments/semantic_pair_v211_20261015/units.jsonl')[0]
    # Choose a nonempty source graph generically, without a named dev entity.
    for unit in read_jsonl(ROOT/'experiments/semantic_pair_v211_20261015/units.jsonl'):
        graph=profile.candidates_for_unit(unit)
        if graph['relations']:return unit,graph
    raise AssertionError('No fixture relation')


def test_scope_retains_every_candidate_and_source_field(case):
    unit,graph=case;rid=graph['relations'][0]['relation_id']
    original=json.loads(ex.build_llm_selection_messages(unit,graph)[1])
    _,text=scope.build_messages(unit,graph,[rid]);q=json.loads(text)
    for key in ('source_unit','candidate_entities','candidate_relations','formula_output_bindings','selection_focus','output_schema'):
        assert q[key]==original[key]
    assert rid in candidate_ids(text)[1]
    assert q['decision_scope']['unscored_relation_ids']==[rid]
    assert q['decision_scope']['version']==scope.VERSION


def test_inference_default_scores_full_graph_without_training_review(case):
    unit,graph=case
    _,text=scope.build_messages(unit,graph)
    assert json.loads(text)['decision_scope']['unscored_relation_ids']==[]
    assert set(candidate_ids(text)[1])=={r['relation_id'] for r in graph['relations']}


def test_unknown_scope_reference_and_positive_conflict_fail(case):
    unit,graph=case
    with pytest.raises(ValueError,match='absent'):
        scope.build_messages(unit,graph,['0'*16])
    rid=graph['relations'][0]['relation_id']
    _,text=scope.build_messages(unit,graph,[rid])
    selection={'selected_entity_ids':[],'selected_relation_ids':[rid]}
    with pytest.raises(ValueError,match='unscored'):
        scope.validate_selection(text,selection)


def test_selection_uses_scoped_question_and_preserves_endpoint_assembly(case):
    unit,graph=case;rid=graph['relations'][0]['relation_id'];calls=[]
    def complete(system,prompt):
        calls.append(prompt)
        return json.dumps({'selected_entity_ids':[],'selected_relation_ids':[rid]})
    selected,raw,ph=scope.select(SimpleNamespace(complete=complete),unit,graph)
    assert selected['relations']==[graph['relations'][0]]
    assert {e['entity_id'] for e in selected['entities']}=={graph['relations'][0][k] for k in ('head_id','tail_id')}
    assert len(calls)==1 and json.loads(calls[0])['decision_scope']['unscored_relation_ids']==[]
