"""Same full graph, actual saved scoped prompts; HTTP and weights are synthetic."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import probe_full_graph_scope_20261019 as probe
from test_full_graph_scope_dev_20261019 import pair as base_pair
from test_reviewed_pair_evaluation import http


@pytest.fixture(scope='module')
def cases(): return probe.build_cases(ROOT)


@pytest.fixture
def pair(base_pair,tmp_path,monkeypatch):
    lock=probe.ev.read(ROOT/probe.LOCK)
    lock['training_manifest_sha256']=probe.ev.EXPECTED_PAIR_SHA
    file=tmp_path/'probe_protocol.json';probe.ev.save(file,lock)
    monkeypatch.setattr(probe,'LOCK',str(file))
    return base_pair


def test_pairs_keep_identical_graph_and_only_change_actual_scope(cases):
    assert len(cases)==16
    full={c['unit']['unit_id']:c for c in cases if c['view']=='full'}
    paired=[c for c in cases if c['view']=='as_supervised']
    assert len(full)==10 and len(paired)==6
    assert sum(c['split']=='train' for c in full.values())==7
    assert sum(len(c['positive_ids']) for c in full.values() if c['split']=='train')==15
    assert sum(len(c['positive_ids']) for c in full.values() if c['split']=='validation')==6
    for c in paired:
        f=full[c['unit']['unit_id']]
        assert c['graph']==f['graph'] and c['candidate_hash']==f['candidate_hash']
        assert c['messages']==c['saved_messages']
        for name in ('positive_ids','confirmed_negative_ids','unknown_ids','semantic_candidate_ids'):
            assert c[name]==f[name]
        a,b=[json.loads(x['messages'][1]['content']) for x in (f,c)]
        assert a['decision_scope']['unscored_relation_ids']==[]
        assert b['decision_scope']['unscored_relation_ids']
        a['decision_scope']['unscored_relation_ids']=b['decision_scope']['unscored_relation_ids']
        assert a==b and f['messages'][0]==c['messages'][0]


def test_unknown_is_not_confirmed_negative_and_scope_violation_is_recorded(cases):
    c=next(c for c in cases if c['view']=='as_supervised' and c['unknown_ids'])
    ids={r['relation_id'] for r in c['graph']['relations']}
    raw=json.dumps({'selected_entity_ids':[],'selected_relation_ids':sorted(ids)})
    pred=probe.assemble_case(c,raw)
    pred.update(experiment_id='fixture',config_hash='fixture',candidate_hash=c['candidate_hash'],
                prompt_hash=c['prompt_hash'],model='fixture',seed=42)
    scored=probe.score_case(c,{r['relation_id'] for r in pred['relations']})
    assert scored['selected_unscored_ids']
    assert scored['selected_unknown_ids']==c['unknown_ids']
    assert not set(scored['selected_unknown_ids'])&set(scored['selected_confirmed_negative_ids'])
    assert probe._validate_unit_prediction(c['unit'],pred)==[]


def test_invalid_member_and_duplicate_selection_rejected(cases):
    c=cases[0];rid=c['graph']['relations'][0]['relation_id']
    for ids in (['not-a-member'],[rid,rid]):
        with pytest.raises(ValueError):probe.assemble_case(c,json.dumps({'selected_entity_ids':[],'selected_relation_ids':ids}))


def test_case_tampering_rejected(cases):
    probe.verify_cases(ROOT,cases)
    changed=deepcopy(cases);changed[0]['positive_ids']=[]
    with pytest.raises(AssertionError,match='cases changed'):probe.verify_cases(ROOT,changed)


def test_real_48_scoped_prompts_proofs_and_diagnostics_no_gate(pair,tmp_path,http,cases):
    with patch('urllib.request.urlopen',side_effect=http['respond']):
        result=probe.run_probe(ROOT,pair,tmp_path/'probe')
    assert len(http['calls'])==result['model_requests']==48
    assert [p for _,_,p in http['calls'][:16]]==[p for _,_,p in http['calls'][16:32]]==[p for _,_,p in http['calls'][32:]]
    assert [p for _,_,p in http['calls'][:16]]==[c['messages'][1]['content'] for c in cases]
    assert not result['training_started'] and not result['three_seed_eligible']
    assert not result['new_checkpoint_selected'] and not result['dev_gate_replaced']
    assert 'gate' not in result
    for arm in probe.ev.MODELS:
        assert result['runs'][arm]['by_split_view']['train:as_supervised']['selected_unscored_ids']>0
    assert len(probe.ev.read_jsonl(tmp_path/'probe/raw_responses.jsonl'))==48


@pytest.mark.parametrize('error',['fail','bad_proof','models'])
def test_failure_has_no_success_summary(pair,tmp_path,http,error):
    http[error]=[] if error=='models' else True
    with patch('urllib.request.urlopen',side_effect=http['respond']),pytest.raises((AssertionError,TimeoutError,ValueError)):
        probe.run_probe(ROOT,pair,tmp_path/'probe')
    assert not (tmp_path/'probe/summary.json').exists()
    if error!='models':assert probe.ev.read_jsonl(tmp_path/'probe/raw_responses.jsonl')[0]['status']=='error'


def test_diagnostic_context_explicit_and_weights_still_selected_by_train(pair):
    cmd=probe.deploy_command(ROOT,probe.ev.inspect_pair(ROOT,pair),8018)
    assert cmd[cmd.index('--max_length')+1]=='16384'
    assert cmd[cmd.index('--max_new_tokens')+1]=='3072'
    assert all(any(v.startswith(probe.ev.MODELS[a]+'=') and v.endswith('checkpoint-120') for v in cmd) for a in ('control','repair'))
