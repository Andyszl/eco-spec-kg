"""Test source-only diagnostics with synthetic HTTP and adapters, no GPU."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import probe_cross_source_learning_20261017 as probe
import evaluate_cross_source_pair_20261016 as ev
from test_cross_source_dev_20261016 import pair
from test_reviewed_pair_evaluation import http


@pytest.fixture
def probe_pair(pair,tmp_path,monkeypatch):
    lock=ev.read(ROOT/probe.LOCK)
    lock['training_manifest_sha256']=ev.EXPECTED_PAIR_SHA
    target=tmp_path/'probe_lock.json';ev.save(target,lock)
    monkeypatch.setattr(probe,'LOCK',str(target))
    return pair


def test_cases_are_positive_source_units_and_exact_training_questions():
    cases=probe.build_cases(ROOT)
    full=[c for c in cases if c['view']=='full']
    assert len(full)==10 and sum(c['split']=='train' for c in full)==7
    assert sum(len(c['positive_ids']) for c in full)==21
    assert all(c['split'] in ('train','validation') and c['positive_ids'] for c in cases)
    for c in cases:
        assert not set(c['positive_ids']) & set(c['unknown_ids'])
        assert not set(c['confirmed_negative_ids']) & set(c['unknown_ids'])
        system,prompt=ev.ex.build_llm_selection_messages(c['unit'],c['graph'])
        assert c['messages']==[{'role':'system','content':system},{'role':'user','content':prompt}]
        if c['view']=='as_supervised':assert c['messages']==c['saved_messages']


def test_fixed_adapters_both_views_all_models_no_gate_change(probe_pair,tmp_path,http):
    with patch('urllib.request.urlopen',side_effect=http['respond']):
        result=probe.run_probe(ROOT,probe_pair,tmp_path/'probe')
    cases=probe.build_cases(ROOT);n=len(cases)
    assert len(http['calls'])==n*3 and result['model_requests']==n*3
    prompts=[p for _,_,p in http['calls']]
    assert prompts[:n]==prompts[n:2*n]==prompts[2*n:]
    assert not result['training_started'] and not result['dev_gate_replaced'] and not result['new_checkpoint_selected']
    for arm in ev.MODELS:
        for value in result['runs'][arm]['cases']:
            assert not value['missed_positive_ids']
            assert set(value['selected_unknown_ids']).isdisjoint(value['selected_confirmed_negative_ids'])
    assert (tmp_path/'probe/raw_responses.jsonl').is_file()


@pytest.mark.parametrize('failure',['fail','bad_proof','models'])
def test_invalid_response_cannot_emit_summary(probe_pair,tmp_path,http,failure):
    http[failure]=[] if failure=='models' else True
    with patch('urllib.request.urlopen',side_effect=http['respond']),pytest.raises((AssertionError,ValueError,TimeoutError)):
        probe.run_probe(ROOT,probe_pair,tmp_path/'failed')
    assert not (tmp_path/'failed/summary.json').exists()


def test_unknown_is_not_scored_negative():
    cases=probe.build_cases(ROOT)
    c=next(c for c in cases if c['view']=='full' and c['unknown_ids'])
    value=probe.score_case(c,set(c['unknown_ids'])|set(c['positive_ids']))
    assert value['selected_unknown_ids'] and not value['selected_confirmed_negative_ids']
    assert not value['missed_positive_ids']


def test_diagnostic_context_does_not_replace_dev(pair):
    manifest=ev.inspect_pair(ROOT,pair)
    expected=ev.deploy_command(ROOT,manifest,8018)
    expected[expected.index('--max_length')+1]='16384'
    assert probe.deploy_command(ROOT,manifest,8018)==expected


def test_changed_case_hash_is_rejected():
    cases=probe.build_cases(ROOT)
    cases[0]['prompt_hash']='changed'
    with pytest.raises(AssertionError,match='Probe cases changed'):
        probe.verify_cases(ROOT,cases)
