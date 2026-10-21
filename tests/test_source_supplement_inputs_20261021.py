from copy import deepcopy
import json
import pytest
import source_supplement_preflight_20261021 as t


def test_packet_covers_focused_edges_and_whole_source_split():
    p=t.check_packet(t.ROOT)
    assert len(p['units'])==5
    assert p['counts']=={'train':3,'validation':2}
    assert p['focus_positive_counts']=={'train':8,'validation':2}
    assert not set(p['source_sets']['train']) & set(p['source_sets']['validation'])
    assert p['train_source_count']==2 and p['validation_source_count']==1


def test_packet_rejects_cross_source_split_change():
    p=t.check_packet(t.ROOT)
    bad=deepcopy(p['review']);bad['cases'][0]['split']='validation'
    with pytest.raises(AssertionError):t.validate_cases(p['units'],bad,p['source_manifest'],p['annotations'])


def test_packet_rejects_positive_missing_from_real_candidates():
    p=t.check_packet(t.ROOT)
    bad=deepcopy(p['review']);bad['cases'][0]['positive'].append(['invented','obtained_by','invented'])
    with pytest.raises(AssertionError):t.validate_cases(p['units'],bad,p['source_manifest'],p['annotations'])


def test_chain_equation_and_branch_outputs_are_explicit():
    p=t.check_packet(t.ROOT)
    gs={u['provenance']['section']:t.profile.candidates_for_unit(u) for u in p['units'].values()}
    assert {r['tail_name'] for r in gs['5.3.1-soil-conservation']['relations'] if r['relation_type']=='has_output'}=={'A_c'}
    v=gs['6.2.2'];pairs={(r['head_name'],r['tail_name']) for r in v['relations'] if r['relation_type']=='has_output'}
    assert pairs=={('公式（2）','VQI_乔'),('公式（3）','M_林冠层'),('公式（4）','VQI_乔')}
    area=gs['A.1'];inputs={r['tail_name'] for r in area['relations'] if r['relation_type']=='has_input'}
    assert 'n' in inputs and 'i' not in inputs
    assert not any(r['head_name']=='i' and r['relation_type']=='sourced_from' for r in area['relations'])


def test_saved_inputs_rebuild_and_shared_new_examples():
    m=t.verify(t.ROOT)
    assert m['counts']=={'control_train':629,'repair_train':629,'validation':74}
    assert m['paired_changed_training_targets']==6
    a=t.rows(t.ROOT/t.FOLDER/'control_train.jsonl');b=t.rows(t.ROOT/t.FOLDER/'repair_train.jsonl')
    assert a[-3:]==b[-3:]
    assert all(x['messages'][:2]==y['messages'][:2] for x,y in zip(a,b))
    for row in a+b+t.rows(t.ROOT/t.FOLDER/'validation.jsonl'):
        meta=json.loads(row['channel']);q=row['messages'][1]['content'];sel=t.policy.selection_for(q,row['messages'][2]['content'])
        assert not set(meta['unknown_relation_ids']) & set(sel['selected_relation_ids'])
        assert not set(meta['unknown_entity_ids']) & set(sel['selected_entity_ids'])


def test_cpu_guard_and_existing_output_protection(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(AssertionError,match='CPU only'):t.preflight(t.ROOT,tmp_path,tmp_path/'new')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(AssertionError,match='Preserve'):t.preflight(t.ROOT,tmp_path,tmp_path)
