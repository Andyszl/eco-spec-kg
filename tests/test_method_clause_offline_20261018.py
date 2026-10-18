"""Verify train witnesses and refuse altered saved results before output creation."""
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_method_clause_guard_20261018 as audit


def test_train_only_rule_witnesses_and_positive_names():
    result=audit.review_training(ROOT)
    assert result['records']==626 and result['known_positive_method_occurrences']>0
    assert result['rejected_known_positive_names']==[]
    assert set(result['negative_witnesses'])=={'leading_fragment','deictic_formula','descriptive_clause'}
    assert all(result['negative_witnesses'].values())


def test_changed_result_is_rejected_before_output(tmp_path,monkeypatch):
    probe=tmp_path/'probe';dev=tmp_path/'dev';probe.mkdir();dev.mkdir()
    (probe/'sample.json').write_text('changed',encoding='utf-8')
    lock=tmp_path/'lock.json'
    audit.save(lock,{'source_sha256_lf':{},'files':{'probe/sample.json':'0'*64}})
    monkeypatch.setattr(audit,'LOCK',str(lock))
    target=tmp_path/'out'
    with pytest.raises(AssertionError,match='Saved result changed'):
        audit.run(ROOT,probe,dev,target)
    assert not target.exists()
