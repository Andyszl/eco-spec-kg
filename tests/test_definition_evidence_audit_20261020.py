import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_definition_evidence_20261020 as audit


def test_training_witnesses_use_only_locked_train_records():
    result=audit.review_training(ROOT)
    assert result['records']==626
    assert result['witnesses']
    assert all(w['definition_span']['span_id'] in w['approved_span_ids'] for w in result['witnesses'])
    assert result['validation_dev_test_used_for_rule_derivation'] is False


def test_tampered_saved_prediction_blocked_before_review_or_output(tmp_path,monkeypatch):
    dev=tmp_path/'dev';dev.mkdir();out=tmp_path/'out'
    file=dev/'summary.json';file.write_text('{}')
    lock=tmp_path/'lock.json'
    lock.write_text(json.dumps({'source_sha256_lf':{},'dev_files_sha256':{'summary.json':'0'*64}}))
    monkeypatch.setattr(audit,'LOCK',str(lock))
    with pytest.raises(AssertionError,match='Saved dev result changed'):
        audit.run(ROOT,dev,out)
    assert not out.exists()
