import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import verify_named_method_review_20261016 as verify


def test_all_train_sources_preserve_existing_candidates_and_review_is_not_training():
    result=verify.verify(ROOT)
    assert result["checked_units_and_views"]==696
    assert result["changed_units"]==1 and result["new_method_entities"]==1 and result["new_method_relations"]==2
    assert result["training_ready"] is False and result["new_annotation_used_in_training"] is False


def test_changed_review_definition_is_rejected_before_candidate_replay(monkeypatch):
    original=verify.read
    def changed(path):
        data=original(path)
        if path.name=="review.json":data["source_definition"]["definition"]="altered"
        return data
    monkeypatch.setattr(verify,"read",changed)
    with pytest.raises(AssertionError,match="definition"):
        verify.verify(ROOT)
