import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_method_scope_20261016 as verification


def test_full_training_and_validation_replay_preserves_known_labels():
    result = verification.audit(ROOT)
    assert result["train_records"] == 625 and result["validation_records"] == 70
    assert result["reviewed_unknown_original_edges"] == 16
    assert result["quarantined_canonical_training_edges"] == 10
    assert result["known_positive_edges_lost"] == 0
    assert result["unknown_removed_from_inference"] == 0
    assert result["training_ready"] is False


def test_review_cannot_include_validation_or_nonexistent_unit(monkeypatch):
    original = verification.read
    def tampered(path):
        value = original(path)
        if path.name == "source_review.json":
            value["cases"][0]["unit_id"] = "9fc3c8a70f70b39e"
        return value
    monkeypatch.setattr(verification, "read", tampered)
    with pytest.raises(AssertionError, match="train"):
        verification.audit(ROOT)
