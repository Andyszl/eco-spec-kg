"""The repair audit preserves inputs/results and keeps dev labels out of training."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_selection_repairs_v28 as audit


def test_existing_output_is_rejected_before_work_and_left_intact(tmp_path, monkeypatch):
    marker = tmp_path / "old-result.json"
    marker.write_text("history", encoding="utf-8")
    check = Mock()
    monkeypatch.setattr(audit, "check_delivery", check)
    with pytest.raises(ValueError, match="output already exists"):
        audit.run(tmp_path)
    check.assert_not_called()
    assert marker.read_text(encoding="utf-8") == "history"


def test_invalid_frozen_delivery_is_rejected_before_any_outputs(tmp_path, monkeypatch):
    check = Mock(side_effect=ValueError("delivery hash mismatch"))
    prepare = Mock()
    monkeypatch.setattr(audit, "check_delivery", check)
    monkeypatch.setattr(audit, "prepare_lora_training_v2", prepare)
    output = tmp_path / "fresh"
    with pytest.raises(ValueError, match="delivery hash mismatch"):
        audit.run(output)
    prepare.assert_not_called()
    assert not output.exists()
    assert check.call_args.kwargs["expected_candidate_generator"] == "structure-aware-rule-v2.6"


def test_new_version_cannot_be_used_as_the_historical_baseline(monkeypatch):
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k:
                        'CANDIDATE_GENERATOR_VERSION = "structure-aware-rule-v2.8"\n')
    with pytest.raises(ValueError, match="baseline version"):
        audit.load_baseline()


def test_audit_cannot_pass_if_repairs_gain_no_coverage(tmp_path, monkeypatch):
    real_compare = audit.compare_dev
    monkeypatch.setattr(audit, "compare_dev", lambda units, labels, baseline, current:
                        real_compare(units, labels, baseline, baseline))
    with pytest.raises(ValueError, match="expected coverage"):
        audit.run(tmp_path / "no-repairs")
    saved = json.loads((tmp_path / "no-repairs/verification.json").read_text(encoding="utf-8"))
    assert saved["passed"] is False


def test_frozen_repair_audit_replays_both_splits_but_trains_only_on_train(tmp_path, monkeypatch):
    real_prepare = audit.prepare_lora_training_v2
    prepare = Mock(wraps=real_prepare)
    monkeypatch.setattr(audit, "prepare_lora_training_v2", prepare)
    result = audit.run(tmp_path / "fresh")
    units, gold, _ = prepare.call_args.args
    assert units.name == "train_units.jsonl"
    assert gold.name == "train_annotations.jsonl"
    assert prepare.call_count == 1
    assert result["passed"] is True
    assert result["frozen_inputs_unchanged"] is True
    assert result["model_training_started"] is False
    assert result["training"]["training_records"] == 693
    # These service table rows belong to dev only. Projection must not import
    # their labels to manufacture additional has_indicator training positives.
    indicator_focus = result["training"]["selection_focus_coverage"]["has_indicator"]
    assert indicator_focus["selected_count"] == 3
    assert indicator_focus["selected_unit_count"] == 1
    assert result["dev_coverage"]["baseline"]["covered_relation_count"] == 24
    assert result["dev_coverage"]["current"]["covered_relation_count"] == 33
    assert result["dev_coverage"]["current"]["covered_entity_count"] == 54
    assert result["train_coverage"]["lost_coverage_by_type"] == {}
    assert result["rule_validation"] == {"train": 693, "dev": 22}
    saved = json.loads((tmp_path / "fresh/verification.json").read_text(encoding="utf-8"))
    assert saved == result
