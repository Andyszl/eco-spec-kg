import json
import shutil
from pathlib import Path

import pytest

from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.extractor_v2 import extract_v2
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2


@pytest.fixture
def confirmed_evaluation(tmp_path):
    public = Path(__file__).resolve().parents[1] / "deliveries/human_confirmed_20261002"
    data = tmp_path / "delivery"
    for relative in (
        "blind/dev_units.jsonl", "gold/dev_annotations.jsonl", "schema_v2.json", "manifest.json"
    ):
        target = data / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(public / relative, target)
    units = data / "blind/dev_units.jsonl"
    gold = data / "gold/dev_annotations.jsonl"
    run = tmp_path / "run"
    extract_v2(units, run)
    validation = validate_predictions_v2(
        units, run / "predictions.jsonl", data / "schema_v2.json", run / "validation"
    )
    assert validation["passed"]
    arguments = (
        gold, run / "validation/validated_predictions.jsonl", units,
        run / "validation/validation_report.json"
    )
    return data, arguments


def test_evaluation_recognizes_confirmed_train_dev_delivery(confirmed_evaluation, tmp_path):
    data, arguments = confirmed_evaluation
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    report = evaluate_v2(*arguments, tmp_path / "evaluation")
    for field in ("dataset_version", "package_id", "gold_nature", "review_method"):
        assert report["dataset"][field] == manifest[field]
    assert report["dataset"]["human_expert_review_required_for_publication"] is False


def test_evaluation_delivery_metadata_does_not_change_scores(confirmed_evaluation, tmp_path):
    data, arguments = confirmed_evaluation
    confirmed = evaluate_v2(*arguments, tmp_path / "confirmed")
    (data / "manifest.json").unlink()
    unknown = evaluate_v2(*arguments, tmp_path / "unknown")
    assert unknown["dataset"]["gold_nature"] == "unknown"
    for field in (
        "entities", "relations", "by_standard", "by_unit_type", "evidence",
        "paths", "unit_exact_match", "error_count"
    ):
        assert confirmed[field] == unknown[field]


def test_evaluation_rejects_modified_delivery_gold(confirmed_evaluation, tmp_path):
    _, arguments = confirmed_evaluation
    with arguments[0].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="dataset file hash differs"):
        evaluate_v2(*arguments, tmp_path / "evaluation")


def test_evaluation_rejects_unregistered_delivery_units(confirmed_evaluation, tmp_path):
    data, arguments = confirmed_evaluation
    path = data / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != "blind/dev_units.jsonl"
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="file is not registered"):
        evaluate_v2(*arguments, tmp_path / "evaluation")


def test_evaluation_rejects_mixed_delivery_packages(confirmed_evaluation, tmp_path):
    data, arguments = confirmed_evaluation
    other = tmp_path / "other"
    (other / "gold").mkdir(parents=True)
    other_gold = other / "gold/dev_annotations.jsonl"
    shutil.copyfile(arguments[0], other_gold)
    shutil.copyfile(data / "manifest.json", other / "manifest.json")
    with pytest.raises(ValueError, match="different packages"):
        evaluate_v2(other_gold, *arguments[1:], tmp_path / "evaluation")
