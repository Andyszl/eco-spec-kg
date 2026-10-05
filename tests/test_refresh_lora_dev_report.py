import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.extractor_v2 import extract_v2
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/refresh_lora_dev_report.py"
DATA = ROOT / "deliveries/human_confirmed_20261002"
SCORE_FIELDS = (
    "entities", "relations", "paths", "evidence", "by_standard",
    "by_unit_type", "unit_exact_match", "error_count",
)


@pytest.fixture
def old_run(tmp_path):
    run = tmp_path / "existing_run"
    units = DATA / "blind/dev_units.jsonl"
    extract_v2(units, run)
    validated = validate_predictions_v2(
        units, run / "predictions.jsonl", DATA / "schema_v2.json", run / "validation"
    )
    assert validated["passed"]
    report = evaluate_v2(
        DATA / "gold/dev_annotations.jsonl",
        run / "validation/validated_predictions.jsonl", units,
        run / "validation/validation_report.json", run / "evaluation",
    )
    report["dataset"] = {"dataset_version": "unknown", "gold_nature": "unknown"}
    (run / "evaluation/metrics.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return run, report


def invoke(run, cwd):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run", str(run)], cwd=cwd,
        env=env, capture_output=True, text=True, encoding="utf-8",
    )


def file_hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def test_refresh_runs_from_home_without_pythonpath_and_preserves_old_files(old_run, tmp_path):
    run, old = old_run
    original = file_hashes(run)
    result = invoke(run, tmp_path)
    assert result.returncode == 0, result.stderr
    outputs = list(run.glob("evaluation_confirmed_*/metrics.json"))
    assert len(outputs) == 1
    new = json.loads(outputs[0].read_text(encoding="utf-8"))
    for field in SCORE_FIELDS:
        assert new[field] == old[field]
    assert new["dataset"]["gold_nature"] == "human_expert_gold"
    assert new["dataset"]["review_method"] == "ai_assisted_human_confirmed"
    assert new["dataset"]["human_expert_review_required_for_publication"] is False
    assert all(file_hashes(run)[name] == digest for name, digest in original.items())
    assert str(outputs[0]) in result.stdout


def test_refresh_missing_prediction_stops_before_output_creation(old_run, tmp_path):
    run, _ = old_run
    (run / "validation/validated_predictions.jsonl").unlink()
    original = file_hashes(run)
    result = invoke(run, tmp_path)
    assert result.returncode != 0
    assert "validated_predictions.jsonl" in result.stderr
    assert not list(run.glob("evaluation_confirmed_*"))
    assert file_hashes(run) == original


def test_refresh_reports_score_difference_without_overwriting_original(old_run, tmp_path):
    run, old = old_run
    old["error_count"] += 1
    metrics = run / "evaluation/metrics.json"
    metrics.write_text(json.dumps(old), encoding="utf-8")
    original = file_hashes(run)
    result = invoke(run, tmp_path)
    assert result.returncode != 0
    assert "error_count" in result.stderr
    assert all(file_hashes(run)[name] == digest for name, digest in original.items())


def test_refresh_repeat_creates_separate_reports(old_run, tmp_path):
    run, _ = old_run
    first = invoke(run, tmp_path)
    assert first.returncode == 0, first.stderr
    first_files = file_hashes(run)
    second = invoke(run, tmp_path)
    assert second.returncode == 0, second.stderr
    assert len(list(run.glob("evaluation_confirmed_*/metrics.json"))) == 2
    assert all(file_hashes(run)[name] == digest for name, digest in first_files.items())
