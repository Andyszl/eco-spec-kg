"""Refresh confirmed dev metadata using saved predictions; no model requests."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from ecospec_kg.evaluation_v2 import evaluate_v2


SCORE_FIELDS = (
    "entities", "relations", "paths", "evidence", "by_standard",
    "by_unit_type", "unit_exact_match", "error_count",
)


def refresh_report(run: Path) -> Path:
    run = run.resolve()
    data = ROOT / "deliveries/human_confirmed_20261002"
    gold = data / "gold/dev_annotations.jsonl"
    units = data / "blind/dev_units.jsonl"
    predictions = run / "validation/validated_predictions.jsonl"
    validation = run / "validation/validation_report.json"
    previous = run / "evaluation/metrics.json"
    manifest_path = data / "manifest.json"
    for path in (gold, units, predictions, validation, previous, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    old = json.loads(previous.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gold_nature") != "human_expert_gold" or manifest.get("review_method") != "ai_assisted_human_confirmed":
        raise ValueError("Delivery metadata does not match the confirmed review method")
    out = run / ("evaluation_confirmed_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(exist_ok=False)
    new = evaluate_v2(gold, predictions, units, validation, out)
    for field in SCORE_FIELDS:
        if old[field] != new[field]:
            raise ValueError(f"Metrics differ: {field}; inspect the new report at {out}")
    for field in ("dataset_version", "package_id", "gold_nature", "review_method"):
        if new["dataset"][field] != manifest[field]:
            raise ValueError(f"Dataset metadata differs: {field}")
    if new["dataset"]["human_expert_review_required_for_publication"] is not False:
        raise ValueError("Confirmed review metadata was not recognized")
    return out / "metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing dev run directory")
    args = parser.parse_args()
    try:
        metrics_path = refresh_report(args.run)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Report refresh failed: {exc}", file=sys.stderr)
        return 1
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps(report["dataset"], ensure_ascii=False, indent=2))
    print("Original predictions and metric values are unchanged.")
    print(f"New report: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
