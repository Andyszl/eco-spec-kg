from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation_v2 import evaluate_v2
from .experiment_data_v2 import prepare_experiment_package_v2
from .extractor_v2 import extract_v2
from .foundation_quality import run_foundation_gate
from .io_utils import write_json
from .layout_parser import parse_layout_corpus
from .ontology_v2 import schema_quality_report, schema_rows_v2
from .prediction_validation_v2 import validate_predictions_v2
from .source_units import build_units_from_layout_dir
from .training_v2 import prepare_lora_training_v2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ecospec_kg.cli_v2")
    commands = parser.add_subparsers(dest="command", required=True)

    parse_layout = commands.add_parser("parse-layout")
    parse_layout.add_argument("--source", type=Path, required=True)
    parse_layout.add_argument("--out", type=Path, required=True)
    parse_layout.add_argument("--standards", nargs="*")

    build_units = commands.add_parser("build-units")
    build_units.add_argument("--layouts", type=Path, required=True)
    build_units.add_argument("--out", type=Path, required=True)

    schema = commands.add_parser("schema-report")
    schema.add_argument("--out", type=Path)

    gate = commands.add_parser("validate-foundations")
    gate.add_argument("--source", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    gate.add_argument("--pilot", type=Path)

    prepare = commands.add_parser("prepare-experiment-v2")
    prepare.add_argument("--source-units", type=Path, required=True)
    prepare.add_argument("--annotations", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--dataset-version", default="v2.1")
    prepare.add_argument("--gold-nature", default="ai_expert_pre_gold")

    extract = commands.add_parser("extract-v2")
    extract.add_argument("--units", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--config", type=Path)
    extract.add_argument("--backend", choices=("rule", "llm"))
    extract.add_argument("--model")
    extract.add_argument("--seed", type=int)

    prepare_lora = commands.add_parser("prepare-lora-v2")
    prepare_lora.add_argument("--units", type=Path, required=True)
    prepare_lora.add_argument("--annotations", type=Path, required=True)
    prepare_lora.add_argument("--out", type=Path, required=True)

    validate = commands.add_parser("validate-predictions-v2")
    validate.add_argument("--units", type=Path, required=True)
    validate.add_argument("--predictions", type=Path, required=True)
    validate.add_argument("--schema", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)

    evaluate = commands.add_parser("evaluate-v2")
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--units", type=Path, required=True)
    evaluate.add_argument("--validation-report", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "parse-layout":
        summary = parse_layout_corpus(
            args.source,
            args.out,
            set(args.standards) if args.standards else None,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "build-units":
        summary = build_units_from_layout_dir(args.layouts, args.out)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "schema-report":
        report = {
            **schema_quality_report(),
            "schema_rows": schema_rows_v2(),
        }
        if args.out:
            write_json(args.out, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["passed"] else 1
    if args.command == "validate-foundations":
        report = run_foundation_gate(
            args.source,
            args.out,
            args.pilot,
        )
        summary = {
            "passed": report["passed"],
            "document_parser_quality": report["document_parser_quality"],
            "schema_professionalism": report["schema_professionalism"],
            "report": str(args.out / "foundation_quality_report.json"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if report["passed"] else 1
    if args.command == "prepare-experiment-v2":
        manifest = prepare_experiment_package_v2(
            args.source_units,
            args.annotations,
            args.out,
            dataset_version=args.dataset_version,
            gold_nature=args.gold_nature,
        )
        summary = {
            "package_id": manifest["package_id"],
            "dataset_version": manifest["dataset_version"],
            "gold_nature": manifest["gold_nature"],
            "unit_split_counts": manifest["unit_split_counts"],
            "relation_split_counts": manifest["relation_split_counts"],
            "manifest": str(args.out / "manifest.json"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "extract-v2":
        manifest = extract_v2(
            args.units,
            args.out,
            config_path=args.config,
            backend=args.backend,
            model=args.model,
            seed=args.seed,
        )
        print(json.dumps(manifest["summary"], ensure_ascii=False))
        return 0 if manifest["status"] == "complete" else 1
    if args.command == "prepare-lora-v2":
        manifest = prepare_lora_training_v2(
            args.units,
            args.annotations,
            args.out,
        )
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    if args.command == "validate-predictions-v2":
        report = validate_predictions_v2(
            args.units,
            args.predictions,
            args.schema,
            args.out,
        )
        summary = {
            "passed": report["passed"],
            "source_unit_count": report["source_unit_count"],
            "valid_prediction_unit_count": report[
                "valid_prediction_unit_count"
            ],
            "failure_count": report["failure_count"],
            "report": str(args.out / "validation_report.json"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if report["passed"] else 1
    if args.command == "evaluate-v2":
        report = evaluate_v2(
            args.gold,
            args.predictions,
            args.units,
            args.validation_report,
            args.out,
        )
        summary = {
            "gold_nature": report["dataset"]["gold_nature"],
            "entity_f1": report["entities"]["micro"]["f1"],
            "relation_f1": report["relations"]["strict_micro"]["f1"],
            "macro_relation_f1": report["relations"][
                "macro_relation_f1"
            ],
            "report": str(args.out / "metrics.json"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
