from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import (
    MicrosoftGraphRAGAdapter,
    NativeGraphRAGAdapter,
    load_chunks,
    load_relations,
)
from .app import launch
from .corpus import build_corpus
from .experiments import evaluate_files, write_ablation_plan
from .extraction import RuleExtractor
from .io_utils import read_jsonl, write_csv, write_json, write_jsonl
from .models import Relation
from .reporting import render_report
from .training import LoRASettings, prepare_training_data, run_lora


ANNOTATION_FIELDS = [
    "record_id",
    "path_id",
    "head_name",
    "head_type",
    "relation_type",
    "tail_name",
    "tail_type",
    "standard_code",
    "page",
    "section",
    "evidence_text",
    "source_chunk_id",
    "source_sha256",
    "extraction_method",
    "confidence",
    "reviewer_1_decision",
    "reviewer_1_comment",
    "reviewer_2_decision",
    "reviewer_2_comment",
    "consensus_decision",
    "review_status",
]


def _annotation_row(relation: Relation) -> dict[str, object]:
    return {
        "record_id": relation.relation_id,
        "path_id": relation.path_id,
        "head_name": relation.head_name,
        "head_type": relation.head_type,
        "relation_type": relation.relation_type,
        "tail_name": relation.tail_name,
        "tail_type": relation.tail_type,
        "standard_code": relation.evidence.standard_code,
        "page": relation.evidence.page,
        "section": relation.evidence.section,
        "evidence_text": relation.evidence.evidence_text,
        "source_chunk_id": relation.evidence.chunk_id,
        "source_sha256": relation.evidence.source_sha256,
        "extraction_method": relation.extraction_method,
        "confidence": relation.confidence,
        "review_status": "pending",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eco-spec-kg")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus")
    corpus.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("ECOSPEC_CORPUS_DIR", ".")),
    )
    corpus.add_argument("--out", type=Path, required=True)

    annotate = sub.add_parser("annotate")
    annotate.add_argument("--chunks", type=Path, required=True)
    annotate.add_argument("--relations", type=Path)
    annotate.add_argument("--out", type=Path, required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--chunks", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)

    index = sub.add_parser("index")
    index.add_argument("--adapter", choices=["native", "microsoft"], required=True)
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--relations", type=Path)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--run", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--relations", type=Path, required=True)
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--smoke-limit", type=int)
    train.add_argument("--run", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--chunks", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)

    ablate = sub.add_parser("ablate")
    ablate.add_argument("--out", type=Path, required=True)

    report = sub.add_parser("report")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--data", type=Path, default=Path("data"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "corpus":
        print(json.dumps(build_corpus(args.source, args.out), ensure_ascii=False))
        return 0
    if args.command == "predict":
        chunks = load_chunks(args.chunks)
        result = RuleExtractor().extract(chunks)
        write_jsonl(args.out, (item.to_dict() for item in result.accepted))
        write_json(args.out.with_suffix(".rejected.json"), result.rejected)
        print(json.dumps({"accepted": len(result.accepted), "rejected": len(result.rejected)}))
        return 0
    if args.command == "annotate":
        chunks = load_chunks(args.chunks)
        relations = load_relations(args.relations)
        if not relations:
            relations = RuleExtractor().extract(chunks).accepted
        write_csv(
            args.out,
            [_annotation_row(item) for item in relations],
            ANNOTATION_FIELDS,
        )
        print(json.dumps({"records": len(relations)}))
        return 0
    if args.command == "index":
        chunks = load_chunks(args.chunks)
        relations = load_relations(args.relations)
        adapter = (
            NativeGraphRAGAdapter()
            if args.adapter == "native"
            else MicrosoftGraphRAGAdapter()
        )
        print(
            json.dumps(
                adapter.index(chunks, relations, args.out, run=args.run),
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "train":
        count = prepare_training_data(args.relations, args.prepared)
        if not args.run:
            write_json(
                args.out / "run_manifest.json",
                {"status": "prepared", "training_records": count},
            )
            print(json.dumps({"status": "prepared", "training_records": count}))
            return 0
        manifest = run_lora(
            args.prepared,
            args.out,
            LoRASettings(seed=args.seed),
            smoke_limit=args.smoke_limit,
        )
        print(json.dumps(manifest))
        return 0
    if args.command == "evaluate":
        result = evaluate_files(args.gold, args.predictions, args.chunks)
        write_json(args.out, result)
        print(json.dumps(result))
        return 0
    if args.command == "ablate":
        write_ablation_plan(args.out)
        return 0
    if args.command == "report":
        render_report(args.results, args.out)
        return 0
    if args.command == "serve":
        launch(args.data, args.host, args.port)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")

