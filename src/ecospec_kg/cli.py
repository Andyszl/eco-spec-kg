from __future__ import annotations

import argparse
import json
import os
import sys
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
from .extraction import LLMExtractor, RuleExtractor
from .io_utils import read_jsonl, write_csv, write_json, write_jsonl
from .models import DocumentChunk, Relation
from .providers import MockProvider, OpenAICompatibleProvider, TransformersProvider
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

    extract_llm = sub.add_parser("extract-llm")
    extract_llm.add_argument("--chunks", type=Path, required=True)
    extract_llm.add_argument("--out", type=Path, required=True)
    extract_llm.add_argument(
        "--provider",
        choices=["openai", "transformers", "mock"],
        default=os.environ.get("ECOSPEC_LLM_PROVIDER", "openai"),
    )
    extract_llm.add_argument(
        "--model",
        default=os.environ.get("ECOSPEC_COMPLETION_MODEL", "Qwen/Qwen3-0.6B"),
    )
    extract_llm.add_argument("--limit", type=int)
    extract_llm.add_argument("--start", type=int, default=0)
    extract_llm.add_argument("--standards", nargs="*")
    extract_llm.add_argument("--max-relations-per-chunk", type=int, default=20)
    extract_llm.add_argument("--retries", type=int, default=2)
    extract_llm.add_argument("--retry-sleep", type=float, default=1.0)
    extract_llm.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-chunk progress logs for LLM extraction.",
    )

    index = sub.add_parser("index")
    index.add_argument("--adapter", choices=["native", "microsoft"], required=True)
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--relations", type=Path)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--run", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--relations", type=Path)
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--smoke-limit", type=int)
    train.add_argument(
        "--model",
        default=os.environ.get("ECOSPEC_BASE_MODEL", "Qwen/Qwen3.5-9B"),
    )
    train.add_argument(
        "--trainer", choices=("transformers", "swift"), default="transformers"
    )
    train.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    train.add_argument("--max-length", type=int, default=4096)
    train.add_argument("--train-batch-size", type=int, default=1)
    train.add_argument("--eval-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--epochs", type=int, default=5)
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


def _make_provider(kind: str, model: str):
    if kind == "openai":
        provider = OpenAICompatibleProvider.from_env()
        provider.model = model
        return provider
    if kind == "transformers":
        return TransformersProvider(model)
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider: {kind}")


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "-" * width
    filled = min(width, max(0, round(width * done / total)))
    return "#" * filled + "-" * (width - filled)


def _llm_progress_logger(
    index: int,
    total: int,
    chunk: DocumentChunk,
    accepted_delta: int,
    rejected_delta: int,
    accepted_total: int,
    rejected_total: int,
) -> None:
    percent = (index / total * 100) if total else 100.0
    section = chunk.section or "-"
    print(
        "[{bar}] {index}/{total} {percent:5.1f}% "
        "chunk={chunk_id} standard={standard} page={page} section={section} "
        "+accepted={accepted_delta} +rejected={rejected_delta} "
        "accepted={accepted_total} rejected={rejected_total}".format(
            bar=_progress_bar(index, total),
            index=index,
            total=total,
            percent=percent,
            chunk_id=chunk.chunk_id,
            standard=chunk.standard_code,
            page=chunk.page,
            section=section,
            accepted_delta=accepted_delta,
            rejected_delta=rejected_delta,
            accepted_total=accepted_total,
            rejected_total=rejected_total,
        ),
        file=sys.stderr,
        flush=True,
    )


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
    if args.command == "extract-llm":
        chunks = load_chunks(args.chunks)
        if args.standards:
            wanted = set(args.standards)
            chunks = [chunk for chunk in chunks if chunk.standard_code in wanted]
        if args.start:
            chunks = chunks[args.start :]
        if args.limit:
            chunks = chunks[: args.limit]
        provider = _make_provider(args.provider, args.model)
        if not args.quiet:
            print(
                "extract-llm start: provider={provider} model={model} chunks={chunks} "
                "standards={standards} start={start} limit={limit} retries={retries} "
                "out={out}".format(
                    provider=args.provider,
                    model=args.model,
                    chunks=len(chunks),
                    standards=",".join(args.standards or ["ALL"]),
                    start=args.start,
                    limit=args.limit if args.limit is not None else "ALL",
                    retries=args.retries,
                    out=args.out,
                ),
                file=sys.stderr,
                flush=True,
            )
        result = LLMExtractor(
            provider,
            max_relations_per_chunk=args.max_relations_per_chunk,
            retries=args.retries,
            retry_sleep_seconds=args.retry_sleep,
        ).extract(
            chunks,
            on_progress=None if args.quiet else _llm_progress_logger,
        )
        write_jsonl(args.out, (item.to_dict() for item in result.accepted))
        write_json(args.out.with_suffix(".rejected.json"), result.rejected)
        print(
            json.dumps(
                {
                    "chunks": len(chunks),
                    "accepted": len(result.accepted),
                    "rejected": len(result.rejected),
                },
                ensure_ascii=False,
            )
        )
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
        if args.relations:
            count = prepare_training_data(args.relations, args.prepared)
        elif args.prepared.exists():
            count = len(read_jsonl(args.prepared))
        else:
            raise ValueError("--relations is required unless --prepared already exists")
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
            LoRASettings(
                model_name=args.model,
                trainer=args.trainer,
                precision=args.precision,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                learning_rate=args.learning_rate,
                max_epochs=args.epochs,
                max_sequence_length=args.max_length,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed,
            ),
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
