from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from .models import Relation


@dataclass(slots=True)
class LoRASettings:
    model_name: str = "Qwen/Qwen3.5-9B"
    trainer: str = "transformers"
    precision: str = "bf16"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 2e-4
    max_epochs: int = 5
    early_stopping_patience: int = 2
    max_sequence_length: int = 4096
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 42
    data_seed: int = 42


SYSTEM_PROMPT = (
    "你是生态评估技术规范知识抽取模型。仅根据输入证据输出一个JSON关系，"
    "不得补充证据之外的信息。"
)


def relation_to_example(relation: Relation) -> dict[str, object]:
    prompt = (
        f"标准：{relation.evidence.standard_code}\n"
        f"章节：{relation.evidence.section}\n"
        f"证据：{relation.evidence.evidence_text}\n"
        "输出头实体、头类型、关系、尾实体、尾类型。"
    )
    completion = json.dumps(
        {
            "head_name": relation.head_name,
            "head_type": relation.head_type,
            "relation_type": relation.relation_type,
            "tail_name": relation.tail_name,
            "tail_type": relation.tail_type,
        },
        ensure_ascii=False,
    )
    return {
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
    }


def prepare_training_data(relations_path: Path, output_path: Path) -> int:
    relations = [Relation.from_dict(row) for row in read_jsonl(relations_path)]
    reviewed = [
        item
        for item in relations
        if item.review_status in {"accepted", "consensus_accepted"}
    ]
    write_jsonl(output_path, (relation_to_example(item) for item in reviewed))
    return len(reviewed)


def run_lora(
    training_path: Path,
    output_dir: Path,
    settings: LoRASettings,
    smoke_limit: int | None = None,
) -> dict[str, object]:
    if settings.trainer == "swift":
        return run_swift_lora(training_path, output_dir, settings, smoke_limit)
    if settings.trainer != "transformers":
        raise ValueError("trainer must be 'transformers' or 'swift'")
    if settings.precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be 'bf16' or 'fp16'")
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("install the 'ml' optional dependencies") from exc

    rows = read_jsonl(training_path)
    if smoke_limit:
        rows = rows[:smoke_limit]
    if len(rows) < 2:
        raise RuntimeError("at least two reviewed training records are required")

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    import torch

    dtype = torch.bfloat16 if settings.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name, torch_dtype=dtype
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    def format_row(row: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        }

    dataset = Dataset.from_list([format_row(row) for row in rows])
    dataset = dataset.train_test_split(test_size=max(1, len(rows) // 10), seed=settings.seed)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=settings.max_sequence_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.max_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=settings.per_device_eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=settings.seed,
        report_to=[],
        bf16=settings.precision == "bf16",
        fp16=settings.precision == "fp16",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=settings.early_stopping_patience
            )
        ],
    )
    trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    manifest = {
        "settings": asdict(settings),
        "training_records": len(rows),
        "smoke_limit": smoke_limit,
        "status": "complete",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_swift_command(
    training_path: Path,
    output_dir: Path,
    settings: LoRASettings,
) -> list[str]:
    if settings.precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be 'bf16' or 'fp16'")
    return [
        "swift",
        "sft",
        "--model",
        settings.model_name,
        "--dataset",
        str(training_path),
        "--tuner_type",
        "lora",
        "--torch_dtype",
        "bfloat16" if settings.precision == "bf16" else "float16",
        "--num_train_epochs",
        str(settings.max_epochs),
        "--per_device_train_batch_size",
        str(settings.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(settings.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(settings.gradient_accumulation_steps),
        "--learning_rate",
        str(settings.learning_rate),
        "--lora_rank",
        str(settings.rank),
        "--lora_alpha",
        str(settings.alpha),
        "--lora_dropout",
        str(settings.dropout),
        "--target_modules",
        "all-linear",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--add_non_thinking_prefix",
        "true",
        "--padding_free",
        "false",
        "--packing",
        "false",
        "--sequence_parallel_size",
        "1",
        "--split_dataset_ratio",
        "0.1",
        "--eval_strategy",
        "epoch",
        "--save_strategy",
        "epoch",
        "--load_best_model_at_end",
        "true",
        "--metric_for_best_model",
        "loss",
        "--greater_is_better",
        "false",
        "--early_stop_interval",
        str(settings.early_stopping_patience),
        "--warmup_ratio",
        "0.1",
        "--truncation_strategy",
        "delete",
        "--max_length",
        str(settings.max_sequence_length),
        "--seed",
        str(settings.seed),
        "--data_seed",
        str(settings.data_seed),
        "--save_total_limit",
        "2",
        "--output_dir",
        str(output_dir),
    ]


def run_swift_lora(
    training_path: Path,
    output_dir: Path,
    settings: LoRASettings,
    smoke_limit: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    rows = read_jsonl(training_path)
    if smoke_limit:
        rows = rows[:smoke_limit]
    if len(rows) < 2:
        raise RuntimeError("at least two reviewed training records are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_training_path = training_path
    if smoke_limit:
        effective_training_path = output_dir / "smoke_training.jsonl"
        write_jsonl(effective_training_path, rows)

    command = build_swift_command(effective_training_path, output_dir, settings)
    manifest: dict[str, object] = {
        "settings": asdict(settings),
        "input": {
            "source_path": str(training_path),
            "source_sha256": sha256_file(training_path),
            "effective_path": str(effective_training_path),
            "effective_sha256": sha256_file(effective_training_path),
        },
        "training_records": len(rows),
        "smoke_limit": smoke_limit,
        "backend": "ms-swift",
        "command": command,
        "status": "running",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    try:
        runner(command, check=True, text=True)
    except FileNotFoundError as exc:
        manifest["status"] = "failed"
        manifest["error"] = "swift command not found; install ms-swift"
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError(str(manifest["error"])) from exc
    except subprocess.CalledProcessError as exc:
        manifest["status"] = "failed"
        manifest["returncode"] = exc.returncode
        write_json(output_dir / "run_manifest.json", manifest)
        raise

    manifest["status"] = "complete"
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
