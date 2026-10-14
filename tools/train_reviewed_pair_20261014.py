"""Launch exactly one reviewed seed42 control/repair pair, with fixed update budgets."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

DATA_DIR = "experiments/candidate_scope_v29_20261013"
REVIEW_NAME = "supervision_review_v1_20261014.json"
REVIEW_SHA = "52f11de69bdc2415f4920cd7295292b2c3536b31edb9cc460653e3d805ef3444"
BASELINE_CODE = "a4bcf6816849223e4238999cf6903367a731d8af"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unit_id(row):
    return json.loads(row["messages"][1]["content"])["source_unit"]["unit_id"]


def verify_delivery(root, report_path):
    folder = root / DATA_DIR
    review_path = folder / REVIEW_NAME
    assert sha(review_path) == REVIEW_SHA, "Supervision review changed"
    review, manifest = read(review_path), read(folder / "input_manifest.json")
    assert review["pilot_allowed"] and review["real_preflight_passed"]
    assert not review["frozen_annotations_changed"] and not review["training_input_files_changed"]
    assert review["label_revisions"] == []
    assert review["input_files_sha256"] == manifest["files"]
    assert review["source_sha256_lf"] == manifest["source_sha256_lf"]
    assert sha(report_path) == review["preflight_report_sha256"], "Wrong or changed preflight report"
    report = read(report_path)
    assert report["input_manifest"] == manifest and report["code_commit"] == BASELINE_CODE
    assert report["passed"] and report["checked_records"] == 1387 and not report["failures"]
    assert report["original_inputs_unchanged"]
    for name, expected in manifest["files"].items():
        assert Path(name).name == name and sha(folder / name) == expected, f"Changed input: {name}"
    for name, expected in manifest["source_sha256_lf"].items():
        assert hashlib.sha256((root / "src/ecospec_kg" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, f"Changed source: {name}"
    for name, expected in manifest["frozen_sha256"].items():
        assert sha(root / "deliveries/human_confirmed_20261002" / name) == expected, f"Changed frozen labels/source: {name}"
    identities = {}
    for name, expected_count in (("control_train", 623), ("repair_train", 626), ("control_validation", 69), ("repair_validation", 69)):
        records = [json.loads(s) for s in (folder / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [unit_id(r) for r in records]
        actual = [r for r in report["encodings"] if r["file"] == name]
        assert len(ids) == len(set(ids)) == len(actual) == expected_count
        assert ids == [r["unit_id"] for r in actual]
        assert all(0 < r["supervised_tokens"] <= r["tokens"] <= 8192 for r in actual)
        identities[name] = ids
    assert identities["control_validation"] == identities["repair_validation"]
    for arm in ("control", "repair"):
        assert set(identities[f"{arm}_train"]).isdisjoint(identities[f"{arm}_validation"])
    assert manifest["parent_unit_id"] not in identities["repair_train"]
    assert set(identities["repair_train"]) - set(identities["control_train"]) == set(manifest["child_unit_ids"])
    return {"status": "ready_for_reviewed_seed42_pilot", "review": review, "input_manifest": manifest,
            "preflight_report": str(report_path), "preflight_sha256": sha(report_path),
            "counts": manifest["counts"]}


def build_command(root, model, output, arm):
    assert arm in {"control", "repair"}
    folder = root / DATA_DIR
    options = {
        "model": str(model), "dataset": str(folder / f"{arm}_train.jsonl"),
        "val_dataset": str(folder / f"{arm}_validation.jsonl"), "split_dataset_ratio": "0",
        "tuner_type": "lora", "torch_dtype": "bfloat16", "max_steps": "200",
        "per_device_train_batch_size": "1", "per_device_eval_batch_size": "1",
        "gradient_accumulation_steps": "16", "learning_rate": "0.0002",
        "lora_rank": "8", "lora_alpha": "16", "lora_dropout": "0.05",
        "target_modules": "all-linear", "freeze_vit": "true", "freeze_aligner": "true",
        "add_non_thinking_prefix": "true", "loss_scale": "default+ignore_empty_think",
        "padding_free": "false", "packing": "false", "sequence_parallel_size": "1",
        "dataset_shuffle": "false", "val_dataset_shuffle": "false", "train_dataloader_shuffle": "true",
        "eval_strategy": "steps", "save_strategy": "steps", "eval_steps": "40", "save_steps": "40",
        "load_best_model_at_end": "true", "metric_for_best_model": "loss", "greater_is_better": "false",
        "early_stop_interval": "0", "warmup_ratio": "0.1", "truncation_strategy": "delete",
        "strict": "true", "max_length": "8192", "seed": "42", "data_seed": "42",
        "save_total_limit": "2", "logging_steps": "5", "report_to": "none",
        "add_version": "false", "output_dir": str(output),
    }
    return ["swift", "sft", *[part for k, v in options.items() for part in ("--" + k, v)]]


def inspect_completion(output):
    states = [p for p in output.glob("checkpoint-*/trainer_state.json") if read(p).get("global_step") == 200]
    assert len(states) == 1, "No unique completed 200-step trainer state"
    state = read(states[0])
    best = Path(state["best_model_checkpoint"]).resolve()
    assert best.is_relative_to(output.resolve()) and best.is_dir(), "Best checkpoint outside this new run"
    weights, config = best / "adapter_model.safetensors", best / "adapter_config.json"
    assert weights.is_file() and weights.stat().st_size > 0 and config.is_file(), "Incomplete selected adapter"
    cfg = read(config)
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 16 and cfg["lora_dropout"] == 0.05
    assert math.isfinite(float(state["best_metric"]))
    evaluations = [v for v in state["log_history"] if "eval_loss" in v]
    assert evaluations and all(math.isfinite(float(v["eval_loss"])) for v in evaluations)
    return {"global_step": 200, "best_checkpoint": str(best), "internal_validation_loss": state["best_metric"],
            "adapter_bytes": weights.stat().st_size, "adapter_sha256": sha(weights), "trainer_state_sha256": sha(states[0])}


def stream_launch(command, log):
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def execute_pair(root, model, destination, verification, launch=stream_launch):
    assert not destination.exists(), "Refuse to overwrite previous run"
    destination.mkdir(parents=True)
    result = {"status": "running", "seed": 42, "max_steps_per_arm": 200, "arms": {},
              "verification": verification, "expansion_to_three_seeds": False, "dev_evaluation_completed": False}
    save(destination / "pair_manifest.json", result)
    try:
        for arm in ("control", "repair"):
            output = destination / arm
            assert not output.exists()
            command = build_command(root, model, output, arm)
            result["arms"][arm] = {"status": "running", "command": command, "output": str(output)}
            save(destination / "pair_manifest.json", result)
            print(f"STARTING_ARM={arm} SEED=42 MAX_STEPS=200", flush=True)
            rc = launch(command, destination / f"{arm}.log")
            if rc:
                raise RuntimeError(f"{arm} training failed with exit code {rc}; subsequent arm not started")
            completion = inspect_completion(output)
            result["arms"][arm].update(status="complete", completion=completion)
            save(destination / "pair_manifest.json", result)
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(destination / "pair_manifest.json", result)
        raise
    save(destination / "pair_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verification = verify_delivery(root, args.preflight)
    assert not args.out.exists(), "Output already exists"
    assert importlib.metadata.version("ms-swift") == "4.4.2", "Expected audited ms-swift 4.4.2"
    for name, expected in verification["input_manifest"]["tokenizer_config_sha256"].items():
        assert Path(name).name == name and sha(args.model / name) == expected, f"Tokenizer changed: {name}"
    # Validate flags against the installed parser without instantiating a model or trainer.
    from dataclasses import fields
    from swift.arguments import SftArguments
    supported = {f.name for f in fields(SftArguments)}
    for arm in ("control", "repair"):
        command = build_command(root, args.model, args.out / arm, arm)
        assert all(name[2:] in supported for name in command[2::2]), "Installed Swift parser missing audited options"
    # Load only the local datasets. Prove explicit train/validation counts stay intact.
    from swift.dataset import load_dataset
    for name, count in (("control_train", 623), ("repair_train", 626), ("control_validation", 69), ("repair_validation", 69)):
        dataset, validation = load_dataset([str(root / DATA_DIR / f"{name}.jsonl")], split_dataset_ratio=0,
                                           seed=42, shuffle=False, strict=True)
        assert len(dataset) == count and validation is None, "Swift changed explicit partition counts"
    verification.update(installed_swift_flags_checked=True, swift_loaded_counts_verified=True,
                        code_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip())
    if not args.run:
        print(json.dumps({"status": "dry_run_verified", "training_started": False,
                          "commands": [build_command(root, args.model, args.out / a, a) for a in ("control", "repair")]}, ensure_ascii=False, indent=2))
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Set CUDA_VISIBLE_DEVICES=0 explicitly"
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "This pilot requires single-process training"
    assert shutil.which("swift"), "swift command missing in active conda environment"
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    assert gpu.returncode == 0, f"GPU query failed: {gpu.stderr.strip()}"
    busy = [pid.strip() for pid in gpu.stdout.splitlines() if pid.strip() and pid.strip() != str(os.getpid())]
    assert not busy, f"GPU 0 occupied by processes: {', '.join(busy)}; stop the relevant service before rerunning"
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 unavailable"
    check = torch.ones(1, device="cuda", dtype=torch.bfloat16)
    assert (check + check).item() == 2
    del check
    torch.cuda.empty_cache()
    result = execute_pair(root, args.model, args.out, verification)
    print(f"PAIRED_TRAINING_OK={args.out}", flush=True)
    for arm, value in result["arms"].items():
        print(f"{arm.upper()}_BEST_CHECKPOINT={value['completion']['best_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
