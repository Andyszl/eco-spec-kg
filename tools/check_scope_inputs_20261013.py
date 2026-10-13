"""Check saved paired draft inputs using the real CPU tokenizer; never train."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_id(row):
    users = [m for m in row["messages"] if m["role"] == "user"]
    assert len(users) == 1
    return json.loads(users[0]["content"])["source_unit"]["unit_id"]


def encode_files(template, files):
    encodings, failures = [], []
    for name, values in files.items():
        for row in values:
            identity = {"file": name, "unit_id": unit_id(row)}
            try:
                encoded = template.encode(copy.deepcopy(row))
                length = len(encoded["input_ids"])
                labels = encoded["labels"]
                supervised = sum(token != -100 for token in labels)
                assert 0 < length <= 8192 and len(labels) == length and supervised > 0, "Invalid length or empty supervision"
                encodings.append({**identity, "tokens": length, "supervised_tokens": supervised})
            except Exception as exc:
                failures.append({**identity, "error_type": type(exc).__name__, "error": str(exc)})
            count = len(encodings) + len(failures)
            if count % 100 == 0:
                print(f"ENCODED_OR_RECORDED_FAILURE={count}", flush=True)
    return {"passed": not failures, "checked_records": len(encodings) + len(failures), "encodings": encodings, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-prep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Output already exists"
    root = Path(__file__).resolve().parents[1]
    folder = root / "experiments/candidate_scope_v29_20261013"
    manifest = read(folder / "input_manifest.json")
    for name, expected in manifest["files"].items():
        assert Path(name).name == name and sha(folder / name) == expected, f"Input hash mismatch: {name}"
    for name, expected in manifest["source_sha256_lf"].items():
        assert hashlib.sha256((root / "src/ecospec_kg" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, f"Source changed: {name}"
    pf_path = args.original_prep / "training_preflight_20261012_113739.json"
    assert sha(pf_path) == manifest["preflight_sha256"], "Saved original split changed"
    pf = read(pf_path)
    for name, expected in manifest["tokenizer_config_sha256"].items():
        assert Path(name).name == name and sha(args.model / name) == expected, f"Tokenizer changed: {name}"
    assert importlib.metadata.version("ms-swift") == "4.4.2", "Expected original Swift version"
    names = ["control_train", "control_validation", "repair_train", "repair_validation"]
    inputs = {name: rows(folder / f"{name}.jsonl") for name in names}
    ids = {name: [unit_id(row) for row in values] for name, values in inputs.items()}
    parent, children = manifest["parent_unit_id"], manifest["child_unit_ids"]
    assert ids["control_train"] == [i for i in pf["split_unit_ids"]["train"] if i != parent]
    assert ids["repair_train"] == [child for i in pf["split_unit_ids"]["train"] for child in (children if i == parent else [i])]
    assert ids["control_validation"] == ids["repair_validation"] == pf["split_unit_ids"]["validation"]
    assert all(len(value) == len(set(value)) for value in ids.values())
    assert set(ids["repair_train"]).isdisjoint(ids["repair_validation"])
    original_path = args.original_prep / "qwen35_train.jsonl"
    assert sha(original_path) == "a4c71472c5f19d002015cdbdcaab33471fea4e7907bfb1e7a77a0e069d01a22c"
    originals = {unit_id(row): row for row in rows(original_path)}
    for name in ("control_train", "control_validation"):
        assert inputs[name] == [originals[i] for i in ids[name]], "Control content changed"
    protected = [folder / f"{n}.jsonl" for n in names] + [pf_path, original_path]
    before = {p: sha(p) for p in protected}
    from swift.model import get_model_processor
    from swift.template import get_template
    unloaded, processor = get_model_processor(str(args.model), load_model=False, download_model=False)
    assert unloaded is None, "Model weights must not load"
    template = get_template(processor, max_length=8192, truncation_strategy="raise",
                            add_non_thinking_prefix=True, loss_scale="default+ignore_empty_think")
    assert template.template_meta.template_type == "qwen3_5"
    template.set_mode("train")
    report = encode_files(template, inputs)
    assert report["checked_records"] == 1387
    assert {p: sha(p) for p in protected} == before, "Input changed during check"
    report.update(model_weights_loaded=False, model_training_started=False, ready_for_training=False,
                  original_inputs_unchanged=True, input_manifest=manifest,
                  code_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                  max_tokens_by_file={n: max((r["tokens"] for r in report["encodings"] if r["file"] == n), default=0) for n in names},
                  remaining_gate="Resolve residual supervision coverage and review explicit paired launcher before training")
    args.output.mkdir(parents=True)
    (args.output / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "coverage_audit.json").write_bytes((folder / "coverage_audit.json").read_bytes())
    print(json.dumps({"output": str(args.output), "length_check_passed": report["passed"],
                      "checked_records": report["checked_records"], "failures": report["failures"],
                      "max_tokens": report["max_tokens_by_file"], "ready_for_training": False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
