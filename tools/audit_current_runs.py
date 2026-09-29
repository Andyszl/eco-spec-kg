"""Audit existing development runs without inference, training or test-set reads."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    result = {r["unit_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate unit_id: {path}")
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), f"Output already exists: {args.out}")
    method = args.method_run.resolve()
    require(method.name.endswith("_method_only"), "Expected a _method_only run directory")
    roots = {
        "control": method.with_name(method.name.removesuffix("_method_only")),
        "method": method,
        "indicator": method.with_name(method.name + "_indicator_evidence"),
    }
    summaries, predictions, manifests, units_by_run = {}, {}, {}, {}
    target_ids = {"3194374504e7f58a", "78bd15d38602a8f7"}
    snapshots = {}
    for label, root in roots.items():
        manifest = read_json(root / "run_manifest.json")
        metrics = read_json(root / "evaluation/metrics.json")
        config = read_json(root / "resolved_config.json")
        units_path = Path(manifest["input"]["path"])
        require(units_path.name == "dev_units.jsonl" and units_path.parent.name == "blind",
                f"Only blind/dev_units.jsonl is allowed: {units_path}")
        gold_path = Path(metrics["inputs"]["gold_path"])
        require(gold_path.name == "dev_annotations.jsonl" and gold_path.parent.name == "gold",
                f"Only gold/dev_annotations.jsonl is allowed: {gold_path}")
        require(sha(units_path) == manifest["input"]["sha256"] == metrics["inputs"]["units_sha256"],
                f"Source hash mismatch: {label}")
        require(sha(gold_path) == metrics["inputs"]["gold_sha256"], f"Gold hash mismatch: {label}")
        for filename, field in (("predictions.jsonl", "predictions"), ("resolved_config.json", "config")):
            require(sha(root / filename) == manifest[field]["sha256"], f"Changed file: {root / filename}")
        validated = root / "validation/validated_predictions.jsonl"
        require(sha(validated) == metrics["inputs"]["predictions_sha256"], f"Evaluation prediction mismatch: {label}")
        validation_path = root / "validation/validation_report.json"
        require(read_json(validation_path).get("passed") is True, f"Validation not passed: {label}")
        require(sha(validation_path) == metrics["inputs"]["validation_report_sha256"],
                f"Changed validation report: {label}")
        units = read_rows(units_path)
        pred = read_rows(root / "predictions.jsonl")
        require(set(units) == set(pred), f"Unit coverage mismatch: {label}")
        require(all(r.get("status") == "success" for r in pred.values()), f"Failed prediction: {label}")
        predictions[label], manifests[label], units_by_run[label] = pred, manifest, units_path
        summaries[label] = {
            "run": str(root), "units_path": str(units_path),
            "units_sha256": sha(units_path), "gold_sha256": metrics["inputs"]["gold_sha256"],
            "dataset": metrics["dataset"],
            "config": {k: config.get(k) for k in (
                "backend", "model", "seed", "temperature", "max_tokens", "candidate_generator", "selection_policy")},
            "entity_f1": metrics["entities"]["micro"]["f1"],
            "relation_f1": metrics["relations"]["strict_micro"]["f1"],
            "macro_relation_f1": metrics["relations"]["macro_relation_f1"],
            "error_count": metrics["error_count"], "paths": metrics.get("paths"),
            "implementation": manifest.get("implementation", {}),
            "prompt_addition_sha256": sha(root / "prompt_addition.txt") if (root / "prompt_addition.txt").is_file() else None,
        }
        snapshots[label] = {
            "source_units": [r for k, r in units.items() if k in target_ids],
            "predictions": [r for k, r in pred.items() if k in target_ids],
        }
    require(len({s["units_sha256"] for s in summaries.values()}) == 1, "Runs use different source data")
    require(len({s["gold_sha256"] for s in summaries.values()}) == 1, "Runs use different evaluation gold")
    candidate_comparisons = {}
    for label in ("control", "indicator"):
        candidate_comparisons[label + "_vs_method"] = {
            "missing_hash_units": [uid for uid, r in predictions["method"].items()
                                   if not r.get("candidate_hash") or not predictions[label][uid].get("candidate_hash")],
            "different_hash_units": [uid for uid, r in predictions["method"].items()
                                     if r.get("candidate_hash") != predictions[label][uid].get("candidate_hash")],
        }
    package = units_by_run["method"].parent.parent
    package_manifest = read_json(package / "manifest.json")
    report = {
        "scope": "development audit; no test records read; no model invoked",
        "runs": summaries, "candidate_comparisons": candidate_comparisons,
        "package": {k: package_manifest.get(k) for k in (
            "package_id", "dataset_version", "gold_nature", "human_expert_review_required_for_publication", "unit_split_counts")},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, value in (("audit_report.json", report), ("target_units.json", snapshots)):
        (args.out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = {
        "CONTROL_RUN": roots["control"], "METHOD_RUN": roots["method"],
        "INDICATOR_RUN": roots["indicator"], "SOURCE_UNITS": units_by_run["method"],
        "DATA_PACKAGE": package, "AUDIT_DIR": args.out.resolve(),
    }
    (args.out / "session.env.sh").write_text(
        "# Generated from verified development run manifests. Source in Bash.\n" +
        "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in env.items()), encoding="utf-8")
    print("方案\t实体F1\t关系F1\t宏平均F1\t总错误")
    for label, s in summaries.items():
        print(label, s["entity_f1"], s["relation_f1"], s["macro_relation_f1"], s["error_count"], sep="\t")
    print(json.dumps(candidate_comparisons, ensure_ascii=False))
    print(f"Audit saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"Audit stopped: {exc}")
