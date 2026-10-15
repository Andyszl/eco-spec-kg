"""Training-only semantic revisions, grouped validation, real-tokenizer gate and seed42 pair."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.io_utils import stable_id
from check_scope_inputs_20261013 import encode_files
import train_reviewed_pair_20261014 as previous

FOLDER = "experiments/semantic_pair_v211_20261015"
SEED_FOLDER = "experiments/semantic_supervision_v1_20261015"
HOLDOUT = "9fc3c8a70f70b39e"
PARENT = "3b87291bafd7da90"
ROLES = ("model_variable|obtained_by|method", "model_variable|sourced_from|data_source")


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def rows(p): return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p): return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
def write_rows(p, values): p.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8", newline="\n")
def key(r): return tuple(r[n] for n in ("head_name", "head_type", "relation_type", "tail_name", "tail_type"))
def uid(row): return json.loads(row["messages"][1]["content"])["source_unit"]["unit_id"]


def project(unit, annotation):
    c = ex.RuleCandidateExtractorV2().predict_unit(unit)
    edges = {key(r) for r in annotation["relations"]}
    entities = {(e["name"], e["entity_type"]) for e in annotation["entities"]}
    selected = [r for r in c["relations"] if key(r) in edges]
    endpoints = {r[s + "_id"] for r in selected for s in ("head", "tail")}
    target = {"selected_entity_ids": sorted(e["entity_id"] for e in c["entities"]
              if (e["name"], e["entity_type"]) in entities and e["entity_id"] not in endpoints),
              "selected_relation_ids": sorted(r["relation_id"] for r in selected)}
    system, prompt = ex.build_llm_selection_messages(unit, c)
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt},
                          {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)}]}


def coverage(values):
    counts = Counter()
    parents = {role: set() for role in ROLES}
    for row in values:
        q, target = json.loads(row["messages"][1]["content"]), json.loads(row["messages"][-1]["content"])
        es = {e["id"]: e for e in q["candidate_entities"]}
        for r in q["candidate_relations"]:
            role = "|".join((es[r["head"]]["type"], r["type"], es[r["tail"]]["type"]))
            if role in ROLES and r["id"] in target["selected_relation_ids"]:
                counts[role] += 1
                ident = q["source_unit"]["unit_id"]
                parents[role].add(PARENT if ident in CHILDREN else ident)
    return {role: {"positive_edges": counts[role], "positive_parent_count": len(parents[role]),
                   "positive_parent_ids": sorted(parents[role])} for role in ROLES}


CHILDREN = {"6f2c40ba174a9b41", "42d8dd466182a9ea", "44bfa21cdeba8677"}


def revise(unit, original, head_name, relation_type, tail_name, tail_type, spans, quote, rationale):
    annotation = deepcopy(original)
    span_map = ex._all_spans(unit)
    assert spans and set(spans) <= set(span_map), "Revision evidence outside training parent"
    compact = lambda s: "".join(s.split())
    assert compact(quote) in compact(" ".join(span_map[s].get("text", "") for s in spans)), "Review quote absent"
    candidates = ex.RuleCandidateExtractorV2().predict_unit(unit)
    candidate = next(r for r in candidates["relations"] if
        (r["head_name"], r["relation_type"], r["tail_name"], r["tail_type"]) == (head_name, relation_type, tail_name, tail_type))
    assert key(candidate) not in {key(r) for r in annotation["relations"]}
    by_key = {(e["name"], e["entity_type"]): e for e in annotation["entities"]}
    for side in ("head", "tail"):
        ek = (candidate[side + "_name"], candidate[side + "_type"])
        if ek not in by_key:
            e = {"entity_id": stable_id("assistant-source-entity-20261015", unit["unit_id"], *ek),
                 "name": ek[0], "entity_type": ek[1], "evidence_span_ids": list(spans)}
            annotation["entities"].append(e)
            by_key[ek] = e
    relation = {n: candidate[n] for n in ("head_name", "head_type", "relation_type", "tail_name", "tail_type")}
    relation.update(head_id=by_key[(relation["head_name"], relation["head_type"])]["entity_id"],
                    tail_id=by_key[(relation["tail_name"], relation["tail_type"])]["entity_id"], evidence_span_ids=list(spans))
    annotation["relations"].append(relation)
    annotation.update(annotation_version="assistant-semantic-revision-v2-20261015", annotator_id="assistant_source_review",
                      review_status="assistant_reviewed_not_independent_human_gold")
    return annotation, {"unit_id": unit["unit_id"], "added_relation": relation, "quote": quote, "rationale": rationale,
                        "source_standard": unit["provenance"]["standard_code"],
                        "source_pdf_sha256": unit["provenance"]["source_sha256"]}


def build_delivery(root, destination):
    assert not destination.exists(), "Delivery exists; preserve original files"
    assert ex.CANDIDATE_GENERATOR_VERSION == "structure-aware-rule-v2.11"
    data, old = root / "deliveries/human_confirmed_20261002", root / "experiments/candidate_scope_v29_20261013"
    old_manifest = read(old / "input_manifest.json")
    assert all(sha(data / n) == h for n, h in old_manifest["frozen_sha256"].items())
    assert all(sha(old / n) == h for n, h in old_manifest["files"].items())
    seed = root / SEED_FOLDER
    long_manifest = read(old / "long_views.manifest.json")
    assert sha(seed / "long_source_units.jsonl") == long_manifest["source_units_sha256"], "Long source hash changed"
    assert sha(seed / "long_annotations.jsonl") == long_manifest["annotations_sha256"], "Long annotation hash changed"
    units = {u["unit_id"]: u for u in rows(data / "blind/train_units.jsonl")}
    originals = {g["unit_id"]: g for g in rows(data / "gold/train_annotations.jsonl")}
    for u in rows(seed / "long_source_units.jsonl"): units[u["unit_id"]] = u
    for g in rows(seed / "long_annotations.jsonl"): originals[g["unit_id"]] = g
    old_ids = [uid(r) for r in rows(old / "repair_train.jsonl")]
    validation_ids = [uid(r) for r in rows(old / "repair_validation.jsonl")]
    assert HOLDOUT in old_ids and HOLDOUT not in validation_ids
    train_ids = [i for i in old_ids if i != HOLDOUT]
    validation_ids += [HOLDOUT]
    assert len(train_ids) == 625 and len(validation_ids) == 70
    assert not set(train_ids) & set(validation_ids)
    assert PARENT not in train_ids and CHILDREN <= set(train_ids)
    holdout_lines = {line for s in ex._all_spans(units[HOLDOUT]).values() for line in s.get("line_ids", [])}
    assert holdout_lines and all(not holdout_lines & {line for s in ex._all_spans(units[i]).values() for line in s.get("line_ids", [])} for i in train_ids), "Held-out source lines overlap training"
    amended, audit = deepcopy(originals), []
    cases = [
        ("113680dd91edf495", "K_EPIC", "obtained_by", "侵蚀-生产力评价模型", "method",
         ["0cb53d9db64b10df"], "采用侵蚀-生产力评价模型计算的土壤可蚀性因子", "变量定义直接命名获取模型；K与K_EPIC不合并。"),
        ("e10dfb07c04c0214", "Q_sr", "obtained_by", "修正土壤流失方程（RUSLE）", "method",
         ["f1c05e793826b95c", "c241caa674dbaae6"], "基于修正土壤流失方程（RUSLE）计算。", "段落明示模型，公式主输出Q_sr与土壤保持量定义绑定；不推及所有中间量。"),
        ("6f2c40ba174a9b41", "SR", "obtained_by", "修正风蚀方程", "method",
         ["cbfe31a12d3f456e", "aa32a74ef98a3e3c"], "采用修正风蚀方程 RWEQ 进行评价。", "同一父来源明示采用模型评价防风固沙量，SR为该量；三个子视图仍归同一train父来源。"),
        ("5954b773ec29497b", "P_i,j,k", "sourced_from", "降雨量资料", "data_source",
         ["99857f4808ce5903", "9dc71160c9931b63", "08f0c8f4b7ade06f"], "所用降雨资料的年份", "年份i对应所用降雨资料，P_i,j,k明示该年第半月逐日降雨量；区别实测输入与推导的R，保留冻结R来源口径。"),
        (HOLDOUT, "FVC", "sourced_from", "遥感图像", "data_source",
         ["bf8dda7ae0ea7747", "e6c7b1d19301c870"], "计算获得的植被覆盖所占像元比例即为该像元的植被覆盖度", "原文完整声明遥感图像像元分解计算流程，FVC是该流程输出；这是明确流程来源，不据此补NDVI土壤/植被端元的直接来源。"),
    ]
    for ident, *arguments in cases:
        amended[ident], decision = revise(units[ident], amended[ident], *arguments)
        audit.append(decision)
    ordered = train_ids + validation_ids
    prepared = {"control_train": [project(units[i], originals[i]) for i in train_ids],
                "repair_train": [project(units[i], amended[i]) for i in train_ids],
                "validation": [project(units[i], amended[i]) for i in validation_ids]}
    changed = sum(a != b for a, b in zip(prepared["control_train"], prepared["repair_train"]))
    assert changed == 4
    assert all(a["messages"][:2] == b["messages"][:2] for a, b in zip(prepared["control_train"], prepared["repair_train"]))
    supervised = {n: coverage(r) for n, r in prepared.items()}
    assert all(supervised["repair_train"][r]["positive_edges"] >= 3 and supervised["validation"][r]["positive_edges"] >= 1 for r in ROLES)
    assert all(set(supervised["repair_train"][r]["positive_parent_ids"]).isdisjoint(supervised["validation"][r]["positive_parent_ids"]) for r in ROLES)
    destination.mkdir(parents=True)
    for n, values in prepared.items(): write_rows(destination / f"{n}.jsonl", values)
    write_rows(destination / "units.jsonl", [units[i] for i in ordered])
    write_rows(destination / "control_annotations.jsonl", [originals[i] for i in train_ids] + [amended[i] for i in validation_ids])
    write_rows(destination / "repair_annotations.jsonl", [amended[i] for i in ordered])
    source_names = ["src/ecospec_kg/" + n for n in ("extractor_v2.py", "formula_symbols.py", "ontology_v2.py", "prediction_contract_v2.py", "selection_decoding_v2.py", "structural_selection_v2.py")]
    source_names += ["tools/semantic_pair_20261015.py", "tools/check_scope_inputs_20261013.py", "tools/train_reviewed_pair_20261014.py"]
    manifest = {"version": "semantic-pair-v211-20261015", "candidate_generator": ex.CANDIDATE_GENERATOR_VERSION,
        "selection_policy": ex.SELECTION_POLICY_VERSION, "dev_content_used_for_build": False,
        "counts": {n: len(v) for n, v in prepared.items()}, "unit_ids": {"control_train": train_ids, "repair_train": train_ids, "validation": validation_ids},
        "training_parent_ids": sorted(set(train_ids) - CHILDREN | {PARENT}), "held_out_parent_ids": [HOLDOUT],
        "frozen_sha256": old_manifest["frozen_sha256"], "tokenizer_config_sha256": old_manifest["tokenizer_config_sha256"],
        "source_sha256_lf": {n: lfsha(root / n) for n in source_names}, "old_input_manifest_sha256": sha(old / "input_manifest.json"),
        "files": {p.name: sha(p) for p in destination.iterdir() if p.is_file()}, "annotation_revisions": audit,
        "changed_training_targets": changed, "added_training_positive_relations": 4, "supervision": supervised,
        "validation_policy": {"original_internal_validation_preserved": 69, "additional_held_out_training_parents": [HOLDOUT],
            "both_arms_exclude_entire_parent": True, "independent_final_test": False,
            "heldout_source_line_count": len(holdout_lines), "source_line_overlap_with_train": 0,
            "checkpoint_selection": "minimum loss on the same fixed 70 validation records; no dev-based checkpoint choice",
            "scope": "training-source development validation; previously used source, not a new independent test"},
        "paired_protocol": {"seed": 42, "max_steps_per_arm": 200, "same_budget": True, "validation_shared": True,
            "effect": "reviewed supervision additions; both arms share new candidates, evidence and long views",
            "full_dev_gate": ["three bindings after common structure", "entity and relation F1/macro nondecrease", "FP nonincrease", "at least one complete expanded method/source path"],
            "automatic_three_seed_expansion": False},
        "limitations": ["Only one model-variable source training parent and one held-out source parent; coverage is a small pilot, not broad generalization.",
                        "Assistant source review is not new independent human annotation.",
                        "Unresolved endmember sources and method intermediates are not promoted to positives."],
        "training_ready": False, "requires_real_tokenizer_preflight": True}
    save(destination / "input_manifest.json", manifest)
    return manifest


def verify_delivery(root, folder=None):
    folder = folder or root / FOLDER
    manifest = read(folder / "input_manifest.json")
    for n, h in manifest["files"].items():
        assert Path(n).name == n and sha(folder / n) == h, "Input hash changed: " + n
    for n, h in manifest["source_sha256_lf"].items(): assert lfsha(root / n) == h, "Source hash changed: " + n
    for n, h in manifest["frozen_sha256"].items(): assert sha(root / "deliveries/human_confirmed_20261002" / n) == h, "Frozen hash changed: " + n
    units = {u["unit_id"]: u for u in rows(folder / "units.jsonl")}
    for arm in ("control", "repair"):
        labels = {g["unit_id"]: g for g in rows(folder / f"{arm}_annotations.jsonl")}
        for name in (arm + "_train", "validation"):
            values = rows(folder / f"{name}.jsonl")
            assert [uid(r) for r in values] == manifest["unit_ids"][name]
            assert values == [project(units[uid(r)], labels[uid(r)]) for r in values], "Target projection changed"
    train, validation = manifest["unit_ids"]["repair_train"], manifest["unit_ids"]["validation"]
    assert len(train) == len(set(train)) == 625 and len(validation) == len(set(validation)) == 70
    assert not set(train) & set(validation) and HOLDOUT in validation and HOLDOUT not in manifest["training_parent_ids"]
    for name in ("control_train", "repair_train", "validation"):
        assert coverage(rows(folder / f"{name}.jsonl")) == manifest["supervision"][name]
    return {"status": "input_audit_passed", "manifest_sha256": sha(folder / "input_manifest.json"), "manifest": manifest}


def verify_preflight(root, folder, report):
    assert report.get("passed") and report.get("real_tokenizer") is True, "Real passing tokenizer preflight required"
    verification = verify_delivery(root, folder)
    assert report["input_manifest_sha256"] == verification["manifest_sha256"], "Wrong preflight inputs"
    assert report["input_manifest"] == verification["manifest"] and report["source_sha256_lf"] == verification["manifest"]["source_sha256_lf"]
    assert report["swift_version"] == "4.4.2"
    assert report["model_weights_loaded"] is False and report["model_training_started"] is False
    assert report["original_inputs_unchanged"] and not report["failures"] and report["checked_records"] == 1320
    for name in ("control_train", "repair_train", "validation"):
        actual = [r for r in report["encodings"] if r["file"] == name]
        assert [r["unit_id"] for r in actual] == verification["manifest"]["unit_ids"][name]
        assert all(0 < r["supervised_tokens"] <= r["tokens"] <= 8192 for r in actual)
    return verification


def preflight(root, model, out):
    assert not out.exists(), "Preflight output exists"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight must hide GPUs"
    checked = verify_delivery(root)
    assert importlib.metadata.version("ms-swift") == "4.4.2"
    for n, h in checked["manifest"]["tokenizer_config_sha256"].items(): assert sha(model / n) == h, "Tokenizer changed: " + n
    from swift.model import get_model_processor
    from swift.template import get_template
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None, "Preflight must not load weights"
    template = get_template(processor, max_length=8192, truncation_strategy="raise", add_non_thinking_prefix=True, loss_scale="default+ignore_empty_think")
    assert template.template_meta.template_type == "qwen3_5"
    template.set_mode("train")
    files = {n: rows(root / FOLDER / f"{n}.jsonl") for n in ("control_train", "repair_train", "validation")}
    report = encode_files(template, files)
    assert report["checked_records"] == 1320
    assert verify_delivery(root) == checked, "Inputs changed during real encoding"
    report.update(real_tokenizer=True, model_weights_loaded=False, model_training_started=False,
        original_inputs_unchanged=True, input_manifest_sha256=checked["manifest_sha256"],
        source_sha256_lf=checked["manifest"]["source_sha256_lf"], input_manifest=checked["manifest"],
        swift_version="4.4.2", code_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        max_tokens_by_file={n: max((r["tokens"] for r in report["encodings"] if r["file"] == n), default=0) for n in files},
        training_ready=False, next_step="Review received real preflight before the guarded seed42 paired launch")
    out.mkdir(parents=True)
    save(out / "preflight_report.json", report)
    print(json.dumps({"PREFLIGHT_PASSED": report["passed"], "CHECKED_RECORDS": report["checked_records"], "MAX_TOKENS": report["max_tokens_by_file"], "TRAINING_STARTED": False}, ensure_ascii=False), flush=True)
    return report


def build_command(root, model, output, arm):
    cmd = previous.build_command(root, model, output, arm)
    for option, value in (("--dataset", root / FOLDER / f"{arm}_train.jsonl"), ("--val_dataset", root / FOLDER / "validation.jsonl")):
        cmd[cmd.index(option) + 1] = str(value)
    return cmd


def train(root, model, out, report_path, run=False, launch=previous.stream_launch):
    verification = verify_preflight(root, root / FOLDER, read(report_path))
    assert not out.exists(), "Training output exists"
    assert importlib.metadata.version("ms-swift") == "4.4.2"
    for n, h in verification["manifest"]["tokenizer_config_sha256"].items(): assert sha(model / n) == h
    from dataclasses import fields
    from swift.arguments import SftArguments
    supported = {f.name for f in fields(SftArguments)}
    for arm in ("control", "repair"):
        cmd = build_command(root, model, out / arm, arm)
        assert all(n[2:] in supported for n in cmd[2::2])
    from swift.dataset import load_dataset
    for name, count in verification["manifest"]["counts"].items():
        dataset, validation = load_dataset([str(root / FOLDER / f"{name}.jsonl")], split_dataset_ratio=0, seed=42, shuffle=False, strict=True)
        assert len(dataset) == count and validation is None
    if not run:
        return {"status": "dry_run_verified", "training_started": False, "commands": [build_command(root, model, out / a, a) for a in ("control", "repair")]}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("WORLD_SIZE", "1") == "1"
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), "GPU0 is occupied or unavailable"
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out.mkdir(parents=True)
    result = {"status": "running", "seed": 42, "max_steps_per_arm": 200, "arms": {}, "verification": verification,
              "preflight_sha256": sha(report_path), "expansion_to_three_seeds": False, "dev_evaluation_completed": False}
    try:
        for arm in ("control", "repair"):
            command = build_command(root, model, out / arm, arm)
            result["arms"][arm] = {"status": "running", "command": command, "output": str(out / arm)}
            save(out / "pair_manifest.json", result)
            rc = launch(command, out / f"{arm}.log")
            assert rc == 0, f"{arm} failed; subsequent arm not started, code={rc}"
            result["arms"][arm].update(status="complete", completion=previous.inspect_completion(out / arm))
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(out / "pair_manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify", "preflight", "train"))
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.action == "build": result = build_delivery(ROOT, args.out or ROOT / FOLDER)
    elif args.action == "verify": result = verify_delivery(ROOT)
    elif args.action == "preflight":
        assert args.model and args.out
        result = preflight(ROOT, args.model, args.out)
    else:
        assert args.model and args.out and args.preflight
        result = train(ROOT, args.model, args.out, args.preflight, args.run)
    print(json.dumps({k: result[k] for k in ("status", "counts", "supervision", "passed", "checked_records", "training_started") if k in result}, ensure_ascii=False, indent=2))
    if args.action == "preflight" and not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
