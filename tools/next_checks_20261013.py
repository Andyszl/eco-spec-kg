"""Read-only training coverage and CPU tokenizer checks, then archive evidence.

No training input is replaced and no model weights are loaded.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tarfile
import time

HEAD = "89d4258b18451ce80cf63182f502002635996a20"
TRAIN_SHA = "a4c71472c5f19d002015cdbdcaab33471fea4e7907bfb1e7a77a0e069d01a22c"
PARENT = "3b87291bafd7da90"
SUMMARY_NAME = "structural_offline_20261012_175956"
LABELS = ["base", "old42", "old43", "old44", "new42", "new43", "new44"]
SOURCE = {
    "src/ecospec_kg/extractor_v2.py": "761c1839756e1c272776c6701f2669fe9e657ba789a98a3f81dea3879fab7df2",
    "src/ecospec_kg/selection_decoding_v2.py": "e899bf628713a2e8e42cac0106a0dc739ec7541415ceef5a7f1921192da14a73",
    "src/ecospec_kg/formula_symbols.py": "1d7678df8177d6e0becae0b4ebe6ca9c99af8fcd07cf5d96e05bb2f46c630a38",
    "src/ecospec_kg/evaluation_v2.py": "5d1564f358e6e325c3a5531c9147874de597da27ee3ef1c2ac98570016edaa8a",
    "src/ecospec_kg/experiment_io_v2.py": "dbb7a20c4b5680e9e6ca7dc75a7df9e037adca6ffda663b5544ccb2afad5532c",
    "src/ecospec_kg/prediction_validation_v2.py": "bfd0470c54e7772d8c954fa633bd7639ed1e0b37501fa784fb289e2e07dca1df",
}
FROZEN = {
    "blind/train_units.jsonl": "6cb286dd48a36d478f68eaf2ed7e1ec28cd8bbd31fdef9ae33359e0446a52162",
    "blind/dev_units.jsonl": "a1776a04775ca50fbdc17dcb80ef3223f0214028b3b5f425243783d9e47cd9e8",
    "gold/train_annotations.jsonl": "dde317b9f1c57cea2a35c5721638cca14b4ddcd0c495c8b38b45d9b16e2eb6c0",
    "gold/dev_annotations.jsonl": "9047b685b2ccb76b986fb2132a3259d231b5cc8a1798cffc1e802bf982d4338a",
    "schema_v2.json": "fca3a421a97f7a47d37d9bed88b39f29ad9f1d33834a4e57fc26ebd7fdcd602f",
}
VIEW_HASHES = {
    "source_units.jsonl": "657a3712c08d006a30df9d3ef4209338613f3068e5df1d32664e7cc9f183caa7",
    "derived_train_annotations.jsonl": "918fb1f70243c7b8b30589cbecf942a86de433ed121056d596f7f20c4656db19",
    "qwen35_replacements.jsonl": "4d5a38f914a8a9f002b7fae7cf3e204728745d062370983e30196c847b1c37fa",
    "derivation.json": "45336ff599fb9f390e12aa5d3975e8059dcb24c486bfde5bb7dd697a5c925cf4",
}
VIEW_FILES = list(VIEW_HASHES)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_rows(value):
    return [json.loads(line) for line in value.splitlines() if line.strip()]


def rows(path):
    return parse_rows(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def uid(row):
    return json.loads(row["messages"][1]["content"])["source_unit"]["unit_id"]


def validate_views(view_data, parent_gold):
    from ecospec_kg.evaluation_v2 import _entity_key, _relation_key
    from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, build_llm_selection_messages
    assert set(view_data) == set(VIEW_FILES), "Incomplete view payload"
    assert all(hashlib.sha256(view_data[n].encode("utf-8")).hexdigest() == h
               for n, h in VIEW_HASHES.items()), "Changed view payload"
    units = parse_rows(view_data["source_units.jsonl"])
    gold = parse_rows(view_data["derived_train_annotations.jsonl"])
    training = parse_rows(view_data["qwen35_replacements.jsonl"])
    derivation = json.loads(view_data["derivation.json"])
    ids = {u["unit_id"] for u in units}
    assert len(units) == len(gold) == len(training) == len(ids) == 3, "Expected three views"
    assert ids == {g["unit_id"] for g in gold} == {uid(r) for r in training}, "View membership"
    assert derivation["parent_unit_id"] == PARENT and all(
        m["parent_unit_id"] == PARENT for m in derivation["child_mapping"]), "View parent mismatch"
    assert {m["unit_id"] for m in derivation["child_mapping"]} == ids, "View mapping mismatch"
    relation_keys = lambda gs: {_relation_key("parent", r) for g in gs for r in g["relations"]}
    entity_keys = lambda gs: {_entity_key("parent", e) for g in gs for e in g["entities"]}
    assert relation_keys(gold) == relation_keys([parent_gold]), "Lost parent relations"
    assert entity_keys(gold) == entity_keys([parent_gold]), "Lost parent entities"
    assert len(relation_keys(gold)) == 34, "Expected 34 distinct preserved relations"
    by_gold = {g["unit_id"]: g for g in gold}
    by_training = {uid(r): r for r in training}
    for u in units:
        c = RuleCandidateExtractorV2().predict_unit(u)
        system, prompt = build_llm_selection_messages(u, c)
        messages = by_training[u["unit_id"]]["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"], "View messages"
        assert messages[0]["content"] == system and json.loads(messages[1]["content"]) == json.loads(prompt), "View prompt mismatch"
        g = by_gold[u["unit_id"]]
        assert g["split"] == "train" and g["derived_from_unit_id"] == PARENT, "View gold provenance"
        selected_r = [r for r in c["relations"] if _relation_key("parent", r) in relation_keys([g])]
        required = {r[side + "_id"] for r in selected_r for side in ("head", "tail")}
        selected_e = {e["entity_id"] for e in c["entities"] if
                      _entity_key("parent", e) in entity_keys([g]) and e["entity_id"] not in required}
        target = json.loads(messages[2]["content"])
        assert set(target["selected_relation_ids"]) == {r["relation_id"] for r in selected_r}, "View relation targets"
        assert set(target["selected_entity_ids"]) == selected_e, "View entity targets"
    return training


def coverage_report(units, gold, training, split_ids):
    from ecospec_kg.evaluation_v2 import _relation_key
    from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
    by_gold = {g["unit_id"]: g for g in gold}
    by_training = {uid(r): r for r in training}
    groups = {"all693": set(by_gold), "effective_train623": set(split_ids["train"]) - {PARENT},
              "validation69": set(split_ids["validation"])}
    report = {name: {"unit_count": len(ids), "roles": {}, "missing_gold_relations": []}
              for name, ids in groups.items()}
    for u in units:
        unit = u["unit_id"]
        c = RuleCandidateExtractorV2().predict_unit(u)
        ck = {_relation_key(unit, r) for r in c["relations"]}
        gk = {_relation_key(unit, r) for r in by_gold[unit]["relations"]}
        target = json.loads(by_training[unit]["messages"][-1]["content"])
        selected = set(target["selected_relation_ids"])
        sk = {_relation_key(unit, r) for r in c["relations"] if r["relation_id"] in selected}
        assert sk == gk & ck, "Existing training target/candidate mismatch"
        for name, ids in groups.items():
            if unit not in ids:
                continue
            output = report[name]
            for key in sorted(gk | ck):
                role = "|".join((key[2], key[3], key[5]))
                bucket = output["roles"].setdefault(role, {"gold_count": 0, "covered_gold_count": 0,
                                                         "candidate_count": 0, "supervised_positive_count": 0})
                bucket["gold_count"] += int(key in gk)
                bucket["covered_gold_count"] += int(key in gk & ck)
                bucket["candidate_count"] += int(key in ck)
                bucket["supervised_positive_count"] += int(key in sk)
                if key in gk - ck:
                    output["missing_gold_relations"].append(list(key))
    return report


def run(root, prep, model, dest, view_data):
    assert not dest.exists(), "Output already exists"
    assert dest.resolve().is_relative_to(prep.resolve()), "Output must be inside prepared directory"
    assert importlib.metadata.version("ms-swift") == "4.4.2", "Expected ms-swift 4.4.2"
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == HEAD, "Wrong baseline"
    assert all(hashlib.sha256((root / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == h
               for n, h in SOURCE.items()), "Source mismatch"
    data = root / "deliveries/human_confirmed_20261002"
    assert all(sha(data / n) == h for n, h in FROZEN.items()), "Frozen input mismatch"
    training_path = prep / "qwen35_train.jsonl"
    assert sha(training_path) == TRAIN_SHA, "Training file mismatch"
    training = rows(training_path)
    ids = {uid(r) for r in training}
    assert len(training) == len(ids) == 693, "Expected 693 unique inputs"
    comparison_path = prep / "length_policy_comparison_8192.json"
    comparison = read(comparison_path)
    assert comparison["passed"] and comparison["policy"] == {
        "max_length": 8192, "truncation_strategy": "delete", "split_dataset_ratio": 0.1, "data_seed": 42}, "Length policy mismatch"
    assert comparison["effective_records"] == {"train": 623, "validation": 69} and comparison["excluded_unit_ids"] == [PARENT], "Filtered membership mismatch"
    assert comparison["training_sha256"] == TRAIN_SHA, "Comparison training mismatch"
    pf_path = Path(comparison["preflight_path"])
    assert pf_path.resolve().is_relative_to(prep.resolve()) and sha(pf_path) == comparison["preflight_sha256"], "Preflight hash mismatch"
    pf = read(pf_path)
    assert pf["ms_swift_version"] == "4.4.2" and pf["training_sha256"] == TRAIN_SHA, "Preflight environment mismatch"
    assert pf["data_seed"] == 42 and pf["split_dataset_ratio"] == 0.1, "Preflight policy mismatch"
    partition = pf["split_unit_ids"]
    tr, va = set(partition["train"]), set(partition["validation"])
    assert len(partition["train"]) == len(tr) == 624 and len(partition["validation"]) == len(va) == 69, "Invalid partition sizes"
    assert not tr & va and tr | va == ids and PARENT in tr, "Invalid parent partition"
    assert pf["encoded_records"] == {"train": 623, "validation": 69}, "Preflight counts mismatch"
    assert len(pf["encoding_failures"]) == 1 and all(
        pf["encoding_failures"][0][k] == v for k, v in
        {"unit_id": PARENT, "split": "train", "error_type": "MaxLengthError"}.items()), "Unexpected original encoding failures"
    tokenizer_hashes = pf["tokenizer_config_sha256"]
    assert tokenizer_hashes and "config.json" in tokenizer_hashes and tokenizer_hashes == comparison["tokenizer_config_sha256"], "Tokenizer records mismatch"
    assert all((model / n).resolve().is_relative_to(model.resolve()) and sha(model / n) == h
               for n, h in tokenizer_hashes.items()), "Tokenizer files changed"
    units, gold = rows(data / "blind/train_units.jsonl"), rows(data / "gold/train_annotations.jsonl")
    assert len(units) == len(gold) == 693 and {u["unit_id"] for u in units} == {g["unit_id"] for g in gold} == ids, "Frozen membership mismatch"
    parent_gold = next(g for g in gold if g["unit_id"] == PARENT)
    views = validate_views(view_data, parent_gold)
    assert not {uid(r) for r in views} & ids, "Child collides with original input"
    summary_dir = prep / SUMMARY_NAME
    summary = read(summary_dir / "summary.json")
    assert summary["status"] == "complete" and summary["model_calls"] == 0 and summary["history_unchanged"], "Structural result incomplete"
    assert set(summary["runs"]) == set(LABELS), "Expected seven saved runs"
    result_root = prep.parent.resolve()
    run_dirs = [summary_dir] + [Path(summary["runs"][label]["baseline"]) for label in LABELS]
    assert len({p.resolve() for p in run_dirs}) == 8 and all(
        p.is_dir() and p.resolve().is_relative_to(result_root) for p in run_dirs), "Invalid result directory"
    result_files = [p for folder in run_dirs for p in folder.rglob("*") if p.is_file()]
    assert all(not p.is_symlink() and p.resolve().is_relative_to(result_root) for p in result_files), "Result path escapes directory"
    originals = [*result_files, training_path, comparison_path, pf_path,
                 *[data / n for n in FROZEN], *[root / n for n in SOURCE], *[model / n for n in tokenizer_hashes]]
    before = {p: sha(p) for p in originals}
    print("CHECKING_TRAIN_CANDIDATE_COVERAGE", flush=True)
    coverage = coverage_report(units, gold, training, partition)
    dest.mkdir()
    save(dest / "train_candidate_coverage.json", coverage)
    view_dest = dest / "long_views"
    view_dest.mkdir()
    for name in VIEW_FILES:
        (view_dest / name).write_text(view_data[name], encoding="utf-8")
    from swift.model import get_model_processor
    from swift.template import get_template
    print("CHECKING_THREE_VIEWS_WITH_CPU_TOKENIZER", flush=True)
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None, "Unexpected model weight loading"
    template = get_template(processor, max_length=8192, truncation_strategy="raise",
                            add_non_thinking_prefix=True, loss_scale="default+ignore_empty_think")
    assert template.template_meta.template_type == "qwen3_5", "Unexpected model template"
    template.set_mode("train")
    encodings, failures = [], []
    for row in views:
        unit = uid(row)
        try:
            encoded = template.encode(copy.deepcopy(row))
            length = len(encoded["input_ids"])
            labels = encoded["labels"]
            supervised = sum(v != -100 for v in labels)
            assert len(labels) == length and 0 < length <= 8192 and supervised > 0, "Length or supervision check failed"
            encodings.append({"unit_id": unit, "parent_unit_id": PARENT, "split": "train",
                              "tokens": length, "supervised_tokens": supervised})
        except Exception as exc:
            failures.append({"unit_id": unit, "parent_unit_id": PARENT, "split": "train",
                             "error_type": type(exc).__name__, "error": str(exc)})
    archive = result_root / ("next_step_results_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())[-9:] + ".tar.gz")
    report = {"status": "checks_complete", "output": str(dest), "archive": str(archive),
              "model_calls": 0, "model_weights_loaded": False, "model_training_started": False,
              "length_check_passed": not failures, "view_encodings": encodings, "encoding_failures": failures,
              "parent_unit_id": PARENT, "relation_union_preserved": 34,
              "original_effective_records": {"train": 623, "validation": 69},
              "proposed_effective_records": {"train": 626, "validation": 69} if not failures else None,
              "partition_rule": "inherit saved parent split; no second random split",
              "ready_for_training": False,
              "pending": ["review original predictions and structural proofs", "repair training-source candidate coverage",
                          "resolve ambiguous method-space annotation policy", "define paired update budget before seed42 pilot"],
              "tokenizer_config_sha256": tokenizer_hashes, "source_sha256_lf": SOURCE,
              "frozen_sha256": FROZEN, "view_sha256_utf8_lf": VIEW_HASHES,
              "training_sha256": TRAIN_SHA, "preflight_sha256": sha(pf_path), "history_unchanged": True}
    assert all(sha(p) == h for p, h in before.items()), "Historical input changed during checks"
    save(dest / "long_view_preflight.json", report)
    save(dest / "original_file_inventory.json", {str(p): h for p, h in before.items()})
    print("PACKAGING_ORIGINAL_RESULTS_AND_REPORTS", flush=True)
    excluded_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    with tarfile.open(archive, "x:gz") as package:
        for p in sorted(set(result_files + [training_path, comparison_path, pf_path] + list(dest.rglob("*")))):
            if p.is_file() and p.suffix.lower() not in excluded_suffixes:
                package.add(p, arcname=p.relative_to(result_root).as_posix(), recursive=False)
        for n in FROZEN:
            package.add(data / n, arcname="review_inputs/" + n, recursive=False)
        for n in SOURCE:
            package.add(root / n, arcname="review_sources/" + n, recursive=False)
    assert all(sha(p) == h for p, h in before.items()), "Historical input changed during archive"
    return report


def main(view_data):
    root = Path("/home/hello/szl/eco-spec-kg-bindings-v26_20261012_110721")
    prep = Path("/home/hello/szl/eco-spec-results/bindings_v26_20261012_110721")
    model = Path("/home/hello/szl/models/Qwen3.5-9B")
    dest = prep / ("next_checks_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())[-9:])
    report = run(root, prep, model, dest, view_data)
    print(json.dumps({k: report[k] for k in ("output", "length_check_passed", "view_encodings", "encoding_failures", "proposed_effective_records", "ready_for_training")}, ensure_ascii=False, indent=2), flush=True)
    print("DOWNLOAD=" + report["archive"], flush=True)


if __name__ == "__main__":
    import base64
    import zlib
    data = zlib.decompress(base64.b64decode("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"))
    assert hashlib.sha256(data).hexdigest() == "52f2a83c6717b499123808cdd507900ef070f20ec64891932649039b7f292c90", "Embedded payload mismatch"
    main(json.loads(data.decode("utf-8")))
