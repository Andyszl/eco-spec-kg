"""Full-graph conditional supervision: rebuild, audit, CPU tokenization only."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecospec_kg import extractor_v2 as ex
from ecospec_kg import candidate_profile_v214 as profile
from ecospec_kg import full_graph_scope_v219 as scope
from ecospec_kg.experiment_io_v2 import sha256_json
import cross_source_pair_20261016 as previous
from cross_source_pair_20261016 import read, rows, sha, lfsha, save, dumps, uid
from check_scope_inputs_20261013 import encode_files

OLD = previous.FOLDER
FOLDER = "experiments/full_graph_scope_pair_20261018"
OLD_MANIFEST_SHA = "c9a4a6479728ca90b257cc3fdc3968a0ce260ef0bfc1808070343776df27cc4f"
TEMPLATE_CONTRACT = {
    "template_type": "qwen3_5", "mode": "train", "max_length": 8192,
    "truncation_strategy": "raise", "add_non_thinking_prefix": True,
    "loss_scale": "default+ignore_empty_think",
}
SOURCES = (
    "src/ecospec_kg/full_graph_scope_v219.py",
    "tools/full_graph_scope_pair_20261018.py",
    "tools/run_full_graph_scope_preflight_20261018.sh",
    "tools/check_scope_inputs_20261013.py",
    "src/ecospec_kg/selection_decoding_v2.py",
)


def compose(root):
    assert sha(root / OLD / "input_manifest.json") == OLD_MANIFEST_SHA, "Old manifest changed"
    old = previous.verify_inputs(root)["manifest"]
    all_units = rows(root / previous.OLD / "units.jsonl") + rows(root / previous.SEED / "source_units.jsonl")
    units = {u["unit_id"]: u for u in all_units}
    graphs = {}; scopes = {}; prepared = {}; counts = {"train": 0, "validation": 0}
    for name, identities in old["unit_ids"].items():
        original = rows(root / OLD / (name + ".jsonl"))
        assert [uid(r) for r in original] == identities
        records = []
        for row in original:
            ident = uid(row); unit = units[ident]
            if ident not in graphs:
                graphs[ident] = profile.candidates_for_unit(unit)
            graph = graphs[ident]
            assert sha256_json(graph) == old["candidate_hashes"][ident], "Full graph differs from reviewed graph"
            q = json.loads(row["messages"][1]["content"])
            full = json.loads(ex.build_llm_selection_messages(unit, graph)[1])
            assert q["candidate_entities"] == full["candidate_entities"]
            prior_relations = {r["id"]: r for r in q["candidate_relations"]}
            full_relations = {r["id"]: r for r in full["candidate_relations"]}
            assert all(full_relations.get(i) == r for i, r in prior_relations.items())
            unscored = sorted(set(full_relations) - set(prior_relations))
            if ident in scopes:
                assert scopes[ident] == unscored, "Pair scope differs"
            else:
                scopes[ident] = unscored
                counts["validation" if name == "validation" else "train"] += len(unscored)
            system, prompt = scope.build_messages(unit, graph, unscored)
            scope.validate_selection(prompt, json.loads(row["messages"][-1]["content"]))
            rebuilt = deepcopy(row)
            rebuilt["messages"][0]["content"] = system
            rebuilt["messages"][1]["content"] = prompt
            records.append(rebuilt)
        prepared[name] = records
    assert counts == old["quarantine_counts"] == {"train": 16, "validation": 5}
    a, b = prepared["control_train"], prepared["repair_train"]
    assert all(x["messages"][:2] == y["messages"][:2] for x, y in zip(a, b))
    changed = sum(x["messages"][-1] != y["messages"][-1] for x, y in zip(a, b))
    assert changed == old["paired_changed_training_targets"] == 6
    assert set(old["unit_ids"]["repair_train"]).isdisjoint(old["unit_ids"]["validation"])
    # The old locked input builder audits original splits, long views and FVC holdout.
    # No dev/test units or labels are read to construct these records.
    m = {
        "version": scope.VERSION, "candidate_profile": profile.VERSION,
        "previous_input_manifest_sha256": OLD_MANIFEST_SHA,
        "counts": {n: len(v) for n, v in prepared.items()}, "unit_ids": old["unit_ids"],
        "tokenizer_config_sha256": old["tokenizer_config_sha256"],
        "source_sha256_lf": {n: lfsha(root / n) for n in SOURCES},
        "candidate_hashes": old["candidate_hashes"],
        "graph_sizes": {i: {"entities": len(g["entities"]), "relations": len(g["relations"])} for i, g in graphs.items()},
        "unscored_relation_ids": scopes, "unscored_counts": counts,
        "paired_changed_training_targets": changed, "assistant_targets_unchanged": True,
        "full_graph_preserved": True, "entity_supervision_unchanged": True,
        "question_contract_changed": True, "token_level_loss_masking": False,
        "unknown_relations_are_negatives": False, "inference_unscored_relation_ids": [],
        "template_contract": TEMPLATE_CONTRACT,
        "paired_protocol": old["paired_protocol"],
        "supervision": old["supervision"],
        "clause_guard_applied_to_inputs": False,
        "training_ready": False, "model_training_started": False,
        "requires_real_tokenizer_preflight": True,
        "limitations": [
            "Conditional question scope, not token-level loss masking; actual effects remain untested.",
            "Training scope lists differ from inference empty scope, despite identical full candidate context.",
            "No new gold, positive coverage or held-out sources added; original entity supervision unchanged.",
            "Control remains the declared incomplete-supervision ablation, not independently verified gold.",
            "Internal validation has been used in design and shares sources with training; no final-test claim.",
            "Historic inference entrypoints unchanged; future base/control/repair evaluation must use scoped select equally.",
        ],
    }
    return prepared, m


def build_inputs(root, out):
    assert not out.exists(), "Preserve existing input delivery"
    files, manifest = compose(root)
    out.mkdir(parents=True)
    for n, values in files.items():
        (out / (n + ".jsonl")).write_text(dumps(values), encoding="utf-8", newline="\n")
    manifest["files"] = {n + ".jsonl": sha(out / (n + ".jsonl")) for n in files}
    save(out / "input_manifest.json", manifest)
    return manifest


def verify_inputs(root, folder=None):
    folder = folder or root / FOLDER
    m = read(folder / "input_manifest.json")
    for n, h in m["files"].items():
        assert Path(n).name == n and sha(folder / n) == h, "Input changed: " + n
    for n, h in m["source_sha256_lf"].items():
        assert lfsha(root / n) == h, "Code changed: " + n
    prepared, expected = compose(root)
    assert {k: v for k, v in m.items() if k != "files"} == expected, "Input metadata changed"
    assert all(rows(folder / (n + ".jsonl")) == v for n, v in prepared.items()), "Projection differs from reviewed inputs"
    return {"manifest_sha256": sha(folder / "input_manifest.json"), "manifest": m}


def verify_preflight(root, folder, report):
    assert report.get("passed") and report.get("real_tokenizer") is True, "Passing real tokenizer report required"
    checked = verify_inputs(root, folder); m = checked["manifest"]
    assert report["input_manifest_sha256"] == checked["manifest_sha256"] and report["input_manifest"] == m
    assert report["checked_records"] == sum(m["counts"].values()) == len(report["encodings"]) == 1324
    assert not report["failures"] and report["swift_version"] == "4.4.2"
    assert report["template_contract"] == TEMPLATE_CONTRACT
    assert report["model_weights_loaded"] is False and report["model_training_started"] is False
    assert report["training_ready"] is False and report["original_inputs_unchanged"] is True
    assert report["source_sha256_lf"] == m["source_sha256_lf"]
    assert [(r["file"], r["unit_id"]) for r in report["encodings"]] == [
        (n, i) for n, ids in m["unit_ids"].items() for i in ids]
    assert all(0 < r["supervised_tokens"] <= r["tokens"] <= 8192 for r in report["encodings"])
    return checked


def preflight(root, model, out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight must hide GPUs"
    assert not out.exists(), "Preserve existing report"
    checked = verify_inputs(root); m = checked["manifest"]
    assert importlib.metadata.version("ms-swift") == "4.4.2"
    for n, h in m["tokenizer_config_sha256"].items():
        assert Path(n).name == n and sha(model / n) == h, "Tokenizer/config changed: " + n
    from swift.model import get_model_processor
    from swift.template import get_template
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None, "Do not load model weights"
    kwargs = {k: v for k, v in TEMPLATE_CONTRACT.items() if k not in ("template_type", "mode")}
    template = get_template(processor, **kwargs)
    assert template.template_meta.template_type == TEMPLATE_CONTRACT["template_type"]
    template.set_mode(TEMPLATE_CONTRACT["mode"])
    files = {n: rows(root / FOLDER / (n + ".jsonl")) for n in m["counts"]}
    report = encode_files(template, files)
    assert report["checked_records"] == 1324
    assert verify_inputs(root) == checked, "Inputs changed during encoding"
    report.update(
        real_tokenizer=True, input_manifest_sha256=checked["manifest_sha256"], input_manifest=m,
        source_sha256_lf=m["source_sha256_lf"], swift_version="4.4.2", template_contract=TEMPLATE_CONTRACT,
        model_weights_loaded=False, model_training_started=False, original_inputs_unchanged=True, training_ready=False,
        code_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        max_tokens_by_file={n: max((r["tokens"] for r in report["encodings"] if r["file"] == n), default=0) for n in files},
        next_step="Audit returned real tokenizer report before preparing guarded fixed-seed paired training; do not train now",
    )
    out.mkdir(parents=True)
    save(out / "preflight_report.json", report)
    if report["passed"]:
        verify_preflight(root, root / FOLDER, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify", "preflight"))
    p.add_argument("--out", type=Path); p.add_argument("--model", type=Path)
    a = p.parse_args()
    if a.action == "build":
        result = build_inputs(ROOT, a.out or ROOT / FOLDER)
    elif a.action == "verify":
        result = verify_inputs(ROOT)
    else:
        assert a.out and a.model
        result = preflight(ROOT, a.model, a.out)
    brief = result.get("manifest", result)
    print(json.dumps({k: brief[k] for k in ("counts", "unscored_counts", "passed", "checked_records", "max_tokens_by_file", "failures", "training_ready") if k in brief}, ensure_ascii=False, indent=2))
    print("FULL_GRAPH_SCOPE_CHECK_OK=" + ("true" if brief.get("passed", True) else "false"))
    print("TRAINING_READY=false")
    if a.action == "preflight" and not result["passed"]:
        raise SystemExit(1)
