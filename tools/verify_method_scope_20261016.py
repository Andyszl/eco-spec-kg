"""Replay source-local alias normalization and symmetric training quarantine.

No model/tokenizer import, no data rebuild, no training or inference requests.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.method_supervision_scope_v2 import canonicalize_methods, training_selection_view
from ecospec_kg.experiment_io_v2 import sha256_json
from verify_named_method_review_20261016 import verify as verify_previous

FOLDER = "experiments/method_scope_review_20261016"
PREVIOUS = "experiments/semantic_pair_v211_20261015"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(relation):
    return tuple(relation[k] for k in ("head_name", "head_type", "relation_type", "tail_name", "tail_type"))


def project(candidates, mapping, annotation):
    """Project known reference selections through the canonical graph."""
    known = {key(r) for r in annotation["relations"]}
    selected = {mapping["relation_id_map"][r["relation_id"]] for r in candidates["relations"] if key(r) in known}
    es = {(e["name"], e["entity_type"]) for e in annotation["entities"]}
    selected_es = {mapping["entity_id_map"][e["entity_id"]] for e in candidates["entities"]
                   if (e["name"], e["entity_type"]) in es}
    return {"selected_entity_ids": sorted(selected_es), "selected_relation_ids": sorted(selected)}


def audit(root):
    # Recheck immutable inputs and the prior, separately published v2.12 fix.
    verify_previous(root)
    source = root / PREVIOUS
    manifest = read(source / "input_manifest.json")
    packet = read(root / FOLDER / "source_review.json")
    assert sha(source / "input_manifest.json") == packet["current_manifest_sha256"]
    units = {u["unit_id"]: u for u in rows(source / "units.jsonl")}
    train = set(manifest["unit_ids"]["repair_train"])
    validation = set(manifest["unit_ids"]["validation"])
    assert not train & validation and len(train) == 625 and len(validation) == 70
    labels = {arm: {g["unit_id"]: g for g in rows(source / f"{arm}_annotations.jsonl")}
              for arm in ("control", "repair")}
    old_targets = {}
    for arm in ("control", "repair"):
        records = rows(source / f"{arm}_train.jsonl") + rows(source / "validation.jsonl")
        old_targets[arm] = {json.loads(r["messages"][1]["content"])["source_unit"]["unit_id"]:
                            json.loads(r["messages"][-1]["content"]) for r in records}
    addition = rows(root / "experiments/semantic_method_review_20261016/annotation_revision.jsonl")[0]
    assert addition["unit_id"] in train
    labels["repair"][addition["unit_id"]] = addition
    raw = {ident: ex.RuleCandidateExtractorV2().predict_unit(u) for ident, u in units.items()}
    unresolved = {}
    for case in packet["cases"]:
        ident = case["unit_id"]
        assert ident in train, "Review must address current train only"
        assert units[ident]["provenance"]["source_sha256"] == case["source_sha256"]
        spans = ex._all_spans(units[ident])
        assert all(s["span_id"] in spans and spans[s["span_id"]] == s for s in case["source_spans"]), "Training review evidence changed"
        edge = next(r for r in raw[ident]["relations"] if r["relation_id"] == case["candidate_relation_id"])
        assert (edge["head_name"], edge["relation_type"], edge["tail_name"]) == (case["head"], case["type"], case["tail"])
        assert (key(edge) in {key(r) for r in labels["repair"][ident]["relations"]}) == case["selected_in_current_target"]
        if case["review_status"] == "unresolved_attribution_not_confirmed_negative":
            unresolved.setdefault(ident, set()).add(edge["relation_id"])
    # The new named method supports K_r directly; its other offered variable
    # relation has no adjudicated positive or negative label yet.
    ident = addition["unit_id"]
    positive = {key(r) for r in addition["relations"]}
    new_method = read(root / "experiments/semantic_method_review_20261016/review.json")["added_entity"]["name"]
    for r in raw[ident]["relations"]:
        if r["relation_type"] == "obtained_by" and r["tail_name"] == new_method and key(r) not in positive:
            unresolved.setdefault(ident, set()).add(r["relation_id"])
    counts, details, preview_hashes = Counter(), [], {}
    for ident, unit in units.items():
        c, mapping = canonicalize_methods(unit, raw[ident])
        unknown = {mapping["relation_id_map"][r] for r in unresolved.get(ident, [])}
        assert unknown <= {r["relation_id"] for r in c["relations"]}, "Inference lost an unknown relation"
        assert canonicalize_methods(unit, c)[0] == c, "Canonicalization must be idempotent"
        merged_e = len(raw[ident]["entities"]) - len(c["entities"])
        merged_r = len(raw[ident]["relations"]) - len(c["relations"])
        counts["merged_method_entities"] += merged_e
        counts["merged_relations"] += merged_r
        counts["quarantined_canonical_training_edges"] += len(unknown)
        prompts, targets = [], []
        for arm in ("control", "repair"):
            target = project(raw[ident], mapping, labels[arm][ident])
            expected = {mapping["relation_id_map"][r] for r in old_targets[arm][ident]["selected_relation_ids"]}
            if arm == "repair" and ident == addition["unit_id"]:
                reviewed_key = key(read(root / "experiments/semantic_method_review_20261016/review.json")["added_relation"])
                expected.update(mapping["relation_id_map"][r["relation_id"]] for r in raw[ident]["relations"] if key(r) == reviewed_key)
            assert set(target["selected_relation_ids"]) == expected, "Existing selected relation lost or unreviewed positive added"
            target["selected_entity_ids"] = sorted(set(target["selected_entity_ids"]) -
                {r[s + "_id"] for r in c["relations"] if r["relation_id"] in target["selected_relation_ids"] for s in ("head", "tail")})
            view, kept = training_selection_view(c, target, unknown if ident in train else [])
            assert kept == target, "Known positive supervision lost"
            assert unknown.isdisjoint({r["relation_id"] for r in view["relations"]})
            system, prompt = ex.build_llm_selection_messages(unit, view)
            q = json.loads(prompt)
            assert not unknown & {r["id"] for r in q["candidate_relations"]}
            assert set(kept["selected_relation_ids"]) <= {r["id"] for r in q["candidate_relations"]}
            prompts.append((system, prompt))
            targets.append(kept)
            counts[arm + "_known_train_method_source_positives"] += sum(
                r["relation_id"] in kept["selected_relation_ids"] and r["head_type"] == "model_variable"
                and r["relation_type"] in {"obtained_by", "sourced_from"} for r in view["relations"]) if ident in train else 0
            preview_hashes[ident + ":" + arm] = sha256_json({"system": system, "prompt": prompt, "target": kept})
        assert prompts[0] == prompts[1], "Arms must share candidate questions"
        counts["paired_changed_training_targets"] += int(ident in train and targets[0] != targets[1])
        if merged_e or merged_r or unknown:
            details.append({"unit_id": ident, "split": "train" if ident in train else "validation",
                            "merged_entities": merged_e, "merged_relations": merged_r,
                            "unknown_training_relation_ids": sorted(unknown),
                            "entity_aliases": {k: v for k, v in mapping["entity_id_map"].items() if k != v}})
    return {"status": "method_scope_review_verified", "train_records": len(train), "validation_records": len(validation),
            "reviewed_unknown_original_edges": sum(map(len, unresolved.values())), **dict(counts),
            "known_positive_edges_lost": 0, "unknown_removed_from_inference": 0,
            "input_preview_sha256": sha256_json(preview_hashes), "changed_units": details,
            "source_review": "assistant_review_not_independent_human_gold",
            "training_inputs_written": False, "cross_source_supplement_complete": False, "training_ready": False}


def verify(root):
    folder = root / FOLDER
    lock = read(folder / "verification_lock.json")
    for name, expected in lock["sha256_lf"].items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual == expected, "Scope review file changed: " + name
    report = audit(root)
    assert report == read(folder / "verified_report.json"), "Scope replay differs from published report"
    return report


if __name__ == "__main__":
    result = verify(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("METHOD_SCOPE_REVIEW_OK=true")
    print("QUARANTINED_TRAINING_EDGES=" + str(result["quarantined_canonical_training_edges"]))
    print("TRAINING_READY=false")
