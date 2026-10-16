"""Probe existing rare-role supervision with fixed adapters; no training/dev gate change."""
import argparse
from dataclasses import fields
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess

import evaluate_semantic_pair_20261016 as ev
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.io_utils import write_jsonl
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2

semantic = ev.semantic
ROOT = Path(__file__).resolve().parents[1]


def semantic_relations(prompt):
    entities = {e["id"]: e for e in prompt["candidate_entities"]}
    return {r["id"]: r for r in prompt["candidate_relations"] if
        "|".join((entities[r["head"]]["type"], r["type"], entities[r["tail"]]["type"])) in semantic.ROLES}


def build_cases(root):
    semantic.verify_delivery(root)
    folder = root / semantic.FOLDER
    units = {u["unit_id"]: u for u in semantic.rows(folder / "units.jsonl")}
    control = {semantic.uid(r): r for name in ("control_train", "validation") for r in semantic.rows(folder / f"{name}.jsonl")}
    cases = []
    for split, name in (("train", "repair_train"), ("validation", "validation")):
        for row in semantic.rows(folder / f"{name}.jsonl"):
            uid = semantic.uid(row)
            prompt = json.loads(row["messages"][1]["content"])
            relations = semantic_relations(prompt)
            reference = sorted(set(json.loads(row["messages"][-1]["content"])["selected_relation_ids"]) & relations.keys())
            if not reference:
                continue
            original = set(json.loads(control[uid]["messages"][-1]["content"])["selected_relation_ids"])
            candidate = ev.ex.RuleCandidateExtractorV2().predict_unit(units[uid])
            system, user = ev.ex.build_llm_selection_messages(units[uid], candidate)
            assert row["messages"][:2] == [{"role": "system", "content": system}, {"role": "user", "content": user}], "Probe prompts changed"
            bound = len(json.dumps({"selected_entity_ids": [e["id"] for e in prompt["candidate_entities"]],
                "selected_relation_ids": [r["id"] for r in prompt["candidate_relations"]]})) + 1
            assert bound <= 3072, "Probe needs a larger completion budget"
            cases.append({"unit": units[uid], "split": split, "messages": row["messages"][:2],
                "reference_positive_ids": reference, "control_target_positive_ids": sorted(original & relations.keys()),
                "new_positive_ids": sorted(set(reference) - original) if split == "train" else [],
                "semantic_candidate_ids": sorted(relations), "completion_bound": bound,
                "prompt_hash": sha256_json({"system": system, "prompt": user}),
                "candidate_hash": sha256_json({k: candidate[k] for k in ("entities", "relations")})})
    assert len(cases) == 6 and len({c["unit"]["unit_id"] for c in cases}) == 6
    assert sum(len(c["new_positive_ids"]) for c in cases) == 4
    return cases


def deploy_command(root, manifest, port):
    command = ev.deploy_command(root, manifest, port)
    # Training targets fit 8192. Near-limit prompts also need space for every
    # allowed grammar continuation; this diagnostic reserves context, not new text.
    command[command.index("--max_length") + 1] = "16384"
    return command


def run_probe(root, train, out, base_url="http://127.0.0.1:8018/v1"):
    assert not out.exists(), "Probe output exists"
    manifest = ev.inspect_pair(root, train)
    capability = ev.check_service(base_url)
    cases = build_cases(root)
    cfg = {"experiment_id": "semantic-supervision-fit-probe", "backend": "llm", "model": ev.MODELS["base"],
        **ev.read(root / ev.PROTOCOL)["selection_protocol"], "selection_decoding": ev.VERSION,
        "use_schema": True, "use_layout": True, "use_evidence": True}
    out.mkdir(parents=True)
    source = out / "probe_units.jsonl"
    write_jsonl(source, [c["unit"] for c in cases])
    ev.save(out / "probe_config.json", cfg)
    ev.save(out / "probe_cases.json", cases)
    ev.save(out / "probe_protocol.json", {"training_manifest_sha256": ev.sha(train / "pair_manifest.json"),
        "protocol_sha256": ev.sha(root / ev.PROTOCOL), "capability": capability,
        "code_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "inference_context_limit": 16384, "training_length_limit_unchanged": 8192,
        "scope": "all existing positive rare-role train units plus internal heldout positive unit; not dev or independent validation"})
    results = {}
    factory = ev.Provider.__dict__["from_env"]
    try:
        ev.Provider.from_env = classmethod(lambda cls: ev.Provider(base_url=base_url, api_key="local-token", timeout=600, enable_thinking=False))
        for arm, model in ev.MODELS.items():
            dest = out / arm
            m = ev.ex.extract_v2(source, dest, config_path=out / "probe_config.json", model=model)
            assert m["status"] == "complete" and m["summary"]["success_count"] == m["summary"]["unit_count"] == 6 and m["summary"]["error_count"] == 0, "Probe extraction failed"
            assert ev.read(dest / "resolved_config.json") == {**cfg, "model": model}
            predictions = semantic.rows(dest / "predictions.jsonl")
            validation = validate_predictions_v2(source, dest / "predictions.jsonl", root / "deliveries/human_confirmed_20261002/schema_v2.json", dest / "validation")
            assert validation["passed"] and validation["valid_prediction_unit_count"] == 6
            by_id = {p["unit_id"]: p for p in predictions}
            assert len(by_id) == 6 and set(by_id) == {c["unit"]["unit_id"] for c in cases}
            values = []
            for c in cases:
                uid = c["unit"]["unit_id"]
                p = by_id[uid]
                assert p["prompt_hash"] == c["prompt_hash"] and p["candidate_hash"] == c["candidate_hash"] and p["seed"] == 42
                selected = {r["relation_id"] for r in p["relations"]} & set(c["semantic_candidate_ids"])
                reference = set(c["reference_positive_ids"])
                prompt = json.loads(c["messages"][1]["content"])
                es = {e["id"]: e for e in prompt["candidate_entities"]}
                relations = semantic_relations(prompt)
                values.append({"unit_id": uid, "split": c["split"],
                    "reference_positive_ids": c["reference_positive_ids"], "selected_reference_ids": sorted(selected & reference),
                    "missed_reference_ids": sorted(reference - selected), "selected_new_positive_ids": sorted(selected & set(c["new_positive_ids"])),
                    "extra_selected_semantic_ids": sorted(selected - reference),
                    "relation_details": [{"id": rid, "head": es[r["head"]]["name"], "type": r["type"], "tail": es[r["tail"]]["name"],
                        "reference_positive": rid in reference, "selected": rid in selected,
                        "in_own_supervision": rid in (set(c["control_target_positive_ids"]) if arm == "control" else reference) if arm != "base" else None}
                        for rid, r in relations.items()]})
            results[arm] = {"cases": values, "train_reference_selected": sum(len(v["selected_reference_ids"]) for v in values if v["split"] == "train"),
                "validation_reference_selected": sum(len(v["selected_reference_ids"]) for v in values if v["split"] == "validation"),
                "new_positives_selected": sum(len(v["selected_new_positive_ids"]) for v in values)}
    finally:
        ev.Provider.from_env = factory
    assert ev.inspect_pair(root, train) == manifest, "Training artifacts changed"
    summary = {"status": "complete", "runs": results, "model_requests": 18, "training_started": False,
        "new_checkpoint_selected": False, "three_seed_eligible": False, "dev_gate_replaced": False,
        "reference_status": "existing assistant-reviewed/reused supervision, not new independent human gold",
        "scope": "training fit and single internal heldout source diagnostic; not independent generalization"}
    ev.save(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "requests": 18,
        "runs": {a: {k: v for k, v in r.items() if k != "cases"} for a, r in results.items()}}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    manifest = ev.inspect_pair(ROOT, args.train)
    assert importlib.metadata.version("ms-swift") == "4.4.2"
    for name, digest in manifest["verification"]["manifest"]["tokenizer_config_sha256"].items():
        assert ev.sha(Path(manifest["arms"]["control"]["command"][3]) / name) == digest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("WORLD_SIZE", "1") == "1"
    command = deploy_command(ROOT, manifest, 8018)
    from swift.arguments import DeployArguments
    supported = {f.name for f in fields(DeployArguments)}
    assert all(n[2:] in supported for n in command if n.startswith("--"))
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), "GPU0 unavailable or occupied"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8018))
    args.out.mkdir(parents=True)
    ev.save(args.out / "launch.json", {"command": command, "training_manifest_sha256": ev.EXPECTED_PAIR_SHA,
        "inference_context_limit": 16384, "training_length_limit": 8192, "not_a_dev_gate_run": True})
    try:
        with ev.owned_service(command, args.out / "deploy.log", "http://127.0.0.1:8018/v1"):
            run_probe(ROOT, args.train, args.out / "probe")
    except BaseException as error:
        ev.save(args.out / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__": main()
