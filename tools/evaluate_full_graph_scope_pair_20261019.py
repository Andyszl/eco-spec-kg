"""Full dev22 for the scoped seed42 pair; common structure and separate frozen clause postfilter."""
import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_full_graph_scope_pair_20261019 as training
from ecospec_kg import candidate_profile_v214 as profile
from cross_source_structural_metrics_20261016 import compare_one
from ecospec_kg import extractor_v2 as ex
from ecospec_kg import full_graph_scope_v219 as scope
from ecospec_kg.method_clause_guard_v218 import filter_prediction, VERSION as CLAUSE_VERSION
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.io_utils import read_jsonl, write_jsonl
from ecospec_kg.providers import OpenAICompatibleProvider as Provider
from ecospec_kg.prediction_validation_v2 import validate_predictions_v2
from ecospec_kg.evaluation_v2 import evaluate_v2
from ecospec_kg.selection_decoding_v2 import VERSION, implementation_hashes

PROTOCOL = "experiments/full_graph_scope_dev_20261019/protocol.json"
EXPECTED_PAIR_SHA = "4c10145133573de50925d859f10152ef04c7396cec0f4a55d855ce21ad6186df"
MODELS = {"base": "eco-pair-base", "control": "eco-pair-control", "repair": "eco-pair-repair"}


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def inspect_pair(root, train):
    assert sha(train / "pair_manifest.json") == EXPECTED_PAIR_SHA, "Wrong or changed training manifest"
    manifest = read(train / "pair_manifest.json")
    lock = read(root / PROTOCOL)
    assert lock["training_manifest_sha256"] == EXPECTED_PAIR_SHA
    assert manifest == lock["training_manifest"], "Training provenance changed"
    assert manifest["status"] == "complete" and manifest["seed"] == 42 and manifest["max_steps_per_arm"] == 200
    assert manifest["preflight_sha256"] == lock["preflight_sha256"], "Preflight changed"
    assert not manifest["expansion_to_three_seeds"] and not manifest["dev_evaluation_completed"]
    assert manifest["training_code_commit"] == "669108d799e373c95b9b36ff632d13a22a31198e"
    assert manifest["candidate_profile"] == profile.VERSION
    assert manifest["decision_scope_version"] == scope.VERSION == lock["decision_scope_version"]
    assert manifest["question_contract_changed"] and not manifest["token_level_loss_masking"]
    assert manifest["clause_guard_used_in_training"] is False
    assert manifest["control_is_incomplete_supervision_ablation"] is True
    assert manifest["verification"] == training.verify_approved(root, root / training.REPORT), "Supervision/input audit changed"
    for name, digest in lock["evaluation_input_sha256"].items():
        assert sha(root / name) == digest, "Evaluation input changed: " + name
    for name, digest in lock["source_sha256_lf"].items():
        assert hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, f"Changed source: {name}"
    for arm in ("control", "repair"):
        saved = manifest["arms"][arm]
        assert saved["status"] == "complete" and Path(saved["output"]).resolve() == (train / arm).resolve()
        cmd = saved["command"]
        training_root = Path(cmd[cmd.index("--dataset") + 1]).parents[2]
        assert cmd == training.build_command(training_root, Path(cmd[3]), Path(saved["output"]), arm), "Training command changed"
        assert training.inspect_completion(train / arm) == saved["completion"], f"Changed {arm} completion/adapter"
    return manifest


def deploy_command(root, manifest, port):
    specs = [MODELS[a] + "=" + manifest["arms"][a]["completion"]["best_checkpoint"] for a in ("control", "repair")]
    model = manifest["arms"]["control"]["command"][3]
    assert model == manifest["arms"]["repair"]["command"][3]
    return ["swift", "deploy", "--model", model, "--adapters", *specs,
            "--infer_backend", "transformers", "--served_model_name", MODELS["base"],
            "--torch_dtype", "bfloat16", "--enable_thinking", "false", "--temperature", "0",
            "--max_new_tokens", "3072", "--max_length", "8192", "--seed", "42", "--max_batch_size", "1",
            "--external_plugins", str(root / "tools/swift_unique_selection_plugin.py"),
            "--host", "127.0.0.1", "--port", str(port), "--api_key", "local-token", "--verbose", "false"]


def get_json(base_url, endpoint):
    req = urllib.request.Request(base_url + endpoint, headers={"Authorization": "Bearer local-token"})
    with urllib.request.urlopen(req, timeout=10) as reply:
        return json.load(reply)


def check_service(base_url):
    assert {x["id"] for x in get_json(base_url, "/models")["data"]} == set(MODELS.values()), "Wrong service models"
    cap = get_json(base_url, "/ecospec-selection-decoding")
    assert (cap.get("version"), cap.get("backend"), cap.get("swift_version")) == (VERSION, "transformers", "4.4.2"), "Wrong service plugin"
    assert cap.get("implementation") == implementation_hashes(), "Wrong service implementation"
    return cap


def strict_gate(control, repair):
    checks = {}
    for stage in ("before", "after"):
        c, r = control[stage], repair[stage]
        for kind in ("entities", "relations"):
            checks[f"{stage}_{kind}_f1_non_decrease"] = r[kind]["f1"] >= c[kind]["f1"]
            checks[f"{stage}_{kind}_fp_non_increase"] = r[kind]["false_positive"] <= c[kind]["false_positive"]
        checks[f"{stage}_macro_non_decrease"] = r["macro_relation_f1"] >= c["macro_relation_f1"]
    edges = repair["indicator_formula_edges"]
    checks["three_target_bindings_after_structure"] = len(edges) == 3 and all(e["after"] for e in edges)
    checks["expanded_method_source_path_after_structure"] = repair["formula_paths_after"]["expanded_complete_evidenced_paths"] >= 1
    return {"comparison": "repair versus control, both without and with the same structural module",
            "checks": checks, "passed": all(checks.values()), "failed_checks": [k for k, v in checks.items() if not v],
            "three_seed_eligible": all(checks.values()), "three_seed_training_started": False,
            "not_independent_test": True}


def extract_profile(units, dest, config):
    """Use the existing audited selector/decoder, supplying the full canonical graph."""
    dest.mkdir(parents=True)
    save(dest / "resolved_config.json", config)
    provider = Provider.from_env()
    for name in ("model", "seed", "temperature", "max_tokens", "enable_thinking", "selection_decoding"):
        setattr(provider, name, config[name])
    predictions, raw_responses = [], []
    try:
        for unit in units:
            graph = profile.candidates_for_unit(unit)
            prediction, raw, prompt_hash = scope.select(provider, unit, graph)
            raw_responses.append({"unit_id": unit["unit_id"], "prompt_hash": prompt_hash,
                "response_sha256": sha256_json(raw), "raw_response": raw,
                "provider_response": provider.last_raw_response})
            prediction.update(experiment_id=config["experiment_id"], config_hash=sha256_json(config),
                candidate_hash=sha256_json({k:graph[k] for k in ("entities","relations")}),
                model=config["model"], seed=config["seed"], prompt_hash=prompt_hash)
            assert prediction["status"] == "success"
            predictions.append(prediction)
    except BaseException as error:
        save(dest / "extraction_failure.json", {"type":type(error).__name__,"message":str(error),
            "unit_id":unit["unit_id"],"provider_response":provider.last_raw_response})
        raise
    finally:
        write_jsonl(dest / "raw_responses.jsonl", raw_responses)
        write_jsonl(dest / "predictions.jsonl", sorted(predictions, key=lambda p:p["unit_id"]))
    result={"status":"complete","summary":{"unit_count":len(units),"success_count":len(predictions),"error_count":0}}
    save(dest / "manifest.json", {**result,"candidate_profile":profile.VERSION,"config":config,
        "predictions_sha256":sha(dest/"predictions.jsonl"),"raw_responses_sha256":sha(dest/"raw_responses.jsonl")})
    return result


def clause_guard_report(units, gold, config, predictions, dest):
    """Same frozen postfilter for every model; never changes the primary gate."""
    dest.mkdir()
    by_id = {u["unit_id"]: u for u in units}
    filtered, filter_proofs = [], []
    for prediction in predictions:
        new, proof = filter_prediction(prediction)
        assert not _validate_unit_prediction(by_id[new["unit_id"]], new), "Filtered prediction invalid"
        filtered.append(new)
        filter_proofs.append({"unit_id": prediction["unit_id"], **proof})
    write_jsonl(dest / "predictions.jsonl", filtered)
    write_jsonl(dest / "filter_proofs.jsonl", filter_proofs)
    report, merged, proofs = compare_one(units, gold, {**config, "postprocessor": CLAUSE_VERSION}, filtered)
    report.update(postprocessor=CLAUSE_VERSION, postprocessor_model_requests=0,
                  removed_entities=sum(len(p["removed_entity_ids"]) for p in filter_proofs),
                  removed_relations=sum(len(p["removed_relation_ids"]) for p in filter_proofs))
    hybrid = dest / "hybrid"
    hybrid.mkdir()
    write_jsonl(hybrid / "predictions.jsonl", merged)
    write_jsonl(hybrid / "structural_proofs.jsonl", proofs)
    save(dest / "metrics_and_changes.json", report)
    return report


def evaluate(root, train, out, base_url="http://127.0.0.1:8018/v1"):
    assert not out.exists(), "Evaluation output exists"
    manifest = inspect_pair(root, train)
    cap = check_service(base_url)
    assert ex.CANDIDATE_GENERATOR_VERSION == "structure-aware-rule-v2.12"
    assert ex.SELECTION_POLICY_VERSION == "ecospec-selection-v2.6"
    assert Path(ex.__file__).resolve() == (root / "src/ecospec_kg/extractor_v2.py").resolve()
    data = root / "deliveries/human_confirmed_20261002"
    units = read_jsonl(data / "blind/dev_units.jsonl")
    gold = read_jsonl(data / "gold/dev_annotations.jsonl")
    assert len(units) == len(gold) == 22
    provenance = {}
    for unit in units:
        c = profile.candidates_for_unit(unit)
        system, prompt = scope.build_messages(unit, c)
        provenance[unit["unit_id"]] = {"candidate_hash": sha256_json({k: c[k] for k in ("entities", "relations")}),
                                      "prompt_hash": sha256_json({"system": system, "prompt": prompt}), "seed": 42}
    assert len(provenance) == 22
    cfg = {"experiment_id": "full-graph-scope-seed42-pair-dev22", "backend": "llm", "model": MODELS["base"],
           **read(root / PROTOCOL)["selection_protocol"], "selection_decoding": VERSION,
           "decision_scope_version": scope.VERSION, "question_contract_changed": True,
           "use_schema": True, "use_layout": True, "use_evidence": True}
    out.mkdir(parents=True)
    save(out / "evaluation_config.json", cfg)
    save(out / "evaluation_protocol.json", {"training_manifest_sha256": sha(train / "pair_manifest.json"),
         "training_run": str(train), "arms": manifest["arms"], "capability": cap, "prompt_provenance": provenance,
         "protocol_sha256": sha(root / PROTOCOL), "code_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
         "scope": "fixed full dev22 pilot; not independent; structure and frozen clause postfilter reported separately"})
    factory = Provider.__dict__["from_env"]
    results, clause_results = {}, {}
    try:
        Provider.from_env = classmethod(lambda cls: Provider(base_url=base_url, api_key="local-token", timeout=600, enable_thinking=False))
        for label, model in MODELS.items():
            dest = out / label
            print(f"Evaluating {label}: full dev22, model={model}", flush=True)
            m = extract_profile(units, dest, {**cfg, "model": model})
            assert m["status"] == "complete" and m["summary"]["success_count"] == m["summary"]["unit_count"] == 22 and m["summary"]["error_count"] == 0, "Extraction failures; no gate"
            actual_cfg = read(dest / "resolved_config.json")
            assert actual_cfg == {**cfg, "model": model}, "Inference settings changed"
            predictions = read_jsonl(dest / "predictions.jsonl")
            assert len(predictions) == 22 and {p["unit_id"] for p in predictions} == set(provenance)
            assert all(p[k] == v for p in predictions for k, v in provenance[p["unit_id"]].items()), "Prompts/candidates changed"
            v = validate_predictions_v2(data / "blind/dev_units.jsonl", dest / "predictions.jsonl", data / "schema_v2.json", dest / "validation")
            assert v["passed"] and v["valid_prediction_unit_count"] == 22 and v["failure_count"] == 0
            metric = evaluate_v2(data / "gold/dev_annotations.jsonl", dest / "validation/validated_predictions.jsonl", data / "blind/dev_units.jsonl", dest / "validation/validation_report.json", dest / "evaluation")
            report, merged, proofs = compare_one(units, gold, actual_cfg, predictions)
            assert report["before"]["entities"] == metric["entities"]["micro"]
            assert report["before"]["relations"] == metric["relations"]["strict_micro"]
            assert report["before"]["macro_relation_f1"] == metric["relations"]["macro_relation_f1"]
            hybrid = dest / "hybrid"
            hybrid.mkdir()
            write_jsonl(hybrid / "predictions.jsonl", merged)
            write_jsonl(hybrid / "structural_proofs.jsonl", proofs)
            save(hybrid / "metrics_and_changes.json", report)
            results[label] = report
            clause_results[label] = clause_guard_report(units, gold, actual_cfg, predictions, dest / "clause_guard")
            print(json.dumps({"model": label, "before": report["before"], "after": report["after"]}, ensure_ascii=False), flush=True)
    finally:
        Provider.from_env = factory
    assert inspect_pair(root, train) == manifest, "Training artifacts changed"
    secondary_gate = strict_gate(clause_results["control"], clause_results["repair"])
    secondary_gate.update(decision_role="secondary_diagnostic_only", three_seed_eligible=False)
    summary = {"status": "complete", "runs": results, "gate": strict_gate(results["control"], results["repair"]),
               "decision_scope_version": scope.VERSION, "clause_guard_runs": clause_results,
               "clause_guard_gate": secondary_gate, "clause_guard_model_requests": 0,
               "primary_gate_uses_clause_guard": False,
               "model_requests": 66, "training_started": False, "output": str(out),
               "historical_results_unchanged": True, "scope": "full dev22 paired diagnostic, not independent validation"}
    save(out / "summary.json", summary)
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2), flush=True)
    return summary


@contextmanager
def owned_service(command, log, base_url):
    # Only this newly created child process group is stopped; never existing GPU tasks.
    with log.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline, next_notice = time.monotonic() + 900, 0
            while True:
                assert child.poll() is None, f"Deployment exited; see {log}"
                try:
                    check_service(base_url)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    assert time.monotonic() < deadline, f"Deployment timeout; see {log}"
                    if time.monotonic() >= next_notice:
                        print(f"Waiting for model service; log={log}", flush=True)
                        next_notice = time.monotonic() + 30
                    time.sleep(1)
            yield
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists(), "Output exists"
    manifest = inspect_pair(ROOT, args.train)
    assert importlib.metadata.version("ms-swift") == "4.4.2"
    for name, digest in manifest["verification"]["manifest"]["tokenizer_config_sha256"].items():
        assert sha(Path(manifest["arms"]["control"]["command"][3]) / name) == digest, "Tokenizer config changed: " + name
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("WORLD_SIZE", "1") == "1"
    command = deploy_command(ROOT, manifest, 8018)
    from swift.arguments import DeployArguments
    supported = {f.name for f in fields(DeployArguments)}
    assert all(x[2:] in supported for x in command if x.startswith("--")), "Unsupported deploy option"
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), f"GPU0 unavailable or occupied: {gpu.stdout} {gpu.stderr}"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8018))
    args.out.mkdir(parents=True)
    save(args.out / "launch.json", {"command": command, "training_manifest_sha256": EXPECTED_PAIR_SHA})
    try:
        with owned_service(command, args.out / "deploy.log", "http://127.0.0.1:8018/v1"):
            evaluate(ROOT, args.train, args.out / "evaluation")
    except BaseException as error:
        save(args.out / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
