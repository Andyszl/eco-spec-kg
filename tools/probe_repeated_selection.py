"""Replay one truncated selection with an explicit decoding change; never score it."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ecospec_kg.experiment_io_v2 import (
    assert_blind_records, runtime_metadata, sha256_json, sha256_path, utc_now,
)
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, build_llm_selection_messages
from ecospec_kg.io_utils import read_json, read_jsonl, write_json
from ecospec_kg.providers import OpenAICompatibleProvider


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_selection(text, candidates):
    text = text.strip()
    if text.startswith("<think>"):
        _, closing, text = text.partition("</think>")
        require(closing, "unclosed think block")
    selection = json.loads(text.strip())
    keys = {"selected_entity_ids": "entities", "selected_relation_ids": "relations"}
    require(isinstance(selection, dict) and set(selection) == set(keys), "unexpected selection fields")
    for key, field in keys.items():
        ids = selection[key]
        require(isinstance(ids, list) and all(isinstance(i, str) for i in ids), f"{key} must be a string array")
        require(len(ids) == len(set(ids)), f"duplicate ids in {key}")
        id_field = "entity_id" if field == "entities" else "relation_id"
        require(set(ids) <= {r[id_field] for r in candidates[field]}, f"unknown candidate ids in {key}")
    return selection


def probe(previous_run: Path, out: Path, *, repetition_penalty=1.1, timeout=600):
    previous_run, out = previous_run.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    require(type(repetition_penalty) in (int, float) and math.isfinite(repetition_penalty)
            and repetition_penalty > 1, "probe repetition_penalty must be finite and greater than 1")
    require(type(timeout) is int and timeout > 0, "timeout must be a positive integer")
    previous = read_json(previous_run / "run_manifest.json")
    for filename, key in (("predictions.jsonl", "predictions"), ("raw_responses.jsonl", "raw_responses"),
                          ("resolved_config.json", "config")):
        require(sha256_path(previous_run / filename) == previous[key]["sha256"], f"saved {filename} hash mismatch")
    units_path = Path(previous["input"]["path"])
    require(sha256_path(units_path) == previous["input"]["sha256"], "source units hash mismatch")
    rows = read_jsonl(previous_run / "predictions.jsonl")
    units = read_jsonl(units_path)
    bad = [r for r in rows if r.get("status") != "success"]
    require(previous["status"] == "completed_with_errors" and previous["summary"]["error_count"] == len(bad) == 1,
            "expected exactly one remaining failed unit")
    require(len(rows) == previous["summary"]["unit_count"] == len(units)
            and {r["unit_id"] for r in rows} == {u["unit_id"] for u in units}, "saved unit counts mismatch")
    failed = bad[0]
    require(failed.get("error_type") == "CompletionTruncatedError", "remaining failure is not output truncation")
    matches = [u for u in units if u["unit_id"] == failed["unit_id"]]
    require(len(matches) == 1, "failed source unit is not unique")
    unit = matches[0]
    assert_blind_records([unit])
    config = read_json(previous_run / "resolved_config.json")
    require(config["backend"] == "llm" and config.get("enable_thinking") is False,
            "expected a non-thinking LLM run")
    require("repetition_penalty" not in config, "previous run already changed repetition control")
    src = REPO / "src/ecospec_kg"
    for name, expected in previous["implementation"].items():
        if name not in {"providers.py", "extractor_v2.py"}:
            require(sha256_path(src / name) == expected, f"implementation hash mismatch: {name}")
    candidates = RuleCandidateExtractorV2().predict_unit(unit)
    system, prompt = build_llm_selection_messages(unit, candidates)
    candidate_hash = sha256_json({k: candidates[k] for k in ("entities", "relations")})
    prompt_hash = sha256_json({"system": system, "prompt": prompt})
    require(candidate_hash == failed["candidate_hash"] and prompt_hash == failed["prompt_hash"],
            "candidate or prompt hash mismatch")
    require(sha256_json(config) == failed["config_hash"], "resolved config hash mismatch")

    provider = OpenAICompatibleProvider.from_env()
    require(not provider.append_no_think, "unset ECOSPEC_LLM_APPEND_NO_THINK before replay")
    provider.model, provider.seed = config["model"], config["seed"]
    provider.temperature, provider.max_tokens = config["temperature"], config["max_tokens"]
    provider.enable_thinking, provider.repetition_penalty, provider.timeout = False, repetition_penalty, timeout
    request = urllib.request.Request(provider.base_url.rstrip("/") + "/models",
                                    headers={"Authorization": f"Bearer {provider.api_key}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        ids = [r["id"] for r in json.load(response)["data"]]
    require(provider.model in ids, f"target adapter is not listed: {provider.model}")

    out.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    write_json(out / "messages.json", messages)
    write_json(out / "resolved_config.json", {**config, "repetition_penalty": repetition_penalty})
    result = {
        "kind": "single_unit_decoding_probe", "formal_evaluation": False, "probe_passed": False,
        "started_at": utc_now(), "unit_id": unit["unit_id"], "unit_type": unit["unit_type"],
        "previous_run": str(previous_run), "previous_manifest_sha256": sha256_path(previous_run / "run_manifest.json"),
        "input": previous["input"], "candidate_hash": candidate_hash, "prompt_hash": prompt_hash,
        "original_config_sha256": previous["config"]["sha256"],
        "config_sha256": sha256_path(out / "resolved_config.json"),
        "decoding_change": {"repetition_penalty": repetition_penalty}, "request_timeout_seconds": timeout,
        "runtime": runtime_metadata(REPO),
        "implementation": {n: sha256_path(src / n) for n in previous["implementation"]},
        "probe_tool_sha256": sha256_path(Path(__file__)), "out": str(out),
    }
    write_json(out / "probe_manifest.json", result)
    print(f"Single-unit probe; repetition_penalty={repetition_penalty}; output: {out}", flush=True)
    try:
        text = provider.complete(system, prompt)
        (out / "response.txt").write_text(text, encoding="utf-8")
        selection = checked_selection(text, candidates)
        require(provider.last_raw_response["choices"][0].get("finish_reason") == "stop", "response did not finish normally")
        write_json(out / "selection.json", selection)
        result.update(probe_passed=True, selected_entity_count=len(selection["selected_entity_ids"]),
                      selected_relation_count=len(selection["selected_relation_ids"]))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error_message=str(exc))
    finally:
        raw = provider.last_raw_response
        if raw is not None:
            write_json(out / "provider_response.json", raw)
            result["provider_response_sha256"] = sha256_path(out / "provider_response.json")
            result["usage"] = raw.get("usage")
            result["finish_reason"] = (raw.get("choices") or [{}])[0].get("finish_reason")
        result["finished_at"] = utc_now()
        write_json(out / "probe_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = probe(args.previous_run, args.out, repetition_penalty=args.repetition_penalty, timeout=args.timeout)
    summary_keys = ("probe_passed", "unit_id", "decoding_change", "finish_reason", "usage",
                    "selected_entity_count", "selected_relation_count", "error_type", "error_message", "out")
    print(json.dumps({k: result[k] for k in summary_keys if k in result}, ensure_ascii=False, indent=2))
    return 0 if result["probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
