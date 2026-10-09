"""Decoding controls must reach the request without changing legacy runs."""
import io
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from test_experiment_chain_v2 import source_unit
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.extractor_v2 import extract_v2
from ecospec_kg.io_utils import read_json, read_jsonl, write_json, write_jsonl
from ecospec_kg.providers import CompletionTruncatedError, OpenAICompatibleProvider


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def completion(content=None, finish_reason="stop"):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content or
            '{"selected_entity_ids":[],"selected_relation_ids":[]}'}}],
        "usage": {"completion_tokens": 3072 if finish_reason == "length" else 18},
    }


def test_optional_penalty_reaches_request_and_run_provenance(tmp_path):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    payloads = []

    def send(request, **_):
        payloads.append(json.loads(request.data))
        return Reply(json.dumps(completion()).encode())

    config = {"backend": "llm", "seed": 42, "temperature": 0, "max_tokens": 3072,
              "enable_thinking": False}
    with patch("urllib.request.urlopen", side_effect=send):
        for name, extra in (("original", {}), ("controlled", {"repetition_penalty": 1.1})):
            write_json(tmp_path / "config.json", {**config, **extra})
            run = extract_v2(tmp_path / "units.jsonl", tmp_path / name,
                             config_path=tmp_path / "config.json")
            assert run["status"] == "complete"
    assert "repetition_penalty" not in payloads[0]
    assert payloads[1].pop("repetition_penalty") == 1.1
    assert payloads[0] == payloads[1]
    old = read_jsonl(tmp_path / "original/predictions.jsonl")[0]
    new = read_jsonl(tmp_path / "controlled/predictions.jsonl")[0]
    assert (old["candidate_hash"], old["prompt_hash"]) == (new["candidate_hash"], new["prompt_hash"])
    assert old["config_hash"] != new["config_hash"]
    resolved = read_json(tmp_path / "controlled/resolved_config.json")
    assert resolved["repetition_penalty"] == 1.1
    assert new["config_hash"] == sha256_json(resolved)
    assert "repetition_penalty" not in read_json(tmp_path / "original/resolved_config.json")


@pytest.mark.parametrize("value", [True, "1.1", None, 0, -1, float("nan"), float("inf")])
def test_invalid_penalty_is_rejected_before_network_or_output(tmp_path, value):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "repetition_penalty": value})
    with patch("urllib.request.urlopen") as send:
        with pytest.raises(ValueError, match="repetition_penalty"):
            extract_v2(tmp_path / "units.jsonl", tmp_path / "out",
                       config_path=tmp_path / "config.json")
    send.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_penalty_does_not_accept_or_repair_truncated_repetition():
    provider = OpenAICompatibleProvider("http://localhost/v1", "local-token")
    provider.repetition_penalty = 1.1
    raw = completion('{"selected_relation_ids":["abc","abc",', "length")
    with patch("urllib.request.urlopen", return_value=Reply(json.dumps(raw).encode())):
        with pytest.raises(CompletionTruncatedError):
            provider.complete("system", "prompt")
    assert provider.last_raw_response == raw


@pytest.fixture
def probe_module():
    path = Path(__file__).resolve().parents[1] / "tools/probe_repeated_selection.py"
    spec = importlib.util.spec_from_file_location("probe_repeated_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def previous_run(tmp_path):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "model": "eco-lora-v28-seed44",
        "max_tokens": 3072, "enable_thinking": False})
    raw = completion('{"selected_entity_ids":[],"selected_relation_ids":["abc","abc",', "length")
    with patch("urllib.request.urlopen", return_value=Reply(json.dumps(raw).encode())):
        run = extract_v2(tmp_path / "units.jsonl", tmp_path / "old",
                         config_path=tmp_path / "config.json")
    assert run["summary"]["error_count"] == 1
    return tmp_path / "old"


def probe_http(response, listed=True):
    calls = []

    def send(request, timeout):
        calls.append((request, timeout))
        if request.full_url.endswith("/models"):
            models = [{"id": "eco-lora-v28-seed44"}] if listed else []
            return Reply(json.dumps({"data": models}).encode())
        return Reply(json.dumps(response).encode())

    return send, calls


def test_probe_replays_same_prompt_once_and_preserves_history(probe_module, previous_run, tmp_path):
    before = {p: p.read_bytes() for p in previous_run.iterdir() if p.is_file()}
    send, calls = probe_http(completion())
    with patch("urllib.request.urlopen", side_effect=send):
        result = probe_module.probe(previous_run, tmp_path / "probe")
    assert result["probe_passed"] is True
    assert result["formal_evaluation"] is False
    assert len(calls) == 2 and calls[-1][1] == 600
    payload = json.loads(calls[-1][0].data)
    assert payload["repetition_penalty"] == 1.1
    assert (payload["seed"], payload["temperature"], payload["max_tokens"]) == (42, 0, 3072)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"] == read_json(tmp_path / "probe/messages.json")
    assert result["prompt_hash"] == read_jsonl(previous_run / "predictions.jsonl")[0]["prompt_hash"]
    assert read_json(tmp_path / "probe/provider_response.json") == completion()
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        probe_module.probe(previous_run, tmp_path / "probe")


@pytest.mark.parametrize("content,reason,error", [
    ('{"selected_entity_ids":[],"selected_relation_ids":["abc","abc",', "length", "CompletionTruncatedError"),
    ('{"selected_entity_ids":[],"selected_relation_ids":["unknown-id"]}', "stop", "ValueError"),
    ('{"selected_entity_ids":[],"selected_relation_ids":[', "stop", "JSONDecodeError"),
    ('{"selected_entity_ids":"wrong-type","selected_relation_ids":[]}', "stop", "ValueError"),
])
def test_probe_keeps_failed_response_without_salvaging(probe_module, previous_run, tmp_path, content, reason, error):
    raw = completion(content, reason)
    send, calls = probe_http(raw)
    with patch("urllib.request.urlopen", side_effect=send):
        result = probe_module.probe(previous_run, tmp_path / "probe")
    assert result["probe_passed"] is False
    assert result["error_type"] == error
    assert len(calls) == 2
    assert read_json(tmp_path / "probe/provider_response.json") == raw
    assert not (tmp_path / "probe/evaluation").exists()


def test_probe_rejects_duplicate_real_ids(probe_module, previous_run, tmp_path):
    from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
    relation_id = RuleCandidateExtractorV2().predict_unit(source_unit())["relations"][0]["relation_id"]
    raw = completion(json.dumps({"selected_entity_ids": [], "selected_relation_ids": [relation_id, relation_id]}))
    send, _ = probe_http(raw)
    with patch("urllib.request.urlopen", side_effect=send):
        result = probe_module.probe(previous_run, tmp_path / "probe")
    assert result["probe_passed"] is False
    assert "duplicate" in result["error_message"]


def test_probe_preserves_valid_selection_and_empty_think_block(probe_module, previous_run, tmp_path):
    from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2
    candidate = RuleCandidateExtractorV2().predict_unit(source_unit())
    selection = {"selected_entity_ids": [candidate["entities"][0]["entity_id"]],
                 "selected_relation_ids": [candidate["relations"][0]["relation_id"]]}
    raw = completion("<think>\n\n</think>\n\n" + json.dumps(selection))
    send, _ = probe_http(raw)
    with patch("urllib.request.urlopen", side_effect=send):
        result = probe_module.probe(previous_run, tmp_path / "probe")
    assert result["probe_passed"] is True
    assert read_json(tmp_path / "probe/selection.json") == selection
    assert read_json(tmp_path / "probe/provider_response.json") == raw


@pytest.mark.parametrize("kind", ["timeout", "http"])
def test_probe_records_transport_failure(probe_module, previous_run, tmp_path, kind):
    send, _ = probe_http(completion())
    error = (TimeoutError("timed out") if kind == "timeout" else
             HTTPError("http://localhost/v1/chat/completions", 400, "Bad Request", {},
                       io.BytesIO(b'{"error":"unsupported parameter"}')))

    def fail(request, timeout):
        if request.full_url.endswith("/models"):
            return send(request, timeout)
        raise error

    with patch("urllib.request.urlopen", side_effect=fail):
        result = probe_module.probe(previous_run, tmp_path / "probe")
    assert result["probe_passed"] is False
    assert result["error_type"] == ("TimeoutError" if kind == "timeout" else "CompletionHTTPError")
    if kind == "http":
        assert read_json(tmp_path / "probe/provider_response.json")["http_status"] == 400
    else:
        assert not (tmp_path / "probe/provider_response.json").exists()


def test_probe_wrong_adapter_stops_before_generation(probe_module, previous_run, tmp_path):
    send, calls = probe_http(completion(), listed=False)
    with patch("urllib.request.urlopen", side_effect=send):
        with pytest.raises(ValueError, match="adapter"):
            probe_module.probe(previous_run, tmp_path / "probe")
    assert len(calls) == 1 and not (tmp_path / "probe").exists()


@pytest.mark.parametrize("file", ["predictions.jsonl", "raw_responses.jsonl", "resolved_config.json", "units.jsonl"])
def test_probe_changed_saved_input_stops_before_network(probe_module, previous_run, tmp_path, file):
    path = tmp_path / file if file == "units.jsonl" else previous_run / file
    path.write_bytes(path.read_bytes() + b" ")
    with patch("urllib.request.urlopen") as send:
        with pytest.raises(ValueError, match="hash"):
            probe_module.probe(previous_run, tmp_path / "probe")
    send.assert_not_called()
    assert not (tmp_path / "probe").exists()


def test_probe_changed_prompt_stops_before_network(probe_module, previous_run, tmp_path):
    original = probe_module.build_llm_selection_messages

    def changed(unit, candidates):
        system, prompt = original(unit, candidates)
        return system + "changed", prompt

    with patch.object(probe_module, "build_llm_selection_messages", side_effect=changed), \
            patch("urllib.request.urlopen") as send:
        with pytest.raises(ValueError, match="prompt hash"):
            probe_module.probe(previous_run, tmp_path / "probe")
    send.assert_not_called()
