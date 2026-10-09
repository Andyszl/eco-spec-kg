"""Constrain generation before a repeated ID can consume the output budget."""
import asyncio
from dataclasses import dataclass
import importlib
import importlib.util
import io
import itertools
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

from test_experiment_chain_v2 import source_unit
from ecospec_kg.extractor_v2 import RuleCandidateExtractorV2, build_llm_selection_messages, extract_v2
from ecospec_kg.io_utils import read_json, read_jsonl, write_json, write_jsonl
from ecospec_kg.providers import OpenAICompatibleProvider
from test_repetition_control_v2 import previous_run, Reply

VERSION = "unique-candidate-json-v1"
E1, E2, R1, R2 = (c * 16 for c in "1234")


@pytest.fixture
def decoding():
    return importlib.import_module("ecospec_kg.selection_decoding_v2")


class AsciiTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids, **_):
        return "".join(chr(i) for i in ids if i != 999)


def constrained(decoding, entities=(E1, E2), relations=(R1, R2)):
    return decoding.UniqueSelectionConstraint(AsciiTokenizer(), entities, relations, [999])


def path_ids(decoder, entities, relations):
    # Chunks are deliberate: a segment tokenizer may differ from whole-string BPE.
    text = '{"selected_entity_ids": ['
    text += ", ".join(json.dumps(i) for i in entities)
    text += '], "selected_relation_ids": ['
    text += ", ".join(json.dumps(i) for i in relations) + "]}"
    return decoder.tokenizer.encode(text)


def assert_path(decoder, ids):
    for n, token in enumerate(ids + [999]):
        assert token in decoder.allowed(ids[:n])


def test_constrained_config_requires_server_proof_before_success(tmp_path):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "enable_thinking": False,
        "selection_decoding": VERSION, "max_tokens": 3072})
    response = {"choices": [{"finish_reason": "stop", "message": {"content":
        '{"selected_entity_ids":[],"selected_relation_ids":[]}'}}]}
    calls = []

    class Reply(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *_): self.close()

    def send(request, **_):
        calls.append(json.loads(request.data))
        return Reply(json.dumps(response).encode())

    with patch("urllib.request.urlopen", side_effect=send):
        manifest = extract_v2(tmp_path / "units.jsonl", tmp_path / "out",
                              config_path=tmp_path / "config.json")
    assert manifest["summary"]["error_count"] == 1
    assert calls[0]["chat_template_kwargs"]["ecospec_selection_decoding"] == VERSION
    assert read_jsonl(tmp_path / "out/raw_responses.jsonl")[0]["provider_response"] == response


def test_full_extraction_preserves_server_proof_and_decoder_provenance(decoding, tmp_path):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "enable_thinking": False,
        "selection_decoding": VERSION, "max_tokens": 3072})
    system, prompt = wire_prompt()
    proof = {**decoding.request_identity(system, prompt), "prefix_calls": 54, "max_completion_tokens": 200}
    response = {"choices": [{"finish_reason": "stop", "message": {"content":
        '{"selected_entity_ids": [], "selected_relation_ids": []}'}}], "ecospec_selection_decoding": proof}
    with patch("urllib.request.urlopen", return_value=Reply(json.dumps(response).encode())):
        manifest = extract_v2(tmp_path / "units.jsonl", tmp_path / "out", config_path=tmp_path / "config.json")
    assert manifest["status"] == "complete"
    assert manifest["decoding_implementation"] == decoding.implementation_hashes()
    assert "swift_unique_selection_plugin.py" not in manifest["implementation"]
    assert manifest["implementation"]["selection_decoding_v2.py"] == decoding.implementation_hashes()["selection_decoding_v2.py"]
    assert read_jsonl(tmp_path / "out/raw_responses.jsonl")[0]["provider_response"] == response
    assert read_json(tmp_path / "out/resolved_config.json")["selection_decoding"] == VERSION


def test_all_subsets_and_orders_remain_selectable(decoding):
    decoder = constrained(decoding)
    options = [()] + list(itertools.permutations((E1, E2), 1)) + list(itertools.permutations((E1, E2)))
    relations = [()] + list(itertools.permutations((R1, R2), 1)) + list(itertools.permutations((R1, R2)))
    for es, rs in itertools.product(options, relations):
        assert_path(decoder, path_ids(decoder, es, rs))


def test_repetition_preference_is_finite_and_cannot_repeat_ids(decoding):
    decoder = constrained(decoding)
    ids = []
    # Prefers quote/comma over closing brackets, hence tries to keep listing IDs.
    for _ in range(decoder.max_tokens):
        next_token = min(decoder.allowed(ids))
        if next_token == 999:
            break
        ids.append(next_token)
    else:
        pytest.fail("generation did not terminate within its calculated bound")
    result = json.loads(decoder.tokenizer.decode(ids))
    assert result == {"selected_entity_ids": [E1, E2], "selected_relation_ids": [R1, R2]}
    assert len(ids) + 1 == decoder.max_tokens
    with pytest.raises(ValueError, match="prefix"):
        decoder.allowed(path_ids(decoder, (E1, E1), ()))
    with pytest.raises(ValueError, match="prefix"):
        decoder.allowed(path_ids(decoder, (), (R1, R1)))


def test_empty_candidates_close_both_arrays(decoding):
    decoder = constrained(decoding, (), ())
    ids = path_ids(decoder, (), ())
    assert_path(decoder, ids)
    assert len(ids) + 1 == decoder.max_tokens


@pytest.mark.parametrize("text", [
    '{"selected_entity_ids": ["unknown"], "selected_relation_ids": []}',
    '{"selected_entity_ids": [], "selected_relation_ids": ["' + R1 + '","' + R1 + '"]}',
    '{"selected_entity_ids": [], "selected_relation_ids": [], "extra": 1}',
    '{"selected_entity_ids": [], "selected_entity_ids": [], "selected_relation_ids": []}',
    '{"selected_entity_ids": true, "selected_relation_ids": []}',
    '{"selected_entity_ids": [], "selected_relation_ids": [',
])
def test_invalid_outputs_are_rejected_without_salvaging(decoding, text):
    with pytest.raises(ValueError):
        decoding.check_selection(text, (E1, E2), (R1, R2))


@pytest.mark.parametrize("entities,relations", [((E1, E1), ()), (("bad",), ()), ((), (None,))])
def test_invalid_candidate_ids_rejected(decoding, entities, relations):
    with pytest.raises(ValueError):
        constrained(decoding, entities, relations)


def test_tokenizer_must_round_trip_json_chunks(decoding):
    class Broken(AsciiTokenizer):
        def decode(self, ids, **_):
            return " " + super().decode(ids)
    with pytest.raises(ValueError, match="tokenizer"):
        decoding.UniqueSelectionConstraint(Broken(), (E1,), (), [999])


def test_multi_character_tokens_are_supported(decoding):
    class Merged(AsciiTokenizer):
        def encode(self, text, **_):
            return [1000] if text == json.dumps(E1) else super().encode(text)
        def decode(self, ids, **_):
            return "".join(json.dumps(E1) if i == 1000 else chr(i) for i in ids)
    decoder = decoding.UniqueSelectionConstraint(Merged(), (E1,), (), [999])
    ids = decoder.tokenizer.encode('{"selected_entity_ids": [') + [1000]
    ids += decoder.tokenizer.encode('], "selected_relation_ids": []}')
    assert_path(decoder, ids)


def wire_prompt():
    return build_llm_selection_messages(source_unit(), RuleCandidateExtractorV2().predict_unit(source_unit()))


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message
    finish_reason: str = "stop"


@dataclass
class Response:
    choices: list


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
    def tolist(self): return self.values


@pytest.fixture
def hooked(decoding):
    class Engine:
        tokenizer = AsciiTokenizer()
        max_batch_size = 1
        use_prefix = True

        def _infer(self, infer_requests, request_config, *, adapter_request=None, pre_infer_hook=None):
            if not infer_requests[0].chat_template_kwargs.get("ecospec_selection_decoding"):
                return ["legacy result"]
            kwargs = {"inputs": {"input_ids": Tensor([11, 22, 33])},
                "generation_config": SimpleNamespace(eos_token_id=[999], max_new_tokens=request_config.max_tokens)}
            kwargs = pre_infer_hook(kwargs)
            prefix = kwargs["inputs"]["prefix_allowed_tokens_fn"]
            ids = []
            if self.use_prefix:
                for _ in range(request_config.max_tokens):
                    token = min(prefix(0, Tensor([11, 22, 33] + ids)))
                    if token == 999: break
                    ids.append(token)
                text = self.tokenizer.decode(ids)
            else:
                text = '{"selected_entity_ids": [], "selected_relation_ids": []}'
            return [Response([Choice(Message(text))])]

    class App:
        routes = {}
        def get(self, path):
            return lambda fn: self.routes.setdefault(path, fn)

    class Deploy:
        args = SimpleNamespace(infer_backend="transformers")
        infer_engine = Engine()
        app = App()
        def _register_app(self): pass
        async def create_chat_completion(self, request, raw_request, *, return_cmpl_response=False):
            return self.infer_engine._infer([request], request)[0]

    decoding.install_swift_hooks(Engine, Deploy, swift_version="4.4.2",
        json_response=lambda payload, status_code=200: {"status_code": status_code, "body": payload})
    system, prompt = wire_prompt()
    request = SimpleNamespace(messages=[{"role": "system", "content": system},
        {"role": "user", "content": prompt}], chat_template_kwargs={"enable_thinking": False,
        "ecospec_selection_decoding": VERSION}, stream=False, n=1, num_beams=1, temperature=0, max_tokens=3072)
    return Engine, Deploy, request


def test_swift_hook_passes_constraint_and_returns_verified_proof(decoding, hooked):
    Engine, Deploy, request = hooked
    service = Deploy()
    service._register_app()
    capabilities = asyncio.run(service.app.routes["/v1/ecospec-selection-decoding"]())
    assert capabilities["version"] == VERSION
    assert capabilities["implementation"] == decoding.implementation_hashes()
    original_hook_calls = []
    response = Engine()._infer([request], request,
        pre_infer_hook=lambda kwargs: original_hook_calls.append(kwargs) or kwargs)[0]
    assert original_hook_calls
    reply = asyncio.run(service.create_chat_completion(request, None))
    assert reply["status_code"] == 200
    metadata = reply["body"]["ecospec_selection_decoding"]
    assert metadata["prefix_calls"] > 0
    decoding.verify_response(reply["body"], request.messages[0]["content"], request.messages[1]["content"])
    assert Engine()._infer([SimpleNamespace(chat_template_kwargs={})], request) == ["legacy result"]


@pytest.mark.parametrize("fault", ["budget", "sampling", "stream", "thinking", "unsupported", "missing_callback"])
def test_hook_fails_closed_instead_of_claiming_unconstrained_success(hooked, fault):
    Engine, Deploy, request = hooked
    service = Deploy()
    if fault == "budget": request.max_tokens = 10
    elif fault == "sampling": request.temperature = 0.7
    elif fault == "stream": request.stream = True
    elif fault == "thinking": request.chat_template_kwargs["enable_thinking"] = True
    elif fault == "unsupported": request.chat_template_kwargs["ecospec_selection_decoding"] = "other"
    elif fault == "missing_callback": service.infer_engine.use_prefix = False
    reply = asyncio.run(service.create_chat_completion(request, None))
    assert reply["status_code"] == 400
    # An error must not kill the engine worker or change unmarked requests.
    assert Engine()._infer([SimpleNamespace(chat_template_kwargs={})], request) == ["legacy result"]


@pytest.mark.parametrize("field", ["version", "prompt_hash", "implementation", "prefix_calls"])
def test_mismatched_server_proof_rejected(decoding, hooked, field):
    _, Deploy, request = hooked
    reply = asyncio.run(Deploy().create_chat_completion(request, None))["body"]
    reply["ecospec_selection_decoding"][field] = None
    with pytest.raises(ValueError):
        decoding.verify_response(reply, request.messages[0]["content"], request.messages[1]["content"])


@pytest.mark.parametrize("extra", [
    {"selection_decoding": "unknown"},
    {"selection_decoding": VERSION, "enable_thinking": True},
    {"selection_decoding": VERSION, "temperature": 0.8},
    {"selection_decoding": VERSION, "backend": "rule"},
])
def test_invalid_decoding_config_rejected_before_network(tmp_path, extra):
    write_jsonl(tmp_path / "units.jsonl", [source_unit()])
    write_json(tmp_path / "config.json", {"backend": "llm", "enable_thinking": False, **extra})
    with patch("urllib.request.urlopen") as send:
        with pytest.raises(ValueError, match="selection_decoding"):
            extract_v2(tmp_path / "units.jsonl", tmp_path / "out", config_path=tmp_path / "config.json")
    send.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.fixture
def probe_module():
    path = Path(__file__).resolve().parents[1] / "tools/probe_repeated_selection.py"
    spec = importlib.util.spec_from_file_location("unique_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unique_probe_checks_capability_and_saves_complete_proof(decoding, probe_module, previous_run, tmp_path):
    system, prompt = wire_prompt()
    proof = {**decoding.request_identity(system, prompt), "prefix_calls": 54, "max_completion_tokens": 200}
    raw = {"choices": [{"finish_reason": "stop", "message": {"content":
        '{"selected_entity_ids": [], "selected_relation_ids": []}'}}], "ecospec_selection_decoding": proof}
    calls = []
    before = {p: p.read_bytes() for p in previous_run.iterdir() if p.is_file()}

    def send(request, timeout):
        calls.append((request, timeout))
        if request.full_url.endswith("/models"):
            result = {"data": [{"id": "eco-lora-v28-seed44"}]}
        elif request.full_url.endswith("/ecospec-selection-decoding"):
            result = {"version": VERSION, "implementation": decoding.implementation_hashes(), "backend": "transformers"}
        else:
            payload = json.loads(request.data)
            assert "repetition_penalty" not in payload
            assert payload["chat_template_kwargs"] == {"enable_thinking": False, "ecospec_selection_decoding": VERSION}
            result = raw
        return Reply(json.dumps(result).encode())

    with patch("urllib.request.urlopen", side_effect=send):
        result = probe_module.probe(previous_run, tmp_path / "unique", unique_candidates=True)
    assert len(calls) == 3
    assert result["probe_passed"] and result["formal_evaluation"] is False
    assert result["decoding_change"] == {"selection_decoding": VERSION}
    assert result["server_decoding_proof"] == proof
    assert read_json(tmp_path / "unique/resolved_config.json")["selection_decoding"] == VERSION
    assert read_json(tmp_path / "unique/provider_response.json") == raw
    assert all(p.read_bytes() == content for p, content in before.items())


def test_unique_probe_missing_plugin_stops_before_inference(decoding, probe_module, previous_run, tmp_path):
    calls = []
    def send(request, **_):
        calls.append(request)
        result = {"data": [{"id": "eco-lora-v28-seed44"}]} if request.full_url.endswith("/models") else {}
        return Reply(json.dumps(result).encode())
    with patch("urllib.request.urlopen", side_effect=send):
        with pytest.raises(ValueError, match="plugin"):
            probe_module.probe(previous_run, tmp_path / "unique", unique_candidates=True)
    assert len(calls) == 2 and all(r.data is None for r in calls)
    assert not (tmp_path / "unique").exists()
