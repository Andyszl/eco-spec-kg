"""Exact-length binary decisions; grammar restricts syntax, never labels.

Reuse the frozen ASCII/token intersection, replacing only its JSON grammar.
Every decision admits both 0 and 1, including candidates masked in training.
"""
from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request

from . import partial_label_v226 as policy
from .experiment_io_v2 import sha256_json
from .providers import OpenAICompatibleProvider, CompletionHTTPError, CompletionTruncatedError
from .selection_decoding_v2 import UniqueSelectionConstraint, _ids, require

VERSION = 'binary-candidate-json-v230'
MARKER = 'ecospec_binary_decoding'


def candidate_ids(prompt):
    data = json.loads(prompt)
    require(data.get('decision_format_version') == policy.VERSION, 'Wrong binary question version')
    require(set(data.get('response_schema', {})) == set(policy.FIELDS), 'Wrong binary response schema')
    require(not any(k in data for k in ('channel', 'unknown_entity_ids', 'unknown_relation_ids')),
            'Supervision metadata is forbidden during inference')
    return tuple(_ids(v) for v in policy.ordered_ids(prompt))


def grammar(entities, relations):
    return json.dumps(dict(zip(policy.FIELDS, (' '.join('?' for _ in group)
                      for group in (entities, relations)))), separators=(',', ':'))


class BinaryDecisionConstraint(UniqueSelectionConstraint):
    def __init__(self, tokenizer, entities, relations, eos_token_ids):
        super().__init__(tokenizer, entities, relations, eos_token_ids)
        self.pattern = grammar(*self.ids)
        self.max_tokens = len(self.pattern) + 1
        self.initial = ('step', 0)
        self._next_cache = {}
        self._cached_tokens, self._cached_text, self._cached_state = (), '', self.initial

    def _next_chars(self, state):
        if state[0] == 'done':
            return {}
        pos = state[1]
        following = ('done', pos + 1) if pos + 1 == len(self.pattern) else ('step', pos + 1)
        return {c: following for c in ('01' if self.pattern[pos] == '?' else self.pattern[pos])}


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    import hashlib
    names = ['src/ecospec_kg/binary_decoding_v230.py', 'src/ecospec_kg/selection_decoding_v2.py',
             'src/ecospec_kg/partial_label_v226.py', 'tools/swift_binary_decision_plugin_20261021.py']
    return {n: hashlib.sha256((root / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}


def request_identity(system, prompt):
    ids = candidate_ids(prompt)
    return dict(version=VERSION, implementation=implementation_hashes(),
                prompt_hash=sha256_json({'system': system, 'prompt': prompt}), candidate_ids_hash=sha256_json(ids))


def _decision_content(response):
    text = response['choices'][0]['message']['content']
    require(isinstance(text, str), 'Binary response must contain text')
    # Swift 4.4.2 Qwen3_5Template may prepend this exact non-thinking prefix
    # during response decoding. Strip only that empty SDK wrapper, once.
    prefix = '<think>\n\n</think>\n\n'
    return text[len(prefix):] if text.startswith(prefix) else text


def verify_response(response, system, prompt):
    proof = response.get(MARKER)
    require(isinstance(proof, dict) and all(proof.get(k) == v for k, v in request_identity(system, prompt).items()),
            'Binary decoding server proof missing or mismatched')
    calls, bound = proof.get('prefix_calls'), proof.get('max_completion_tokens')
    require(type(calls) is int and type(bound) is int and 0 < calls <= bound
            and bound == len(grammar(*candidate_ids(prompt))) + 1, 'Binary constraint was not applied')
    choice = response['choices'][0]
    require(choice.get('finish_reason') == 'stop', 'Binary response did not stop normally')
    text = _decision_content(response)
    chosen = policy.selection_for(prompt, text)
    require(text == policy.answer_for(prompt, chosen), 'Noncanonical binary response')
    return chosen


class BinaryProvider(OpenAICompatibleProvider):
    """Explicit wire protocol; failure never falls back to unconstrained output."""
    def complete(self, system, prompt):
        self.last_raw_response = None
        require(self.selection_decoding == VERSION and self.enable_thinking is False
                and self.temperature == 0 and not self.append_no_think and self.repetition_penalty is None,
                'Binary decoding requires non-thinking greedy generation')
        candidate_ids(prompt)
        payload = dict(model=self.model, messages=[dict(role='system', content=system), dict(role='user', content=prompt)],
                       temperature=0, max_tokens=self.max_tokens, seed=self.seed,
                       chat_template_kwargs={'enable_thinking': False, MARKER: VERSION})
        request = urllib.request.Request(self.base_url.rstrip('/') + '/chat/completions',
                    data=json.dumps(payload).encode('utf8'), method='POST',
                    headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                response = json.load(reply)
        except urllib.error.HTTPError as error:
            body = error.read().decode('utf8', errors='replace')
            self.last_raw_response = dict(http_status=error.code, error_body=body)
            raise CompletionHTTPError(f'HTTP {error.code}: {body[:2000]}') from error
        self.last_raw_response = response
        if response['choices'][0].get('finish_reason') == 'length':
            raise CompletionTruncatedError('Binary completion exceeded its output limit')
        verify_response(response, system, prompt)
        return _decision_content(response)


from dataclasses import asdict
from functools import wraps
import inspect


def _messages(request):
    messages = request.messages
    require(len(messages) == 2 and [r["role"] for r in messages] == ["system", "user"],
            "binary_decoding requires one system and one user message")
    require(all(isinstance(r["content"], str) for r in messages), "binary_decoding requires text messages")
    return messages[0]["content"], messages[1]["content"]


def install_swift_hooks(engine_class, deploy_class, *, swift_version, json_response):
    """Narrow adapter for the verified Swift 4.4.2 hook and non-streaming HTTP route."""
    require(swift_version == "4.4.2", "binary_decoding plugin supports ms-swift 4.4.2 only")
    require("pre_infer_hook" in inspect.signature(engine_class._infer).parameters,
            "unsupported Swift TransformersEngine hook")
    if getattr(engine_class, "_ecospec_binary_decision_installed", False):
        return
    original_infer = engine_class._infer
    original_create = deploy_class.create_chat_completion
    original_register = deploy_class._register_app

    @wraps(original_infer)
    def infer(engine, infer_requests, request_config, *, adapter_request=None, pre_infer_hook=None):
        marked = [r.chat_template_kwargs.get(MARKER) for r in infer_requests]
        if not any(marked):
            return original_infer(engine, infer_requests, request_config,
                                  adapter_request=adapter_request, pre_infer_hook=pre_infer_hook)
        try:
            require(marked == [VERSION], "binary_decoding requires a single marked request")
            system, prompt = _messages(infer_requests[0])
            identity = request_identity(system, prompt)
            holder = {}

            def hook(kwargs):
                if pre_infer_hook: kwargs = pre_infer_hook(kwargs)
                inputs, config = kwargs["inputs"], kwargs["generation_config"]
                require("input_ids" in inputs and inputs["input_ids"].shape[0] == 1,
                        "binary_decoding requires a single text input")
                decoder = BinaryDecisionConstraint(engine.tokenizer, *candidate_ids(prompt), config.eos_token_id)
                require(config.max_new_tokens >= decoder.max_tokens,
                        f"binary_decoding budget too small: need {decoder.max_tokens} tokens for all candidates")
                require("prefix_allowed_tokens_fn" not in inputs, "another prefix constraint is already present")
                inputs["prefix_allowed_tokens_fn"] = decoder.prefix_callback(inputs["input_ids"].shape[-1])
                holder["decoder"] = decoder
                return kwargs

            replies = original_infer(engine, infer_requests, request_config,
                                     adapter_request=adapter_request, pre_infer_hook=hook)
            decoder = holder.get("decoder")
            require(decoder is not None and decoder.calls > 0, "binary_decoding callback was not applied")
            require(len(replies) == 1 and hasattr(replies[0], "choices"), "binary_decoding generation failed")
            setattr(replies[0], "_ecospec_binary_proof", {**identity, "prefix_calls": decoder.calls,
                    "max_completion_tokens": decoder.max_tokens})
            return replies
        except Exception as exc:
            # Swift's worker expects a result list. Never terminate its thread on a
            # constraint error; the HTTP wrapper below turns it into a clear 400.
            return [exc]

    @wraps(original_create)
    async def create(service, request, raw_request, *, return_cmpl_response=False):
        marker = request.chat_template_kwargs.get(MARKER)
        if marker is None:
            return await original_create(service, request, raw_request, return_cmpl_response=return_cmpl_response)
        try:
            require(marker == VERSION, "unsupported binary_decoding version")
            require(service.args.infer_backend == "transformers" and service.infer_engine.max_batch_size == 1,
                    "binary_decoding requires Transformers with max_batch_size=1")
            require(not request.stream and request.n == request.num_beams == 1 and not return_cmpl_response,
                    "binary_decoding requires non-streaming chat, n=1, num_beams=1")
            require(request.temperature == 0 and request.chat_template_kwargs.get("enable_thinking") is False,
                    "binary_decoding requires temperature=0 and enable_thinking=false")
            _messages(request)
            result = await original_create(service, request, raw_request, return_cmpl_response=False)
            if isinstance(result, Exception): raise result
            proof = getattr(result, "_ecospec_binary_proof", None)
            require(proof is not None, "binary_decoding generation did not return server proof")
            return json_response({**asdict(result), MARKER: proof})
        except Exception as exc:
            return json_response({"message": f"binary_decoding: {type(exc).__name__}: {exc}"}, status_code=400)

    @wraps(original_register)
    def register(service):
        original_register(service)

        async def capabilities():
            return {"version": VERSION, "implementation": implementation_hashes(), "swift_version": swift_version,
                    "backend": service.args.infer_backend}
        service.app.get("/v1/ecospec-binary-decoding")(capabilities)

    engine_class._infer = infer
    engine_class._ecospec_binary_decision_installed = True
    deploy_class.create_chat_completion = create
    deploy_class._register_app = register
