"""Finite token paths for two JSON arrays of unique, source-local candidate IDs.

This restricts syntax and membership only. Every subset and ordering is allowed;
no gold labels or relation types influence the allowed tokens.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import wraps
import inspect
import json
from pathlib import Path
import re

from .experiment_io_v2 import sha256_json, sha256_path

VERSION = "unique-candidate-json-v2"
MARKER = "ecospec_selection_decoding"
KEYS = ("selected_entity_ids", "selected_relation_ids")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(values):
    values = tuple(values)
    require(all(isinstance(i, str) and re.fullmatch(r"[0-9a-f]{16}", i) for i in values),
            "candidate IDs must be 16 lowercase hexadecimal characters")
    require(len(values) == len(set(values)), "duplicate candidate IDs")
    return values


def candidate_ids(prompt):
    data = json.loads(prompt)
    require(data.get("selection_policy", {}).get("version") == "ecospec-selection-v2.5",
            "unsupported selection prompt version")
    require(set(data.get("output_schema", {})) == set(KEYS), "unexpected output schema")
    return tuple(_ids(r["id"] for r in data[field])
                 for field in ("candidate_entities", "candidate_relations"))


def implementation_hashes():
    module = Path(__file__).resolve()
    plugin = module.parents[2] / "tools/swift_unique_selection_plugin.py"
    return {module.name: sha256_path(module), plugin.name: sha256_path(plugin)}


def request_identity(system, prompt):
    entities, relations = candidate_ids(prompt)
    return {"version": VERSION, "implementation": implementation_hashes(),
            "prompt_hash": sha256_json({"system": system, "prompt": prompt}),
            "candidate_ids_hash": sha256_json([entities, relations])}


def _unique_object(pairs):
    require(len(pairs) == len({key for key, _ in pairs}), "duplicate JSON keys")
    return dict(pairs)


def check_selection(text, entities, relations):
    text = text.strip()
    if text.startswith("<think>"):
        reasoning, closing, text = text[len("<think>"):].partition("</think>")
        require(closing and not reasoning.strip(), "unexpected thinking output")
    result = json.loads(text.strip(), object_pairs_hook=_unique_object)
    require(isinstance(result, dict) and set(result) == set(KEYS), "unexpected selection fields")
    for key, candidates in zip(KEYS, (entities, relations)):
        values = result[key]
        require(isinstance(values, list) and all(isinstance(i, str) for i in values),
                "selection must contain string arrays")
        require(len(values) == len(set(values)), "duplicate selected IDs")
        require(set(values) <= set(candidates), "unknown selected IDs")
    return result


def verify_response(response, system, prompt):
    proof = response.get(MARKER)
    require(isinstance(proof, dict), "selection_decoding server proof is missing; load the Swift plugin")
    require(all(proof.get(k) == v for k, v in request_identity(system, prompt).items()),
            "selection_decoding server proof mismatch")
    calls, bound = proof.get("prefix_calls"), proof.get("max_completion_tokens")
    require(type(calls) is int and type(bound) is int and 0 < calls <= bound,
            "selection_decoding prefix constraint was not applied")
    choice = response["choices"][0]
    require(choice.get("finish_reason") == "stop", "selection_decoding response did not stop normally")
    return check_selection(choice["message"]["content"], *candidate_ids(prompt))


_ALPHABET = frozenset('{"selected_entity_ids": [], "selected_relation_ids": []}0123456789abcdef')


class _AsciiTokens:
    """Vocabulary trie permits tokens spanning JSON punctuation and ID boundaries."""

    def __init__(self, tokenizer):
        self.texts, self.trie = {}, {}
        ids = sorted(set(tokenizer.get_vocab().values()))
        for offset in range(0, len(ids), 4096):
            batch = ids[offset:offset + 4096]
            kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            if hasattr(tokenizer, "batch_decode"):
                texts = tokenizer.batch_decode([[i] for i in batch], **kwargs)
            else:
                texts = [tokenizer.decode([i], **kwargs) for i in batch]
            require(len(texts) == len(batch), "tokenizer returned an incomplete vocabulary decode")
            for token, text in zip(batch, texts):
                if not text or not set(text) <= _ALPHABET:
                    continue
                self.texts[token] = text
                node = self.trie
                for char in text:
                    node = node.setdefault(char, {})
                node.setdefault(None, []).append(token)


class UniqueSelectionConstraint:
    """Finite ASCII JSON grammar intersected with the tokenizer's vocabulary."""

    def __init__(self, tokenizer, entities, relations, eos_token_ids):
        self.tokenizer = tokenizer
        self.ids = (_ids(entities), _ids(relations))
        self.eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids or [])
        require(self.eos and all(type(i) is int and i >= 0 for i in self.eos), "missing EOS token IDs")
        self.calls = 0
        self.start = '{"selected_entity_ids": ['
        self.close = ('], "selected_relation_ids": [', "]}")
        self.items = tuple({i: json.dumps(i) for i in group} for group in self.ids)
        # Every accepted token consumes at least one ASCII character; even the
        # least compressed legal path fits this bound. There is no whitespace loop.
        self.max_tokens = len(json.dumps(dict(zip(KEYS, self.ids)))) + 1
        for char in _ALPHABET:
            ids = tokenizer.encode(char, add_special_tokens=False)
            require(len(ids) == 1 and ids[0] not in self.eos and self._decode(ids) == char,
                    "tokenizer cannot round-trip individual JSON characters")
        tokens = getattr(tokenizer, "_ecospec_ascii_tokens", None)
        if tokens is None:
            tokens = _AsciiTokens(tokenizer)
            tokenizer._ecospec_ascii_tokens = tokens
        self.tokens = tokens
        self.initial = ("start", 0, frozenset(), frozenset(), "")
        self._next_cache = {}
        self._cached_tokens, self._cached_text, self._cached_state = (), "", self.initial

    def _decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _next_chars(self, state):
        if state in self._next_cache:
            return self._next_cache[state]
        stage, group, used0, used1, prefix = state
        if stage == "done":
            return {}
        if stage == "start":
            options = [(self.start, "first", None)]
        else:
            used = (used0, used1)[group]
            remaining = [(text, "after", item) for item, text in self.items[group].items() if item not in used]
            close = [(self.close[group], "next_group" if group == 0 else "done", None)]
            if stage == "first": options = remaining + close
            elif stage == "item": options = remaining
            elif stage == "after": options = ([(", ", "item", None)] if remaining else []) + close
            else: raise ValueError("invalid selection grammar state")
        options = [(text, target, item) for text, target, item in options if text.startswith(prefix)]
        following = {}
        for char in {text[len(prefix)] for text, _, _ in options}:
            new_prefix = prefix + char
            complete = [(target, item) for text, target, item in options if text == new_prefix]
            if complete:
                require(len(complete) == 1 and not any(text.startswith(new_prefix) and text != new_prefix
                        for text, _, _ in options), "ambiguous selection grammar prefix")
                target, item = complete[0]
                used = [used0, used1]
                if item is not None: used[group] = used[group] | {item}
                if target == "next_group": target, next_group = "first", 1
                else: next_group = group
                following[char] = (target, next_group, *used, "")
            else:
                following[char] = (stage, group, used0, used1, new_prefix)
        self._next_cache[state] = following
        return following

    def allowed(self, generated):
        generated = tuple(generated)
        if generated[:len(self._cached_tokens)] == self._cached_tokens:
            state, text, offset = self._cached_state, self._cached_text, len(self._cached_tokens)
        else:
            state, text, offset = self.initial, "", 0
        for token in generated[offset:]:
            chunk = self.tokens.texts.get(token)
            require(chunk is not None and token not in self.eos, "invalid selection token prefix")
            for char in chunk:
                state = self._next_chars(state).get(char)
                require(state is not None, "invalid selection token prefix")
            text += chunk
        require(self._decode(generated) == text, "tokenizer changes concatenated ASCII tokens")
        self._cached_tokens, self._cached_text, self._cached_state = generated, text, state
        if state[0] == "done":
            return self.eos
        stack, allowed = [(self.tokens.trie, state)], []
        while stack:
            node, current = stack.pop()
            allowed.extend(token for token in node.get(None, ()) if token not in self.eos)
            for char, following in self._next_chars(current).items():
                if char in node: stack.append((node[char], following))
        require(allowed, "tokenizer has no valid selection continuation")
        return sorted(allowed)

    def prefix_callback(self, prompt_length):
        def allowed_tokens(batch_id, input_ids):
            require(batch_id == 0, "selection_decoding requires a single generation sequence")
            self.calls += 1
            return self.allowed(input_ids.tolist()[prompt_length:])
        return allowed_tokens


def _messages(request):
    messages = request.messages
    require(len(messages) == 2 and [r["role"] for r in messages] == ["system", "user"],
            "selection_decoding requires one system and one user message")
    require(all(isinstance(r["content"], str) for r in messages), "selection_decoding requires text messages")
    return messages[0]["content"], messages[1]["content"]


def install_swift_hooks(engine_class, deploy_class, *, swift_version, json_response):
    """Narrow adapter for the verified Swift 4.4.2 hook and non-streaming HTTP route."""
    require(swift_version == "4.4.2", "selection_decoding plugin supports ms-swift 4.4.2 only")
    require("pre_infer_hook" in inspect.signature(engine_class._infer).parameters,
            "unsupported Swift TransformersEngine hook")
    if getattr(engine_class, "_ecospec_unique_selection_installed", False):
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
            require(marked == [VERSION], "selection_decoding requires a single marked request")
            system, prompt = _messages(infer_requests[0])
            identity = request_identity(system, prompt)
            holder = {}

            def hook(kwargs):
                if pre_infer_hook: kwargs = pre_infer_hook(kwargs)
                inputs, config = kwargs["inputs"], kwargs["generation_config"]
                require("input_ids" in inputs and inputs["input_ids"].shape[0] == 1,
                        "selection_decoding requires a single text input")
                decoder = UniqueSelectionConstraint(engine.tokenizer, *candidate_ids(prompt), config.eos_token_id)
                require(config.max_new_tokens >= decoder.max_tokens,
                        f"selection_decoding budget too small: need {decoder.max_tokens} tokens for all candidates")
                require("prefix_allowed_tokens_fn" not in inputs, "another prefix constraint is already present")
                inputs["prefix_allowed_tokens_fn"] = decoder.prefix_callback(inputs["input_ids"].shape[-1])
                holder["decoder"] = decoder
                return kwargs

            replies = original_infer(engine, infer_requests, request_config,
                                     adapter_request=adapter_request, pre_infer_hook=hook)
            decoder = holder.get("decoder")
            require(decoder is not None and decoder.calls > 0, "selection_decoding callback was not applied")
            require(len(replies) == 1 and hasattr(replies[0], "choices"), "selection_decoding generation failed")
            setattr(replies[0], "_ecospec_selection_proof", {**identity, "prefix_calls": decoder.calls,
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
            require(marker == VERSION, "unsupported selection_decoding version")
            require(service.args.infer_backend == "transformers" and service.infer_engine.max_batch_size == 1,
                    "selection_decoding requires Transformers with max_batch_size=1")
            require(not request.stream and request.n == request.num_beams == 1 and not return_cmpl_response,
                    "selection_decoding requires non-streaming chat, n=1, num_beams=1")
            require(request.temperature == 0 and request.chat_template_kwargs.get("enable_thinking") is False,
                    "selection_decoding requires temperature=0 and enable_thinking=false")
            _messages(request)
            result = await original_create(service, request, raw_request, return_cmpl_response=False)
            if isinstance(result, Exception): raise result
            proof = getattr(result, "_ecospec_selection_proof", None)
            require(proof is not None, "selection_decoding generation did not return server proof")
            return json_response({**asdict(result), MARKER: proof})
        except Exception as exc:
            return json_response({"message": f"selection_decoding: {type(exc).__name__}: {exc}"}, status_code=400)

    @wraps(original_register)
    def register(service):
        original_register(service)

        async def capabilities():
            return {"version": VERSION, "implementation": implementation_hashes(), "swift_version": swift_version,
                    "backend": service.args.infer_backend}
        service.app.get("/v1/ecospec-selection-decoding")(capabilities)

    engine_class._infer = infer
    engine_class._ecospec_unique_selection_installed = True
    deploy_class.create_chat_completion = create
    deploy_class._register_app = register
