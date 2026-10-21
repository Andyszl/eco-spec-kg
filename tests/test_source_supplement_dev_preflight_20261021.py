"""Exercise encoding guards with explicit tokenizer/template substitutes."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import source_supplement_dev_preflight_20261021 as pf
from ecospec_kg import binary_decoding_v230 as binary
from ecospec_kg import partial_label_v226 as policy
from ecospec_kg.candidate_profile_v228 import candidates_for_unit
from ecospec_kg.providers import CompletionTruncatedError
from test_unique_selection_v2 import AsciiTokenizer
from test_experiment_chain_v2 import source_unit


class Template:
    tokenizer = AsciiTokenizer()
    tokenizer.eos_token_id = 999
    def __init__(self, fault=None):
        self.fault = fault
    def set_mode(self, mode):
        self.mode = mode
    def encode(self, row):
        prefix = list(sha256(json.dumps(row['messages'][:2]).encode()).digest())
        if self.fault == 'length':
            prefix *= 300
        if self.mode == 'transformers':
            if self.fault == 'infer':
                prefix.append(8)
            return dict(input_ids=prefix)
        if self.fault == 'train':
            prefix[0] += 1
        tokens = prefix + self.tokenizer.encode(row['messages'][-1]['content']) + [999]
        labels = [-100] * len(prefix) + tokens[len(prefix):]
        if self.fault == 'labels':
            labels[-2] = -100
        return dict(input_ids=tokens, labels=labels)


def test_inference_prefix_and_complete_binary_paths_have_no_gold_or_masks():
    units = [source_unit()]
    before = deepcopy(units)
    rows = pf.check_prompts(units, Template(), Template())
    assert units == before and len(rows) == 1
    assert rows[0]['prompt_tokens'] + rows[0]['worst_case_output_tokens'] <= 8192
    assert set(rows[0]) == {'unit_id', 'prompt_tokens', 'worst_case_output_tokens', 'prefix_sha256',
                           'prompt_hash', 'entity_candidates', 'relation_candidates'}


@pytest.mark.parametrize('fault', ['length', 'infer', 'train', 'labels'])
def test_encoding_mismatch_or_overflow_blocks_generation(fault):
    with pytest.raises(AssertionError):
        pf.check_prompts([source_unit()], Template(), Template(fault))


@pytest.mark.parametrize('fault', ['version', 'thinking', 'sampling', 'metadata'])
def test_invalid_wire_configuration_fails_before_network(fault):
    unit = source_unit()
    system, prompt = policy.build_messages(unit, candidates_for_unit(unit))
    provider = binary.BinaryProvider(base_url='http://invalid', api_key='local-token',
               enable_thinking=False, seed=42, selection_decoding=binary.VERSION)
    if fault == 'version':
        provider.selection_decoding = 'old-unique-ID-version'
    elif fault == 'thinking':
        provider.enable_thinking = True
    elif fault == 'sampling':
        provider.temperature = .7
    else:
        q = json.loads(prompt); q['unknown_entity_ids'] = []; prompt = json.dumps(q)
    with patch('urllib.request.urlopen') as send, pytest.raises(ValueError):
        provider.complete(system, prompt)
    send.assert_not_called()


def test_truncated_reply_preserved_and_not_repaired_or_retried():
    from test_repetition_control_v2 import Reply
    unit = source_unit()
    system, prompt = policy.build_messages(unit, candidates_for_unit(unit))
    response = dict(choices=[dict(finish_reason='length', message=dict(content='{"entity_decisions":'))])
    provider = binary.BinaryProvider(base_url='http://invalid', api_key='local-token',
               enable_thinking=False, seed=42, selection_decoding=binary.VERSION)
    with patch('urllib.request.urlopen', return_value=Reply(json.dumps(response).encode())) as send:
        with pytest.raises(CompletionTruncatedError):
            provider.complete(system, prompt)
    assert provider.last_raw_response == response and send.call_count == 1
