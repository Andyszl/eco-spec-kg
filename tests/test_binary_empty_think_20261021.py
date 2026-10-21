"""Replay the real SDK response that blocked dev; no model generation here."""
from copy import deepcopy
import io
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from ecospec_kg import binary_decoding_v230 as binary
from ecospec_kg import partial_label_v226 as policy
from ecospec_kg.candidate_profile_v228 import candidates_for_unit
from ecospec_kg.io_utils import read_jsonl
from test_source_supplement_dev_20261021 import pair, ev
from test_binary_decoding_20261021 import binary_http

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'experiments/source_supplement_dev_empty_think_fix_20261021/actual_extraction_failure.json'
SDK_PREFIX = '<think>\n\n</think>\n\n'


class Reply(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def captured():
    failure = json.loads(FIXTURE.read_text(encoding='utf8'))
    unit = next(u for u in read_jsonl(ROOT / 'deliveries/human_confirmed_20261002/blind/dev_units.jsonl')
                if u['unit_id'] == failure['unit_id'])
    system, prompt = policy.build_messages(unit, candidates_for_unit(unit))
    response = deepcopy(failure['provider_response'])
    # Only the implementation fingerprint changes for the patched adapter.
    # Raw content, bit decisions, token counts and finish_reason remain captured.
    response[binary.MARKER].update(binary.request_identity(system, prompt))
    return unit, system, prompt, response


def test_real_server_empty_think_wrapper_replays_without_altering_any_decision():
    unit, system, prompt, response = captured()
    raw = response['choices'][0]['message']['content']
    assert raw.startswith(SDK_PREFIX)
    expected_text = raw[len(SDK_PREFIX):]
    expected = policy.selection_for(prompt, expected_text)
    assert binary.verify_response(response, system, prompt) == expected
    provider = binary.BinaryProvider(base_url='http://fixture/v1', api_key='local-token', seed=42,
                                    enable_thinking=False, selection_decoding=binary.VERSION)
    with patch('urllib.request.urlopen', return_value=Reply(json.dumps(response).encode())) as send:
        prediction, text, digest = policy.select(provider, unit, candidates_for_unit(unit))
    assert send.call_count == 1 and text == expected_text
    assert provider.last_raw_response == response
    assert provider.last_raw_response['choices'][0]['message']['content'] == raw
    assert [r['relation_id'] for r in prediction['relations']] == expected['selected_relation_ids']


@pytest.mark.parametrize('content', [
    '<think>actual reasoning</think>\n{body}',
    '<think>\n{body}',
    '```json\n{body}\n```',
    'explanation\n{body}',
    SDK_PREFIX + SDK_PREFIX + '{body}',
    SDK_PREFIX + '{body}{{}}',
    SDK_PREFIX + '{{"entity_decisions":"2","relation_decisions":""}}',
    SDK_PREFIX + '{{"entity_decisions":"","entity_decisions":"","relation_decisions":""}}',
])
def test_prefix_compatibility_never_salvages_reasoning_or_invalid_json(content):
    _, system, prompt, response = captured()
    body = response['choices'][0]['message']['content'][len(SDK_PREFIX):]
    response['choices'][0]['message']['content'] = content.format(body=body)
    with pytest.raises(ValueError):
        binary.verify_response(response, system, prompt)


def test_wrapped_output_still_requires_matching_server_proof():
    _, system, prompt, response = captured()
    response[binary.MARKER]['prompt_hash'] = 'bad'
    with pytest.raises(ValueError, match='proof'):
        binary.verify_response(response, system, prompt)


def test_all_66_wrapped_http_replies_complete_same_unchanged_dev(pair, tmp_path, binary_http):
    def respond(request, **kwargs):
        reply = binary_http['respond'](request, **kwargs)
        value = json.load(reply)
        if request.data is not None:
            value['choices'][0]['message']['content'] = SDK_PREFIX + value['choices'][0]['message']['content']
        return Reply(json.dumps(value).encode())
    out = tmp_path / 'evaluation'
    with patch('urllib.request.urlopen', side_effect=respond):
        report = ev.evaluate(ROOT, pair, out)
    assert report['status'] == 'complete' and report['model_requests'] == len(binary_http['calls']) == 66
    prompts = [p for _, _, p in binary_http['calls']]
    assert prompts[:22] == prompts[22:44] == prompts[44:]
    for arm in ev.MODELS:
        records = read_jsonl(out / arm / 'raw_responses.jsonl')
        assert len(records) == 22
        for r in records:
            assert r['provider_response']['choices'][0]['message']['content'] == SDK_PREFIX + r['raw_response']
