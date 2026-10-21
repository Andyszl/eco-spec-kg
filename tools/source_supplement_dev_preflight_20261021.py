"""Check dev prompts with the installed tokenizer before any model weights load."""
from copy import deepcopy
from pathlib import Path
from ecospec_kg import alias_partial_label_v227 as labels
from ecospec_kg import partial_label_v226 as policy
from ecospec_kg.candidate_profile_v228 import candidates_for_unit
from ecospec_kg.binary_decoding_v230 import BinaryDecisionConstraint, grammar
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.io_utils import read_jsonl
from cross_source_pair_20261016 import save


def check_prompts(units, plain, masked):
    records = []
    for unit in units:
        graph = candidates_for_unit(unit)
        system, prompt = policy.build_messages(unit, graph)
        messages = [dict(role='system', content=system), dict(role='user', content=prompt)]
        ids = policy.ordered_ids(prompt)
        bound = len(grammar(*ids)) + 1
        assert bound <= 3072, 'Binary output exceeds fixed 3072-token budget'
        plain.set_mode('transformers'); masked.set_mode('transformers')
        x, y = (t.encode(dict(messages=deepcopy(messages))) for t in (plain, masked))
        assert x['input_ids'] == y['input_ids'] and not x.get('labels') and not y.get('labels'), 'Inference templates differ'
        assert 'channel' not in x and 'channel' not in y, 'Inference contains supervision metadata'
        assert len(x['input_ids']) + bound <= 8192, 'Prompt plus worst-case binary output exceeds 8192'
        decoder = BinaryDecisionConstraint(plain.tokenizer, *ids, plain.tokenizer.eos_token_id)
        assert decoder.max_tokens == bound
        for bit in ('0', '1'):
            selection = dict(zip(('selected_entity_ids', 'selected_relation_ids'),
                                 (group if bit == '1' else [] for group in ids)))
            answer = policy.answer_for(prompt, selection)
            tokens = plain.tokenizer.encode(answer, add_special_tokens=False)
            assert len(tokens) + 1 <= bound
            for offset, token in enumerate(tokens):
                assert token in decoder.allowed(tokens[:offset]), 'Tokenizer path rejected by binary grammar'
            assert set(decoder.allowed(tokens)) == set(decoder.eos)
        # Synthetic all-zero answer tests prefix alignment only, never gold.
        answer = policy.answer_for(prompt, dict(selected_entity_ids=[], selected_relation_ids=[]))
        row = dict(messages=deepcopy(messages) + [dict(role='assistant', content=answer)])
        plain.set_mode('train'); masked.set_mode('train')
        a = plain.encode(deepcopy(row))
        row['channel'] = labels.supervision_channel(prompt, answer, [], [])
        b = masked.encode(row)
        assert a['input_ids'] == b['input_ids'] and a['labels'] == b['labels'], 'Training templates differ on known bits'
        assert a['input_ids'][:len(x['input_ids'])] == x['input_ids'], 'Train and inference prefixes differ'
        records.append(dict(unit_id=unit['unit_id'], prompt_tokens=len(x['input_ids']),
                            worst_case_output_tokens=bound, prefix_sha256=sha256_json(x['input_ids']),
                            prompt_hash=sha256_json(dict(system=system, prompt=prompt)),
                            entity_candidates=len(ids[0]), relation_candidates=len(ids[1])))
    assert len(records) == len({r['unit_id'] for r in records}) == len(units)
    return records


def preflight_prompts(root, model, report_path):
    from swift.model import get_model_processor
    from swift.template import get_template
    import swift_alias_partial_label_plugin_20261020  # registers the audited training template
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None
    kwargs = dict(max_length=8192, truncation_strategy='raise', add_non_thinking_prefix=True,
                  enable_thinking=False, loss_scale='default+ignore_empty_think')
    plain = get_template(processor, template_type='qwen3_5', **kwargs)
    masked = get_template(processor, template_type=labels.TEMPLATE, **kwargs)
    units = read_jsonl(root / 'deliveries/human_confirmed_20261002/blind/dev_units.jsonl')
    assert len(units) == 22
    report = dict(passed=False, model_weights_loaded=False, model_calls=0, records=[])
    try:
        report['records'] = check_prompts(units, plain, masked)
        report['passed'] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(report_path, report)
    print('BINARY_DEV_PROMPTS_PREFLIGHT_OK=true CHECKED=22 MODEL_CALLS=0', flush=True)
    return report
