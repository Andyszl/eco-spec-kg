"""Rebuild immutable partial-label inputs and check the actual CPU SDK pipeline.

No model weights, training, dev responses or GPU services are used here.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from ecospec_kg import partial_label_v226 as policy
from ecospec_kg import candidate_profile_v214 as profile
from ecospec_kg.experiment_io_v2 import sha256_json
import full_graph_scope_pair_20261018 as previous
from cross_source_pair_20261016 import read, rows, sha, lfsha, save, dumps, uid

ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'experiments/partial_label_v226_20261020'
OLD_SHA = '0b3c79e533a57fa596a49e4bca0c53f1772c5c31ff270eabf78348395fd93ba5'
SOURCES = ['src/ecospec_kg/partial_label_v226.py', 'tools/swift_partial_label_plugin_20261020.py',
           'tools/partial_label_preflight_20261020.py', 'tools/run_partial_label_preflight_20261020.sh']


def verify_runtime_lock(root):
    lock = read(root / FOLDER / 'runtime_source_lock.json')
    assert lock['version'] == '4.4.2' and lock['file_count'] == 40
    assert lock['code_commit'] == '328bd7d2c14396e50bd65e29cfb26891b69f86cd'
    return lock


def compose(root):
    old = previous.verify_inputs(root)
    assert old['manifest_sha256'] == OLD_SHA, 'Frozen complete graph manifest changed'
    m = old['manifest']; patch = read(root / FOLDER / 'name_boundary_patch.json')
    assert patch['frozen_input_manifest_sha256'] == OLD_SHA
    all_units = rows(root / previous.previous.OLD / 'units.jsonl') + rows(root / previous.previous.SEED / 'source_units.jsonl')
    units = {u['unit_id']: u for u in all_units}
    graphs = {}; prepared = {}; changes = 0; coverage = {}; sources = {}
    for name, identities in m['unit_ids'].items():
        original = rows(root / previous.FOLDER / (name + '.jsonl'))
        assert [uid(r) for r in original] == identities
        if name in patch['original_input_sha256']:
            assert sha(root / previous.FOLDER / (name + '.jsonl')) == patch['original_input_sha256'][name]
        records = []; totals = {'positive_entities': 0, 'positive_relations': 0,
                                'retained_unselected_relations': 0, 'unknown_relations_masked': 0}
        source_ids = set(); by_type = {}
        for row in original:
            ident = uid(row); unit = units[ident]
            if ident not in graphs:
                graphs[ident] = profile.candidates_for_unit(unit)
            graph = graphs[ident]
            assert sha256_json(graph) == m['candidate_hashes'][ident]
            oldq = json.loads(row['messages'][1]['content'])
            selection = json.loads(row['messages'][2]['content'])
            if ident == patch['unit_id'] and name != 'validation':
                entry = next(p for p in patch['patches'] if p['arm'] == name)
                assert unit['provenance'] == patch['source_provenance']
                assert row['messages'][2]['content'] == entry['before_assistant']
                assert policy.digest(json.dumps(row['messages'][:2], ensure_ascii=False, sort_keys=True)) == entry['system_and_user_messages_sha256']
                selection = json.loads(entry['after_assistant']); changes += 1
                assert selection['selected_relation_ids'] == json.loads(entry['before_assistant'])['selected_relation_ids']
            system, prompt = policy.build_messages(unit, graph); q = json.loads(prompt)
            for field in ('candidate_entities', 'candidate_relations', 'source_unit', 'formula_output_bindings'):
                assert q.get(field) == oldq.get(field), 'Source or complete graph changed: ' + field
            unknown = m['unscored_relation_ids'][ident]
            assert not set(selection['selected_relation_ids']) & set(unknown)
            # Old standalone entity lists omitted relation endpoints by design.
            # Preserve the effective selected graph rather than label endpoints 0.
            selection = policy.effective_selection(graph, selection)
            source_ids.add(unit['provenance']['source_sha256'])
            totals['positive_entities'] += len(selection['selected_entity_ids'])
            for rel in graph['relations']:
                key = ('unknown_relations_masked' if rel['relation_id'] in unknown else
                       'positive_relations' if rel['relation_id'] in selection['selected_relation_ids'] else
                       'retained_unselected_relations')
                totals[key] += 1
                count = by_type.setdefault(rel['relation_type'], {k: 0 for k in totals if k != 'positive_entities'})
                count[key] += 1
            answer = policy.answer_for(prompt, selection)
            assert policy.selection_for(prompt, answer) == {
                k: [i for i in policy.ordered_ids(prompt)[j] if i in selection[k]]
                for j, k in enumerate(('selected_entity_ids', 'selected_relation_ids'))}
            records.append({'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt},
                                          {'role': 'assistant', 'content': answer}],
                            'channel': policy.supervision_channel(prompt, answer, unknown)})
        prepared[name] = records
        coverage[name] = {'totals': totals, 'by_relation_type': by_type, 'source_count': len(source_ids)}
        sources[name] = sorted(source_ids)
    assert changes == 2
    assert all(a['messages'][:2] == b['messages'][:2] for a, b in zip(prepared['control_train'], prepared['repair_train']))
    changed = sum(a['messages'][2] != b['messages'][2] for a, b in zip(prepared['control_train'], prepared['repair_train']))
    assert changed == 6
    manifest = {'version': policy.VERSION, 'counts': m['counts'], 'unit_ids': m['unit_ids'],
                'candidate_hashes': m['candidate_hashes'], 'old_manifest_sha256': OLD_SHA,
                'unscored_relation_ids': m['unscored_relation_ids'], 'unscored_counts': m['unscored_counts'],
                'tokenizer_config_sha256': m['tokenizer_config_sha256'],
                'source_sha256_lf': {n: lfsha(root / n) for n in SOURCES},
                'review_sha256_lf': {n: lfsha(root / FOLDER / n) for n in ('name_boundary_patch.json', 'runtime_source_lock.json')},
                'paired_changed_training_targets': changed, 'name_patch_records': changes,
                'full_graph_preserved': True, 'same_training_inference_question': True,
                'entity_targets_are_effective_graph_members': True,
                'selected_graphs_preserved_except_shared_name_revision': True,
                'unknown_metadata_in_prompt': False, 'token_level_loss_masking': True,
                'template_type': policy.TEMPLATE, 'max_length': 8192, 'truncation_strategy': 'raise',
                'new_supervision_sources': 0, 'model_weights_loaded': False, 'training_ready': False,
                'supervision_coverage': coverage, 'source_sha256_by_file': sources,
                'shared_train_validation_sources': sorted(set(sources['repair_train']) & set(sources['validation'])),
                'limitations': ['Real tokenizer, loader, collator and synthetic loss check pending until server report.',
                               'Masked placeholder bits still occur in autoregressive teacher-forcing context.',
                               'Known supervision, including the control ablation, retained except the shared name correction.',
                               'No additional positive source coverage or independent held-out validation created.',
                               'New output protocol requires fresh base/control/repair comparisons; old dev scores do not transfer.',
                               'Strict dev gate and independent-source evaluation remain unsatisfied.']}
    return prepared, manifest


def build(root):
    out = root / FOLDER
    assert not (out / 'input_manifest.json').exists(), 'Preserve existing delivery'
    prepared, manifest = compose(root)
    for name, values in prepared.items():
        (out / (name + '.jsonl')).write_text(dumps(values), encoding='utf-8', newline='\n')
    manifest['files'] = {n + '.jsonl': sha(out / (n + '.jsonl')) for n in prepared}
    save(out / 'input_manifest.json', manifest)
    return manifest


def verify(root):
    m = read(root / FOLDER / 'input_manifest.json')
    for n, h in m['files'].items():
        assert Path(n).name == n and sha(root / FOLDER / n) == h, 'New input changed: ' + n
    prepared, expected = compose(root)
    assert {k: v for k, v in m.items() if k != 'files'} == expected
    assert all(rows(root / FOLDER / (n + '.jsonl')) == values for n, values in prepared.items())
    return m


def check_loaded_row(source, row):
    # The observed RowPreprocessor Arrow schema may add null loss/loss_scale
    # fields. These are padding, not a changed message; non-null overrides fail.
    normalized = [{k: v for k, v in msg.items() if not (k in ('loss', 'loss_scale') and v is None)}
                  for msg in row['messages']]
    assert normalized == source['messages'] and row['channel'] == source['channel'], 'Loader changed question/answer/mask'


def synthetic_loss_check(template, encoded, proof):
    """Use real SDK trainer/Template loss route with tiny synthetic logits.

    This is a mask/causal-shift gradient check, not a Qwen forward or model eval.
    """
    import torch
    from types import SimpleNamespace
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    batch = template.data_collator([deepcopy(encoded)])
    assert 'channel' not in batch
    actual = batch['labels'][0].tolist()
    assert actual[:len(encoded['labels'])] == encoded['labels'], 'Collator changed labels'
    labels = batch['labels'].clone()
    labels[labels != -100] %= 3
    logits = torch.zeros((*labels.shape, 3), requires_grad=True)

    class Output(dict):
        logits = None
        @property
        def loss(self):
            return self['loss']
        @loss.setter
        def loss(self, value):
            self['loss'] = value

    class SyntheticModel:
        training = True
        model_info = SimpleNamespace(is_moe_model=False)
        def __call__(self, **inputs):
            assert torch.equal(inputs['labels'], labels)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1), ignore_index=-100)
            return Output(loss=loss)

    model = SyntheticModel()
    trainer = SimpleNamespace(template=template, model=model, label_smoother=None,
                              args=SimpleNamespace(enable_dft_loss=False, enable_channel_loss=False,
                                                   past_index=-1, average_tokens_across_devices=False))
    loss = Seq2SeqTrainer.compute_loss(trainer, model, {'labels': labels})
    assert torch.isfinite(loss)
    loss.backward()
    gradient = logits.grad.abs().sum(-1)[0]
    unknown = proof['unknown_token_indices']; known = [i for i in proof['known_token_indices'] if i > 0]
    assert all(i > 0 and gradient[i - 1].item() == 0 for i in unknown), 'Unknown bit received direct loss gradient'
    assert known and all(gradient[i - 1].item() > 0 for i in known), 'Known bit lost gradient'
    return {'synthetic_loss': float(loss.detach()), 'unknown_zero_gradient': len(unknown), 'known_nonzero_gradient': len(known),
            'route': 'real Seq2SeqTrainer.compute_loss + template.compute_sft_loss; synthetic 3-class logits, no Qwen weights'}


def preflight(root, model, out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only; hide GPUs'
    assert not out.exists(), 'Preserve existing report'
    out.mkdir(parents=True)
    m = verify(root); lock = verify_runtime_lock(root)
    assert importlib.metadata.version('ms-swift') == '4.4.2'
    sdkroot = Path(next(iter(importlib.util.find_spec('swift').submodule_search_locations)))
    for name, info in lock['files'].items():
        assert sha(sdkroot.parent / name) == info['sha256'], 'SDK source changed: ' + name
    for name, h in m['tokenizer_config_sha256'].items():
        assert Path(name).name == name and sha(model / name) == h, 'Tokenizer changed'
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.dataset import load_dataset
    import swift_partial_label_plugin_20261020
    unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
    assert unloaded is None
    kwargs = dict(max_length=8192, truncation_strategy='raise', add_non_thinking_prefix=True,
                  loss_scale='default+ignore_empty_think')
    template = get_template(processor, template_type=policy.TEMPLATE, **kwargs)
    plain = get_template(processor, template_type='qwen3_5', **kwargs)
    template.set_mode('train'); plain.set_mode('train')
    report = {'passed': False, 'encodings': [], 'failures': [], 'loss_checks': [], 'inference_checks': 0,
              'real_tokenizer': True, 'swift_version': '4.4.2', 'input_manifest_sha256': sha(root / FOLDER / 'input_manifest.json'),
              'input_manifest': m, 'runtime_source_lock': lock, 'model_weights_loaded': False,
              'model_training_started': False, 'training_ready': False, 'model_calls': 0,
              'code_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()}
    try:
        for name in m['counts']:
            saved = rows(root / FOLDER / (name + '.jsonl'))
            loaded, validation = load_dataset([str(root / FOLDER / (name + '.jsonl'))], split_dataset_ratio=0,
                                               seed=42, shuffle=False, strict=True)
            assert validation is None and len(loaded) == len(saved)
            for source, row in zip(saved, loaded):
                identity = {'file': name, 'unit_id': uid(source)}
                try:
                    check_loaded_row(source, row)
                    prompt, answer = row['messages'][1]['content'], row['messages'][2]['content']
                    baseline = plain.encode(deepcopy(row))
                    expected, proof = policy.mask_encoded(baseline, template.tokenizer, prompt, answer, row['channel'])
                    encoded = template.encode(deepcopy(row))
                    assert encoded['input_ids'] == baseline['input_ids'], 'Mask changed tokenization'
                    assert encoded['labels'] == expected['labels'], 'Plugin lost mask'
                    assert not encoded.get('loss_scale'), 'Expected binary loss labels, not weighted trainer branch'
                    assert 'channel' not in encoded
                    assert len(encoded['input_ids']) <= 8192 and any(x != -100 for x in encoded['labels'])
                    batch = template.data_collator([deepcopy(encoded)])
                    assert batch['labels'][0].tolist()[:len(encoded['labels'])] == encoded['labels']
                    assert 'channel' not in batch
                    if proof['unknown_count']:
                        report['loss_checks'].append({**identity, **synthetic_loss_check(template, encoded, proof)})
                        # Flip only unknown placeholders, keeping question and mask identity.
                        changed = json.loads(answer); ids = policy.ordered_ids(prompt)[1]
                        bits = changed['relation_decisions'].split()
                        unknown_ids = json.loads(row['channel'])['unknown_relation_ids']
                        for j, ident in enumerate(ids):
                            if ident in unknown_ids:
                                bits[j] = '1' if bits[j] == '0' else '0'
                        changed['relation_decisions'] = ' '.join(bits)
                        other_answer = json.dumps(changed, separators=(',', ':'))
                        other_row = deepcopy(row); other_row['messages'][2]['content'] = other_answer
                        other_row['channel'] = policy.supervision_channel(prompt, other_answer, unknown_ids)
                        other = template.encode(other_row)
                        _, other_proof = policy.mask_encoded(plain.encode(deepcopy(other_row)), template.tokenizer, prompt, other_answer, other_row['channel'])
                        assert proof == other_proof and encoded['labels'] == other['labels'], 'Unknown placeholder affects supervised labels'
                    # Identical system/user and tokenizer prefix during formal inference.
                    infer = {'messages': deepcopy(row['messages'][:2])}
                    template.set_mode('transformers'); plain.set_mode('transformers')
                    try:
                        a, b = template.encode(deepcopy(infer)), plain.encode(deepcopy(infer))
                        assert a['input_ids'] == b['input_ids'] and not a.get('labels') and 'channel' not in a
                        generation = a['input_ids']; training = encoded['input_ids']
                        assert training[:len(generation)] == generation, 'Training/inference prompt prefix differs'
                        report['inference_checks'] += 1
                    finally:
                        template.set_mode('train'); plain.set_mode('train')
                    report['encodings'].append({**identity, 'tokens': len(encoded['input_ids']),
                                                'supervised_tokens': sum(x != -100 for x in encoded['labels']),
                                                'unknown_bits_masked': proof['unknown_count'],
                                                'known_bits_supervised': len(proof['known_token_indices'])})
                except Exception as exc:
                    report['failures'].append({**identity, 'error_type': type(exc).__name__, 'error': str(exc)})
                total = len(report['encodings']) + len(report['failures'])
                if total % 100 == 0:
                    print('CHECKED_OR_RECORDED_FAILURE=' + str(total), flush=True)
        assert len(report['encodings']) + len(report['failures']) == 1324
        assert verify(root) == m
        report['original_inputs_unchanged'] = True
        report['max_tokens_by_file'] = {n: max((r['tokens'] for r in report['encodings'] if r['file'] == n), default=0) for n in m['counts']}
        passed = not report['failures'] and len(report['encodings']) == report['inference_checks'] == 1324
        if passed:
            for name in m['counts']:
                observed = sum(r['unknown_bits_masked'] for r in report['encodings'] if r['file'] == name)
                assert observed == m['supervision_coverage'][name]['totals']['unknown_relations_masked']
            assert sum(r['unknown_zero_gradient'] for r in report['loss_checks']) == 37
        report['passed'] = passed
    finally:
        save(out / 'preflight_report.json', report)
    print('PARTIAL_LABEL_PREFLIGHT_OK=' + str(report['passed']).lower())
    print('TRAINING_READY=false')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'verify', 'preflight'))
    p.add_argument('--model', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'build':
        print(json.dumps(build(ROOT)['counts']))
    elif a.action == 'verify':
        print(json.dumps(verify(ROOT)['counts']))
    else:
        assert a.model and a.out
        raise SystemExit(0 if preflight(ROOT, a.model, a.out)['passed'] else 1)
