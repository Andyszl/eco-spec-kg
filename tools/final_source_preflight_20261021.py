"""CPU-only check of new, partially reviewed source passages; never infer/train."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess

FOLDER = Path('experiments/final_sources_v232_20261021')


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def verify_packet(root):
    folder = root / FOLDER
    manifest = read(folder / 'manifest.json')
    for rel, expected in manifest['files'].items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and '\\' not in rel and ':' not in rel
        assert sha(folder / rel) == expected, 'Changed material: ' + rel
    protocol = read(folder / 'protocol.json')
    units = rows(folder / 'source_units.jsonl')
    gold = rows(folder / 'reviewed_annotations.jsonl')
    assert len(units) == len(gold) == protocol['primary_passages'] == 2
    assert [u['unit_id'] for u in units] == [g['unit_id'] for g in gold]
    source_manifest = read(folder / 'source_manifest.json')
    assert {s['sha256'] for s in source_manifest['sources']} == {u['provenance']['source_sha256'] for u in units}
    for source in source_manifest['sources']:
        assert sha(folder / source['file']) == source['sha256']
    for u, g in zip(units, gold):
        spans = {s['span_id'] for s in u['provenance']['evidence_spans']}
        endpoints = {e['entity_id']: e for e in g['entities']}
        assert len(endpoints) == len(g['entities'])
        for item in [*g['entities'], *g['relations']]:
            assert item['evidence_span_ids'] and set(item['evidence_span_ids']) <= spans
        for r in g['relations']:
            for side in ('head', 'tail'):
                e = endpoints[r[side + '_id']]
                assert (r[side + '_name'], r[side + '_type']) == (e['name'], e['entity_type'])
    assert protocol['scope_before_model_outputs'] and protocol['unresolved_not_implicit_negative']
    assert not protocol['exhaustive_full_source_gold'] and not protocol['final_inference_ready']
    return dict(units=units, gold=gold, protocol=protocol, manifest=manifest)


def candidate_diagnostic(units, gold):
    from ecospec_kg.candidate_profile_v228 import candidates_for_unit
    from ecospec_kg.evaluation_v2 import _entity_key, _relation_key
    from ecospec_kg.structural_selection_v2 import structural_relation_proofs
    records = []
    for u, g in zip(units, gold):
        c = candidates_for_unit(u)
        sets = {}
        for field, key in (('entities', _entity_key), ('relations', _relation_key)):
            expected = {key(u['unit_id'], x) for x in g[field]}
            available = {key(u['unit_id'], x) for x in c[field]}
            sets['gold_' + field] = len(expected)
            sets['candidate_' + field] = len(available)
            sets['covered_' + field] = len(expected & available)
            sets['missing_' + field] = sorted(expected - available)
        proofs = structural_relation_proofs(u, c)
        relation_map = {r['relation_id']: r for r in c['relations']}
        proof_keys = {_relation_key(u['unit_id'], relation_map[p['relation_id']]) for p in proofs}
        gold_keys = {_relation_key(u['unit_id'], r) for r in g['relations']}
        records.append(dict(unit_id=u['unit_id'], **sets, structural_proofs=len(proofs),
                            structural_known_positive=len(proof_keys & gold_keys),
                            structural_outside_known_positive=sorted(proof_keys - gold_keys)))
    return dict(model_calls=0, model_performance_measured=False, records=records,
                note='Candidate coverage and syntax-only proofs, not model scores. Unknown annotations are not negative labels.')


def preflight(root, model, out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only: hide GPUs'
    assert not out.exists(), 'Preserve old reports'
    out.mkdir(parents=True)
    report = dict(passed=False, model_calls=0, model_weights_loaded=False, training_started=False,
                  final_inference_ready=False, records=[])
    try:
        packet = verify_packet(root)
        report['input_manifest_sha256'] = sha(root / FOLDER / 'manifest.json')
        report['code_commit'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        for rel, expected in packet['protocol']['frozen']['source_sha256_lf'].items():
            assert hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected, 'Frozen component changed: ' + rel
        report['candidate_diagnostic'] = candidate_diagnostic(packet['units'], packet['gold'])
        # Reuse the previously audited runtime/tokenizer locks, not a fresh SDK installation.
        import train_source_supplement_pair_20261021 as training
        approved = training.verify_approved(root, root / training.REPORT)
        runtime = training.inputs.prior.verify_runtime_lock(root)
        assert importlib.metadata.version('ms-swift') == '4.4.2'
        sdk = Path(next(iter(importlib.util.find_spec('swift').submodule_search_locations)))
        for rel, entry in runtime['files'].items():
            assert sha(sdk.parent / rel) == entry['sha256'], 'SDK changed: ' + rel
        for name, expected in approved['manifest']['tokenizer_config_sha256'].items():
            assert Path(name).name == name and sha(model / name) == expected, 'Tokenizer changed: ' + name
        from swift.model import get_model_processor
        from swift.template import get_template
        import swift_alias_partial_label_plugin_20261020  # noqa: F401
        from ecospec_kg import alias_partial_label_v227 as labels
        from source_supplement_dev_preflight_20261021 import check_prompts
        unloaded, processor = get_model_processor(str(model), load_model=False, download_model=False)
        assert unloaded is None
        options = dict(max_length=8192, truncation_strategy='raise', add_non_thinking_prefix=True,
                       enable_thinking=False, loss_scale='default+ignore_empty_think')
        plain = get_template(processor, template_type='qwen3_5', **options)
        masked = get_template(processor, template_type=labels.TEMPLATE, **options)
        report['records'] = check_prompts(packet['units'], plain, masked)
        assert len(report['records']) == 2
        report.update(passed=True, real_tokenizer=True, checked_units=2, swift_version='4.4.2',
                      maximum_prompt_plus_output=max(r['prompt_tokens'] + r['worst_case_output_tokens'] for r in report['records']))
        verify_packet(root)
        report['original_inputs_unchanged'] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(out / 'report.json', report)
    print('FINAL_SOURCE_PREFLIGHT_OK=true CHECKED_UNITS=2 MODEL_CALLS=0', flush=True)
    print('FINAL_INFERENCE_READY=false TRAINING_READY=false', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    preflight(Path(__file__).resolve().parents[1], args.model, args.out)
