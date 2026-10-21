"""Six frozen inference requests on two partially annotated local passages; no training."""
import argparse
from copy import deepcopy
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import final_source_preflight_20261021 as cpu
import evaluate_source_supplement_pair_20261021 as prior
from ecospec_kg import candidate_profile_v228 as profile, partial_label_v226 as scope
from ecospec_kg.binary_decoding_v230 import VERSION
from ecospec_kg.definition_evidence_v223 import attach_definition_evidence
from ecospec_kg.evaluation_v2 import _entity_key, _relation_key
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.io_utils import read_jsonl, write_jsonl
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction, _all_spans
from ecospec_kg.structural_selection_v2 import merge_structural_relations, formula_path_metrics

FOLDER = 'experiments/final_source_eval_v233_20261021'
PREFLIGHT = FOLDER + '/actual_preflight.json'
PREFLIGHT_SHA = 'f28d5b4a61a960f5f0bcba8dcb453ddbdf4f03172fe62ea02d47a9aafba70128'
INPUT_SHA = '92f4aa5fcfcf9e50a138fa9da9500e5586b882b5d89a4008ab57e0bd2b59a6c8'
PAIR = Path('/home/hello/szl/eco-spec-runs/lora/source_supplement_seed42_pair_20261021-103244-529738')
MODES = ('raw', 'structure', 'evidence', 'structure_evidence')
read, save, sha = prior.read, prior.save, prior.sha


def prompt_provenance(unit):
    candidates = profile.candidates_for_unit(unit)
    system, prompt = scope.build_messages(unit, candidates)
    return dict(prompt_hash=sha256_json(dict(system=system, prompt=prompt)),
                candidate_hash=sha256_json({k:candidates[k] for k in ('entities', 'relations')}),
                entity_candidates=len(candidates['entities']), relation_candidates=len(candidates['relations']))


def verify_inputs(root):
    assert (root / PREFLIGHT).is_file() and sha(root / PREFLIGHT) == PREFLIGHT_SHA, 'Preflight report changed'
    actual = read(root / PREFLIGHT)
    assert actual['passed'] and actual['real_tokenizer'] and actual['original_inputs_unchanged']
    assert actual['checked_units'] == 2 and actual['swift_version'] == '4.4.2'
    assert actual['model_calls'] == 0 and not actual['model_weights_loaded'] and not actual['training_started']
    assert actual['code_commit'] == '5631faf877a537a254059d138f337f3447a6a88b'
    assert actual['input_manifest_sha256'] == INPUT_SHA == sha(root / cpu.FOLDER / 'manifest.json')
    packet = cpu.verify_packet(root)
    for rel, digest in packet['protocol']['frozen']['source_sha256_lf'].items():
        assert hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest, rel
    for rel, digest in read(root / FOLDER / 'runtime_files.json').items():
        assert hashlib.sha256((root / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest, 'Evaluation source changed: ' + rel
    policy = read(root / FOLDER / 'scoring_policy.json')
    assert policy['mode'] == 'known_positive_only_no_implicit_negatives'
    assert policy['model_requests'] == 6 and policy['postprocessing_requests'] == 0
    assert policy['known_positive_entities'] == sum(len(g['entities']) for g in packet['gold']) == 21
    assert policy['known_positive_relations'] == sum(len(g['relations']) for g in packet['gold']) == 17
    assert policy['unknown_scope'] == packet['protocol']['unresolved_scope']
    assert actual['candidate_diagnostic'] == json.loads(json.dumps(cpu.candidate_diagnostic(packet['units'], packet['gold'])))
    assert len(actual['records']) == len(packet['units']) == 2
    for unit, record in zip(packet['units'], actual['records']):
        p = prompt_provenance(unit)
        assert unit['unit_id'] == record['unit_id']
        assert all(p[k] == record[k] for k in ('prompt_hash', 'entity_candidates', 'relation_candidates')), 'Question changed'
        assert record['prompt_tokens'] + record['worst_case_output_tokens'] <= 8192
        assert record['worst_case_output_tokens'] <= 3072 and len(record['prefix_sha256']) == 64
    return packet, actual


def semantic_keys(row):
    uid = row['unit_id']
    return dict(entities={_entity_key(uid, e) for e in row['entities']},
                relations={_relation_key(uid, r) for r in row['relations']})


def partial_scores(gold, predictions):
    """Positive recovery only. Every unmatched prediction stays visible, never an implicit FP."""
    result = {'scope': 'known_positive_only; extra predictions require independent adjudication'}
    for field, key in (('entities', _entity_key), ('relations', _relation_key)):
        g = {key(row['unit_id'], x) for row in gold for x in row[field]}
        p = {key(row['unit_id'], x) for row in predictions for x in row[field]}
        result[field] = dict(known_positive_support=len(g), recovered_known_positive=len(g & p),
                             missed_known_positive=len(g - p), known_positive_recall=len(g & p)/len(g) if g else None,
                             prediction_count=len(p), unadjudicated_extra_count=len(p - g),
                             unadjudicated_extra_keys=sorted(p - g), missing_known_positive_keys=sorted(g - p),
                             false_positive=None, precision=None, f1=None)
        if field == 'relations':
            labels = sorted({k[3] for k in g | p})
            result['relation_by_type'] = {t:dict(known_positive_support=sum(k[3] == t for k in g),
                recovered_known_positive=sum(k[3] == t for k in g & p),
                unadjudicated_extra_count=sum(k[3] == t for k in p - g)) for t in labels}
            result['macro_relation_f1'] = None
    return result


def _path_complete(unit, gold, prediction, required, overlap=False):
    uid = unit['unit_id']
    pe = {_entity_key(uid, e):e for e in prediction['entities']}
    pr = {_relation_key(uid, r):r for r in prediction['relations']}
    ge = {_entity_key(uid, e):e for e in gold['entities']}
    valid_spans = set(_all_spans(unit))
    for edge in required:
        actual = pr.get(_relation_key(uid, edge))
        if actual is None:
            return False
        objects = [(actual, edge)]
        for side in ('head', 'tail'):
            k = (uid, edge[side+'_name'], edge[side+'_type'])
            if k not in pe:
                return False
            objects.append((pe[k], ge[k]))
        for item, reference in objects:
            spans = set(item.get('evidence_span_ids', []))
            if not spans or not spans <= valid_spans:
                return False
            if overlap and not spans & set(reference.get('evidence_span_ids', [])):
                return False
    return True


def _supported(details):
    n = len(details)
    return dict(support=n, complete=sum(d['complete'] for d in details) if n else None,
                rate=sum(d['complete'] for d in details)/n if n else None, details=details)


def paths(units, gold, predictions):
    by_id = {p['unit_id']:p for p in predictions}
    indicator = formula_path_metrics(units, gold, predictions)
    # The old metric's expanded path can equal its core when no extras exist.
    # Do not present that as evidence of a supported method/source extension.
    indicator = {k:v for k,v in indicator.items() if not k.startswith('expanded_')}
    indicator['details'] = [{k:v for k,v in d.items() if not k.startswith('expanded_')} for d in indicator['details']]
    variables, indicator_methods, variable_methods, sources = [], [], [], []
    for unit, annotation in zip(units, gold):
        uid = unit['unit_id']; prediction = by_id[uid]; rels = annotation['relations']
        indicator_formulas = {r['tail_name'] for r in rels if (r['head_type'], r['relation_type'], r['tail_type']) == ('assessment_indicator','calculated_by','formula')}
        for root in rels:
            if root['relation_type'] != 'calculated_by' or root['tail_type'] != 'formula':
                continue
            kind = root['head_type']
            if kind == 'model_variable' and root['tail_name'] in indicator_formulas:
                continue  # Do not double-count the indicator's own output inverse.
            if kind not in ('model_variable', 'assessment_indicator'):
                continue
            roles = [r for r in rels if r['head_type'] == 'formula' and r['head_name'] == root['tail_name'] and r['relation_type'] in ('has_input', 'has_output')]
            if {r['relation_type'] for r in roles} != {'has_input', 'has_output'}:
                continue
            outputs = {r['tail_name'] for r in roles if r['relation_type'] == 'has_output'}
            inverse = [r for r in rels if r['head_type'] == 'model_variable' and r['head_name'] in outputs and r['relation_type'] == 'calculated_by' and r['tail_name'] == root['tail_name']]
            required = list({_relation_key(uid,r):r for r in [root,*roles,*inverse]}.values())
            detail = dict(root=list(_relation_key(uid,root)), required_edges=len(required),
                          complete=_path_complete(unit,annotation,prediction,required),
                          reference_span_overlap_complete=_path_complete(unit,annotation,prediction,required,True))
            if kind == 'model_variable':
                variables.append(detail)
            nodes = {(r[s+'_name'],r[s+'_type']) for r in required for s in ('head','tail')}
            for relation_type, bucket in [('obtained_by', indicator_methods if kind == 'assessment_indicator' else variable_methods), ('sourced_from', sources)]:
                extra = [r for r in rels if r['relation_type'] == relation_type and (r['head_name'],r['head_type']) in nodes]
                if extra:
                    bucket.append(dict(root=detail['root'], extension_edges=len(extra),
                        complete=_path_complete(unit,annotation,prediction,required+extra),
                        reference_span_overlap_complete=_path_complete(unit,annotation,prediction,required+extra,True)))
    return dict(indicator=indicator, variable=_supported(variables),
                indicator_method_extension=_supported(indicator_methods),
                variable_method_extension=_supported(variable_methods), data_source_extension=_supported(sources),
                note='Presence/coverage of annotated subgraphs, not path precision or evidence entailment; source support zero is unavailable.')


def empty_prediction(unit):
    """Only used by local deterministic tests; the launcher always calls the real extractor."""
    c = profile.candidates_for_unit(unit)
    return {**c, 'entities':[], 'relations':[], 'no_relation_reason':'llm_selected_no_relation',
            'experiment_id':'fixture', 'config_hash':'fixture', 'candidate_hash':'fixture', 'model':'fixture', 'seed':42}


def postprocess(units, predictions):
    assert len(predictions) == len(units) and len({p['unit_id'] for p in predictions}) == len(units)
    by_id = {p['unit_id']:p for p in predictions}
    modes = {mode:dict(predictions=[], proofs=[]) for mode in MODES}
    for unit in units:
        raw = deepcopy(by_id[unit['unit_id']]); candidate = profile.candidates_for_unit(unit)
        structured, proof = merge_structural_relations(unit, candidate, raw)
        evidence, eproof = attach_definition_evidence(unit, candidate, raw)
        both, bproof = attach_definition_evidence(unit, candidate, structured)
        assert semantic_keys(raw) == semantic_keys(evidence)
        assert semantic_keys(structured) == semantic_keys(both)
        for mode, prediction, proofs in [('raw',raw,{}), ('structure',structured,proof), ('evidence',evidence,eproof), ('structure_evidence',both,dict(structure=proof,evidence=bproof))]:
            failures = _validate_unit_prediction(unit,prediction)
            assert not failures, f'Invalid {mode} prediction: {failures}'
            modes[mode]['predictions'].append(prediction)
            modes[mode]['proofs'].append(dict(unit_id=unit['unit_id'], **proofs))
    return modes


def evaluate(root, out, base_url, capability):
    assert not out.exists(), 'Preserve existing results'
    packet, actual = verify_inputs(root)
    units, gold = packet['units'], packet['gold']
    provenance = {u['unit_id']:prompt_provenance(u) for u in units}
    cfg = {'experiment_id':'final-partial-sources-v233', 'backend':'llm', 'model':prior.MODELS['base'],
           **read(root / prior.PROTOCOL)['selection_protocol'], 'selection_decoding':VERSION,
           'decision_scope_version':scope.VERSION, 'use_schema':True, 'use_layout':True, 'use_evidence':True}
    out.mkdir(parents=True)
    save(out/'protocol.json',dict(scoring_policy=read(root/FOLDER/'scoring_policy.json'),
         preflight_sha256=sha(root/PREFLIGHT), input_manifest_sha256=INPUT_SHA,
         capability=capability, prompt_provenance=provenance,
         evaluation_files_sha256=read(root/FOLDER/'runtime_files.json')))
    results = {}
    for label, model in prior.MODELS.items():
        dest = out/label
        print(f'Evaluating {label}: 2 units; frozen model={model}',flush=True)
        config = {**cfg,'model':model}
        manifest = prior.extract_profile(units,dest,config,base_url)
        assert manifest['status'] == 'complete' and manifest['summary'] == dict(unit_count=2,success_count=2,error_count=0)
        predictions = read_jsonl(dest/'predictions.jsonl')
        assert len(predictions) == 2 and {p['unit_id'] for p in predictions} == set(provenance)
        for p in predictions:
            assert p['model'] == model and p['seed'] == 42
            assert all(p[k] == provenance[p['unit_id']][k] for k in ('candidate_hash','prompt_hash'))
        modes = postprocess(units,predictions)
        results[label] = {}
        for mode, data in modes.items():
            folder = dest/mode; folder.mkdir()
            write_jsonl(folder/'predictions.jsonl',data['predictions'])
            write_jsonl(folder/'proofs.jsonl',data['proofs'])
            report = dict(scores=partial_scores(gold,data['predictions']), paths=paths(units,gold,data['predictions']),
                by_source={u['provenance']['standard_code']:dict(unit_id=u['unit_id'],
                    scores=partial_scores([g],[p]),paths=paths([u],[g],[p]))
                    for u,g,p in zip(units,gold,data['predictions'])})
            save(folder/'report.json',report)
            results[label][mode] = report
    verify_inputs(root)
    summary = dict(status='complete',scope='2 local passages from 2 sources; partially reviewed positives',
         sources=2, units=2, printed_formulas=3, model_requests=6, postprocessing_requests=0,
         complete_precision_f1_available=False, extra_predictions_need_human_adjudication=True,
         automatic_three_seed_launch=False, three_seed_eligible=False, training_started=False,
         previous_complete_dev_gate_overridden=False, evidence_overlap_is_entailment=False,
         candidate_diagnostic=actual['candidate_diagnostic'], results=results)
    save(out/'summary.json',summary)
    print('FINAL_SOURCE_EVALUATION_OK=true MODEL_REQUESTS=6 TRAINING_STARTED=false',flush=True)
    return summary


def require_idle_gpu():
    gpu = subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
    assert gpu.returncode == 0 and not gpu.stdout.strip(), f'GPU0 unavailable or occupied: {gpu.stdout} {gpu.stderr}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Preserve existing results'
    args.out.mkdir(parents=True)
    try:
        packet, actual = verify_inputs(ROOT)
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('WORLD_SIZE','1') == '1'
        assert importlib.metadata.version('ms-swift') == '4.4.2'
        manifest = prior.inspect_pair(ROOT,PAIR)
        prior.training.verify_sdk(ROOT)
        model_path = Path(manifest['arms']['control']['command'][3])
        for name,digest in manifest['verification']['manifest']['tokenizer_config_sha256'].items():
            assert sha(model_path/name) == digest, 'Tokenizer changed: ' + name
        for arm in ('control','repair'):
            assert manifest['arms'][arm]['completion'] == packet['protocol']['frozen']['checkpoints'][arm], 'Checkpoint changed'
        command = prior.deploy_command(ROOT,manifest,8018)
        from swift.arguments import DeployArguments
        supported = {f.name for f in fields(DeployArguments)}
        assert all(x[2:] in supported for x in command if x.startswith('--')), 'Unsupported deploy option'
        require_idle_gpu()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',8018))
        save(args.out/'launch.json',dict(command=command, pair_manifest_sha256=prior.EXPECTED_PAIR_SHA,
             code_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
             preflight_sha256=PREFLIGHT_SHA,scoring_policy_sha256=sha(ROOT/FOLDER/'scoring_policy.json')))
        with prior.owned_service(command,args.out/'deploy.log','http://127.0.0.1:8018/v1'):
            capability = prior.check_service('http://127.0.0.1:8018/v1')
            evaluate(ROOT,args.out/'evaluation','http://127.0.0.1:8018/v1',capability)
        assert prior.inspect_pair(ROOT,PAIR) == manifest, 'Training artifacts changed during evaluation'
        verify_inputs(ROOT)
        save(args.out/'completion.json',dict(status='complete',model_requests=6,adapter_hashes_unchanged=True,training_started=False))
    except BaseException as error:
        save(args.out/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error)))
        raise


if __name__ == '__main__':
    main()
