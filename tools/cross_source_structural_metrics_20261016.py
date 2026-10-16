"""Same frozen scoring/merge algorithm with the complete shared v214 candidate profile."""
import hashlib
import json
from pathlib import Path
import sys
from ecospec_kg.evaluation_v2 import _entity_key, _relation_key, _prf
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.candidate_profile_v214 import candidates_for_unit
from ecospec_kg.prediction_validation_v2 import _validate_unit_prediction
from ecospec_kg.structural_selection_v2 import merge_structural_relations, formula_path_metrics, STRUCTURAL_POLICY_VERSION

def scores(gold, predictions):
    result = {}
    for field, key in (('entities', _entity_key), ('relations', _relation_key)):
        g = {key(r['unit_id'], x) for r in gold for x in r[field]}
        p = {key(r['unit_id'], x) for r in predictions for x in r[field]}
        result[field] = _prf(g, p)
        if field == 'relations':
            labels = sorted({k[3] for k in g | p})
            result['relation_by_type'] = {t: _prf({k for k in g if k[3] == t}, {k for k in p if k[3] == t}) for t in labels}
            result['macro_relation_f1'] = round(sum((v['f1'] for v in result['relation_by_type'].values())) / len(labels), 6) if labels else None
    return result

def compare_one(units, gold, cfg, predictions):
    by_id = {p['unit_id']: p for p in predictions}
    merged, audits = ([], [])
    new_config = {'baseline_config': cfg, 'structural_policy': STRUCTURAL_POLICY_VERSION, 'model_calls': 0}
    for u in units:
        candidate = candidates_for_unit(u)
        before = by_id[u['unit_id']]
        after, audit = merge_structural_relations(u, candidate, before)
        after.update(backend='hybrid', experiment_id=before['experiment_id'] + '-' + STRUCTURAL_POLICY_VERSION, config_hash=sha256_json(new_config), baseline_config_hash=before['config_hash'], structural_policy=STRUCTURAL_POLICY_VERSION)
        assert not _validate_unit_prediction(u, after), 'Hybrid evidence/schema validation failed'
        merged.append(after)
        audits.append({'unit_id': u['unit_id'], **audit})
    keys = lambda data: {_relation_key(r['unit_id'], x) for r in data for x in r['relations']}
    g, a, b = (keys(gold), keys(predictions), keys(merged))
    assert a <= b, 'Additive module removed baseline relations'
    result = {'before': scores(gold, predictions), 'after': scores(gold, merged), 'added_correct_relations': len(b - a & g), 'added_false_positive_relations': len(b - a - g), 'recovered_relation_keys': sorted(b - a & g), 'added_false_positive_keys': sorted(b - a - g), 'formula_paths_before': formula_path_metrics(units, gold, predictions), 'formula_paths_after': formula_path_metrics(units, gold, merged), 'indicator_formula_edges': [{'key': list(k), 'before': k in a, 'after': k in b} for k in sorted(g) if (k[2], k[3], k[5]) == ('assessment_indicator', 'calculated_by', 'formula')]}
    return (result, merged, audits)
