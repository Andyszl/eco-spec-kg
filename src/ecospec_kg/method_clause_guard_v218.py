"""Experimental lexical postfilter; no semantic selection or candidate-ID rules."""
from copy import deepcopy
import re
from .experiment_io_v2 import sha256_json

VERSION='training-derived-method-clause-guard-v1-20261018'
RULES=(
    ('leading_fragment',r'^(?:[）)]|的)'),
    ('deictic_formula',r'^如下公式'),
    ('descriptive_clause',r'可采用|可根据|即采用|常用|目前已|等方法进行计算'),
)


def rejection_reason(name):
    text=re.sub(r'\s+','',name)
    return next((label for label,pattern in RULES if re.search(pattern,text)),None)


def filter_prediction(prediction):
    """Drop clause-like method entities and incident edges, never add/relabel edges."""
    result=deepcopy(prediction)
    reasons={e['entity_id']:rejection_reason(e['name']) for e in prediction['entities'] if e['entity_type']=='method'}
    reasons={i:r for i,r in reasons.items() if r}
    removed=[r['relation_id'] for r in prediction['relations'] if {r['head_id'],r['tail_id']}&reasons.keys()]
    proof={'policy':VERSION,'baseline_prediction_sha256':sha256_json(prediction),
        'removed_entity_ids':sorted(reasons),'removed_relation_ids':sorted(removed),'reasons':reasons,'model_calls':0}
    if reasons:
        result['entities']=[e for e in result['entities'] if e['entity_id'] not in reasons]
        result['relations']=[r for r in result['relations'] if r['relation_id'] not in removed]
        if not result['relations']:result['no_relation_reason']=result['no_relation_reason'] or 'no_relations_after_method_clause_guard'
        result.update(config_hash=sha256_json({'baseline_config_hash':prediction['config_hash'],'postprocessor':VERSION}),
                      baseline_config_hash=prediction['config_hash'],postprocessor=VERSION)
    return result,proof
