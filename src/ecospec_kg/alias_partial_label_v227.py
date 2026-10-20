"""Conservatively quarantine unselected literal aliases, without adding positives.

Question, response and graph selection stay exactly v226. Only supervision
metadata and token labels change. No dev/test labels or identifier rules here.
"""
import json
import re
import unicodedata
from . import partial_label_v226 as base

VERSION = 'ecospec-alias-partial-label-v227'
TEMPLATE = 'ecospec_alias_partial_qwen3_5'
build_messages = base.build_messages
ordered_ids = base.ordered_ids
answer_for = base.answer_for
selection_for = base.selection_for
select = base.select
digest = base.digest


def normalized(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


def review_aliases(unit, graph, selections, confirmed_negative_ids=()):
    selected_entities = set().union(*(set(s['selected_entity_ids']) for s in selections))
    selected_relations = set().union(*(set(s['selected_relation_ids']) for s in selections))
    for r in graph['relations']:
        if r['relation_id'] in selected_relations:
            selected_entities.update((r['head_id'], r['tail_id']))
    spans = unit['provenance']['evidence_spans']
    groups = []; unknown_entities = set()
    for entity in graph['entities']:
        if entity['entity_type'] not in ('data_source', 'method'):
            continue
        full = normalized(entity['name'])
        match = re.fullmatch(r'(.+)\(([A-Za-z][A-Za-z0-9_-]*)\)', full)
        if not match:
            continue
        witnesses = [s['span_id'] for s in spans if full in normalized(s['text'])]
        if not witnesses:
            continue
        members = [e for e in graph['entities'] if e['entity_type'] == entity['entity_type']
                   and normalized(e['name']) in {full, *match.groups()}]
        ids = {e['entity_id'] for e in members}
        anchors = ids & selected_entities
        if len(ids) < 2 or not anchors:
            continue
        unknown = ids - selected_entities
        if not unknown:
            continue
        unknown_entities.update(unknown)
        groups.append({'literal': entity['name'], 'entity_type': entity['entity_type'],
                       'members': [{'entity_id': e['entity_id'], 'name': e['name']} for e in members],
                       'positive_anchor_ids': sorted(anchors), 'unknown_entity_ids': sorted(unknown),
                       'source_span_ids': witnesses})
    related = {r['relation_id'] for r in graph['relations']
               if (r['head_id'] in unknown_entities or r['tail_id'] in unknown_entities)
               and r['relation_id'] not in selected_relations}
    retained = related & set(confirmed_negative_ids)
    return {'groups': groups, 'unknown_entity_ids': sorted(unknown_entities),
            'unknown_relation_ids': sorted(related - retained), 'retained_confirmed_negative_ids': sorted(retained)}


def supervision_channel(prompt, answer, unknown_relations, unknown_entities=()):
    # Reuse v226's strict response and membership validation for relations.
    meta = json.loads(base.supervision_channel(prompt, answer, unknown_relations))
    available = set(ordered_ids(prompt)[0])
    if len(unknown_entities) != len(set(unknown_entities)) or not set(unknown_entities) <= available:
        raise ValueError('unknown entities outside candidates or duplicated')
    meta.update(version=VERSION, unknown_entity_ids=sorted(unknown_entities))
    return json.dumps(meta, separators=(',', ':'))


def mask_encoded(encoded, tokenizer, prompt, answer, channel):
    try:
        meta = json.loads(channel)
        expected = {'version', 'prompt_sha256', 'answer_sha256', 'unknown_relation_ids', 'unknown_entity_ids'}
        if set(meta) != expected or meta['version'] != VERSION:
            raise ValueError('incorrect alias supervision metadata')
        if supervision_channel(prompt, answer, meta['unknown_relation_ids'], meta['unknown_entity_ids']) != channel:
            raise ValueError('alias supervision metadata mismatch')
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid alias supervision metadata') from exc
    old_channel = base.supervision_channel(prompt, answer, meta['unknown_relation_ids'])
    result, proof = base.mask_encoded(encoded, tokenizer, prompt, answer, old_channel)
    all_tokens = sorted(proof['known_token_indices'] + proof['unknown_token_indices'])
    decisions = base.decision_positions(prompt, answer)
    if len(all_tokens) != len(decisions):
        raise ValueError('decision alignment count changed')
    entity_tokens = []
    for (ident, field, _), index in zip(decisions, all_tokens):
        if field == 'entity_decisions' and ident in meta['unknown_entity_ids']:
            entity_tokens.append(index); result['labels'][index] = -100
            if result.get('loss_scale') is not None:
                result['loss_scale'][index] = 0
    relation_tokens = proof['unknown_token_indices']
    unknown = sorted(entity_tokens + relation_tokens)
    return result, {**proof, 'unknown_token_indices': unknown,
                    'known_token_indices': [i for i in proof['known_token_indices'] if i not in entity_tokens],
                    'unknown_entity_token_indices': entity_tokens,
                    'unknown_relation_token_indices': relation_tokens, 'unknown_count': len(unknown)}
