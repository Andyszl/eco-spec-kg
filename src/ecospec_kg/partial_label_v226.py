"""Full-graph binary decisions; review metadata never appears in the question.

Unknown bits use a fixed placeholder in teacher forcing and have label -100.
This removes direct negative loss, not their indirect autoregressive context.
"""
from copy import deepcopy
import hashlib
import json
import re

from . import extractor_v2 as ex
from .experiment_io_v2 import sha256_json
from .selection_decoding_v2 import check_selection

VERSION = 'ecospec-partial-label-v226'
TEMPLATE = 'ecospec_partial_qwen3_5'
FIELDS = ('entity_decisions', 'relation_decisions')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_messages(unit, candidates):
    system, text = ex.build_llm_selection_messages(unit, candidates)
    q = json.loads(text)
    q.pop('output_schema')
    q['task'] = ('按候选列表顺序逐项独立判断原文是否支持该实体或关系。'
                 '只返回一个JSON对象，含entity_decisions和relation_decisions两个字符串；'
                 '每个候选对应一位1或0，位之间用一个空格分隔。1表示支持，0表示不支持。'
                 '分别严格覆盖全部实体和全部关系，不省略候选，不限制选择数量。'
                 '实体位表示最终图中保留的实体，包含所选关系的两端实体；'
                 '端点也须填1，不能因关系已选择而将端点填0。context_only实体仅可随明确支持的关系保留。')
    q['response_schema'] = {k: 'space-separated binary decisions in candidate order' for k in FIELDS}
    q['selection_policy']['rules']['independent_candidate_decision'] = (
        '逐项判断全部候选。相似候选互不排斥，选中一个不是排除另一个的理由。关系端点也须判断。')
    q['decision_format_version'] = VERSION
    # The inherited system contains the old output instructions. Replace the
    # output-only paragraph with an explicit final instruction, without dropping
    # the original semantic policy from the question.
    system = ('你是生态规范知识抽取审核员。仅依据给出的原文、结构证据和语义规则判断，'
              '不得引入原文没有的知识。输出格式严格以用户问题的task和response_schema为准；'
              '只返回两个字符串字段entity_decisions和relation_decisions的JSON。')
    return system, json.dumps(q, ensure_ascii=False)


def ordered_ids(prompt):
    q = json.loads(prompt)
    values = [[c['id'] for c in q[k]] for k in ('candidate_entities', 'candidate_relations')]
    if any(len(v) != len(set(v)) for v in values):
        raise ValueError('duplicate candidate identity')
    return values


def effective_selection(graph, selection):
    """Convert old standalone entity IDs to the actual graph membership target."""
    rels = [r for r in graph['relations'] if r['relation_id'] in selection['selected_relation_ids']]
    if len(rels) != len(selection['selected_relation_ids']):
        raise ValueError('selection references absent or duplicate relations')
    entities = set(selection['selected_entity_ids']) | {r[k] for r in rels for k in ('head_id', 'tail_id')}
    ids = [e['entity_id'] for e in graph['entities'] if e['entity_id'] in entities]
    if set(ids) != entities:
        raise ValueError('selection references absent entities')
    return {'selected_entity_ids': ids, 'selected_relation_ids': list(selection['selected_relation_ids'])}


def answer_for(prompt, selection):
    checked = check_selection(json.dumps(selection), *ordered_ids(prompt))
    return json.dumps({f: ' '.join('1' if i in checked[k] else '0' for i in ids)
                       for f, ids, k in zip(FIELDS, ordered_ids(prompt),
                                           ('selected_entity_ids', 'selected_relation_ids'))}, separators=(',', ':'))


def selection_for(prompt, answer):
    try:
        def unique(pairs):
            d = dict(pairs)
            if len(d) != len(pairs):
                raise ValueError('duplicate response key')
            return d
        parsed = json.loads(answer, object_pairs_hook=unique)
        if not isinstance(parsed, dict) or set(parsed) != set(FIELDS):
            raise ValueError('incorrect decision schema')
        out = {}
        for f, ids, k in zip(FIELDS, ordered_ids(prompt), ('selected_entity_ids', 'selected_relation_ids')):
            value = parsed[f]
            if not isinstance(value, str) or (value != '' and not re.fullmatch('[01]( [01])*', value)):
                raise ValueError('decisions must be spaced binary digits')
            bits = value.split() if value else []
            if len(bits) != len(ids):
                raise ValueError('decision count differs from candidate count')
            out[k] = [i for i, bit in zip(ids, bits) if bit == '1']
        return out
    except (TypeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError('invalid decision response') from e


def supervision_channel(prompt, answer, unknown):
    selection_for(prompt, answer)
    ids = ordered_ids(prompt)[1]
    if len(unknown) != len(set(unknown)) or not set(unknown) <= set(ids):
        raise ValueError('unknown IDs outside candidates or duplicated')
    return json.dumps({'version': VERSION, 'prompt_sha256': digest(prompt),
                       'answer_sha256': digest(answer), 'unknown_relation_ids': sorted(unknown)}, separators=(',', ':'))


def decision_positions(prompt, answer):
    parsed = selection_for(prompt, answer)
    # Only canonical serialization is accepted in training, so offset discovery
    # cannot silently bind to an escaped string or a repeated field name.
    if answer_for(prompt, parsed) != answer:
        raise ValueError('noncanonical training answer')
    positions = []
    for f, ids in zip(FIELDS, ordered_ids(prompt)):
        start = answer.index('"' + f + '":"') + len(f) + 4
        positions.extend((ident, f, start + 2 * j) for j, ident in enumerate(ids))
    return positions


def mask_encoded(encoded, tokenizer, prompt, answer, channel):
    try:
        meta = json.loads(channel)
        if set(meta) != {'version', 'prompt_sha256', 'answer_sha256', 'unknown_relation_ids'}:
            raise ValueError('incorrect supervision metadata')
        if meta['version'] != VERSION or meta['prompt_sha256'] != digest(prompt) or meta['answer_sha256'] != digest(answer):
            raise ValueError('supervision metadata mismatch')
        if supervision_channel(prompt, answer, meta['unknown_relation_ids']) != channel:
            raise ValueError('noncanonical supervision metadata')
    except (TypeError, KeyError, json.JSONDecodeError) as e:
        raise ValueError('invalid supervision metadata') from e
    tok = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tok['input_ids'], tok['offset_mapping']
    full = encoded['input_ids']
    starts = [i for i, token in enumerate(full) if ids and token == ids[0] and full[i:i + len(ids)] == ids]
    if len(starts) != 1 or not ids or len(ids) != len(offsets):
        raise ValueError('assistant token sequence must match exactly once; no approximate alignment')
    start = starts[0]
    known, unknown, used = [], [], set()
    result = deepcopy(encoded)
    if len(result['labels']) != len(full):
        raise ValueError('incorrect encoded labels length')
    for ident, field, char in decision_positions(prompt, answer):
        hits = [i for i, (lo, hi) in enumerate(offsets) if lo <= char < hi]
        if len(hits) != 1 or hits[0] in used:
            raise ValueError('decision values not individually tokenized')
        local = hits[0]; lo, hi = offsets[local]
        # A token may contain surrounding whitespace, but never another bit,
        # a quote or a delimiter. Fail rather than masking a known neighbour.
        if answer[lo:hi].strip() not in ('0', '1'):
            raise ValueError('decision token crosses a field boundary')
        used.add(local); index = start + local
        if result['labels'][index] != full[index]:
            raise ValueError('decision token already missing from original supervision')
        if field == 'relation_decisions' and ident in meta['unknown_relation_ids']:
            unknown.append(index); result['labels'][index] = -100
            if result.get('loss_scale') is not None:
                result['loss_scale'][index] = 0
        else:
            known.append(index)
    return result, {'unknown_token_indices': unknown, 'known_token_indices': known,
                    'answer_start_token': start, 'unknown_count': len(unknown)}


def select(provider, unit, candidates):
    system, prompt = build_messages(unit, candidates)
    raw = provider.complete(system, prompt)
    chosen = selection_for(prompt, raw)
    rels = [r for r in candidates['relations'] if r['relation_id'] in chosen['selected_relation_ids']]
    endpoints = set(chosen['selected_entity_ids']) | {r[k] for r in rels for k in ('head_id', 'tail_id')}
    prediction = {**candidates, 'backend': 'llm', 'decision_format_version': VERSION,
                  'entities': [e for e in candidates['entities'] if e['entity_id'] in endpoints],
                  'relations': rels, 'no_relation_reason': '' if rels else 'llm_selected_no_relation'}
    return prediction, raw, sha256_json({'system': system, 'prompt': prompt})
