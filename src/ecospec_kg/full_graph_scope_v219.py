"""Conditional decision scope with the full source graph retained as context.

This is a question contract, not token-level loss masking. Inference defaults to
scoring every relation; it never imports a training review or annotation table.
"""
import json

from . import extractor_v2 as ex
from .experiment_io_v2 import sha256_json
from .selection_decoding_v2 import candidate_ids, check_selection

VERSION = "ecospec-full-graph-decision-scope-v219"


def build_messages(unit, candidates, unscored_relation_ids=()):
    system, prompt = ex.build_llm_selection_messages(unit, candidates)
    question = json.loads(prompt)
    unscored = list(unscored_relation_ids)
    available = {r["id"] for r in question["candidate_relations"]}
    if len(unscored) != len(set(unscored)):
        raise ValueError("duplicate unscored relation ID")
    if not set(unscored) <= available:
        raise ValueError("unscored relation ID absent from full graph")
    question["decision_scope"] = {
        "version": VERSION,
        "unscored_relation_ids": sorted(unscored),
        "semantics": (
            "候选图完整保留。unscored_relation_ids中的关系仅供上下文参考，本题不要求判断，"
            "不得在答案中输出；未输出不表示关系错误。其余关系及全部实体按原文独立判断。"
            "列表为空时判断全部候选关系。不得因一条关系未评分而排除其端点实体或其他关系。"
        ),
    }
    question["task"] = (
        "返回 selected_relation_ids 和 selected_entity_ids。仅选择原文明确支持且在本题评分范围内的候选；"
        "被关系使用的实体无需重复放入 selected_entity_ids。逐个独立判断全部实体及未被decision_scope"
        "列为unscored的关系，不设选择数量上限。"
    )
    question["selection_policy"]["rules"]["independent_candidate_decision"] = (
        "逐个独立判断全部实体和评分范围内的关系，不设选择数量上限；已选择一个相似候选不能作为"
        "排除其他候选的理由。decision_scope列出的未评分关系保留为上下文，不作为正例或负例。"
    )
    # v2.6 names the inherited semantic rules, not an unchanged question contract.
    # The scope version above records this new contract independently.
    system = system.replace("必须逐项检查候选", "必须按decision_scope逐项检查需判断的候选")
    return system, json.dumps(question, ensure_ascii=False)


def validate_selection(prompt, selection):
    selection = check_selection(json.dumps(selection), *candidate_ids(prompt))
    scope = json.loads(prompt)["decision_scope"]
    if scope["version"] != VERSION:
        raise ValueError("unsupported decision scope version")
    if set(selection["selected_relation_ids"]) & set(scope["unscored_relation_ids"]):
        raise ValueError("selected relation is unscored for this question")
    return selection


def select(provider, unit, candidates):
    """Formal inference has no unscored IDs: all full-graph edges are decided."""
    system, prompt = build_messages(unit, candidates)
    raw = provider.complete(system, prompt)
    selection = validate_selection(prompt, check_selection(raw, *candidate_ids(prompt)))
    relations = {r["relation_id"]: r for r in candidates["relations"]}
    selected = [relations[i] for i in selection["selected_relation_ids"]]
    endpoints = {r[k] for r in selected for k in ("head_id", "tail_id")}
    endpoints.update(selection["selected_entity_ids"])
    prediction = {
        **candidates, "backend": "llm", "decision_scope_version": VERSION,
        "entities": [e for e in candidates["entities"] if e["entity_id"] in endpoints],
        "relations": selected,
        "no_relation_reason": "" if selected else "llm_selected_no_relation",
    }
    return prediction, raw, sha256_json({"system": system, "prompt": prompt})
