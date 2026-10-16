"""Source-local method aliases and explicit, training-only unknown quarantine.

These operations never select a relation. Inference uses the full canonical
graph; unresolved training labels must be removed from the candidate question,
not silently encoded as omitted (negative) answers.
"""
from copy import deepcopy
import re
from .io_utils import stable_id

VERSION = "source-local-method-scope-v1-20261016"


def _compact(value):
    return "".join(value.split())


def _named_method(value):
    # Conservative lexical names only; workflow clauses and malformed sentence
    # boundaries require their own review, not alias normalization.
    return (re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·—\- ]{2,32}(?:法|方程|模型|技术)", value)
            and not re.search(r"的|可使用|首先|采用|使用|基于|进行|由|获取|估算|计算", value))


def canonicalize_methods(unit, candidates):
    """Merge only existing names supported by this unit's explicit spelling.

    A bare method and exactly one parenthesized Latin abbreviation may merge.
    A leading 基于 may be dropped only for a complete method name already in
    the graph. Longer workflow descriptions remain separate.
    """
    from .extractor_v2 import _all_spans, _PROMPT_BODY_FIELDS
    text = _compact(" ".join(str(unit.get(k, "")) for k in _PROMPT_BODY_FIELDS)
                    + " " + " ".join(s.get("text", "") for s in _all_spans(unit).values()))
    methods = {e["name"]: e for e in candidates["entities"] if e["entity_type"] == "method"}
    aliases = {name: name for name in methods}
    for name in methods:
        bare = name.removeprefix("基于")
        lexical = re.sub(r"[（(][A-Za-z][A-Za-z0-9 -]{1,15}[）)]$", "", bare)
        if (bare != name and bare in methods and _named_method(lexical)
                and _compact(name) in text):
            aliases[name] = bare
    expanded = {}
    for name in methods:
        if name.startswith("基于"):
            continue
        m = re.fullmatch(r"(.+(?:方法|方程|模型|技术))[（(]([A-Za-z][A-Za-z0-9 -]{1,15})[）)]", name)
        if m and _named_method(m[1]) and _compact(name) in text:
            expanded.setdefault(m[1], []).append(name)
    for name, target in list(aliases.items()):
        full = expanded.get(target, [])
        if len(full) == 1:
            aliases[name] = full[0]
    result = deepcopy(candidates)
    by_id = {e["entity_id"]: e for e in candidates["entities"]}
    id_map = {e["entity_id"]: methods[aliases[e["name"]]]["entity_id"]
              if e["entity_type"] == "method" else e["entity_id"] for e in candidates["entities"]}
    entities = {}
    for e in candidates["entities"]:
        canonical_id = id_map[e["entity_id"]]
        if canonical_id not in entities:
            entities[canonical_id] = deepcopy(by_id[canonical_id])
        if canonical_id != e["entity_id"]:
            entities[canonical_id]["evidence_span_ids"] = sorted(set(
                entities[canonical_id]["evidence_span_ids"] + e["evidence_span_ids"]))
    relations, relation_map = {}, {}
    for r in candidates["relations"]:
        mapped = deepcopy(r)
        for side in ("head", "tail"):
            e = by_id[id_map[r[side + "_id"]]]
            mapped.update({side + "_id": e["entity_id"], side + "_name": e["name"], side + "_type": e["entity_type"]})
        key = tuple(mapped[k] for k in ("head_name", "head_type", "relation_type", "tail_name", "tail_type"))
        ident = stable_id("prediction-relation-v2", unit["unit_id"], *key)
        mapped["relation_id"] = ident
        relation_map[r["relation_id"]] = ident
        if ident in relations:
            relations[ident]["evidence_span_ids"] = sorted(set(
                relations[ident]["evidence_span_ids"] + mapped["evidence_span_ids"]))
        else:
            relations[ident] = mapped
    # Preserve exact objects/order when the graph is already canonical.
    if all(a == b for a, b in id_map.items()):
        return result, {"version": VERSION, "entity_id_map": id_map, "relation_id_map": relation_map}
    result["entities"] = sorted(entities.values(), key=lambda e: (e["entity_type"], e["name"], e["entity_id"]))
    result["relations"] = sorted(relations.values(), key=lambda r: (r["relation_type"], r["head_name"], r["tail_name"], r["relation_id"]))
    return result, {"version": VERSION, "entity_id_map": id_map, "relation_id_map": relation_map}


def training_selection_view(candidates, target, unresolved_relation_ids):
    """Hide only explicitly reviewed unknown edges from a training question."""
    unknown = set(unresolved_relation_ids)
    ids = {r["relation_id"] for r in candidates["relations"]}
    if not unknown <= ids:
        raise ValueError("Unresolved relation absent from candidate graph")
    if unknown & set(target["selected_relation_ids"]):
        raise ValueError("Unresolved relation conflicts with positive supervision")
    if not set(target["selected_relation_ids"]) <= ids:
        raise ValueError("Positive relation absent from candidate graph")
    view = deepcopy(candidates)
    view["relations"] = [r for r in view["relations"] if r["relation_id"] not in unknown]
    view["no_relation_reason"] = "" if view["relations"] else "no_explicit_schema_v2_relation"
    return view, deepcopy(target)
