"""Optional, additive selection from explicit local equations; never reads gold.

The module retains only existing candidates supported by formula syntax and
local variable definitions. It does not apply source-specific alias overlays,
heading-to-formula heuristics, or automatic semantic method/scope selections.
"""
from __future__ import annotations

from copy import deepcopy
import re

from .extractor_v2 import _formula_aliases, _formula_expression_body, _formula_output_bindings, _split_variable
from .formula_symbols import formula_symbols, normalize_symbol
from .prediction_validation_v2 import _all_spans
from .evaluation_v2 import _entity_key, _relation_key

STRUCTURAL_POLICY_VERSION = "explicit-local-formula-v1.0"


def _index(unit, candidates):
    if candidates["unit_id"] != unit["unit_id"]:
        raise ValueError("candidate source mismatch")
    entities = {e["entity_id"]: e for e in candidates["entities"]}
    relations = {r["relation_id"]: r for r in candidates["relations"]}
    if len(entities) != len(candidates["entities"]) or len(relations) != len(candidates["relations"]):
        raise ValueError("duplicate candidate id")
    spans = set(_all_spans(unit))
    for item in [*entities.values(), *relations.values()]:
        evidence = item.get("evidence_span_ids", [])
        if not evidence or not set(evidence) <= spans:
            raise ValueError("invalid candidate evidence")
    for r in relations.values():
        for side in ("head", "tail"):
            e = entities.get(r[side + "_id"])
            if not e or (e["name"], e["entity_type"]) != (r[side + "_name"], r[side + "_type"]):
                raise ValueError("candidate endpoint mismatch")
    return entities, relations, spans


def structural_relation_proofs(unit, candidates):
    entities, relations, spans = _index(unit, candidates)
    if unit.get("unit_type") != "formula_package":
        return []
    variables = [v for raw in unit.get("variable_definitions", []) for v in _split_variable(raw)]
    declared = {normalize_symbol(v["symbol"]) for v in variables}
    by_name = {(e["name"], e["entity_type"]): e["entity_id"] for e in entities.values()}
    lookup = {(r["head_id"], r["relation_type"], r["tail_id"]): r for r in relations.values()}
    proofs = {}

    def retain(relation, rule, evidence):
        if not relation:
            return
        needed = set(evidence)
        if not needed or not needed <= spans:
            return
        # The proof additionally records the definition span. Frozen candidate
        # evidence may cite only the equation; never rewrite that evidence.
        joint = set(relation["evidence_span_ids"])
        for side in ("head", "tail"):
            joint.update(entities[relation[side + "_id"]]["evidence_span_ids"])
        if not needed & joint:
            return
        proofs[relation["relation_id"]] = {"relation_id": relation["relation_id"], "rule": rule,
            "evidence_span_ids": sorted(needed), "policy": STRUCTURAL_POLICY_VERSION}

    for formula in unit.get("formulas", []):
        fid = by_name.get((f"公式（{formula['formula_number']}）", "formula"))
        fspan = formula.get("evidence_span", {}).get("span_id")
        if not fid or fspan not in spans:
            continue
        aliases = _formula_aliases(formula, declared, spans)
        expression = _formula_expression_body(formula.get("expression_text", ""))
        # Restrict output recognition to a complete, declared simple LHS.
        # Compound LHS, chained equalities and objective expressions abstain.
        outputs = set()
        for equation in re.split(r"[;\n]", expression):
            if equation.count("=") != 1:
                continue
            lhs, _ = equation.split("=", 1)
            symbol = normalize_symbol(lhs)
            symbol = aliases.get(symbol, symbol)
            if symbol in declared:
                outputs.add(symbol)
        if not outputs:
            continue
        vocabulary = declared | set(aliases)
        used = {aliases.get(s, s) for s in formula_symbols(expression, vocabulary, exclude_bound_indices=True)}
        for variable in variables:
            symbol = normalize_symbol(variable["symbol"])
            vid = by_name.get((variable["symbol"], "model_variable"))
            vspan = variable.get("evidence_span", {}).get("span_id")
            if not vid or vspan not in spans or symbol not in used:
                continue
            evidence = [fspan, vspan]
            evidence += [s for a in formula.get("symbol_aliases", []) for s in a["evidence_span_ids"]]
            if symbol in outputs:
                retain(lookup.get((fid, "has_output", vid)), "equation_output", evidence)
                retain(lookup.get((vid, "calculated_by", fid)), "equation_output_inverse", evidence)
            else:
                # Training-source review distinguishes physical/empirical
                # inputs from count/limit definitions. Counts remain selectable
                # by the model; this module abstains instead of redefining gold.
                if variable.get("definition_origin") == "equation_structure_transcription" or re.search(r"总数|个数|求和上限", variable.get("definition", "")):
                    continue
                retain(lookup.get((fid, "has_input", vid)), "equation_free_input", evidence)
        for binding in _formula_output_bindings(unit, candidates):
            if binding["formula_id"] != fid:
                continue
            inverse = binding["variable_relation_id"]
            output = lookup.get((fid, "has_output", binding["variable_id"]))
            if inverse in proofs and output and output["relation_id"] in proofs:
                retain(relations[binding["indicator_relation_id"]], "explicit_output_definition",
                       proofs[inverse]["evidence_span_ids"])
    return list(proofs.values())


def merge_structural_relations(unit, candidates, prediction):
    entities, relations, _ = _index(unit, candidates)
    if prediction["unit_id"] != unit["unit_id"] or prediction.get("status") != "success":
        raise ValueError("incompatible baseline prediction")
    for field, id_key, index in (("entities", "entity_id", entities), ("relations", "relation_id", relations)):
        if len({r[id_key] for r in prediction[field]}) != len(prediction[field]):
            raise ValueError("duplicate baseline candidate")
        for item in prediction[field]:
            if item != index.get(item[id_key]):
                raise ValueError("baseline is not an exact candidate subset")
    proofs = structural_relation_proofs(unit, candidates)
    result = deepcopy(prediction)
    selected = {r["relation_id"] for r in result["relations"]}
    added = [p["relation_id"] for p in proofs if p["relation_id"] not in selected]
    result["relations"].extend(deepcopy(relations[r]) for r in added)
    eids = {e["entity_id"] for e in result["entities"]}
    needed = {r[side + "_id"] for r in result["relations"] for side in ("head", "tail")}
    added_entities = [e for e in entities if e in needed and e not in eids]
    result["entities"].extend(deepcopy(entities[e]) for e in added_entities)
    if result["relations"]:
        result["no_relation_reason"] = ""
    return result, {"policy": STRUCTURAL_POLICY_VERSION, "proofs": proofs,
                    "added_relation_ids": added, "added_entity_ids": added_entities}


def formula_path_metrics(units, gold, predictions):
    """Gold-defined formula subpaths plus their annotated method/source context.

    A scorable subpath needs indicator→formula and at least one formula input
    and output in gold. Every annotated role edge and output inverse is needed.
    The expanded path also needs linked units, sources, methods, method scopes
    and constraints, and subject→indicator when annotated. Missing gold context
    is not invented; this is not a claim of complete real-world workflow truth.
    """
    source = {u["unit_id"]: set(_all_spans(u)) for u in units}
    pred = {p["unit_id"]: p for p in predictions}
    details = []
    unscorable = 0
    for annotation in gold:
        uid = annotation["unit_id"]
        p = pred.get(uid, {"relations": [], "entities": []})
        pr = {_relation_key(uid, r): r for r in p["relations"]}
        pe = {_entity_key(uid, e): e for e in p["entities"]}
        ge = {_entity_key(uid, e): e for e in annotation["entities"]}
        all_relations = annotation["relations"]
        for root in all_relations:
            if (root["head_type"], root["relation_type"], root["tail_type"]) != ("assessment_indicator", "calculated_by", "formula"):
                continue
            roles = [r for r in all_relations if r["head_name"] == root["tail_name"] and r["head_type"] == "formula" and r["relation_type"] in {"has_input", "has_output"}]
            if {r["relation_type"] for r in roles} != {"has_input", "has_output"}:
                unscorable += 1
                continue
            outputs = {r["tail_name"] for r in roles if r["relation_type"] == "has_output"}
            required = [root, *roles, *[r for r in all_relations if r["head_name"] in outputs and r["head_type"] == "model_variable" and r["relation_type"] == "calculated_by" and r["tail_name"] == root["tail_name"]]]
            nodes = {(r[side + "_name"], r[side + "_type"]) for r in required for side in ("head", "tail")}
            extras = [r for r in all_relations if (r["head_name"], r["head_type"]) in nodes and r["relation_type"] in {"has_unit", "sourced_from", "obtained_by", "applies_to_space", "applies_to_ecosystem", "constrained_by"}]
            methods = {r["tail_name"] for r in extras if r["tail_type"] == "method"}
            extras += [r for r in all_relations if (r["head_type"] == "method" and r["head_name"] in methods and r["relation_type"] in {"constrained_by", "applies_to_space", "applies_to_ecosystem"}) or (r["relation_type"] == "has_indicator" and r["tail_name"] == root["head_name"])]

            def complete(edges, evidence, reference_overlap=False):
                for r in edges:
                    item = pr.get(_relation_key(uid, r))
                    if not item:
                        return False
                    for side in ("head", "tail"):
                        key = (uid, r[side + "_name"], r[side + "_type"])
                        endpoint = pe.get(key)
                        if not endpoint:
                            return False
                        es = set(endpoint.get("evidence_span_ids", []))
                        if evidence and (not es or not es <= source[uid]):
                            return False
                        if reference_overlap and not es & set(ge.get(key, {}).get("evidence_span_ids", [])):
                            return False
                    es = set(item.get("evidence_span_ids", []))
                    if evidence and (not es or not es <= source[uid]):
                        return False
                    if reference_overlap and not es & set(r.get("evidence_span_ids", [])):
                        return False
                return True

            details.append({"root": list(_relation_key(uid, root)), "required_edges": len(required), "expanded_required_edges": len(required) + len(extras),
                            "complete": complete(required, False), "complete_evidenced": complete(required, True),
                            "reference_span_overlap_complete": complete(required, True, True),
                            "expanded_complete_evidenced": complete(required + extras, True)})
    n = len(details)
    evidenced = sum(d["complete_evidenced"] for d in details)
    expanded = sum(d["expanded_complete_evidenced"] for d in details)
    return {"scope": "annotated_formula_subpath_and_annotated_context", "scorable_paths": n, "unscorable_roots": unscorable,
            "complete_paths": sum(d["complete"] for d in details), "complete_evidenced_paths": evidenced,
            "completion_rate": evidenced / n if n else None, "expanded_complete_evidenced_paths": expanded,
            "reference_span_overlap_complete_paths": sum(d["reference_span_overlap_complete"] for d in details),
            "expanded_completion_rate": expanded / n if n else None, "details": details}
