"""Shared candidate profile for the next paired training and model evaluation."""
from copy import deepcopy
import re
from . import extractor_v2 as ex
from .method_supervision_scope_v2 import canonicalize_methods

VERSION = "structure-aware-v212-local-alias-context-v1-20261016"


def candidates_with_alias_map(unit):
    original = ex.RuleCandidateExtractorV2().predict_unit(unit)
    b = ex.PredictionBuilder(unit, original["backend"])
    b.entities = {(e["name"],e["entity_type"]):deepcopy(e) for e in original["entities"]}
    b.relations = {tuple(r[k] for k in ("head_name","head_type","relation_type","tail_name","tail_type")):deepcopy(r)
                   for r in original["relations"]}
    if unit.get("unit_type") == "formula_package":
        text = "".join(str(unit.get(k,"")) for k in ("introduction","interstitial_text","adjacent_source_text"))
        text = re.sub(r"\s+", "", text)
        spans = ex._all_spans(unit)
        outputs = {r["tail_id"] for r in original["relations"] if r["relation_type"] == "has_output"}
        # A bounded named model after 建立/构建 remains a candidate, not a label.
        named = list(re.finditer(r"(?:建立|构建)([\u4e00-\u9fffA-Za-z0-9\-]{2,16}模型)(?=用于|进行|计算|估算|[，。；;])",text))
        named += list(re.finditer(r"(?<![A-Za-z0-9\-])([A-Za-z][A-Za-z0-9\-]{1,20}模型)(?=[、，。,；;）)]|进行|计算|来)",text))
        for m in named:
            name = m.group(1)
            if (name,"method") in b.entities:
                continue
            evidence = [s["span_id"] for s in spans.values() if name in re.sub(r"\s+","",s.get("text",""))]
            method = b.add_entity(name,"method",evidence)
            for e in original["entities"]:
                if e["entity_id"] in outputs:
                    b.add_relation(e,"obtained_by",method,list(dict.fromkeys(e["evidence_span_ids"]+evidence)))
        for definition in unit.get("variable_definitions",[]):
            noun = definition.get("definition","").strip("，,；;。 ")
            if not re.fullmatch(r"[\u4e00-\u9fff]{2,12}",noun):
                continue
            pattern = re.escape(noun)+r"[^。；;]{0,8}?(?:采用|常用|使用|运用|用)([\u4e00-\u9fffA-Za-z0-9\-]{2,20}?法)(?:来|进行|加以)?(?:计算|估算|得到)"
            for m in re.finditer(pattern,text):
                name = m.group(1)
                evidence = [s["span_id"] for s in spans.values() if m.group(0) in re.sub(r"\s+","",s.get("text",""))]
                if not evidence:
                    continue
                method = b.add_entity(name,"method",evidence)
                for v in ex._split_variable(definition):
                    variable = b.entities.get((v["symbol"],"model_variable"))
                    if variable:
                        b.add_relation(variable,"obtained_by",method,list(dict.fromkeys(variable["evidence_span_ids"]+evidence)))
    raw = b.result()
    canonical, mapping = canonicalize_methods(unit,raw)
    return raw, canonical, mapping


def candidates_for_unit(unit):
    return candidates_with_alias_map(unit)[1]
