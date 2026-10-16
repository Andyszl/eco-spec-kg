"""CPU-only verification of a training-source candidate fix; never launch training."""
from pathlib import Path
import json
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.experiment_io_v2 import sha256_json
from ecospec_kg.prediction_validation_v2 import _all_spans

FOLDER="experiments/semantic_method_review_20261016"
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p):return hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def verify(root):
    folder=root/FOLDER
    lock=read(folder/"verification_lock.json")
    source=root/"experiments/semantic_pair_v211_20261015"
    inputs=read(source/"input_manifest.json")
    assert sha(source/"input_manifest.json")==lock["previous_input_manifest_sha256"]
    for name,h in inputs["files"].items():assert sha(source/name)==h,"Training input changed: "+name
    for name,h in inputs["frozen_sha256"].items():assert sha(root/"deliveries/human_confirmed_20261002"/name)==h,"Frozen input changed: "+name
    for name,h in lock["source_sha256_lf"].items():assert lfsha(root/name)==h,"Code changed: "+name
    assert sha(folder/"annotation_revision.jsonl")==lock["annotation_sha256"]
    assert ex.CANDIDATE_GENERATOR_VERSION=="structure-aware-rule-v2.12"
    units=rows(root/"deliveries/human_confirmed_20261002/blind/train_units.jsonl")
    ids={u["unit_id"] for u in units}
    units.extend(u for u in rows(source/"units.jsonl") if u["unit_id"] not in ids)
    by_id={u["unit_id"]:u for u in units}
    review=read(folder/"review.json")
    unit=by_id[review["unit_id"]]
    definition=next(v for v in unit["variable_definitions"] if v["symbol"]==review["added_relation"]["head_name"])
    assert definition==review["source_definition"],"Reviewed definition changed"
    assert unit["provenance"]["source_sha256"]==review["source_sha256"]
    assert not review["frozen_labels_changed"] and not review["already_used_in_training"] and not review["training_ready"]
    amended=rows(folder/"annotation_revision.jsonl")
    assert len(amended)==1 and amended[0]["unit_id"]==unit["unit_id"]
    original=next(a for a in rows(source/"repair_annotations.jsonl") if a["unit_id"]==unit["unit_id"])
    assert amended[0]["entities"]==original["entities"]+[review["added_entity"]]
    assert amended[0]["relations"]==original["relations"]+[review["added_relation"]]
    assert set(review["added_relation"]["evidence_span_ids"])<=set(_all_spans(unit))
    changed=entities=relations=0
    for uid,u in by_id.items():
        candidate=ex.RuleCandidateExtractorV2().predict_unit(u)
        allowed=lock["additions"].get(uid,{"entity_ids":[],"relation_ids":[]})
        before_e=[e for e in candidate["entities"] if e["entity_id"] not in allowed["entity_ids"]]
        before_r=[r for r in candidate["relations"] if r["relation_id"] not in allowed["relation_ids"]]
        assert sha256_json(before_e)==lock["baseline"][uid]["entities"],"Existing entity/evidence changed: "+uid
        assert sha256_json(before_r)==lock["baseline"][uid]["relations"],"Existing relation/evidence changed: "+uid
        new_e=len(candidate["entities"])-len(before_e);new_r=len(candidate["relations"])-len(before_r)
        entities+=new_e;relations+=new_r;changed+=int(bool(new_e or new_r))
    assert len(by_id)==696 and changed==1 and entities==1 and relations==2
    return {"status":"named_method_review_verified","checked_units_and_views":len(by_id),"changed_units":changed,
        "new_method_entities":entities,"new_method_relations":relations,
        "training_ready":False,"new_annotation_used_in_training":False}


if __name__=="__main__":
    r=verify(ROOT)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    print("NAMED_METHOD_REVIEW_OK=true")
    print("CHECKED_TRAIN_UNITS_AND_VIEWS=696")
    print("TRAINING_READY=false")
