from pathlib import Path
import sys,json
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import cross_source_pair_20261016 as pair


def test_rebuilt_pair_is_symmetric_and_retains_parent_holdout(tmp_path):
    out=tmp_path/"inputs"
    m=pair.build_inputs(ROOT,out)
    assert m["counts"]=={"control_train":626,"repair_train":626,"validation":72}
    assert m["quarantine_counts"]=={"train":16,"validation":5}
    assert m["new_source_train_validation_line_overlap"]==0
    assert m["paired_changed_training_targets"]==6
    assert m["fvc_parent_excluded_from_both_train"]
    assert m["new_source_independent_final_test"] is False
    assert pair.rows(out/"validation.jsonl")[:70]==pair.rows(ROOT/pair.OLD/"validation.jsonl")
    new_val={"fce734e267614afa","cdd361be6b9b6037"}
    assert new_val<=set(m["unit_ids"]["validation"])
    assert new_val.isdisjoint(m["unit_ids"]["control_train"])
    assert new_val.isdisjoint(m["unit_ids"]["repair_train"])
    assert pair.verify_inputs(ROOT,out)["manifest_sha256"]


def test_new_source_validation_parent_and_unknown_ids_cannot_leak_into_training(tmp_path,monkeypatch):
    original=pair.read
    def tampered(p):
        v=original(p)
        if p.name=="supervision_review.json":
            v["cases"][-1]["split"]="train"
        return v
    monkeypatch.setattr(pair,"read",tampered)
    with pytest.raises(AssertionError,match="split"):
        pair.build_inputs(ROOT,tmp_path/"bad")


def test_old_or_failed_preflight_cannot_authorize_new_inputs(tmp_path):
    out=tmp_path/"inputs"
    pair.build_inputs(ROOT,out)
    with pytest.raises(AssertionError,match="passing real"):
        pair.verify_preflight(ROOT,out,{"passed":False,"real_tokenizer":True})


def test_changed_training_target_cannot_pass_even_with_updated_file_hash(tmp_path):
    out=tmp_path/"inputs"
    pair.build_inputs(ROOT,out)
    p=out/"repair_train.jsonl";values=pair.rows(p)
    values[0]["messages"][-1]["content"]='{"selected_entity_ids":[],"selected_relation_ids":[]}'
    p.write_text(pair.dumps(values),encoding="utf-8",newline="\n")
    m=pair.read(out/"input_manifest.json");m["files"][p.name]=pair.sha(p);pair.save(out/"input_manifest.json",m)
    with pytest.raises(AssertionError,match="Projection"):
        pair.verify_inputs(ROOT,out)


def test_preflight_validator_rejects_oversized_and_stale_synthetic_reports(tmp_path):
    out=tmp_path/"inputs"
    pair.build_inputs(ROOT,out);checked=pair.verify_inputs(ROOT,out);m=checked["manifest"]
    # Synthetic metadata fixture exercises the gate only, not tokenization.
    report={"passed":True,"real_tokenizer":True,"checked_records":1324,"failures":[],
            "input_manifest_sha256":checked["manifest_sha256"],"input_manifest":m,
            "source_sha256_lf":m["source_sha256_lf"],"swift_version":"4.4.2",
            "model_weights_loaded":False,"model_training_started":False,"original_inputs_unchanged":True,
            "encodings":[{"file":n,"unit_id":i,"tokens":64,"supervised_tokens":10}
                         for n,ids in m["unit_ids"].items() for i in ids]}
    pair.verify_preflight(ROOT,out,report)
    report["encodings"][0]["tokens"]=8193
    with pytest.raises(AssertionError):pair.verify_preflight(ROOT,out,report)
    report["encodings"][0]["tokens"]=64;report["input_manifest_sha256"]="old-report"
    with pytest.raises(AssertionError):pair.verify_preflight(ROOT,out,report)
