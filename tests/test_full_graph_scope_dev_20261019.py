"""Real blind dev fixtures, synthetic adapters/HTTP only; no GPU inference."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evaluate_full_graph_scope_pair_20261019 as ev
from test_reviewed_pair_evaluation import http


@pytest.fixture
def pair(tmp_path, monkeypatch):
    lock = copy.deepcopy(ev.read(ROOT / ev.PROTOCOL))
    manifest = lock["training_manifest"]
    train = tmp_path / "train"
    for arm, step in (("control", 120), ("repair", 120)):
        out = train / arm
        best = out / f"checkpoint-{step}"
        best.mkdir(parents=True)
        (best / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
        ev.save(best / "adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": .05})
        last = out / "checkpoint-200"
        last.mkdir()
        ev.save(last / "trainer_state.json", {"global_step": 200, "best_model_checkpoint": str(best),
            "best_metric": .02, "log_history": [{"step": s, "eval_loss": .02 if s == step else .03} for s in (40,80,120,160,200)]})
        manifest["arms"][arm]["output"] = str(out)
        manifest["arms"][arm]["command"] = ev.training.build_command(ROOT, Path("/home/hello/szl/models/Qwen3.5-9B"), out, arm)
        manifest["arms"][arm]["completion"] = ev.training.inspect_completion(out)
    ev.save(train / "pair_manifest.json", manifest)
    lock["training_manifest_sha256"] = ev.sha(train / "pair_manifest.json")
    protocol = tmp_path / "protocol.json"
    ev.save(protocol, lock)
    monkeypatch.setattr(ev, "PROTOCOL", str(protocol))
    monkeypatch.setattr(ev, "EXPECTED_PAIR_SHA", lock["training_manifest_sha256"])
    return train


def test_pair_provenance_and_adapter_tampering(pair):
    manifest = ev.inspect_pair(ROOT, pair)
    assert manifest["verification"]["manifest"]["counts"] == {"control_train":626,"repair_train":626,"validation":72}
    (pair / "repair/checkpoint-120/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="completion"):
        ev.inspect_pair(ROOT, pair)


def test_changed_manifest_rejected(pair):
    with (pair / "pair_manifest.json").open("a") as f: f.write(" ")
    with pytest.raises(AssertionError, match="manifest"):
        ev.inspect_pair(ROOT, pair)


def test_real_dev_66_equal_full_profile_prompts_and_same_structure(pair,tmp_path,http):
    with patch("urllib.request.urlopen", side_effect=http["respond"]):
        result = ev.evaluate(ROOT,pair,tmp_path/"evaluation")
    assert result["status"] == "complete" and len(http["calls"]) == 66
    prompts = [p for _,_,p in http["calls"]]
    assert prompts[:22] == prompts[22:44] == prompts[44:]
    units = ev.read_jsonl(ROOT/"deliveries/human_confirmed_20261002/blind/dev_units.jsonl")
    by_id = {u["unit_id"]:u for u in units}
    for _,ident,prompt in http["calls"]:
        graph = ev.profile.candidates_for_unit(by_id[ident])
        assert prompt == ev.scope.build_messages(by_id[ident],graph)[1]
    for label in ev.MODELS:
        dest=tmp_path/"evaluation"/label
        assert len(ev.read_jsonl(dest/"hybrid/structural_proofs.jsonl")) == 22
        cfg=ev.read(dest/"resolved_config.json")
        assert cfg["candidate_generator"] == ev.profile.VERSION
    assert not result["training_started"]
    assert result["decision_scope_version"] == ev.scope.VERSION
    assert set(result["clause_guard_runs"]) == set(ev.MODELS)
    assert result["clause_guard_model_requests"] == 0
    assert result["clause_guard_gate"]["three_seed_eligible"] is False
    for _, _, prompt in http["calls"]:
        assert json.loads(prompt)["decision_scope"]["unscored_relation_ids"] == []
    for arm in ev.MODELS:
        assert len(ev.read_jsonl(tmp_path/"evaluation"/arm/"clause_guard/filter_proofs.jsonl")) == 22
        assert len(ev.read_jsonl(tmp_path/"evaluation"/arm/"clause_guard/hybrid/structural_proofs.jsonl")) == 22


@pytest.mark.parametrize("failure",["fail","bad_proof","models"])
def test_bad_or_incomplete_inference_never_produces_gate(pair,tmp_path,http,failure):
    http[failure]=[] if failure=="models" else True
    with patch("urllib.request.urlopen",side_effect=http["respond"]),pytest.raises((AssertionError,TimeoutError,ValueError)):
        ev.evaluate(ROOT,pair,tmp_path/"bad")
    assert not (tmp_path/"bad/summary.json").exists()


def test_profile_structure_reuses_frozen_algorithm():
    import paired_structural_metrics as frozen
    units=ev.read_jsonl(ROOT/"deliveries/human_confirmed_20261002/blind/dev_units.jsonl")
    gold=ev.read_jsonl(ROOT/"deliveries/human_confirmed_20261002/gold/dev_annotations.jsonl")
    predictions=[]
    for u in units:
        p=ev.profile.candidates_for_unit(u)
        p.update(experiment_id="synthetic",config_hash="fixture",candidate_hash="fixture",model="fixture",seed=42,
                 relations=[],no_relation_reason="synthetic_empty")
        predictions.append(p)
    # With the same supplied graphs the original implementation must match exactly.
    class Candidate:
        def predict_unit(self,u):return ev.profile.candidates_for_unit(u)
    with patch.object(frozen,"RuleCandidateExtractorV2",Candidate):
        expected=frozen.compare_one(units,gold,{},predictions)
    assert ev.compare_one(units,gold,{},predictions)==expected


def test_deploy_uses_selected_120_120_and_fixed_budget(pair):
    cmd=ev.deploy_command(ROOT,ev.inspect_pair(ROOT,pair),8018)
    assert any(x.startswith("eco-pair-control=") and x.endswith("checkpoint-120") for x in cmd)
    assert any(x.startswith("eco-pair-repair=") and x.endswith("checkpoint-120") for x in cmd)
    assert cmd[cmd.index("--max_length")+1]=="8192"
    assert cmd[cmd.index("--max_new_tokens")+1]=="3072"


def test_changed_source_fails_before_inference(pair):
    lock=ev.read(Path(ev.PROTOCOL));lock["source_sha256_lf"]["tools/paired_structural_metrics.py"]="0"*64
    ev.save(Path(ev.PROTOCOL),lock)
    with pytest.raises(AssertionError,match="source"):
        ev.inspect_pair(ROOT,pair)


def test_changed_frozen_dev_lock_rejected_before_service(pair):
    lock=ev.read(Path(ev.PROTOCOL))
    lock["evaluation_input_sha256"]["deliveries/human_confirmed_20261002/blind/dev_units.jsonl"]="0"*64
    ev.save(Path(ev.PROTOCOL),lock)
    with pytest.raises(AssertionError,match="Evaluation input"):
        ev.inspect_pair(ROOT,pair)


def test_secondary_postfilter_cannot_replace_primary_gate(pair,tmp_path,http,monkeypatch):
    original = ev.clause_guard_report
    def degraded_secondary(units,gold,cfg,predictions,dest):
        report=original(units,gold,cfg,predictions,dest)
        if cfg['model']==ev.MODELS['repair']:
            report['before']['relations']['f1']=-1
        return report
    monkeypatch.setattr(ev,'clause_guard_report',degraded_secondary)
    with patch('urllib.request.urlopen',side_effect=http['respond']):
        result=ev.evaluate(ROOT,pair,tmp_path/'evaluation')
    assert result['gate']==ev.strict_gate(result['runs']['control'],result['runs']['repair'])
    assert not result['primary_gate_uses_clause_guard']
    assert not result['clause_guard_gate']['passed']
    assert not result['clause_guard_gate']['three_seed_eligible']
    assert len(http['calls'])==66
