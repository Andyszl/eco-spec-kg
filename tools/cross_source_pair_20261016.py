"""Reviewed cross-source inputs and real CPU tokenizer preflight; no training."""
import argparse
from copy import deepcopy
from collections import Counter
import hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ecospec_kg import extractor_v2 as ex
from ecospec_kg.candidate_profile_v214 import candidates_with_alias_map, VERSION
from ecospec_kg.method_supervision_scope_v2 import training_selection_view
from ecospec_kg.experiment_io_v2 import sha256_json
import verify_method_scope_20261016 as previous
from check_scope_inputs_20261013 import encode_files

SEED="experiments/cross_source_supervision_20261016"
FOLDER="experiments/cross_source_pair_20261016"
OLD="experiments/semantic_pair_v211_20261015"
ROLES=("model_variable|obtained_by|method","model_variable|sourced_from|data_source")
HOLDOUT="9fc3c8a70f70b39e"
LONG_PARENT="3b87291bafd7da90"
LONG_CHILDREN={"6f2c40ba174a9b41","42d8dd466182a9ea","44bfa21cdeba8677"}

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p):return hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
def dumps(values):return "".join(json.dumps(v,ensure_ascii=False)+"\n" for v in values)
def key(r):return tuple(r[k] for k in ("head_name","head_type","relation_type","tail_name","tail_type"))
def short(r):return (r["head_name"],r["relation_type"],r["tail_name"])
def uid(row):return json.loads(row["messages"][1]["content"])["source_unit"]["unit_id"]

def coverage(records,units):
    count=Counter();parents={r:set() for r in ROLES};sources={r:set() for r in ROLES}
    for row in records:
        q=json.loads(row["messages"][1]["content"]);target=json.loads(row["messages"][-1]["content"])
        es={e["id"]:e for e in q["candidate_entities"]};ident=q["source_unit"]["unit_id"]
        for r in q["candidate_relations"]:
            role="|".join((es[r["head"]]["type"],r["type"],es[r["tail"]]["type"]))
            if role in ROLES and r["id"] in target["selected_relation_ids"]:
                count[role]+=1;parents[role].add(LONG_PARENT if ident in LONG_CHILDREN else ident)
                sources[role].add(units[ident]["provenance"]["source_sha256"])
    return {r:{"positive_candidate_edges":count[r],"parent_ids":sorted(parents[r]),"source_pdf_sha256":sorted(sources[r])} for r in ROLES}

def compose(root):
    previous.verify(root)
    old=root/OLD;seed=root/SEED
    old_manifest=read(old/"input_manifest.json")
    source=read(seed/"source_manifest.json")
    assert sha(seed/"HJ1340-2023.pdf")==source["source_sha256"]
    units={u["unit_id"]:u for u in rows(old/"units.jsonl")}
    labels={arm:{g["unit_id"]:g for g in rows(old/f"{arm}_annotations.jsonl")} for arm in ("control","repair")}
    new_units=rows(seed/"source_units.jsonl");new_labels={g["unit_id"]:g for g in rows(seed/"annotations.jsonl")}
    review=read(seed/"supervision_review.json")
    decisions={r["unit_id"]:r for r in review["cases"]}
    assert len(new_units)==len(new_labels)==len(decisions)==3
    assert source["independent_final_test"] is False
    old_sources={u["provenance"]["source_sha256"] for u in units.values()}
    assert source["source_sha256"] not in old_sources,"New source already belongs to old inputs"
    train=list(old_manifest["unit_ids"]["repair_train"]);validation=list(old_manifest["unit_ids"]["validation"])
    assert HOLDOUT not in train and HOLDOUT in validation
    unknown={v["unit_id"]:v["unknown_training_relation_ids"] for v in read(root/previous.FOLDER/"verified_report.json")["changed_units"]}
    kr=rows(root/"experiments/semantic_method_review_20261016/annotation_revision.jsonl")[0]
    labels["repair"][kr["unit_id"]]=kr
    new_train_lines=set();new_val_lines=set();new_positive_train=0
    for u in new_units:
        ident=u["unit_id"];d=decisions[ident]
        assert ident not in units and u["provenance"]["source_sha256"]==d["source_sha256"]==source["source_sha256"]
        section=u["provenance"]["section"]
        expected="train" if section in source["train_sections"] else "validation"
        assert d["split"]==expected,"New source split changed"
        assert section in source["train_sections"]+source["internal_validation_sections"]
        units[ident]=u;labels["repair"][ident]=new_labels[ident];labels["control"][ident]=deepcopy(new_labels[ident])
        lines={line for s in ex._all_spans(u).values() for line in s["line_ids"]}
        if expected=="train":
            train.append(ident);new_train_lines.update(lines)
            additions={tuple(r) for r in d["positive"]}
            labels["control"][ident]["relations"]=[r for r in labels["control"][ident]["relations"] if short(r) not in additions]
            new_positive_train+=len(additions)
        else:
            validation.append(ident);new_val_lines.update(lines)
        unknown[ident]=d["unknown_relation_ids"]
    assert not new_train_lines & new_val_lines,"New source train/validation line overlap"
    assert len(train)==len(set(train))==626 and len(validation)==len(set(validation))==72 and not set(train)&set(validation)
    prepared={"control_train":[],"repair_train":[],"validation":[]};candidate_hashes={};quarantine=Counter()
    for ident in train+validation:
        u=units[ident];raw,c,mapping=candidates_with_alias_map(u)
        candidate_hashes[ident]=sha256_json(c)
        unresolved=unknown.get(ident,[])
        if ident not in decisions:
            base=ex.RuleCandidateExtractorV2().predict_unit(u)
            from ecospec_kg.method_supervision_scope_v2 import canonicalize_methods
            before=canonicalize_methods(u,base)[0]
            assert c==before,"Context candidate expansion changed a previously audited unit: "+ident
        else:
            d=decisions[ident]
            known={r["relation_id"] for r in new_labels[ident]["relations"]}
            negative={r["relation_id"] for r in d["confirmed_negative_relations"]}
            assert not (known&negative or known&set(unresolved) or negative&set(unresolved))
            assert known|negative|set(unresolved)=={r["relation_id"] for r in c["relations"]},"Unreviewed new-source relation"
        quarantine["train" if ident in train else "validation"]+=len(unresolved)
        prompts=[]
        for arm in ("control","repair"):
            target=previous.project(raw,mapping,labels[arm][ident])
            endpoints={r[s+"_id"] for r in c["relations"] if r["relation_id"] in target["selected_relation_ids"] for s in ("head","tail")}
            target["selected_entity_ids"]=sorted(set(target["selected_entity_ids"])-endpoints)
            view,target=training_selection_view(c,target,unresolved)
            system,prompt=ex.build_llm_selection_messages(u,view)
            q=json.loads(prompt)
            assert not set(unresolved)&{r["id"] for r in q["candidate_relations"]}
            assert set(target["selected_relation_ids"])<={r["id"] for r in q["candidate_relations"]}
            row={"messages":[{"role":"system","content":system},{"role":"user","content":prompt},
                              {"role":"assistant","content":json.dumps(target,ensure_ascii=False,sort_keys=True)}]}
            prompts.append((system,prompt))
            if ident in train:prepared[arm+"_train"].append(row)
            elif arm=="repair":prepared["validation"].append(row)
            else:
                control_validation=row
            if ident in validation and arm=="repair":assert row==control_validation,"Validation must be identical"
        assert prompts[0]==prompts[1],"Candidate questions must be identical between arms"
    changed=sum(a!=b for a,b in zip(prepared["control_train"],prepared["repair_train"]))
    assert changed==6 and new_positive_train==7
    assert prepared["validation"][:70]==rows(old/"validation.jsonl"),"Original 70 validation records changed"
    supervised={name:coverage(records,units) for name,records in prepared.items()}
    assert all(len(supervised["repair_train"][r]["source_pdf_sha256"])>=2 for r in ROLES)
    assert all(len(supervised["validation"][r]["parent_ids"])>=2 for r in ROLES)
    assert len(supervised["validation"][ROLES[1]]["source_pdf_sha256"])==2
    sources=list(old_manifest["source_sha256_lf"])+["src/ecospec_kg/method_supervision_scope_v2.py",
            "src/ecospec_kg/candidate_profile_v214.py","tools/verify_named_method_review_20261016.py",
            "tools/verify_method_scope_20261016.py","tools/cross_source_pair_20261016.py",
            "tools/run_cross_source_preflight_20261016.sh"]
    metadata={"version":"reviewed-cross-source-pair-v1-20261016","candidate_profile":VERSION,
        "counts":{n:len(v) for n,v in prepared.items()},"unit_ids":{"control_train":train,"repair_train":train,"validation":validation},
        "frozen_sha256":old_manifest["frozen_sha256"],"tokenizer_config_sha256":old_manifest["tokenizer_config_sha256"],
        "previous_input_manifest_sha256":sha(old/"input_manifest.json"),
        "review_files_sha256":{str(p.relative_to(root)).replace("\\","/"):sha(p) for p in sorted(seed.iterdir()) if p.is_file()},
        "source_sha256_lf":{name:lfsha(root/name) for name in dict.fromkeys(sources)},
        "candidate_hashes":candidate_hashes,"quarantine_counts":dict(quarantine),"supervision":supervised,
        "paired_changed_training_targets":changed,"new_train_positive_candidate_edges":7,
        "fvc_parent_excluded_from_both_train":True,"new_source_train_validation_line_overlap":0,
        "new_source_independent_final_test":False,"new_validation_source_also_seen_in_training":True,
        "new_validation_partial_review":True,"original_70_validation_units_preserved":True,
        "control_policy":"Declared incomplete-supervision ablation: same questions, reviewed added semantic positives withheld; control is not new gold.",
        "repair_policy":"Existing semantic revisions plus K_r and 7 new source-reviewed training edges; shared unknown quarantine.",
        "paired_protocol":{"seed":42,"max_steps_per_arm":200,"validation_shared":True,"checkpoint_selection":"minimum shared validation loss",
            "three_seed_expansion":False,"full_dev_gate_unchanged":True},
        "limitations":["Assistant source review is not independent human gold.","CHM has three source spellings for one logical DEM source, not three independent examples.",
            "New NPP/CHM validation masks five unresolved method/identity edges; original 70 validation units have no quarantine.",
            "New source training/validation sections share one PDF; this is not source-held-out final generalization.",
            "FVC internal holdout remains one parent and was previously used in design."],
        "training_ready":False,"requires_real_tokenizer_preflight":True,"model_training_started":False}
    return prepared,metadata

def build_inputs(root,out):
    assert not out.exists(),"Preserve existing input delivery"
    files,m=compose(root)
    out.mkdir(parents=True)
    for n,values in files.items():(out/(n+".jsonl")).write_text(dumps(values),encoding="utf-8",newline="\n")
    m["files"]={n+".jsonl":sha(out/(n+".jsonl")) for n in files}
    save(out/"input_manifest.json",m)
    return m

def verify_inputs(root,folder=None):
    folder=folder or root/FOLDER;m=read(folder/"input_manifest.json")
    for n,h in m["files"].items():assert Path(n).name==n and sha(folder/n)==h,"Input changed: "+n
    for n,h in m["review_files_sha256"].items():assert sha(root/n)==h,"Source review changed: "+n
    for n,h in m["source_sha256_lf"].items():assert lfsha(root/n)==h,"Code changed: "+n
    prepared,expected=compose(root)
    assert {k:v for k,v in m.items() if k!="files"}==expected,"Input audit metadata changed"
    assert all(rows(folder/(n+".jsonl"))==values for n,values in prepared.items()),"Projection differs from reviewed sources"
    return {"manifest_sha256":sha(folder/"input_manifest.json"),"manifest":m}

def verify_preflight(root,folder,report):
    assert report.get("passed") and report.get("real_tokenizer") is True,"A passing real tokenizer report is required"
    checked=verify_inputs(root,folder);m=checked["manifest"]
    assert report["input_manifest_sha256"]==checked["manifest_sha256"] and report["input_manifest"]==m
    assert report["checked_records"]==sum(m["counts"].values())==1324 and not report["failures"]
    assert report["swift_version"]=="4.4.2" and report["model_weights_loaded"] is False and report["model_training_started"] is False
    assert report["original_inputs_unchanged"] and report["source_sha256_lf"]==m["source_sha256_lf"]
    for name,identities in m["unit_ids"].items():
        records=[r for r in report["encodings"] if r["file"]==name]
        assert [r["unit_id"] for r in records]==identities
        assert all(0<r["supervised_tokens"]<=r["tokens"]<=8192 for r in records)
    return checked

def preflight(root,model,out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU preflight must hide GPUs"
    assert not out.exists(),"Preserve existing report"
    checked=verify_inputs(root);m=checked["manifest"]
    assert importlib.metadata.version("ms-swift")=="4.4.2"
    for n,h in m["tokenizer_config_sha256"].items():assert sha(model/n)==h,"Tokenizer/model config changed: "+n
    from swift.model import get_model_processor
    from swift.template import get_template
    unloaded,processor=get_model_processor(str(model),load_model=False,download_model=False)
    assert unloaded is None,"Do not load model weights"
    template=get_template(processor,max_length=8192,truncation_strategy="raise",add_non_thinking_prefix=True,loss_scale="default+ignore_empty_think")
    assert template.template_meta.template_type=="qwen3_5"
    template.set_mode("train")
    files={n:rows(root/FOLDER/(n+".jsonl")) for n in m["counts"]}
    report=encode_files(template,files)
    assert report["checked_records"]==1324
    assert verify_inputs(root)==checked,"Inputs changed during encoding"
    report.update(real_tokenizer=True,input_manifest_sha256=checked["manifest_sha256"],input_manifest=m,
        source_sha256_lf=m["source_sha256_lf"],swift_version="4.4.2",model_weights_loaded=False,model_training_started=False,
        original_inputs_unchanged=True,training_ready=False,code_commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
        max_tokens_by_file={n:max(r["tokens"] for r in report["encodings"] if r["file"]==n) for n in files},
        next_step="Audit returned real preflight, then validate and deliver guarded seed42 paired training commands")
    out.mkdir(parents=True);save(out/"preflight_report.json",report)
    if report["passed"]:verify_preflight(root,root/FOLDER,report)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("build","verify","preflight"))
    p.add_argument("--out",type=Path);p.add_argument("--model",type=Path);a=p.parse_args()
    if a.action=="build":result=build_inputs(ROOT,a.out or ROOT/FOLDER)
    elif a.action=="verify":result=verify_inputs(ROOT)
    else:
        assert a.out and a.model;result=preflight(ROOT,a.model,a.out)
    print(json.dumps({k:result[k] for k in ("counts","passed","checked_records","max_tokens_by_file","failures","training_ready") if k in result},ensure_ascii=False,indent=2))
    if a.action=="preflight" and not result["passed"]:raise SystemExit(1)
