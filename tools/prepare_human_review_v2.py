"""Create a private two-reviewer packet and validate actual human submissions."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ecospec_kg.experiment_io_v2 import assert_blind_records, sha256_path
from ecospec_kg.experiment_data_v2 import prepare_experiment_package_v2, split_for_experiment_unit
from ecospec_kg.pilot_quality_v2 import compare_experts, validate_expert_annotations


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def check_manifest(package: Path) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest.get("files", []):
        item = package / record["path"]
        if sha256_path(item) != record["sha256"]:
            raise ValueError(f"package file hash mismatch: {item}")
    return manifest


def prepare_records(split_rows: dict[str, tuple[list[dict], list[dict]]]) -> dict[str, Any]:
    sources: list[dict] = []
    candidates: list[dict] = []
    seen: set[str] = set()
    for split in ("train", "dev", "test"):
        units, labels = split_rows[split]
        if [u["unit_id"] for u in units] != [a["unit_id"] for a in labels]:
            raise ValueError(f"source/annotation inventory mismatch: {split}")
        if any(u["unit_id"] in seen for u in units):
            raise ValueError("duplicate unit across splits")
        seen.update(u["unit_id"] for u in units)
        assert_blind_records(units)
        sources.extend(copy.deepcopy(units))
        for annotation in labels:
            row = copy.deepcopy(annotation)
            row["split"] = split
            candidates.append(row)
    templates: dict[str, list[dict]] = {}
    for reviewer in ("A", "B"):
        rows = copy.deepcopy(candidates)
        for row in rows:
            row["annotator_id"] = f"PENDING_EXPERT_{reviewer}"
            row["review_status"] = "pending_human_review"
            row["human_review"] = {"reviewer_id": "", "reviewed_at": "", "comment": ""}
        templates[reviewer] = rows
    return {"sources": sources, "candidates": candidates, "templates": templates,
            "split_counts": {split: len(split_rows[split][0]) for split in ("train", "dev", "test")}}


def _reviewer_id(rows: list[dict], label: str) -> str:
    ids = {str(row.get("annotator_id", "")) for row in rows}
    if len(ids) != 1 or not next(iter(ids)).strip():
        raise ValueError(f"{label}: inconsistent reviewer identity")
    reviewer_id = next(iter(ids))
    if reviewer_id.startswith("PENDING_"):
        raise ValueError(f"{label}: pending reviewer identity")
    for row in rows:
        details = row.get("human_review", {})
        if (row.get("review_status") != "human_reviewed"
                or details.get("reviewer_id") != reviewer_id
                or not str(details.get("reviewed_at", "")).strip()):
            raise ValueError(f"{label}: pending or incomplete review: {row.get('unit_id')}")
    return reviewer_id


def _keys(row: dict, kind: str) -> set[tuple]:
    if kind == "entities":
        return {(e["name"], e["entity_type"]) for e in row[kind]}
    return {(r["head_name"], r["head_type"], r["relation_type"],
             r["tail_name"], r["tail_type"]) for r in row[kind]}


def _validate_review_rows(sources: list[dict], rows: list[dict], reviewer_id: str) -> None:
    result = validate_expert_annotations(sources, rows, reviewer_id)
    if not result["passed"]:
        raise ValueError(f"review schema/evidence validation failed: {result['failures'][:3]}")
    for row in rows:
        entities = {e["entity_id"]: (e["name"], e["entity_type"])
                    for e in row["entities"]}
        if len(entities) != len(row["entities"]):
            raise ValueError(f"duplicate entity ID: {row['unit_id']}")
        for relation in row["relations"]:
            for side in ("head", "tail"):
                if entities.get(relation[side + "_id"]) != (
                    relation[side + "_name"], relation[side + "_type"]
                ):
                    raise ValueError(f"relation endpoint ID mismatch: {row['unit_id']}")


def compare_submissions(packet: Path, expert_a: Path, expert_b: Path,
                        out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("output exists; refusing to overwrite")
    check_manifest(packet)
    sources = read_rows(packet / "sources/all_units.jsonl")
    a, b = read_rows(expert_a), read_rows(expert_b)
    a_id, b_id = _reviewer_id(a, "expert_A"), _reviewer_id(b, "expert_B")
    if a_id == b_id:
        raise ValueError("reviewers must be distinct human experts")
    expected = [u["unit_id"] for u in sources]
    if [r["unit_id"] for r in a] != expected or [r["unit_id"] for r in b] != expected:
        raise ValueError("full source inventory and order required")
    _validate_review_rows(sources, a, a_id)
    _validate_review_rows(sources, b, b_id)
    agreement, disagreements = compare_experts(sources, a, b)
    out.mkdir(parents=True)
    write_rows(out / "disagreements.jsonl", disagreements)
    summary = {"reviewer_ids": [a_id, b_id], "disagreement_count": len(disagreements),
               "agreement": agreement, "expert_A_sha256": sha256_path(expert_a),
               "expert_B_sha256": sha256_path(expert_b)}
    write_json(out / "agreement.json", summary)
    return summary


def validate_submissions(sources: list[dict], expert_a: list[dict], expert_b: list[dict],
                         adjudicated: list[dict], provenance: dict,
                         decision_log: list[dict] | None = None) -> dict[str, Any]:
    a_id = _reviewer_id(expert_a, "expert_A")
    b_id = _reviewer_id(expert_b, "expert_B")
    if a_id == b_id:
        raise ValueError("reviewers must be distinct human experts")
    if [r["unit_id"] for r in expert_a] != [u["unit_id"] for u in sources] or \
       [r["unit_id"] for r in expert_b] != [u["unit_id"] for u in sources] or \
       [r["unit_id"] for r in adjudicated] != [u["unit_id"] for u in sources]:
        raise ValueError("full source inventory and order required")
    reviewers = {str(r.get("reviewer_id", "")): r for r in provenance.get("reviewers", [])}
    for reviewer_id in (a_id, b_id):
        reviewer = reviewers.get(reviewer_id, {})
        if reviewer.get("reviewer_type") != "human_domain_expert" or reviewer.get("human_expert") is not True:
            raise ValueError("signed human domain expert provenance required")
        for field in ("qualification_summary", "identity_verification_reference", "signed_at"):
            if not str(reviewer.get(field, "")).strip():
                raise ValueError(f"human reviewer provenance missing {field}")
    if provenance.get("schema_version") != "ecospec-review-provenance-v2.0":
        raise ValueError("unsupported human review provenance schema")
    adjudication = provenance.get("adjudication", {})
    adjudicator_id = str(adjudication.get("adjudicator_id", ""))
    if adjudicator_id not in (a_id, b_id):
        raise ValueError("human adjudicator must be one of the signed reviewers")
    if provenance.get("gold_nature") != "human_expert_gold" or provenance.get("claims_human_expert_review") is not True:
        raise ValueError("human review provenance declaration required")
    if adjudication.get("unresolved_count") != 0:
        raise ValueError("unresolved expert disagreements")
    for rows, reviewer_id in ((expert_a, a_id), (expert_b, b_id),
                              (adjudicated, adjudicator_id)):
        _validate_review_rows(sources, rows, reviewer_id)
    for row in adjudicated:
        if (row.get("review_status") != "human_adjudicated"
                or row.get("human_review", {}).get("reviewer_id") != adjudicator_id
                or not str(row.get("human_review", {}).get("reviewed_at", "")).strip()):
            raise ValueError(f"adjudication incomplete: {row['unit_id']}")
    agreement, disagreements = compare_experts(sources, expert_a, expert_b)
    disputed = {r["unit_id"] for r in disagreements}
    logs = decision_log or []
    if {r.get("unit_id") for r in logs} != disputed or len(logs) != len(disputed):
        raise ValueError("all disagreements need one adjudication log entry")
    for log in logs:
        if (log.get("adjudicator_id") != adjudicator_id
                or not str(log.get("reason", "")).strip()
                or not str(log.get("signed_at", "")).strip()):
            raise ValueError("disagreement adjudication reason/signature missing")
    for left, right, final in zip(expert_a, expert_b, adjudicated):
        if left["unit_id"] not in disputed:
            for kind in ("entities", "relations"):
                if _keys(final, kind) != _keys(left, kind):
                    raise ValueError("agreed unit changed during adjudication")
    return {"passed": True, "unit_count": len(sources), "reviewer_ids": [a_id, b_id],
            "adjudicator_id": adjudicator_id, "disagreement_count": len(disputed),
            "agreement": agreement}


def prepare(rc2: Path, frozen: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("output exists; refusing to overwrite")
    rc2_manifest = check_manifest(rc2)
    frozen_manifest = check_manifest(frozen)
    split_rows = {}
    for split in ("train", "dev", "test"):
        package = rc2 if split == "train" else frozen
        split_rows[split] = (read_rows(package / "blind" / f"{split}_units.jsonl"),
                             read_rows(package / "gold" / f"{split}_annotations.jsonl"))
    packet = prepare_records(split_rows)
    out.mkdir(parents=True)
    write_rows(out / "sources/all_units.jsonl", packet["sources"])
    write_rows(out / "reference/ai_candidate_annotations.jsonl", packet["candidates"])
    for reviewer in ("A", "B"):
        write_rows(out / f"templates/expert_{reviewer}.jsonl", packet["templates"][reviewer])
    files = [{"path": p.relative_to(out).as_posix(), "sha256": sha256_path(p)}
             for p in sorted(out.rglob("*.jsonl"))]
    manifest = {"status": "pending_human_review", "split_counts": packet["split_counts"],
                "unit_count": len(packet["sources"]), "human_expert_review_completed": False,
                "rc2_dataset_version": rc2_manifest["dataset_version"],
                "rc2_train_hashes": {key: sha256_path(rc2 / path) for key, path in
                                      (("units", "blind/train_units.jsonl"),
                                       ("annotations", "gold/train_annotations.jsonl"))},
                "frozen_dataset_version": frozen_manifest["dataset_version"],
                "files": files}
    write_json(out / "manifest.json", manifest)
    return manifest


def validate(packet: Path, expert_a: Path, expert_b: Path, adjudicated: Path,
             provenance: Path, decision_log: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("output exists; refusing to overwrite")
    check_manifest(packet)
    sources = read_rows(packet / "sources/all_units.jsonl")
    a, b, final = read_rows(expert_a), read_rows(expert_b), read_rows(adjudicated)
    review = json.loads(provenance.read_text(encoding="utf-8"))
    result = validate_submissions(sources, a, b, final, review, read_rows(decision_log))
    result["source_hashes"] = {"packet": sha256_path(packet / "manifest.json"),
                               "expert_A": sha256_path(expert_a), "expert_B": sha256_path(expert_b),
                               "adjudicated": sha256_path(adjudicated),
                               "provenance": sha256_path(provenance),
                               "adjudication_log": sha256_path(decision_log)}
    out.mkdir(parents=True)
    write_json(out / "human_review_validation.json", result)
    return result


def freeze(packet: Path, expert_a: Path, expert_b: Path, adjudicated: Path,
           provenance: Path, decision_log: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("output exists; refusing to overwrite")
    packet_manifest = check_manifest(packet)
    sources = read_rows(packet / "sources/all_units.jsonl")
    final = read_rows(adjudicated)
    result = validate_submissions(
        sources, read_rows(expert_a), read_rows(expert_b), final,
        json.loads(provenance.read_text(encoding="utf-8")), read_rows(decision_log))
    result["source_hashes"] = {
        "packet": sha256_path(packet / "manifest.json"),
        "expert_A": sha256_path(expert_a),
        "expert_B": sha256_path(expert_b),
        "adjudicated": sha256_path(adjudicated),
        "provenance": sha256_path(provenance),
        "adjudication_log": sha256_path(decision_log),
    }
    for source, label in zip(sources, final):
        if split_for_experiment_unit(source) != label["split"]:
            raise ValueError("split policy changed; refusing to freeze")
    manifest = prepare_experiment_package_v2(
        packet / "sources/all_units.jsonl", adjudicated, out,
        dataset_version="v2.2-human-reviewed-internal-rc1",
        gold_nature="human_expert_gold", review_provenance_path=provenance)
    if manifest["unit_split_counts"] != packet_manifest["split_counts"]:
        raise ValueError("frozen package inventory changed")
    result["frozen_package_id"] = manifest["package_id"]
    write_json(out / "human_review_validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for flag in ("rc2", "frozen", "out"):
        prep.add_argument("--" + flag, type=Path, required=True)
    comparison = commands.add_parser("compare")
    for flag in ("packet", "expert-a", "expert-b", "out"):
        comparison.add_argument("--" + flag, type=Path, required=True)
    for command in ("validate", "freeze"):
        check = commands.add_parser(command)
        for flag in ("packet", "expert-a", "expert-b", "adjudicated", "provenance", "decision-log", "out"):
            check.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.rc2, args.frozen, args.out)
    elif args.command == "compare":
        result = compare_submissions(args.packet, args.expert_a, args.expert_b, args.out)
    else:
        action = validate if args.command == "validate" else freeze
        result = action(args.packet, args.expert_a, args.expert_b, args.adjudicated,
                        args.provenance, args.decision_log, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
