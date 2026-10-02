"""Human review packets remain pending until independent, evidenced decisions arrive."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_human_review_v2 import (
    prepare_records, validate_submissions, compare_submissions, freeze, write_json, write_rows,
)


def sample():
    span = {"span_id": "s", "page": 1, "text": "X"}
    unit = {"unit_id": "u", "provenance": {"standard_code": "HJ 1173-2021", "evidence_spans": [span]},
            "unit_type": "formula_package"}
    annotation = {"unit_id": "u", "split": "train", "annotator_id": "ai",
                  "entities": [{"entity_id": "e", "name": "X", "entity_type": "model_variable",
                                "evidence_span_ids": ["s"]}], "relations": []}
    return {"train": ([unit], [annotation]), "dev": ([], []), "test": ([], [])}


def test_packet_does_not_promote_candidate_to_reviewed():
    records = prepare_records(sample())
    assert records["sources"][0]["unit_id"] == "u"
    assert records["candidates"][0]["entities"][0]["name"] == "X"
    for reviewer in ("A", "B"):
        row = records["templates"][reviewer][0]
        assert row["review_status"] == "pending_human_review"
        assert row["human_review"]["reviewed_at"] == ""
        assert row["annotator_id"] == f"PENDING_EXPERT_{reviewer}"


def test_unreviewed_templates_cannot_pass_gate():
    packet = prepare_records(sample())
    with pytest.raises(ValueError, match="pending|review"):
        validate_submissions(packet["sources"], packet["templates"]["A"],
                             packet["templates"]["B"], packet["templates"]["A"], {})


def test_one_person_cannot_fill_both_review_slots():
    packet = prepare_records(sample())
    rows = packet["templates"]["A"]
    for row in rows:
        row["annotator_id"] = "same_person"
        row["review_status"] = "human_reviewed"
        row["human_review"] = {"reviewer_id": "same_person", "reviewed_at": "2026-10-01T10:00:00+08:00"}
    with pytest.raises(ValueError, match="distinct"):
        validate_submissions(packet["sources"], rows, rows, rows, {})


def test_two_signed_reviews_and_agreed_adjudication_pass():
    packet = prepare_records(sample())
    a, b = packet["templates"]["A"], packet["templates"]["B"]
    for rows, reviewer in ((a, "person_A"), (b, "person_B")):
        for row in rows:
            row["annotator_id"] = reviewer
            row["review_status"] = "human_reviewed"
            row["human_review"] = {"reviewer_id": reviewer, "reviewed_at": "2026-10-01T10:00:00+08:00"}
    final = [dict(row, review_status="human_adjudicated") for row in a]
    provenance = {"schema_version": "ecospec-review-provenance-v2.0",
                  "gold_nature": "human_expert_gold", "claims_human_expert_review": True,
                  "reviewers": [{"reviewer_id": reviewer, "reviewer_type": "human_domain_expert",
                                 "human_expert": True, "qualification_summary": "ecology",
                                 "identity_verification_reference": "signed record",
                                 "signed_at": "2026-10-01T10:00:00+08:00"}
                                for reviewer in ("person_A", "person_B")],
                  "adjudication": {"adjudicator_id": "person_A", "unresolved_count": 0}}
    result = validate_submissions(packet["sources"], a, b, final, provenance)
    assert result["passed"]
    assert result["agreement"]["annotation_nature"] == "human_expert_reviewed"
    assert result["agreement"]["human_expert_review_required_for_publication"] is False
    provenance["review_method"] = "ai_assisted_human_confirmed"
    assisted = validate_submissions(packet["sources"], a, b, final, provenance)
    assert assisted["agreement"]["annotation_nature"] == "ai_assisted_human_confirmed"
    assert assisted["agreement"]["review_provenance_validated"] is True


def test_disputed_unit_requires_logged_adjudication():
    packet = prepare_records(sample())
    a, b = packet["templates"]["A"], packet["templates"]["B"]
    for rows, reviewer in ((a, "person_A"), (b, "person_B")):
        for row in rows:
            row["annotator_id"] = reviewer
            row["review_status"] = "human_reviewed"
            row["human_review"] = {"reviewer_id": reviewer, "reviewed_at": "2026-10-01T10:00:00+08:00"}
    b[0]["entities"] = []
    final = [dict(a[0], review_status="human_adjudicated")]
    provenance = {"schema_version": "ecospec-review-provenance-v2.0",
                  "gold_nature": "human_expert_gold", "claims_human_expert_review": True,
                  "reviewers": [{"reviewer_id": reviewer, "reviewer_type": "human_domain_expert",
                                 "human_expert": True, "qualification_summary": "ecology",
                                 "identity_verification_reference": "signed record",
                                 "signed_at": "2026-10-01T10:00:00+08:00"}
                                for reviewer in ("person_A", "person_B")],
                  "adjudication": {"adjudicator_id": "person_A", "unresolved_count": 0}}
    with pytest.raises(ValueError, match="disagreements"):
        validate_submissions(packet["sources"], a, b, final, provenance)


def test_compare_writes_only_actual_disagreements(tmp_path):
    packet = prepare_records(sample())
    folder = tmp_path / "packet"
    write_rows(folder / "sources/all_units.jsonl", packet["sources"])
    write_json(folder / "manifest.json", {"files": []})
    for reviewer in ("A", "B"):
        row = packet["templates"][reviewer][0]
        row["annotator_id"] = f"person_{reviewer}"
        row["review_status"] = "human_reviewed"
        row["human_review"] = {"reviewer_id": f"person_{reviewer}",
                               "reviewed_at": "2026-10-01T10:00:00+08:00"}
    packet["templates"]["B"][0]["entities"] = []
    write_rows(tmp_path / "A.jsonl", packet["templates"]["A"])
    write_rows(tmp_path / "B.jsonl", packet["templates"]["B"])
    summary = compare_submissions(folder, tmp_path / "A.jsonl", tmp_path / "B.jsonl",
                                  tmp_path / "difference")
    assert summary["disagreement_count"] == 1
    assert summary["agreement"]["annotation_nature"] == "reviewed_submissions"
    assert summary["agreement"]["review_provenance_validated"] is False
    assert (tmp_path / "difference/disagreements.jsonl").exists()


def test_signed_reviews_freeze_only_matching_split(tmp_path):
    rows = sample()
    rows["train"][0][0]["provenance"]["standard_code"] = "HJ 1167-2021"
    packet = prepare_records(rows)
    folder = tmp_path / "packet"
    write_rows(folder / "sources/all_units.jsonl", packet["sources"])
    a, b = packet["templates"]["A"], packet["templates"]["B"]
    for records, reviewer in ((a, "person_A"), (b, "person_B")):
        records[0]["annotator_id"] = reviewer
        records[0]["review_status"] = "human_reviewed"
        records[0]["human_review"] = {"reviewer_id": reviewer, "reviewed_at": "2026-10-01T10:00:00+08:00"}
    final = [dict(a[0], review_status="human_adjudicated")]
    provenance = {"schema_version": "ecospec-review-provenance-v2.0",
                  "gold_nature": "human_expert_gold", "claims_human_expert_review": True,
                  "reviewers": [{"reviewer_id": reviewer, "reviewer_type": "human_domain_expert",
                                 "human_expert": True, "qualification_summary": "ecology",
                                 "identity_verification_reference": "signed record",
                                 "signed_at": "2026-10-01T10:00:00+08:00"}
                                for reviewer in ("person_A", "person_B")],
                  "adjudication": {"adjudicator_id": "person_A", "unresolved_count": 0}}
    write_rows(tmp_path / "a.jsonl", a)
    write_rows(tmp_path / "b.jsonl", b)
    write_rows(tmp_path / "final.jsonl", final)
    write_rows(tmp_path / "log.jsonl", [])
    write_json(tmp_path / "provenance.json", provenance)
    write_json(folder / "manifest.json", {"split_counts": {"train": 1, "dev": 0, "test": 0}, "files": []})
    result = freeze(folder, tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "final.jsonl",
                    tmp_path / "provenance.json", tmp_path / "log.jsonl", tmp_path / "frozen")
    assert result["passed"]
    assert (tmp_path / "frozen/manifest.json").exists()


def test_relation_endpoint_id_must_match_reviewed_entity(tmp_path):
    rows = sample()
    source = rows["train"][0][0]
    label = rows["train"][1][0]
    label["relations"] = [{"head_id": "WRONG", "head_name": "X", "head_type": "model_variable",
                           "relation_type": "calculated_by", "tail_id": "formula-id",
                           "tail_name": "公式（1）", "tail_type": "formula", "evidence_span_ids": ["s"]}]
    label["entities"].append({"entity_id": "formula-id", "name": "公式（1）",
                              "entity_type": "formula", "evidence_span_ids": ["s"]})
    packet = prepare_records(rows)
    write_rows(tmp_path / "sources/all_units.jsonl", [source])
    write_json(tmp_path / "manifest.json", {"files": []})
    for reviewer in ("A", "B"):
        review = packet["templates"][reviewer][0]
        review["annotator_id"] = f"person_{reviewer}"
        review["review_status"] = "human_reviewed"
        review["human_review"] = {"reviewer_id": f"person_{reviewer}",
                                  "reviewed_at": "2026-10-01T10:00:00+08:00"}
        write_rows(tmp_path / f"{reviewer}.jsonl", [review])
    with pytest.raises(ValueError, match="endpoint ID"):
        compare_submissions(tmp_path, tmp_path / "A.jsonl", tmp_path / "B.jsonl",
                            tmp_path / "disagreements")
