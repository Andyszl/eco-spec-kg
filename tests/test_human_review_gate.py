"""Human review packets remain pending until independent, evidenced decisions arrive."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_human_review_v2 import prepare_records, validate_submissions, freeze, write_json, write_rows


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
    assert validate_submissions(packet["sources"], a, b, final, provenance)["passed"]


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
