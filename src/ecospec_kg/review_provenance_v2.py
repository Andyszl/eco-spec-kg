from __future__ import annotations

from pathlib import Path
from typing import Any

from .experiment_io_v2 import read_json, sha256_path, write_json


AI_ADJUDICATED_GOLD = "ai_expert_adjudicated_gold"
AI_PRE_GOLD = "ai_expert_pre_gold"
HUMAN_EXPERT_GOLD = "human_expert_gold"
SUPPORTED_GOLD_NATURES = frozenset(
    {AI_PRE_GOLD, AI_ADJUDICATED_GOLD, HUMAN_EXPERT_GOLD}
)


def _human_reviewers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        reviewer
        for reviewer in payload.get("reviewers", [])
        if reviewer.get("reviewer_type") == "human_domain_expert"
        and reviewer.get("human_expert") is True
    ]


def _validate_human_expert_gold(payload: dict[str, Any]) -> None:
    reviewers = _human_reviewers(payload)
    reviewer_ids = {str(reviewer.get("reviewer_id", "")) for reviewer in reviewers}
    if len(reviewer_ids - {""}) < 2:
        raise ValueError(
            "human_expert_gold requires at least two distinct human domain experts"
        )
    required_fields = {
        "qualification_summary",
        "identity_verification_reference",
        "signed_at",
    }
    for reviewer in reviewers:
        missing = sorted(
            field for field in required_fields if not str(reviewer.get(field, "")).strip()
        )
        if missing:
            raise ValueError(
                "human expert reviewer is missing provenance fields: "
                f"reviewer_id={reviewer.get('reviewer_id')} fields={missing}"
            )
    adjudicator_id = str(payload.get("adjudication", {}).get("adjudicator_id", ""))
    if adjudicator_id not in reviewer_ids:
        raise ValueError(
            "human_expert_gold requires a human adjudicator registered in reviewers"
        )


def _validate_ai_adjudicated_gold(payload: dict[str, Any]) -> None:
    ai_reviewers = [
        reviewer
        for reviewer in payload.get("reviewers", [])
        if reviewer.get("reviewer_type") == "ai_simulated_domain_reviewer"
        and reviewer.get("human_expert") is False
    ]
    if not ai_reviewers:
        raise ValueError(
            "ai_expert_adjudicated_gold requires an AI simulated domain reviewer"
        )
    if payload.get("claims_human_expert_review") is not False:
        raise ValueError(
            "AI adjudication provenance must explicitly set "
            "claims_human_expert_review=false"
        )


def validate_review_provenance_v2(
    gold_nature: str,
    provenance_path: Path | None,
) -> dict[str, Any] | None:
    if gold_nature not in SUPPORTED_GOLD_NATURES:
        raise ValueError(f"unsupported gold_nature: {gold_nature}")
    if gold_nature == AI_PRE_GOLD:
        if provenance_path is None:
            return None
    elif provenance_path is None:
        raise ValueError(f"{gold_nature} requires --review-provenance")

    assert provenance_path is not None
    payload = read_json(provenance_path)
    if payload.get("schema_version") != "ecospec-review-provenance-v2.0":
        raise ValueError("unsupported review provenance schema_version")
    if payload.get("gold_nature") != gold_nature:
        raise ValueError("review provenance gold_nature does not match CLI argument")
    if gold_nature == HUMAN_EXPERT_GOLD:
        _validate_human_expert_gold(payload)
    elif gold_nature == AI_ADJUDICATED_GOLD:
        _validate_ai_adjudicated_gold(payload)
    return payload


def write_package_review_provenance(
    out_dir: Path,
    provenance_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    output_path = out_dir / "review_provenance.json"
    write_json(output_path, payload)
    return {
        "path": output_path.relative_to(out_dir).as_posix(),
        "sha256": sha256_path(output_path),
        "source_path": str(provenance_path.resolve()),
        "source_sha256": sha256_path(provenance_path),
        "reviewer_count": len(payload.get("reviewers", [])),
        "reviewer_types": sorted(
            {
                str(reviewer.get("reviewer_type", ""))
                for reviewer in payload.get("reviewers", [])
                if reviewer.get("reviewer_type")
            }
        ),
    }


__all__ = [
    "AI_ADJUDICATED_GOLD",
    "AI_PRE_GOLD",
    "HUMAN_EXPERT_GOLD",
    "SUPPORTED_GOLD_NATURES",
    "validate_review_provenance_v2",
    "write_package_review_provenance",
]
