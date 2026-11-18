from __future__ import annotations

import pytest

from src.data_utils import validate_research_splits


def _row(recording, label, *, environment="Shoplifting", statement_source=None, statement_env=None):
    row = {
        "image_path": f"/images/{recording}.png",
        "statement": (
            f"description for {recording if statement_source is None else statement_source} "
            f"({label})"
        ),
        "label": label,
        "camera_id": f"cam_{recording}",
        "environment_id": environment,
        "source_recording_id": recording,
    }
    if statement_source is not None:
        row["statement_source_recording_id"] = statement_source
    if statement_env is not None:
        row["statement_source_environment_id"] = statement_env
    return row


def test_split_audit_requires_recording_disjointness_but_reports_category_overlap():
    splits = {
        "fit": [_row("fit-1", "MATCH"), _row("fit-1", "CONTRADICTION")],
        "calibration": [_row("cal-1", "MATCH"), _row("cal-1", "CONTRADICTION")],
        "test": [_row("test-1", "MATCH"), _row("test-1", "CONTRADICTION")],
    }
    audit = validate_research_splits(splits)
    assert audit["source_recording_disjoint"] is True
    assert audit["camera_id_is_recording_alias"] is True
    assert audit["camera_generalization_claim_supported"] is False
    assert audit["pairwise_group_overlap_counts"]["environment_id"]["fit__test"] == 1


def test_split_audit_rejects_recording_leakage():
    splits = {
        "fit": [_row("same-recording", "MATCH"), _row("fit-other", "CONTRADICTION")],
        "test": [_row("same-recording", "MATCH"), _row("test-other", "CONTRADICTION")],
    }
    with pytest.raises(ValueError, match="Source recording leakage"):
        validate_research_splits(splits)


def test_split_audit_rejects_statement_annotation_cross_partition_leakage():
    splits = {
        "fit": [
            _row("fit-1", "MATCH", environment="A", statement_source="fit-1", statement_env="A"),
            _row("fit-1", "CONTRADICTION", environment="A", statement_source="fit-2", statement_env="B"),
            _row("fit-2", "MATCH", environment="B", statement_source="fit-2", statement_env="B"),
            _row("fit-2", "CONTRADICTION", environment="B", statement_source="fit-1", statement_env="A"),
        ],
        "test": [
            _row("test-1", "MATCH", environment="A", statement_source="test-1", statement_env="A"),
            _row("test-1", "CONTRADICTION", environment="A", statement_source="fit-1", statement_env="B"),
        ],
    }
    with pytest.raises(ValueError, match="outside their own split"):
        validate_research_splits(splits)


def test_split_audit_checks_proxy_pairing_semantics():
    bad_match = _row("fit-1", "MATCH", statement_source="fit-2", statement_env="Shoplifting")
    splits = {
        "fit": [
            bad_match,
            _row("fit-1", "CONTRADICTION", statement_source="fit-2", statement_env="Arson"),
            _row("fit-2", "MATCH", statement_source="fit-2", statement_env="Arson"),
            _row("fit-2", "CONTRADICTION", statement_source="fit-1", statement_env="Shoplifting"),
        ],
        "test": [
            _row("test-1", "MATCH", statement_source="test-1", statement_env="Shoplifting"),
            _row("test-1", "CONTRADICTION", statement_source="test-2", statement_env="Arson"),
            _row("test-2", "MATCH", statement_source="test-2", statement_env="Arson"),
            _row("test-2", "CONTRADICTION", statement_source="test-1", statement_env="Shoplifting"),
        ],
    }
    with pytest.raises(ValueError, match="MATCH rows must use their own"):
        validate_research_splits(splits)

def test_split_audit_rejects_duplicate_pairs_within_same_split():
    fit_match = _row("fit-1", "MATCH")
    duplicate_match = dict(fit_match)

    splits = {
        "fit": [
            fit_match,
            _row("fit-1", "CONTRADICTION"),
            duplicate_match,
        ],
        "calibration": [
            _row("cal-1", "MATCH"),
            _row("cal-1", "CONTRADICTION"),
        ],
        "test": [
            _row("test-1", "MATCH"),
            _row("test-1", "CONTRADICTION"),
        ],
    }

    with pytest.raises(ValueError, match="duplicate image/statement pair"):
        validate_research_splits(splits)


def test_split_audit_rejects_duplicate_pairs_across_splits():
    fit_match = _row("fit-1", "MATCH")

    duplicate_in_calibration = _row("cal-1", "MATCH")
    duplicate_in_calibration["image_path"] = fit_match["image_path"]
    duplicate_in_calibration["statement"] = fit_match["statement"]

    splits = {
        "fit": [
            fit_match,
            _row("fit-1", "CONTRADICTION"),
        ],
        "calibration": [
            duplicate_in_calibration,
            _row("cal-1", "CONTRADICTION"),
        ],
        "test": [
            _row("test-1", "MATCH"),
            _row("test-1", "CONTRADICTION"),
        ],
    }

    with pytest.raises(ValueError, match="duplicate image/statement pair"):
        validate_research_splits(splits)

def test_split_audit_accepts_artifact_reference_recording_ids():
    splits = {
        "test": [
            _row("test-1", "MATCH"),
            _row("test-1", "CONTRADICTION"),
        ]
    }

    audit = validate_research_splits(
        splits,
        reference_recording_ids={
            "fit": ["fit-1", "fit-2"],
            "calibration": ["cal-1", "cal-2"],
        },
    )

    assert audit["source_recording_disjoint"] is True
    assert audit["reference_splits"]["fit"]["source_recording_count"] == 2
    assert audit["reference_splits"]["calibration"]["source_recording_count"] == 2


def test_split_audit_rejects_overlap_with_artifact_reference_ids():
    splits = {
        "test": [
            _row("test-1", "MATCH"),
            _row("test-1", "CONTRADICTION"),
        ]
    }

    with pytest.raises(ValueError, match="Source recording leakage"):
        validate_research_splits(
            splits,
            reference_recording_ids={
                "fit": ["test-1"],
                "calibration": ["cal-1"],
            },
        )