from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from neo4j_connector import KnowledgeGraphEngine


def make_engine() -> KnowledgeGraphEngine:
    return KnowledgeGraphEngine(connect=False)


def add_record(engine, case_id, filename="camera.mp4", statement="A person entered.", **kwargs):
    return engine.create_evidential_link(
        case_id=case_id,
        evidence_filename=filename,
        statement_text=statement,
        sinkhorn_cost=kwargs.get("sinkhorn_cost", 0.12),
        assessment_status=kwargs.get("assessment_status", "INDETERMINATE"),
        evidence_sha256=kwargs.get("evidence_sha256", "a" * 64),
        visual_quality_metrics={"width": 640, "height": 480},
        investigator_advisory="Human review required.",
    )


def test_evidence_and_statement_ids_are_case_scoped():
    engine = make_engine()
    first = add_record(engine, "CASE-A")
    second = add_record(engine, "CASE-B")

    assert first["evidence_id"] != second["evidence_id"]
    assert first["statement_id"] != second["statement_id"]
    assert len(engine.get_case_graph("CASE-A")["evidence_list"]) == 1
    assert len(engine.get_case_graph("CASE-B")["evidence_list"]) == 1


def test_same_statement_for_different_evidence_has_distinct_pair_nodes():
    engine = make_engine()
    first = add_record(engine, "CASE-A", filename="first.jpg", evidence_sha256="1" * 64)
    second = add_record(engine, "CASE-A", filename="second.jpg", evidence_sha256="2" * 64)

    assert first["statement_text"] == second["statement_text"]
    assert first["statement_id"] != second["statement_id"]
    assert len(engine.get_case_graph("CASE-A")["evidence_list"]) == 2


def test_clear_evidence_keeps_case_metadata_and_cross_case_links():
    engine = make_engine()
    engine.create_case("CASE-A", name="Alpha")
    engine.create_case("CASE-B", name="Beta")
    add_record(engine, "CASE-A")
    engine.add_cross_case_relation("CASE-A", "CASE-B")

    result = engine.clear_case_evidence("CASE-A")

    assert result["case_deleted"] is False
    assert result["database_cleared"] is False  # no Neo4j server in this test
    assert engine.get_case_graph("CASE-A")["evidence_list"] == []
    assert any(case["id"] == "CASE-A" for case in engine.get_cases())
    assert len(engine.get_cross_case_relations()) == 1


def test_delete_case_removes_case_from_memory_and_links():
    engine = make_engine()
    engine.create_case("CASE-A")
    engine.create_case("CASE-B")
    add_record(engine, "CASE-A")
    engine.add_cross_case_relation("CASE-A", "CASE-B")

    result = engine.delete_case("CASE-A")

    assert result["case_deleted"] is True
    assert result["database_deleted"] is False
    assert all(case["id"] != "CASE-A" for case in engine.get_cases())
    assert engine.get_case_graph("CASE-A")["evidence_list"] == []
    assert engine.get_cross_case_relations() == []


def test_memory_fallback_is_reported_as_volatile():
    engine = make_engine()
    record = add_record(engine, "CASE-A")

    assert record["persisted"] is False
    assert record["storage_backend"] == "memory"
    assert "process memory" in record["persistence_message"]
    assert engine.get_storage_status()["durable"] is False


def test_invalid_assessment_status_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="Unsupported assessment_status"):
        add_record(engine, "CASE-A", assessment_status="MAYBE")


def test_update_case_preserves_unsupplied_metadata():
    engine = make_engine()
    engine.create_case(
        "CASE-A",
        name="Original name",
        description="Original description",
        category="Original category",
    )

    updated = engine.update_case("CASE-A", name="Updated name")

    assert updated["name"] == "Updated name"
    assert updated["description"] == "Original description"
    assert updated["category"] == "Original category"


def test_update_missing_case_does_not_create_a_new_case():
    engine = make_engine()
    with pytest.raises(KeyError, match="does not exist"):
        engine.update_case("MISSING", name="Should not be created")
    assert engine.get_cases() == []


def test_invalid_sha256_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        add_record(engine, "CASE-A", evidence_sha256="not-a-hash")


def test_invalid_uncertainty_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="between 0 and 1"):
        engine.create_evidential_link(
            case_id="CASE-A",
            evidence_filename="frame.jpg",
            statement_text="A person entered.",
            sinkhorn_cost=0.2,
            model_epistemic_uncertainty=1.2,
        )


def test_non_finite_sinkhorn_value_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="finite and non-negative"):
        engine.create_evidential_link(
            case_id="CASE-A",
            evidence_filename="frame.jpg",
            statement_text="A person entered.",
            sinkhorn_cost=math.inf,
        )


def test_cross_case_relationship_requires_two_existing_cases():
    engine = make_engine()
    engine.create_case("CASE-A")
    with pytest.raises(KeyError, match="CASE-B"):
        engine.add_cross_case_relation("CASE-A", "CASE-B")


def test_cross_case_confidence_is_range_checked():
    engine = make_engine()
    engine.create_case("CASE-A")
    engine.create_case("CASE-B")
    with pytest.raises(ValueError, match="between 0 and 1"):
        engine.add_cross_case_relation("CASE-A", "CASE-B", confidence=1.5)


def test_all_evidence_records_include_memory_fallback_records():
    engine = make_engine()
    add_record(engine, "CASE-A", filename="one.jpg", evidence_sha256="1" * 64)
    add_record(engine, "CASE-B", filename="two.jpg", evidence_sha256="2" * 64)

    records = engine.get_all_evidence_records()

    assert len(records) == 2
    assert {record["case_id"] for record in records} == {"CASE-A", "CASE-B"}


def test_legacy_missing_status_defaults_to_indeterminate_in_graph():
    engine = make_engine()
    add_record(engine, "CASE-A")
    engine.memory_store["CASE-A"][0].pop("assessment_status")
    graph = engine.get_case_graph("CASE-A")

    assert graph["evidence_list"][0]["assessment_status"] == "INDETERMINATE"
    statement_nodes = [node for node in graph["nodes"] if node["data"].get("type") == "statement"]
    assert statement_nodes[0]["data"]["assessment_status"] == "INDETERMINATE"
    assert statement_nodes[0]["data"]["is_contradiction"] is None


def test_keyword_overlap_link_is_not_marked_as_investigator_confidence():
    engine = make_engine()
    engine.create_case("CASE-A")
    engine.create_case("CASE-B")

    link = engine.add_cross_case_relation(
        "CASE-A",
        "CASE-B",
        relation_type="POTENTIAL_KEYWORD_OVERLAP",
        confidence=0.0,
    )

    assert link["confidence_interpretation"] == "heuristic_candidate_not_calibrated"
