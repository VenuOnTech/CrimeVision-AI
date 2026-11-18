"""Live Neo4j integration test. Skipped unless NEO4J_TEST_URI is set."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.neo4j

URI = os.getenv("NEO4J_TEST_URI")
USER = os.getenv("NEO4J_TEST_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_TEST_PASSWORD", "")


@pytest.mark.skipif(not URI, reason="NEO4J_TEST_URI not set")
def test_case_and_evidence_roundtrip_against_live_neo4j():
    from neo4j_connector import KnowledgeGraphEngine

    engine = KnowledgeGraphEngine(
        uri=URI, user=USER, password=PASSWORD, connect=True, ensure_constraints=True
    )
    if not engine.get_storage_status()["neo4j_connected"]:
        pytest.skip("Neo4j is not reachable")

    case_id = "TEST-INTEGRATION-001"
    try:
        engine.delete_case(case_id)
    except KeyError:
        pass

    try:
        engine.create_case(case_id, name="Integration")
        engine.create_evidential_link(
            case_id=case_id,
            evidence_filename="frame.jpg",
            statement_text="A person entered.",
            sinkhorn_cost=0.1,
            assessment_status="MATCH",
            evidence_sha256="a" * 64,
        )
        graph = engine.get_case_graph(case_id)
        assert len(graph["evidence_list"]) == 1
        assert graph["assessment_counts"]["MATCH"] == 1
        assert engine.get_storage_status()["neo4j_connected"] is True
    finally:
        try:
            engine.delete_case(case_id)
        finally:
            engine.close()