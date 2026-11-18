from __future__ import annotations

from pathlib import Path

from memory_snapshot import MemorySnapshot
from neo4j_connector import KnowledgeGraphEngine


def test_snapshot_round_trip(tmp_path: Path):
    store = MemorySnapshot(tmp_path / "snap.json")
    assert store.load() is None
    assert store.save(
        memory_store={"CASE-A": [{"statement_id": "s1", "statement_text": "x"}]},
        cases_metadata={"CASE-A": {"id": "CASE-A", "name": "Alpha"}},
        cross_case_links=[],
    )
    loaded = store.load()
    assert loaded is not None
    assert "CASE-A" in loaded["memory_store"]
    assert loaded["cases_metadata"]["CASE-A"]["name"] == "Alpha"


def test_engine_loads_snapshot_when_neo4j_unavailable(tmp_path: Path):
    snapshot_path = tmp_path / "snap.json"
    MemorySnapshot(snapshot_path).save(
        memory_store={
            "CASE-A": [{
                "case_id": "CASE-A",
                "statement_id": "s1",
                "statement_text": "x",
                "evidence_filename": "f.jpg",
                "assessment_status": "MATCH",
                "sinkhorn_cost": 0.12,
            }]
        },
        cases_metadata={"CASE-A": {"id": "CASE-A", "name": "Alpha"}},
        cross_case_links=[],
    )
    engine = KnowledgeGraphEngine(connect=False, snapshot_path=str(snapshot_path))
    graph = engine.get_case_graph("CASE-A")
    assert any(r["statement_text"] == "x" for r in graph["evidence_list"])
    assert any(c["id"] == "CASE-A" for c in engine.get_cases())


def test_engine_flushes_snapshot(tmp_path: Path):
    snapshot_path = tmp_path / "snap.json"
    engine = KnowledgeGraphEngine(connect=False, snapshot_path=str(snapshot_path))
    engine.create_case("CASE-A", name="Alpha")
    engine.create_evidential_link(
        case_id="CASE-A",
        evidence_filename="f.jpg",
        statement_text="x",
        sinkhorn_cost=0.1,
        assessment_status="MATCH",
        evidence_sha256="a" * 64,
    )
    result = engine.flush_snapshot()
    assert result["written"] is True
    assert Path(result["path"]).is_file()


def test_flush_is_noop_when_snapshot_not_configured():
    engine = KnowledgeGraphEngine(connect=False)
    result = engine.flush_snapshot()
    assert result["written"] is False
    assert "not configured" in result["reason"]