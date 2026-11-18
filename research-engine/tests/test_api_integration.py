"""End-to-end API wiring test. Does not require CLIP or Neo4j."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


class FakeEngine:
    def analyze_image(self, image, statement):
        return {
            "assessment_status": "INDETERMINATE",
            "indeterminate_reason": "test: no checkpoint",
            "is_contradiction": None,
            "sinkhorn_divergence": 0.1,
            "standard_entropic_ot_cost": 0.2,
            "global_clip_similarity": 0.3,
            "probabilities": None,
            "model_epistemic_uncertainty": None,
            "visual_quality_metrics": {
                "width": 2, "height": 2,
                "mean_absolute_high_frequency_residual": 0.0,
                "laplacian_variance": 0.0,
                "luminance_standard_deviation": 0.0,
            },
            "tokens": ["a"],
            "patches": ["p1"],
            "heatmap_data": [[0.0]],
            "transport_plan": [[1.0]],
            "ot_diagnostics": {},
            "model_name": "fake",
            "model_revision": "fake",
            "checkpoint_sha256": None,
            "calibration_version": None,
            "feature_schema_version": 4,
            "conformal": None,
        }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_STORAGE_ROOT", str(tmp_path / "evidence"))
    monkeypatch.setenv("MEMORY_SNAPSHOT_PATH", str(tmp_path / "snapshot.json"))
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("EVIDENTIAL_HEAD_PATH", raising=False)
    monkeypatch.delenv("EVIDENTIAL_CALIBRATION_PATH", raising=False)
    monkeypatch.delenv("CONFORMAL_ARTIFACT_PATH", raising=False)
    monkeypatch.setenv("RESEARCH_API_HOST", "127.0.0.1")

    sys.modules.pop("main", None)
    import main  # noqa: F401
    monkeypatch.setattr(main, "get_inference_engine", lambda: FakeEngine())

    with TestClient(main.app) as test_client:
        yield test_client, tmp_path / "evidence"


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_evidence_end_to_end_without_checkpoint(client):
    test_client, evidence_root = client
    response = test_client.post(
        "/analyze_evidence/",
        data={"case_id": "CASE-API-TEST", "statement_text": "A person entered the room."},
        files={"evidence_file": ("frame.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["assessment_status"] == "INDETERMINATE"
    assert body["case_id"] == "CASE-API-TEST"
    assert isinstance(body["graph_status"], dict)
    assert "backend" in body["graph_status"] and "persisted" in body["graph_status"]
    assert body.get("conformal") is None

    # storage_path is case-scoped: "<case_id>/<sha256>.<ext>". The file lives
    # under evidence_root / <case_id> / ..., not directly under evidence_root.
    stored = Path(body["storage_path"])
    stored_path = evidence_root / stored
    assert stored_path.is_file()
    assert stored_path.stat().st_size > 0