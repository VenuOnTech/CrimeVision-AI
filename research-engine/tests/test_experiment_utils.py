from __future__ import annotations

import numpy as np
import pytest

from experiment_utils import cache_fingerprint, load_feature_cache, save_feature_cache


def test_cache_fingerprint_changes_when_csv_or_model_changes(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,label\n1,MATCH\n", encoding="utf-8")
    common = {
        "csv_path": csv_path,
        "model_name": "clip-test",
        "model_revision": "revision-a",
        "feature_schema_version": 4,
        "inference_config": {"trust_mode": "heterogeneity"},
    }
    first = cache_fingerprint(**common)
    assert cache_fingerprint(**{**common, "model_revision": "revision-b"}) != first
    assert cache_fingerprint(**{**common, "inference_config": {"trust_mode": "absolute"}}) != first
    csv_path.write_text("id,label\n2,CONTRADICTION\n", encoding="utf-8")
    assert cache_fingerprint(**common) != first


def test_feature_cache_round_trip_and_fingerprint_rejection(tmp_path):
    path = tmp_path / "features.npz"
    X = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    y = np.asarray([0, 1], dtype=np.int64)
    save_feature_cache(path, X, y, "fingerprint-a")
    X_loaded, y_loaded = load_feature_cache(path, "fingerprint-a")
    np.testing.assert_allclose(X_loaded, X)
    np.testing.assert_array_equal(y_loaded, y)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_feature_cache(path, "fingerprint-b")


def test_cache_fingerprint_changes_when_image_bytes_change(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("image_path,label\nframe.png,MATCH\n", encoding="utf-8")
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"image-bytes-a")
    common = {
        "csv_path": csv_path,
        "model_name": "clip-test",
        "model_revision": "revision-a",
        "feature_schema_version": 4,
        "inference_config": {"trust_mode": "heterogeneity"},
        "image_paths": [image_path],
    }
    first = cache_fingerprint(**common)
    image_path.write_bytes(b"image-bytes-b")
    assert cache_fingerprint(**common) != first


def test_ablation_subset_keeps_complete_recording_groups():
    from ablate_conformal_scores import _select_rows

    rows = []
    for recording in ("rec-a", "rec-b", "rec-c"):
        rows.extend([
            {"source_recording_id": recording, "label": "MATCH"},
            {"source_recording_id": recording, "label": "CONTRADICTION"},
        ])
    selected = _select_rows(rows, max_examples=2, seed=21)
    selected_groups = {row["source_recording_id"] for row in selected}
    assert len(selected_groups) == 2
    assert len(selected) == 4
    assert all(
        {row["label"] for row in selected if row["source_recording_id"] == group}
        == {"MATCH", "CONTRADICTION"}
        for group in selected_groups
    )
