"""Unit tests for the split conformal prediction module (schema 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.conformal import (
    ConformalArtifact,
    FEATURE_NAMES,
    coverage_and_abstention,
    extract_conformal_features,
    fit_conformal,
    prediction_set_to_assessment,
    _conformal_quantile,
)


def _make_features(n: int, center, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, len(FEATURE_NAMES)))
    return base * scale + np.asarray(center, dtype=float)


def _make_split(n: int, seed: int):
    match = _make_features(n, center=(0.3, 0.1, 0.4), scale=0.1, seed=seed)
    contra = _make_features(n, center=(0.9, 0.05, 0.7), scale=0.1, seed=seed + 1)
    features = np.vstack([match, contra])
    labels = np.array([0] * n + [1] * n, dtype=np.int64)
    return features, labels


def test_fit_conformal_uses_separate_fit_and_calibration_sets():
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_features, cal_labels = _make_split(60, seed=3)
    artifact = fit_conformal(fit_features, fit_labels, cal_features, cal_labels, alpha=0.10)
    assert artifact.fit_sample_count == 120
    assert artifact.calibration_sample_count == 120
    assert artifact.fit_class_counts == {"MATCH": 60, "CONTRADICTION": 60}
    assert artifact.calibration_class_counts == {"MATCH": 60, "CONTRADICTION": 60}
    assert len(artifact.calibration_scores) == 120


def test_fit_conformal_q_hat_depends_on_fit_set_only_via_scores():
    """If we change only the calibration set, q_hat should change; the fit
    densities should not. This is the observable consequence of the split."""
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_a, cal_a_labels = _make_split(60, seed=3)
    cal_b, cal_b_labels = _make_split(60, seed=7)

    art_a = fit_conformal(fit_features, fit_labels, cal_a, cal_a_labels, alpha=0.10)
    art_b = fit_conformal(fit_features, fit_labels, cal_b, cal_b_labels, alpha=0.10)

    np.testing.assert_allclose(art_a.density_match.mean, art_b.density_match.mean)
    np.testing.assert_allclose(art_a.density_contradiction.mean, art_b.density_contradiction.mean)
    assert art_a.q_hat != art_b.q_hat


def test_fit_conformal_rejects_too_few_fit_samples():
    features, labels = _make_split(10, seed=1)
    with pytest.raises(ValueError, match="fit examples"):
        fit_conformal(features, labels, features, labels, alpha=0.1)


def test_fit_conformal_rejects_too_few_calibration_samples():
    features, labels = _make_split(60, seed=1)
    small, small_labels = _make_split(5, seed=2)
    with pytest.raises(ValueError, match="calibration examples"):
        fit_conformal(features, labels, small, small_labels, alpha=0.1)


def test_fit_conformal_rejects_single_class_in_fit():
    features, labels = _make_split(60, seed=1)
    fit_features = features[:60]
    fit_labels = np.zeros(60, dtype=np.int64)
    with pytest.raises(ValueError, match="fit labels must include both classes"):
        fit_conformal(fit_features, fit_labels, features, labels, alpha=0.1)


def test_prediction_set_separates_well_when_classes_are_distinct():
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_features, cal_labels = _make_split(60, seed=3)
    artifact = fit_conformal(fit_features, fit_labels, cal_features, cal_labels, alpha=0.10)
    match_test = _make_features(30, center=(0.3, 0.1, 0.4), scale=0.1, seed=100)
    contra_test = _make_features(30, center=(0.9, 0.05, 0.7), scale=0.1, seed=200)
    match_singletons = sum(1 for s in artifact.predict_set(match_test) if s == [0])
    contra_singletons = sum(1 for s in artifact.predict_set(contra_test) if s == [1])
    assert match_singletons >= 20
    assert contra_singletons >= 20


def test_artifact_round_trip(tmp_path: Path):
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_features, cal_labels = _make_split(60, seed=3)
    artifact = fit_conformal(fit_features, fit_labels, cal_features, cal_labels, alpha=0.10)
    path = tmp_path / "conformal.json"
    artifact.save(path)
    loaded = ConformalArtifact.load(path)
    assert loaded.alpha == pytest.approx(artifact.alpha)
    assert loaded.q_hat == pytest.approx(artifact.q_hat)
    assert loaded.fit_sample_count == artifact.fit_sample_count
    assert loaded.calibration_sample_count == artifact.calibration_sample_count
    assert len(loaded.calibration_scores) == len(artifact.calibration_scores)
    np.testing.assert_allclose(loaded.density_match.mean, artifact.density_match.mean)


def test_artifact_rejects_incompatible_feature_names():
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_features, cal_labels = _make_split(60, seed=3)
    artifact = fit_conformal(fit_features, fit_labels, cal_features, cal_labels, alpha=0.10)
    payload = artifact.to_json()
    payload["feature_names"] = ["wrong", "names", "here"]
    with pytest.raises(ValueError, match="incompatible feature names"):
        ConformalArtifact.from_json(payload)


def test_artifact_rejects_schema_2():
    with pytest.raises(ValueError, match="requires schema_version=3"):
        ConformalArtifact.from_json({"schema_version": 2})


def test_artifact_rejects_schema_1():
    payload = {
        "schema_version": 1,
        "method": "split_conformal_classification",
        "alpha": 0.1, "q_hat": 0.5,
        "feature_names": list(FEATURE_NAMES),
        "class_to_index": {"MATCH": 0, "CONTRADICTION": 1},
        "density_match": {}, "density_contradiction": {},
        "fit_sample_count": 0, "fit_class_counts": {},
        "calibration_sample_count": 0, "calibration_class_counts": {},
        "calibration_scores": [],
    }
    with pytest.raises(ValueError, match="schema_version"):
        ConformalArtifact.from_json(payload)


def test_with_alpha_recomputes_q_hat_from_stored_scores():
    fit_features, fit_labels = _make_split(60, seed=1)
    cal_features, cal_labels = _make_split(60, seed=3)
    artifact = fit_conformal(fit_features, fit_labels, cal_features, cal_labels, alpha=0.10)
    strict = artifact.with_alpha(0.05)
    loose = artifact.with_alpha(0.20)
    assert loose.q_hat < strict.q_hat
    assert loose.alpha == pytest.approx(0.20)
    assert strict.alpha == pytest.approx(0.05)


def test_prediction_set_to_assessment_maps_correctly():
    assert prediction_set_to_assessment([0]) == ("MATCH", None)
    assert prediction_set_to_assessment([1]) == ("CONTRADICTION", None)
    status, reason = prediction_set_to_assessment([0, 1])
    assert status == "INDETERMINATE" and reason is not None
    status, reason = prediction_set_to_assessment([])
    assert status == "INDETERMINATE" and reason is not None


def test_coverage_and_abstention_reports_selective_accuracy_and_set_size():
    sets = [[0], [1], [0, 1], [0]]
    labels = [0, 1, 1, 1]
    report = coverage_and_abstention(sets, labels, alpha=0.1)
    assert report["sample_count"] == 4
    assert report["singleton_count"] == 3
    assert report["both_classes_count"] == 1
    assert report["abstention_count"] == 1
    assert report["empirical_coverage"] == pytest.approx(0.75)
    # 2 of 3 singletons correct (the [0,1] row and the [0] on label 1 are the errors)
    assert report["selective_accuracy"] == pytest.approx(2 / 3)
    assert report["average_set_size"] == pytest.approx((1 + 1 + 2 + 1) / 4)


def test_extract_conformal_features_produces_finite_vector():
    features = extract_conformal_features({
        "sinkhorn_divergence": 0.3,
        "discarded_mass": 0.1,
        "global_clip_similarity": 0.7,
    })
    assert features.shape == (3,)
    assert np.isfinite(features).all()
    assert features[2] == pytest.approx(0.3)


def test_extract_conformal_features_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        extract_conformal_features({
            "sinkhorn_divergence": float("nan"),
            "discarded_mass": 0.1,
            "global_clip_similarity": 0.7,
        })


def test_quantile_returns_inf_for_too_small_n():
    scores = np.array([0.1, 0.2, 0.3])
    assert _conformal_quantile(scores, alpha=0.01) == float("inf")

def test_recording_group_calibration_uses_max_score_per_recording():
    fit_features, fit_labels = _make_split(30, seed=31)
    calibration_features, calibration_labels = _make_split(30, seed=32)
    fit_groups = [f"fit-{i if i < 30 else i - 30:03d}" for i in range(len(fit_labels))]
    calibration_groups = [f"cal-{i if i < 30 else i - 30:03d}" for i in range(len(calibration_labels))]

    artifact = fit_conformal(
        fit_features, fit_labels,
        calibration_features, calibration_labels,
        alpha=0.10,
        fit_group_ids=fit_groups,
        calibration_group_ids=calibration_groups,
    )

    example_scores = artifact.nonconformity_scores(calibration_features)
    true_scores = np.where(calibration_labels == 0, example_scores[:, 0], example_scores[:, 1])
    expected = []
    for group in sorted(set(calibration_groups)):
        expected.append(max(
            score for score, group_id in zip(true_scores, calibration_groups)
            if group_id == group
        ))

    assert artifact.calibration_unit == "source_recording_id_max"
    assert artifact.calibration_sample_count == 60
    assert artifact.calibration_group_count == 30
    assert len(artifact.calibration_scores) == 30
    np.testing.assert_allclose(artifact.calibration_scores, expected)
    assert artifact.q_hat == pytest.approx(_conformal_quantile(np.asarray(expected), 0.10))


def test_recording_group_calibration_rejects_fit_calibration_group_overlap():
    fit_features, fit_labels = _make_split(30, seed=41)
    calibration_features, calibration_labels = _make_split(30, seed=42)
    fit_groups = [f"shared-{i // 2:03d}" for i in range(len(fit_labels))]
    calibration_groups = [f"shared-{i // 2:03d}" for i in range(len(calibration_labels))]
    with pytest.raises(ValueError, match="fit/calibration group leakage"):
        fit_conformal(
            fit_features, fit_labels,
            calibration_features, calibration_labels,
            alpha=0.10,
            fit_group_ids=fit_groups,
            calibration_group_ids=calibration_groups,
        )


def test_group_calibrated_artifact_round_trip_preserves_group_ids(tmp_path: Path):
    fit_features, fit_labels = _make_split(30, seed=51)
    calibration_features, calibration_labels = _make_split(30, seed=52)
    fit_groups = [f"fit-{i if i < 30 else i - 30:03d}" for i in range(len(fit_labels))]
    calibration_groups = [f"cal-{i if i < 30 else i - 30:03d}" for i in range(len(calibration_labels))]
    artifact = fit_conformal(
        fit_features, fit_labels,
        calibration_features, calibration_labels,
        alpha=0.10,
        cli_model_revision="test-revision",
        feature_schema_version=4,
        fit_group_ids=fit_groups,
        calibration_group_ids=calibration_groups,
    )
    path = tmp_path / "group-conformal.json"
    artifact.save(path)
    restored = ConformalArtifact.load(path)
    assert restored.calibration_unit == "source_recording_id_max"
    assert restored.calibration_group_count == 30
    assert restored.fit_group_ids == tuple(sorted(set(fit_groups)))
    assert restored.calibration_group_ids == tuple(sorted(set(calibration_groups)))
    assert restored.cli_model_revision == "test-revision"
    assert restored.with_alpha(0.20).q_hat == pytest.approx(
        _conformal_quantile(np.asarray(restored.calibration_scores), 0.20)
    )


def test_group_coverage_measures_simultaneous_coverage_per_recording():
    from src.conformal import coverage_by_group

    result = coverage_by_group(
        prediction_sets=[[0], [1], [0, 1], [0]],
        true_labels=[0, 0, 1, 0],
        group_ids=["rec-a", "rec-a", "rec-b", "rec-b"],
    )
    assert result["group_count"] == 2
    assert result["simultaneously_covered_group_count"] == 1
    assert result["simultaneous_group_coverage"] == pytest.approx(0.5)
    assert result["group_sizes"] == {"min": 2, "max": 2}
