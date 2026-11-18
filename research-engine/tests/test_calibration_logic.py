import numpy as np
import pytest

from calibrate_decision_threshold import (
    calibrate_binary_baseline,
    choose_selective_thresholds,
)


def test_selective_calibration_finds_zero_error_operating_point():
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.10, 0.20, 0.30, 0.70, 0.80, 0.90])
    uncertainty = np.asarray([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])

    probability_threshold, uncertainty_limit, metrics = choose_selective_thresholds(
        y_true=labels,
        p_contradiction=probabilities,
        epistemic_uncertainty=uncertainty,
        max_error_rate=0.0,
        min_accepted_per_class=2,
    )

    predictions = (probabilities >= probability_threshold).astype(np.int64)
    assert np.array_equal(predictions, labels)
    assert uncertainty_limit == pytest.approx(0.10)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["empirical_error_rate"] == pytest.approx(0.0)


def test_selective_calibration_never_accepts_quality_ood_examples():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.10, 0.20, 0.80, 0.90])
    uncertainty = np.asarray([0.10, 0.10, 0.10, 0.10])
    quality_in_domain = np.asarray([True, False, True, False])

    _, _, metrics = choose_selective_thresholds(
        y_true=labels,
        p_contradiction=probabilities,
        epistemic_uncertainty=uncertainty,
        max_error_rate=0.0,
        min_accepted_per_class=1,
        quality_in_domain=quality_in_domain,
    )

    assert metrics["accepted_sample_count"] == 2
    assert metrics["coverage"] == pytest.approx(0.5)


def test_selective_calibration_fails_when_minimum_per_class_is_unavailable():
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.2, 0.9])
    uncertainty = np.asarray([0.1, 0.1, 0.1])

    with pytest.raises(RuntimeError, match="No selective operating point"):
        choose_selective_thresholds(
            y_true=labels,
            p_contradiction=probabilities,
            epistemic_uncertainty=uncertainty,
            max_error_rate=0.0,
            min_accepted_per_class=2,
        )


def test_baseline_calibration_selects_direction_and_threshold_on_labels():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    similarities = np.asarray([0.8, 0.9, 0.1, 0.2])

    threshold, direction, score = calibrate_binary_baseline(similarities, labels)

    assert direction == "below_or_equal"
    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(1.0)


def test_selective_calibration_enforces_per_class_error_limit():
    # One false negative is only ~1% of 101 examples, but it is a 100%
    # false-negative rate for the minority contradiction class.
    labels = np.asarray([0] * 100 + [1], dtype=np.int64)
    probabilities = np.asarray([0.5] * 100 + [0.4], dtype=float)
    uncertainty = np.asarray([0.1] * 101, dtype=float)

    with pytest.raises(RuntimeError, match="overall and per-class empirical error limits"):
        choose_selective_thresholds(
            y_true=labels,
            p_contradiction=probabilities,
            epistemic_uncertainty=uncertainty,
            max_error_rate=0.02,
            min_accepted_per_class=1,
            max_class_error_rate=0.10,
        )


def test_evaluation_reports_false_positive_and_false_negative_rates():
    from evaluate_research_engine import binary_metrics

    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
