import pytest
import torch

from src.inference import EvidentialInferenceEngine
from src.evidential_fusion import (
    EvidentialOptimalTransport,
    EvidentialPairHead,
    LogDomainSinkhorn,
    SinkhornConvergenceError,
    evidential_classification_loss,
)


def test_sinkhorn_divergence_is_zero_for_identical_feature_sets():
    torch.manual_seed(7)
    features = torch.randn(9, 32)
    solver = EvidentialOptimalTransport(epsilon=0.1, tolerance=1e-7)
    divergence, plan, cost = solver(features, features.clone())
    assert abs(float(divergence)) < 1e-6
    assert plan.shape == (9, 9)
    assert cost.shape == (9, 9)


def test_sinkhorn_divergence_is_symmetric():
    torch.manual_seed(11)
    visual = torch.randn(7, 24)
    text = torch.randn(5, 24)
    solver = EvidentialOptimalTransport(epsilon=0.1, tolerance=1e-7)
    forward, _, _ = solver(visual, text)
    reverse, _, _ = solver(text, visual)
    assert float(forward) >= 0.0
    assert float(reverse) >= 0.0
    assert abs(float(forward - reverse)) < 1e-5


def test_log_sinkhorn_respects_marginals():
    torch.manual_seed(13)
    cost = torch.rand(8, 6, dtype=torch.float64) * 2.0
    result = LogDomainSinkhorn(epsilon=0.1, max_iterations=2000, tolerance=1e-8).solve(cost)
    expected_rows = torch.full((8,), 1 / 8, dtype=torch.float64)
    expected_cols = torch.full((6,), 1 / 6, dtype=torch.float64)
    assert torch.allclose(result.plan.sum(dim=1), expected_rows, atol=1e-8)
    assert torch.allclose(result.plan.sum(dim=0), expected_cols, atol=1e-8)
    assert torch.isfinite(result.regularized_cost)


def test_solver_rejects_non_finite_costs():
    cost = torch.tensor([[0.0, float("nan")], [1.0, 0.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        LogDomainSinkhorn().solve(cost)


def test_solver_raises_when_not_converged():
    torch.manual_seed(29)
    cost = torch.rand(5, 5, dtype=torch.float64) * 2.0
    solver = LogDomainSinkhorn(epsilon=0.001, max_iterations=1, tolerance=1e-14, check_interval=1)
    with pytest.raises(SinkhornConvergenceError):
        solver.solve(cost)


def test_evidential_head_and_loss_are_finite_and_differentiable():
    torch.manual_seed(17)
    model = EvidentialPairHead()
    features = torch.randn(6, EvidentialPairHead.INPUT_DIM)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    alpha, probabilities, uncertainty = model(features)
    loss = evidential_classification_loss(alpha, labels, annealing=0.01)
    assert torch.all(alpha >= 1.0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.all((uncertainty > 0.0) & (uncertainty <= 1.0))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_pair_head_input_includes_explicit_quality_features():
    # Schema 4: 512 abs-diff + 512 product + divergence + discarded mass = 1026
    # plus 3 quality features = 1029.
    assert EvidentialPairHead.PAIR_FEATURE_DIM == 1026
    assert EvidentialPairHead.QUALITY_FEATURE_DIM == 3
    assert EvidentialPairHead.INPUT_DIM == 1029
    model = EvidentialPairHead()
    alpha, probabilities, uncertainty = model(torch.randn(3, 1029))
    assert alpha.shape == (3, 2)
    assert probabilities.shape == (3, 2)
    assert uncertainty.shape == (3,)


def test_pair_head_rejects_wrong_feature_dimension():
    model = EvidentialPairHead()
    with pytest.raises(ValueError, match="pair_features must have shape"):
        model(torch.randn(2, 512))


def test_partial_ot_discards_low_trust_mass():
    torch.manual_seed(31)
    visual = torch.randn(9, 16)
    text = torch.randn(7, 16)
    solver = EvidentialOptimalTransport(epsilon=0.1, tolerance=1e-5)
    trust = torch.full((9,), 0.1)
    divergence, plan, cost = solver(visual, text, visual_trust=trust)
    assert plan.shape == (9, 7)
    assert cost.shape == (9, 7)
    assert solver.last_diagnostics["discarded_mass"] > 0.8
    assert solver.last_diagnostics["transported_mass"] < 0.2
    assert torch.isfinite(divergence)


def test_full_trust_reduces_exactly_to_full_ot():
    torch.manual_seed(37)
    visual = torch.randn(6, 12)
    text = torch.randn(5, 12)
    solver = EvidentialOptimalTransport(epsilon=0.1, tolerance=1e-5)
    full_div, full_plan, _ = solver(visual, text)
    partial_div, partial_plan, _ = solver(
        visual, text,
        visual_trust=torch.ones(6), text_trust=torch.ones(5),
    )
    assert torch.allclose(full_div, partial_div, atol=1e-9)
    assert torch.allclose(full_plan, partial_plan, atol=1e-9)


def test_quality_feature_standardization_is_finite_and_clipped():
    raw = torch.tensor([[10.0, 100.0, 20.0], [12.0, 200.0, 30.0]])
    mean = torch.tensor([11.0, 150.0, 25.0])
    std = torch.tensor([1.0, 50.0, 5.0])
    standardized = EvidentialInferenceEngine.standardize_quality_features(raw, mean, std)
    assert standardized.shape == (2, 3)
    assert torch.isfinite(standardized).all()
    assert torch.allclose(standardized[0], torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(standardized[1], torch.tensor([1.0, 1.0, 1.0]))


def test_quality_feature_standardization_rejects_zero_std():
    raw = torch.ones((1, 3))
    mean = torch.zeros(3)
    std = torch.tensor([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="standard deviations must be positive"):
        EvidentialInferenceEngine.standardize_quality_features(raw, mean, std)


def test_quality_features_outside_training_range_are_rejected():
    mean = torch.zeros(3)
    std = torch.ones(3)
    in_range = torch.tensor([[0.0, 2.0, -3.0]])
    out_of_range = torch.tensor([[0.0, 11.0, -3.0]])
    assert EvidentialInferenceEngine.quality_features_in_training_domain(in_range, mean, std)
    assert not EvidentialInferenceEngine.quality_features_in_training_domain(out_of_range, mean, std)


def test_evidential_loss_rejects_non_integer_labels():
    model = EvidentialPairHead()
    alpha, _, _ = model(torch.randn(2, EvidentialPairHead.INPUT_DIM))
    with pytest.raises(TypeError, match="integer tensor dtype"):
        evidential_classification_loss(alpha, torch.tensor([0.0, 1.0]))