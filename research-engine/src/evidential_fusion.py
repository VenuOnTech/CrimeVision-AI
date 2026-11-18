"""Numerically stable, debiased entropic optimal transport and evidential head.

This module deliberately does not turn image-quality degradation into agreement.
Image-quality metrics and learned model uncertainty are reported separately by
inference.py. A trained EvidentialPairHead is required before a class decision
can be issued.

Novelty: quality-aware PARTIAL optimal transport. A per-patch trust weight
derives (a) non-uniform source/target marginals and (b) the transported mass
fraction. Low-trust patches are permitted to be discarded into a dustbin rather
than forced to match. The discarded mass is exposed as a feature to the
evidential head.

The dustbin cost is deliberately large (default 10.0) so that the augmented
Sinkhorn problem effectively isolates the dustbin cells from the real cells.
With a small dustbin cost, discarded mass leaks back into real cells and
Sinkhorn converges very slowly; with a large one, the dustbin decouples and
convergence is dominated by the real block.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornConvergenceError(RuntimeError):
    """Raised when the Sinkhorn solver cannot satisfy its marginal constraints."""


@dataclass(frozen=True)
class SinkhornResult:
    plan: torch.Tensor
    regularized_cost: torch.Tensor
    row_marginal_error: float
    column_marginal_error: float
    iterations: int
    transported_mass: float = 1.0
    discarded_mass: float = 0.0


class LogDomainSinkhorn:
    """Solve entropic OT in log space using KL(P || a x b) regularization."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 2000,
        tolerance: float = 1e-7,
        check_interval: int = 10,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be strictly positive")
        if check_interval < 1:
            raise ValueError("check_interval must be at least 1")

        self.epsilon = float(epsilon)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.check_interval = int(check_interval)

    @staticmethod
    def _validate_cost(cost: torch.Tensor) -> None:
        if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
            raise ValueError("cost must be a non-empty 2-D tensor")
        if not torch.is_floating_point(cost):
            raise TypeError("cost must be a floating-point tensor")
        if not torch.isfinite(cost).all():
            raise ValueError("cost contains NaN or infinite values")

    @staticmethod
    def _validate_marginals(
        masses: torch.Tensor, expected: int, name: str, device: torch.device
    ) -> torch.Tensor:
        if masses.shape != (expected,):
            raise ValueError(f"{name} must have shape ({expected},)")
        if not torch.isfinite(masses).all():
            raise ValueError(f"{name} must be finite")
        if torch.any(masses <= 0):
            raise ValueError(f"{name} must be strictly positive")
        # Tolerance accommodates float32 -> float64 conversion of trust-derived
        # masses. A true normalization bug produces an error far larger than 1e-6.
        if not torch.isclose(
            masses.sum(),
            torch.tensor(1.0, device=device, dtype=masses.dtype),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"{name} must sum to 1")
        return masses

    def solve(
        self,
        cost: torch.Tensor,
        source_mass: torch.Tensor | None = None,
        target_mass: torch.Tensor | None = None,
    ) -> SinkhornResult:
        self._validate_cost(cost)
        original_dtype = cost.dtype
        work_cost = cost.to(dtype=torch.float64)
        n, m = work_cost.shape
        device = work_cost.device

        if source_mass is None:
            a = torch.full((n,), 1.0 / n, dtype=torch.float64, device=device)
        else:
            a = source_mass.to(device=device, dtype=torch.float64)
        if target_mass is None:
            b = torch.full((m,), 1.0 / m, dtype=torch.float64, device=device)
        else:
            b = target_mass.to(device=device, dtype=torch.float64)

        a = self._validate_marginals(a, n, "source_mass", device)
        b = self._validate_marginals(b, m, "target_mass", device)

        log_a = torch.log(a)
        log_b = torch.log(b)
        log_kernel = -work_cost / self.epsilon
        log_v = torch.zeros_like(b)
        tiny = torch.finfo(torch.float64).tiny
        converged = False
        row_error = float("inf")
        column_error = float("inf")
        final_plan = None
        used_iterations = self.max_iterations

        for iteration in range(1, self.max_iterations + 1):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_kernel.transpose(0, 1) + log_u.unsqueeze(0), dim=1)

            if iteration % self.check_interval == 0 or iteration == self.max_iterations:
                log_plan = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
                plan = torch.exp(log_plan)
                row_error = float(torch.max(torch.abs(plan.sum(dim=1) - a)).item())
                column_error = float(torch.max(torch.abs(plan.sum(dim=0) - b)).item())
                final_plan = plan
                if max(row_error, column_error) <= self.tolerance:
                    converged = True
                    used_iterations = iteration
                    break

        if not converged or final_plan is None:
            raise SinkhornConvergenceError(
                "Sinkhorn did not converge within "
                f"{self.max_iterations} iterations; row error={row_error:.3e}, "
                f"column error={column_error:.3e}. Increase epsilon/iterations "
                "or inspect the cost matrix and feature inputs."
            )

        log_plan = torch.log(final_plan.clamp_min(tiny))
        log_reference = log_a.unsqueeze(1) + log_b.unsqueeze(0)
        kl_value = torch.sum(final_plan * (log_plan - log_reference))
        transport_cost = torch.sum(final_plan * work_cost)
        objective = transport_cost + self.epsilon * kl_value

        return SinkhornResult(
            plan=final_plan.to(dtype=original_dtype),
            regularized_cost=objective,
            row_marginal_error=row_error,
            column_marginal_error=column_error,
            iterations=used_iterations,
            transported_mass=1.0,
            discarded_mass=0.0,
        )

    def solve_partial(
        self,
        cost: torch.Tensor,
        transported_mass: float,
        dustbin_cost: float = 10.0,
        source_mass: torch.Tensor | None = None,
        target_mass: torch.Tensor | None = None,
    ) -> SinkhornResult:
        """Partial OT with a dustbin that absorbs discarded mass.

        Augmented cost is (N+1) x (M+1): original block, dustbin column,
        dustbin row, and a zero dustbin-dustbin entry. Source and target
        marginals are scaled by `transported_mass`; the remaining mass goes to
        the dustbin on each side. With transported_mass == 1 this reduces
        exactly to the full solver.
        """
        self._validate_cost(cost)
        if not math.isfinite(transported_mass) or not (0.0 < transported_mass <= 1.0):
            raise ValueError("transported_mass must be in (0, 1]")
        if not math.isfinite(dustbin_cost) or dustbin_cost < 0:
            raise ValueError("dustbin_cost must be finite and non-negative")

        if transported_mass >= 1.0 - 1e-9:
            result = self.solve(cost, source_mass, target_mass)
            return SinkhornResult(
                plan=result.plan,
                regularized_cost=result.regularized_cost,
                row_marginal_error=result.row_marginal_error,
                column_marginal_error=result.column_marginal_error,
                iterations=result.iterations,
                transported_mass=1.0,
                discarded_mass=0.0,
            )

        n, m = cost.shape
        device = cost.device
        if source_mass is None:
            a = torch.full((n,), 1.0 / n, dtype=torch.float64, device=device)
        else:
            a = source_mass.to(device=device, dtype=torch.float64)
        if target_mass is None:
            b = torch.full((m,), 1.0 / m, dtype=torch.float64, device=device)
        else:
            b = target_mass.to(device=device, dtype=torch.float64)

        a = self._validate_marginals(a, n, "source_mass", device)
        b = self._validate_marginals(b, m, "target_mass", device)

        mass = float(transported_mass)
        dust = 1.0 - mass
        work_cost = cost.to(dtype=torch.float64)

        aug_cost = torch.empty((n + 1, m + 1), dtype=torch.float64, device=device)
        aug_cost[:n, :m] = work_cost
        aug_cost[:n, m] = float(dustbin_cost)
        aug_cost[n, :m] = float(dustbin_cost)
        aug_cost[n, m] = 0.0

        aug_a = torch.cat([a * mass, torch.tensor([dust], dtype=torch.float64, device=device)])
        aug_b = torch.cat([b * mass, torch.tensor([dust], dtype=torch.float64, device=device)])

        result = self.solve(aug_cost, aug_a, aug_b)
        original_plan = result.plan[:n, :m]
        transported = float(original_plan.sum().item())
        return SinkhornResult(
            plan=original_plan,
            regularized_cost=result.regularized_cost,
            row_marginal_error=result.row_marginal_error,
            column_marginal_error=result.column_marginal_error,
            iterations=result.iterations,
            transported_mass=transported,
            discarded_mass=max(0.0, 1.0 - transported),
        )


class EvidentialOptimalTransport(nn.Module):
    """Debiased Sinkhorn divergence with quality-aware partial transport."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 2000,
        tolerance: float = 1e-7,
        check_interval: int = 10,
        negative_tolerance: float = 1e-5,
        dustbin_cost: float = 10.0,
    ) -> None:
        super().__init__()
        if negative_tolerance < 0:
            raise ValueError("negative_tolerance cannot be negative")
        if dustbin_cost < 0 or not math.isfinite(dustbin_cost):
            raise ValueError("dustbin_cost must be finite and non-negative")
        self.solver = LogDomainSinkhorn(
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
            check_interval=check_interval,
        )
        self.negative_tolerance = float(negative_tolerance)
        self.dustbin_cost = float(dustbin_cost)
        self.last_diagnostics: Dict[str, float | int] = {}

    @staticmethod
    def _normalise(features: torch.Tensor, name: str) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"{name} must have shape (count, feature_dim), count > 0")
        if not torch.is_floating_point(features):
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        norms = torch.linalg.vector_norm(features, ord=2, dim=-1, keepdim=True)
        if torch.any(norms <= torch.finfo(features.dtype).eps):
            raise ValueError(f"{name} contains a zero-norm feature vector")
        return features / norms

    @staticmethod
    def _trust_tensor(
        trust, expected_len: int, name: str, device, dtype
    ) -> torch.Tensor:
        if trust is None:
            return torch.ones(expected_len, device=device, dtype=dtype)
        tensor = torch.as_tensor(trust, device=device, dtype=dtype).flatten()
        if tensor.numel() != expected_len:
            raise ValueError(f"{name} must have {expected_len} entries, got {tensor.numel()}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return tensor.clamp(min=1e-6, max=1.0)

    @staticmethod
    def compute_cost_matrix(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if source_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError("source_features and target_features must be 2-D")
        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError("source and target features must share the same feature dimension")
        source = EvidentialOptimalTransport._normalise(source_features, "source_features")
        target = EvidentialOptimalTransport._normalise(target_features, "target_features")
        return (1.0 - source @ target.transpose(0, 1)).clamp_min(0.0)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_trust: torch.Tensor | None = None,
        text_trust: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("visual and text features must both be 2-D tensors")
        if visual_features.device != text_features.device:
            raise ValueError("visual and text features must be on the same device")
        if visual_features.shape[1] != text_features.shape[1]:
            raise ValueError("visual and text features must share the same feature dimension")

        visual = self._normalise(visual_features, "visual_features")
        text = self._normalise(text_features, "text_features")

        v_trust = self._trust_tensor(
            visual_trust, visual.shape[0], "visual_trust", visual.device, visual.dtype
        )
        t_trust = self._trust_tensor(
            text_trust, text.shape[0], "text_trust", text.device, text.dtype
        )
        visual_mass = v_trust / v_trust.sum()
        text_mass = t_trust / t_trust.sum()
        transported = float(
            torch.minimum(v_trust.mean(), t_trust.mean()).clamp(min=1e-6, max=1.0).item()
        )

        cross_cost = (1.0 - visual @ text.transpose(0, 1)).clamp_min(0.0)
        visual_self_cost = (1.0 - visual @ visual.transpose(0, 1)).clamp_min(0.0)
        text_self_cost = (1.0 - text @ text.transpose(0, 1)).clamp_min(0.0)

        cross_result = self.solver.solve_partial(
            cross_cost, transported, self.dustbin_cost, visual_mass, text_mass
        )
        visual_self_result = self.solver.solve_partial(
            visual_self_cost, transported, self.dustbin_cost, visual_mass, visual_mass
        )
        text_self_result = self.solver.solve_partial(
            text_self_cost, transported, self.dustbin_cost, text_mass, text_mass
        )

        divergence = (
            cross_result.regularized_cost
            - 0.5 * visual_self_result.regularized_cost
            - 0.5 * text_self_result.regularized_cost
        )

        value = float(divergence.detach().cpu().item())
        if value < -self.negative_tolerance:
            raise RuntimeError(
                "Computed Sinkhorn divergence is materially negative "
                f"({value:.6g}); inspect solver convergence and feature validity."
            )
        if value < 0.0:
            divergence = divergence.new_zeros(())

        self.last_diagnostics = {
            "cross_regularized_ot": float(cross_result.regularized_cost.detach().cpu().item()),
            "visual_self_regularized_ot": float(visual_self_result.regularized_cost.detach().cpu().item()),
            "text_self_regularized_ot": float(text_self_result.regularized_cost.detach().cpu().item()),
            "cross_row_marginal_error": cross_result.row_marginal_error,
            "cross_column_marginal_error": cross_result.column_marginal_error,
            "cross_sinkhorn_iterations": cross_result.iterations,
            "transported_mass": float(cross_result.transported_mass),
            "discarded_mass": float(cross_result.discarded_mass),
            "dustbin_cost": self.dustbin_cost,
        }
        return divergence, cross_result.plan, cross_cost


class EvidentialPairHead(nn.Module):
    """Supervised two-class evidential head for MATCH vs CONTRADICTION pairs.

    This network is randomly initialized when constructed. It MUST be trained
    on human-labelled image/statement pairs and loaded from a checkpoint before
    it is used for inference.
    """

    PAIR_FEATURE_DIM = 1026  # 512 abs-diff + 512 elementwise + divergence + discarded mass
    QUALITY_FEATURE_DIM = 3
    INPUT_DIM = PAIR_FEATURE_DIM + QUALITY_FEATURE_DIM
    NUM_CLASSES = 2

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        if input_dim != self.INPUT_DIM:
            raise ValueError(f"Expected input_dim={self.INPUT_DIM}, received {input_dim}")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, self.NUM_CLASSES),
        )

    def forward(self, pair_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pair_features.ndim != 2 or pair_features.shape[0] == 0 or pair_features.shape[1] != self.INPUT_DIM:
            raise ValueError(f"pair_features must have shape (non-empty batch, {self.INPUT_DIM})")
        if not torch.isfinite(pair_features).all():
            raise ValueError("pair_features must contain only finite values")
        evidence = F.softplus(self.network(pair_features))
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        probabilities = alpha / strength
        epistemic_uncertainty = self.NUM_CLASSES / strength
        return alpha, probabilities, epistemic_uncertainty.squeeze(-1)


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(ones)), returned per sample."""
    if (
        alpha.ndim != 2
        or alpha.shape[0] == 0
        or alpha.shape[1] < 2
        or not torch.isfinite(alpha).all()
        or torch.any(alpha < 1.0)
    ):
        raise ValueError("alpha must be a non-empty 2-D tensor with at least 2 classes and concentration >= 1")
    classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1, keepdim=True)
    log_normalizer = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        - torch.lgamma(torch.tensor(float(classes), device=alpha.device, dtype=alpha.dtype))
    )
    digamma_delta = torch.digamma(alpha) - torch.digamma(strength)
    return (log_normalizer + ((alpha - 1.0) * digamma_delta).sum(dim=-1, keepdim=True)).squeeze(-1)


def evidential_classification_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    annealing: float = 0.01,
) -> torch.Tensor:
    """Expected categorical cross-entropy plus annealed KL regularization."""
    if (
        alpha.ndim != 2
        or alpha.shape[0] == 0
        or alpha.shape[1] < 2
        or not torch.isfinite(alpha).all()
        or torch.any(alpha < 1.0)
    ):
        raise ValueError("alpha must be finite, non-empty, 2-D and have at least 2 classes with concentration >= 1")
    if labels.ndim != 1 or labels.shape[0] != alpha.shape[0]:
        raise ValueError("labels must have shape (batch,)")
    if labels.device != alpha.device:
        raise ValueError("labels and alpha must be on the same device")
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("labels must use an integer tensor dtype")
    if torch.any(labels < 0) or torch.any(labels >= alpha.shape[-1]):
        raise ValueError("labels contain an invalid class index")
    if not math.isfinite(float(annealing)) or annealing < 0:
        raise ValueError("annealing must be finite and non-negative")

    strength = alpha.sum(dim=-1, keepdim=True)
    expected_nll = torch.digamma(strength).squeeze(-1) - torch.digamma(alpha).gather(
        1, labels.long().unsqueeze(1)
    ).squeeze(1)

    one_hot = F.one_hot(labels.long(), num_classes=alpha.shape[-1]).to(dtype=alpha.dtype)
    adjusted_alpha = one_hot + (1.0 - one_hot) * alpha
    kl = dirichlet_kl_to_uniform(adjusted_alpha)
    return (expected_nll + float(annealing) * kl).mean()