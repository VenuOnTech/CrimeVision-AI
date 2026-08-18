"""
CrimeVision AI - Research Engine Core
Module: Evidential Optimal Transport for Contradiction Resolution (Track 1)

Mathematical Formulations:
1. Dirichlet Uncertainty Parameterization:
   alpha_i = ReLU(W * x_i) + 1
   S = sum(alpha_i)
   u = K / S  (Epistemic Uncertainty)

2. Uncertainty-Dampened Cost Matrix:
   C_tilde_{ij} = (1 - sim(v_i, t_j)) * exp(-gamma * max(u_v_i, u_t_j)) + lambda * M_neg

3. Sinkhorn Divergence Objective:
   L_OT = min_P <P, C_tilde> + epsilon * H(P)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import ot  # Python Optimal Transport Library
except ImportError:
    ot = None


class EvidentialUncertaintyEncoder(nn.Module):
    """
    Maps feature embeddings to Dirichlet concentration parameters
    to calculate Epistemic Uncertainty (Doubt).
    """
    def __init__(self, feature_dim: int, num_classes: int = 10):
        super(EvidentialUncertaintyEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Input:  x of shape (N, feature_dim)
        Output: alpha of shape (N, num_classes), uncertainty u of shape (N, 1)
        """
        # Non-negative evidence activation
        evidence = F.relu(self.evidence_layer(x))
        # Dirichlet concentration parameters
        alpha = evidence + 1.0
        # Dirichlet strength S
        S = torch.sum(alpha, dim=-1, keepdim=True)
        # Epistemic Uncertainty u = K / S
        uncertainty = self.num_classes / S
        return alpha, uncertainty


class EvidentialOptimalTransport(nn.Module):
    """
    Calculates the Uncertainty-Dampened Cost Matrix and solves
    the Sinkhorn Divergence transport plan.
    """
    def __init__(self, gamma: float = 2.0, epsilon: float = 0.1, lambda_neg: float = 0.5):
        super(EvidentialOptimalTransport, self).__init__()
        self.gamma = gamma          # Uncertainty damping coefficient
        self.epsilon = epsilon      # Entropic regularization parameter
        self.lambda_neg = lambda_neg # Semantic negation weight

    def compute_cost_matrix(
        self, 
        visual_feats: torch.Tensor, 
        text_feats: torch.Tensor, 
        u_visual: torch.Tensor, 
        u_text: torch.Tensor,
        negation_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Input:
            visual_feats: (N_patches, d) - Visual patch embeddings
            text_feats:   (M_tokens, d)  - Text token embeddings
            u_visual:     (N_patches, 1) - Visual patch uncertainties
            u_text:       (M_tokens, 1)  - Text token uncertainties
            negation_matrix: (N, M)      - Optional semantic negation mask
        Output:
            C_tilde:      (N_patches, M_tokens) - Dampened Cost Matrix
        """
        # 1. Cosine Dissimilarity: (1 - CosineSimilarity)
        visual_norm = F.normalize(visual_feats, p=2, dim=-1)
        text_norm = F.normalize(text_feats, p=2, dim=-1)
        cosine_sim = torch.mm(visual_norm, text_norm.t()) # (N, M)
        base_dissimilarity = 1.0 - cosine_sim

        # 2. Pairwise Maximum Uncertainty: max(u_v_i, u_t_j)
        N, M = cosine_sim.shape
        u_v_expanded = u_visual.expand(N, M)
        u_t_expanded = u_text.t().expand(N, M)
        max_uncertainty = torch.max(u_v_expanded, u_t_expanded)

        # 3. Uncertainty Damping Factor: exp(-gamma * max_u)
        damping_factor = torch.exp(-self.gamma * max_uncertainty)

        # 4. Uncertainty-Dampened Cost Matrix
        C_tilde = base_dissimilarity * damping_factor

        # 5. Optional Semantic Negation Penalty
        if negation_matrix is not None:
            C_tilde = C_tilde + self.lambda_neg * negation_matrix

        return C_tilde

    def solve_sinkhorn(self, C_tilde: torch.Tensor) -> torch.Tensor:
        """
        Solves entropy-regularized Optimal Transport using Sinkhorn-Knopp algorithm.
        """
        N, M = C_tilde.shape
        # Uniform marginal distributions
        a = torch.ones(N, device=C_tilde.device) / N
        b = torch.ones(M, device=C_tilde.device) / M

        C_np = C_tilde.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        if ot is not None:
            # Python Optimal Transport solver
            P_plan = ot.sinkhorn(a_np, b_np, C_np, reg=self.epsilon)
        else:
            # Fallback PyTorch Sinkhorn Implementation
            K_mat = torch.exp(-C_tilde / self.epsilon)
            u = torch.ones(N, device=C_tilde.device) / N
            for _ in range(50):
                v = b / (torch.mm(K_mat.t(), u.unsqueeze(1)).squeeze() + 1e-8)
                u = a / (torch.mm(K_mat, v.unsqueeze(1)).squeeze() + 1e-8)
            P_plan = torch.mm(torch.diag(u), torch.mm(K_mat, torch.diag(v))).cpu().numpy()

        return torch.tensor(P_plan, dtype=torch.float32)

    def forward(
        self, 
        visual_feats: torch.Tensor, 
        text_feats: torch.Tensor, 
        u_visual: torch.Tensor, 
        u_text: torch.Tensor
    ):
        C_tilde = self.compute_cost_matrix(visual_feats, text_feats, u_visual, u_text)
        P_plan = self.solve_sinkhorn(C_tilde)
        
        # Overall Contradiction Score
        contradiction_score = torch.sum(torch.tensor(P_plan) * C_tilde.cpu())
        return contradiction_score, P_plan, C_tilde