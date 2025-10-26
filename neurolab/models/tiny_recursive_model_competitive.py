"""
TinyRecursiveModelCompetitive (v1) - Base recursive model with competitive learning.

Base model for LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn


class TinyRecursiveModelCompetitive(nn.Module):
    """
    Base recursive model with competitive learning mechanism and affect modulation.

    This is the foundation model (v1) that implements:
    - Recursive refinement over K outer iterations
    - Inner iteration loops for latent state updates
    - Affect (emotion) modulation through gating
    - Competitive learning dynamics

    Args:
        dim (int): Dimension of embeddings. Default: 128
        inner (int): Number of inner iterations per outer iteration. Default: 4
        affect_w (float): Weight for affect modulation. Default: 0.3

    Returns:
        Tuple[torch.Tensor, list]:
            - Output answer state of shape [batch, dim]
            - List of confidence values (one per outer iteration)
    """

    def __init__(self, dim=128, inner=4, affect_w=0.3):
        super().__init__()
        self.dim = dim
        self.inner = inner
        self.affect_w = affect_w

        self.ln_latent = nn.LayerNorm(dim*3)
        self.ln_answer = nn.LayerNorm(dim*2)
        self.ln_z = nn.LayerNorm(dim)

        self.latent = nn.Sequential(
            nn.Linear(dim*3, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

        self.answer = nn.Sequential(
            nn.Linear(dim*2, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

        self.affect_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.Tanh()
        )
        self.affect_gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y_init, affect_vec=None, K=5):
        """
        Forward pass with recursive refinement.

        Args:
            x (torch.Tensor): Input question/context of shape [batch, dim]
            y_init (torch.Tensor): Initial answer state of shape [batch, dim]
            affect_vec (torch.Tensor, optional): Affect modulation vector of shape [batch, 3]
                Typically PAD (Pleasure-Arousal-Dominance) values
            K (int): Number of outer recursive iterations. Default: 5

        Returns:
            Tuple[torch.Tensor, list]:
                - Final answer state y of shape [batch, dim]
                - List of K confidence values
        """
        y = y_init.clone()
        z = torch.zeros_like(y)
        confidences = []

        for k in range(K):
            # Inner iterations for latent state refinement
            for n in range(self.inner):
                latent_input = self.ln_latent(torch.cat([x, y, z], dim=-1))
                z_delta = self.latent(latent_input)
                z = z + 0.3 * z_delta
                z = self.ln_z(z)

            # Affect modulation (if provided)
            if affect_vec is not None:
                affect_latent = self.affect_proj(affect_vec)
                gate = self.affect_gate(torch.cat([z, affect_latent], dim=-1))
                z = z + self.affect_w * gate * affect_latent
                confidences.append(gate.mean().item())
            else:
                confidences.append(0.0)

            # Update answer state
            answer_input = self.ln_answer(torch.cat([y, z], dim=-1))
            y_delta = self.answer(answer_input)
            y = y + 0.4 * y_delta

        return y, confidences
