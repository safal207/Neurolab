"""
TinyRecursiveModelTRMv3 - Heartbeat-TRM v3 with PAD head output.

Extends v2 with explicit PAD (Pleasure-Arousal-Dominance) emotion prediction.
Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn
from .self_attention_tiny import SelfAttentionTiny
from .pad_regression_head import PADRegressionHead


class TinyRecursiveModelTRMv3(nn.Module):
    """
    Heartbeat-TRM v3 - adds PAD head output.

    Improvements over v2:
    - Explicit PAD (Pleasure-Arousal-Dominance) emotion prediction
    - Regression head for 3D emotion space output

    Args:
        dim (int): Dimension of embeddings. Default: 128
        inner (int): Number of inner iterations per outer iteration. Default: 4
        affect_w (float): Weight for affect modulation. Default: 0.3

    Returns:
        Tuple[torch.Tensor, list, torch.Tensor]:
            - Output answer state of shape [batch, dim]
            - List of confidence values (one per outer iteration)
            - PAD emotion values of shape [batch, 3]
    """

    def __init__(self, dim=128, inner=4, affect_w=0.3):
        super().__init__()
        self.dim = dim
        self.inner = inner
        self.affect_w = affect_w

        self.ln_latent = nn.LayerNorm(dim * 3)
        self.ln_answer = nn.LayerNorm(dim * 2)
        self.ln_z = nn.LayerNorm(dim)

        self.latent = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.answer = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.affect_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.Tanh()
        )
        self.affect_gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )

        self.attn_core = SelfAttentionTiny(dim)
        self.pad_head = PADRegressionHead(dim)

    def forward(self, x, y_init, affect_vec=None, K=5):
        """
        Forward pass with PAD emotion prediction.

        Args:
            x (torch.Tensor): Input question/context of shape [batch, dim]
            y_init (torch.Tensor): Initial answer state of shape [batch, dim]
            affect_vec (torch.Tensor, optional): Affect modulation vector of shape [batch, 3]
            K (int): Number of outer recursive iterations. Default: 5

        Returns:
            Tuple[torch.Tensor, list, torch.Tensor]:
                - Final answer state y of shape [batch, dim]
                - List of K confidence values
                - PAD emotion prediction of shape [batch, 3]
        """
        y = y_init.clone()
        z = torch.zeros_like(y)
        confs = []
        history = []

        for k in range(K):
            # Inner iterations for latent state refinement
            for n in range(self.inner):
                latent_input = self.ln_latent(torch.cat([x, y, z], dim=-1))
                z_delta = self.latent(latent_input)
                z = self.ln_z(z + 0.3 * z_delta)

            # Affect modulation
            if affect_vec is not None:
                affect_latent = self.affect_proj(affect_vec)
                gate = self.affect_gate(torch.cat([z, affect_latent], dim=-1))
                z = z + self.affect_w * gate * affect_latent
                confs.append(gate.mean().item())
            else:
                confs.append(0.0)

            # Add current state to history and apply attention
            history.append(z.unsqueeze(1))
            if len(history) > 1:
                seq = torch.cat(history, dim=1)
                attn_out, _ = self.attn_core(seq)
                z = self.ln_z(z + attn_out[:, -1, :])

            # Update answer state
            answer_input = self.ln_answer(torch.cat([y, z], dim=-1))
            y = y + 0.4 * self.answer(answer_input)

        # Predict PAD emotion values
        pad = self.pad_head(z)
        return y, confs, pad
