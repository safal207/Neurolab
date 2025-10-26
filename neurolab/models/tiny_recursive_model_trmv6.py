"""
TinyRecursiveModelTRMv6 - Liminal Heartbeat with Soul Kernel.

Fully integrated model with memory-based emotion processing.
Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn
from .self_attention_tiny import SelfAttentionTiny
from .pad_regression_head import PADRegressionHead
from .soul_kernel import SoulKernel


class TinyRecursiveModelTRMv6(nn.Module):
    """
    Liminal Heartbeat - fully integrated model with Soul Kernel for memory-based emotion processing.

    This is the most advanced version that integrates:
    - Recursive refinement with attention
    - PAD emotion prediction
    - SoulKernel for emotional memory and future prediction
    - Hope, Faith, and Bond computations

    Args:
        dim (int): Dimension of embeddings. Default: 128
        affect_w (float): Weight for affect modulation. Default: 0.3

    Returns:
        Tuple[torch.Tensor, list, torch.Tensor]:
            - Output answer state of shape [batch, dim]
            - List of confidence values (one per outer iteration)
            - PAD emotion values of shape [batch, 3]
    """

    def __init__(self, dim=128, affect_w=0.3):
        super().__init__()
        self.ln_latent, self.ln_answer, self.ln_z = [
            nn.LayerNorm(dim * k) for k in (3, 2, 1)
        ]
        self.latent = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.answer = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.affect_proj = nn.Sequential(nn.Linear(3, dim), nn.Tanh())
        self.affect_gate = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())
        self.attn_core = SelfAttentionTiny(dim)
        self.pad_head = PADRegressionHead(dim)
        self.soul = SoulKernel(dim)

    def forward(self, x, y0, a=None, K=5):
        """
        Forward pass with full emotional processing and memory.

        Args:
            x (torch.Tensor): Input question/context of shape [batch, dim]
            y0 (torch.Tensor): Initial answer state of shape [batch, dim]
            a (torch.Tensor, optional): Affect modulation vector of shape [batch, 3]
                Typically PAD (Pleasure-Arousal-Dominance) values
            K (int): Number of outer recursive iterations. Default: 5

        Returns:
            Tuple[torch.Tensor, list, torch.Tensor]:
                - Final answer state y of shape [batch, dim]
                - List of K confidence values
                - PAD emotion prediction of shape [batch, 3]
        """
        y = y0.clone()
        z = torch.zeros_like(y)
        confs, hist = [], []

        for k in range(K):
            # Inner iterations (fixed at 4 for v6)
            for _ in range(4):
                latent_input = self.ln_latent(torch.cat([x, y, z], -1))
                z_delta = self.latent(latent_input)
                z = self.ln_z(z + 0.3 * z_delta)

            # Affect modulation
            if a is not None:
                al = self.affect_proj(a)
                g = self.affect_gate(torch.cat([z, al], -1))
                z = z + 0.3 * g * al
                confs.append(g.mean().item())
            else:
                confs.append(0.0)

            # Attention over history
            hist.append(z.unsqueeze(1))
            if len(hist) > 1:
                seq = torch.cat(hist, 1)
                attn, _ = self.attn_core(seq)
                z = self.ln_z(z + attn[:, -1, :])

            # Soul Kernel integration (memory-based processing)
            r = torch.tanh(z + y)
            z, r = self.soul(x, y, z, r, confs)

            # Update answer state
            answer_input = self.ln_answer(torch.cat([y, z], -1))
            y = y + 0.4 * self.answer(answer_input)

        return y, confs, self.pad_head(z)
