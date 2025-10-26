"""
SelfAttentionTiny - Lightweight self-attention mechanism for sequence processing.

Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionTiny(nn.Module):
    """
    A lightweight self-attention mechanism for sequence processing.

    Args:
        dim (int): Dimension of the input embeddings. Default: 128

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Attention output (same shape as input)
            - Attention weights
    """

    def __init__(self, dim=128):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = [
            nn.Linear(dim, dim) for _ in range(4)
        ]
        self.scale = dim ** -0.5

    def forward(self, x):
        """
        Forward pass through self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape [batch, seq_len, dim]
                - Attention weights of shape [batch, seq_len, seq_len]
        """
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        return self.out_proj(torch.bmm(attn, V)), attn
