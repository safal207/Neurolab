"""
PADRegressionHead - Outputs Pleasure-Arousal-Dominance (PAD) emotion values.

Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn


class PADRegressionHead(nn.Module):
    """
    A head that outputs Pleasure-Arousal-Dominance (PAD) emotion values.

    The PAD model represents emotions in a 3-dimensional space:
    - Pleasure: Positive vs Negative emotional valence
    - Arousal: Intensity of emotion (calm vs excited)
    - Dominance: Sense of control (submissive vs dominant)

    Args:
        dim (int): Dimension of the input latent representation. Default: 128

    Returns:
        torch.Tensor: PAD values of shape [batch, 3] in range [-1, 1]
    """

    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Forward pass to predict PAD values.

        Args:
            z (torch.Tensor): Latent representation of shape [batch, dim]

        Returns:
            torch.Tensor: PAD emotion values of shape [batch, 3]
                - [:, 0]: Pleasure dimension
                - [:, 1]: Arousal dimension
                - [:, 2]: Dominance dimension
        """
        return self.fc(z)
