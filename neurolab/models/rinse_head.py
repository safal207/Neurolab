"""
RINSEHead - Reflective Integrative Neural Self-Evolver.

Provides introspection on attention and affect states.
Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn


class RINSEHead(nn.Module):
    """
    Reflective Integrative Neural Self-Evolver - introspection on attention and affect states.

    RINSE analyzes the model's internal state to provide meta-cognitive insights
    about the emotional processing happening within the network.

    Args:
        dim (int): Dimension of the latent representation. Default: 128

    Returns:
        torch.Tensor: Introspection state of shape [dim // 4]
    """

    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim + 4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.Tanh()
        )

    def forward(self, z, pad, conf_mean):
        """
        Forward pass for reflective introspection.

        Args:
            z (torch.Tensor): Latent state of shape [batch, dim]
            pad (torch.Tensor): PAD emotion values of shape [batch, 3]
            conf_mean (float): Mean confidence across iterations

        Returns:
            torch.Tensor: Introspection state of shape [dim // 4]
        """
        reflect_input = torch.cat([z.mean(dim=0), pad.mean(dim=0), torch.tensor([conf_mean])], dim=0)
        return self.fc(reflect_input)
