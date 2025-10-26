"""
SoulKernel - Memory-based kernel for emotional state processing.

Maintains emotional history and integrates past states into current processing.
Part of the LIMINAL Heartbeat emotion recognition system.
"""

import torch
import torch.nn as nn
from collections import deque


class SoulKernel(nn.Module):
    """
    A memory-based kernel that processes emotional states and maintains history.

    The SoulKernel implements emotional memory by:
    - Maintaining a deque of past emotional states
    - Computing "hope" from emotional alignment
    - Computing "faith" from confidence stability
    - Creating "bonds" between inputs and emotional states
    - Projecting future emotional trajectories

    Args:
        dim (int): Dimension of the latent representation. Default: 128
        mem (int): Maximum memory size (number of past states to remember). Default: 100

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated latent state and response
    """

    def __init__(self, dim=128, mem=100):
        super().__init__()
        self.mem = deque(maxlen=mem)

    def forward(self, x, y, z, r, c):
        """
        Forward pass with emotional memory integration.

        Args:
            x (torch.Tensor): Input representation [batch, dim]
            y (torch.Tensor): Current answer state [batch, dim]
            z (torch.Tensor): Latent state [batch, dim]
            r (torch.Tensor): Response/reflection state [batch, dim]
            c (list): List of confidence values from iterations

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated latent state z
                - Updated response state r
        """
        # Compute "hope" as alignment with target (clamped to valid range)
        hope = torch.clamp(torch.mean(r.detach()), -1, 1)

        # Compute "faith" as stabilized confidence (square root for smoothing)
        faith = (sum(c) / len(c)) ** 0.5 if c else 0.0

        # Strengthen latent state based on faith
        z = z + faith * 0.05 * z

        # Predict future state from recent memory
        if self.mem:
            future = torch.mean(torch.stack(list(self.mem)[-3:]), 0).to(z.device)
        else:
            future = torch.zeros_like(z)

        # Compute "bond" between inputs and emotional states
        bond = torch.tanh(torch.mean(x * z).detach() + torch.mean(y * r).detach())

        # Move current state toward predicted future, weighted by bond strength
        z = z + 0.2 * bond * (future.detach() - z)

        # Update response with future prediction
        r = (r + future.detach() * 0.3).tanh()

        # Store current response in memory for future predictions
        self.mem.append(r[0].detach().clone())

        return z, r
