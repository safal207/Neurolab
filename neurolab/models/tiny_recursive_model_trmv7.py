"""
TinyRecursiveModelTRMv7 - Quantum-Enhanced LIMINAL Heartbeat

Extends TRMv6 with quantum-inspired emotional operators:
- Hermitian operators for Hope, Faith, Love measurements
- Eigenvalue-based stability tracking
- Density matrix purity for emotional coherence

This version bridges quantum mechanics and emotion recognition,
inspired by the Lieberman brothers' quantum consciousness research.

Key additions:
- QuantumEmotionalMetrics for all measurements
- Eigenvalue evolution tracking
- Quantum entanglement measures
- Enhanced interpretability through spectral analysis
"""

import torch
import torch.nn as nn
from .self_attention_tiny import SelfAttentionTiny
from .pad_regression_head import PADRegressionHead
from .soul_kernel import SoulKernel
from .quantum import QuantumEmotionalMetrics


class TinyRecursiveModelTRMv7(nn.Module):
    """
    Quantum-Enhanced LIMINAL Heartbeat (v7)

    Combines all previous innovations with quantum operators:
    - Recursive refinement (K iterations)
    - Self-attention over history
    - PAD emotion prediction
    - SoulKernel memory
    - **NEW**: Quantum emotional metrics (Hope, Faith, Love)

    The quantum operators provide:
    1. Intrinsic measurements (no ground truth needed)
    2. Stability tracking (eigenvalue evolution)
    3. Emotional coherence (density matrix purity)
    4. Interpretability (spectral decomposition)

    Args:
        dim (int): Embedding dimension. Default: 128
        affect_w (float): Affect modulation weight. Default: 0.3
        use_quantum_metrics (bool): Enable quantum measurements. Default: True

    Example:
        >>> model = TinyRecursiveModelTRMv7(dim=128)
        >>> x = torch.randn(4, 128)  # Input embeddings
        >>> y0 = torch.zeros(4, 128)  # Initial state
        >>> a = torch.randn(4, 3)  # Affect vector
        >>>
        >>> # Forward pass
        >>> y, confs, pad, quantum_metrics = model(x, y0, a, K=5)
        >>>
        >>> print(quantum_metrics.keys())
        >>> # ['hope', 'faith', 'love', 'eigenvalues']
    """

    def __init__(self, dim: int = 128, affect_w: float = 0.3, use_quantum_metrics: bool = True):
        super().__init__()
        self.dim = dim
        self.affect_w = affect_w
        self.use_quantum_metrics = use_quantum_metrics

        # Core layers (from v6)
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

        # Advanced components
        self.attn_core = SelfAttentionTiny(dim)
        self.pad_head = PADRegressionHead(dim)
        self.soul = SoulKernel(dim)

        # **NEW**: Quantum emotional metrics
        if use_quantum_metrics:
            self.quantum_metrics = QuantumEmotionalMetrics(dim)

    def forward(
        self,
        x: torch.Tensor,
        y0: torch.Tensor,
        a: torch.Tensor = None,
        K: int = 5,
        return_quantum_metrics: bool = True
    ):
        """
        Forward pass with quantum-enhanced emotional processing.

        Args:
            x: Input embeddings [batch, dim]
            y0: Initial answer state [batch, dim]
            a: Affect modulation vector [batch, 3] (PAD values)
            K: Number of recursive iterations. Default: 5
            return_quantum_metrics: Return quantum measurements. Default: True

        Returns:
            If return_quantum_metrics=True:
                Tuple of (y, confs, pad, quantum_metrics)
            Else:
                Tuple of (y, confs, pad) like v6
        """
        y = y0.clone()
        z = torch.zeros_like(y)
        confs, hist = [], []
        states_sequence = []  # For faith measurement

        # Recursive refinement loop
        for k in range(K):
            # Inner iterations (latent refinement)
            for _ in range(4):
                latent_input = self.ln_latent(torch.cat([x, y, z], -1))
                z_delta = self.latent(latent_input)
                z = self.ln_z(z + 0.3 * z_delta)

            # Affect modulation
            if a is not None:
                al = self.affect_proj(a)
                g = self.affect_gate(torch.cat([z, al], -1))
                z = z + self.affect_w * g * al
                confs.append(g.mean().item())
            else:
                confs.append(0.0)

            # Attention over history
            hist.append(z.unsqueeze(1))
            if len(hist) > 1:
                seq = torch.cat(hist, 1)
                attn, _ = self.attn_core(seq)
                z = self.ln_z(z + attn[:, -1, :])

            # Soul Kernel (memory integration)
            r = torch.tanh(z + y)
            z, r = self.soul(x, y, z, r, confs)

            # Answer update
            answer_input = self.ln_answer(torch.cat([y, z], -1))
            y = y + 0.4 * self.answer(answer_input)

            # Store state for quantum measurements
            states_sequence.append(z.detach())

        # PAD prediction
        pad = self.pad_head(z)

        # **NEW**: Quantum emotional metrics
        if return_quantum_metrics and self.use_quantum_metrics:
            quantum_metrics = self._compute_quantum_metrics(
                z, states_sequence, pad, None, confs  # targets=None during inference
            )
            return y, confs, pad, quantum_metrics
        else:
            return y, confs, pad

    def _compute_quantum_metrics(
        self,
        final_state: torch.Tensor,
        states_sequence: list,
        pad_predictions: torch.Tensor,
        targets: torch.Tensor = None,
        confidences: list = None
    ) -> dict:
        """
        Compute all quantum emotional metrics.

        Args:
            final_state: Final latent state [batch, dim]
            states_sequence: List of K states from iterations
            pad_predictions: Predicted PAD values [batch, 3]
            targets: Ground truth PAD (if available)
            confidences: List of confidence values

        Returns:
            Dictionary with quantum measurements:
            - hope: Quantum hope measurement
            - faith: Eigenvalue stability
            - love: Density matrix purity (if targets available)
            - eigenvalues: Eigenvalue spectra for all operators
        """
        metrics = {}

        # Hope (intrinsic or with ground truth)
        if targets is not None:
            metrics['hope'] = self.quantum_metrics.measure_hope(
                final_state, targets, pad_predictions
            )
        else:
            metrics['hope'] = self.quantum_metrics.measure_hope(final_state)

        # Faith (stability across iterations)
        metrics['faith'] = self.quantum_metrics.measure_faith(states_sequence)

        # Love (harmony, if targets available)
        if targets is not None:
            metrics['love'] = self.quantum_metrics.measure_love(
                final_state, pad_predictions, targets
            )
        else:
            # Intrinsic love (purity only, no classical component)
            metrics['love'] = self.quantum_metrics.love_operator.compute_density_matrix_purity(
                final_state
            )

        # Eigenvalue spectra (for analysis)
        metrics['eigenvalues'] = self.quantum_metrics.get_eigenvalue_spectra()

        # Classical metrics for comparison
        if confidences:
            metrics['classical_faith'] = self.quantum_metrics.faith_operator.measure_confidence_sequence(
                confidences
            )

        return metrics

    def compute_quantum_loss(
        self,
        pad_predictions: torch.Tensor,
        targets: torch.Tensor,
        quantum_metrics: dict,
        alpha_hope: float = 0.1,
        alpha_faith: float = 0.1,
        alpha_love: float = 0.1
    ) -> torch.Tensor:
        """
        Compute loss with quantum metric regularization.

        Total loss = MSE + α_hope * (1-hope) + α_faith * (1-faith) + α_love * (1-love)

        This encourages the model to maximize Hope, Faith, and Love
        during training, implementing virtue-based learning.

        Args:
            pad_predictions: Predicted PAD [batch, 3]
            targets: Ground truth PAD [batch, 3]
            quantum_metrics: Dictionary of quantum measurements
            alpha_hope: Weight for hope regularization. Default: 0.1
            alpha_faith: Weight for faith regularization. Default: 0.1
            alpha_love: Weight for love regularization. Default: 0.1

        Returns:
            Total loss (scalar)
        """
        # Standard MSE loss
        mse_loss = nn.functional.mse_loss(pad_predictions, targets)

        # Quantum regularization terms
        # We subtract from 1 to turn maximization into minimization
        hope_loss = 1.0 - quantum_metrics['hope'].mean()
        faith_loss = 1.0 - quantum_metrics['faith']
        love_loss = 1.0 - quantum_metrics['love']

        # Combined loss
        total_loss = (
            mse_loss
            + alpha_hope * hope_loss
            + alpha_faith * faith_loss
            + alpha_love * love_loss
        )

        return total_loss

    def get_eigenvalue_evolution(self, num_samples: int = 100) -> dict:
        """
        Track eigenvalue evolution over multiple forward passes.

        Useful for analyzing stability and understanding quantum dynamics.

        Args:
            num_samples: Number of random samples to average over

        Returns:
            Dictionary with eigenvalue statistics
        """
        hope_eigs = []
        faith_eigs = []
        love_eigs = []

        for _ in range(num_samples):
            x = torch.randn(1, self.dim)
            y0 = torch.zeros(1, self.dim)

            _, _, _, quantum_metrics = self(x, y0, K=5)

            hope_eigs.append(quantum_metrics['eigenvalues']['hope_eigenvalues'])
            faith_eigs.append(quantum_metrics['eigenvalues']['faith_eigenvalues'])
            love_eigs.append(quantum_metrics['eigenvalues']['love_eigenvalues'])

        return {
            'hope_mean': torch.stack(hope_eigs).mean(dim=0),
            'hope_std': torch.stack(hope_eigs).std(dim=0),
            'faith_mean': torch.stack(faith_eigs).mean(dim=0),
            'faith_std': torch.stack(faith_eigs).std(dim=0),
            'love_mean': torch.stack(love_eigs).mean(dim=0),
            'love_std': torch.stack(love_eigs).std(dim=0),
        }
