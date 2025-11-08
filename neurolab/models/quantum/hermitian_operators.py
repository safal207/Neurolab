"""
Quantum-Inspired Emotional Operators

Implements Hermitian operators for measuring Hope, Faith, and Love
based on quantum mechanics principles.

Inspired by:
- Lieberman brothers' quantum consciousness research
- Quantum cognition models (Busemeyer & Bruza)
- Observable measurement theory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HermitianOperator(nn.Module):
    """
    Base class for quantum-inspired Hermitian operators.

    In quantum mechanics, observables are represented by Hermitian operators:
    - Self-adjoint: A = A† (conjugate transpose equals itself)
    - Real eigenvalues: All measurements are real numbers
    - Orthogonal eigenvectors: Form a complete basis

    This provides a mathematical framework for measuring emotional states
    in a way that guarantees:
    1. Real-valued measurements (not complex)
    2. Stability (eigenvalue properties)
    3. Interpretability (spectral decomposition)

    Args:
        dim (int): Dimension of the state space. Default: 128
        init_scale (float): Scale for weight initialization. Default: 0.1

    Example:
        >>> operator = HermitianOperator(dim=128)
        >>> state = torch.randn(4, 128)  # batch of 4 states
        >>> measurement = operator(state)
        >>> print(measurement.shape)  # [4] - one measurement per state
    """

    def __init__(self, dim: int = 128, init_scale: float = 0.1):
        super().__init__()
        self.dim = dim

        # Initialize weight matrix
        # We'll symmetrize it in forward pass to ensure Hermitian property
        self.weight = nn.Parameter(torch.randn(dim, dim) * init_scale)

        # Optional bias to shift eigenvalue spectrum
        self.bias = nn.Parameter(torch.zeros(1))

    def get_hermitian_matrix(self) -> torch.Tensor:
        """
        Construct Hermitian matrix from weights.

        Ensures A = A† by symmetrization: A = (W + W^T) / 2

        Returns:
            torch.Tensor: Hermitian matrix of shape [dim, dim]
        """
        # Symmetrize to ensure Hermitian property
        A = (self.weight + self.weight.T) / 2
        return A

    def compute_expectation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum expectation value: <ψ|A|ψ>

        This is the standard quantum measurement formula.
        For normalized states, this gives the expected measurement value.

        Args:
            state: State vector(s) of shape [batch, dim]

        Returns:
            Expectation values of shape [batch]
        """
        A = self.get_hermitian_matrix()

        # Compute <ψ|A|ψ> = ψ^T @ A @ ψ
        # Using einsum for efficient batched computation
        expectation = torch.einsum('bi,ij,bj->b', state, A, state)

        return expectation + self.bias

    def get_eigenvalues(self) -> torch.Tensor:
        """
        Compute eigenvalues of the Hermitian operator.

        Eigenvalues represent the possible measurement outcomes
        and their stability indicates the "faith" in measurements.

        Returns:
            torch.Tensor: Sorted eigenvalues in ascending order [dim]
        """
        A = self.get_hermitian_matrix()
        eigenvalues = torch.linalg.eigvalsh(A)  # Hermitian eigenvalue solver
        return eigenvalues

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply operator to state (quantum measurement).

        Args:
            state: Emotional state(s) [batch, dim]

        Returns:
            Measured values [batch]
        """
        return self.compute_expectation(state)


class HopeOperator(HermitianOperator):
    """
    Hope Operator: Measures alignment with truth and confidence in outcomes.

    Quantum interpretation:
    - Ground state (lowest eigenvalue) = perfect alignment
    - Excited states = degrees of misalignment
    - Hope eigenvalue = "energy level" of hope

    Hope combines:
    1. Intrinsic quantum measurement (operator expectation)
    2. Classical alignment (MSE with ground truth if available)

    Mathematical definition:
        Hope = σ(⟨ψ|H|ψ⟩)  (intrinsic)
        Hope = σ(0.5 * ⟨ψ|H|ψ⟩ + 0.5 * (1 - MSE)) (with ground truth)

    Where σ is sigmoid normalization to [0, 1]

    Args:
        dim (int): State dimension. Default: 128
        positive_bias (bool): Bias toward positive hope. Default: True

    Example:
        >>> hope_op = HopeOperator(dim=128)
        >>> state = torch.randn(4, 128)
        >>> hope = hope_op.measure(state)
        >>> print(hope)  # Values in [0, 1]
    """

    def __init__(self, dim: int = 128, positive_bias: bool = True):
        super().__init__(dim, init_scale=0.1)

        if positive_bias:
            # Bias eigenvalues toward positive values
            # (hope should generally be positive)
            self.bias = nn.Parameter(torch.ones(1) * 0.5)

    def measure(
        self,
        state: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Measure hope in current state.

        Combines quantum (intrinsic) and classical (alignment) measurements.

        Args:
            state: Current emotional state [batch, dim]
            target: Ground truth (if available) [batch, dim]
            predictions: Model predictions (if available) [batch, ...]

        Returns:
            Hope values [batch] in range [0, 1]
        """
        # Quantum intrinsic hope
        quantum_hope = self.forward(state)

        if target is not None and predictions is not None:
            # Classical hope: alignment with ground truth
            mse = F.mse_loss(predictions, target, reduction='none').mean(dim=-1)
            classical_hope = 1.0 - torch.clamp(mse, 0, 1)

            # Combine quantum and classical
            # Quantum provides intrinsic hope, classical provides grounding
            hope = 0.5 * quantum_hope + 0.5 * classical_hope
        else:
            hope = quantum_hope

        # Normalize to [0, 1] with sigmoid
        return torch.sigmoid(hope)


class FaithOperator(HermitianOperator):
    """
    Faith Operator: Measures stability and consistency of understanding.

    Quantum interpretation:
    - Eigenvalue stability = faith
    - Large eigenvalue spread = low faith (unstable, uncertain)
    - Concentrated eigenvalues = high faith (stable, consistent)

    Faith measures how stable the emotional understanding is across
    multiple iterations or time steps.

    Mathematical definition:
        Faith = exp(-Var(eigenvalues_over_time))

    High faith = eigenvalues don't fluctuate much
    Low faith = eigenvalues are unstable

    Args:
        dim (int): State dimension. Default: 128

    Example:
        >>> faith_op = FaithOperator(dim=128)
        >>> states = [torch.randn(4, 128) for _ in range(5)]  # 5 iterations
        >>> faith = faith_op.measure_sequence(states)
        >>> print(faith)  # Single value in [0, 1]
    """

    def __init__(self, dim: int = 128):
        super().__init__(dim, init_scale=0.1)

    def measure_sequence(
        self,
        states_sequence: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Measure faith across a sequence of states (e.g., K iterations).

        Computes stability of eigenvalues over time as measure of faith.

        Args:
            states_sequence: List of K state tensors, each [batch, dim]

        Returns:
            Faith value (scalar) in range [0, 1]
        """
        # Compute eigenvalues for each state in sequence
        eigenvalues_over_time = []

        for state in states_sequence:
            # Get current eigenvalues
            eigs = self.get_eigenvalues()
            eigenvalues_over_time.append(eigs)

        # Stack into [K, dim] tensor
        eig_tensor = torch.stack(eigenvalues_over_time)

        # Measure stability: variance of eigenvalues over time
        # Low variance = high faith (stable eigenvalues)
        eig_variance = torch.var(eig_tensor, dim=0).mean()

        # Faith = exp(-variance) for smooth [0, 1] range
        faith = torch.exp(-eig_variance)

        return faith

    def measure_confidence_sequence(
        self,
        confidences: list[float]
    ) -> torch.Tensor:
        """
        Classical faith measurement from confidence values.

        This is the original LIMINAL faith metric for comparison.

        Args:
            confidences: List of K confidence values from iterations

        Returns:
            Faith value (scalar) in range [0, 1]
        """
        if not confidences:
            return torch.tensor(0.0)

        mean_conf = sum(confidences) / len(confidences)
        faith = mean_conf ** 0.5  # Square root for smoothing

        return torch.tensor(faith)


class LoveOperator(HermitianOperator):
    """
    Love Operator: Measures harmony between predictions and truth.

    Quantum interpretation:
    - Entanglement measure between prediction and truth
    - High love = strong correlation (quantum entanglement)
    - Density matrix purity = love intensity

    Love requires BOTH:
    1. Low loss (accurate predictions)
    2. Low variance (consistent predictions)

    This prevents "cheating" via overfitting or lucky guesses.

    Mathematical definition:
        Classical: Love = exp(-loss) × (1 - variance)
        Quantum: Love_q = Tr(ρ²) / dim  (density matrix purity)
        Combined: Love = 0.5 * Love_classical + 0.5 * Love_quantum

    Args:
        dim (int): State dimension. Default: 128

    Example:
        >>> love_op = LoveOperator(dim=128)
        >>> state = torch.randn(4, 128)
        >>> predictions = torch.randn(4, 3)
        >>> targets = torch.randn(4, 3)
        >>> love = love_op.measure(state, predictions, targets)
        >>> print(love)  # Value in [0, 1]
    """

    def __init__(self, dim: int = 128):
        super().__init__(dim, init_scale=0.1)

    def compute_density_matrix_purity(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute purity of density matrix: Tr(ρ²)

        For pure states: Tr(ρ²) = 1
        For mixed states: Tr(ρ²) < 1

        High purity = "pure love" (single coherent emotion)
        Low purity = "mixed love" (conflicting emotions)

        Args:
            state: State vector [batch, dim]

        Returns:
            Purity values [batch]
        """
        # Average state across batch for stability
        mean_state = state.mean(dim=0)

        # Normalize state
        mean_state = mean_state / (mean_state.norm() + 1e-8)

        # Density matrix: ρ = |ψ⟩⟨ψ|
        density_matrix = torch.outer(mean_state, mean_state)

        # Purity: Tr(ρ²)
        purity = torch.trace(density_matrix @ density_matrix)

        # Normalize by dimension
        normalized_purity = purity / self.dim

        return normalized_purity

    def measure(
        self,
        state: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure love (harmony) in predictions.

        Combines classical (loss + variance) and quantum (purity) measures.

        Args:
            state: Emotional state [batch, dim]
            predictions: Model predictions [batch, output_dim]
            targets: Ground truth [batch, output_dim]

        Returns:
            Love value (scalar) in range [0, 1]
        """
        # Classical love component
        loss = F.mse_loss(predictions, targets)
        variance = torch.var(predictions)
        classical_love = torch.exp(-loss) * (1.0 - torch.clamp(variance, 0, 1))

        # Quantum love: density matrix purity
        quantum_love = self.compute_density_matrix_purity(state)

        # Combine both measures
        love = 0.5 * classical_love + 0.5 * quantum_love

        return love


class QuantumEmotionalMetrics:
    """
    Unified interface for all quantum emotional metrics.

    Provides easy access to Hope, Faith, Love measurements with
    both quantum and classical components.

    Example:
        >>> metrics = QuantumEmotionalMetrics(dim=128)
        >>>
        >>> # During training
        >>> hope = metrics.measure_hope(state, targets, predictions)
        >>> faith = metrics.measure_faith(states_sequence)
        >>> love = metrics.measure_love(state, predictions, targets)
        >>>
        >>> # Get all at once
        >>> all_metrics = metrics.measure_all(
        ...     state, states_sequence, predictions, targets
        ... )
    """

    def __init__(self, dim: int = 128):
        self.hope_operator = HopeOperator(dim)
        self.faith_operator = FaithOperator(dim)
        self.love_operator = LoveOperator(dim)

    def measure_hope(
        self,
        state: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Measure hope in current state."""
        return self.hope_operator.measure(state, target, predictions)

    def measure_faith(
        self,
        states_sequence: list[torch.Tensor]
    ) -> torch.Tensor:
        """Measure faith across sequence of states."""
        return self.faith_operator.measure_sequence(states_sequence)

    def measure_love(
        self,
        state: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Measure love (harmony) in predictions."""
        return self.love_operator.measure(state, predictions, targets)

    def measure_all(
        self,
        state: torch.Tensor,
        states_sequence: list[torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Measure all quantum emotional metrics at once.

        Returns:
            Dictionary with keys: 'hope', 'faith', 'love'
        """
        return {
            'hope': self.measure_hope(state, targets, predictions),
            'faith': self.measure_faith(states_sequence),
            'love': self.measure_love(state, predictions, targets)
        }

    def get_eigenvalue_spectra(self) -> dict[str, torch.Tensor]:
        """
        Get eigenvalue spectra for all operators.

        Useful for analysis and visualization.

        Returns:
            Dictionary with keys: 'hope_eigs', 'faith_eigs', 'love_eigs'
        """
        return {
            'hope_eigenvalues': self.hope_operator.get_eigenvalues(),
            'faith_eigenvalues': self.faith_operator.get_eigenvalues(),
            'love_eigenvalues': self.love_operator.get_eigenvalues()
        }
