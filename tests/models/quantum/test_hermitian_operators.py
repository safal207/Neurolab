"""
Unit tests for quantum-inspired Hermitian operators.

Tests verify:
1. Hermitian property (A = A†)
2. Real eigenvalues
3. Proper measurement behavior
4. Normalization to [0, 1]
5. Integration with training
"""

import pytest
import torch
import torch.nn as nn
from neurolab.models.quantum import (
    HermitianOperator,
    HopeOperator,
    FaithOperator,
    LoveOperator,
    QuantumEmotionalMetrics,
)


class TestHermitianOperator:
    """Tests for base HermitianOperator class."""

    def test_initialization(self):
        """Test operator initializes correctly."""
        op = HermitianOperator(dim=128)
        assert op.dim == 128
        assert op.weight.shape == (128, 128)
        assert op.bias.shape == (1,)

    def test_hermitian_property(self):
        """Test that operator matrix is Hermitian (A = A†)."""
        op = HermitianOperator(dim=64)
        A = op.get_hermitian_matrix()

        # Check A = A^T (Hermitian property for real matrices)
        assert torch.allclose(A, A.T, atol=1e-6)

    def test_real_eigenvalues(self):
        """Test that eigenvalues are real."""
        op = HermitianOperator(dim=32)
        eigenvalues = op.get_eigenvalues()

        # Should be real (no imaginary part)
        assert eigenvalues.dtype == torch.float32 or eigenvalues.dtype == torch.float64
        assert eigenvalues.shape == (32,)

    def test_forward_pass(self):
        """Test forward pass produces correct shape."""
        op = HermitianOperator(dim=128)
        state = torch.randn(4, 128)  # batch of 4

        measurement = op(state)

        assert measurement.shape == (4,)  # One measurement per batch item

    def test_expectation_value_properties(self):
        """Test quantum expectation value properties."""
        op = HermitianOperator(dim=64)

        # Normalized state
        state = torch.randn(1, 64)
        state = state / state.norm()

        measurement = op(state)

        # Should be finite
        assert torch.isfinite(measurement).all()


class TestHopeOperator:
    """Tests for HopeOperator."""

    def test_initialization(self):
        """Test Hope operator initializes with positive bias."""
        hope_op = HopeOperator(dim=128, positive_bias=True)

        assert hope_op.dim == 128
        assert hope_op.bias > 0  # Should be positive

    def test_measure_intrinsic_hope(self):
        """Test measuring intrinsic hope (no ground truth)."""
        hope_op = HopeOperator(dim=64)
        state = torch.randn(4, 64)

        hope = hope_op.measure(state)

        # Should be in [0, 1] due to sigmoid
        assert hope.shape == (4,)
        assert (hope >= 0).all() and (hope <= 1).all()

    def test_measure_with_ground_truth(self):
        """Test measuring hope with ground truth."""
        hope_op = HopeOperator(dim=64)
        state = torch.randn(4, 64)
        predictions = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        hope = hope_op.measure(state, targets, predictions)

        assert hope.shape == (4,)
        assert (hope >= 0).all() and (hope <= 1).all()

    def test_perfect_alignment_gives_high_hope(self):
        """Test that perfect predictions give high hope."""
        hope_op = HopeOperator(dim=64)
        state = torch.randn(4, 64)
        predictions = torch.randn(4, 3)
        targets = predictions.clone()  # Perfect match

        hope = hope_op.measure(state, targets, predictions)

        # Should be close to 1.0 (high hope)
        assert (hope > 0.5).all()  # At least above 0.5


class TestFaithOperator:
    """Tests for FaithOperator."""

    def test_initialization(self):
        """Test Faith operator initializes correctly."""
        faith_op = FaithOperator(dim=128)
        assert faith_op.dim == 128

    def test_measure_sequence_faith(self):
        """Test measuring faith across sequence."""
        faith_op = FaithOperator(dim=64)

        # Create sequence of 5 states (K=5 iterations)
        states = [torch.randn(4, 64) for _ in range(5)]

        faith = faith_op.measure_sequence(states)

        # Should be scalar in [0, 1]
        assert faith.shape == ()
        assert faith >= 0 and faith <= 1

    def test_stable_sequence_gives_high_faith(self):
        """Test that stable eigenvalues give high faith."""
        faith_op = FaithOperator(dim=32)

        # Create very similar states (stable)
        base_state = torch.randn(4, 32)
        states = [base_state + torch.randn(4, 32) * 0.01 for _ in range(5)]

        faith = faith_op.measure_sequence(states)

        # Should be high (close to 1.0)
        assert faith > 0.5

    def test_classical_faith_from_confidences(self):
        """Test classical faith measurement."""
        faith_op = FaithOperator(dim=64)

        confidences = [0.8, 0.82, 0.85, 0.83, 0.84]  # Stable confidences
        faith = faith_op.measure_confidence_sequence(confidences)

        assert faith > 0.5  # Should be reasonably high


class TestLoveOperator:
    """Tests for LoveOperator."""

    def test_initialization(self):
        """Test Love operator initializes correctly."""
        love_op = LoveOperator(dim=128)
        assert love_op.dim == 128

    def test_measure_love(self):
        """Test measuring love."""
        love_op = LoveOperator(dim=64)
        state = torch.randn(4, 64)
        predictions = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        love = love_op.measure(state, predictions, targets)

        # Should be scalar in [0, 1]
        assert love.shape == ()
        assert love >= 0 and love <= 1

    def test_density_matrix_purity(self):
        """Test density matrix purity computation."""
        love_op = LoveOperator(dim=32)

        # Pure state (single coherent emotion)
        pure_state = torch.randn(1, 32)
        pure_state = pure_state / pure_state.norm()

        purity = love_op.compute_density_matrix_purity(pure_state)

        # Purity should be high for pure state
        assert purity > 0.5
        assert purity <= 1.0

    def test_perfect_predictions_give_high_love(self):
        """Test that perfect predictions with low variance give high love."""
        love_op = LoveOperator(dim=64)
        state = torch.randn(4, 64)

        # Perfect predictions
        targets = torch.randn(4, 3)
        predictions = targets.clone()

        love = love_op.measure(state, predictions, targets)

        # Should be high (loss=0, low variance)
        assert love > 0.5


class TestQuantumEmotionalMetrics:
    """Tests for unified QuantumEmotionalMetrics interface."""

    def test_initialization(self):
        """Test metrics interface initializes all operators."""
        metrics = QuantumEmotionalMetrics(dim=128)

        assert isinstance(metrics.hope_operator, HopeOperator)
        assert isinstance(metrics.faith_operator, FaithOperator)
        assert isinstance(metrics.love_operator, LoveOperator)

    def test_measure_all(self):
        """Test measuring all metrics at once."""
        metrics = QuantumEmotionalMetrics(dim=64)

        state = torch.randn(4, 64)
        states_sequence = [torch.randn(4, 64) for _ in range(5)]
        predictions = torch.randn(4, 3)
        targets = torch.randn(4, 3)

        all_metrics = metrics.measure_all(
            state, states_sequence, predictions, targets
        )

        # Should have all three metrics
        assert 'hope' in all_metrics
        assert 'faith' in all_metrics
        assert 'love' in all_metrics

        # All should be in valid range
        assert (all_metrics['hope'] >= 0).all() and (all_metrics['hope'] <= 1).all()
        assert all_metrics['faith'] >= 0 and all_metrics['faith'] <= 1
        assert all_metrics['love'] >= 0 and all_metrics['love'] <= 1

    def test_get_eigenvalue_spectra(self):
        """Test retrieving eigenvalue spectra."""
        metrics = QuantumEmotionalMetrics(dim=32)

        spectra = metrics.get_eigenvalue_spectra()

        assert 'hope_eigenvalues' in spectra
        assert 'faith_eigenvalues' in spectra
        assert 'love_eigenvalues' in spectra

        # Each should have dim eigenvalues
        assert spectra['hope_eigenvalues'].shape == (32,)
        assert spectra['faith_eigenvalues'].shape == (32,)
        assert spectra['love_eigenvalues'].shape == (32,)


class TestIntegrationWithTraining:
    """Integration tests with training scenarios."""

    def test_gradient_flow(self):
        """Test that gradients flow through operators."""
        hope_op = HopeOperator(dim=32)
        state = torch.randn(4, 32, requires_grad=True)

        hope = hope_op.measure(state).mean()
        hope.backward()

        # Gradients should exist
        assert state.grad is not None
        assert torch.isfinite(state.grad).all()

    def test_optimizer_update(self):
        """Test that operators can be optimized."""
        hope_op = HopeOperator(dim=32)
        optimizer = torch.optim.Adam(hope_op.parameters(), lr=0.01)

        initial_weight = hope_op.weight.clone()

        # Simulate training step
        state = torch.randn(4, 32)
        hope = hope_op.measure(state).mean()
        loss = -hope  # Maximize hope

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(hope_op.weight, initial_weight)

    def test_hermitian_preserved_after_update(self):
        """Test that Hermitian property is preserved after optimization."""
        hope_op = HopeOperator(dim=32)
        optimizer = torch.optim.Adam(hope_op.parameters(), lr=0.01)

        for _ in range(10):
            state = torch.randn(4, 32)
            hope = hope_op.measure(state).mean()
            loss = -hope

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should still be Hermitian
        A = hope_op.get_hermitian_matrix()
        assert torch.allclose(A, A.T, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
