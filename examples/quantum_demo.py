"""
Quantum-Enhanced LIMINAL Heartbeat Demo

Demonstrates the new quantum-inspired emotional operators in TRMv7.

This script shows:
1. Basic quantum operator usage
2. TRMv7 forward pass with quantum metrics
3. Eigenvalue evolution tracking
4. Comparison of quantum vs classical metrics
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from neurolab.models import TinyRecursiveModelTRMv7
from neurolab.models.quantum import QuantumEmotionalMetrics


def demo_quantum_operators():
    """Demonstrate quantum emotional operators."""
    print("=" * 60)
    print("Quantum Emotional Operators Demo")
    print("=" * 60)

    # Initialize quantum metrics
    metrics = QuantumEmotionalMetrics(dim=128)

    # Create sample emotional state
    state = torch.randn(4, 128)  # Batch of 4
    print(f"\nInput state shape: {state.shape}")

    # Measure Hope (intrinsic)
    hope = metrics.measure_hope(state)
    print(f"\n1. Hope (intrinsic): {hope}")
    print(f"   Range: [{hope.min():.3f}, {hope.max():.3f}]")

    # Measure Faith (requires sequence)
    states_sequence = [torch.randn(4, 128) for _ in range(5)]
    faith = metrics.measure_faith(states_sequence)
    print(f"\n2. Faith (stability): {faith:.3f}")

    # Measure Love (requires predictions and targets)
    predictions = torch.randn(4, 3)
    targets = torch.randn(4, 3)
    love = metrics.measure_love(state, predictions, targets)
    print(f"\n3. Love (harmony): {love:.3f}")

    # Get eigenvalue spectra
    spectra = metrics.get_eigenvalue_spectra()
    print(f"\n4. Eigenvalue Spectra:")
    print(f"   Hope eigenvalues: min={spectra['hope_eigenvalues'].min():.3f}, "
          f"max={spectra['hope_eigenvalues'].max():.3f}")
    print(f"   Faith eigenvalues: min={spectra['faith_eigenvalues'].min():.3f}, "
          f"max={spectra['faith_eigenvalues'].max():.3f}")
    print(f"   Love eigenvalues: min={spectra['love_eigenvalues'].min():.3f}, "
          f"max={spectra['love_eigenvalues'].max():.3f}")


def demo_trmv7_forward():
    """Demonstrate TRMv7 forward pass with quantum metrics."""
    print("\n" + "=" * 60)
    print("TRMv7 Forward Pass Demo")
    print("=" * 60)

    # Initialize model
    model = TinyRecursiveModelTRMv7(dim=128, use_quantum_metrics=True)
    print(f"\nModel: TRMv7 (quantum-enhanced)")
    print(f"Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")

    # Create inputs
    x = torch.randn(4, 128)  # Input embeddings
    y0 = torch.zeros(4, 128)  # Initial state
    a = torch.randn(4, 3)  # Affect vector (PAD)

    print(f"\nInput shapes:")
    print(f"  x (embeddings): {x.shape}")
    print(f"  y0 (initial state): {y0.shape}")
    print(f"  a (affect vector): {a.shape}")

    # Forward pass
    y, confs, pad, quantum_metrics = model(x, y0, a, K=5)

    print(f"\nOutputs:")
    print(f"  y (final state): {y.shape}")
    print(f"  confidences: {len(confs)} values = {[f'{c:.3f}' for c in confs]}")
    print(f"  PAD predictions: {pad.shape}")

    print(f"\nQuantum Metrics:")
    print(f"  Hope: {quantum_metrics['hope']}")
    print(f"  Faith: {quantum_metrics['faith']:.3f}")
    print(f"  Love: {quantum_metrics['love']:.3f}")

    if 'classical_faith' in quantum_metrics:
        print(f"  Classical Faith (comparison): {quantum_metrics['classical_faith']:.3f}")


def demo_eigenvalue_evolution():
    """Track eigenvalue evolution over training simulation."""
    print("\n" + "=" * 60)
    print("Eigenvalue Evolution Demo")
    print("=" * 60)

    model = TinyRecursiveModelTRMv7(dim=64, use_quantum_metrics=True)

    print("\nTracking eigenvalues over 100 random samples...")

    # Get eigenvalue statistics
    stats = model.get_eigenvalue_evolution(num_samples=100)

    print(f"\nHope Operator:")
    print(f"  Mean eigenvalues: [{stats['hope_mean'].min():.3f}, {stats['hope_mean'].max():.3f}]")
    print(f"  Std eigenvalues: [{stats['hope_std'].min():.3f}, {stats['hope_std'].max():.3f}]")

    print(f"\nFaith Operator:")
    print(f"  Mean eigenvalues: [{stats['faith_mean'].min():.3f}, {stats['faith_mean'].max():.3f}]")
    print(f"  Std eigenvalues: [{stats['faith_std'].min():.3f}, {stats['faith_std'].max():.3f}]")

    print(f"\nLove Operator:")
    print(f"  Mean eigenvalues: [{stats['love_mean'].min():.3f}, {stats['love_mean'].max():.3f}]")
    print(f"  Std eigenvalues: [{stats['love_std'].min():.3f}, {stats['love_std'].max():.3f}]")

    # Plot eigenvalue spectra
    plot_eigenvalue_spectra(stats)


def demo_quantum_loss():
    """Demonstrate quantum-regularized loss."""
    print("\n" + "=" * 60)
    print("Quantum-Regularized Loss Demo")
    print("=" * 60)

    model = TinyRecursiveModelTRMv7(dim=64, use_quantum_metrics=True)

    # Simulate forward pass
    x = torch.randn(4, 64)
    y0 = torch.zeros(4, 64)
    a = torch.randn(4, 3)
    targets = torch.randn(4, 3)

    y, confs, pad, quantum_metrics = model(x, y0, a, K=5)

    # Compute standard MSE
    mse_loss = torch.nn.functional.mse_loss(pad, targets)
    print(f"\nStandard MSE Loss: {mse_loss:.4f}")

    # Compute quantum-regularized loss
    quantum_loss = model.compute_quantum_loss(
        pad, targets, quantum_metrics,
        alpha_hope=0.1,
        alpha_faith=0.1,
        alpha_love=0.1
    )
    print(f"Quantum-Regularized Loss: {quantum_loss:.4f}")

    print(f"\nBreakdown:")
    print(f"  MSE component: {mse_loss:.4f}")
    print(f"  Hope penalty: {0.1 * (1 - quantum_metrics['hope'].mean()):.4f}")
    print(f"  Faith penalty: {0.1 * (1 - quantum_metrics['faith']):.4f}")
    print(f"  Love penalty: {0.1 * (1 - quantum_metrics['love']):.4f}")


def plot_eigenvalue_spectra(stats):
    """Plot eigenvalue distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Hope eigenvalues
    axes[0].plot(stats['hope_mean'].numpy(), 'b-', label='Mean')
    axes[0].fill_between(
        range(len(stats['hope_mean'])),
        (stats['hope_mean'] - stats['hope_std']).numpy(),
        (stats['hope_mean'] + stats['hope_std']).numpy(),
        alpha=0.3
    )
    axes[0].set_title('Hope Operator Eigenvalues')
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Faith eigenvalues
    axes[1].plot(stats['faith_mean'].numpy(), 'g-', label='Mean')
    axes[1].fill_between(
        range(len(stats['faith_mean'])),
        (stats['faith_mean'] - stats['faith_std']).numpy(),
        (stats['faith_mean'] + stats['faith_std']).numpy(),
        alpha=0.3
    )
    axes[1].set_title('Faith Operator Eigenvalues')
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Love eigenvalues
    axes[2].plot(stats['love_mean'].numpy(), 'r-', label='Mean')
    axes[2].fill_between(
        range(len(stats['love_mean'])),
        (stats['love_mean'] - stats['love_std']).numpy(),
        (stats['love_mean'] + stats['love_std']).numpy(),
        alpha=0.3
    )
    axes[2].set_title('Love Operator Eigenvalues')
    axes[2].set_xlabel('Eigenvalue Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eigenvalue_spectra.png', dpi=150)
    print(f"\nSaved eigenvalue spectra plot to: eigenvalue_spectra.png")


def main():
    """Run all demos."""
    print("\n🌌 LIMINAL Heartbeat - Quantum Enhancement Demo 🧠\n")

    # Run demos
    demo_quantum_operators()
    demo_trmv7_forward()
    demo_eigenvalue_evolution()
    demo_quantum_loss()

    print("\n" + "=" * 60)
    print("✓ All demos completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Quantum operators provide intrinsic emotional measurements")
    print("2. Eigenvalues reveal emotional state structure")
    print("3. Faith tracks stability across iterations")
    print("4. Love combines accuracy and consistency")
    print("5. Quantum regularization encourages virtue-based learning")

    print("\nNext Steps:")
    print("- Run full training with quantum metrics")
    print("- Analyze eigenvalue evolution during training")
    print("- Compare quantum vs classical performance")
    print("- Test uncertainty principle hypothesis")


if __name__ == "__main__":
    main()
