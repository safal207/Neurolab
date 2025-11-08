"""
Quantum-Inspired Emotion Recognition Models

This module implements quantum mechanics-inspired approaches to emotion recognition,
inspired by the Lieberman brothers' quantum consciousness research.

Key components:
- HermitianOperator: Base class for quantum observables
- HopeOperator: Measures alignment with truth (quantum ground state)
- FaithOperator: Measures stability of understanding (eigenvalue consistency)
- LoveOperator: Measures harmony (density matrix purity + classical metrics)
- QuantumEmotionalMetrics: Unified interface for all quantum metrics

Example:
    >>> from neurolab.models.quantum import QuantumEmotionalMetrics
    >>>
    >>> metrics = QuantumEmotionalMetrics(dim=128)
    >>> hope = metrics.measure_hope(state)
    >>> faith = metrics.measure_faith([state1, state2, state3, state4, state5])
    >>> love = metrics.measure_love(state, predictions, targets)
"""

from .hermitian_operators import (
    HermitianOperator,
    HopeOperator,
    FaithOperator,
    LoveOperator,
    QuantumEmotionalMetrics,
)

__all__ = [
    "HermitianOperator",
    "HopeOperator",
    "FaithOperator",
    "LoveOperator",
    "QuantumEmotionalMetrics",
]

__version__ = "0.1.0"
