"""
LIMINAL Heartbeat Model Architectures

This module contains all model architectures for emotion recognition in text.
Models are organized in evolutionary order from v1 to v6.
"""

from .self_attention_tiny import SelfAttentionTiny
from .pad_regression_head import PADRegressionHead
from .rinse_head import RINSEHead
from .soul_kernel import SoulKernel
from .tiny_recursive_model_competitive import TinyRecursiveModelCompetitive
from .tiny_recursive_model_trmv2 import TinyRecursiveModelTRMv2
from .tiny_recursive_model_trmv3 import TinyRecursiveModelTRMv3
from .tiny_recursive_model_trmv4 import TinyRecursiveModelTRMv4
from .tiny_recursive_model_trmv6 import TinyRecursiveModelTRMv6

__all__ = [
    # Base components
    "SelfAttentionTiny",
    "PADRegressionHead",
    "RINSEHead",
    "SoulKernel",
    # Model versions
    "TinyRecursiveModelCompetitive",  # v1
    "TinyRecursiveModelTRMv2",  # v2
    "TinyRecursiveModelTRMv3",  # v3
    "TinyRecursiveModelTRMv4",  # v4
    "TinyRecursiveModelTRMv6",  # v6 (latest)
]
