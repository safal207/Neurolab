"""
Data loading and processing utilities for LIMINAL Heartbeat.
"""

from .emobank_loader import (
    EmoBankDataset,
    download_emobank,
    load_emobank,
    create_dataloaders,
)
from .embeddings import (
    EmbeddingGenerator,
    SentenceTransformerEmbedder,
    TransformerEmbedder,
    ProjectionLayer,
    create_embedder,
)

__all__ = [
    # EmoBank dataset
    "EmoBankDataset",
    "download_emobank",
    "load_emobank",
    "create_dataloaders",
    # Embeddings
    "EmbeddingGenerator",
    "SentenceTransformerEmbedder",
    "TransformerEmbedder",
    "ProjectionLayer",
    "create_embedder",
]
