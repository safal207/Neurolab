"""
Data loading and processing utilities for LIMINAL Heartbeat.
"""

from .emobank_loader import (
    EmoBankDataset,
    download_emobank,
    load_emobank,
    create_dataloaders,
)

def __getattr__(name):
    """Load optional transformer dependencies only when that backend is used."""
    if name in {
        "EmbeddingGenerator", "SentenceTransformerEmbedder", "TransformerEmbedder",
        "ProjectionLayer", "create_embedder",
    }:
        from . import embeddings
        return getattr(embeddings, name)
    raise AttributeError(name)

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
