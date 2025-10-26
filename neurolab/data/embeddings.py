"""
Embedding Generation Utilities

Tools for generating embeddings from text using various models.
"""

import torch
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Base class for generating embeddings from text.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings

        Returns:
            torch.Tensor: Embeddings of shape [batch, dim]
        """
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbeddingGenerator):
    """
    Generate embeddings using Sentence-Transformers models.

    Args:
        model_name (str): Name of the sentence-transformers model
        device (str, optional): Device to use ('cuda' or 'cpu')
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        super().__init__(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Generate sentence embeddings.

        Args:
            texts (List[str]): List of text strings

        Returns:
            torch.Tensor: Embeddings of shape [batch, embedding_dim]
        """
        embeddings = self.model.encode(
            texts, convert_to_tensor=True, device=self.device
        )
        return embeddings


class TransformerEmbedder(EmbeddingGenerator):
    """
    Generate embeddings using HuggingFace Transformers models (e.g., GPT-2, BERT).

    Args:
        model_name (str): Name of the HuggingFace model
        device (str, optional): Device to use ('cuda' or 'cpu')
        pooling (str): Pooling strategy ('mean', 'max', 'cls')
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        pooling: str = "mean",
    ):
        super().__init__(device)
        self.model_name = model_name
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Set pad token for GPT-2 and similar models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings using transformer models.

        Args:
            texts (List[str]): List of text strings

        Returns:
            torch.Tensor: Embeddings of shape [batch, hidden_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Apply pooling
        if self.pooling == "mean":
            # Mean pooling (excluding padding)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling == "max":
            # Max pooling
            embeddings = hidden_states.max(dim=1)[0]
        elif self.pooling == "cls":
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return embeddings


class ProjectionLayer(torch.nn.Module):
    """
    Linear projection layer to project embeddings to target dimension.

    Args:
        input_dim (int): Input embedding dimension
        output_dim (int): Target embedding dimension (default: 128 for LIMINAL)
    """

    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to target dimension.

        Args:
            x (torch.Tensor): Input embeddings of shape [batch, input_dim]

        Returns:
            torch.Tensor: Projected embeddings of shape [batch, output_dim]
        """
        return self.projection(x)


def create_embedder(
    embedder_type: str = "sentence-transformer",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.

    Args:
        embedder_type (str): Type of embedder ('sentence-transformer', 'transformer')
        model_name (str, optional): Name of the model to use
        device (str, optional): Device to use ('cuda' or 'cpu')
        **kwargs: Additional arguments for the embedder

    Returns:
        EmbeddingGenerator: Configured embedding generator
    """
    if embedder_type == "sentence-transformer":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    elif embedder_type == "transformer":
        model_name = model_name or "gpt2"
        return TransformerEmbedder(model_name=model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            "Choose 'sentence-transformer' or 'transformer'"
        )
