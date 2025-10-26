"""
EmoBank Dataset Loader

Utilities for loading and processing the EmoBank emotion-annotated corpus.
"""

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class EmoBankDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for EmoBank corpus.

    EmoBank is an emotion-annotated corpus with VAD (Valence, Arousal, Dominance)
    labels for each text sample.

    Args:
        data_path (str): Path to the EmoBank CSV file
        split (str): 'train', 'test', or 'all'
        test_size (float): Fraction of data to use for testing (if splitting)
        random_state (int): Random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        split: str = "all",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.split = split

        # Load data
        self.df = pd.read_csv(data_path)

        # Split if needed
        if split in ["train", "test"]:
            train_df, test_df = train_test_split(
                self.df, test_size=test_size, random_state=random_state
            )
            self.df = train_df if split == "train" else test_df

        # Extract text and VAD labels
        self.texts = self.df["text"].values if "text" in self.df.columns else []
        self.vad_labels = self._extract_vad_labels()

    def _extract_vad_labels(self) -> Optional[torch.Tensor]:
        """Extract VAD (Valence, Arousal, Dominance) labels from dataframe."""
        vad_columns = ["V", "A", "D"]  # Common column names
        alt_columns = ["Valence", "Arousal", "Dominance"]

        # Try primary column names
        if all(col in self.df.columns for col in vad_columns):
            return torch.tensor(self.df[vad_columns].values, dtype=torch.float32)
        # Try alternative column names
        elif all(col in self.df.columns for col in alt_columns):
            return torch.tensor(self.df[alt_columns].values, dtype=torch.float32)
        else:
            print(
                f"Warning: VAD columns not found. Available columns: {self.df.columns.tolist()}"
            )
            return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple[str, torch.Tensor]:
                - Text string
                - VAD labels as tensor of shape [3]
        """
        text = self.texts[idx]
        vad = self.vad_labels[idx] if self.vad_labels is not None else torch.zeros(3)
        return text, vad


def download_emobank(save_dir: str = "./data", force: bool = False) -> str:
    """
    Download EmoBank dataset from GitHub.

    Args:
        save_dir (str): Directory to save the dataset
        force (bool): Force re-download even if file exists

    Returns:
        str: Path to the downloaded CSV file
    """
    import urllib.request

    os.makedirs(save_dir, exist_ok=True)

    # EmoBank CSV URL (adjust based on actual source)
    url = "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
    save_path = os.path.join(save_dir, "emobank.csv")

    if os.path.exists(save_path) and not force:
        print(f"EmoBank already exists at {save_path}")
        return save_path

    print(f"Downloading EmoBank from {url}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded to {save_path}")
    except Exception as e:
        print(f"Error downloading EmoBank: {e}")
        print("Please download manually from: https://github.com/JULIELab/EmoBank")

    return save_path


def load_emobank(
    data_path: Optional[str] = None,
    download: bool = True,
    split: str = "all",
    test_size: float = 0.2,
    random_state: int = 42,
) -> EmoBankDataset:
    """
    Load EmoBank dataset with automatic download if needed.

    Args:
        data_path (str, optional): Path to EmoBank CSV. If None, will download.
        download (bool): Automatically download if file not found
        split (str): 'train', 'test', or 'all'
        test_size (float): Fraction for test set
        random_state (int): Random seed

    Returns:
        EmoBankDataset: PyTorch Dataset object
    """
    if data_path is None:
        if download:
            data_path = download_emobank()
        else:
            raise ValueError(
                "data_path is None and download=False. Cannot load dataset."
            )

    if not os.path.exists(data_path):
        if download:
            data_path = download_emobank()
        else:
            raise FileNotFoundError(f"EmoBank file not found at {data_path}")

    return EmoBankDataset(
        data_path=data_path,
        split=split,
        test_size=test_size,
        random_state=random_state,
    )


def create_dataloaders(
    data_path: Optional[str] = None,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test DataLoaders for EmoBank.

    Args:
        data_path (str, optional): Path to EmoBank CSV
        batch_size (int): Batch size for DataLoaders
        test_size (float): Fraction for test set
        random_state (int): Random seed
        num_workers (int): Number of workers for data loading

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders
    """
    train_dataset = load_emobank(
        data_path=data_path, split="train", test_size=test_size, random_state=random_state
    )

    test_dataset = load_emobank(
        data_path=data_path, split="test", test_size=test_size, random_state=random_state
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
