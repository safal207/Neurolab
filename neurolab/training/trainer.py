"""
Training utilities for LIMINAL Heartbeat models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
import csv
from pathlib import Path
from tqdm import tqdm


class EmotionalMetrics:
    """
    Custom emotional metrics for LIMINAL models.

    Tracks Hope, Faith, and Love alongside standard ML metrics.
    """

    @staticmethod
    def compute_hope(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Hope = 1 - mean absolute error (normalized)

        High hope indicates low error, representing confidence in outcomes.
        """
        mae = torch.mean(torch.abs(predictions - targets))
        hope = 1.0 - torch.clamp(mae, 0, 1)
        return hope.item()

    @staticmethod
    def compute_faith(confidences: List[float]) -> float:
        """
        Faith = stabilized confidence (square root of mean)

        Measures stability and consistency of model confidence.
        """
        if not confidences:
            return 0.0
        mean_conf = sum(confidences) / len(confidences)
        faith = mean_conf ** 0.5
        return faith

    @staticmethod
    def compute_love(loss: float, variance: float) -> float:
        """
        Love = exp(-loss) * (1 - variance)

        Low loss with low variance represents alignment and harmony.
        """
        love = torch.exp(torch.tensor(-loss)) * (1.0 - torch.clamp(torch.tensor(variance), 0, 1))
        return love.item()


class LIMINALTrainer:
    """
    Trainer for LIMINAL Heartbeat models.

    Handles training loop, validation, metrics tracking, and checkpointing.

    Args:
        model (nn.Module): The LIMINAL model to train
        optimizer (optim.Optimizer): Optimizer for training
        scheduler (optim.lr_scheduler, optional): Learning rate scheduler
        device (str): Device to train on ('cuda' or 'cpu')
        log_dir (str): Directory to save logs and checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "./logs",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.metrics = EmotionalMetrics()

        # Initialize logging
        self.log_file = self.log_dir / "emotional_log.csv"
        self._init_log_file()

    def _init_log_file(self):
        """Initialize CSV log file for emotional metrics."""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_hope",
                    "val_hope",
                    "train_faith",
                    "val_faith",
                    "train_love",
                    "val_love",
                    "lr",
                ]
            )

    def train_epoch(
        self,
        train_loader: DataLoader,
        embedder: Callable,
        projection: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader
            embedder (Callable): Function to generate embeddings from text
            projection (nn.Module, optional): Projection layer for embeddings

        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_confidences = []

        pbar = tqdm(train_loader, desc="Training")
        for texts, vad_labels in pbar:
            # Generate embeddings
            with torch.no_grad():
                embeddings = embedder(texts)
                if projection is not None:
                    embeddings = projection(embeddings)
                embeddings = embeddings.to(self.device)

            vad_labels = vad_labels.to(self.device)

            # Forward pass
            y_init = torch.zeros_like(embeddings)
            _, confidences, pad_pred = self.model(embeddings, y_init, vad_labels)

            # Compute loss
            loss = self.criterion(pad_pred, vad_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_predictions.append(pad_pred.detach())
            all_targets.append(vad_labels.detach())
            all_confidences.extend(confidences)

            pbar.set_postfix({"loss": loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        hope = self.metrics.compute_hope(all_predictions, all_targets)
        faith = self.metrics.compute_faith(all_confidences)
        variance = torch.var(all_predictions).item()
        love = self.metrics.compute_love(avg_loss, variance)

        return {
            "loss": avg_loss,
            "hope": hope,
            "faith": faith,
            "love": love,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        embedder: Callable,
        projection: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader
            embedder (Callable): Function to generate embeddings from text
            projection (nn.Module, optional): Projection layer for embeddings

        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_confidences = []

        pbar = tqdm(val_loader, desc="Validation")
        for texts, vad_labels in pbar:
            # Generate embeddings
            embeddings = embedder(texts)
            if projection is not None:
                embeddings = projection(embeddings)
            embeddings = embeddings.to(self.device)

            vad_labels = vad_labels.to(self.device)

            # Forward pass
            y_init = torch.zeros_like(embeddings)
            _, confidences, pad_pred = self.model(embeddings, y_init, vad_labels)

            # Compute loss
            loss = self.criterion(pad_pred, vad_labels)

            # Track metrics
            total_loss += loss.item()
            all_predictions.append(pad_pred)
            all_targets.append(vad_labels)
            all_confidences.extend(confidences)

            pbar.set_postfix({"loss": loss.item()})

        # Compute validation metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        hope = self.metrics.compute_hope(all_predictions, all_targets)
        faith = self.metrics.compute_faith(all_confidences)
        variance = torch.var(all_predictions).item()
        love = self.metrics.compute_love(avg_loss, variance)

        return {
            "loss": avg_loss,
            "hope": hope,
            "faith": faith,
            "love": love,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        embedder: Callable,
        projection: Optional[nn.Module] = None,
        epochs: int = 50,
        save_best: bool = True,
    ):
        """
        Full training loop.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            embedder (Callable): Function to generate embeddings from text
            projection (nn.Module, optional): Projection layer for embeddings
            epochs (int): Number of epochs to train
            save_best (bool): Save best model based on validation loss
        """
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'=' * 60}")

            # Train
            train_metrics = self.train_epoch(train_loader, embedder, projection)

            # Validate
            val_metrics = self.validate(val_loader, embedder, projection)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            lr = self.optimizer.param_groups[0]["lr"]
            self._log_metrics(epoch, train_metrics, val_metrics, lr)

            # Print summary
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Hope: {train_metrics['hope']:.4f} | Val Hope: {val_metrics['hope']:.4f}")
            print(f"Train Faith: {train_metrics['faith']:.4f} | Val Faith: {val_metrics['faith']:.4f}")
            print(f"Train Love: {train_metrics['love']:.4f} | Val Love: {val_metrics['love']:.4f}")
            print(f"Learning Rate: {lr:.6f}")

            # Save best model
            if save_best and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, "best_model.pt")
                print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pt")

    def _log_metrics(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float
    ):
        """Log metrics to CSV file."""
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    train_metrics["hope"],
                    val_metrics["hope"],
                    train_metrics["faith"],
                    val_metrics["faith"],
                    train_metrics["love"],
                    val_metrics["love"],
                    lr,
                ]
            )

    def save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]
