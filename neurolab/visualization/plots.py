"""
Visualization utilities for LIMINAL Heartbeat.

Includes emotion field mapping, breathing animations, and training visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import torch


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training and validation curves from log file.

    Args:
        log_file (str): Path to emotional_log.csv
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    df = pd.read_csv(log_file)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LIMINAL Heartbeat Training Progress", fontsize=16, fontweight="bold")

    # Loss curves
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (MSE)")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Hope curves
    axes[0, 1].plot(df["epoch"], df["train_hope"], label="Train Hope", linewidth=2)
    axes[0, 1].plot(df["epoch"], df["val_hope"], label="Val Hope", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Hope")
    axes[0, 1].set_title("Hope: Emotional Confidence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Faith curves
    axes[1, 0].plot(df["epoch"], df["train_faith"], label="Train Faith", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["val_faith"], label="Val Faith", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Faith")
    axes[1, 0].set_title("Faith: Stability & Consistency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Love curves
    axes[1, 1].plot(df["epoch"], df["train_love"], label="Train Love", linewidth=2)
    axes[1, 1].plot(df["epoch"], df["val_love"], label="Val Love", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Love")
    axes[1, 1].set_title("Love: Alignment & Harmony")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_emotional_field(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot PAD emotion predictions in 3D space.

    Args:
        predictions (torch.Tensor): Predicted PAD values [N, 3]
        targets (torch.Tensor): Target PAD values [N, 3]
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    # 3D scatter plot for predictions
    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        pred_np[:, 0],
        pred_np[:, 1],
        pred_np[:, 2],
        c=np.linalg.norm(pred_np, axis=1),
        cmap="viridis",
        alpha=0.6,
    )
    ax1.set_xlabel("Pleasure")
    ax1.set_ylabel("Arousal")
    ax1.set_zlabel("Dominance")
    ax1.set_title("Predicted Emotions (PAD Space)")
    plt.colorbar(scatter1, ax=ax1, label="Magnitude")

    # 3D scatter plot for targets
    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter(
        target_np[:, 0],
        target_np[:, 1],
        target_np[:, 2],
        c=np.linalg.norm(target_np, axis=1),
        cmap="plasma",
        alpha=0.6,
    )
    ax2.set_xlabel("Pleasure")
    ax2.set_ylabel("Arousal")
    ax2.set_zlabel("Dominance")
    ax2.set_title("Target Emotions (PAD Space)")
    plt.colorbar(scatter2, ax=ax2, label="Magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved emotional field plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot attention weights as heatmap.

    Args:
        attention_weights (torch.Tensor): Attention weights [batch, seq_len, seq_len]
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    # Average over batch
    attn_avg = attention_weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_avg,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Attention Weight"},
    )
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Attention Weights Heatmap")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved attention heatmap to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confidence_evolution(
    confidences_list: List[List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot evolution of confidence across iterations.

    Args:
        confidences_list (List[List[float]]): List of confidence values per sample
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot each sample's confidence evolution
    for i, confs in enumerate(confidences_list[:20]):  # Plot first 20 samples
        iterations = list(range(1, len(confs) + 1))
        plt.plot(iterations, confs, alpha=0.3, color="blue")

    # Plot mean confidence
    max_len = max(len(c) for c in confidences_list)
    mean_confs = []
    for i in range(max_len):
        values = [c[i] for c in confidences_list if i < len(c)]
        mean_confs.append(np.mean(values))

    iterations = list(range(1, max_len + 1))
    plt.plot(iterations, mean_confs, linewidth=3, color="red", label="Mean Confidence")

    plt.xlabel("Iteration")
    plt.ylabel("Confidence")
    plt.title("Confidence Evolution Across Iterations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confidence evolution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_pad_distribution(
    pad_values: torch.Tensor,
    dimension_names: List[str] = ["Pleasure", "Arousal", "Dominance"],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot distribution of PAD dimensions.

    Args:
        pad_values (torch.Tensor): PAD values [N, 3]
        dimension_names (List[str]): Names for the 3 dimensions
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
    """
    pad_np = pad_values.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, (name, color) in enumerate(zip(dimension_names, colors)):
        axes[i].hist(pad_np[:, i], bins=50, color=color, alpha=0.7, edgecolor="black")
        axes[i].axvline(pad_np[:, i].mean(), color="red", linestyle="--", linewidth=2)
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{name} Distribution")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved PAD distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_breathing_animation(
    log_file: str,
    save_path: Optional[str] = None,
    interval: int = 200,
):
    """
    Create animated "breathing" visualization of training progress.

    Args:
        log_file (str): Path to emotional_log.csv
        save_path (str, optional): Path to save the animation (as GIF)
        interval (int): Delay between frames in milliseconds
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    df = pd.read_csv(log_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(frame):
        ax.clear()

        # Get data up to current frame
        current_df = df.iloc[: frame + 1]

        # Plot with breathing effect (pulse size based on loss)
        loss = current_df["train_loss"].values
        normalized_loss = (loss - loss.min()) / (loss.max() - loss.min() + 1e-8)
        sizes = 100 * (1 - normalized_loss)  # Larger when loss is smaller

        ax.scatter(
            current_df["epoch"],
            current_df["train_loss"],
            s=sizes,
            alpha=0.6,
            color="blue",
            label="Train Loss",
        )
        ax.plot(current_df["epoch"], current_df["train_loss"], alpha=0.3, color="blue")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"LIMINAL Heartbeat - Epoch {frame + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(df) + 1)
        ax.set_ylim(df["train_loss"].min() * 0.9, df["train_loss"].max() * 1.1)

    anim = FuncAnimation(fig, animate, frames=len(df), interval=interval, repeat=True)

    if save_path:
        writer = PillowWriter(fps=5)
        anim.save(save_path, writer=writer)
        print(f"Saved breathing animation to {save_path}")

    plt.show()
