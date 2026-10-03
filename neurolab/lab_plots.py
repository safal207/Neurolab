"""Portable figures for the executed experiment; no external plotting services."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(output_dir):
    output = Path(output_dir)
    report = json.loads((output / "report.json").read_text())
    metrics = pd.DataFrame(report["test_metrics"])
    history = pd.read_csv(output / "training.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#273244", "text.color": "#273244"})
    palette = {"Train mean": "#a7afb9", "Ridge": "#58667a",
               "Neurolab K=1": "#2c6fbb", "Neurolab K=5": "#b78925"}
    figures = []

    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    bars = ax.bar(metrics["model"], metrics["MAE_mean"],
                  color=[palette.get(name, "#2c6fbb") for name in metrics["model"]], width=0.58)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in metrics["MAE_mean"]], padding=6)
    ax.set(ylim=(0, metrics["MAE_mean"].max() * 1.22),
           ylabel="Mean absolute error · V/A/D points (1–5)",
           title=f"Held-out text prediction · {report['data']['used_counts']['test']} test rows")
    ax.yaxis.grid(True, color="#e4e8ed")
    ax.set_axisbelow(True)
    fig.savefig(output / "test_mae.png", dpi=160)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.3), layout="constrained")
    for name, group in history.groupby("model", sort=False):
        style = "-" if name.endswith("=1") else "--"
        ax.plot(group["epoch"], group["dev_MAE"], style, marker="o", markersize=4,
                label=name, color=palette.get(name, "#2c6fbb"))
        chosen = report["selection_on_dev_only"][name]["epoch"]
        row = group.loc[group["epoch"] == chosen].iloc[0]
        ax.plot(chosen, row["dev_MAE"], "o", markersize=9, markerfacecolor="white",
                markeredgewidth=2, color=palette.get(name, "#2c6fbb"))
    ax.set(xlabel="Training epoch · open markers are selected checkpoints",
           ylabel="Development MAE · V/A/D points (1–5)",
           title="Checkpoint selection uses development data")
    ax.set_xticks(sorted(history["epoch"].unique()))
    ax.grid(True, color="#e4e8ed")
    ax.legend(frameon=False)
    fig.savefig(output / "development_mae.png", dpi=160)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.7), layout="constrained")
    comparisons = report["paired_comparisons_to_ridge"]
    for i, (name, row) in enumerate(comparisons.items()):
        lower, upper = row["row_bootstrap_95pct"]
        delta = row["delta_MAE"]
        ax.errorbar(delta, i, xerr=[[max(0, delta - lower)], [max(0, upper - delta)]],
                    fmt="o", capsize=5, color=palette.get(name, "#2c6fbb"), markersize=7)
    ax.axvline(0, color="#273244", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(comparisons)), list(comparisons))
    ax.set(xlabel="MAE difference from Ridge · negative favours Neurolab",
           title="Paired 95% row-bootstrap intervals · one trained seed")
    ax.xaxis.grid(True, color="#e4e8ed")
    fig.savefig(output / "paired_difference.png", dpi=160)
    figures.append(fig)
    return figures
