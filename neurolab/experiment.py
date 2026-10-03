"""Train-only-fitted EmoBank experiment with baselines and reloadable inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .models import TinyRecursiveModelTRMv6

DATA_COMMIT = "248ce2a43e165a66d31aeaed83cff9641d6654e0"
DATA_BLOB = "e810731bef6967c14daefb84bd904c76628442d7"
DATA_URL = f"https://raw.githubusercontent.com/JULIELab/EmoBank/{DATA_COMMIT}/corpus/emobank.csv"
AXES = ("V", "A", "D")


@dataclass(frozen=True)
class LabConfig:
    seed: int = 42
    max_train: int = 3000
    epochs: int = 10
    batch_size: int = 128
    dim: int = 64
    vocabulary: int = 8000
    learning_rate: float = 0.001
    iterations: tuple[int, ...] = (1, 5)


def git_blob_hash(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def download_data(path="data/emobank.csv"):
    """Verify pinned bytes on both download and cache hits."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
        if git_blob_hash(raw) != DATA_BLOB:
            raise ValueError("Downloaded EmoBank differs from the pinned source")
        path.write_bytes(raw)
    if git_blob_hash(path.read_bytes()) != DATA_BLOB:
        raise ValueError("Cached EmoBank differs from the pinned source")
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def read_splits(path, config):
    # Literal text such as 'NA' is a sentence, not a missing-value marker.
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"id", "split", "text", *AXES}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any() or frame["id"].duplicated().any():
        raise ValueError("Dataset has missing fields or duplicate IDs")
    if not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every text must be a non-empty string")
    labels = frame[list(AXES)].to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all() or (labels < 1).any() or (labels > 5).any():
        raise ValueError("V/A/D labels must be finite and lie in [1, 5]")
    if set(frame["split"]) != {"train", "dev", "test"}:
        raise ValueError("Declare non-empty train/dev/test splits")
    splits = {name: frame.loc[frame["split"] == name].copy() for name in ("train", "dev", "test")}
    if config.max_train:
        splits["train"] = splits["train"].sample(
            n=min(config.max_train, len(splits["train"])), random_state=config.seed)
    seen = set(splits["train"]["text"].str.strip().str.casefold())
    excluded = {}
    for name in ("dev", "test"):
        keys = splits[name]["text"].str.strip().str.casefold()
        overlap = keys.isin(seen)
        excluded[name] = int(overlap.sum())
        seen.update(keys)
        splits[name] = splits[name].loc[~overlap].copy()
    if any(len(part) < 2 for part in splits.values()):
        raise ValueError("A split has fewer than two usable rows")
    quality = {
        "source_rows": len(frame), "official_counts": frame["split"].value_counts().to_dict(),
        "used_counts": {name: len(part) for name, part in splits.items()},
        "exact_text_overlap_excluded": excluded,
        "text_overlap_policy": "strip/casefold; dev excludes train; test excludes train and original dev",
        "split_id_hashes": {name: hashlib.sha256("\n".join(part["id"].astype(str)).encode()).hexdigest()
                            for name, part in splits.items()},
    }
    return splits, quality


def fit_features(splits, config):
    """Vocabulary, SVD and scaling see training text only."""
    features = make_pipeline(
        TfidfVectorizer(max_features=config.vocabulary, ngram_range=(1, 2), min_df=2),
        TruncatedSVD(n_components=config.dim, random_state=config.seed), StandardScaler())
    arrays = {"train": features.fit_transform(splits["train"]["text"]).astype(np.float32)}
    if arrays["train"].shape[1] != config.dim:
        raise ValueError("Training data is too small for the configured dimension")
    for name in ("dev", "test"):
        arrays[name] = features.transform(splits[name]["text"]).astype(np.float32)
    return features, arrays


def scores(target, prediction):
    """MAE is in original EmoBank 1–5 points, not classification accuracy."""
    error = np.abs(np.asarray(target) - np.clip(prediction, 1, 5))
    return {**{f"MAE_{axis}": float(error[:, i].mean()) for i, axis in enumerate(AXES)},
            "MAE_mean": float(error.mean())}


@torch.no_grad()
def neural_predict(model, values, k, batch_size=128):
    model.eval()
    chunks = []
    for start in range(0, len(values), batch_size):
        inputs = torch.from_numpy(values[start:start + batch_size])
        _, _, output = model(inputs, torch.zeros_like(inputs), K=k)
        chunks.append(output.numpy() * 2 + 3)
    return np.concatenate(chunks)


def train_neural(arrays, labels, config, k):
    seed_everything(config.seed)
    model = TinyRecursiveModelTRMv6(dim=config.dim, use_memory=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    inputs = torch.from_numpy(arrays["train"])
    targets = torch.from_numpy(((labels["train"] - 3) / 2).astype(np.float32))
    generator = torch.Generator().manual_seed(config.seed)
    best_mae, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start:start + config.batch_size]
            batch = inputs[rows]
            optimizer.zero_grad()
            _, _, prediction = model(batch, torch.zeros_like(batch), K=k)
            loss = nn.functional.mse_loss(prediction, targets[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(rows)
        dev = scores(labels["dev"], neural_predict(model, arrays["dev"], k))
        history.append({"model": f"Neurolab K={k}", "epoch": epoch,
                        "train_MSE_centered": total_loss / len(inputs), "dev_MAE": dev["MAE_mean"]})
        if dev["MAE_mean"] < best_mae:
            best_mae, best_epoch = dev["MAE_mean"], epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history, best_epoch


def paired_interval(target, candidate, baseline, seed=42, repetitions=1000):
    """Paired row-bootstrap; this does not cover variability across training seeds."""
    differences = (np.abs(target - candidate) - np.abs(target - baseline)).mean(axis=1)
    rng = np.random.default_rng(seed)
    estimates = np.array([differences[rng.integers(len(differences), size=len(differences))].mean()
                          for _ in range(repetitions)])
    return {"delta_MAE": float(differences.mean()),
            "row_bootstrap_95pct": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "resamples": repetitions, "negative_means": "candidate has lower error",
            "scope": "test rows at this trained seed, not uncertainty across training runs"}


def run_experiment(data_path, output_dir, config=LabConfig()):
    if config.epochs < 1 or config.dim < 8 or config.batch_size < 1 or config.max_train < 0:
        raise ValueError("Invalid experiment configuration")
    if not config.iterations or any(k < 1 for k in config.iterations):
        raise ValueError("Provide positive recursion depths")
    seed_everything(config.seed)
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, quality = read_splits(data_path, config)
    labels = {name: part[list(AXES)].to_numpy(dtype=np.float32) for name, part in splits.items()}
    features, arrays = fit_features(splits, config)
    joblib.dump(features, output / "features.joblib")
    rows, predictions, history, selection = [], {}, [], {}
    predictions["Train mean"] = np.repeat(labels["train"].mean(axis=0)[None, :], len(labels["test"]), axis=0)
    rows.append({"model": "Train mean", **scores(labels["test"], predictions["Train mean"])})
    candidates = []
    for alpha in (1.0, 10.0, 100.0):
        ridge = Ridge(alpha=alpha).fit(arrays["train"], labels["train"])
        candidates.append((scores(labels["dev"], ridge.predict(arrays["dev"]))["MAE_mean"], alpha, ridge))
    ridge_dev, ridge_alpha, ridge = min(candidates, key=lambda row: row[0])
    selection["Ridge"] = {"alpha": ridge_alpha, "dev_MAE": ridge_dev}
    predictions["Ridge"] = np.clip(ridge.predict(arrays["test"]), 1, 5)
    rows.append({"model": "Ridge", **scores(labels["test"], predictions["Ridge"])})
    joblib.dump(ridge, output / "ridge.joblib")
    for k in config.iterations:
        model, curve, epoch = train_neural(arrays, labels, config, k)
        name = f"Neurolab K={k}"
        selection[name] = {"epoch": epoch, "dev_MAE": min(item["dev_MAE"] for item in curve)}
        history.extend(curve)
        torch.save({"state_dict": model.state_dict(), "dim": config.dim, "K": k,
                    "trained": True, "seed": config.seed, "epoch": epoch,
                    "label_transform": "(raw - 3) / 2", "use_memory": False}, output / f"model_k{k}.pt")
        predictions[name] = np.clip(neural_predict(model, arrays["test"], k), 1, 5)
        rows.append({"model": name, **scores(labels["test"], predictions[name])})
    selected = min((f"Neurolab K={k}" for k in config.iterations), key=lambda name: selection[name]["dev_MAE"])
    raw = Path(data_path).read_bytes()
    root = Path(__file__).resolve().parents[1]
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "unavailable"
    report = {
        "config": asdict(config), "data": quality,
        "data_provenance": {"repository": "JULIELab/EmoBank", "commit": DATA_COMMIT,
                            "expected_blob": DATA_BLOB, "actual_blob": git_blob_hash(raw),
                            "matches_pinned_source": git_blob_hash(raw) == DATA_BLOB,
                            "sha256": hashlib.sha256(raw).hexdigest(), "license": "CC-BY-SA-4.0"},
        "code_commit": code_commit,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((root / "neurolab").rglob("*.py"))},
        "environment": {name: importlib.metadata.version(name) for name in
                        ("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib")},
        "selection_on_dev_only": selection, "selected_neural": selected,
        "selected_K": int(selected.split("=")[-1]), "test_metrics": rows,
        "paired_comparisons_to_ridge": {
            name: paired_interval(labels["test"], prediction, predictions["Ridge"], config.seed)
            for name, prediction in predictions.items() if name.startswith("Neurolab")},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "interpretation": "One English-text lexical-feature experiment; no consciousness, clinical, multilingual or SOTA claim.",
    }
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training.csv", index=False)
    np.savez_compressed(output / "test_predictions.npz", target=labels["test"], **predictions)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output / "ATTRIBUTION.txt").write_text(
        "EmoBank — Sven Buechel and Udo Hahn, JULIE Lab, 2017. CC-BY-SA 4.0.\n"
        "https://github.com/JULIELab/EmoBank\nhttps://aclanthology.org/E17-2092/\n"
        "Derived labels use (raw - 3) / 2; source corpus is not redistributed.\n")
    return report


def load_predictor(output_dir):
    """Read a bundle created by this lab. Joblib artifacts must be locally trusted."""
    output = Path(output_dir)
    report = json.loads((output / "report.json").read_text())
    k = report["selected_K"]
    checkpoint = torch.load(output / f"model_k{k}.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("trained") is not True or checkpoint.get("use_memory") is not False:
        raise ValueError("Not a trained independent-text lab checkpoint")
    model = TinyRecursiveModelTRMv6(dim=checkpoint["dim"], use_memory=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, joblib.load(output / "features.joblib"), k


def predict_texts(output_dir, texts):
    texts = list(texts)
    model, features, k = load_predictor(output_dir)
    values = features.transform(texts).astype(np.float32)
    return pd.DataFrame(np.clip(neural_predict(model, values, k), 1, 5), columns=AXES).assign(text=texts)[["text", *AXES]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", default="artifacts/neurolab-lab")
    parser.add_argument("--max-train", type=int, default=3000, help="0 uses all official training rows")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_experiment(args.data or download_data(), args.output_dir,
                            LabConfig(max_train=args.max_train, epochs=args.epochs, seed=args.seed))
    print(pd.DataFrame(report["test_metrics"]).round(4).to_string(index=False))
    print(f"Saved trained bundle: {args.output_dir}")


if __name__ == "__main__":
    main()
