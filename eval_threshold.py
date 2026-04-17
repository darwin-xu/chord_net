"""
eval_threshold.py — Find the optimal sigmoid threshold for a saved checkpoint.

Evaluates the model on the validation set at multiple thresholds and prints
the precision/recall/F1 for each, so you can pick the best one for inference.

Usage
-----
::

    python eval_threshold.py --checkpoint checkpoints/best_resnet.pt --arch resnet
    python eval_threshold.py --checkpoint checkpoints/last_resnet.pt --arch resnet
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from model import ChordNet
from model_resnet import ChordResNet
from prepare_maestro import MaestroDataset

ARCH_REGISTRY = {"chordnet": ChordNet, "resnet": ChordResNet}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a full validation pass, return raw logits and targets as numpy arrays."""
    model.eval()
    all_logits = []
    all_targets = []

    for patches, labels in tqdm(loader, desc="  Collecting logits", unit="batch"):
        patches = patches.to(device)
        logits = model(patches)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def sweep_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(np.float32)
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        })
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep sigmoid thresholds for a checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--arch",       default="resnet", choices=list(ARCH_REGISTRY))
    p.add_argument("--data-dir",   default=str(CFG.data_dir))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        help="List of thresholds to evaluate (default: 0.3 to 0.8 in steps of 0.05)",
    )
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = ARCH_REGISTRY[args.arch]().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    saved_f1 = ckpt.get("val_f1", "?")
    print(f"Loaded: {args.checkpoint}  (epoch={epoch}, saved_F1={saved_f1})")

    # Validation data
    val_ds = MaestroDataset(args.data_dir, split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Val patches: {len(val_ds)}")

    # Collect logits once, sweep thresholds in numpy (fast)
    logits, targets = collect_logits(model, val_loader, device)

    results = sweep_thresholds(logits, targets, args.thresholds)

    best = max(results, key=lambda r: r["f1"])

    print()
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 44)
    for r in results:
        marker = " ◄ best" if r["threshold"] == best["threshold"] else ""
        print(
            f"{r['threshold']:>10.2f}  "
            f"{r['precision']:>10.3f}  "
            f"{r['recall']:>8.3f}  "
            f"{r['f1']:>8.3f}"
            f"{marker}"
        )
    print()
    print(
        f"Best threshold: {best['threshold']:.2f}  →  "
        f"P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}"
    )
    print()
    print(f"Use this threshold in iOS AudioEngine.swift and inference.py:")
    print(f"  threshold = {best['threshold']:.2f}")


if __name__ == "__main__":
    main()
