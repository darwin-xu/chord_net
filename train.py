"""
train.py – Training and validation loop for ChordNet.

Usage
-----
::

    python train.py                         # uses defaults from config.py
    python train.py --epochs 100 --lr 5e-4  # override on the CLI
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from dataset import ChordDataset
from model import ChordNet
from prepare_maestro import MaestroDataset


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if CFG.device:
        return torch.device(CFG.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, and F1 for multi-label predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs, shape ``(B, 88)``.
    targets : torch.Tensor
        Ground-truth binary labels, shape ``(B, 88)``.
    threshold : float
        Sigmoid threshold for converting logits to binary predictions.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


# ────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one training epoch.

    Returns
    -------
    dict
        ``loss``, ``precision``, ``recall``, ``f1`` averaged over
        the epoch.
    """
    model.train()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * patches.size(0)
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_targets))
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run one validation pass.

    Returns
    -------
    dict
        ``loss``, ``precision``, ``recall``, ``f1`` averaged over
        the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        running_loss += loss.item() * patches.size(0)
        all_logits.append(logits)
        all_targets.append(labels)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_targets))
    metrics["loss"] = epoch_loss
    return metrics


# ────────────────────────────────────────────────────────────────────────
# Main training driver
# ────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Complete training run: data → model → optimiser → loop → save."""

    device = get_device()
    print(f"[ChordNet] Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    if args.maestro:
        # Use preprocessed MAESTRO .npz / .npy files.
        train_ds = MaestroDataset(args.data_dir, split="train")
        val_ds = MaestroDataset(args.data_dir, split="val")
    else:
        # Use raw WAV + .npy label files.
        train_ds = ChordDataset(
            audio_dir=Path(args.data_dir) / "train" / "audio",
            label_dir=Path(args.data_dir) / "train" / "labels",
        )
        val_ds = ChordDataset(
            audio_dir=Path(args.data_dir) / "val" / "audio",
            label_dir=Path(args.data_dir) / "val" / "labels",
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[ChordNet] Train patches : {len(train_ds)}")
    print(f"[ChordNet] Val   patches : {len(val_ds)}")

    # ── Model / loss / optimiser ────────────────────────────────────────
    model = ChordNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning-rate scheduler: reduce on plateau of validation loss.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )

    # ── Checkpoint directory ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)

        scheduler.step(val_m["loss"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d}  "
            f"train_loss={train_m['loss']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  "
            f"val_P={val_m['precision']:.3f}  "
            f"val_R={val_m['recall']:.3f}  "
            f"val_F1={val_m['f1']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # Save the best model (by validation F1).
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                },
                best_path,
            )
            print(f"  ↳ Saved best model (F1={best_val_f1:.4f}) → {best_path}")

    # Always save the final checkpoint.
    final_path = ckpt_dir / "last.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"[ChordNet] Training complete. Final checkpoint → {final_path}")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChordNet")
    p.add_argument("--data-dir",       type=str,   default=str(CFG.data_dir))
    p.add_argument("--checkpoint-dir", type=str,   default=str(CFG.checkpoint_dir))
    p.add_argument("--epochs",         type=int,   default=CFG.epochs)
    p.add_argument("--batch-size",     type=int,   default=CFG.batch_size)
    p.add_argument("--lr",             type=float, default=CFG.learning_rate)
    p.add_argument("--weight-decay",   type=float, default=CFG.weight_decay)
    p.add_argument("--num-workers",    type=int,   default=CFG.num_workers)
    p.add_argument("--maestro",        action="store_true",
                   help="Use preprocessed MAESTRO data (MaestroDataset).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
