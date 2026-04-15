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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import ChordDataset
from model import ChordNet
from model_resnet import ChordResNet
from prepare_maestro import MaestroDataset


# ────────────────────────────────────────────────────────────────────────
# Architecture registry
# ────────────────────────────────────────────────────────────────────────

ARCH_REGISTRY: dict[str, type] = {
    "chordnet": ChordNet,
    "resnet": ChordResNet,
}


def build_model(arch: str) -> nn.Module:
    """Instantiate a model by architecture name."""
    cls = ARCH_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {', '.join(ARCH_REGISTRY)}"
        )
    return cls()


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
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = patches.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        # Running metric accumulators.
        preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
        tp_sum += (preds * labels).sum().item()
        fp_sum += (preds * (1 - labels)).sum().item()
        fn_sum += ((1 - preds) * labels).sum().item()

        # Update progress bar.
        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    epoch_loss = running_loss / n_samples
    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"loss": epoch_loss, "precision": precision, "recall": recall, "f1": f1}


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
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        bs = patches.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp_sum += (preds * labels).sum().item()
        fp_sum += (preds * (1 - labels)).sum().item()
        fn_sum += ((1 - preds) * labels).sum().item()

        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    epoch_loss = running_loss / n_samples
    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"loss": epoch_loss, "precision": precision, "recall": recall, "f1": f1}


def compute_pos_weight(
    args: argparse.Namespace,
    train_ds,
    max_weight: float,
) -> torch.Tensor:
    """Estimate per-note positive class weights for BCEWithLogitsLoss.

    For MAESTRO this reads labels directly from labels_all.npy (fast, vectorized).
    For generic datasets it falls back to sampling labels from the dataset.
    """
    eps = 1e-6

    if args.maestro:
        labels_path = Path(args.data_dir) / "train" / "labels_all.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing label file for pos_weight: {labels_path}. "
                "Run prepare_maestro.py first."
            )
        labels = np.load(labels_path, mmap_mode="r")
        pos_frac = labels.mean(axis=0).astype(np.float64)
    else:
        # Keep fallback bounded so startup does not become too slow.
        n = min(len(train_ds), 100_000)
        idx = np.linspace(0, len(train_ds) - 1, num=n, dtype=np.int64)
        pos_sum = None
        for i in idx:
            _, y = train_ds[int(i)]
            y_np = y.numpy().astype(np.float64)
            if pos_sum is None:
                pos_sum = np.zeros_like(y_np)
            pos_sum += y_np
        pos_frac = pos_sum / float(n)

    pos_frac = np.clip(pos_frac, eps, 1.0 - eps)
    weights = (1.0 - pos_frac) / pos_frac
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


# ────────────────────────────────────────────────────────────────────────
# Main training driver
# ────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Complete training run: data → model → optimiser → loop → save."""

    device = get_device()
    print(f"[ChordNet] Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    if args.maestro:
        # Use preprocessed MAESTRO memory-mapped arrays.
        train_ds = MaestroDataset(args.data_dir, split="train", augment=True)
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

    # pin_memory accelerates host→GPU copies on CUDA but is not
    # supported on MPS — disable it there to silence the warning.
    pin = device.type == "cuda"

    # With memory-mapped data, num_workers=0 is fastest — the main
    # process does a simple numpy array index (0.03 ms/sample) and
    # avoids fork+mmap issues on macOS.  Multi-worker loading only
    # helps if preprocessing is CPU-heavy (not our case).
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    print(f"[ChordNet] Train patches : {len(train_ds)}")
    print(f"[ChordNet] Val   patches : {len(val_ds)}")

    # ── Model / loss / optimiser ────────────────────────────────────────
    arch = args.arch
    model = build_model(arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ChordNet] Architecture  : {arch} ({n_params:,} params)")
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(args, train_ds, args.max_pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(
            "[ChordNet] Using pos_weight in BCEWithLogitsLoss "
            f"(min={pos_weight.min().item():.2f}, "
            f"mean={pos_weight.mean().item():.2f}, max={pos_weight.max().item():.2f})"
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning-rate scheduler: reduce on plateau of validation loss.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Checkpoint directory ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    start_epoch = 1

    # Architecture-specific checkpoint filenames for easy identification.
    best_ckpt_name = f"best_{arch}.pt"
    last_ckpt_name = f"last_{arch}.pt"

    # ── Resume from checkpoint ──────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        saved_arch = ckpt.get("arch", "chordnet")
        if saved_arch != arch:
            raise ValueError(
                f"Checkpoint was saved with arch='{saved_arch}' but "
                f"--arch='{arch}' was requested."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("val_f1")
        if best_val_f1 is None:
            best_path = ckpt_dir / best_ckpt_name
            if best_path.exists():
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
                best_val_f1 = float(best_ckpt.get("val_f1", 0.0))
            else:
                best_val_f1 = 0.0
        print(f"[ChordNet] Resumed from {resume_path} (epoch {start_epoch - 1}, best F1={best_val_f1:.4f})")

    # Early-stopping bookkeeping.
    epochs_without_improvement = 0

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        should_validate = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_m = None
        if should_validate:
            val_m = validate(model, val_loader, criterion, device)
            scheduler.step(val_m["loss"])

        elapsed = time.time() - t0
        if val_m is None:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val=skipped  "
                f"({elapsed:.1f}s)"
            )
        else:
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
                epochs_without_improvement = 0
                best_path = ckpt_dir / best_ckpt_name
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": arch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_f1": best_val_f1,
                    },
                    best_path,
                )
                print(f"  ↳ Saved best model (F1={best_val_f1:.4f}) → {best_path}")
            else:
                epochs_without_improvement += 1

        # Save rolling checkpoint every epoch for safe interruption/resume.
        rolling_path = ckpt_dir / last_ckpt_name
        torch.save(
            {
                "epoch": epoch,
                "arch": arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1": best_val_f1,
            },
            rolling_path,
        )

        # ── Early stopping ──────────────────────────────────────────────
        if args.early_stop > 0 and epochs_without_improvement >= args.early_stop:
            print(
                f"[ChordNet] Early stopping: no F1 improvement for "
                f"{args.early_stop} validated epochs."
            )
            break

    print(f"[ChordNet] Training complete. Final checkpoint → {rolling_path}")


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
    p.add_argument("--val-every",      type=int,   default=1,
                   help="Run validation every N epochs (always validates final epoch).")
    p.add_argument("--use-pos-weight", action="store_true",
                   help="Use per-note positive class weighting in BCEWithLogitsLoss.")
    p.add_argument("--max-pos-weight", type=float, default=20.0,
                   help="Upper bound for per-note pos_weight to avoid instability.")
    p.add_argument("--arch",           type=str,   default="chordnet",
                   choices=list(ARCH_REGISTRY),
                   help="Model architecture: chordnet (shallow CNN) or resnet.")
    p.add_argument("--maestro",        action="store_true",
                   help="Use preprocessed MAESTRO data (MaestroDataset).")
    p.add_argument("--resume",         type=str,   default=None,
                   help="Path to checkpoint (.pt) to resume training from.")
    p.add_argument("--early-stop",     type=int,   default=0,
                   help="Stop after N validated epochs without F1 improvement (0=disabled).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
