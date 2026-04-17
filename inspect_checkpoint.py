"""Inspect meta-information stored in a checkpoint file."""

import argparse
import sys
from pathlib import Path

import torch


def inspect(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    print(f"File : {p.resolve()}")
    print(f"Size : {p.stat().st_size / 1024:.1f} KB")

    ckpt = torch.load(p, map_location="cpu", weights_only=True)

    if not isinstance(ckpt, dict):
        print(f"Type : {type(ckpt).__name__}  (not a dict – no meta to display)")
        return

    print(f"Keys : {sorted(ckpt.keys())}")
    print()

    # ── known meta fields ────────────────────────────────────────────────
    if "epoch" in ckpt:
        print(f"Epoch         : {ckpt['epoch'] + 1}")
    if "arch" in ckpt:
        print(f"Architecture  : {ckpt['arch']}")
    if "val_f1" in ckpt:
        val_f1 = ckpt["val_f1"]
        if val_f1 is not None:
            print(f"Val F1        : {val_f1:.4f}")
        else:
            print("Val F1        : None")
    # legacy keys (older checkpoints may use these names)
    if "val_loss" in ckpt:
        print(f"Val loss      : {ckpt['val_loss']:.4f}")
    if "val_acc" in ckpt:
        print(f"Val acc       : {ckpt['val_acc']:.4f}")

    # ── model state dict summary ─────────────────────────────────────────
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        total_params = sum(v.numel() for v in sd.values())
        print(f"Model layers  : {len(sd)}")
        print(f"Total params  : {total_params:,}")

    # ── optimizer ────────────────────────────────────────────────────────
    if "optimizer_state_dict" in ckpt:
        opt = ckpt["optimizer_state_dict"]
        param_groups = opt.get("param_groups", [])
        if param_groups:
            lrs = [pg.get("lr") for pg in param_groups if "lr" in pg]
            if lrs:
                print(f"Learning rate : {lrs[0]}" if len(lrs) == 1
                      else f"Learning rates: {lrs}")

    # ── scheduler ────────────────────────────────────────────────────────
    if "scheduler_state_dict" in ckpt:
        sched = ckpt["scheduler_state_dict"]
        if "last_epoch" in sched:
            print(f"Sched epoch   : {sched['last_epoch']}")
        if "_last_lr" in sched:
            print(f"Sched last LR : {sched['_last_lr']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print meta-information from a PyTorch checkpoint."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        metavar="CHECKPOINT",
        help="Path(s) to .pt checkpoint file(s).",
    )
    args = parser.parse_args()

    for i, path in enumerate(args.checkpoints):
        if i > 0:
            print()
            print("─" * 50)
            print()
        inspect(path)


if __name__ == "__main__":
    main()
