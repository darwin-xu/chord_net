#!/usr/bin/env python3
"""
transfer_to_ios.py – Convert a ChordNet checkpoint to CoreML for iOS.

Usage
-----
::

    # Default: export best.pt → iOS/ChordNet/ChordNetModel.mlpackage
    python transfer_to_ios.py

    # Explicit checkpoint
    python transfer_to_ios.py --checkpoint checkpoints/last.pt

    # Float16 (smaller, faster on Neural Engine)
    python transfer_to_ios.py --float16
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Project imports (model & config live in the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
from model import ChordNet


class _ChordNetExport(nn.Module):
    """Thin wrapper that appends sigmoid so the CoreML model outputs
    probabilities in [0, 1] rather than raw logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ChordNet checkpoint to CoreML (.mlpackage)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to a .pt checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="iOS/ChordNet/ChordNetModel.mlpackage",
        help="Output .mlpackage path",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Use Float16 compute precision (smaller model, faster on Neural Engine)",
    )
    args = parser.parse_args()

    # ── Check coremltools ───────────────────────────────────────────────
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools is not installed.")
        print("  pip install coremltools")
        sys.exit(1)

    # ── Load checkpoint ─────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = ChordNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    export_model = _ChordNetExport(model)
    export_model.eval()

    # ── Trace ───────────────────────────────────────────────────────────
    example = torch.randn(1, 1, CFG.n_mels, CFG.n_time_frames)
    traced = torch.jit.trace(export_model, example)

    # ── Convert to CoreML ───────────────────────────────────────────────
    precision = ct.precision.FLOAT16 if args.float16 else ct.precision.FLOAT32
    print(f"Converting to CoreML (precision={precision}) …")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=(1, 1, CFG.n_mels, CFG.n_time_frames),
            )
        ],
        convert_to="mlprogram",
        compute_precision=precision,
    )

    # ── Save ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    print(f"Saved → {out_path}")
    print(f"  Input : mel_spectrogram  shape (1, 1, {CFG.n_mels}, {CFG.n_time_frames})")
    print(f"  Output: probabilities    shape (1, {CFG.n_notes})  [sigmoid applied]")
    if ckpt.get("val_f1"):
        print(f"  Source checkpoint F1 = {ckpt['val_f1']:.4f}")


if __name__ == "__main__":
    main()
