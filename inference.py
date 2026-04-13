"""
inference.py – Run ChordNet inference on audio files.

Usage
-----
::

    # Single file
    python inference.py --checkpoint checkpoints/best.pt --input song.wav

    # Whole directory
    python inference.py --checkpoint checkpoints/best.pt --input audio_dir/
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from config import CFG
from model import ChordNet
from preprocess import preprocess, extract_patches


# ────────────────────────────────────────────────────────────────────────
# Note name mapping (MIDI 21–108 → A0 … C8)
# ────────────────────────────────────────────────────────────────────────

NOTE_NAMES = []
_SEMITONE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

for midi in range(21, 109):
    octave = (midi // 12) - 1
    name = _SEMITONE[midi % 12]
    NOTE_NAMES.append(f"{name}{octave}")


# ────────────────────────────────────────────────────────────────────────
# Core inference
# ────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> ChordNet:
    """Load a trained ChordNet from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pt`` checkpoint saved by ``train.py``.
    device : torch.device
        Target device.

    Returns
    -------
    ChordNet
        Model in eval mode with loaded weights.
    """
    model = ChordNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_file(
    model: ChordNet,
    audio_path: str,
    device: torch.device,
    threshold: float = CFG.inference_threshold,
) -> List[Dict]:
    """Run inference on a single audio file.

    Parameters
    ----------
    model : ChordNet
        Trained model in eval mode.
    audio_path : str
        Path to the audio file.
    device : torch.device
        Compute device.
    threshold : float
        Sigmoid probability threshold to consider a note active.

    Returns
    -------
    list of dict
        One dict per time-patch with keys:
        - ``patch_idx``   (int)   – zero-based patch index
        - ``start_sec``   (float) – patch start time in seconds
        - ``end_sec``     (float) – patch end time in seconds
        - ``notes``       (list[str]) – active note names
        - ``midi``        (list[int]) – active MIDI numbers
        - ``probs``       (list[float]) – sigmoid probabilities
    """
    # Preprocess & slice into patches.
    log_mel = preprocess(audio_path)
    patches = extract_patches(log_mel)  # (N, 1, F, T)
    patches_tensor = torch.from_numpy(patches).to(device)

    # Forward pass (may be batched if many patches).
    logits = model(patches_tensor)            # (N, 88)
    probs = torch.sigmoid(logits).cpu().numpy()  # (N, 88)

    # Seconds per patch.
    patch_duration = CFG.n_time_frames * CFG.hop_length / CFG.target_sr

    results = []
    for i, row in enumerate(probs):
        active_indices = np.where(row >= threshold)[0]
        notes = [NOTE_NAMES[j] for j in active_indices]
        midi_nums = [int(j) + 21 for j in active_indices]
        note_probs = [float(row[j]) for j in active_indices]

        results.append(
            {
                "patch_idx": i,
                "start_sec": round(i * patch_duration, 4),
                "end_sec": round((i + 1) * patch_duration, 4),
                "notes": notes,
                "midi": midi_nums,
                "probs": note_probs,
            }
        )

    return results


def pretty_print(results: List[Dict], file_name: str) -> None:
    """Print detection results in a readable table."""
    print(f"\n{'=' * 60}")
    print(f"  {file_name}")
    print(f"{'=' * 60}")
    for r in results:
        if r["notes"]:
            note_str = ", ".join(
                f"{n} ({p:.2f})" for n, p in zip(r["notes"], r["probs"])
            )
        else:
            note_str = "—"
        print(
            f"  [{r['start_sec']:7.3f}s – {r['end_sec']:7.3f}s]  {note_str}"
        )
    print()


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChordNet inference")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pt model checkpoint.",
    )
    p.add_argument(
        "--input", type=str, required=True,
        help="Path to an audio file or a directory of audio files.",
    )
    p.add_argument(
        "--threshold", type=float, default=CFG.inference_threshold,
        help=f"Sigmoid threshold (default: {CFG.inference_threshold}).",
    )
    p.add_argument(
        "--device", type=str, default="",
        help="Force device (cuda / mps / cpu). Auto-detect if omitted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device.
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[ChordNet] Using device: {device}")

    # Load model.
    model = load_model(args.checkpoint, device)

    # Gather audio files.
    input_path = Path(args.input)
    if input_path.is_dir():
        audio_files = sorted(input_path.glob("*.wav"))
        if not audio_files:
            print(f"No .wav files found in {input_path}")
            return
    else:
        audio_files = [input_path]

    # Run inference on each file.
    for audio_file in audio_files:
        results = predict_file(model, str(audio_file), device, args.threshold)
        pretty_print(results, audio_file.name)


if __name__ == "__main__":
    main()
