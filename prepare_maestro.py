"""
prepare_maestro.py – Convert the MAESTRO v3 dataset into ChordNet training format.

The MAESTRO dataset provides paired WAV + MIDI files of piano performances.
This script:

  1. Reads the MAESTRO CSV metadata to get file paths and train/val/test splits.
  2. For each recording:
     a. Loads the WAV, resamples to mono 22.05 kHz.
     b. Applies DC removal and RMS normalisation.
     c. Computes the log-Mel spectrogram.
     d. Slices the spectrogram into fixed-length patches (229 × 32).
     e. Parses the MIDI file to build a binary (N_patches, 88) label matrix
        indicating which piano keys are active in each patch.
  3. Saves each file's patches and labels as .npz to the output directory.

Usage
-----
::

    python prepare_maestro.py \\
        --maestro-dir /Users/darwin/Desktop/maestro-v3.0.0 \\
        --output-dir  data \\
        --max-files 0          # 0 = process all files

The output layout matches what ``dataset.py`` expects::

    data/
    ├── train/
    │   ├── audio/       (symlinks or copies — NOT used at runtime)
    │   └── labels/
    │       ├── <stem>.npy
    │       └── …
    ├── val/
    │   ├── audio/
    │   └── labels/
    └── test/
        ├── audio/
        └── labels/

We also save the preprocessed spectrogram patches alongside the labels
in .npz files so that training can load them directly without re-computing
spectrograms every epoch.  A companion ``MaestroDataset`` class is provided
at the bottom of this file for convenience.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Ensure project modules are importable ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
from preprocess import load_audio, remove_dc, rms_normalize, compute_log_mel, extract_patches

# We use mido for MIDI parsing (pip install mido).
import mido


# ────────────────────────────────────────────────────────────────────────
# MIDI → label matrix
# ────────────────────────────────────────────────────────────────────────

# Piano MIDI range: note 21 (A0) to note 108 (C8) — 88 keys.
MIDI_LOW = 21
MIDI_HIGH = 108


def midi_to_note_intervals(midi_path: str) -> List[Tuple[int, float, float]]:
    """Parse a MIDI file and extract (note, onset_sec, offset_sec) tuples.

    Parameters
    ----------
    midi_path : str
        Path to a ``.midi`` file.

    Returns
    -------
    list of (int, float, float)
        Each tuple is ``(midi_note, onset_seconds, offset_seconds)``.
    """
    mid = mido.MidiFile(midi_path)
    intervals = []

    # Track active notes: note_number → onset_time.
    active: Dict[int, float] = {}
    current_time = 0.0

    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_time
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            onset = active.pop(msg.note, None)
            if onset is not None:
                intervals.append((msg.note, onset, current_time))

    # Close any notes that are still held at the end.
    for note, onset in active.items():
        intervals.append((note, onset, current_time))

    return intervals


def build_label_matrix(
    note_intervals: List[Tuple[int, float, float]],
    n_patches: int,
    patch_duration: float,
) -> np.ndarray:
    """Build a binary label matrix from MIDI note intervals.

    A note is considered active in a patch if it overlaps with the patch's
    time window by any amount.

    Parameters
    ----------
    note_intervals : list of (int, float, float)
        ``(midi_note, onset_sec, offset_sec)`` tuples.
    n_patches : int
        Number of spectrogram patches.
    patch_duration : float
        Duration of each patch in seconds.

    Returns
    -------
    np.ndarray
        Binary matrix of shape ``(n_patches, 88)``, dtype float32.
    """
    labels = np.zeros((n_patches, 88), dtype=np.float32)

    for midi_note, onset, offset in note_intervals:
        # Skip notes outside the 88-key piano range.
        if midi_note < MIDI_LOW or midi_note > MIDI_HIGH:
            continue
        key_index = midi_note - MIDI_LOW  # 0–87

        # Find which patches this note overlaps.
        start_patch = int(onset / patch_duration)
        end_patch = int(offset / patch_duration)

        # Clamp to valid range.
        start_patch = max(0, start_patch)
        end_patch = min(n_patches - 1, end_patch)

        labels[start_patch : end_patch + 1, key_index] = 1.0

    return labels


# ────────────────────────────────────────────────────────────────────────
# Main processing
# ────────────────────────────────────────────────────────────────────────

def process_one(
    wav_path: str,
    midi_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single WAV + MIDI pair.

    Returns
    -------
    patches : np.ndarray
        Shape ``(N, 1, n_mels, n_time_frames)``.
    labels : np.ndarray
        Shape ``(N, 88)``.
    """
    # Audio → log-Mel → patches.
    waveform = load_audio(wav_path)
    waveform = remove_dc(waveform)
    waveform = rms_normalize(waveform)
    log_mel = compute_log_mel(waveform)
    patches = extract_patches(log_mel)  # (N, 1, F, T)

    n_patches = patches.shape[0]
    patch_duration = CFG.n_time_frames * CFG.hop_length / CFG.target_sr

    # MIDI → label matrix.
    note_intervals = midi_to_note_intervals(midi_path)
    labels = build_label_matrix(note_intervals, n_patches, patch_duration)

    return patches, labels


def read_maestro_csv(maestro_dir: Path) -> List[dict]:
    """Read the MAESTRO CSV and return a list of row dicts."""
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    maestro_dir = Path(args.maestro_dir)
    output_dir = Path(args.output_dir)

    rows = read_maestro_csv(maestro_dir)
    total = len(rows)

    if args.max_files > 0:
        rows = rows[: args.max_files]

    # Map MAESTRO split names to our directory names.
    split_map = {"train": "train", "validation": "val", "test": "test"}

    # Create output directory structure.
    for split_name in split_map.values():
        (output_dir / split_name / "audio").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    print(f"[prepare_maestro] MAESTRO dir : {maestro_dir}")
    print(f"[prepare_maestro] Output dir  : {output_dir}")
    print(f"[prepare_maestro] Files to process: {len(rows)} / {total}")
    print()

    stats = {"train": 0, "val": 0, "test": 0}

    for i, row in enumerate(rows):
        split = split_map.get(row["split"])
        if split is None:
            print(f"  Skipping unknown split: {row['split']}")
            continue

        wav_path = str(maestro_dir / row["audio_filename"])
        midi_path = str(maestro_dir / row["midi_filename"])
        stem = Path(row["audio_filename"]).stem

        # Skip if already processed.
        label_out = output_dir / split / "labels" / f"{stem}.npy"
        audio_out = output_dir / split / "audio" / f"{stem}.npz"
        if label_out.exists() and audio_out.exists():
            print(f"  [{i+1}/{len(rows)}] SKIP (exists) {stem}")
            stats[split] += 1
            continue

        try:
            patches, labels = process_one(wav_path, midi_path)
        except Exception as e:
            print(f"  [{i+1}/{len(rows)}] ERROR {stem}: {e}")
            continue

        # Save labels as .npy and patches as uncompressed .npz for fast loading.
        # Use np.savez (not savez_compressed) so workers can load files without
        # decompression overhead — this cuts per-file load time ~10x.
        np.save(str(label_out), labels)
        np.savez(str(audio_out), patches=patches)

        stats[split] += 1
        print(
            f"  [{i+1}/{len(rows)}] {split:5s}  {stem}  "
            f"patches={patches.shape[0]}  "
            f"active_notes={int(labels.sum())}"
        )

    print()
    print(f"[prepare_maestro] Done.  train={stats['train']}  "
          f"val={stats['val']}  test={stats['test']}")

    # ── Consolidate per-file outputs into single memory-mapped arrays ───
    print()
    for split_name in split_map.values():
        consolidate_split(output_dir, split_name)


def consolidate_split(output_dir: Path, split: str) -> None:
    """Merge per-file .npz + .npy into two big memory-mappable .npy files.

    Creates::

        data/{split}/patches_all.npy   ← (N_total, 1, 229, 32) float32
        data/{split}/labels_all.npy    ← (N_total, 88)          float32

    These files can be opened with ``np.load(..., mmap_mode='r')`` for
    zero-copy, O(1) random access during training.
    """
    audio_dir = output_dir / split / "audio"
    label_dir = output_dir / split / "labels"
    npz_files = sorted(audio_dir.glob("*.npz"))
    if not npz_files:
        print(f"  [{split}] No .npz files — skipping consolidation.")
        return

    patches_out = output_dir / split / "patches_all.npy"
    labels_out = output_dir / split / "labels_all.npy"

    # Skip if already consolidated.
    if patches_out.exists() and labels_out.exists():
        print(f"  [{split}] Already consolidated — skipping.")
        return

    # First pass: count total patches to pre-allocate.
    total = 0
    for npz_path in npz_files:
        stem = npz_path.stem
        lbl = label_dir / f"{stem}.npy"
        if not lbl.exists():
            continue
        n = np.load(str(lbl), mmap_mode="r").shape[0]
        total += n

    if total == 0:
        print(f"  [{split}] 0 patches — skipping.")
        return

    print(f"  [{split}] Consolidating {total:,} patches from "
          f"{len(npz_files)} files …")

    # Allocate on disk with np.lib.format.
    p_shape = (total, 1, CFG.n_mels, CFG.n_time_frames)
    l_shape = (total, CFG.n_notes)

    # Create empty files with the right shape, then memory-map for writing.
    _create_empty_npy(str(patches_out), p_shape, np.float32)
    _create_empty_npy(str(labels_out), l_shape, np.float32)

    p_mmap = np.load(str(patches_out), mmap_mode="r+")
    l_mmap = np.load(str(labels_out), mmap_mode="r+")

    offset = 0
    for npz_path in npz_files:
        stem = npz_path.stem
        lbl_path = label_dir / f"{stem}.npy"
        if not lbl_path.exists():
            continue
        patches = np.load(str(npz_path))["patches"]
        labels = np.load(str(lbl_path))
        n = patches.shape[0]
        p_mmap[offset : offset + n] = patches
        l_mmap[offset : offset + n] = labels
        offset += n

    # Flush to disk.
    del p_mmap, l_mmap
    print(f"  [{split}] Wrote {patches_out} and {labels_out}")


def _create_empty_npy(path: str, shape: tuple, dtype) -> None:
    """Create a .npy file with a valid header and pre-allocated zero data."""
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=shape,
    )
    del arr  # flush header


# ────────────────────────────────────────────────────────────────────────
# MaestroDataset – memory-mapped Dataset for fast random access
# ────────────────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset


class MaestroDataset(Dataset):
    """PyTorch Dataset backed by consolidated memory-mapped ``.npy`` files.

    After ``prepare_maestro.py`` runs, each split directory contains::

        data/{split}/patches_all.npy   ← (N, 1, 229, 32) float32
        data/{split}/labels_all.npy    ← (N, 88)          float32

    These are opened with ``mmap_mode='r'`` so that the OS pages data in
    on demand — no per-file open overhead, no decompression, and the OS
    page cache handles caching automatically.

    Parameters
    ----------
    data_dir : str | Path
        Root data directory (e.g. ``data``).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        split_dir = Path(data_dir) / split
        patches_path = split_dir / "patches_all.npy"
        labels_path = split_dir / "labels_all.npy"

        if not patches_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Consolidated files not found in {split_dir}. "
                f"Run prepare_maestro.py first."
            )

        # Memory-mapped read-only access — data is paged in by the OS.
        self._patches = np.load(str(patches_path), mmap_mode="r")
        self._labels = np.load(str(labels_path), mmap_mode="r")

        assert self._patches.shape[0] == self._labels.shape[0], (
            f"Patch/label count mismatch: "
            f"{self._patches.shape[0]} vs {self._labels.shape[0]}"
        )

    def __len__(self) -> int:
        return self._patches.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # np.array() copies from the mmap into a contiguous buffer so
        # the tensor doesn't hold a reference to the entire mmap region.
        patch = torch.from_numpy(np.array(self._patches[idx]))  # (1, F, T)
        label = torch.from_numpy(np.array(self._labels[idx]))   # (88,)
        return patch, label


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MAESTRO v3 dataset to ChordNet training format."
    )
    p.add_argument(
        "--maestro-dir", type=str, required=True,
        help="Path to the maestro-v3.0.0 directory.",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(CFG.data_dir),
        help=f"Output directory (default: {CFG.data_dir}).",
    )
    p.add_argument(
        "--max-files", type=int, default=0,
        help="Process at most N files (0 = all). Useful for testing.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
