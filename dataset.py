"""
dataset.py – PyTorch Dataset for piano-note detection.

Expected directory layout
-------------------------
::

    data/
    ├── train/
    │   ├── audio/
    │   │   ├── 001.wav
    │   │   └── ...
    │   └── labels/
    │       ├── 001.npy      ← (N_patches, 88) float32  binary vectors
    │       └── ...
    └── val/
        ├── audio/
        └── labels/

Each ``.npy`` label file is a 2-D array of shape ``(N, 88)`` whose rows
correspond 1-to-1 with the non-overlapping spectrogram patches extracted
from the matching audio file.  A value of 1.0 means the note is active
in that patch; 0.0 means inactive.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG
from preprocess import preprocess, extract_patches


class ChordDataset(Dataset):
    """Yields ``(patch, label)`` pairs for training / validation.

    On first access every audio file is preprocessed and sliced into
    fixed-length patches; the results are cached in memory so that
    subsequent epochs are fast.

    Parameters
    ----------
    audio_dir : str | Path
        Directory containing audio files (``*.wav``).
    label_dir : str | Path
        Directory containing label files (``*.npy``).
    """

    def __init__(self, audio_dir: str | Path, label_dir: str | Path) -> None:
        self.audio_dir = Path(audio_dir)
        self.label_dir = Path(label_dir)

        # Discover files – sort for reproducibility.
        self.audio_files: List[Path] = sorted(self.audio_dir.glob("*.wav"))
        if not self.audio_files:
            raise FileNotFoundError(
                f"No .wav files found in {self.audio_dir}"
            )

        # Pre-compute all patches and labels.
        self.patches: List[np.ndarray] = []   # Each element: (1, F, T)
        self.labels: List[np.ndarray] = []    # Each element: (88,)

        for audio_path in self.audio_files:
            stem = audio_path.stem
            label_path = self.label_dir / f"{stem}.npy"
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Label file missing for {audio_path.name}: "
                    f"expected {label_path}"
                )

            # Preprocess audio → log-Mel → patches.
            log_mel = preprocess(str(audio_path))
            file_patches = extract_patches(log_mel)  # (N, 1, F, T)

            # Load the corresponding binary label matrix.
            file_labels = np.load(str(label_path)).astype(np.float32)

            # Safety check: patch / label counts must match.
            n_patches = file_patches.shape[0]
            n_labels = file_labels.shape[0]
            if n_patches != n_labels:
                raise ValueError(
                    f"{audio_path.name}: {n_patches} patches but "
                    f"{n_labels} label rows.  They must be equal."
                )

            for i in range(n_patches):
                self.patches.append(file_patches[i])
                self.labels.append(file_labels[i])

    # ── Dataset interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = torch.from_numpy(self.patches[idx])   # (1, F, T)
        label = torch.from_numpy(self.labels[idx])     # (88,)
        return patch, label
