"""
config.py – Central configuration for ChordNet.

All tuneable hyper-parameters and paths live here so that every other
module can simply ``from config import CFG``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Single source of truth for every configurable value."""

    # ── Paths ───────────────────────────────────────────────────────────
    # Root directory that contains audio files for training / validation.
    # Expected layout:  data_dir / {train,val} / *.wav
    data_dir: Path = Path("data")

    # Where model checkpoints are saved during training.
    checkpoint_dir: Path = Path("checkpoints")

    # ── Audio I/O ───────────────────────────────────────────────────────
    # Supported input sample rates (Hz).  Files at these rates are
    # accepted and resampled to `target_sr`.
    supported_sample_rates: List[int] = field(
        default_factory=lambda: [16_000, 22_050, 44_100, 48_000]
    )

    # Target sample rate after resampling (Hz).
    target_sr: int = 22_050

    # Target number of audio channels (1 = mono).
    target_channels: int = 1

    # Target sample width in bytes (2 = 16-bit PCM).
    target_sample_width: int = 2

    # ── Spectrogram (Log-Mel) ───────────────────────────────────────────
    # FFT window size in samples.
    n_fft: int = 2048

    # Hop length between successive frames (samples).
    hop_length: int = 512

    # Number of Mel filter-bank bands.
    n_mels: int = 229

    # Number of time-frames per input patch fed to the CNN.
    n_time_frames: int = 32

    # Small constant added before log to avoid log(0).
    log_offset: float = 1e-6

    # ── Model architecture ──────────────────────────────────────────────
    # Number of piano keys (MIDI notes 21–108).
    n_notes: int = 88

    # Channel counts for each conv block.
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Kernel size used in every Conv2d layer.
    conv_kernel_size: int = 3

    # Padding for every Conv2d (keeps spatial dims before pooling).
    conv_padding: int = 1

    # Hidden dimension in the classifier MLP head.
    fc_hidden: int = 128

    # Dropout probability in the classifier head.
    dropout: float = 0.3

    # ── Training ────────────────────────────────────────────────────────
    # Total number of training epochs.
    epochs: int = 50

    # Mini-batch size.
    # Larger batches keep the GPU better utilised; 256 is a good default
    # for MPS (Apple Silicon).  Reduce if you run out of memory.
    batch_size: int = 256

    # Initial learning rate for Adam.
    learning_rate: float = 1e-3

    # Weight-decay (L2 regularisation).
    weight_decay: float = 1e-4

    # Fraction of training data held out for validation when no explicit
    # validation set is provided.
    val_split: float = 0.1

    # Number of data-loader workers.  0 = main process only (fastest with
    # memory-mapped data on macOS / MPS).
    num_workers: int = 0

    # Device for training / inference ("cuda", "mps", or "cpu").
    # An empty string means auto-detect.
    device: str = ""

    # ── Inference ───────────────────────────────────────────────────────
    # Sigmoid threshold: a note is considered "active" when its
    # probability exceeds this value.
    inference_threshold: float = 0.5


# ── Global singleton ────────────────────────────────────────────────────
CFG = Config()
