"""
preprocess.py – Audio preprocessing pipeline.

Pipeline:  load → stereo→mono → resample → 16-bit PCM
           → DC removal → RMS normalise → log-Mel spectrogram
"""

import numpy as np
import librosa
import soundfile as sf

from config import CFG


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load an audio file, convert to mono 22.05 kHz 16-bit PCM.

    Parameters
    ----------
    path : str
        Path to a WAV / FLAC / MP3 file.

    Returns
    -------
    np.ndarray
        1-D float32 waveform normalised to [-1, 1].
    """
    # librosa.load returns float32 mono by default.
    waveform, sr = librosa.load(path, sr=CFG.target_sr, mono=True)
    return waveform


def remove_dc(waveform: np.ndarray) -> np.ndarray:
    """Subtract the mean (DC offset) from the waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 audio signal.

    Returns
    -------
    np.ndarray
        DC-free waveform.
    """
    return waveform - np.mean(waveform)


def rms_normalize(waveform: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale waveform so that its RMS equals *target_rms*.

    If the signal is silent (RMS ≈ 0) it is returned unchanged to avoid
    division by zero.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 audio signal.
    target_rms : float
        Desired RMS level.

    Returns
    -------
    np.ndarray
        RMS-normalised waveform.
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-8:
        return waveform
    return waveform * (target_rms / rms)


def compute_log_mel(waveform: np.ndarray) -> np.ndarray:
    """Compute a log-Mel spectrogram from a preprocessed waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 audio signal (already mono, resampled, DC-removed,
        RMS-normalised).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_mels, T)`` where *T* depends on the
        waveform length.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=CFG.target_sr,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        n_mels=CFG.n_mels,
    )
    # Stabilised log scale.
    log_mel = np.log(mel + CFG.log_offset)
    return log_mel


def preprocess(path: str) -> np.ndarray:
    """Full pipeline: file → log-Mel spectrogram.

    Parameters
    ----------
    path : str
        Path to an audio file.

    Returns
    -------
    np.ndarray
        Log-Mel spectrogram of shape ``(n_mels, T)``.
    """
    waveform = load_audio(path)
    waveform = remove_dc(waveform)
    waveform = rms_normalize(waveform)
    log_mel = compute_log_mel(waveform)
    return log_mel


def extract_patches(log_mel: np.ndarray) -> np.ndarray:
    """Slice a full spectrogram into fixed-length patches.

    Each patch has shape ``(1, n_mels, n_time_frames)`` – ready for the
    CNN (batch dim is added later by the DataLoader).

    Non-overlapping windows are used.  The last partial window, if any,
    is zero-padded on the right.

    Parameters
    ----------
    log_mel : np.ndarray
        Full spectrogram of shape ``(n_mels, T)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 1, n_mels, n_time_frames)``.
    """
    n_mels, total_frames = log_mel.shape
    T = CFG.n_time_frames
    patches = []

    for start in range(0, total_frames, T):
        patch = log_mel[:, start : start + T]
        # Zero-pad the last patch if it is shorter than T frames.
        if patch.shape[1] < T:
            pad_width = T - patch.shape[1]
            patch = np.pad(patch, ((0, 0), (0, pad_width)), mode="constant")
        # Add channel dimension → (1, F, T).
        patches.append(patch[np.newaxis, :, :])

    return np.array(patches, dtype=np.float32)
