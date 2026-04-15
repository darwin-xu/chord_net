# ChordNet – Piano Note & Chord Detection

A CNN-based neural network that detects which piano keys are pressed from
an audio waveform.  Input is raw audio from a microphone; output is an
88-dimensional binary vector (one element per piano key, MIDI 21–108).

---

## Project Structure

```
chord_net/
├── config.py              # All tuneable hyper-parameters
├── preprocess.py          # Audio → log-Mel spectrogram pipeline
├── dataset.py             # PyTorch Dataset (raw WAV + .npy labels)
├── prepare_maestro.py     # MAESTRO v3 → ChordNet training data
├── model.py               # ChordNet CNN architecture
├── train.py               # Training & validation loop
├── inference.py           # Run trained model on audio files
├── requirements.txt       # Python dependencies
└── README.md              # ← you are here
```

---

## 1. Setup

```bash
# Create a virtual environment and install dependencies.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, torchaudio, librosa, soundfile, numpy, mido.

---

## 2. Preparing Training Data

### 2.1 Data Requirements

ChordNet needs paired **audio** and **labels**:

| Item   | Format | Description |
|--------|--------|-------------|
| Audio  | WAV (any supported SR / channels / bit-depth) | Piano performance recordings |
| Labels | `.npy` – shape `(N_patches, 88)`, float32 | Binary matrix: 1 = note active, 0 = inactive |

Each label row corresponds to one spectrogram **patch** (229 mel-bins × 32 time-frames ≈ 0.74 s at 22.05 kHz).

### 2.2 Collecting Your Own Data

1. **Record or obtain** piano audio (WAV preferred).
   - Supported input: 16 kHz / 22.05 kHz / 44.1 kHz / 48 kHz, mono or stereo, 16/24/32-bit.
   - All audio is internally resampled to **mono, 22.05 kHz, 16-bit PCM**.

2. **Create ground-truth labels.**  Options:
   - **From MIDI files** — if you have aligned MIDI, use the MIDI-to-label
     conversion in `prepare_maestro.py` as a reference.
   - **Manual annotation** — tools like [Sonic Visualiser](https://sonicvisualiser.org/)
     or [tony](https://code.soundsoftware.ac.uk/projects/tony) can help.
   - **Synthetic data** — render MIDI → WAV with a piano soundfont, then
     derive labels from the same MIDI.

3. **Organise** into the expected layout:

```
data/
├── train/
│   ├── audio/
│   │   ├── 001.wav
│   │   └── ...
│   └── labels/
│       ├── 001.npy      # (N_patches, 88)
│       └── ...
└── val/
    ├── audio/
    │   └── ...
    └── labels/
        └── ...
```

> **Important:** The stem of each `.wav` must match the stem of its `.npy`
> label file (e.g. `song1.wav` ↔ `song1.npy`), and the number of rows in
> the `.npy` must equal the number of patches extracted from the audio.

### 2.3 Using the MAESTRO v3 Dataset (Recommended)

The [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) dataset
provides ~200 hours of virtuosic piano performances with perfectly aligned
MIDI.  It is the easiest way to get a large, high-quality training set.

**Download:**
```bash
# ~120 GB uncompressed.  Download from:
# https://magenta.tensorflow.org/datasets/maestro
# and unpack to e.g. ~/Desktop/maestro-v3.0.0
```

The dataset contains:
- **1,276 recordings** split into train (962), validation (137), test (177).
- Each recording has a paired `.wav` + `.midi` file.
- Audio: stereo, 44.1 kHz, 16-bit PCM.
- MIDI: type-1, note_on/note_off events with precise timestamps.

**Preprocess MAESTRO → ChordNet format:**

```bash
# Process ALL files (may take several hours on the full set).
python prepare_maestro.py \
    --maestro-dir /path/to/maestro-v3.0.0 \
    --output-dir  data

# Process only the first 10 files (for a quick test).
python prepare_maestro.py \
    --maestro-dir /path/to/maestro-v3.0.0 \
    --output-dir  data \
    --max-files   10
```

**What the script does (per recording):**

1. Loads the WAV → resamples to mono 22.05 kHz.
2. DC removal → RMS normalisation.
3. Computes log-Mel spectrogram (n_fft=2048, hop=512, 229 mels).
4. Slices into non-overlapping patches of 32 time-frames each.
5. Parses the aligned MIDI → builds a binary `(N_patches, 88)` label matrix.
6. Saves patches as `.npz` and labels as `.npy`.

**Output layout:**
```
data/
├── train/
│   ├── audio/<stem>.npz      ← compressed spectrogram patches
│   └── labels/<stem>.npy     ← binary note labels
├── val/
│   ├── audio/
│   └── labels/
└── test/
    ├── audio/
    └── labels/
```

The script is **resumable** — it skips files that already have both `.npz`
and `.npy` outputs, so you can safely re-run it if interrupted.

---

## 3. Training

### 3.1 Train with MAESTRO data (recommended)

```bash
python train.py \
    --data-dir data \
    --maestro \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3
```

The `--maestro` flag tells the training script to load the preprocessed
memory-mapped `.npy` files via `MaestroDataset` (from `prepare_maestro.py`).

### 3.2 Train with raw WAV + .npy labels

```bash
python train.py \
    --data-dir data \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3
```

Without `--maestro`, it uses `ChordDataset` (from `dataset.py`) which
expects raw `.wav` in `audio/` and pre-built `.npy` in `labels/`.

### 3.3 CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Root data directory |
| `--checkpoint-dir` | `checkpoints` | Where to save model checkpoints |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 256 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--weight-decay` | 1e-4 | L2 regularisation |
| `--num-workers` | 0 | DataLoader worker processes |
| `--maestro` | off | Use MaestroDataset instead of ChordDataset |
| `--resume` | — | Path to checkpoint (`.pt`) to resume training from |

### 3.4 What Happens During Training

- **Loss:** `BCEWithLogitsLoss` (binary cross-entropy with built-in sigmoid).
- **Optimiser:** Adam with weight decay.
- **Scheduler:** ReduceLROnPlateau (halves LR after 5 epochs without
  validation-loss improvement).
- **Metrics:** Precision, Recall, F1 (micro-averaged over all 88 notes).
- **Checkpoints:**
  - `checkpoints/best.pt` — model with the highest validation F1.
  - `checkpoints/last.pt` — model state at the final epoch.

### 3.5 Resuming Training

To continue training from a previous run, use `--resume`:

```bash
python train.py \
    --data-dir data \
    --maestro \
    --epochs 50 \
    --resume checkpoints/last.pt
```

This restores the model, optimiser, and scheduler state and resumes from
the next epoch.  `--epochs` is the **total** epoch count, not the number
of additional epochs (e.g. if the checkpoint is from epoch 3 and you pass
`--epochs 50`, training continues from epoch 4 to 50).

### 3.6 Expected Output

```
[ChordNet] Using device: mps
[ChordNet] Train patches : 562440
[ChordNet] Val   patches : 81320
Epoch 001/050  train_loss=0.1832  val_loss=0.1245  val_P=0.612  val_R=0.534  val_F1=0.570  (42.3s)
  ↳ Saved best model (F1=0.5700) → checkpoints/best.pt
Epoch 002/050  ...
```

---

## 4. Validation / Testing

Validation runs automatically at the end of every training epoch.
To evaluate a saved checkpoint on the test split separately:

```bash
python -c "
import torch
from torch.utils.data import DataLoader
from config import CFG
from model import ChordNet
from prepare_maestro import MaestroDataset
from train import validate, get_device

device = get_device()
model = ChordNet().to(device)
ckpt = torch.load('checkpoints/best.pt', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])

test_ds = MaestroDataset('data', split='test')
test_loader = DataLoader(test_ds, batch_size=64, num_workers=4)

metrics = validate(model, test_loader, torch.nn.BCEWithLogitsLoss(), device)
print(f'Test  loss={metrics[\"loss\"]:.4f}  P={metrics[\"precision\"]:.3f}  '
      f'R={metrics[\"recall\"]:.3f}  F1={metrics[\"f1\"]:.3f}')
"
```

---

## 5. Inference

Run a trained model on new audio files:

```bash
# Single file
python inference.py \
    --checkpoint checkpoints/best.pt \
    --input recording.wav

# All WAV files in a directory
python inference.py \
    --checkpoint checkpoints/best.pt \
    --input audio_folder/

# Custom threshold and device
python inference.py \
    --checkpoint checkpoints/best.pt \
    --input recording.wav \
    --threshold 0.4 \
    --device cpu
```

### 5.1 Output Format

```
============================================================
  recording.wav
============================================================
  [  0.000s –   0.743s]  C4 (0.92), E4 (0.87), G4 (0.81)
  [  0.743s –   1.486s]  C4 (0.95), E4 (0.90), G4 (0.85)
  [  1.486s –   2.229s]  —
  ...
```

Each line shows the time window and the detected notes with their
sigmoid probabilities.  Notes below the threshold are suppressed.

### 5.2 CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to a `.pt` checkpoint |
| `--input` | (required) | Audio file or directory |
| `--threshold` | 0.5 | Sigmoid probability threshold |
| `--device` | auto | Force `cuda` / `mps` / `cpu` |

---

## 6. Configuration Reference

All hyper-parameters live in [`config.py`](config.py).  Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_sr` | 22050 | Target sample rate (Hz) |
| `n_fft` | 2048 | FFT window size |
| `hop_length` | 512 | Hop between frames |
| `n_mels` | 229 | Mel filter-bank bands |
| `n_time_frames` | 32 | Frames per CNN input patch |
| `n_notes` | 88 | Piano keys (MIDI 21–108) |
| `conv_channels` | [32, 64, 128, 256] | Channels per conv block |
| `fc_hidden` | 128 | MLP hidden size |
| `dropout` | 0.3 | Classifier dropout |
| `epochs` | 50 | Training epochs |
| `batch_size` | 64 | Batch size |
| `learning_rate` | 1e-3 | Adam LR |
| `inference_threshold` | 0.5 | Note detection threshold |

---

## 7. Model Architecture

```
Input  (B, 1, 229, 32)
 │
 ├─ Block 1: Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2)
 ├─ Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2)
 ├─ Block 3: Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2)
 ├─ Block 4: Conv2d(128→256, 3×3) → BN → ReLU
 │
 ├─ AdaptiveAvgPool2d(1,1) → flatten → (B, 256)
 │
 └─ Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→88)

Output (B, 88)  ← raw logits; apply sigmoid for probabilities
```

Total parameters: **433,048**.

---

## 8. Audio Processing Pipeline

```
Raw audio (any supported format)
  │
  ├─ Resample → mono, 22.05 kHz
  ├─ DC removal (subtract mean)
  ├─ RMS normalisation (target RMS = 0.1)
  ├─ Log-Mel spectrogram (2048 FFT, 512 hop, 229 mels)
  └─ Slice into patches of 32 frames (~0.74 s each)
```
