#!/bin/bash
# vertex/entrypoint.sh — runs inside the Vertex AI training container.
#
# Environment variables (set by submit_job.py):
#   GCS_DATA_BUCKET  — GCS bucket name (no gs:// prefix)
#   ARCH             — model architecture: chordnet | resnet  (default: resnet)
#   EPOCHS           — number of training epochs               (default: 100)
#   BATCH_SIZE       — training batch size                     (default: 512)
#   NUM_WORKERS      — DataLoader worker processes             (default: 4)
#   EARLY_STOP       — early-stop patience in validated epochs (default: 15)

set -euo pipefail

echo "=== ChordNet Vertex AI Training ==="
echo "ARCH=${ARCH:-resnet}  EPOCHS=${EPOCHS:-100}  BATCH_SIZE=${BATCH_SIZE:-512}"
echo "Host: $(hostname)  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo 'none')"

# ── 1. Download training data from GCS ───────────────────────────────────────
echo ""
echo "--- Downloading data from gs://${GCS_DATA_BUCKET}/data/ ---"
python /app/vertex/gcs_io.py download

# ── 2. Train ─────────────────────────────────────────────────────────────────
echo ""
echo "--- Starting training ---"
python /app/train.py \
  --arch        "${ARCH:-resnet}" \
  --maestro \
  --data-dir    /tmp/chord_data \
  --checkpoint-dir /tmp/checkpoints \
  --epochs      "${EPOCHS:-100}" \
  --batch-size  "${BATCH_SIZE:-512}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --use-pos-weight \
  --early-stop  "${EARLY_STOP:-15}"

# ── 3. Upload checkpoints back to GCS ────────────────────────────────────────
echo ""
echo "--- Uploading checkpoints ---"
python /app/vertex/gcs_io.py upload

echo ""
echo "=== Training complete ==="
