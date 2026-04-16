"""
vertex/gcs_io.py — GCS helpers that run inside the training container.

Called by entrypoint.sh:
    python gcs_io.py download   # fetch .npy files from GCS before training
    python gcs_io.py upload     # push checkpoint .pt files to GCS after training
"""

import os
import sys
from pathlib import Path

from google.cloud import storage


# ── Download ──────────────────────────────────────────────────────────────────

def download_data() -> None:
    bucket_name = os.environ["GCS_DATA_BUCKET"]
    local_root = Path("/tmp/chord_data")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = [
        ("data/train/patches_all.npy", local_root / "train" / "patches_all.npy"),
        ("data/train/labels_all.npy",  local_root / "train" / "labels_all.npy"),
        ("data/val/patches_all.npy",   local_root / "val"   / "patches_all.npy"),
        ("data/val/labels_all.npy",    local_root / "val"   / "labels_all.npy"),
    ]

    for gcs_path, local_path in files:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(gcs_path)
        size_gb = blob.size / 1024**3 if blob.size else 0.0
        print(f"  gs://{bucket_name}/{gcs_path}  ({size_gb:.2f} GB)  →  {local_path}")
        blob.download_to_filename(str(local_path))
        actual_gb = local_path.stat().st_size / 1024**3
        print(f"    OK  ({actual_gb:.2f} GB on disk)")


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_checkpoints() -> None:
    local_ckpt = Path("/tmp/checkpoints")
    if not local_ckpt.exists():
        print("  No checkpoint directory found; skipping upload.")
        return

    # Vertex AI sets AIP_MODEL_DIR (e.g. "gs://bucket/outputs/12345/").
    # Fall back to gs://<GCS_DATA_BUCKET>/checkpoints/.
    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "").rstrip("/")
    bucket_name = os.environ["GCS_DATA_BUCKET"]
    client = storage.Client()

    if aip_model_dir.startswith("gs://"):
        stripped = aip_model_dir[len("gs://"):]
        out_bucket_name, _, prefix = stripped.partition("/")
        out_bucket = client.bucket(out_bucket_name)
    else:
        out_bucket = client.bucket(bucket_name)
        prefix = "checkpoints"

    for ckpt_file in sorted(local_ckpt.glob("*.pt")):
        gcs_path = f"{prefix}/{ckpt_file.name}" if prefix else ckpt_file.name
        size_mb = ckpt_file.stat().st_size / 1024**2
        print(f"  {ckpt_file.name}  ({size_mb:.1f} MB)  →  gs://{out_bucket.name}/{gcs_path}")
        blob = out_bucket.blob(gcs_path)
        blob.upload_from_filename(str(ckpt_file), timeout=600)
        print(f"    OK")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("download", "upload"):
        print("Usage: python gcs_io.py <download|upload>")
        sys.exit(1)

    if sys.argv[1] == "download":
        download_data()
    else:
        upload_checkpoints()
