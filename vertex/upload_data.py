"""
vertex/upload_data.py — One-time upload of preprocessed .npy files to GCS.

Run this ONCE from your local machine before submitting the training job.
The script uploads patches_all.npy and labels_all.npy for each split.

Usage:
    pip install google-cloud-storage
    python vertex/upload_data.py --bucket my-gcs-bucket --data-dir data
"""

import argparse
from pathlib import Path

from google.cloud import storage


def upload_file(
    bucket: storage.Bucket,
    local_path: Path,
    gcs_path: str,
) -> None:
    size_gb = local_path.stat().st_size / 1024**3
    print(f"  Uploading  {local_path}  ({size_gb:.2f} GB)  →  gs://{bucket.name}/{gcs_path}")
    blob = bucket.blob(gcs_path)
    # Use resumable upload (default for large files) with a generous timeout.
    blob.upload_from_filename(str(local_path), timeout=7200)
    print(f"    Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload ChordNet .npy data to GCS")
    p.add_argument("--bucket",   required=True, help="GCS bucket name (no gs:// prefix)")
    p.add_argument("--data-dir", default="data", help="Local data directory (default: data)")
    p.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits to upload (default: train val)",
    )
    args = p.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    data_dir = Path(args.data_dir)

    total_bytes = 0
    for split in args.splits:
        for fname in ("patches_all.npy", "labels_all.npy"):
            local_path = data_dir / split / fname
            if not local_path.exists():
                print(f"  Skipping {local_path}  (not found)")
                continue
            total_bytes += local_path.stat().st_size
            upload_file(bucket, local_path, f"data/{split}/{fname}")

    total_gb = total_bytes / 1024**3
    print(f"\nUpload complete.  Total: {total_gb:.2f} GB  →  gs://{args.bucket}/data/")


if __name__ == "__main__":
    main()
