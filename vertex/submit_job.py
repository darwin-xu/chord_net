"""
vertex/submit_job.py — Submit a ChordNet Custom Training job to Vertex AI.

Prerequisites (run once):
    pip install google-cloud-aiplatform
    gcloud auth application-default login

Usage:
    python vertex/submit_job.py \\
        --project  my-gcp-project \\
        --bucket   my-gcs-bucket \\
        --image    us-central1-docker.pkg.dev/my-gcp-project/chord-net/trainer:latest \\
        [--region  us-central1] \\
        [--machine t4|v100|a100] \\
        [--arch    resnet] \\
        [--epochs  100]
"""

import argparse

from google.cloud import aiplatform


# ── Machine / accelerator presets ─────────────────────────────────────────────
# Approximate on-demand pricing (us-central1, April 2026):
#   T4   ~$0.35/hr  — good for testing / cost-sensitive runs
#   V100 ~$0.90/hr  — good balance for medium jobs
#   A100 ~$2.50/hr  — fastest; recommended for production training
MACHINE_CONFIGS = {
    "t4": {
        "machine_type":      "n1-standard-8",
        "accelerator_type":  "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
        "batch_size_hint":   512,
    },
    "v100": {
        "machine_type":      "n1-standard-8",
        "accelerator_type":  "NVIDIA_TESLA_V100",
        "accelerator_count": 1,
        "batch_size_hint":   512,
    },
    "a100": {
        "machine_type":      "a2-highgpu-1g",
        "accelerator_type":  "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "batch_size_hint":   1024,
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description="Submit ChordNet training to Vertex AI")
    p.add_argument("--project",    required=True, help="GCP project ID")
    p.add_argument("--bucket",     required=True, help="GCS bucket name (no gs:// prefix)")
    p.add_argument("--image",      required=True, help="Full Artifact Registry image URI")
    p.add_argument("--region",     default="us-central1", help="GCP region (default: us-central1)")
    p.add_argument("--machine",    default="t4", choices=list(MACHINE_CONFIGS),
                   help="GPU machine preset: t4 | v100 | a100  (default: t4)")
    p.add_argument("--arch",       default="resnet", choices=["chordnet", "resnet"],
                   help="Model architecture (default: resnet)")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument("--job-name",   default="chord-net-train",
                   help="Display name for the Vertex AI job")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Override batch size (default: 512 for T4/V100, 1024 for A100)")
    args = p.parse_args()

    mc = MACHINE_CONFIGS[args.machine]
    batch_size = args.batch_size if args.batch_size > 0 else mc["batch_size_hint"]

    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.job_name,
        container_uri=args.image,
    )

    job.run(
        machine_type=mc["machine_type"],
        accelerator_type=mc["accelerator_type"],
        accelerator_count=mc["accelerator_count"],
        base_output_dir=f"gs://{args.bucket}/outputs",
        environment_variables={
            "GCS_DATA_BUCKET": args.bucket,
            "ARCH":            args.arch,
            "EPOCHS":          str(args.epochs),
            "BATCH_SIZE":      str(batch_size),
            "NUM_WORKERS":     "4",
            "EARLY_STOP":      str(args.early_stop),
        },
        replica_count=1,
        sync=False,  # returns immediately; tail logs in Cloud Console
    )

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
        f"?project={args.project}"
    )
    print(f"Job submitted: {job.resource_name}")
    print(f"Monitor at:    {console_url}")
    print(f"Checkpoints → gs://{args.bucket}/outputs/.../  (also gs://{args.bucket}/checkpoints/)")


if __name__ == "__main__":
    main()
