# Training ChordNet on Google Vertex AI

## Do You Need Docker?

**Short answer: Yes, for this project. But here is why, and what the alternative looks like.**

Vertex AI offers two ways to run custom training:

| Approach | What you provide | Docker required? |
|---|---|---|
| **Custom Container** (what we use) | A Docker image pushed to Artifact Registry | Yes |
| **Python Package** | A `.tar.gz` Python package + a pre-built PyTorch container URI | No |

### Why the Python Package approach does NOT work well here

Google's pre-built PyTorch containers include torch and torchaudio, but NOT `librosa`, `mido`, `soundfile`, or `google-cloud-storage`. You can pass a `requirements.txt` to `CustomPythonPackageTrainingJob`, but those packages are installed fresh at every job start — adding 2–4 minutes of pip install time per run. More importantly:

- The project is not structured as a Python package (no `setup.py` / `pyproject.toml`). You would need to add that boilerplate.
- The entrypoint logic (download data → train → upload checkpoints) lives in `vertex/entrypoint.sh`. Replicating that without a shell entrypoint and Docker requires more restructuring.
- Python version mismatch: Google pre-built containers use Python 3.10/3.11. Our local code targets 3.11+ syntax (e.g. `tuple[int, ...]` type hints). This is fine, but worth knowing.

### Why Docker is the right choice

- **Self-contained**: all dependencies are baked in. The container starts and begins downloading data immediately — no pip install delay.
- **No code restructuring needed**: the existing `.py` files and `vertex/entrypoint.sh` work as-is.
- **Reproducible**: the exact same image runs locally (`docker run …`) and on Vertex AI.
- **Cached layers**: if you only change `train.py`, Docker rebuilds in seconds because the dependency layers are already cached in Artifact Registry.

### If you ever want to avoid Docker

You would need to:
1. Add a `setup.py` or `pyproject.toml` at the project root
2. Run `python setup.py sdist` to produce a `.tar.gz`
3. Upload that to GCS
4. Use `CustomPythonPackageTrainingJob` and pass `requirements=["librosa", "mido", "soundfile", "google-cloud-storage"]`
5. Rewrite `vertex/entrypoint.sh` logic in Python inside a `train_task()` function

For this project Docker is simpler. Move on.

---

## GCS Storage Cost for 25 GB

Standard Storage in `us-central1`:

| Item | Rate | 25 GB/month |
|---|---|---|
| Storage | $0.020 / GB / month | **$0.50 / month** |
| Inbound transfer (upload from laptop) | Free | $0.00 |
| Outbound to Vertex AI (same region) | Free | $0.00 |
| Class A operations (4 uploads) | $0.005 / 1000 ops | < $0.01 |
| **Total** | | **≈ $0.50 / month** |

The `.npy` data is written once and read once per training job. Keeping it in GCS for a month costs about **50 cents**. Negligible.

---

## Full Step-by-Step Guide

### Prerequisites (one-time installs)

```bash
# Google Cloud SDK
brew install --cask google-cloud-sdk

# Python client libraries (local machine only, not inside Docker)
pip install -r vertex/requirements.txt
```

### Step 1 — Authenticate and configure your project

```bash
PROJECT=my-gcp-project   # ← replace with your actual GCP project ID
REGION=us-central1
BUCKET=chord-net-$PROJECT  # must be globally unique

gcloud auth login
gcloud auth application-default login
gcloud config set project $PROJECT
```

### Step 2 — Enable required APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

This is a one-time command. It takes about 30 seconds.

### Step 3 — Create GCS bucket and Artifact Registry repo

```bash
# GCS bucket — keep it in the same region as your training jobs
# to avoid inter-region transfer charges
gcloud storage buckets create gs://$BUCKET \
  --location=$REGION \
  --uniform-bucket-level-access

# Docker image registry
gcloud artifacts repositories create chord-net \
  --repository-format=docker \
  --location=$REGION \
  --description="ChordNet training images"
```

### Step 4 — Grant Vertex AI permission to read the bucket

Vertex AI runs with a service account. It needs read access to your data bucket
and write access to the output bucket.

```bash
# Get the Vertex AI service account email for your project
SA="service-$(gcloud projects describe $PROJECT --format='value(projectNumber)')@gcp-sa-aiplatform.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectAdmin"
```

### Step 5 — Upload training data to GCS (one-time, ~25 GB)

Run from the project root. The script uploads `patches_all.npy` and `labels_all.npy`
for train and val splits.

```bash
python vertex/upload_data.py --bucket $BUCKET --data-dir data
```

Upload time depends on your internet connection — estimate 10–30 minutes for 25 GB.
This only needs to be done once. If you regenerate `data/` (e.g. re-run `prepare_maestro.py`),
run this command again.

### Step 6 — Build the Docker image

```bash
IMAGE=$REGION-docker.pkg.dev/$PROJECT/chord-net/trainer:latest

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build from project root (Dockerfile is at the root)
docker build -t $IMAGE .

# Push to Artifact Registry
docker push $IMAGE
```

The first build takes 3–5 minutes (downloading the PyTorch base image and installing
dependencies). Subsequent rebuilds are fast because Docker caches layers — only the
`COPY *.py` layer is invalidated when you change source code.

### Step 7 — Submit a training job

```bash
# Option A — T4 GPU (~$0.35/hr, ~10 min/epoch, cheapest)
python vertex/submit_job.py \
  --project  $PROJECT \
  --bucket   $BUCKET \
  --image    $IMAGE \
  --machine  t4 \
  --arch     resnet \
  --epochs   100

# Option B — A100 GPU (~$2.50/hr, ~3 min/epoch, fastest)
python vertex/submit_job.py \
  --project  $PROJECT \
  --bucket   $BUCKET \
  --image    $IMAGE \
  --machine  a100 \
  --batch-size 1024 \
  --arch     resnet \
  --epochs   100
```

`submit_job.py` returns immediately and prints a link to Cloud Console. The job runs
asynchronously. Monitor progress at:

```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=YOUR_PROJECT
```

Click the job → **View logs** for live tqdm output identical to what you see locally.

### Step 8 — Download checkpoints when training is done

```bash
mkdir -p checkpoints

# The entrypoint uploads checkpoints to gs://BUCKET/outputs/<job-id>/
# and also to gs://BUCKET/checkpoints/ as a convenience.
gsutil -m cp "gs://$BUCKET/checkpoints/*.pt" checkpoints/

# Or download from the auto-generated output path:
gsutil -m cp "gs://$BUCKET/outputs/**/best_resnet.pt" checkpoints/
```

### Step 9 — Continue training (resume)

If the job was stopped early or you want to train more epochs, download the
`last_resnet.pt` checkpoint and resume locally or on Vertex AI:

```bash
# Locally
python train.py --arch resnet --maestro --resume checkpoints/last_resnet.pt --epochs 200

# On Vertex AI: upload the checkpoint first, then add --resume to entrypoint.sh
```

---

## GPU Comparison

| GPU | VRAM | ~min/epoch (est.) | On-demand price | 100-epoch cost |
|-----|------|-------------------|-----------------|----------------|
| T4  | 16 GB | 10 min | ~$0.35 / hr | ~$6 |
| V100 | 16 GB | 5 min | ~$0.90 / hr | ~$8 |
| A100 | 40 GB | 3 min | ~$2.50 / hr | ~$13 |

Estimates assume batch_size=512 (T4/V100) or 1024 (A100) and early stopping around epoch 30–40,
so actual cost is typically 30–40% of the 100-epoch figure.

**Recommendation**: start with T4 for a short verification run (5 epochs), then switch to A100
for the full training run.

---

## Files Added to This Project

```
Dockerfile                      # Training container definition
.dockerignore                   # Excludes data/, .venv/, iOS/ from Docker context
vertex/
  entrypoint.sh                 # Container entry point: download → train → upload
  gcs_io.py                     # GCS download/upload helpers (runs inside container)
  upload_data.py                # One-time local → GCS data uploader
  submit_job.py                 # Vertex AI job submission script
  requirements.txt              # Local-only: google-cloud-aiplatform, google-cloud-storage
```

---

## Common Issues

**`Permission denied` when Vertex job reads GCS**
Re-run Step 4 to grant the Vertex AI service account `storage.objectAdmin` on the bucket.

**`docker push` fails with auth error**
Re-run `gcloud auth configure-docker $REGION-docker.pkg.dev`.

**Job fails immediately with exit code 1**
Click "View logs" in Cloud Console. Most common cause: a missing environment variable in `submit_job.py` or a broken import in the copied `.py` files.

**`patches_all.npy` not found in container**
The download step in `vertex/gcs_io.py` expects the files at `data/train/patches_all.npy`
inside the bucket. Run Step 5 again with `--data-dir data` pointing to your local data directory.
