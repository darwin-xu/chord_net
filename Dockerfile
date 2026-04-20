# ──────────────────────────────────────────────────────────────────────────────
# ChordNet training container
#
# Base image: official PyTorch + CUDA 12.1 + cuDNN 8 (runtime variant).
# Python 3.11 is provided by the base image.
#
# Build & push (from project root):
#   docker build -t REGION-docker.pkg.dev/PROJECT/chord-net/trainer:latest .
#   docker push    REGION-docker.pkg.dev/PROJECT/chord-net/trainer:latest
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#FROM gcr.io/complete-tube-271302/pytorch-base:2.3.0-cuda12.1

WORKDIR /app

# System libraries required by librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch / torchaudio already in base image — pip skips them)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir google-cloud-storage>=2.10

# Copy training source code
COPY *.py ./
COPY vertex/ vertex/
RUN chmod +x vertex/entrypoint.sh

ENTRYPOINT ["bash", "/app/vertex/entrypoint.sh"]
