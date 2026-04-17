"""
train.py – Training and validation loop for ChordNet.

Usage
-----
::

    python train.py                         # uses defaults from config.py
    python train.py --epochs 100 --lr 5e-4  # override on the CLI
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import ChordDataset
from model import ChordNet
from model_resnet import ChordResNet
from prepare_maestro import MaestroDataset


# ────────────────────────────────────────────────────────────────────────
# Architecture registry
# ────────────────────────────────────────────────────────────────────────

ARCH_REGISTRY: dict[str, type] = {
    "chordnet": ChordNet,
    "resnet": ChordResNet,
}

# Threshold candidates swept after each validation epoch.
SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def build_model(arch: str) -> nn.Module:
    """Instantiate a model by architecture name."""
    cls = ARCH_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {', '.join(ARCH_REGISTRY)}"
        )
    return cls()


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if CFG.device:
        return torch.device(CFG.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, and F1 for multi-label predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs, shape ``(B, 88)``.
    targets : torch.Tensor
        Ground-truth binary labels, shape ``(B, 88)``.
    threshold : float
        Sigmoid threshold for converting logits to binary predictions.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def find_best_threshold(
    logits_np: np.ndarray,
    targets_np: np.ndarray,
) -> tuple[float, float]:
    """Sweep SWEEP_THRESHOLDS and return (best_threshold, best_f1)."""
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    best_t, best_f1 = 0.5, 0.0
    for t in SWEEP_THRESHOLDS:
        preds = (probs >= t).astype(np.float32)
        tp = (preds * targets_np).sum()
        fp = (preds * (1 - targets_np)).sum()
        fn = ((1 - preds) * targets_np).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t  = float(t)
    return best_t, best_f1


# ────────────────────────────────────────────────────────────────────────
# Train / Validate one epoch
# ────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one training epoch.

    Returns
    -------
    dict
        ``loss``, ``precision``, ``recall``, ``f1`` averaged over
        the epoch.
    """
    model.train()
    running_loss = 0.0
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = patches.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        # Running metric accumulators.
        preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
        tp_sum += (preds * labels).sum().item()
        fp_sum += (preds * (1 - labels)).sum().item()
        fn_sum += ((1 - preds) * labels).sum().item()

        # Update progress bar.
        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    epoch_loss = running_loss / n_samples
    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"loss": epoch_loss, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run one validation pass and collect raw logits for threshold sweep.

    Returns
    -------
    tuple of (metrics_dict, logits_np, targets_np)
        ``metrics_dict`` has keys ``loss``, ``precision``, ``recall``, ``f1``
        (all at threshold=0.5). ``logits_np`` and ``targets_np`` are raw
        numpy arrays used by ``find_best_threshold``.
    """
    model.eval()
    running_loss = 0.0
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    n_samples = 0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        bs = patches.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp_sum += (preds * labels).sum().item()
        fp_sum += (preds * (1 - labels)).sum().item()
        fn_sum += ((1 - preds) * labels).sum().item()

        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    epoch_loss = running_loss / n_samples
    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    metrics = {"loss": epoch_loss, "precision": precision, "recall": recall, "f1": f1}
    return metrics, np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def compute_pos_weight(
    args: argparse.Namespace,
    train_ds,
    max_weight: float,
) -> torch.Tensor:
    """Estimate per-note positive class weights for BCEWithLogitsLoss.

    For MAESTRO this reads labels directly from labels_all.npy (fast, vectorized).
    For generic datasets it falls back to sampling labels from the dataset.
    """
    eps = 1e-6

    if args.maestro:
        labels_path = Path(args.data_dir) / "train" / "labels_all.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing label file for pos_weight: {labels_path}. "
                "Run prepare_maestro.py first."
            )
        labels = np.load(labels_path, mmap_mode="r")
        pos_frac = labels.mean(axis=0).astype(np.float64)
    else:
        # Keep fallback bounded so startup does not become too slow.
        n = min(len(train_ds), 100_000)
        idx = np.linspace(0, len(train_ds) - 1, num=n, dtype=np.int64)
        pos_sum = None
        for i in idx:
            _, y = train_ds[int(i)]
            y_np = y.numpy().astype(np.float64)
            if pos_sum is None:
                pos_sum = np.zeros_like(y_np)
            pos_sum += y_np
        pos_frac = pos_sum / float(n)

    pos_frac = np.clip(pos_frac, eps, 1.0 - eps)
    weights = (1.0 - pos_frac) / pos_frac
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def write_html_log(csv_path: Path, html_path: Path, arch: str) -> None:
    """Regenerate the HTML training dashboard from the current CSV log.

    The file auto-refreshes every 60 seconds when opened in a browser,
    so you can watch training progress in real time.
    """
    if not csv_path.exists():
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def col(key: str) -> list:
        return [float(r[key]) for r in rows]

    epochs  = [int(r["epoch"]) for r in rows]
    last    = rows[-1]
    e_json  = json.dumps(epochs)
    tl_json = json.dumps(col("train_loss"))
    vl_json = json.dumps(col("val_loss"))
    tf_json = json.dumps(col("train_f1"))
    vf_json = json.dumps(col("val_f1"))
    bf_json = json.dumps(col("best_f1"))
    vp_json = json.dumps(col("val_precision"))
    vr_json = json.dumps(col("val_recall"))
    th_json = json.dumps(col("best_threshold"))
    lr_json = json.dumps([float(r["lr"]) * 1000 for r in rows])

    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['epoch']}</td>"
        f"<td>{float(r['train_loss']):.4f}</td>"
        f"<td>{float(r['train_f1']):.4f}</td>"
        f"<td>{float(r['val_loss']):.4f}</td>"
        f"<td>{float(r['val_f1']):.4f}</td>"
        f"<td class='hi'>{float(r['best_f1']):.4f}</td>"
        f"<td>{float(r['best_threshold']):.2f}</td>"
        f"<td>{float(r['val_precision']):.4f}</td>"
        f"<td>{float(r['val_recall']):.4f}</td>"
        f"<td>{float(r['lr']):.2e}</td>"
        "</tr>"
        for r in reversed(rows)
    )

    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='UTF-8'>\n"
        "  <meta http-equiv='refresh' content='60'>\n"
        f"  <title>ChordNet [{arch}] Training</title>\n"
        "  <script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'></script>\n"
        "  <style>\n"
        "    body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;margin:0;padding:20px}\n"
        "    h1{color:#7eb8f7;margin-bottom:4px}\n"
        "    .sub{color:#888;margin-bottom:24px;font-size:13px}\n"
        "    .stats{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}\n"
        "    .stat{background:#1a1a2e;border-radius:8px;padding:12px 20px;min-width:120px}\n"
        "    .sl{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px}\n"
        "    .sv{font-size:22px;font-weight:700;color:#7eb8f7;margin-top:4px}\n"
        "    .charts{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}\n"
        "    .cb{background:#1a1a2e;border-radius:8px;padding:16px}\n"
        "    .cb h3{margin:0 0 12px;font-size:12px;color:#aaa;font-weight:500;text-transform:uppercase;letter-spacing:.5px}\n"
        "    canvas{max-height:200px}\n"
        "    table{width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:8px;overflow:hidden;font-size:12px}\n"
        "    th{background:#16213e;padding:8px 12px;text-align:right;color:#888;font-weight:500;font-size:11px;text-transform:uppercase}\n"
        "    th:first-child{text-align:center}\n"
        "    td{padding:7px 12px;text-align:right;border-top:1px solid #222}\n"
        "    td:first-child{text-align:center;color:#7eb8f7;font-weight:600}\n"
        "    tr:hover td{background:#16213e}\n"
        "    .hi{color:#5af78e;font-weight:700}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>ChordNet [{arch.upper()}] Training Dashboard</h1>\n"
        f"  <div class='sub'>Auto-refreshes every 60 s &nbsp;&middot;&nbsp; Last update: epoch {last['epoch']}</div>\n"
        "  <div class='stats'>\n"
        f"    <div class='stat'><div class='sl'>Epoch</div><div class='sv'>{last['epoch']}</div></div>\n"
        f"    <div class='stat'><div class='sl'>Best F1</div><div class='sv'>{float(last['best_f1']):.4f}</div></div>\n"
        f"    <div class='stat'><div class='sl'>Threshold</div><div class='sv'>{float(last['best_threshold']):.2f}</div></div>\n"
        f"    <div class='stat'><div class='sl'>Val Loss</div><div class='sv'>{float(last['val_loss']):.4f}</div></div>\n"
        f"    <div class='stat'><div class='sl'>Precision</div><div class='sv'>{float(last['val_precision']):.3f}</div></div>\n"
        f"    <div class='stat'><div class='sl'>Recall</div><div class='sv'>{float(last['val_recall']):.3f}</div></div>\n"
        f"    <div class='stat'><div class='sl'>LR</div><div class='sv'>{float(last['lr']):.2e}</div></div>\n"
        "  </div>\n"
        "  <div class='charts'>\n"
        "    <div class='cb'><h3>Loss</h3><canvas id='lc'></canvas></div>\n"
        "    <div class='cb'><h3>F1 Score</h3><canvas id='fc'></canvas></div>\n"
        "    <div class='cb'><h3>Precision &amp; Recall</h3><canvas id='pc'></canvas></div>\n"
        "    <div class='cb'><h3>Best Threshold &amp; LR (&#xD7;1000)</h3><canvas id='tc'></canvas></div>\n"
        "  </div>\n"
        "  <table>\n"
        "    <thead><tr>\n"
        "      <th>Epoch</th><th>Train Loss</th><th>Train F1</th>\n"
        "      <th>Val Loss</th><th>Val F1@0.5</th><th>Best F1</th>\n"
        "      <th>Threshold</th><th>Precision</th><th>Recall</th><th>LR</th>\n"
        "    </tr></thead>\n"
        "    <tbody>\n"
        f"      {table_rows}\n"
        "    </tbody>\n"
        "  </table>\n"
        "  <script>\n"
        f"    const E={e_json},TL={tl_json},VL={vl_json};\n"
        f"    const TF={tf_json},VF={vf_json},BF={bf_json};\n"
        f"    const VP={vp_json},VR={vr_json},TH={th_json},LR={lr_json};\n"
        "    const O={responsive:true,maintainAspectRatio:true,animation:false,\n"
        "      plugins:{legend:{labels:{color:'#aaa',boxWidth:12,font:{size:11}}}},\n"
        "      scales:{x:{ticks:{color:'#666'},grid:{color:'#222'}},\n"
        "              y:{ticks:{color:'#666'},grid:{color:'#222'}}}};\n"
        "    const D=(l,d,c,dash=[])=>({label:l,data:d,borderColor:c,\n"
        "      backgroundColor:c+'22',borderWidth:2,pointRadius:3,tension:.3,\n"
        "      borderDash:dash,fill:false});\n"
        "    new Chart('lc',{type:'line',data:{labels:E,datasets:[\n"
        "      D('Train Loss',TL,'#7eb8f7'),D('Val Loss',VL,'#f97316')]},options:O});\n"
        "    new Chart('fc',{type:'line',data:{labels:E,datasets:[\n"
        "      D('Train F1',TF,'#7eb8f7'),D('Val F1@0.5',VF,'#f97316'),\n"
        "      D('Best F1',BF,'#5af78e')]},options:O});\n"
        "    new Chart('pc',{type:'line',data:{labels:E,datasets:[\n"
        "      D('Precision',VP,'#a78bfa'),D('Recall',VR,'#fb7185')]},options:O});\n"
        "    new Chart('tc',{type:'line',data:{labels:E,datasets:[\n"
        "      D('Threshold',TH,'#fbbf24'),D('LR x1000',LR,'#94a3b8',[5,5])]},options:O});\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    with open(html_path, "w") as f:
        f.write(html)


# ────────────────────────────────────────────────────────────────────────
# Main training driver
# ────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Complete training run: data → model → optimiser → loop → save."""

    device = get_device()
    print(f"[ChordNet] Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    if args.maestro:
        # Use preprocessed MAESTRO memory-mapped arrays.
        train_ds = MaestroDataset(args.data_dir, split="train", augment=True)
        val_ds = MaestroDataset(args.data_dir, split="val")
    else:
        # Use raw WAV + .npy label files.
        train_ds = ChordDataset(
            audio_dir=Path(args.data_dir) / "train" / "audio",
            label_dir=Path(args.data_dir) / "train" / "labels",
        )
        val_ds = ChordDataset(
            audio_dir=Path(args.data_dir) / "val" / "audio",
            label_dir=Path(args.data_dir) / "val" / "labels",
        )

    # pin_memory accelerates host→GPU copies on CUDA but is not
    # supported on MPS — disable it there to silence the warning.
    pin = device.type == "cuda"

    # With memory-mapped data, num_workers=0 is fastest — the main
    # process does a simple numpy array index (0.03 ms/sample) and
    # avoids fork+mmap issues on macOS.  Multi-worker loading only
    # helps if preprocessing is CPU-heavy (not our case).
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    print(f"[ChordNet] Train patches : {len(train_ds)}")
    print(f"[ChordNet] Val   patches : {len(val_ds)}")

    # ── Model / loss / optimiser ────────────────────────────────────────
    arch = args.arch
    model = build_model(arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ChordNet] Architecture  : {arch} ({n_params:,} params)")
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(args, train_ds, args.max_pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(
            "[ChordNet] Using pos_weight in BCEWithLogitsLoss "
            f"(min={pos_weight.min().item():.2f}, "
            f"mean={pos_weight.mean().item():.2f}, max={pos_weight.max().item():.2f})"
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning-rate scheduler: reduce on plateau of validation loss.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ── Checkpoint directory ────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_val_f1_opt = 0.0   # best F1 at optimal sigmoid threshold
    best_threshold = 0.5    # threshold for the saved best checkpoint
    start_epoch = 1

    # Architecture-specific checkpoint filenames for easy identification.
    best_ckpt_name = f"best_{arch}.pt"
    last_ckpt_name = f"last_{arch}.pt"

    # ── Resume from checkpoint ──────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        saved_arch = ckpt.get("arch", "chordnet")
        if saved_arch != arch:
            raise ValueError(
                f"Checkpoint was saved with arch='{saved_arch}' but "
                f"--arch='{arch}' was requested."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("val_f1")
        if best_val_f1 is None:
            best_path = ckpt_dir / best_ckpt_name
            if best_path.exists():
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
                best_val_f1 = float(best_ckpt.get("val_f1", 0.0))
            else:
                best_val_f1 = 0.0
        best_val_f1_opt = float(ckpt.get("val_f1_opt", best_val_f1))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"[ChordNet] Resumed from {resume_path} (epoch {start_epoch - 1}, best F1={best_val_f1:.4f}, threshold={best_threshold:.2f})")

    # Early-stopping bookkeeping.
    epochs_without_improvement = 0

    # ── CSV / HTML training log ───────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / f"{arch}_training.csv"
    html_path = log_dir / f"{arch}_training.html"
    _csv_mode = "a" if (args.resume and csv_path.exists()) else "w"
    _csv_file = open(csv_path, _csv_mode, newline="")  # noqa: SIM115
    _csv_fields = [
        "epoch", "train_loss", "train_f1",
        "val_loss", "val_f1", "val_precision", "val_recall",
        "best_threshold", "best_f1", "lr",
    ]
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
    if _csv_mode == "w":
        _csv_writer.writeheader()

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        should_validate = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_m = None
        # Carry forward previous best values on non-validation epochs.
        epoch_best_t  = best_threshold
        epoch_best_f1 = best_val_f1_opt
        if should_validate:
            val_m, val_logits, val_targets = validate(model, val_loader, criterion, device)
            scheduler.step(val_m["loss"])
            epoch_best_t, epoch_best_f1 = find_best_threshold(val_logits, val_targets)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        if val_m is None:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val=skipped  lr={current_lr:.2e}  ({elapsed:.1f}s)"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"val_P={val_m['precision']:.3f}  val_R={val_m['recall']:.3f}  "
                f"val_F1@0.5={val_m['f1']:.3f}  "
                f"best_F1={epoch_best_f1:.3f}@t={epoch_best_t:.2f}  "
                f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # ── Log to CSV and regenerate HTML dashboard ──────────────────
            _csv_writer.writerow({
                "epoch":          epoch,
                "train_loss":     f"{train_m['loss']:.6f}",
                "train_f1":       f"{train_m['f1']:.6f}",
                "val_loss":       f"{val_m['loss']:.6f}",
                "val_f1":         f"{val_m['f1']:.6f}",
                "val_precision":  f"{val_m['precision']:.6f}",
                "val_recall":     f"{val_m['recall']:.6f}",
                "best_threshold": f"{epoch_best_t:.2f}",
                "best_f1":        f"{epoch_best_f1:.6f}",
                "lr":             f"{current_lr:.8f}",
            })
            _csv_file.flush()
            write_html_log(csv_path, html_path, arch)

            # ── Save the best model by best-threshold F1 ──────────────────
            if epoch_best_f1 > best_val_f1_opt:
                best_val_f1_opt = epoch_best_f1
                best_threshold  = epoch_best_t
                best_val_f1     = val_m["f1"]
                epochs_without_improvement = 0
                best_path = ckpt_dir / best_ckpt_name
                torch.save(
                    {
                        "epoch":                epoch,
                        "arch":                 arch,
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_f1":               best_val_f1,
                        "val_f1_opt":           best_val_f1_opt,
                        "best_threshold":       best_threshold,
                    },
                    best_path,
                )
                print(
                    f"  ↳ Saved best model "
                    f"(F1={best_val_f1_opt:.4f}@t={best_threshold:.2f}) → {best_path}"
                )
            else:
                epochs_without_improvement += 1

        # Save rolling checkpoint every epoch for safe interruption/resume.
        rolling_path = ckpt_dir / last_ckpt_name
        torch.save(
            {
                "epoch":                epoch,
                "arch":                 arch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1":               best_val_f1,
                "val_f1_opt":           best_val_f1_opt,
                "best_threshold":       best_threshold,
            },
            rolling_path,
        )

        # ── Early stopping ──────────────────────────────────────────────
        if args.early_stop > 0 and epochs_without_improvement >= args.early_stop:
            print(
                f"[ChordNet] Early stopping: no best-threshold F1 improvement "
                f"for {args.early_stop} validated epochs."
            )
            break

    _csv_file.close()
    print(f"[ChordNet] Training complete. Final checkpoint → {rolling_path}")
    print(f"[ChordNet] Training log  → {csv_path}")
    print(f"[ChordNet] Dashboard     → {html_path}  (open in browser)")


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ChordNet")
    p.add_argument("--data-dir",       type=str,   default=str(CFG.data_dir))
    p.add_argument("--checkpoint-dir", type=str,   default=str(CFG.checkpoint_dir))
    p.add_argument("--epochs",         type=int,   default=CFG.epochs)
    p.add_argument("--batch-size",     type=int,   default=CFG.batch_size)
    p.add_argument("--lr",             type=float, default=CFG.learning_rate)
    p.add_argument("--weight-decay",   type=float, default=CFG.weight_decay)
    p.add_argument("--num-workers",    type=int,   default=CFG.num_workers)
    p.add_argument("--val-every",      type=int,   default=1,
                   help="Run validation every N epochs (always validates final epoch).")
    p.add_argument("--use-pos-weight", action="store_true",
                   help="Use per-note positive class weighting in BCEWithLogitsLoss.")
    p.add_argument("--max-pos-weight", type=float, default=20.0,
                   help="Upper bound for per-note pos_weight to avoid instability.")
    p.add_argument("--arch",           type=str,   default="chordnet",
                   choices=list(ARCH_REGISTRY),
                   help="Model architecture: chordnet (shallow CNN) or resnet.")
    p.add_argument("--maestro",        action="store_true",
                   help="Use preprocessed MAESTRO data (MaestroDataset).")
    p.add_argument("--resume",         type=str,   default=None,
                   help="Path to checkpoint (.pt) to resume training from.")
    p.add_argument("--early-stop",     type=int,   default=0,
                   help="Stop after N validated epochs without F1 improvement (0=disabled).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
