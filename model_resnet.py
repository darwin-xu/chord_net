"""
model_resnet.py – ResNet-style CNN for multi-label piano-note detection.

Architecture
------------
::

    Input  (1, 229, 32)
    ├─ Stem Conv 3×3      →  (32, 229, 32)
    ├─ ResStage 1 (×2)    →  (32, 115, 16)    ← stride-2 downsample
    ├─ ResStage 2 (×2)    →  (64, 58, 8)
    ├─ ResStage 3 (×2)    →  (128, 29, 4)
    ├─ ResStage 4 (×2)    →  (256, 15, 2)
    ├─ AdaptiveAvgPool2d(1,1) → (256,)
    ├─ Dropout(0.3)
    └─ Linear(256, 88)    → raw logits

Total: ~1.5 M parameters (vs 433 K for ChordNet).
"""

import torch
import torch.nn as nn

from config import CFG


class ResBlock(nn.Module):
    """Pre-activation residual block (BN → ReLU → Conv) × 2.

    When ``downsample=True`` the first conv uses stride 2 and a 1×1
    projection shortcut matches dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False) -> None:
        super().__init__()
        stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if downsample or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class ChordResNet(nn.Module):
    """ResNet-style model for 88-key piano-note detection.

    Accepts a single-channel log-Mel spectrogram patch of shape
    ``(B, 1, 229, 32)`` and outputs raw logits of shape ``(B, 88)``.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()

        # ── Stem ────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Residual stages ─────────────────────────────────────────────
        stages: list[nn.Module] = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            for b in range(blocks_per_stage):
                downsample = (b == 0 and i > 0)  # first block of stages 2–4
                stages.append(ResBlock(in_ch, out_ch, downsample=downsample))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # ── Head ────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(CFG.dropout),
            nn.Linear(channels[-1], CFG.n_notes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
