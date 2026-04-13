"""
model.py – ChordNet CNN for multi-label piano-note detection.

Architecture
------------
::

    Input  (1, 229, 32)
    ├─ ConvBlock 1  →  (32, 114, 16)
    ├─ ConvBlock 2  →  (64,  57,  8)
    ├─ ConvBlock 3  →  (128, 28,  4)
    ├─ ConvBlock 4  →  (256, 28,  4)   ← no pooling
    ├─ AdaptiveAvgPool2d(1,1) → (256,)
    └─ MLP head     → (88,)            ← raw logits
"""

import torch
import torch.nn as nn

from config import CFG


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (→ optional MaxPool2d).

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    pool : bool
        If ``True`` a 2×2 max-pool is appended.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=CFG.conv_kernel_size,
                padding=CFG.conv_padding,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChordNet(nn.Module):
    """4-block CNN for 88-key piano-note detection.

    Accepts a single-channel log-Mel spectrogram patch of shape
    ``(B, 1, 229, 32)`` and outputs raw logits of shape ``(B, 88)``.
    Apply ``torch.sigmoid`` during inference to obtain probabilities.
    """

    def __init__(self) -> None:
        super().__init__()

        ch = CFG.conv_channels  # [32, 64, 128, 256]

        # ── Convolutional feature extractor ─────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(1,     ch[0], pool=True),    # Block 1
            ConvBlock(ch[0], ch[1], pool=True),    # Block 2
            ConvBlock(ch[1], ch[2], pool=True),    # Block 3
            ConvBlock(ch[2], ch[3], pool=False),   # Block 4 – no pooling
        )

        # ── Global average pooling ──────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier MLP head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(ch[3], CFG.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.fc_hidden, CFG.n_notes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of spectrogram patches, shape ``(B, 1, 229, 32)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, 88)``.
        """
        x = self.features(x)       # (B, 256, H', W')
        x = self.gap(x)            # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.classifier(x)     # (B, 88)
        return x
