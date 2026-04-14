"""
DanoocNet — a convolutional neural network for image classification.

Architecture:
  3 convolutional blocks (conv → batchnorm → relu → maxpool)
  followed by a fully-connected classifier head.
"""
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DanoocNet(nn.Module):
    """
    Small CNN designed for 32×32 RGB images (e.g. CIFAR-10).

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout probability for the classifier head.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()

        self.features = nn.Sequential(
            _ConvBlock(3, 32),    # 32×32 → 16×16
            _ConvBlock(32, 64),   # 16×16 → 8×8
            _ConvBlock(64, 128),  # 8×8  → 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
