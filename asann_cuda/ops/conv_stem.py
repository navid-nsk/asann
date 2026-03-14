import torch
import torch.nn as nn
from typing import Tuple


class ConvStemCUDA(nn.Module):
    """Drop-in CUDA replacement for ConvStem.

    Uses standard PyTorch ops (Conv2d, BN, ReLU) which already have CUDA support.
    Conv2d(C_in, C_stem, 3, padding=1) -> BN -> ReLU
    """

    def __init__(self, C_in: int, C_stem: int, H: int, W: int):
        super().__init__()
        self.C_in = C_in
        self.C_stem = C_stem
        self.H = H
        self.W = W
        self.conv = nn.Conv2d(C_in, C_stem, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_stem)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    @property
    def out_features(self) -> int:
        return self.C_stem * self.H * self.W

    @property
    def in_features(self) -> int:
        return self.C_in * self.H * self.W

    @property
    def out_channels(self) -> int:
        return self.C_stem

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return (self.C_stem, self.H, self.W)
