import torch
import torch.nn as nn
import torch.nn.functional as F
import asann_cuda_ops
from .gated_residual import GatedResidualFunction


class DepthwiseSeparableConv2dOpCUDA(nn.Module):
    """Drop-in CUDA replacement for DepthwiseSeparableConv2dOp.

    Uses a hybrid approach: CUDA kernels for the gated residual,
    PyTorch's native ops for dw_conv -> BN -> ReLU -> pw_conv
    (since BN backward requires save_mean/save_invstd from forward).
    """

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise: each channel independently
        self.dw_conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                                 groups=C, bias=False)
        self.bn = nn.BatchNorm2d(C)
        # Pointwise: cross-channel mixing (1x1 conv)
        self.pw_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)

        # Near-identity init for depthwise: center weight = 1.0
        nn.init.zeros_(self.dw_conv.weight)
        center = kernel_size // 2
        for c in range(C):
            self.dw_conv.weight.data[c, 0, center, center] = 1.0
        self.dw_conv.weight.data += 0.01 * torch.randn_like(self.dw_conv.weight.data)

        # Near-identity init for pointwise: I + noise
        nn.init.eye_(self.pw_conv.weight.data.view(C, C))
        self.pw_conv.weight.data += 0.01 * torch.randn_like(self.pw_conv.weight.data)
        nn.init.zeros_(self.pw_conv.bias)

        # Gated residual
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's native ops for the pipeline (BN needs autograd support)
        out = self.dw_conv(x)
        out = F.relu(self.bn(out))
        out = self.pw_conv(out)
        # Use CUDA gated residual
        return GatedResidualFunction.apply(
            x.reshape(x.size(0), -1),
            out.reshape(x.size(0), -1),
            self.gate_logit
        ).view_as(x)

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, kernel_size={self.kernel_size}"
