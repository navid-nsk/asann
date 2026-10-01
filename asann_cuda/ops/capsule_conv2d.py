"""CUDA-accelerated Capsule Conv2d operation.

Uses custom CUDA kernel for fused squash activation, PyTorch native ops
for depthwise conv and pointwise conv (cuDNN-accelerated), and CUDA
GatedResidualFunction for the residual blending.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import asann_cuda_ops
from .gated_residual import GatedResidualFunction


class CapsuleSquashFunction(torch.autograd.Function):
    """Autograd function for fused capsule squash activation."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, cap_dim, eps):
        results = asann_cuda_ops.capsule_squash_forward(
            input.contiguous(), cap_dim, eps)
        output = results[0]
        ctx.save_for_backward(input)
        ctx.cap_dim = cap_dim
        ctx.eps = eps
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grads = asann_cuda_ops.capsule_squash_backward(
            grad_output.contiguous(), input.contiguous(),
            ctx.cap_dim, ctx.eps)
        return grads[0], None, None


class CapsuleConv2dOpCUDA(nn.Module):
    """Capsule-aware Depthwise Separable Conv2d with CUDA squash activation.

    Uses CUDA kernel for fused squash, PyTorch native ops for depthwise
    and pointwise convolutions, and CUDA GatedResidualFunction for blending.

    Squash: v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)
    """

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3, cap_dim: int = 4):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        self.cap_dim = min(cap_dim, C) if C >= 2 else 1
        if self.cap_dim < 2:
            self.cap_dim = 1

        self.C_padded = ((C + self.cap_dim - 1) // self.cap_dim) * self.cap_dim
        self.needs_padding = (self.C_padded != C)
        self.num_capsules = self.C_padded // self.cap_dim

        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                                 groups=C, bias=False)
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

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))
        self._squash_eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Step 1: Depthwise conv
        out = self.dw_conv(x)

        # Step 2: CUDA squash activation
        if self.needs_padding:
            pad_c = self.C_padded - C
            out = F.pad(out, (0, 0, 0, 0, 0, pad_c))

        out = CapsuleSquashFunction.apply(out, self.cap_dim, self._squash_eps)

        if self.needs_padding:
            out = out[:, :C, :, :]

        # Step 3: Pointwise conv
        out = self.pw_conv(out)

        # CUDA gated residual
        return GatedResidualFunction.apply(
            x.reshape(B, -1),
            out.reshape(B, -1),
            self.gate_logit
        ).view_as(x)

    def extra_repr(self) -> str:
        return (f"C={self.C}, H={self.H}, W={self.W}, "
                f"kernel_size={self.kernel_size}, cap_dim={self.cap_dim}")
