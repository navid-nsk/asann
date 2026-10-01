import torch
import torch.nn as nn
import asann_cuda_ops


class GAPFlattenFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x):
        output = asann_cuda_ops.gap_flatten_forward(x)
        ctx.save_for_backward(x)
        ctx.shape = x.shape
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        B, C, H, W = ctx.shape
        grad_x = asann_cuda_ops.gap_flatten_backward(grad_output, B, C, H, W)
        return grad_x


class GAPFlattenCUDA(nn.Module):
    """Drop-in CUDA replacement for adaptive_avg_pool2d(x, 1).flatten(1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GAPFlattenFunction.apply(x)
