import torch
import torch.nn as nn
import asann_cuda_ops


class Conv1dBlockFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, bias, kernel_size):
        output = asann_cuda_ops.conv1d_block_forward(x, weight, bias, kernel_size)
        ctx.save_for_backward(x, weight, bias)
        ctx.kernel_size = kernel_size
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.conv1d_block_backward(
            grad_output, x, weight, bias, ctx.kernel_size
        )
        return grads[0], grads[1], grads[2], None  # None for kernel_size


class Conv1dBlockCUDA(nn.Module):
    """Drop-in CUDA replacement for Conv1dBlock."""

    def __init__(self, d: int, kernel_size: int = 3):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size, padding=self.padding, bias=True,
        )
        # Near-identity init: center weight = 1.0, rest = small noise
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        self.conv.weight.data[0, 0, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Conv1dBlockFunction.apply(
            x, self.conv.weight, self.conv.bias, self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}"
