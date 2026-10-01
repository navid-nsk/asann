import torch
import torch.nn as nn
import asann_cuda_ops


class SpatialConv2dFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, conv_weight, conv_bias, gate_logit, kernel_size):
        results = asann_cuda_ops.spatial_conv2d_forward(
            x, conv_weight, conv_bias, gate_logit, kernel_size
        )
        output, conv_out = results[0], results[1]
        ctx.save_for_backward(x, conv_out, conv_weight, conv_bias, gate_logit)
        ctx.kernel_size = kernel_size
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, conv_out, conv_weight, conv_bias, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.spatial_conv2d_backward(
            grad_output, x, conv_out, conv_weight, conv_bias,
            gate_logit, ctx.kernel_size
        )
        return grads[0], grads[1], grads[2], grads[3], None


class SpatialConv2dOpCUDA(nn.Module):
    """Drop-in CUDA replacement for SpatialConv2dOp."""

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                              groups=C, bias=True)
        # Near-identity init
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        for c in range(C):
            self.conv.weight.data[c, 0, center, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SpatialConv2dFunction.apply(
            x, self.conv.weight, self.conv.bias,
            self.gate_logit, self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, kernel_size={self.kernel_size}"
