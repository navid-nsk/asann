import torch
import torch.nn as nn
import asann_cuda_ops


class Conv2dBlockFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, conv_weight, conv_bias, gate_logit, C, H, W, kernel_size):
        results = asann_cuda_ops.conv2d_block_forward(
            x, conv_weight, conv_bias, gate_logit, C, H, W, kernel_size
        )
        output, img, conv_out, transformed = results[0], results[1], results[2], results[3]
        ctx.save_for_backward(x, img, conv_out, transformed, conv_weight, conv_bias, gate_logit)
        ctx.C = C
        ctx.H = H
        ctx.W = W
        ctx.kernel_size = kernel_size
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, img, conv_out, transformed, conv_weight, conv_bias, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.conv2d_block_backward(
            grad_output, x, img, conv_out, transformed,
            conv_weight, conv_bias, gate_logit,
            ctx.C, ctx.H, ctx.W, ctx.kernel_size
        )
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None


class Conv2dBlockCUDA(nn.Module):
    """Drop-in CUDA replacement for Conv2dBlock."""

    def __init__(self, d: int, kernel_size: int = 3, spatial_shape=None):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size

        if spatial_shape and d == spatial_shape[0] * spatial_shape[1] * spatial_shape[2]:
            self.C, self.H, self.W = spatial_shape
        else:
            self.C = 1
            side = int(d ** 0.5)
            while side > 1 and d % side != 0:
                side -= 1
            self.H = side
            self.W = d // side

        padding = kernel_size // 2
        self.conv = nn.Conv2d(self.C, self.C, kernel_size, padding=padding,
                              groups=self.C, bias=True)
        # Near-identity init
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        for c in range(self.C):
            self.conv.weight.data[c, 0, center, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Conv2dBlockFunction.apply(
            x, self.conv.weight, self.conv.bias, self.gate_logit,
            self.C, self.H, self.W, self.kernel_size
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}, spatial=({self.C},{self.H},{self.W})"
