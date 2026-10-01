import torch
import torch.nn as nn
import asann_cuda_ops


class DilatedConv1dFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, bias, gate_logit, kernel_size, dilation):
        output = asann_cuda_ops.dilated_conv1d_forward(
            x, weight, bias, gate_logit, kernel_size, dilation
        )
        ctx.save_for_backward(x, weight, bias, gate_logit)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, weight, bias, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.dilated_conv1d_backward(
            grad_output, x, weight, bias, gate_logit,
            ctx.kernel_size, ctx.dilation
        )
        # grads: [grad_x, grad_w, grad_b, grad_gate_logit]
        return grads[0], grads[1], grads[2], grads[3], None, None


class DilatedConv1dBlockCUDA(nn.Module):
    """Drop-in CUDA replacement for DilatedConv1dBlock."""

    def __init__(self, d: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size, padding=self.padding,
            dilation=dilation, bias=True,
        )
        # Near-identity init: center weight = 1.0, rest = small noise
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        self.conv.weight.data[0, 0, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DilatedConv1dFunction.apply(
            x, self.conv.weight, self.conv.bias, self.gate_logit,
            self.kernel_size, self.dilation
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}, dilation={self.dilation}"
