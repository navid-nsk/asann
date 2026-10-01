import torch
import torch.nn as nn
import asann_cuda_ops


class PointwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, conv_weight, conv_bias, gate_logit):
        results = asann_cuda_ops.pointwise_conv2d_forward(
            x, conv_weight, conv_bias, gate_logit
        )
        output, conv_out = results[0], results[1]
        ctx.save_for_backward(x, conv_out, conv_weight, conv_bias, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, conv_out, conv_weight, conv_bias, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.pointwise_conv2d_backward(
            grad_output, x, conv_out, conv_weight, conv_bias, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3]


class PointwiseConv2dOpCUDA(nn.Module):
    """Drop-in CUDA replacement for PointwiseConv2dOp."""

    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        # Near-identity: start as identity mapping + small noise
        nn.init.eye_(self.conv.weight.data.view(C, C))
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PointwiseConv2dFunction.apply(
            x, self.conv.weight, self.conv.bias, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}"
