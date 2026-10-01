import torch
import torch.nn as nn
import asann_cuda_ops


class TemporalDiffFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, gate_logit):
        output = asann_cuda_ops.temporal_diff_forward(x, gate_logit)
        ctx.save_for_backward(x, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.temporal_diff_backward(grad_output, x, gate_logit)
        # grads: [grad_x, grad_gate_logit]
        return grads[0], grads[1]


class TemporalDiffCUDA(nn.Module):
    """Drop-in CUDA replacement for TemporalDiff."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TemporalDiffFunction.apply(x, self.gate_logit)

    def extra_repr(self) -> str:
        return f"d={self.d}"
