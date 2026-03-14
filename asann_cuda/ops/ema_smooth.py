import torch
import torch.nn as nn
import asann_cuda_ops


class EMASmoothFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, alpha_logit, gate_logit):
        output = asann_cuda_ops.ema_smooth_forward(x, alpha_logit, gate_logit)
        ctx.save_for_backward(x, alpha_logit, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, alpha_logit, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.ema_smooth_backward(
            grad_output, x, alpha_logit, gate_logit
        )
        # grads: [grad_x, grad_alpha_logit, grad_gate_logit]
        return grads[0], grads[1], grads[2]


class EMASmoothCUDA(nn.Module):
    """Drop-in CUDA replacement for EMASmooth."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.alpha_logit = nn.Parameter(torch.zeros(d))  # sigmoid(0) = 0.5
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return EMASmoothFunction.apply(x, self.alpha_logit, self.gate_logit)

    def extra_repr(self) -> str:
        return f"d={self.d}"
