import torch
import torch.nn as nn
import asann_cuda_ops


class GatedResidualFunction(torch.autograd.Function):
    """Autograd function for gated residual: (1-gate)*x + gate*transformed."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, transformed, gate_logit):
        output = asann_cuda_ops.gated_residual_forward(x, transformed, gate_logit)
        ctx.save_for_backward(x, transformed, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, transformed, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_transformed, grad_gate_logit = asann_cuda_ops.gated_residual_backward(
            grad_output, x, transformed, gate_logit
        )
        return grad_x, grad_transformed, grad_gate_logit
