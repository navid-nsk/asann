import torch
import torch.nn as nn
import asann_cuda_ops


class GeometricEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, radial_scale, bias, gate_logit):
        results = asann_cuda_ops.geometric_embedding_forward(
            x, radial_scale, bias, gate_logit
        )
        output, scaled, norm, poincare = results[0], results[1], results[2], results[3]
        ctx.save_for_backward(x, scaled, norm, poincare, radial_scale, bias, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, scaled, norm, poincare, radial_scale, bias, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.geometric_embedding_backward(
            grad_output, x, scaled, norm, poincare,
            radial_scale, bias, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3]


class GeometricEmbeddingCUDA(nn.Module):
    """Drop-in CUDA replacement for GeometricEmbedding."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.radial_scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GeometricEmbeddingFunction.apply(
            x, self.radial_scale, self.bias, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"d={self.d}"
