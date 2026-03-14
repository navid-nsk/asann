import torch
import torch.nn as nn
import asann_cuda_ops


class FactoredEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, U, V):
        output = asann_cuda_ops.factored_embedding_forward(x, U, V)
        ctx.save_for_backward(x, U, V)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, U, V = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x, grad_U, grad_V = asann_cuda_ops.factored_embedding_backward(
            grad_output, x, U, V
        )
        return grad_x, grad_U, grad_V


class FactoredEmbeddingCUDA(nn.Module):
    """Drop-in CUDA replacement for FactoredEmbedding."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rank = max(d // 4, 2)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.U = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.V = nn.Parameter(torch.randn(self.rank, d) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FactoredEmbeddingFunction.apply(x, self.U, self.V)

    def extra_repr(self) -> str:
        return f"d={self.d}, rank={self.rank}"
