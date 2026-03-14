import torch
import torch.nn as nn
import asann_cuda_ops


class PositionalEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, pos_emb):
        output = asann_cuda_ops.positional_embedding_forward(x, pos_emb)
        ctx.save_for_backward(x, pos_emb)
        ctx.batch_size = x.size(0)
        ctx.d = x.size(1)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_x, grad_pos_emb = asann_cuda_ops.positional_embedding_backward(
            grad_output, ctx.batch_size, ctx.d
        )
        return grad_x, grad_pos_emb


class PositionalEmbeddingCUDA(nn.Module):
    """Drop-in CUDA replacement for PositionalEmbedding."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.pos_emb = nn.Parameter(torch.randn(d) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PositionalEmbeddingFunction.apply(x, self.pos_emb)

    def extra_repr(self) -> str:
        return f"d={self.d}"
