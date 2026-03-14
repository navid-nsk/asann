import torch
import torch.nn as nn
import asann_cuda_ops


class CrossAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, Q_emb, mem_keys, mem_values, gate_logit):
        results = asann_cuda_ops.cross_attention_forward(
            x, Q_emb, mem_keys, mem_values, gate_logit
        )
        output, Q, attn, out = results[0], results[1], results[2], results[3]
        ctx.save_for_backward(x, Q, attn, out, Q_emb, mem_keys, mem_values, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, Q, attn, out, Q_emb, mem_keys, mem_values, gate_logit = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.cross_attention_backward(
            grad_output, x, Q, attn, out,
            Q_emb, mem_keys, mem_values, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3], grads[4]


class CrossAttentionOpCUDA(nn.Module):
    """Drop-in CUDA replacement for CrossAttentionOp."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.num_memories = max(d // 4, 4)
        self.rank = max(d // 4, 4)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.mem_keys = nn.Parameter(torch.randn(self.num_memories, self.rank) * init_scale)
        self.mem_values = nn.Parameter(torch.randn(self.num_memories, d) * init_scale)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CrossAttentionFunction.apply(
            x, self.Q_emb, self.mem_keys, self.mem_values, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, memories={self.num_memories}, rank={self.rank}"
