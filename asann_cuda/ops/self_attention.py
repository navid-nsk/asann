import torch
import torch.nn as nn
import asann_cuda_ops


class SelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, Q_emb, K_emb, V_emb, out_proj, gate_logit):
        results = asann_cuda_ops.self_attention_forward(
            x, Q_emb, K_emb, V_emb, out_proj, gate_logit
        )
        output = results[0]
        Q, K, V, attn, weighted, out = results[1], results[2], results[3], results[4], results[5], results[6]
        ctx.save_for_backward(x, Q, K, V, attn, weighted, out,
                              Q_emb, K_emb, V_emb, out_proj, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x, Q, K, V, attn, weighted, out,
         Q_emb, K_emb, V_emb, out_proj, gate_logit) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.self_attention_backward(
            grad_output, x, Q, K, V, attn, weighted, out,
            Q_emb, K_emb, V_emb, out_proj, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]


class SelfAttentionOpCUDA(nn.Module):
    """Drop-in CUDA replacement for SelfAttentionOp."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rank = max(d // 4, 4)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.K_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.V_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.out_proj = nn.Parameter(torch.randn(self.rank) * init_scale)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SelfAttentionFunction.apply(
            x, self.Q_emb, self.K_emb, self.V_emb,
            self.out_proj, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, rank={self.rank}"
