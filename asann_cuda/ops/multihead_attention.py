import torch
import torch.nn as nn
import asann_cuda_ops


class MultiHeadAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, Q_emb, K_emb, V_emb, out_proj, head_weights, gate_logit):
        results = asann_cuda_ops.multihead_attention_forward(
            x, Q_emb, K_emb, V_emb, out_proj, head_weights, gate_logit
        )
        output = results[0]
        Q, K, V = results[1], results[2], results[3]
        attn_4d, weighted_4d = results[4], results[5]
        head_out, hw, out = results[6], results[7], results[8]
        ctx.save_for_backward(x, Q, K, V, attn_4d, weighted_4d,
                              head_out, hw, out,
                              Q_emb, K_emb, V_emb, out_proj, head_weights, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x, Q, K, V, attn_4d, weighted_4d,
         head_out, hw, out,
         Q_emb, K_emb, V_emb, out_proj, head_weights, gate_logit) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.multihead_attention_backward(
            grad_output, x, Q, K, V, attn_4d, weighted_4d,
            head_out, hw, out,
            Q_emb, K_emb, V_emb, out_proj, head_weights, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6]


class MultiHeadAttentionOpCUDA(nn.Module):
    """Drop-in CUDA replacement for MultiHeadAttentionOp."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.num_heads = max(d // 8, 2)
        self.head_rank = max(d // (self.num_heads * 2), 2)
        init_scale = 0.1 / (self.head_rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        self.K_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        self.V_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        self.out_proj = nn.Parameter(torch.randn(self.num_heads, self.head_rank) * init_scale)
        self.head_weights = nn.Parameter(torch.zeros(self.num_heads))
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MultiHeadAttentionFunction.apply(
            x, self.Q_emb, self.K_emb, self.V_emb,
            self.out_proj, self.head_weights, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, heads={self.num_heads}, head_rank={self.head_rank}"
