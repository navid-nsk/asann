import torch
import torch.nn as nn
import asann_cuda_ops


class ChannelAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_logit):
        results = asann_cuda_ops.channel_attention_forward(
            x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_logit
        )
        output = results[0]
        squeezed, fc1_out, relu_out = results[1], results[2], results[3]
        fc2_out, attn_weights, rescaled = results[4], results[5], results[6]
        ctx.save_for_backward(x, squeezed, fc1_out, relu_out, fc2_out,
                              attn_weights, rescaled,
                              fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_logit)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x, squeezed, fc1_out, relu_out, fc2_out,
         attn_weights, rescaled,
         fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_logit) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.channel_attention_backward(
            grad_output, x, squeezed, fc1_out, relu_out, fc2_out,
            attn_weights, rescaled,
            fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_logit
        )
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]


class ChannelAttentionOpCUDA(nn.Module):
    """Drop-in CUDA replacement for ChannelAttentionOp (SE-block)."""

    def __init__(self, C: int, reduction_ratio: int = 4):
        super().__init__()
        self.C = C
        self.reduction_ratio = reduction_ratio
        self.reduction = max(C // reduction_ratio, 2)
        self.fc1 = nn.Linear(C, self.reduction)
        self.fc2 = nn.Linear(self.reduction, C)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ChannelAttentionFunction.apply(
            x, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias, self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"C={self.C}, reduction={self.reduction}, ratio={self.reduction_ratio}"
