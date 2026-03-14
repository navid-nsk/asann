import torch
import torch.nn as nn
import asann_cuda_ops


class MLPEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, encode_weight, encode_bias, decode_weight, decode_bias):
        results = asann_cuda_ops.mlp_embedding_forward(
            x, encode_weight, encode_bias, decode_weight, decode_bias
        )
        output, encoded, hidden_act = results[0], results[1], results[2]
        ctx.save_for_backward(x, encoded, hidden_act,
                              encode_weight, encode_bias,
                              decode_weight, decode_bias)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x, encoded, hidden_act,
         encode_weight, encode_bias,
         decode_weight, decode_bias) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.mlp_embedding_backward(
            grad_output, x, encoded, hidden_act,
            encode_weight, encode_bias, decode_weight, decode_bias
        )
        grad_x = grads[0]
        grad_encode_weight = grads[1]
        grad_encode_bias = grads[2]
        grad_decode_weight = grads[3]
        grad_decode_bias = grads[4]
        return grad_x, grad_encode_weight, grad_encode_bias, grad_decode_weight, grad_decode_bias


class MLPEmbeddingCUDA(nn.Module):
    """Drop-in CUDA replacement for MLPEmbedding."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.hidden = max(d // 2, 4)
        self.encode = nn.Linear(d, self.hidden)
        self.decode = nn.Linear(self.hidden, d)
        init_scale = 0.1 / (self.hidden ** 0.5)
        nn.init.normal_(self.encode.weight, std=init_scale)
        nn.init.zeros_(self.encode.bias)
        nn.init.normal_(self.decode.weight, std=init_scale)
        nn.init.zeros_(self.decode.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MLPEmbeddingFunction.apply(
            x, self.encode.weight, self.encode.bias,
            self.decode.weight, self.decode.bias
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, hidden={self.hidden}"
