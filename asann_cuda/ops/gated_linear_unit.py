import torch
import torch.nn as nn
import asann_cuda_ops


class GatedLinearUnitFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, gate_weight, gate_bias, value_weight, value_bias, outer_gate_logit):
        output = asann_cuda_ops.gated_linear_unit_forward(
            x, gate_weight, gate_bias, value_weight, value_bias, outer_gate_logit
        )
        # We need to save intermediates for backward.
        # The CUDA backward recomputes some, but needs gate_sigmoid, gate_linear, value_linear.
        # Recompute them here to save for backward (matching the CUDA kernel's expectations).
        gate_linear = torch.nn.functional.linear(x, gate_weight, gate_bias)
        value_linear = torch.nn.functional.linear(x, value_weight, value_bias)
        gate_sigmoid = torch.sigmoid(gate_linear)
        ctx.save_for_backward(
            x, gate_weight, gate_bias, value_weight, value_bias,
            outer_gate_logit, gate_sigmoid, gate_linear, value_linear
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (x, gate_weight, gate_bias, value_weight, value_bias,
         outer_gate_logit, gate_sigmoid, gate_linear, value_linear) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = asann_cuda_ops.gated_linear_unit_backward(
            grad_output, x, gate_weight, gate_bias, value_weight, value_bias,
            outer_gate_logit, gate_sigmoid, gate_linear, value_linear
        )
        # grads: [grad_x, grad_gate_w, grad_gate_b, grad_value_w, grad_value_b, grad_outer_gate]
        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]


class GatedLinearUnitCUDA(nn.Module):
    """Drop-in CUDA replacement for GatedLinearUnit."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.gate_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d)
        # Near-identity init for value path, small init for gate path
        nn.init.eye_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.zeros_(self.gate_proj.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GatedLinearUnitFunction.apply(
            x, self.gate_proj.weight, self.gate_proj.bias,
            self.value_proj.weight, self.value_proj.bias,
            self.gate_logit
        )

    def extra_repr(self) -> str:
        return f"d={self.d}"
