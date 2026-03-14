#include "common.cuh"

// =============================================================================
// GatedLinearUnit: GLU(x) = sigmoid(Wx+b) * (Vx+c) with outer gated residual
// out = (1-outer_gate)*x + outer_gate * [sigmoid(gate_proj(x)) * value_proj(x)]
// Uses at::linear for projections (cuBLAS), custom kernel for sigmoid*elementwise
// =============================================================================

// Forward declarations from gated_residual.cu
torch::Tensor gated_residual_forward(
    torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor transformed, torch::Tensor gate_logit);

template <typename scalar_t>
__global__ void glu_elementwise_forward_kernel(
    const scalar_t* __restrict__ gate_linear,   // sigmoid input (Wx+b)
    const scalar_t* __restrict__ value_linear,  // Vx+c
    scalar_t* __restrict__ glu_out,
    scalar_t* __restrict__ gate_sigmoid,  // save for backward
    const int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float g = asann_sigmoid(to_float(gate_linear[idx]));
    gate_sigmoid[idx] = from_float<scalar_t>(g);
    glu_out[idx] = from_float<scalar_t>(g * to_float(value_linear[idx]));
}

template <typename scalar_t>
__global__ void glu_elementwise_backward_kernel(
    const scalar_t* __restrict__ grad_glu_out,
    const scalar_t* __restrict__ gate_sigmoid,  // saved from forward
    const scalar_t* __restrict__ value_linear,  // Vx+c
    const scalar_t* __restrict__ gate_linear,   // Wx+b (for dsigmoid)
    scalar_t* __restrict__ grad_gate_linear,
    scalar_t* __restrict__ grad_value_linear,
    const int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float go = to_float(grad_glu_out[idx]);
    float g = to_float(gate_sigmoid[idx]);
    float v = to_float(value_linear[idx]);

    // glu = sigmoid(gate) * value
    // d/d(value) = sigmoid(gate)
    grad_value_linear[idx] = from_float<scalar_t>(go * g);
    // d/d(gate_linear) = value * sigmoid(gate) * (1 - sigmoid(gate))
    grad_gate_linear[idx] = from_float<scalar_t>(go * v * g * (1.0f - g));
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor gated_linear_unit_forward(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor value_weight,
    torch::Tensor value_bias,
    torch::Tensor outer_gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(gate_weight);
    ASANN_CHECK_INPUT(gate_bias);
    ASANN_CHECK_INPUT(value_weight);
    ASANN_CHECK_INPUT(value_bias);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int total = x.numel();

    // Linear projections: gate_linear = Wx+b, value_linear = Vx+c
    auto gate_linear = at::linear(x, gate_weight, gate_bias);
    auto value_linear = at::linear(x, value_weight, value_bias);

    // GLU elementwise: glu_out = sigmoid(gate_linear) * value_linear
    auto glu_out = torch::empty_like(x);
    auto gate_sigmoid = torch::empty_like(x);  // save for backward

    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "glu_elementwise_forward", [&] {
        glu_elementwise_forward_kernel<scalar_t><<<blocks, threads>>>(
            gate_linear.data_ptr<scalar_t>(),
            value_linear.data_ptr<scalar_t>(),
            glu_out.data_ptr<scalar_t>(),
            gate_sigmoid.data_ptr<scalar_t>(),
            total
        );
    });
    ASANN_DEBUG_SYNC();

    // Outer gated residual: (1-gate)*x + gate*glu_out
    return gated_residual_forward(x, glu_out, outer_gate_logit);
}

std::vector<torch::Tensor> gated_linear_unit_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor value_weight,
    torch::Tensor value_bias,
    torch::Tensor outer_gate_logit,
    torch::Tensor gate_sigmoid,
    torch::Tensor gate_linear,
    torch::Tensor value_linear
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int total = x.numel();
    int B = x.size(0);
    int d = x.size(1);

    // Recompute glu_out for gated residual backward
    auto glu_out = gate_sigmoid * value_linear;

    // Backward through outer gated residual
    auto gr_grads = gated_residual_backward(grad_output, x, glu_out, outer_gate_logit);
    auto grad_x_residual = gr_grads[0];
    auto grad_glu_out = gr_grads[1];
    auto grad_outer_gate = gr_grads[2];

    // Backward through GLU elementwise
    auto grad_gate_linear = torch::empty_like(x);
    auto grad_value_linear = torch::empty_like(x);

    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "glu_elementwise_backward", [&] {
        glu_elementwise_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_glu_out.data_ptr<scalar_t>(),
            gate_sigmoid.data_ptr<scalar_t>(),
            value_linear.data_ptr<scalar_t>(),
            gate_linear.data_ptr<scalar_t>(),
            grad_gate_linear.data_ptr<scalar_t>(),
            grad_value_linear.data_ptr<scalar_t>(),
            total
        );
    });
    ASANN_DEBUG_SYNC();

    // Backward through linear projections
    // Cast weights to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto gw_bwd = gate_weight.to(act_dtype);
    auto vw_bwd = value_weight.to(act_dtype);

    // gate_linear = x @ gate_weight.T + gate_bias
    auto grad_x_gate = at::mm(grad_gate_linear, gw_bwd);        // [B,d] @ [d,d]
    auto grad_gate_w = at::mm(grad_gate_linear.t(), x);          // [d,B] @ [B,d] = [d,d]
    auto grad_gate_b = grad_gate_linear.sum(0);                  // [d]

    // value_linear = x @ value_weight.T + value_bias
    auto grad_x_value = at::mm(grad_value_linear, vw_bwd);      // [B,d] @ [d,d]
    auto grad_value_w = at::mm(grad_value_linear.t(), x);        // [d,d]
    auto grad_value_b = grad_value_linear.sum(0);                // [d]

    // Total grad_x
    auto grad_x = grad_x_residual + grad_x_gate + grad_x_value;

    return {grad_x, grad_gate_w, grad_gate_b, grad_value_w, grad_value_b, grad_outer_gate};
}
