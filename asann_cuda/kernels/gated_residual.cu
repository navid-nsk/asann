#include "common.cuh"

// =============================================================================
// Gated Residual: out = (1 - gate) * x + gate * transformed
// where gate = sigmoid(gate_logit)
//
// Used by: Conv2dBlock, GeometricEmbedding, SelfAttentionOp,
//          MultiHeadAttentionOp, CrossAttentionOp, CausalAttentionOp,
//          SpatialConv2dOp, PointwiseConv2dOp, DepthwiseSeparableConv2dOp,
//          ChannelAttentionOp
// =============================================================================

template <typename scalar_t>
__global__ void gated_residual_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ transformed,
    scalar_t* __restrict__ output,
    const float gate,       // pre-computed sigmoid(gate_logit), always FP32
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float xi = to_float(x[idx]);
    float ti = to_float(transformed[idx]);
    output[idx] = from_float<scalar_t>((1.0f - gate) * xi + gate * ti);
}

template <typename scalar_t>
__global__ void gated_residual_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ transformed,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_transformed,
    const float gate,       // pre-computed sigmoid(gate_logit)
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float go = to_float(grad_output[idx]);
    // d(out)/d(x) = (1 - gate)
    grad_x[idx] = from_float<scalar_t>((1.0f - gate) * go);
    // d(out)/d(transformed) = gate
    grad_transformed[idx] = from_float<scalar_t>(gate * go);
}

// Kernel to compute per-element contribution to grad_gate_logit
// grad_gate_logit = sum_i [ grad_output[i] * (transformed[i] - x[i]) * gate * (1-gate) ]
template <typename scalar_t>
__global__ void gated_residual_gate_grad_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ transformed,
    scalar_t* __restrict__ gate_grad_contrib,  // per-element contribution
    const float gate,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float dsigmoid = gate * (1.0f - gate);  // sigmoid derivative
    float go = to_float(grad_output[idx]);
    float ti = to_float(transformed[idx]);
    float xi = to_float(x[idx]);
    gate_grad_contrib[idx] = from_float<scalar_t>(go * (ti - xi) * dsigmoid);
}

// =============================================================================
// C++ Interface Functions (called from bindings)
// =============================================================================

torch::Tensor gated_residual_forward(
    torch::Tensor x,
    torch::Tensor transformed,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(transformed);
    TORCH_CHECK(x.sizes() == transformed.sizes(),
                "x and transformed must have same shape");

    // Under AMP, x (residual) and transformed (conv output) may have different dtypes
    // e.g., x is Float32 from previous stage, transformed is FP16 from autocast conv
    if (transformed.scalar_type() != x.scalar_type()) {
        transformed = transformed.to(x.scalar_type());
    }

    float gate = 1.0f / (1.0f + expf(-gate_logit.item<float>()));

    auto output = torch::empty_like(x);
    int total = x.numel();
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gated_residual_forward", [&] {
        gated_residual_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            transformed.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gate,
            total
        );
    });
    ASANN_DEBUG_SYNC();

    return output;
}

std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor transformed,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(transformed);

    // Ensure all tensors have matching dtype (same AMP mismatch as forward)
    auto target_dtype = x.scalar_type();
    if (grad_output.scalar_type() != target_dtype) {
        grad_output = grad_output.to(target_dtype);
    }
    if (transformed.scalar_type() != target_dtype) {
        transformed = transformed.to(target_dtype);
    }

    float gate = 1.0f / (1.0f + expf(-gate_logit.item<float>()));

    auto grad_x = torch::empty_like(x);
    auto grad_transformed = torch::empty_like(x);  // match x's dtype, not transformed's

    int total = x.numel();
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    // Compute grad_x and grad_transformed
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gated_residual_backward", [&] {
        gated_residual_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            transformed.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_transformed.data_ptr<scalar_t>(),
            gate,
            total
        );
    });
    ASANN_DEBUG_SYNC();

    // Compute grad_gate_logit via per-element contribution + sum
    auto gate_grad_contrib = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gated_residual_gate_grad", [&] {
        gated_residual_gate_grad_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            transformed.data_ptr<scalar_t>(),
            gate_grad_contrib.data_ptr<scalar_t>(),
            gate,
            total
        );
    });
    ASANN_DEBUG_SYNC();

    auto grad_gate_logit = gate_grad_contrib.sum().unsqueeze(0);

    return {grad_x, grad_transformed, grad_gate_logit};
}
