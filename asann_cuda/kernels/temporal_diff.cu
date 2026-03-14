#include "common.cuh"

// =============================================================================
// TemporalDiff: First-order difference with gated residual
// diff[b][0] = 0, diff[b][i] = x[b][i] - x[b][i-1] for i > 0
// out = (1-gate)*x + gate*diff
// =============================================================================

// Forward declarations from gated_residual.cu
torch::Tensor gated_residual_forward(
    torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor transformed, torch::Tensor gate_logit);

template <typename scalar_t>
__global__ void temporal_diff_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ diff,
    const int B,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * d;
    if (idx >= total) return;

    int i = idx % d;

    if (i == 0) {
        diff[idx] = from_float<scalar_t>(0.0f);
    } else {
        diff[idx] = from_float<scalar_t>(to_float(x[idx]) - to_float(x[idx - 1]));
    }
}

template <typename scalar_t>
__global__ void temporal_diff_backward_kernel(
    const scalar_t* __restrict__ grad_diff,
    scalar_t* __restrict__ grad_x,
    const int B,
    const int d
) {
    // diff[b][i] = x[b][i] - x[b][i-1] for i > 0, diff[b][0] = 0
    // d(diff[i])/d(x[i]) = 1 for i > 0
    // d(diff[i+1])/d(x[i]) = -1 for i < d-1
    // So grad_x[b][i] = grad_diff[b][i] (from diff[i]) - grad_diff[b][i+1] (from diff[i+1])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * d;
    if (idx >= total) return;

    int i = idx % d;

    float g = 0.0f;
    if (i > 0) {
        g += to_float(grad_diff[idx]);  // from diff[i] = x[i] - x[i-1], d/dx[i] = 1
    }
    if (i < d - 1) {
        g -= to_float(grad_diff[idx + 1]);  // from diff[i+1] = x[i+1] - x[i], d/dx[i] = -1
    }
    grad_x[idx] = from_float<scalar_t>(g);
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor temporal_diff_forward(
    torch::Tensor x,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);
    auto diff = torch::empty_like(x);

    int total = B * d;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "temporal_diff_forward", [&] {
        temporal_diff_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            diff.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    return gated_residual_forward(x, diff, gate_logit);
}

std::vector<torch::Tensor> temporal_diff_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int d = x.size(1);

    // Recompute diff for gated residual backward
    auto diff = torch::empty_like(x);
    int total = B * d;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "temporal_diff_recompute", [&] {
        temporal_diff_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            diff.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    // Backward through gated residual
    auto gr_grads = gated_residual_backward(grad_output, x, diff, gate_logit);
    auto grad_x_residual = gr_grads[0];
    auto grad_diff = gr_grads[1];
    auto grad_gate_logit = gr_grads[2];

    // Backward through temporal diff
    auto grad_x_diff = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "temporal_diff_backward", [&] {
        temporal_diff_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_diff.data_ptr<scalar_t>(),
            grad_x_diff.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    auto grad_x = grad_x_residual + grad_x_diff;

    return {grad_x, grad_gate_logit};
}
