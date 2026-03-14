#include "common.cuh"

// =============================================================================
// SkipConnection: optional projection + optional adaptive_avg_pool2d + scale
// Forward: output = scale * projection(x) [with optional spatial pooling]
// The projection is either Linear or Conv2d(1x1), handled in Python wrapper.
// This kernel handles the scale multiplication.
// =============================================================================

// Simple scale kernel
template <typename scalar_t>
__global__ void skip_scale_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const float scale,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    output[idx] = from_float<scalar_t>(scale * to_float(x[idx]));
}

template <typename scalar_t>
__global__ void skip_scale_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_scale_contrib,
    const float scale,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float go = to_float(grad_output[idx]);
    float xi = to_float(x[idx]);
    grad_x[idx] = from_float<scalar_t>(scale * go);
    grad_scale_contrib[idx] = from_float<scalar_t>(go * xi);
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor skip_connection_forward(
    torch::Tensor x,
    float scale
) {
    ASANN_CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int total = x.numel();
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "skip_connection_forward", [&] {
        skip_scale_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scale,
            total
        );
    });
    ASANN_DEBUG_SYNC();

    return output;
}

std::vector<torch::Tensor> skip_connection_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    float scale
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    auto grad_x = torch::empty_like(x);
    auto grad_scale_contrib = torch::empty_like(x);

    int total = x.numel();
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "skip_connection_backward", [&] {
        skip_scale_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_scale_contrib.data_ptr<scalar_t>(),
            scale,
            total
        );
    });
    ASANN_DEBUG_SYNC();

    auto grad_scale = grad_scale_contrib.sum().unsqueeze(0);

    return {grad_x, grad_scale};
}
