#include "common.cuh"

// =============================================================================
// GAP + Flatten: adaptive_avg_pool2d(x, (1,1)).flatten(1)
// x: [B, C, H, W] -> output: [B, C]
// Forward: out[b][c] = mean(x[b][c][h][w]) for all h,w
// Backward: grad_x[b][c][h][w] = grad_out[b][c] / (H * W)
// =============================================================================

template <typename scalar_t>
__global__ void gap_flatten_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int C,
    const int H,
    const int W
) {
    // Each thread computes one output element: out[b][c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * C) return;

    int b = idx / C;
    int c = idx % C;
    int HW = H * W;

    float sum = 0.0f;  // FP32 accumulator for precision
    int base = b * C * HW + c * HW;
    for (int hw = 0; hw < HW; hw++) {
        sum += to_float(x[base + hw]);
    }
    output[idx] = from_float<scalar_t>(sum / (float)HW);
}

template <typename scalar_t>
__global__ void gap_flatten_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_x,
    const int batch_size,
    const int C,
    const int H,
    const int W
) {
    // Each thread handles one input element: grad_x[b][c][h][w]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * C * H * W;
    if (idx >= total) return;

    int HW = H * W;
    int b = idx / (C * HW);
    int c = (idx / HW) % C;

    float inv_hw = 1.0f / (float)HW;
    grad_x[idx] = from_float<scalar_t>(to_float(grad_output[b * C + c]) * inv_hw);
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor gap_flatten_forward(
    torch::Tensor x
) {
    ASANN_CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({B, C}, x.options());

    int total = B * C;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gap_flatten_forward", [&] {
        gap_flatten_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, C, H, W
        );
    });
    ASANN_DEBUG_SYNC();

    return output;
}

torch::Tensor gap_flatten_backward(
    torch::Tensor grad_output,
    int B, int C, int H, int W
) {
    ASANN_CHECK_INPUT(grad_output);

    auto grad_x = torch::empty({B, C, H, W}, grad_output.options());

    int total = B * C * H * W;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "gap_flatten_backward", [&] {
        gap_flatten_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            B, C, H, W
        );
    });
    ASANN_DEBUG_SYNC();

    return grad_x;
}
