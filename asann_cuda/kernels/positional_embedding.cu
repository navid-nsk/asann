#include "common.cuh"

// =============================================================================
// PositionalEmbedding: out = x + pos_emb
// x: [B, d], pos_emb: [d]
// Forward: out[b][i] = x[b][i] + pos_emb[i]
// Backward: grad_x = grad_out, grad_pos_emb = sum_b(grad_out[b])
// =============================================================================

template <typename scalar_t>
__global__ void positional_embedding_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ pos_emb,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * d) return;

    int i = idx % d;  // feature index
    output[idx] = from_float<scalar_t>(to_float(x[idx]) + to_float(pos_emb[i]));
}

template <typename scalar_t>
__global__ void positional_embedding_backward_pos_emb_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_pos_emb,
    const int batch_size,
    const int d
) {
    // Each thread handles one feature dimension, sums across batch
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float sum = 0.0f;  // FP32 accumulator
    for (int b = 0; b < batch_size; b++) {
        sum += to_float(grad_output[b * d + i]);
    }
    grad_pos_emb[i] = from_float<scalar_t>(sum);
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor positional_embedding_forward(
    torch::Tensor x,
    torch::Tensor pos_emb
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(pos_emb);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");
    TORCH_CHECK(pos_emb.dim() == 1, "pos_emb must be 1D [d]");
    TORCH_CHECK(x.size(1) == pos_emb.size(0), "dimension mismatch");

    int B = x.size(0);
    int d = x.size(1);
    auto output = torch::empty_like(x);

    int total = B * d;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "positional_embedding_forward", [&] {
        positional_embedding_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            pos_emb.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    return output;
}

std::vector<torch::Tensor> positional_embedding_backward(
    torch::Tensor grad_output,
    int batch_size,
    int d
) {
    ASANN_CHECK_INPUT(grad_output);

    // grad_x = grad_output (pass-through)
    auto grad_x = grad_output.clone();

    // grad_pos_emb = sum across batch dimension
    auto grad_pos_emb = torch::zeros({d}, grad_output.options());

    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(d, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "positional_embedding_backward", [&] {
        positional_embedding_backward_pos_emb_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_pos_emb.data_ptr<scalar_t>(),
            batch_size, d
        );
    });
    ASANN_DEBUG_SYNC();

    return {grad_x, grad_pos_emb};
}
