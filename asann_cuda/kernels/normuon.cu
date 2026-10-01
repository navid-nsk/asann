#include "common.cuh"

// =============================================================================
// NorMuon: Per-neuron gradient normalization
//
// 4D (Conv2d) [C_out, C_in, k, k]:
//   - Compute L2 norm of each filter (row = flatten over C_in*k*k)
//   - Compute mean norm across all filters
//   - Normalize: grad[i] = grad[i] / max(row_norm[i], eps) * mean_norm
//
// 2D (Linear) [d_out, d_in] where d_out >= 128:
//   - Same logic but rows are d_in-dimensional
//
// Two-phase design:
//   Phase 1: One thread block per row → compute L2 norm per row
//            (FP32 accumulator for precision, works with FP16 inputs)
//   Phase 2: Element-wise normalize using row norms and mean norm
// =============================================================================

// Phase 1: Compute L2 norm of each row using warp-shuffle + shared memory
// Always accumulates in FP32 for numerical stability
template <typename scalar_t>
__global__ void normuon_compute_norms_kernel(
    const scalar_t* __restrict__ grad,
    float* __restrict__ row_norms,  // always FP32
    const int num_rows,
    const int row_len
) {
    // One block per row
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const scalar_t* row_ptr = grad + row * row_len;

    // Each thread accumulates partial sum of squares in FP32
    float partial_sum = 0.0f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float val = to_float(row_ptr[i]);
        partial_sum += val * val;
    }

    // Warp-level reduction
    for (int offset = ASANN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }

    // Shared memory for inter-warp reduction
    __shared__ float warp_sums[8];  // max 256 threads / 32 = 8 warps
    int warp_id = threadIdx.x / ASANN_WARP_SIZE;
    int lane_id = threadIdx.x % ASANN_WARP_SIZE;

    if (lane_id == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        int num_warps = (blockDim.x + ASANN_WARP_SIZE - 1) / ASANN_WARP_SIZE;
        float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = ASANN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            row_norms[row] = sqrtf(val);
        }
    }
}

// Phase 2: Normalize each element: grad[i] = grad[i] / max(row_norm, eps) * mean_norm
template <typename scalar_t>
__global__ void normuon_normalize_kernel(
    scalar_t* __restrict__ grad,
    const float* __restrict__ row_norms,  // always FP32
    const float mean_norm,
    const float eps,
    const int num_rows,
    const int row_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_len;
    if (idx >= total) return;

    int row = idx / row_len;
    float norm = row_norms[row];
    float safe_norm = fmaxf(norm, eps);
    float g = to_float(grad[idx]);
    grad[idx] = from_float<scalar_t>(g / safe_norm * mean_norm);
}

// =============================================================================
// C++ Interface
// =============================================================================

void normuon_normalize(
    torch::Tensor grad,
    float eps
) {
    ASANN_CHECK_INPUT(grad);
    ASANN_CHECK_FLOAT_OR_HALF(grad);

    int ndim = grad.dim();
    int num_rows, row_len;

    if (ndim == 4) {
        // Conv2d [C_out, C_in, k, k]
        num_rows = grad.size(0);
        row_len = grad.size(1) * grad.size(2) * grad.size(3);
    } else if (ndim == 2) {
        // Linear [d_out, d_in]
        num_rows = grad.size(0);
        row_len = grad.size(1);
    } else {
        TORCH_CHECK(false, "normuon_normalize: expected 2D or 4D tensor, got ", ndim, "D");
    }

    if (num_rows == 0 || row_len == 0) return;

    // Allocate temporary row norms (always FP32)
    auto row_norms = torch::empty({num_rows}, grad.options().dtype(at::kFloat));

    // Phase 1: Compute row norms (one block per row)
    int threads_p1 = ASANN_THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "normuon_compute_norms", [&] {
        normuon_compute_norms_kernel<scalar_t><<<num_rows, threads_p1>>>(
            grad.data_ptr<scalar_t>(),
            row_norms.data_ptr<float>(),
            num_rows,
            row_len
        );
    });
    ASANN_DEBUG_SYNC();

    // Compute mean norm using ATen (small tensor, negligible cost)
    float mean_norm = row_norms.mean().item<float>();

    // Phase 2: Normalize elements
    int total = num_rows * row_len;
    int threads_p2 = ASANN_THREADS_PER_BLOCK;
    int blocks_p2 = asann_get_blocks(total, threads_p2);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "normuon_normalize", [&] {
        normuon_normalize_kernel<scalar_t><<<blocks_p2, threads_p2>>>(
            grad.data_ptr<scalar_t>(),
            row_norms.data_ptr<float>(),
            mean_norm,
            eps,
            num_rows,
            row_len
        );
    });
    ASANN_DEBUG_SYNC();
}
