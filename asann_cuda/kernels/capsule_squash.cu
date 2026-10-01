#include "common.cuh"

// =============================================================================
// Capsule Squash Activation:
//   Groups channels into capsule vectors of size cap_dim (2 or 4)
//   Computes per-group: v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)
//
// Input/Output: [B, C_padded, H, W] where C_padded = num_capsules * cap_dim
// One thread per (batch, capsule_index, spatial_position)
// cap_dim fits in registers — no shared memory needed
// =============================================================================

// =============================================================================
// Forward Kernel
// =============================================================================

template <typename scalar_t, int CAP_DIM>
__global__ void capsule_squash_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int num_capsules,
    const int HW,
    const int total,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int spatial_pos = idx % HW;
    int tmp = idx / HW;
    int cap_idx = tmp % num_capsules;
    int b = tmp / num_capsules;

    int C_padded = num_capsules * CAP_DIM;

    // Compute squared norm for this capsule
    float sq_norm = 0.0f;
    #pragma unroll
    for (int d = 0; d < CAP_DIM; d++) {
        int channel = cap_idx * CAP_DIM + d;
        int offset = b * C_padded * HW + channel * HW + spatial_pos;
        float val = to_float(input[offset]);
        sq_norm += val * val;
    }

    // Squash scale: sq_norm / (1 + sq_norm) / sqrt(sq_norm + eps)
    float norm = sqrtf(sq_norm + eps);
    float scale = sq_norm / (1.0f + sq_norm) / norm;

    // Apply scale to each component
    #pragma unroll
    for (int d = 0; d < CAP_DIM; d++) {
        int channel = cap_idx * CAP_DIM + d;
        int offset = b * C_padded * HW + channel * HW + spatial_pos;
        float val = to_float(input[offset]);
        output[offset] = from_float<scalar_t>(val * scale);
    }
}

// =============================================================================
// Backward Kernel
//
// v = squash(s) = f * s  where  f = ||s|| / (1 + ||s||^2)
//
// grad_s_j = f * grad_v_j + s_j * (1 - ||s||^2) / (||s|| * (1+||s||^2)^2) * dot(grad_v, s)
// =============================================================================

template <typename scalar_t, int CAP_DIM>
__global__ void capsule_squash_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_input,
    const int num_capsules,
    const int HW,
    const int total,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int spatial_pos = idx % HW;
    int tmp = idx / HW;
    int cap_idx = tmp % num_capsules;
    int b = tmp / num_capsules;

    int C_padded = num_capsules * CAP_DIM;

    // Load capsule values and gradients into registers
    float s[CAP_DIM], gv[CAP_DIM];
    float sq_norm = 0.0f;
    float dot_gv_s = 0.0f;

    #pragma unroll
    for (int d = 0; d < CAP_DIM; d++) {
        int channel = cap_idx * CAP_DIM + d;
        int offset = b * C_padded * HW + channel * HW + spatial_pos;
        s[d] = to_float(input[offset]);
        gv[d] = to_float(grad_output[offset]);
        sq_norm += s[d] * s[d];
        dot_gv_s += gv[d] * s[d];
    }

    float norm = sqrtf(sq_norm + eps);
    float one_plus_sq = 1.0f + sq_norm;
    float f = norm / one_plus_sq;
    float coeff = (1.0f - sq_norm) / (norm * one_plus_sq * one_plus_sq) * dot_gv_s;

    #pragma unroll
    for (int d = 0; d < CAP_DIM; d++) {
        int channel = cap_idx * CAP_DIM + d;
        int offset = b * C_padded * HW + channel * HW + spatial_pos;
        float gs = f * gv[d] + coeff * s[d];
        grad_input[offset] = from_float<scalar_t>(gs);
    }
}

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> capsule_squash_forward(
    torch::Tensor input,
    int cap_dim,
    float eps
) {
    ASANN_CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 4, "capsule_squash: input must be 4D [B, C_padded, H, W]");

    int B = input.size(0);
    int C_padded = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    int num_capsules = C_padded / cap_dim;

    TORCH_CHECK(C_padded % cap_dim == 0,
                "capsule_squash: C_padded must be divisible by cap_dim");

    auto output = torch::empty_like(input);
    int total = B * num_capsules * HW;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "capsule_squash_fwd", [&] {
        if (cap_dim == 2) {
            capsule_squash_forward_kernel<scalar_t, 2><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                num_capsules, HW, total, eps);
        } else if (cap_dim == 4) {
            capsule_squash_forward_kernel<scalar_t, 4><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                num_capsules, HW, total, eps);
        } else {
            TORCH_CHECK(false, "capsule_squash: only cap_dim=2 and cap_dim=4 supported");
        }
    });
    ASANN_DEBUG_SYNC();

    return {output};
}

std::vector<torch::Tensor> capsule_squash_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    int cap_dim,
    float eps
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(input);

    int B = input.size(0);
    int C_padded = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    int num_capsules = C_padded / cap_dim;

    auto grad_input = torch::empty_like(input);
    int total = B * num_capsules * HW;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "capsule_squash_bwd", [&] {
        if (cap_dim == 2) {
            capsule_squash_backward_kernel<scalar_t, 2><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                num_capsules, HW, total, eps);
        } else if (cap_dim == 4) {
            capsule_squash_backward_kernel<scalar_t, 4><<<blocks, threads>>>(
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                num_capsules, HW, total, eps);
        } else {
            TORCH_CHECK(false, "capsule_squash: only cap_dim=2 and cap_dim=4 supported");
        }
    });
    ASANN_DEBUG_SYNC();

    return {grad_input};
}
