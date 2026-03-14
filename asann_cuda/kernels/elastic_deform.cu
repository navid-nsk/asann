#include "common.cuh"

// =============================================================================
// Elastic Deformation: Separable Gaussian Blur for displacement fields
//
// Input:  displacement field [N, H, W] where N = B*2 (dx and dy stacked)
// Output: blurred displacement field [N, H, W]
//
// Two-pass separable: horizontal blur -> vertical blur
// Kernel weights stored in constant memory (max 64 elements)
// Forward-only (data augmentation, no backward needed)
// =============================================================================

// 1D Gaussian kernel weights in constant memory (shared across all threads)
__constant__ float d_elastic_kernel[64];

// Reflect-pad index: mirrors at boundaries
__device__ __forceinline__ int elastic_reflect(int i, int N) {
    // One or two bounces handles kernel_size <= 64 on image_size >= 4
    if (i < 0) i = -i;
    if (i >= N) i = 2 * (N - 1) - i;
    if (i < 0) i = 0;  // safety clamp for extreme cases
    return i;
}

// -----------------------------------------------------------------------------
// Horizontal 1D Gaussian blur
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void elastic_blur_h_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N, const int H, const int W,
    const int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H * W);
    int h = (idx / W) % H;
    int w = idx % W;

    int half_k = kernel_size / 2;
    float sum = 0.0f;
    int base = n * H * W + h * W;

    for (int k = 0; k < kernel_size; k++) {
        int src_w = elastic_reflect(w + k - half_k, W);
        sum += to_float(input[base + src_w]) * d_elastic_kernel[k];
    }
    output[idx] = from_float<scalar_t>(sum);
}

// -----------------------------------------------------------------------------
// Vertical 1D Gaussian blur
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void elastic_blur_v_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N, const int H, const int W,
    const int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H * W);
    int h = (idx / W) % H;
    int w = idx % W;

    int half_k = kernel_size / 2;
    float sum = 0.0f;
    int base_nw = n * H * W + w;

    for (int k = 0; k < kernel_size; k++) {
        int src_h = elastic_reflect(h + k - half_k, H);
        sum += to_float(input[base_nw + src_h * W]) * d_elastic_kernel[k];
    }
    output[idx] = from_float<scalar_t>(sum);
}

// =============================================================================
// C++ Interface
// =============================================================================

void elastic_set_kernel_weights(torch::Tensor weights_1d) {
    TORCH_CHECK(weights_1d.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(weights_1d.scalar_type() == at::kFloat, "weights must be float32");
    int n = weights_1d.size(0);
    TORCH_CHECK(n <= 64, "kernel_size must be <= 64, got ", n);

    // Copy from CPU tensor to constant memory
    const float* src = weights_1d.data_ptr<float>();
    ASANN_CUDA_CHECK(cudaMemcpyToSymbol(
        d_elastic_kernel, src, n * sizeof(float), 0, cudaMemcpyHostToDevice
    ));
}

torch::Tensor elastic_blur_separable(
    torch::Tensor displacement,
    int kernel_size
) {
    ASANN_CHECK_INPUT(displacement);
    TORCH_CHECK(displacement.dim() == 3,
                "displacement must be 3D [N, H, W], got ", displacement.dim(), "D");
    TORCH_CHECK(kernel_size > 0 && kernel_size <= 64 && kernel_size % 2 == 1,
                "kernel_size must be odd and 1..63, got ", kernel_size);

    int N = displacement.size(0);
    int H = displacement.size(1);
    int W = displacement.size(2);

    // Intermediate buffer for horizontal pass
    auto temp = torch::empty_like(displacement);
    auto output = torch::empty_like(displacement);

    int total = N * H * W;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    // Pass 1: Horizontal blur
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        displacement.scalar_type(), "elastic_blur_h", [&] {
        elastic_blur_h_kernel<scalar_t><<<blocks, threads>>>(
            displacement.data_ptr<scalar_t>(),
            temp.data_ptr<scalar_t>(),
            N, H, W, kernel_size
        );
    });
    ASANN_DEBUG_SYNC();

    // Pass 2: Vertical blur on horizontal result
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        temp.scalar_type(), "elastic_blur_v", [&] {
        elastic_blur_v_kernel<scalar_t><<<blocks, threads>>>(
            temp.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, H, W, kernel_size
        );
    });
    ASANN_DEBUG_SYNC();

    return output;
}
