#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

// =============================================================================
// Error Checking Macros
// =============================================================================

#define ASANN_CUDA_CHECK(call)                                                 \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "ASANN CUDA Error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            TORCH_CHECK(false, "ASANN CUDA Error: ", cudaGetErrorString(err)); \
        }                                                                      \
    } while (0)

#define ASANN_CHECK_CUDA_ERROR()                                               \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "ASANN CUDA kernel error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            TORCH_CHECK(false, "ASANN CUDA kernel error: ",                    \
                        cudaGetErrorString(err));                              \
        }                                                                      \
    } while (0)

// Debug-mode synchronize + check (expensive, only for debugging)
#ifdef ASANN_CUDA_DEBUG
#define ASANN_DEBUG_SYNC()                                                     \
    do {                                                                        \
        ASANN_CUDA_CHECK(cudaDeviceSynchronize());                            \
        ASANN_CHECK_CUDA_ERROR();                                              \
    } while (0)
#else
#define ASANN_DEBUG_SYNC() ASANN_CHECK_CUDA_ERROR()
#endif

// =============================================================================
// Tensor Validation Macros
// =============================================================================

#define ASANN_CHECK_CUDA(x)                                                    \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define ASANN_CHECK_CONTIGUOUS(x)                                              \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define ASANN_CHECK_INPUT(x)                                                   \
    do {                                                                        \
        ASANN_CHECK_CUDA(x);                                                   \
        ASANN_CHECK_CONTIGUOUS(x);                                             \
    } while (0)

#define ASANN_CHECK_FLOAT(x)                                                   \
    TORCH_CHECK((x).scalar_type() == at::kFloat,                               \
                #x " must be float32 tensor")

#define ASANN_CHECK_FLOAT_OR_HALF(x)                                           \
    TORCH_CHECK((x).scalar_type() == at::kFloat ||                             \
                (x).scalar_type() == at::kHalf,                                \
                #x " must be float32 or float16 tensor")

// =============================================================================
// Thread/Block Configuration Helpers
// =============================================================================

constexpr int ASANN_THREADS_PER_BLOCK = 256;
constexpr int ASANN_WARP_SIZE = 32;

inline int asann_get_blocks(int total_threads,
                            int threads_per_block = ASANN_THREADS_PER_BLOCK) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}

// =============================================================================
// Math Utilities (device functions)
// =============================================================================

__device__ __forceinline__ float asann_safe_div(float a, float b,
                                                 float eps = 1e-8f) {
    return a / (b + eps);
}

__device__ __forceinline__ float asann_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float asann_sigmoid_backward(float sigmoid_val) {
    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    return sigmoid_val * (1.0f - sigmoid_val);
}

__device__ __forceinline__ float asann_tanh_backward(float tanh_val) {
    // d(tanh)/dx = 1 - tanh^2
    return 1.0f - tanh_val * tanh_val;
}

__device__ __forceinline__ float asann_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ __forceinline__ float asann_relu_backward(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ __forceinline__ float asann_clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// =============================================================================
// FP16 / Mixed-Precision Helpers (templated)
// =============================================================================

// Convert any scalar_t to float for FP32 accumulation
template<typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template<>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }

// Convert float back to scalar_t
template<typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }

template<>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
