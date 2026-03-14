#include "common.cuh"

// =============================================================================
// Fused Optimizer Step Kernel
//
// Fuses all per-element operations of the ASANNOptimizer inner loop into a
// single kernel launch per parameter:
//   1. Update 3 momentum buffers (m_fast, m_medium, m_slow)
//   2. Update second moment v
//   3. Compute bias-corrected combined momentum
//   4. Either: write combined to buf (for Newton-Schulz path)
//      Or: apply weight decay + adaptive step directly to p_data
//
// All math is identical to the Python implementation.
// =============================================================================

__global__ void fused_optimizer_step_kernel(
    float* __restrict__ p_data,
    const float* __restrict__ grad,
    float* __restrict__ m_fast,
    float* __restrict__ m_medium,
    float* __restrict__ m_slow,
    float* __restrict__ v,
    float* __restrict__ buf,
    const float beta1_fast,
    const float beta1_med,
    const float beta1_slow,
    const float beta2,
    const float inv_bc_fast,   // 0.5 / bc_fast
    const float inv_bc_med,    // 0.35 / bc_med
    const float inv_bc_slow,   // 0.15 / bc_slow
    const float inv_sqrt_bc_v, // 1.0 / sqrt(bc_v)
    const float lr,
    const float weight_decay,
    const float newborn_mult,
    const float eps,
    const int N,
    const bool write_buf_only  // true = Newton-Schulz path (write buf, skip update)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = grad[idx];

    // Update momentum buffers (in-place EMA)
    float mf = m_fast[idx] * beta1_fast + g * (1.0f - beta1_fast);
    float mm = m_medium[idx] * beta1_med + g * (1.0f - beta1_med);
    float ms = m_slow[idx] * beta1_slow + g * (1.0f - beta1_slow);

    m_fast[idx] = mf;
    m_medium[idx] = mm;
    m_slow[idx] = ms;

    // Update second moment
    float vi = v[idx] * beta2 + g * g * (1.0f - beta2);
    v[idx] = vi;

    // Combined bias-corrected momentum
    float combined = mf * inv_bc_fast + mm * inv_bc_med + ms * inv_bc_slow;

    if (write_buf_only) {
        // Newton-Schulz path: just write combined to buf for later processing
        buf[idx] = combined;
        return;
    }

    // Normal path: apply weight decay and adaptive step
    float p = p_data[idx];

    // Decoupled weight decay: p -= lr * weight_decay * p
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    // Adaptive step: p -= lr * newborn_mult * combined / (sqrt(v/bc_v) + eps)
    float denom = sqrtf(vi * inv_sqrt_bc_v * inv_sqrt_bc_v) + eps;
    p -= lr * newborn_mult * combined / denom;

    p_data[idx] = p;
}

// =============================================================================
// Companion kernel: Apply update after Newton-Schulz orthogonalization
//
// Called when the Newton-Schulz path was used: buf now contains the
// orthogonalized combined momentum. Apply weight decay + adaptive step.
// =============================================================================

__global__ void optimizer_apply_update_kernel(
    float* __restrict__ p_data,
    const float* __restrict__ buf,
    const float* __restrict__ v,
    const float inv_sqrt_bc_v,
    const float lr,
    const float weight_decay,
    const float newborn_mult,
    const float eps,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float p = p_data[idx];

    // Decoupled weight decay
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    // Adaptive step with orthogonalized buf
    float denom = sqrtf(v[idx] * inv_sqrt_bc_v * inv_sqrt_bc_v) + eps;
    p -= lr * newborn_mult * buf[idx] / denom;

    p_data[idx] = p;
}

// =============================================================================
// C++ Interfaces
// =============================================================================

void fused_optimizer_step(
    torch::Tensor p_data,
    torch::Tensor grad,
    torch::Tensor m_fast,
    torch::Tensor m_medium,
    torch::Tensor m_slow,
    torch::Tensor v,
    torch::Tensor buf,
    float beta1_fast,
    float beta1_med,
    float beta1_slow,
    float beta2,
    float inv_bc_fast,
    float inv_bc_med,
    float inv_bc_slow,
    float inv_sqrt_bc_v,
    float lr,
    float weight_decay,
    float newborn_mult,
    float eps,
    bool write_buf_only
) {
    ASANN_CHECK_INPUT(p_data);
    ASANN_CHECK_INPUT(grad);
    ASANN_CHECK_INPUT(m_fast);
    ASANN_CHECK_INPUT(m_medium);
    ASANN_CHECK_INPUT(m_slow);
    ASANN_CHECK_INPUT(v);
    ASANN_CHECK_INPUT(buf);
    ASANN_CHECK_FLOAT(p_data);

    int N = p_data.numel();
    TORCH_CHECK(grad.numel() == N, "grad size mismatch");
    TORCH_CHECK(m_fast.numel() == N, "m_fast size mismatch");

    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(N, threads);

    fused_optimizer_step_kernel<<<blocks, threads>>>(
        p_data.data_ptr<float>(),
        grad.data_ptr<float>(),
        m_fast.data_ptr<float>(),
        m_medium.data_ptr<float>(),
        m_slow.data_ptr<float>(),
        v.data_ptr<float>(),
        buf.data_ptr<float>(),
        beta1_fast,
        beta1_med,
        beta1_slow,
        beta2,
        inv_bc_fast,
        inv_bc_med,
        inv_bc_slow,
        inv_sqrt_bc_v,
        lr,
        weight_decay,
        newborn_mult,
        eps,
        N,
        write_buf_only
    );
    ASANN_DEBUG_SYNC();
}

void optimizer_apply_update(
    torch::Tensor p_data,
    torch::Tensor buf,
    torch::Tensor v,
    float inv_sqrt_bc_v,
    float lr,
    float weight_decay,
    float newborn_mult,
    float eps
) {
    ASANN_CHECK_INPUT(p_data);
    ASANN_CHECK_INPUT(buf);
    ASANN_CHECK_INPUT(v);
    ASANN_CHECK_FLOAT(p_data);

    int N = p_data.numel();
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(N, threads);

    optimizer_apply_update_kernel<<<blocks, threads>>>(
        p_data.data_ptr<float>(),
        buf.data_ptr<float>(),
        v.data_ptr<float>(),
        inv_sqrt_bc_v,
        lr,
        weight_decay,
        newborn_mult,
        eps,
        N
    );
    ASANN_DEBUG_SYNC();
}
