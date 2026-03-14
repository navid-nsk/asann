#include "common.cuh"

// =============================================================================
// GeometricEmbedding: Poincaré-inspired geometric embedding
// x: [B, d], radial_scale: [d], bias: [d], gate_logit: scalar
//
// Forward:
//   gate = sigmoid(gate_logit)
//   scaled = radial_scale * x + bias
//   norm = scaled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
//   poincare = scaled * (tanh(norm) / norm)
//   out = (1 - gate) * x + gate * poincare
//
// Backward: chain through gated residual, tanh, norm, element-wise ops
// =============================================================================

// Compute L2 norm per batch element: norm[b] = sqrt(sum_i scaled[b][i]^2)
// Input is scalar_t, output norm is always FP32 for precision
template <typename scalar_t>
__global__ void geometric_norm_kernel(
    const scalar_t* __restrict__ scaled,
    float* __restrict__ norm_out,
    const int batch_size,
    const int d
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float sum_sq = 0.0f;  // FP32 accumulator
    for (int i = 0; i < d; i++) {
        float s = to_float(scaled[b * d + i]);
        sum_sq += s * s;
    }
    float n = sqrtf(sum_sq);
    norm_out[b] = fmaxf(n, 1e-8f);  // clamp(min=1e-8)
}

// Compute poincare = scaled * (tanh(norm) / norm)
// Input scaled is scalar_t, norm is FP32, output is scalar_t
template <typename scalar_t>
__global__ void geometric_poincare_transform_kernel(
    const scalar_t* __restrict__ scaled,
    const float* __restrict__ norm,
    scalar_t* __restrict__ poincare_out,
    const int batch_size,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * d) return;

    int b = idx / d;
    float n = norm[b];
    float factor = tanhf(n) / n;
    poincare_out[idx] = from_float<scalar_t>(to_float(scaled[idx]) * factor);
}

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> geometric_embedding_forward(
    torch::Tensor x,
    torch::Tensor radial_scale,
    torch::Tensor bias,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(radial_scale);
    ASANN_CHECK_INPUT(bias);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);

    float gate = 1.0f / (1.0f + expf(-gate_logit.item<float>()));

    // Step 1: scaled = radial_scale * x + bias (broadcasting [d] over [B, d])
    auto scaled = x * radial_scale.unsqueeze(0) + bias.unsqueeze(0);

    int total = B * d;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    // Step 2: norm = scaled.norm(dim=-1).clamp(min=1e-8) — always FP32
    auto norm = torch::empty({B}, x.options().dtype(at::kFloat));
    int norm_blocks = asann_get_blocks(B, threads);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scaled.scalar_type(), "geometric_norm", [&] {
        geometric_norm_kernel<scalar_t><<<norm_blocks, threads>>>(
            scaled.data_ptr<scalar_t>(),
            norm.data_ptr<float>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    // Step 3: poincare = scaled * (tanh(norm) / norm)
    auto poincare = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scaled.scalar_type(), "geometric_poincare", [&] {
        geometric_poincare_transform_kernel<scalar_t><<<blocks, threads>>>(
            scaled.data_ptr<scalar_t>(),
            norm.data_ptr<float>(),
            poincare.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    // Step 4: out = (1 - gate) * x + gate * poincare
    auto output = (1.0f - gate) * x + gate * poincare;

    // Return output + intermediates for backward
    return {output, scaled, norm, poincare};
}

std::vector<torch::Tensor> geometric_embedding_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor scaled,
    torch::Tensor norm,
    torch::Tensor poincare,
    torch::Tensor radial_scale,
    torch::Tensor bias,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(scaled);
    ASANN_CHECK_INPUT(norm);
    ASANN_CHECK_INPUT(poincare);
    ASANN_CHECK_INPUT(radial_scale);

    int B = x.size(0);
    int d = x.size(1);

    float gate_val = 1.0f / (1.0f + expf(-gate_logit.item<float>()));
    float dsigmoid = gate_val * (1.0f - gate_val);

    // Forward:
    //   out = (1 - gate) * x + gate * poincare
    //
    // Backward:
    //   grad_x_from_residual = (1 - gate) * grad_output
    //   grad_poincare = gate * grad_output
    //   grad_gate_logit = sum( grad_output * (poincare - x) ) * dsigmoid

    auto grad_x_residual = (1.0f - gate_val) * grad_output;
    auto grad_poincare = gate_val * grad_output;

    // grad_gate_logit
    auto gate_contrib = (grad_output * (poincare - x)).sum() * dsigmoid;
    auto grad_gate_logit = gate_contrib.unsqueeze(0);

    // Backward through poincare = scaled * (tanh(norm) / norm)
    // Let factor = tanh(norm) / norm, shape [B]
    // poincare[b][i] = scaled[b][i] * factor[b]
    auto norm_unsq = norm.unsqueeze(1);  // [B, 1]
    auto tanh_norm = torch::tanh(norm_unsq);  // [B, 1]
    auto factor = tanh_norm / norm_unsq;  // [B, 1]

    auto sech2 = 1.0f - tanh_norm * tanh_norm;  // [B, 1]
    auto dfactor_dnorm = (sech2 * norm_unsq - tanh_norm) / (norm_unsq * norm_unsq);  // [B, 1]

    auto grad_poincare_dot_scaled = (grad_poincare * scaled).sum(1, true);  // [B, 1]
    auto grad_scaled = grad_poincare * factor
                     + grad_poincare_dot_scaled * dfactor_dnorm * scaled / norm_unsq;

    // Backward through scaled = radial_scale * x + bias
    auto grad_x_from_scaled = grad_scaled * radial_scale.unsqueeze(0);
    auto grad_radial_scale = (grad_scaled * x).sum(0);
    auto grad_bias = grad_scaled.sum(0);

    // Total grad_x
    auto grad_x = grad_x_residual + grad_x_from_scaled;

    return {grad_x, grad_radial_scale, grad_bias, grad_gate_logit};
}
