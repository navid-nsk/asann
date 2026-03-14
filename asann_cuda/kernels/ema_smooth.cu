#include "common.cuh"

// =============================================================================
// EMASmooth: Exponential Moving Average over feature dimension with gated residual
// y[b][0] = x[b][0]
// y[b][i] = alpha[i] * x[b][i] + (1 - alpha[i]) * y[b][i-1]  for i > 0
// out = (1-gate)*x + gate*y
//
// alpha = sigmoid(alpha_logit), per-channel learnable
// Sequential scan per batch element (d is small, 32-512)
// =============================================================================

// Forward declarations from gated_residual.cu
torch::Tensor gated_residual_forward(
    torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor transformed, torch::Tensor gate_logit);

// Each thread handles one batch element's full EMA scan
// FP32 running accumulator for numerical stability in sequential scan
template <typename scalar_t>
__global__ void ema_smooth_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ alpha,  // [d] already sigmoid'd
    scalar_t* __restrict__ y,
    const int B,
    const int d
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const scalar_t* x_row = x + b * d;
    scalar_t* y_row = y + b * d;

    float prev = to_float(x_row[0]);
    y_row[0] = from_float<scalar_t>(prev);
    for (int i = 1; i < d; i++) {
        float a = to_float(alpha[i]);
        float xi = to_float(x_row[i]);
        prev = a * xi + (1.0f - a) * prev;
        y_row[i] = from_float<scalar_t>(prev);
    }
}

// Backward: each thread handles one batch element
// Need grad_x and grad_alpha_logit contributions
// FP32 running accumulator for backward scan
template <typename scalar_t>
__global__ void ema_smooth_backward_kernel(
    const scalar_t* __restrict__ grad_y,  // grad w.r.t. EMA output y
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ alpha,   // [d] sigmoid'd
    const scalar_t* __restrict__ y,       // forward output
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_alpha_contrib, // [B, d] per-element contribution
    const int B,
    const int d
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const scalar_t* gy_row = grad_y + b * d;
    const scalar_t* x_row = x + b * d;
    const scalar_t* y_row = y + b * d;
    scalar_t* gx_row = grad_x + b * d;
    scalar_t* ga_row = grad_alpha_contrib + b * d;

    // Reverse-mode through the sequential scan
    // y[i] = alpha[i]*x[i] + (1-alpha[i])*y[i-1]
    // dy[i]/dx[i] = alpha[i]
    // dy[i]/dy[i-1] = (1-alpha[i])
    // dy[i]/dalpha[i] = x[i] - y[i-1]  (need dsigmoid too)

    // Accumulate from the end
    float acc = 0.0f;  // accumulated gradient flowing backward through y chain

    for (int i = d - 1; i >= 0; i--) {
        float total_grad = to_float(gy_row[i]) + acc;

        // grad w.r.t. x[i]
        if (i == 0) {
            gx_row[i] = from_float<scalar_t>(total_grad);  // y[0] = x[0], so dy[0]/dx[0] = 1
            ga_row[i] = from_float<scalar_t>(0.0f);        // no alpha involvement at i=0
        } else {
            float a = to_float(alpha[i]);
            gx_row[i] = from_float<scalar_t>(total_grad * a);
            // grad w.r.t. alpha[i] (before dsigmoid)
            float y_prev = to_float(y_row[i - 1]);
            float xi = to_float(x_row[i]);
            ga_row[i] = from_float<scalar_t>(total_grad * (xi - y_prev));
            // Propagate through y[i-1]
            acc = total_grad * (1.0f - a);
        }
    }
}

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor ema_smooth_forward(
    torch::Tensor x,
    torch::Tensor alpha_logit,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(alpha_logit);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);

    auto alpha = at::sigmoid(alpha_logit);  // [d]
    auto y = torch::empty_like(x);

    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(B, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ema_smooth_forward", [&] {
        ema_smooth_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    return gated_residual_forward(x, y, gate_logit);
}

std::vector<torch::Tensor> ema_smooth_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor alpha_logit,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(alpha_logit);

    int B = x.size(0);
    int d = x.size(1);

    auto alpha = at::sigmoid(alpha_logit);

    // Recompute y for backward
    auto y = torch::empty_like(x);
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks_b = asann_get_blocks(B, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ema_smooth_recompute", [&] {
        ema_smooth_forward_kernel<scalar_t><<<blocks_b, threads>>>(
            x.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    // Backward through gated residual
    auto gr_grads = gated_residual_backward(grad_output, x, y, gate_logit);
    auto grad_x_residual = gr_grads[0];
    auto grad_y = gr_grads[1];
    auto grad_gate_logit = gr_grads[2];

    // Backward through EMA
    auto grad_x_ema = torch::empty_like(x);
    auto grad_alpha_contrib = torch::empty_like(x);  // [B, d]

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ema_smooth_backward", [&] {
        ema_smooth_backward_kernel<scalar_t><<<blocks_b, threads>>>(
            grad_y.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x_ema.data_ptr<scalar_t>(),
            grad_alpha_contrib.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    // Sum alpha grad contributions over batch and apply dsigmoid
    auto grad_alpha = grad_alpha_contrib.sum(0);  // [d]
    auto dsigmoid = alpha * (1.0f - alpha);       // [d]
    auto grad_alpha_logit = grad_alpha * dsigmoid; // [d]

    auto grad_x = grad_x_residual + grad_x_ema;

    return {grad_x, grad_alpha_logit, grad_gate_logit};
}
