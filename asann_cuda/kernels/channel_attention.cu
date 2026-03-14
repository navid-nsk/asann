#include "common.cuh"

// =============================================================================
// ChannelAttentionOp (SE-block):
//   squeeze: GAP -> [B, C]
//   excitation: fc1 -> relu -> fc2 -> sigmoid -> [B, C, 1, 1]
//   rescale: x * attention_weights
//   gated residual: (1-gate)*x + gate*(x * attention_weights)
//
// x: [B, C, H, W]
// fc1: [reduction, C], fc2: [C, reduction]
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// GAP kernel: global average pooling [B, C, H, W] -> [B, C]
// FP32 accumulator for numerical stability
// =============================================================================

template <typename scalar_t>
__global__ void channel_gap_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int C,
    const int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * C) return;

    int b = idx / C;
    int c = idx % C;

    float sum = 0.0f;  // FP32 accumulator
    int base = b * C * HW + c * HW;
    for (int i = 0; i < HW; i++) {
        sum += to_float(x[base + i]);
    }
    output[idx] = from_float<scalar_t>(sum / (float)HW);
}

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> channel_attention_forward(
    torch::Tensor x,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(fc1_weight);
    ASANN_CHECK_INPUT(fc1_bias);
    ASANN_CHECK_INPUT(fc2_weight);
    ASANN_CHECK_INPUT(fc2_bias);
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    // Step 1: Squeeze (GAP) -> [B, C]
    auto squeezed = torch::empty({B, C}, x.options());
    int total = B * C;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total, threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "channel_gap", [&] {
        channel_gap_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            squeezed.data_ptr<scalar_t>(),
            B, C, HW
        );
    });
    ASANN_DEBUG_SYNC();

    // Step 2: Excitation MLP
    // fc1: [B, C] -> [B, reduction]
    auto fc1_out = at::linear(squeezed, fc1_weight, fc1_bias);
    auto relu_out = at::relu(fc1_out);
    // fc2: [B, reduction] -> [B, C]
    auto fc2_out = at::linear(relu_out, fc2_weight, fc2_bias);
    auto attn_weights = at::sigmoid(fc2_out);  // [B, C]

    // Step 3: Rescale: x * attn_weights.unsqueeze(-1).unsqueeze(-1)
    auto attn_4d = attn_weights.unsqueeze(-1).unsqueeze(-1);  // [B, C, 1, 1]
    auto rescaled = x * attn_4d;  // [B, C, H, W]

    // Step 4: Gated residual
    auto output_flat = gated_residual_forward(
        x.reshape({B, -1}),
        rescaled.reshape({B, -1}),
        gate_logit
    );
    auto output = output_flat.view_as(x);

    return {output, squeezed, fc1_out, relu_out, fc2_out, attn_weights, rescaled};
}

std::vector<torch::Tensor> channel_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor squeezed,
    torch::Tensor fc1_out,
    torch::Tensor relu_out,
    torch::Tensor fc2_out,
    torch::Tensor attn_weights,
    torch::Tensor rescaled,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(
        grad_output.reshape({B, -1}),
        x.reshape({B, -1}),
        rescaled.reshape({B, -1}),
        gate_logit
    );
    auto grad_x_gated = gated_grads[0].view_as(x);
    auto grad_rescaled = gated_grads[1].view_as(x);
    auto grad_gate_logit = gated_grads[2];

    // Backward through rescale: rescaled = x * attn_4d
    auto attn_4d = attn_weights.unsqueeze(-1).unsqueeze(-1);
    auto grad_x_from_rescale = grad_rescaled * attn_4d;
    // grad_attn_4d = grad_rescaled * x, then sum over H, W
    auto grad_attn_weights = (grad_rescaled * x).sum(at::IntArrayRef({2, 3}));  // [B, C]

    // Backward through sigmoid: attn_weights = sigmoid(fc2_out)
    auto grad_fc2_out = grad_attn_weights * attn_weights * (1.0f - attn_weights);

    // Cast weights to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto fc2w_bwd = fc2_weight.to(act_dtype);
    auto fc1w_bwd = fc1_weight.to(act_dtype);

    // Backward through fc2: fc2_out = relu_out @ fc2_weight.T + fc2_bias
    auto grad_relu_out = at::matmul(grad_fc2_out, fc2w_bwd);
    auto grad_fc2_weight = at::matmul(grad_fc2_out.t(), relu_out);
    auto grad_fc2_bias = grad_fc2_out.sum(0);

    // Backward through relu
    auto grad_fc1_out = grad_relu_out * (fc1_out > 0).to(grad_relu_out.dtype());

    // Backward through fc1: fc1_out = squeezed @ fc1_weight.T + fc1_bias
    auto grad_squeezed = at::matmul(grad_fc1_out, fc1w_bwd);
    auto grad_fc1_weight = at::matmul(grad_fc1_out.t(), squeezed);
    auto grad_fc1_bias = grad_fc1_out.sum(0);

    // Backward through GAP: squeezed[b][c] = mean(x[b][c][h][w])
    // grad_x_from_gap[b][c][h][w] = grad_squeezed[b][c] / (H*W)
    auto grad_x_from_gap = grad_squeezed.unsqueeze(-1).unsqueeze(-1).expand_as(x) / (float)(HW);

    // Total grad_x
    auto grad_x = grad_x_gated + grad_x_from_rescale + grad_x_from_gap;

    return {grad_x, grad_fc1_weight, grad_fc1_bias,
            grad_fc2_weight, grad_fc2_bias, grad_gate_logit};
}
