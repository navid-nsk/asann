#include "common.cuh"

// =============================================================================
// DilatedConv1dBlock: Like Conv1dBlock but with dilation for larger receptive field
// Reshapes [B, d] -> [B, 1, d], applies dilated conv1d(same-pad), -> [B, d]
// Plus gated residual: out = (1-gate)*x + gate*conv(x)
// Uses at::conv1d (cuDNN) with dilation parameter.
// =============================================================================

// Forward declarations from gated_residual.cu
torch::Tensor gated_residual_forward(
    torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor dilated_conv1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gate_logit,
    int kernel_size,
    int dilation
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(weight);
    ASANN_CHECK_INPUT(bias);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int padding = dilation * (kernel_size / 2);

    // x: [B, d] -> [B, 1, d]
    auto x_3d = x.unsqueeze(1);
    // Dilated conv1d: [B, 1, d] -> [B, 1, d] (same padding with dilation)
    auto out_3d = at::conv1d(x_3d, weight, bias,
                             /*stride=*/at::IntArrayRef({1}),
                             /*padding=*/at::IntArrayRef({padding}),
                             /*dilation=*/at::IntArrayRef({dilation}));
    // [B, 1, d] -> [B, d]
    auto conv_out = out_3d.squeeze(1);

    // Gated residual: (1-gate)*x + gate*conv_out
    return gated_residual_forward(x, conv_out, gate_logit);
}

std::vector<torch::Tensor> dilated_conv1d_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gate_logit,
    int kernel_size,
    int dilation
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(weight);

    int padding = dilation * (kernel_size / 2);

    // Cast weights to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto w_bwd = weight.to(act_dtype);
    auto b_bwd = bias.to(act_dtype);

    // Recompute conv output for gated residual backward
    auto x_3d = x.unsqueeze(1);
    auto out_3d = at::conv1d(x_3d, w_bwd, b_bwd,
                             /*stride=*/at::IntArrayRef({1}),
                             /*padding=*/at::IntArrayRef({padding}),
                             /*dilation=*/at::IntArrayRef({dilation}));
    auto conv_out = out_3d.squeeze(1);

    // Backward through gated residual
    auto gr_grads = gated_residual_backward(grad_output, x, conv_out, gate_logit);
    auto grad_x_residual = gr_grads[0];
    auto grad_conv_out = gr_grads[1];
    auto grad_gate_logit = gr_grads[2];

    // Backward through conv1d
    auto grad_conv_3d = grad_conv_out.unsqueeze(1);

    std::array<bool,3> output_mask = {true, true, true};
    auto grad_inputs = at::convolution_backward(
        grad_conv_3d,
        x_3d,
        w_bwd,
        /*bias_sizes=*/at::OptionalIntArrayRef(at::IntArrayRef({bias.size(0)})),
        /*stride=*/at::IntArrayRef({1}),
        /*padding=*/at::IntArrayRef({padding}),
        /*dilation=*/at::IntArrayRef({dilation}),
        /*transposed=*/false,
        /*output_padding=*/at::IntArrayRef({0}),
        /*groups=*/1,
        /*output_mask=*/output_mask
    );

    auto grad_x_conv = std::get<0>(grad_inputs).squeeze(1);
    auto grad_w = std::get<1>(grad_inputs).to(weight.scalar_type());
    auto grad_b = std::get<2>(grad_inputs).to(bias.scalar_type());

    // Total grad_x = grad from residual path + grad from conv path
    auto grad_x = grad_x_residual + grad_x_conv;

    return {grad_x, grad_w, grad_b, grad_gate_logit};
}
