#include "common.cuh"

// =============================================================================
// SpatialConv2dOp: Depthwise Conv2d on [B, C, H, W] with gated residual
// Forward: out = conv2d(x, groups=C), output = (1-gate)*x + gate*out
// Uses at::conv2d (cuDNN) for the convolution.
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> spatial_conv2d_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gate_logit,
    int kernel_size
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(conv_weight);
    ASANN_CHECK_INPUT(conv_bias);
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");

    int C = x.size(1);
    int padding = kernel_size / 2;

    // Depthwise conv2d
    auto conv_out = at::conv2d(x, conv_weight, conv_bias,
                               /*stride=*/at::IntArrayRef({1, 1}),
                               /*padding=*/at::IntArrayRef({padding, padding}),
                               /*dilation=*/at::IntArrayRef({1, 1}),
                               /*groups=*/C);

    // Flatten for gated residual (works on any shape via .numel())
    // Actually gated_residual works element-wise, so shape doesn't matter
    // We need to reshape to flat for the kernel, or better: keep 4D
    // The gated residual kernel works on total_elements, so any shape is fine
    auto x_flat = x.reshape({-1});
    auto conv_flat = conv_out.reshape({-1});
    auto output_flat = gated_residual_forward(
        x.reshape({x.size(0), -1}),
        conv_out.reshape({x.size(0), -1}),
        gate_logit
    );
    auto output = output_flat.view_as(x);

    return {output, conv_out};
}

std::vector<torch::Tensor> spatial_conv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor conv_out,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gate_logit,
    int kernel_size
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int C = x.size(1);
    int padding = kernel_size / 2;

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(
        grad_output.reshape({x.size(0), -1}),
        x.reshape({x.size(0), -1}),
        conv_out.reshape({x.size(0), -1}),
        gate_logit
    );
    auto grad_x_gated = gated_grads[0].view_as(x);
    auto grad_conv_out = gated_grads[1].view_as(x);
    auto grad_gate_logit = gated_grads[2];

    // Backward through conv2d
    // Cast weight to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto w_bwd = conv_weight.to(act_dtype);

    std::array<bool,3> output_mask = {true, true, true};
    auto conv_grads = at::convolution_backward(
        grad_conv_out,
        x,
        w_bwd,
        /*bias_sizes=*/at::OptionalIntArrayRef(at::IntArrayRef({conv_bias.size(0)})),
        /*stride=*/at::IntArrayRef({1, 1}),
        /*padding=*/at::IntArrayRef({padding, padding}),
        /*dilation=*/at::IntArrayRef({1, 1}),
        /*transposed=*/false,
        /*output_padding=*/at::IntArrayRef({0, 0}),
        /*groups=*/C,
        /*output_mask=*/output_mask
    );

    auto grad_x_conv = std::get<0>(conv_grads);
    auto grad_conv_weight = std::get<1>(conv_grads).to(conv_weight.scalar_type());
    auto grad_conv_bias = std::get<2>(conv_grads).to(conv_bias.scalar_type());

    auto grad_x = grad_x_gated + grad_x_conv;

    return {grad_x, grad_conv_weight, grad_conv_bias, grad_gate_logit};
}
