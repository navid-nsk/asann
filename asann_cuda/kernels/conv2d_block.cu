#include "common.cuh"

// =============================================================================
// Conv2dBlock: flat [B, d] -> reshape [B, C, H, W] -> depthwise conv2d -> reshape -> gated residual
// Uses at::conv2d (cuDNN) for the actual convolution.
// =============================================================================

// Forward: uses gated_residual kernel + at::conv2d
// Backward: at::convolution_backward + gated_residual_backward

// =============================================================================
// C++ Interface
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

std::vector<torch::Tensor> conv2d_block_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gate_logit,
    int C, int H, int W,
    int kernel_size
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(conv_weight);
    ASANN_CHECK_INPUT(conv_bias);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int padding = kernel_size / 2;

    // Reshape [B, d] -> [B, C, H, W]
    auto img = x.view({B, C, H, W});

    // Depthwise conv2d (groups=C)
    auto conv_out = at::conv2d(img, conv_weight, conv_bias,
                               /*stride=*/at::IntArrayRef({1, 1}),
                               /*padding=*/at::IntArrayRef({padding, padding}),
                               /*dilation=*/at::IntArrayRef({1, 1}),
                               /*groups=*/C);

    // Reshape back to [B, d]
    auto transformed = conv_out.reshape({B, -1});

    // Gated residual: (1 - gate) * x + gate * transformed
    auto output = gated_residual_forward(x, transformed, gate_logit);

    // Return output and intermediates
    return {output, img, conv_out, transformed};
}

std::vector<torch::Tensor> conv2d_block_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor img,
    torch::Tensor conv_out,
    torch::Tensor transformed,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gate_logit,
    int C, int H, int W,
    int kernel_size
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int padding = kernel_size / 2;

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(grad_output, x, transformed, gate_logit);
    auto grad_x = gated_grads[0];
    auto grad_transformed = gated_grads[1];
    auto grad_gate_logit = gated_grads[2];

    // Reshape grad_transformed [B, d] -> [B, C, H, W]
    auto grad_conv_out = grad_transformed.view({B, C, H, W});

    // Backward through conv2d
    // Cast weight to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = img.scalar_type();
    auto w_bwd = conv_weight.to(act_dtype);

    std::array<bool,3> output_mask = {true, true, true};
    auto conv_grads = at::convolution_backward(
        grad_conv_out,
        img,
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

    auto grad_img = std::get<0>(conv_grads);
    auto grad_conv_weight = std::get<1>(conv_grads).to(conv_weight.scalar_type());
    auto grad_conv_bias = std::get<2>(conv_grads).to(conv_bias.scalar_type());

    // grad_img: [B, C, H, W] -> [B, d]
    auto grad_x_from_conv = grad_img.reshape({B, -1});

    // Total grad_x
    grad_x = grad_x + grad_x_from_conv;

    return {grad_x, grad_conv_weight, grad_conv_bias, grad_gate_logit};
}
