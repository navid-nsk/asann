#include "common.cuh"

// =============================================================================
// DepthwiseSeparableConv2dOp:
//   dw_conv(x) -> BN -> ReLU -> pw_conv -> gated residual
// x: [B, C, H, W]
// Uses at::conv2d (cuDNN) for convolutions, at::batch_norm for BN.
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> dw_separable_conv2d_forward(
    torch::Tensor x,
    torch::Tensor dw_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor pw_weight,
    torch::Tensor pw_bias,
    torch::Tensor gate_logit,
    int kernel_size,
    bool training,
    float bn_momentum,
    float bn_eps
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(dw_weight);
    ASANN_CHECK_INPUT(pw_weight);
    ASANN_CHECK_INPUT(pw_bias);
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, C, H, W]");

    int C = x.size(1);
    int padding = kernel_size / 2;

    // Step 1: Depthwise conv2d (groups=C, no bias)
    auto dw_out = at::conv2d(x, dw_weight, /*bias=*/at::Tensor(),
                             /*stride=*/at::IntArrayRef({1, 1}),
                             /*padding=*/at::IntArrayRef({padding, padding}),
                             /*dilation=*/at::IntArrayRef({1, 1}),
                             /*groups=*/C);

    // Step 2: BatchNorm2d
    auto bn_out = at::batch_norm(dw_out, bn_weight, bn_bias,
                                  bn_running_mean, bn_running_var,
                                  training, bn_momentum, bn_eps, true);

    // Step 3: ReLU
    auto relu_out = at::relu(bn_out);

    // Step 4: Pointwise conv2d (1x1, groups=1)
    auto pw_out = at::conv2d(relu_out, pw_weight, pw_bias,
                             /*stride=*/at::IntArrayRef({1, 1}),
                             /*padding=*/at::IntArrayRef({0, 0}),
                             /*dilation=*/at::IntArrayRef({1, 1}),
                             /*groups=*/1);

    // Step 5: Gated residual
    auto output_flat = gated_residual_forward(
        x.reshape({x.size(0), -1}),
        pw_out.reshape({x.size(0), -1}),
        gate_logit
    );
    auto output = output_flat.view_as(x);

    return {output, dw_out, bn_out, relu_out, pw_out};
}

std::vector<torch::Tensor> dw_separable_conv2d_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor dw_out,
    torch::Tensor bn_out,
    torch::Tensor relu_out,
    torch::Tensor pw_out,
    torch::Tensor dw_weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    torch::Tensor pw_weight,
    torch::Tensor pw_bias,
    torch::Tensor gate_logit,
    int kernel_size,
    bool training,
    float bn_eps
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int C = x.size(1);
    (void)kernel_size;  // Used only by Python wrapper for full backward

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(
        grad_output.reshape({x.size(0), -1}),
        x.reshape({x.size(0), -1}),
        pw_out.reshape({x.size(0), -1}),
        gate_logit
    );
    auto grad_x_gated = gated_grads[0].view_as(x);
    auto grad_pw_out = gated_grads[1].view_as(x);
    auto grad_gate_logit = gated_grads[2];

    // Backward through pointwise conv2d
    // Cast weight to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = relu_out.scalar_type();
    auto pw_w_bwd = pw_weight.to(act_dtype);

    std::array<bool,3> pw_mask = {true, true, true};
    auto pw_grads = at::convolution_backward(
        grad_pw_out,
        relu_out,
        pw_w_bwd,
        /*bias_sizes=*/at::OptionalIntArrayRef(at::IntArrayRef({pw_bias.size(0)})),
        /*stride=*/at::IntArrayRef({1, 1}),
        /*padding=*/at::IntArrayRef({0, 0}),
        /*dilation=*/at::IntArrayRef({1, 1}),
        /*transposed=*/false,
        /*output_padding=*/at::IntArrayRef({0, 0}),
        /*groups=*/1,
        /*output_mask=*/pw_mask
    );
    auto grad_relu_out = std::get<0>(pw_grads);
    auto grad_pw_weight = std::get<1>(pw_grads).to(pw_weight.scalar_type());
    auto grad_pw_bias = std::get<2>(pw_grads).to(pw_bias.scalar_type());

    // Backward through ReLU
    auto grad_bn_out = grad_relu_out * (bn_out > 0).to(grad_relu_out.dtype());

    // NOTE: BN backward and depthwise conv backward are handled by the Python
    // wrapper using PyTorch autograd (since BN requires save_mean/save_invstd
    // from forward which are not easily accessible from C++).
    // This C++ backward only computes: gated_residual_backward + pw_conv_backward + relu_backward.
    // The Python wrapper uses native PyTorch ops for the full pipeline.

    return {grad_x_gated, grad_pw_weight, grad_pw_bias, grad_gate_logit,
            grad_bn_out, grad_relu_out};
}
