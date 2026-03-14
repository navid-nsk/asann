#include "common.cuh"

// =============================================================================
// Conv1dBlock: Reshapes [B, d] -> [B, 1, d], conv1d(same-pad), -> [B, d]
// Uses at::conv1d (cuDNN) for the actual convolution.
// Forward: x_3d = x.unsqueeze(1), out = conv1d(x_3d).squeeze(1)
// Backward: through at::conv1d backward
// =============================================================================

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor conv1d_block_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(weight);
    ASANN_CHECK_INPUT(bias);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int padding = kernel_size / 2;

    // x: [B, d] -> [B, 1, d]
    auto x_3d = x.unsqueeze(1);
    // conv1d: [B, 1, d] -> [B, 1, d] (same padding)
    auto out_3d = at::conv1d(x_3d, weight, bias, /*stride=*/at::IntArrayRef({1}), /*padding=*/at::IntArrayRef({padding}));
    // [B, 1, d] -> [B, d]
    auto output = out_3d.squeeze(1);

    return output;
}

std::vector<torch::Tensor> conv1d_block_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(weight);

    int padding = kernel_size / 2;

    // Reshape for conv1d backward
    // grad_output: [B, d] -> [B, 1, d]
    auto grad_out_3d = grad_output.unsqueeze(1);
    auto x_3d = x.unsqueeze(1);

    // Cast weight to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto w_bwd = weight.to(act_dtype);

    // Use PyTorch's native backward computation
    std::array<bool,3> output_mask = {true, true, true};
    auto grad_inputs = at::convolution_backward(
        grad_out_3d,
        x_3d,
        w_bwd,
        /*bias_sizes=*/at::OptionalIntArrayRef(at::IntArrayRef({bias.size(0)})),
        /*stride=*/at::IntArrayRef({1}),
        /*padding=*/at::IntArrayRef({padding}),
        /*dilation=*/at::IntArrayRef({1}),
        /*transposed=*/false,
        /*output_padding=*/at::IntArrayRef({0}),
        /*groups=*/1,
        /*output_mask=*/output_mask
    );

    auto grad_x = std::get<0>(grad_inputs).squeeze(1);
    auto grad_w = std::get<1>(grad_inputs).to(weight.scalar_type());
    auto grad_b = std::get<2>(grad_inputs).to(bias.scalar_type());

    return {grad_x, grad_w, grad_b};
}
