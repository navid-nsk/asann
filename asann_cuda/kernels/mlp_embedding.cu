#include "common.cuh"

// =============================================================================
// MLPEmbedding: out = x + decode(relu(encode(x)))
// x: [B, d], encode_w: [hidden, d], encode_b: [hidden]
// decode_w: [d, hidden], decode_b: [d]
// Forward: hidden_act = relu(x @ encode_w.T + encode_b)
//          correction = hidden_act @ decode_w.T + decode_b
//          out = x + correction
// Backward: chain through decode -> relu_backward -> encode
// =============================================================================

// Uses at::linear (cuBLAS) for matrix operations
// Custom kernel only for bookkeeping if needed

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> mlp_embedding_forward(
    torch::Tensor x,
    torch::Tensor encode_weight,
    torch::Tensor encode_bias,
    torch::Tensor decode_weight,
    torch::Tensor decode_bias
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(encode_weight);
    ASANN_CHECK_INPUT(encode_bias);
    ASANN_CHECK_INPUT(decode_weight);
    ASANN_CHECK_INPUT(decode_bias);

    // encode: [B, d] -> [B, hidden]
    auto encoded = at::linear(x, encode_weight, encode_bias);
    // relu
    auto hidden_act = at::relu(encoded);
    // decode: [B, hidden] -> [B, d]
    auto correction = at::linear(hidden_act, decode_weight, decode_bias);
    // residual
    auto output = x + correction;

    // Return output and intermediates needed for backward
    return {output, encoded, hidden_act};
}

std::vector<torch::Tensor> mlp_embedding_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor encoded,
    torch::Tensor hidden_act,
    torch::Tensor encode_weight,
    torch::Tensor encode_bias,
    torch::Tensor decode_weight,
    torch::Tensor decode_bias
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(encoded);
    ASANN_CHECK_INPUT(hidden_act);
    ASANN_CHECK_INPUT(encode_weight);
    ASANN_CHECK_INPUT(decode_weight);

    // Forward: out = x + decode(relu(encode(x)))
    // grad_correction = grad_output (from residual)
    auto grad_correction = grad_output;

    // Cast weights to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto dw_bwd = decode_weight.to(act_dtype);
    auto ew_bwd = encode_weight.to(act_dtype);

    // Backward through decode linear: correction = hidden_act @ decode_weight.T + decode_bias
    // grad_hidden_act = grad_correction @ decode_weight  [B, hidden]
    auto grad_hidden_act = at::matmul(grad_correction, dw_bwd);
    // grad_decode_weight = grad_correction.T @ hidden_act  [d, hidden]
    auto grad_decode_weight = at::matmul(grad_correction.t(), hidden_act);
    // grad_decode_bias = grad_correction.sum(0)  [d]
    auto grad_decode_bias = grad_correction.sum(0);

    // Backward through relu: grad_encoded = grad_hidden_act * (encoded > 0)
    auto grad_encoded = grad_hidden_act * (encoded > 0).to(grad_hidden_act.dtype());

    // Backward through encode linear: encoded = x @ encode_weight.T + encode_bias
    // grad_x_from_encode = grad_encoded @ encode_weight  [B, d]
    auto grad_x_from_encode = at::matmul(grad_encoded, ew_bwd);
    // grad_encode_weight = grad_encoded.T @ x  [hidden, d]
    auto grad_encode_weight = at::matmul(grad_encoded.t(), x);
    // grad_encode_bias = grad_encoded.sum(0)  [hidden]
    auto grad_encode_bias = grad_encoded.sum(0);

    // Total grad_x = grad_output (identity path) + grad from correction path
    auto grad_x = grad_output + grad_x_from_encode;

    return {grad_x, grad_encode_weight, grad_encode_bias,
            grad_decode_weight, grad_decode_bias};
}
