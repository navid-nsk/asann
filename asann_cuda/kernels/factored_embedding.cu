#include "common.cuh"

// =============================================================================
// FactoredEmbedding: out = x + x @ V.T @ U.T
// x: [B, d], U: [d, rank], V: [rank, d]
// Forward: correction = x @ V.T @ U.T, out = x + correction
// Backward: uses chain rule through two matmuls
// =============================================================================

// Forward: uses at::matmul for the matrix multiplications (cuBLAS)
// The C++ interface handles the computation.

// =============================================================================
// C++ Interface
// =============================================================================

torch::Tensor factored_embedding_forward(
    torch::Tensor x,
    torch::Tensor U,
    torch::Tensor V
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(U);
    ASANN_CHECK_INPUT(V);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");
    TORCH_CHECK(U.dim() == 2, "U must be 2D [d, rank]");
    TORCH_CHECK(V.dim() == 2, "V must be 2D [rank, d]");
    TORCH_CHECK(x.size(1) == U.size(0), "x.size(1) must match U.size(0)");
    TORCH_CHECK(U.size(1) == V.size(0), "U.size(1) must match V.size(0)");
    TORCH_CHECK(V.size(1) == x.size(1), "V.size(1) must match x.size(1)");

    // correction = x @ V.T @ U.T
    // Step 1: x @ V.T -> [B, rank]
    auto xV = at::matmul(x, V.t());
    // Step 2: [B, rank] @ U.T -> [B, d]
    auto correction = at::matmul(xV, U.t());
    // out = x + correction
    auto output = x + correction;

    return output;
}

std::vector<torch::Tensor> factored_embedding_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor U,
    torch::Tensor V
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(U);
    ASANN_CHECK_INPUT(V);

    // Forward was: out = x + x @ V.T @ U.T
    // Let correction = x @ V.T @ U.T
    // Let intermediate = x @ V.T   (shape [B, rank])
    // correction = intermediate @ U.T  (shape [B, d])
    //
    // grad_out is dL/d(out), shape [B, d]
    // dL/dx = dL/d(out) * d(out)/dx
    //       = dL/d(out) * (I + V.T @ U.T)  -- but we compute via chain rule
    //       = grad_output + grad_output @ U @ V  (grad through correction path)

    // Backward through correction = intermediate @ U.T:
    // grad_intermediate = grad_output @ U  (shape [B, rank])
    // grad_U = intermediate.T @ grad_output  -- but grad w.r.t. U.T means:
    //   d(loss)/d(U) = grad_output.T @ intermediate  -> transpose
    //   Actually: correction = intermediate @ U.T
    //   d(loss)/d(U.T) = intermediate.T @ grad_output  -> [rank, d]
    //   d(loss)/d(U) = grad_output.T @ intermediate    -> [d, rank]

    // Cast weights to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto U_bwd = U.to(act_dtype);
    auto V_bwd = V.to(act_dtype);

    auto intermediate = at::matmul(x, V_bwd.t());  // [B, rank]

    // grad_correction = grad_output (since out = x + correction)
    auto grad_correction = grad_output;

    // Backward through correction = intermediate @ U.T
    auto grad_intermediate = at::matmul(grad_correction, U_bwd);  // [B, rank]
    // grad_U: correction_j = sum_k intermediate_k * U_j_k
    // d(correction_j)/d(U_j_k) = intermediate_k
    // grad_U = grad_correction.T @ intermediate -> [d, rank]
    auto grad_U = at::matmul(grad_correction.t(), intermediate);  // [d, rank]

    // Backward through intermediate = x @ V.T
    auto grad_x_from_correction = at::matmul(grad_intermediate, V_bwd);  // [B, d]
    // grad_V: intermediate = x @ V.T
    // d(intermediate)/d(V.T) => grad_V.T = x.T @ grad_intermediate -> [d, rank]
    // grad_V = grad_intermediate.T @ x -> [rank, d]
    auto grad_V = at::matmul(grad_intermediate.t(), x);  // [rank, d]

    // Total grad_x = grad_output (from identity) + grad from correction path
    auto grad_x = grad_output + grad_x_from_correction;

    return {grad_x, grad_U, grad_V};
}
