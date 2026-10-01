#include "common.cuh"

// =============================================================================
// CrossAttentionOp: Cross-attention between features and learned memory bank
// x: [B, d], Q_emb: [d, rank], mem_keys: [M, rank], mem_values: [M, d]
// gate_logit: scalar
//
// Forward:
//   Q = x.unsqueeze(-1) * Q_emb.unsqueeze(0)       -> [B, d, rank]
//   attn = softmax(einsum('bdr,mr->bdm', Q, mem_keys) / sqrt(rank))  -> [B, d, M]
//   out = einsum('bdm,md->bd', attn, mem_values)    -> [B, d]
//   output = (1-gate)*x + gate*out
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> cross_attention_forward(
    torch::Tensor x,
    torch::Tensor Q_emb,
    torch::Tensor mem_keys,
    torch::Tensor mem_values,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(Q_emb);
    ASANN_CHECK_INPUT(mem_keys);
    ASANN_CHECK_INPUT(mem_values);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);
    int rank = Q_emb.size(1);
    int M = mem_keys.size(0);
    float scale = sqrtf((float)rank);

    // Q: [B, d, rank]
    auto Q = x.unsqueeze(-1) * Q_emb.unsqueeze(0);

    // Attention to memory: einsum('bdr,mr->bdm')
    // = Q @ mem_keys.T -> [B, d, M]  (batched matmul)
    auto scores = at::matmul(Q, mem_keys.t()) / scale;  // [B, d, M]
    auto attn = at::softmax(scores, -1);  // [B, d, M]

    // Retrieve from memory: einsum('bdm,md->bd')
    // For each b,d: out[b][d_idx] = sum_m attn[b][d_idx][m] * mem_values[m][d_idx]
    // This is element-wise: out = (attn * mem_values.T.unsqueeze(0)).sum(-1)
    // mem_values: [M, d] -> mem_values.T: [d, M]
    auto out = (attn * mem_values.t().unsqueeze(0)).sum(-1);  // [B, d]

    // Gated residual
    auto output = gated_residual_forward(x, out, gate_logit);

    return {output, Q, attn, out};
}

std::vector<torch::Tensor> cross_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor Q, torch::Tensor attn, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor mem_keys, torch::Tensor mem_values,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int d = x.size(1);
    int rank = Q_emb.size(1);
    int M = mem_keys.size(0);
    float scale = sqrtf((float)rank);

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(grad_output, x, out, gate_logit);
    auto grad_x_gated = gated_grads[0];
    auto grad_out = gated_grads[1];
    auto grad_gate_logit = gated_grads[2];

    // Cast parameters to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto mk_bwd = mem_keys.to(act_dtype);
    auto mv_bwd = mem_values.to(act_dtype);
    auto qe_bwd = Q_emb.to(act_dtype);

    // Backward through out = (attn * mem_values.T.unsqueeze(0)).sum(-1)
    // out[b][i] = sum_m attn[b][i][m] * mem_values[m][i]
    // grad_attn[b][i][m] = grad_out[b][i] * mem_values[m][i]
    auto grad_attn = grad_out.unsqueeze(-1) * mv_bwd.t().unsqueeze(0);  // [B, d, M]
    // grad_mem_values[m][i] = sum_b attn[b][i][m] * grad_out[b][i]
    auto grad_mem_values = at::einsum("bdm,bd->md", {attn, grad_out});  // [M, d]

    // Backward through softmax
    auto sum_ga = (grad_attn * attn).sum(-1, true);
    auto grad_scores = attn * (grad_attn - sum_ga) / scale;

    // Backward through scores = Q @ mem_keys.T / scale (scale handled above)
    // grad_Q = grad_scores @ mem_keys  -> [B, d, rank]
    auto grad_Q = at::matmul(grad_scores, mk_bwd);
    // grad_mem_keys = sum_b grad_scores.T @ Q[b]
    // grad_scores: [B, d, M], Q: [B, d, rank]
    // grad_mem_keys[m][r] = sum_{b,d} grad_scores[b][d][m] * Q[b][d][r]
    auto grad_mem_keys = at::einsum("bdm,bdr->mr", {grad_scores, Q});  // [M, rank]

    // Backward through Q = x[:,:,None] * Q_emb[None,:,:]
    auto x_exp = x.unsqueeze(-1);
    auto grad_x_from_Q = (grad_Q * qe_bwd.unsqueeze(0)).sum(-1);  // [B, d]
    auto grad_Q_emb = (grad_Q * x_exp).sum(0);  // [d, rank]

    auto grad_x = grad_x_gated + grad_x_from_Q;

    return {grad_x, grad_Q_emb, grad_mem_keys, grad_mem_values, grad_gate_logit};
}
