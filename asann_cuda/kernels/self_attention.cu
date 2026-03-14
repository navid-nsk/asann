#include "common.cuh"

// =============================================================================
// SelfAttentionOp: Data-dependent feature self-attention
// x: [B, d], Q_emb/K_emb/V_emb: [d, rank], out_proj: [rank], gate_logit: scalar
//
// Forward:
//   Q = x.unsqueeze(-1) * Q_emb.unsqueeze(0)  -> [B, d, rank]
//   K = x.unsqueeze(-1) * K_emb.unsqueeze(0)  -> [B, d, rank]
//   V = x.unsqueeze(-1) * V_emb.unsqueeze(0)  -> [B, d, rank]
//   attn = softmax(bmm(Q, K.T) / sqrt(rank))  -> [B, d, d]
//   weighted = bmm(attn, V)                    -> [B, d, rank]
//   out = (weighted * out_proj).sum(-1)         -> [B, d]
//   output = (1-gate)*x + gate*out
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> self_attention_forward(
    torch::Tensor x,
    torch::Tensor Q_emb,
    torch::Tensor K_emb,
    torch::Tensor V_emb,
    torch::Tensor out_proj,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(Q_emb);
    ASANN_CHECK_INPUT(K_emb);
    ASANN_CHECK_INPUT(V_emb);
    ASANN_CHECK_INPUT(out_proj);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);
    int rank = Q_emb.size(1);
    float scale = sqrtf((float)rank);

    // Q, K, V: [B, d, rank] = x[:, :, None] * emb[None, :, :]
    auto x_exp = x.unsqueeze(-1);  // [B, d, 1]
    auto Q = x_exp * Q_emb.unsqueeze(0);  // [B, d, rank]
    auto K = x_exp * K_emb.unsqueeze(0);  // [B, d, rank]
    auto V = x_exp * V_emb.unsqueeze(0);  // [B, d, rank]

    // Attention scores: [B, d, d]
    auto scores = at::bmm(Q, K.transpose(1, 2)) / scale;
    auto attn = at::softmax(scores, -1);  // [B, d, d]

    // Weighted values: [B, d, rank]
    auto weighted = at::bmm(attn, V);

    // Project to scalar per feature: [B, d]
    auto out = (weighted * out_proj.unsqueeze(0).unsqueeze(0)).sum(-1);

    // Gated residual
    auto output = gated_residual_forward(x, out, gate_logit);

    return {output, Q, K, V, attn, weighted, out};
}

std::vector<torch::Tensor> self_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor attn, torch::Tensor weighted, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor K_emb, torch::Tensor V_emb,
    torch::Tensor out_proj,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int d = x.size(1);
    int rank = Q_emb.size(1);
    float scale = sqrtf((float)rank);

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(grad_output, x, out, gate_logit);
    auto grad_x_gated = gated_grads[0];
    auto grad_out = gated_grads[1];
    auto grad_gate_logit = gated_grads[2];

    // Cast parameters to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto op_bwd = out_proj.to(act_dtype);
    auto qe_bwd = Q_emb.to(act_dtype);
    auto ke_bwd = K_emb.to(act_dtype);
    auto ve_bwd = V_emb.to(act_dtype);

    // Backward through projection: out = (weighted * out_proj).sum(-1)
    // grad_weighted[b][i][r] = grad_out[b][i] * out_proj[r]
    auto grad_weighted = grad_out.unsqueeze(-1) * op_bwd.unsqueeze(0).unsqueeze(0);  // [B, d, rank]
    // grad_out_proj[r] = sum_{b,i} grad_out[b][i] * weighted[b][i][r]
    auto grad_out_proj = (grad_out.unsqueeze(-1) * weighted).sum(at::IntArrayRef({0, 1}));  // [rank]

    // Backward through weighted = bmm(attn, V)
    // grad_attn = grad_weighted @ V.T  -> [B, d, d]
    auto grad_attn = at::bmm(grad_weighted, V.transpose(1, 2));
    // grad_V = attn.T @ grad_weighted -> [B, d, rank]
    auto grad_V = at::bmm(attn.transpose(1, 2), grad_weighted);

    // Backward through softmax
    // grad_scores = attn * (grad_attn - (grad_attn * attn).sum(-1, keepdim=True))
    auto sum_ga = (grad_attn * attn).sum(-1, true);
    auto grad_scores = attn * (grad_attn - sum_ga) / scale;

    // Backward through scores = bmm(Q, K.T) / scale
    // (scale division already handled above)
    auto grad_Q = at::bmm(grad_scores, K);  // [B, d, rank]
    auto grad_K = at::bmm(grad_scores.transpose(1, 2), Q);  // [B, d, rank]

    // Backward through Q = x[:,:,None] * Q_emb[None,:,:]
    // grad_x from Q: sum over rank of (grad_Q * Q_emb)
    auto x_exp = x.unsqueeze(-1);  // [B, d, 1]
    auto grad_x_from_Q = (grad_Q * qe_bwd.unsqueeze(0)).sum(-1);  // [B, d]
    auto grad_Q_emb = (grad_Q * x_exp).sum(0);  // [d, rank]

    auto grad_x_from_K = (grad_K * ke_bwd.unsqueeze(0)).sum(-1);
    auto grad_K_emb = (grad_K * x_exp).sum(0);

    auto grad_x_from_V = (grad_V * ve_bwd.unsqueeze(0)).sum(-1);
    auto grad_V_emb = (grad_V * x_exp).sum(0);

    // Total grad_x
    auto grad_x = grad_x_gated + grad_x_from_Q + grad_x_from_K + grad_x_from_V;

    return {grad_x, grad_Q_emb, grad_K_emb, grad_V_emb, grad_out_proj, grad_gate_logit};
}
