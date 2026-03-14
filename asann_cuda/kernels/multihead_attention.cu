#include "common.cuh"

// =============================================================================
// MultiHeadAttentionOp: Multi-head data-dependent feature attention
// x: [B, d]
// Q_emb/K_emb/V_emb: [H, d, r_h], out_proj: [H, r_h], head_weights: [H]
// gate_logit: scalar
//
// Forward:
//   x_exp = x[:, None, :, None]              -> [B, 1, d, 1]
//   Q = x_exp * Q_emb[None, :, :, :]         -> [B, H, d, r_h]
//   K = x_exp * K_emb[None, :, :, :]
//   V = x_exp * V_emb[None, :, :, :]
//   attn = softmax(einsum('bhir,bhjr->bhij', Q, K) / sqrt(r_h))  -> [B, H, d, d]
//   weighted = einsum('bhij,bhjr->bhir', attn, V)                 -> [B, H, d, r_h]
//   head_out = (weighted * out_proj[None,:,None,:]).sum(-1)        -> [B, H, d]
//   hw = softmax(head_weights)                                     -> [H]
//   out = (head_out * hw[None,:,None]).sum(1)                     -> [B, d]
//   output = (1-gate)*x + gate*out
// =============================================================================

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> multihead_attention_forward(
    torch::Tensor x,
    torch::Tensor Q_emb,
    torch::Tensor K_emb,
    torch::Tensor V_emb,
    torch::Tensor out_proj,
    torch::Tensor head_weights,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(x);
    ASANN_CHECK_INPUT(Q_emb);
    ASANN_CHECK_INPUT(K_emb);
    ASANN_CHECK_INPUT(V_emb);
    ASANN_CHECK_INPUT(out_proj);
    ASANN_CHECK_INPUT(head_weights);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, d]");

    int B = x.size(0);
    int d = x.size(1);
    int H = Q_emb.size(0);
    int r_h = Q_emb.size(2);
    float scale = sqrtf((float)r_h);

    // x_exp: [B, 1, d, 1]
    auto x_exp = x.unsqueeze(1).unsqueeze(-1);
    // Q, K, V: [B, H, d, r_h]
    auto Q = x_exp * Q_emb.unsqueeze(0);
    auto K = x_exp * K_emb.unsqueeze(0);
    auto V = x_exp * V_emb.unsqueeze(0);

    // Attention per head: einsum('bhir,bhjr->bhij') = bmm over (B*H) dimension
    // Reshape: [B*H, d, r_h]
    auto Q_r = Q.reshape({B * H, d, r_h});
    auto K_r = K.reshape({B * H, d, r_h});
    auto V_r = V.reshape({B * H, d, r_h});

    auto scores = at::bmm(Q_r, K_r.transpose(1, 2)) / scale;  // [B*H, d, d]
    auto attn = at::softmax(scores, -1);  // [B*H, d, d]

    // Weighted values: [B*H, d, r_h]
    auto weighted = at::bmm(attn, V_r);
    auto weighted_4d = weighted.reshape({B, H, d, r_h});

    // Project each head to scalar per feature: [B, H, d]
    auto head_out = (weighted_4d * out_proj.unsqueeze(0).unsqueeze(2)).sum(-1);

    // Mix heads: softmax(head_weights) -> [H]
    auto hw = at::softmax(head_weights, 0);
    auto out = (head_out * hw.unsqueeze(0).unsqueeze(-1)).sum(1);  // [B, d]

    // Gated residual
    auto output = gated_residual_forward(x, out, gate_logit);

    // Reshape attn back to [B, H, d, d] for backward
    auto attn_4d = attn.reshape({B, H, d, d});

    return {output, Q, K, V, attn_4d, weighted_4d, head_out, hw, out};
}

std::vector<torch::Tensor> multihead_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor attn_4d, torch::Tensor weighted_4d,
    torch::Tensor head_out, torch::Tensor hw, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor K_emb, torch::Tensor V_emb,
    torch::Tensor out_proj, torch::Tensor head_weights,
    torch::Tensor gate_logit
) {
    ASANN_CHECK_INPUT(grad_output);
    ASANN_CHECK_INPUT(x);

    int B = x.size(0);
    int d = x.size(1);
    int H = Q_emb.size(0);
    int r_h = Q_emb.size(2);
    float scale = sqrtf((float)r_h);

    // Backward through gated residual
    auto gated_grads = gated_residual_backward(grad_output, x, out, gate_logit);
    auto grad_x_gated = gated_grads[0];
    auto grad_out = gated_grads[1];
    auto grad_gate_logit = gated_grads[2];

    // Backward through head mixing: out = (head_out * hw[None,:,None]).sum(1)
    // grad_head_out[b][h][i] = grad_out[b][i] * hw[h]
    auto grad_head_out = grad_out.unsqueeze(1) * hw.unsqueeze(0).unsqueeze(-1);  // [B, H, d]
    // grad_hw[h] = sum_{b,i} grad_out[b][i] * head_out[b][h][i]
    auto grad_hw_raw = (grad_out.unsqueeze(1) * head_out).sum(at::IntArrayRef({0, 2}));  // [H]

    // Backward through softmax(head_weights)
    // grad_head_weights = hw * (grad_hw_raw - (hw * grad_hw_raw).sum())
    auto grad_head_weights = hw * (grad_hw_raw - (hw * grad_hw_raw).sum());

    // Cast parameters to match activation dtype (FP16 under AMP, custom_bwd disables autocast)
    auto act_dtype = x.scalar_type();
    auto op_bwd = out_proj.to(act_dtype);
    auto qe_bwd = Q_emb.to(act_dtype);
    auto ke_bwd = K_emb.to(act_dtype);
    auto ve_bwd = V_emb.to(act_dtype);

    // Backward through head projection: head_out = (weighted_4d * out_proj[None,:,None,:]).sum(-1)
    // grad_weighted_4d[b][h][i][r] = grad_head_out[b][h][i] * out_proj[h][r]
    auto grad_weighted_4d = grad_head_out.unsqueeze(-1) * op_bwd.unsqueeze(0).unsqueeze(2);
    // grad_out_proj[h][r] = sum_{b,i} grad_head_out[b][h][i] * weighted_4d[b][h][i][r]
    auto grad_out_proj = (grad_head_out.unsqueeze(-1) * weighted_4d).sum(at::IntArrayRef({0, 2}));  // [H, r_h]

    // Reshape for bmm operations: [B*H, d, r_h]
    auto grad_weighted = grad_weighted_4d.reshape({B * H, d, r_h});
    auto attn = attn_4d.reshape({B * H, d, d});
    auto V_r = V.reshape({B * H, d, r_h});
    auto Q_r = Q.reshape({B * H, d, r_h});
    auto K_r = K.reshape({B * H, d, r_h});

    // Backward through weighted = bmm(attn, V)
    auto grad_attn = at::bmm(grad_weighted, V_r.transpose(1, 2));  // [B*H, d, d]
    auto grad_V_r = at::bmm(attn.transpose(1, 2), grad_weighted);  // [B*H, d, r_h]

    // Backward through softmax
    auto sum_ga = (grad_attn * attn).sum(-1, true);
    auto grad_scores = attn * (grad_attn - sum_ga) / scale;

    // Backward through scores = bmm(Q, K.T) / scale
    auto grad_Q_r = at::bmm(grad_scores, K_r);
    auto grad_K_r = at::bmm(grad_scores.transpose(1, 2), Q_r);

    // Reshape back to [B, H, d, r_h]
    auto grad_Q = grad_Q_r.reshape({B, H, d, r_h});
    auto grad_K = grad_K_r.reshape({B, H, d, r_h});
    auto grad_V = grad_V_r.reshape({B, H, d, r_h});

    // Backward through Q = x_exp * Q_emb
    // x_exp: [B, 1, d, 1]
    auto x_exp = x.unsqueeze(1).unsqueeze(-1);
    auto grad_x_from_Q = (grad_Q * qe_bwd.unsqueeze(0)).sum(at::IntArrayRef({1, 3}));  // [B, d]
    auto grad_Q_emb = (grad_Q * x_exp).sum(0);  // [H, d, r_h]

    auto grad_x_from_K = (grad_K * ke_bwd.unsqueeze(0)).sum(at::IntArrayRef({1, 3}));
    auto grad_K_emb = (grad_K * x_exp).sum(0);

    auto grad_x_from_V = (grad_V * ve_bwd.unsqueeze(0)).sum(at::IntArrayRef({1, 3}));
    auto grad_V_emb = (grad_V * x_exp).sum(0);

    auto grad_x = grad_x_gated + grad_x_from_Q + grad_x_from_K + grad_x_from_V;

    return {grad_x, grad_Q_emb, grad_K_emb, grad_V_emb,
            grad_out_proj, grad_head_weights, grad_gate_logit};
}
