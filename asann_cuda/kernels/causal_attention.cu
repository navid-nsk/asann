#include "common.cuh"

// =============================================================================
// CausalAttentionOp: Same as SelfAttention but with causal (lower-triangular) mask
// x: [B, d], Q_emb/K_emb/V_emb: [d, rank], out_proj: [rank], gate_logit: scalar
//
// Forward:
//   Q, K, V computed same as SelfAttention
//   scores = bmm(Q, K.T) / sqrt(rank)
//   scores = scores.masked_fill(~causal_mask, -inf)
//   attn = softmax(scores)
//   weighted = bmm(attn, V)
//   out = (weighted * out_proj).sum(-1)
//   output = (1-gate)*x + gate*out
// =============================================================================

// Causal mask kernel: fill upper triangle with -inf
template <typename scalar_t>
__global__ void causal_mask_kernel(
    scalar_t* __restrict__ scores,
    const int B,
    const int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * d * d;
    if (idx >= total) return;

    int i = (idx / d) % d;
    int j = idx % d;

    // Causal: feature i can only attend to features j <= i
    if (j > i) {
        scores[idx] = from_float<scalar_t>(-INFINITY);
    }
}

// Forward declarations for gated residual
torch::Tensor gated_residual_forward(torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);

// =============================================================================
// C++ Interface
// =============================================================================

std::vector<torch::Tensor> causal_attention_forward(
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

    // Q, K, V: [B, d, rank]
    auto x_exp = x.unsqueeze(-1);
    auto Q = x_exp * Q_emb.unsqueeze(0);
    auto K = x_exp * K_emb.unsqueeze(0);
    auto V = x_exp * V_emb.unsqueeze(0);

    // Attention scores: [B, d, d]
    auto scores = at::bmm(Q, K.transpose(1, 2)) / scale;

    // Apply causal mask: upper triangle -> -inf
    int total_mask = B * d * d;
    int threads = ASANN_THREADS_PER_BLOCK;
    int blocks = asann_get_blocks(total_mask, threads);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scores.scalar_type(), "causal_mask", [&] {
        causal_mask_kernel<scalar_t><<<blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            B, d
        );
    });
    ASANN_DEBUG_SYNC();

    auto attn = at::softmax(scores, -1);  // [B, d, d]

    // Weighted values: [B, d, rank]
    auto weighted = at::bmm(attn, V);

    // Project to scalar per feature: [B, d]
    auto out = (weighted * out_proj.unsqueeze(0).unsqueeze(0)).sum(-1);

    // Gated residual
    auto output = gated_residual_forward(x, out, gate_logit);

    return {output, Q, K, V, scores, attn, weighted, out};
}

std::vector<torch::Tensor> causal_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor scores, torch::Tensor attn,
    torch::Tensor weighted, torch::Tensor out,
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

    // Backward through projection
    auto grad_weighted = grad_out.unsqueeze(-1) * op_bwd.unsqueeze(0).unsqueeze(0);
    auto grad_out_proj = (grad_out.unsqueeze(-1) * weighted).sum(at::IntArrayRef({0, 1}));

    // Backward through weighted = bmm(attn, V)
    auto grad_attn = at::bmm(grad_weighted, V.transpose(1, 2));
    auto grad_V = at::bmm(attn.transpose(1, 2), grad_weighted);

    // Backward through softmax (with causal mask)
    // The mask positions contribute 0 gradient (attn is 0 there after softmax with -inf)
    auto sum_ga = (grad_attn * attn).sum(-1, true);
    auto grad_scores = attn * (grad_attn - sum_ga) / scale;

    // Backward through scores = bmm(Q, K.T) / scale
    auto grad_Q = at::bmm(grad_scores, K);
    auto grad_K = at::bmm(grad_scores.transpose(1, 2), Q);

    // Backward through Q/K/V = x_exp * emb
    auto x_exp = x.unsqueeze(-1);
    auto grad_x_from_Q = (grad_Q * qe_bwd.unsqueeze(0)).sum(-1);
    auto grad_Q_emb = (grad_Q * x_exp).sum(0);

    auto grad_x_from_K = (grad_K * ke_bwd.unsqueeze(0)).sum(-1);
    auto grad_K_emb = (grad_K * x_exp).sum(0);

    auto grad_x_from_V = (grad_V * ve_bwd.unsqueeze(0)).sum(-1);
    auto grad_V_emb = (grad_V * x_exp).sum(0);

    auto grad_x = grad_x_gated + grad_x_from_Q + grad_x_from_K + grad_x_from_V;

    return {grad_x, grad_Q_emb, grad_K_emb, grad_V_emb, grad_out_proj, grad_gate_logit};
}
