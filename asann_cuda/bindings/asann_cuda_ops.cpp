#include <torch/extension.h>
#include <vector>

// =============================================================================
// Forward declarations for all CUDA operations
// =============================================================================

// --- Gated Residual ---
torch::Tensor gated_residual_forward(
    torch::Tensor x, torch::Tensor transformed, torch::Tensor gate_logit);
std::vector<torch::Tensor> gated_residual_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor transformed, torch::Tensor gate_logit);

// --- Positional Embedding ---
torch::Tensor positional_embedding_forward(
    torch::Tensor x, torch::Tensor pos_emb);
std::vector<torch::Tensor> positional_embedding_backward(
    torch::Tensor grad_output, int batch_size, int d);

// --- Factored Embedding ---
torch::Tensor factored_embedding_forward(
    torch::Tensor x, torch::Tensor U, torch::Tensor V);
std::vector<torch::Tensor> factored_embedding_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor U, torch::Tensor V);

// --- MLP Embedding ---
std::vector<torch::Tensor> mlp_embedding_forward(
    torch::Tensor x, torch::Tensor encode_weight, torch::Tensor encode_bias,
    torch::Tensor decode_weight, torch::Tensor decode_bias);
std::vector<torch::Tensor> mlp_embedding_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor encoded, torch::Tensor hidden_act,
    torch::Tensor encode_weight, torch::Tensor encode_bias,
    torch::Tensor decode_weight, torch::Tensor decode_bias);

// --- Geometric Embedding ---
std::vector<torch::Tensor> geometric_embedding_forward(
    torch::Tensor x, torch::Tensor radial_scale,
    torch::Tensor bias, torch::Tensor gate_logit);
std::vector<torch::Tensor> geometric_embedding_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor scaled, torch::Tensor norm, torch::Tensor poincare,
    torch::Tensor radial_scale, torch::Tensor bias, torch::Tensor gate_logit);

// --- GAP Flatten ---
torch::Tensor gap_flatten_forward(torch::Tensor x);
torch::Tensor gap_flatten_backward(
    torch::Tensor grad_output, int B, int C, int H, int W);

// --- Conv1d Block ---
torch::Tensor conv1d_block_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int kernel_size);
std::vector<torch::Tensor> conv1d_block_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor weight, torch::Tensor bias, int kernel_size);

// --- Conv2d Block ---
std::vector<torch::Tensor> conv2d_block_forward(
    torch::Tensor x, torch::Tensor conv_weight, torch::Tensor conv_bias,
    torch::Tensor gate_logit, int C, int H, int W, int kernel_size);
std::vector<torch::Tensor> conv2d_block_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor img, torch::Tensor conv_out, torch::Tensor transformed,
    torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor gate_logit,
    int C, int H, int W, int kernel_size);

// --- Spatial Conv2d ---
std::vector<torch::Tensor> spatial_conv2d_forward(
    torch::Tensor x, torch::Tensor conv_weight, torch::Tensor conv_bias,
    torch::Tensor gate_logit, int kernel_size);
std::vector<torch::Tensor> spatial_conv2d_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor conv_out,
    torch::Tensor conv_weight, torch::Tensor conv_bias,
    torch::Tensor gate_logit, int kernel_size);

// --- Pointwise Conv2d ---
std::vector<torch::Tensor> pointwise_conv2d_forward(
    torch::Tensor x, torch::Tensor conv_weight, torch::Tensor conv_bias,
    torch::Tensor gate_logit);
std::vector<torch::Tensor> pointwise_conv2d_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor conv_out,
    torch::Tensor conv_weight, torch::Tensor conv_bias, torch::Tensor gate_logit);

// --- Depthwise Separable Conv2d ---
std::vector<torch::Tensor> dw_separable_conv2d_forward(
    torch::Tensor x, torch::Tensor dw_weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_running_mean, torch::Tensor bn_running_var,
    torch::Tensor pw_weight, torch::Tensor pw_bias,
    torch::Tensor gate_logit,
    int kernel_size, bool training, float bn_momentum, float bn_eps);
std::vector<torch::Tensor> dw_separable_conv2d_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor dw_out, torch::Tensor bn_out,
    torch::Tensor relu_out, torch::Tensor pw_out,
    torch::Tensor dw_weight, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_running_mean, torch::Tensor bn_running_var,
    torch::Tensor pw_weight, torch::Tensor pw_bias, torch::Tensor gate_logit,
    int kernel_size, bool training, float bn_eps);

// --- Channel Attention ---
std::vector<torch::Tensor> channel_attention_forward(
    torch::Tensor x, torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias, torch::Tensor gate_logit);
std::vector<torch::Tensor> channel_attention_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor squeezed, torch::Tensor fc1_out, torch::Tensor relu_out,
    torch::Tensor fc2_out, torch::Tensor attn_weights, torch::Tensor rescaled,
    torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias, torch::Tensor gate_logit);

// --- Self Attention ---
std::vector<torch::Tensor> self_attention_forward(
    torch::Tensor x, torch::Tensor Q_emb, torch::Tensor K_emb,
    torch::Tensor V_emb, torch::Tensor out_proj, torch::Tensor gate_logit);
std::vector<torch::Tensor> self_attention_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor attn, torch::Tensor weighted, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor K_emb, torch::Tensor V_emb,
    torch::Tensor out_proj, torch::Tensor gate_logit);

// --- Multi-Head Attention ---
std::vector<torch::Tensor> multihead_attention_forward(
    torch::Tensor x, torch::Tensor Q_emb, torch::Tensor K_emb,
    torch::Tensor V_emb, torch::Tensor out_proj,
    torch::Tensor head_weights, torch::Tensor gate_logit);
std::vector<torch::Tensor> multihead_attention_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor attn_4d, torch::Tensor weighted_4d,
    torch::Tensor head_out, torch::Tensor hw, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor K_emb, torch::Tensor V_emb,
    torch::Tensor out_proj, torch::Tensor head_weights, torch::Tensor gate_logit);

// --- Cross Attention ---
std::vector<torch::Tensor> cross_attention_forward(
    torch::Tensor x, torch::Tensor Q_emb,
    torch::Tensor mem_keys, torch::Tensor mem_values, torch::Tensor gate_logit);
std::vector<torch::Tensor> cross_attention_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor Q, torch::Tensor attn, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor mem_keys, torch::Tensor mem_values,
    torch::Tensor gate_logit);

// --- Causal Attention ---
std::vector<torch::Tensor> causal_attention_forward(
    torch::Tensor x, torch::Tensor Q_emb, torch::Tensor K_emb,
    torch::Tensor V_emb, torch::Tensor out_proj, torch::Tensor gate_logit);
std::vector<torch::Tensor> causal_attention_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor scores, torch::Tensor attn,
    torch::Tensor weighted, torch::Tensor out,
    torch::Tensor Q_emb, torch::Tensor K_emb, torch::Tensor V_emb,
    torch::Tensor out_proj, torch::Tensor gate_logit);

// --- Skip Connection ---
torch::Tensor skip_connection_forward(torch::Tensor x, float scale);
std::vector<torch::Tensor> skip_connection_backward(
    torch::Tensor grad_output, torch::Tensor x, float scale);

// --- Dilated Conv1d ---
torch::Tensor dilated_conv1d_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gate_logit, int kernel_size, int dilation);
std::vector<torch::Tensor> dilated_conv1d_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gate_logit, int kernel_size, int dilation);

// --- EMA Smooth ---
torch::Tensor ema_smooth_forward(
    torch::Tensor x, torch::Tensor alpha_logit, torch::Tensor gate_logit);
std::vector<torch::Tensor> ema_smooth_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor alpha_logit, torch::Tensor gate_logit);

// --- Gated Linear Unit ---
torch::Tensor gated_linear_unit_forward(
    torch::Tensor x, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor value_weight, torch::Tensor value_bias,
    torch::Tensor outer_gate_logit);
std::vector<torch::Tensor> gated_linear_unit_backward(
    torch::Tensor grad_output, torch::Tensor x,
    torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor value_weight, torch::Tensor value_bias,
    torch::Tensor outer_gate_logit,
    torch::Tensor gate_sigmoid, torch::Tensor gate_linear,
    torch::Tensor value_linear);

// --- Temporal Diff ---
torch::Tensor temporal_diff_forward(
    torch::Tensor x, torch::Tensor gate_logit);
std::vector<torch::Tensor> temporal_diff_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor gate_logit);

// --- NorMuon (per-neuron gradient normalization) ---
void normuon_normalize(torch::Tensor grad, float eps);

// --- Fused Optimizer Step ---
void fused_optimizer_step(
    torch::Tensor p_data, torch::Tensor grad,
    torch::Tensor m_fast, torch::Tensor m_medium, torch::Tensor m_slow,
    torch::Tensor v, torch::Tensor buf,
    float beta1_fast, float beta1_med, float beta1_slow, float beta2,
    float inv_bc_fast, float inv_bc_med, float inv_bc_slow, float inv_sqrt_bc_v,
    float lr, float weight_decay, float newborn_mult, float eps,
    bool write_buf_only);

void optimizer_apply_update(
    torch::Tensor p_data, torch::Tensor buf, torch::Tensor v,
    float inv_sqrt_bc_v, float lr, float weight_decay,
    float newborn_mult, float eps);

// --- Elastic Deformation (separable Gaussian blur) ---
void elastic_set_kernel_weights(torch::Tensor weights_1d);
torch::Tensor elastic_blur_separable(torch::Tensor displacement, int kernel_size);

// --- Capsule Squash ---
std::vector<torch::Tensor> capsule_squash_forward(
    torch::Tensor input, int cap_dim, float eps);
std::vector<torch::Tensor> capsule_squash_backward(
    torch::Tensor grad_output, torch::Tensor input, int cap_dim, float eps);

// =============================================================================
// Python Module Registration
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ASANN Custom CUDA Operations";

    // Gated Residual
    m.def("gated_residual_forward", &gated_residual_forward,
          "Gated residual forward (CUDA)");
    m.def("gated_residual_backward", &gated_residual_backward,
          "Gated residual backward (CUDA)");

    // Positional Embedding
    m.def("positional_embedding_forward", &positional_embedding_forward,
          "Positional embedding forward (CUDA)");
    m.def("positional_embedding_backward", &positional_embedding_backward,
          "Positional embedding backward (CUDA)");

    // Factored Embedding
    m.def("factored_embedding_forward", &factored_embedding_forward,
          "Factored embedding forward (CUDA)");
    m.def("factored_embedding_backward", &factored_embedding_backward,
          "Factored embedding backward (CUDA)");

    // MLP Embedding
    m.def("mlp_embedding_forward", &mlp_embedding_forward,
          "MLP embedding forward (CUDA)");
    m.def("mlp_embedding_backward", &mlp_embedding_backward,
          "MLP embedding backward (CUDA)");

    // Geometric Embedding
    m.def("geometric_embedding_forward", &geometric_embedding_forward,
          "Geometric embedding forward (CUDA)");
    m.def("geometric_embedding_backward", &geometric_embedding_backward,
          "Geometric embedding backward (CUDA)");

    // GAP Flatten
    m.def("gap_flatten_forward", &gap_flatten_forward,
          "Global average pooling + flatten forward (CUDA)");
    m.def("gap_flatten_backward", &gap_flatten_backward,
          "Global average pooling + flatten backward (CUDA)");

    // Conv1d Block
    m.def("conv1d_block_forward", &conv1d_block_forward,
          "Conv1d block forward (CUDA)");
    m.def("conv1d_block_backward", &conv1d_block_backward,
          "Conv1d block backward (CUDA)");

    // Conv2d Block
    m.def("conv2d_block_forward", &conv2d_block_forward,
          "Conv2d block forward (CUDA)");
    m.def("conv2d_block_backward", &conv2d_block_backward,
          "Conv2d block backward (CUDA)");

    // Spatial Conv2d
    m.def("spatial_conv2d_forward", &spatial_conv2d_forward,
          "Spatial Conv2d forward (CUDA)");
    m.def("spatial_conv2d_backward", &spatial_conv2d_backward,
          "Spatial Conv2d backward (CUDA)");

    // Pointwise Conv2d
    m.def("pointwise_conv2d_forward", &pointwise_conv2d_forward,
          "Pointwise Conv2d forward (CUDA)");
    m.def("pointwise_conv2d_backward", &pointwise_conv2d_backward,
          "Pointwise Conv2d backward (CUDA)");

    // Depthwise Separable Conv2d
    m.def("dw_separable_conv2d_forward", &dw_separable_conv2d_forward,
          "Depthwise separable Conv2d forward (CUDA)");
    m.def("dw_separable_conv2d_backward", &dw_separable_conv2d_backward,
          "Depthwise separable Conv2d backward (CUDA)");

    // Channel Attention
    m.def("channel_attention_forward", &channel_attention_forward,
          "Channel attention (SE-block) forward (CUDA)");
    m.def("channel_attention_backward", &channel_attention_backward,
          "Channel attention (SE-block) backward (CUDA)");

    // Self Attention
    m.def("self_attention_forward", &self_attention_forward,
          "Self attention forward (CUDA)");
    m.def("self_attention_backward", &self_attention_backward,
          "Self attention backward (CUDA)");

    // Multi-Head Attention
    m.def("multihead_attention_forward", &multihead_attention_forward,
          "Multi-head attention forward (CUDA)");
    m.def("multihead_attention_backward", &multihead_attention_backward,
          "Multi-head attention backward (CUDA)");

    // Cross Attention
    m.def("cross_attention_forward", &cross_attention_forward,
          "Cross attention forward (CUDA)");
    m.def("cross_attention_backward", &cross_attention_backward,
          "Cross attention backward (CUDA)");

    // Causal Attention
    m.def("causal_attention_forward", &causal_attention_forward,
          "Causal attention forward (CUDA)");
    m.def("causal_attention_backward", &causal_attention_backward,
          "Causal attention backward (CUDA)");

    // Skip Connection
    m.def("skip_connection_forward", &skip_connection_forward,
          "Skip connection forward (CUDA)");
    m.def("skip_connection_backward", &skip_connection_backward,
          "Skip connection backward (CUDA)");

    // Dilated Conv1d
    m.def("dilated_conv1d_forward", &dilated_conv1d_forward,
          "Dilated Conv1d forward (CUDA)");
    m.def("dilated_conv1d_backward", &dilated_conv1d_backward,
          "Dilated Conv1d backward (CUDA)");

    // EMA Smooth
    m.def("ema_smooth_forward", &ema_smooth_forward,
          "EMA smooth forward (CUDA)");
    m.def("ema_smooth_backward", &ema_smooth_backward,
          "EMA smooth backward (CUDA)");

    // Gated Linear Unit
    m.def("gated_linear_unit_forward", &gated_linear_unit_forward,
          "Gated linear unit forward (CUDA)");
    m.def("gated_linear_unit_backward", &gated_linear_unit_backward,
          "Gated linear unit backward (CUDA)");

    // Temporal Diff
    m.def("temporal_diff_forward", &temporal_diff_forward,
          "Temporal diff forward (CUDA)");
    m.def("temporal_diff_backward", &temporal_diff_backward,
          "Temporal diff backward (CUDA)");

    // NorMuon (per-neuron gradient normalization)
    m.def("normuon_normalize", &normuon_normalize,
          "NorMuon per-neuron gradient normalization (CUDA)");

    // Fused Optimizer Step
    m.def("fused_optimizer_step", &fused_optimizer_step,
          "Fused optimizer step with multi-scale momentum (CUDA)");
    m.def("optimizer_apply_update", &optimizer_apply_update,
          "Apply update after Newton-Schulz orthogonalization (CUDA)");

    // Elastic Deformation (separable Gaussian blur)
    m.def("elastic_set_kernel_weights", &elastic_set_kernel_weights,
          "Upload 1D Gaussian kernel weights to constant memory (CUDA)");
    m.def("elastic_blur_separable", &elastic_blur_separable,
          "Separable Gaussian blur on displacement field (CUDA)");

    // Capsule Squash
    m.def("capsule_squash_forward", &capsule_squash_forward,
          "Capsule squash activation forward (CUDA)");
    m.def("capsule_squash_backward", &capsule_squash_backward,
          "Capsule squash activation backward (CUDA)");
}
