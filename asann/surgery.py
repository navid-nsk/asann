
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Optional, Tuple, Any
from .config import ASANNConfig
from .model import (ASANNModel, OperationPipeline, SpatialOperationPipeline,
                     SkipConnection, ConvStem, ASANNLayer, _TRIVIAL_OPS)
from .logger import SurgeryLogger

# Try to import CUDA operation replacements
try:
    from asann_cuda.ops import (
        Conv1dBlockCUDA, Conv2dBlockCUDA, FactoredEmbeddingCUDA,
        MLPEmbeddingCUDA, GeometricEmbeddingCUDA, PositionalEmbeddingCUDA,
        SelfAttentionOpCUDA, MultiHeadAttentionOpCUDA,
        CrossAttentionOpCUDA, CausalAttentionOpCUDA,
        SpatialConv2dOpCUDA, PointwiseConv2dOpCUDA,
        DepthwiseSeparableConv2dOpCUDA, ChannelAttentionOpCUDA,
        CapsuleConv2dOpCUDA,
        DilatedConv1dBlockCUDA, EMASmoothCUDA,
        GatedLinearUnitCUDA, TemporalDiffCUDA,
    )
    _CUDA_OPS_AVAILABLE = True
except ImportError:
    _CUDA_OPS_AVAILABLE = False


# ==================== Utilities ====================

def _safe_num_groups(C: int, max_groups: int = 4) -> int:
    """Find largest divisor of C that is <= max_groups.

    Used for nn.GroupNorm which requires num_channels % num_groups == 0.
    Falls back to 1 (equivalent to LayerNorm over channels) for primes.
    """
    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0:
            return g
    return 1


# ==================== Conv1D Wrapper ====================

class Conv1dBlock(nn.Module):
    """A Conv1d operation that works on flat [batch, d] tensors.

    Reshapes [batch, d] -> [batch, 1, d], applies Conv1d with same-padding,
    then reshapes back to [batch, d]. Uses C_in=1, C_out=1 so the output
    dimension equals the input dimension, fitting naturally into the pipeline.

    Different kernel sizes capture different locality patterns:
      - kernel_size=3: very local patterns
      - kernel_size=5: medium-range patterns
      - kernel_size=7: wider-range patterns
    """

    def __init__(self, d: int, kernel_size: int = 3):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        # same-padding: output length = input length
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size, padding=self.padding, bias=True,
        )
        # Initialize near-identity: center weight = 1.0, rest = small noise
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        self.conv.weight.data[0, 0, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d]
        x_3d = x.unsqueeze(1)  # [batch, 1, d]
        out = self.conv(x_3d)  # [batch, 1, d]
        return out.squeeze(1)  # [batch, d]

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}"


class Conv2dBlock(nn.Module):
    """Conv2d on flat [batch, d] tensors with spatial awareness.

    Reshapes [batch, d] → [batch, C, H, W], applies depthwise Conv2d
    (groups=C, same-padding), reshapes back. Gated residual.

    For layer 0 with d == C*H*W: uses actual spatial dims.
    For hidden layers: treats d as (1, H', W') with H'*W'=d.
    """

    def __init__(self, d: int, kernel_size: int = 3, spatial_shape=None):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size

        if spatial_shape and d == spatial_shape[0] * spatial_shape[1] * spatial_shape[2]:
            self.C, self.H, self.W = spatial_shape
        else:
            # For hidden layers: treat as (1, H', W') with best factorization
            self.C = 1
            side = int(d ** 0.5)
            while side > 1 and d % side != 0:
                side -= 1
            self.H = side
            self.W = d // side

        padding = kernel_size // 2
        self.conv = nn.Conv2d(self.C, self.C, kernel_size, padding=padding,
                              groups=self.C, bias=True)
        # Near-identity init
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        for c in range(self.C):
            self.conv.weight.data[c, 0, center, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        B = x.shape[0]
        img = x.view(B, self.C, self.H, self.W)
        out = self.conv(img).view(B, -1)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}, spatial=({self.C},{self.H},{self.W})"


# ==================== Embedding Operations ====================
# Learned feature transformations that capture different structural patterns.
# All embeddings: [batch, d] → [batch, d], residual connections, near-identity init.

class FactoredEmbedding(nn.Module):
    """Low-rank matrix factorization embedding.

    Learns a low-rank residual correction: out = x + x @ V.T @ U.T
    Captures linear feature correlations via factorized transformation.
    Rank = max(d//4, 2) balances expressiveness with parameter efficiency.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rank = max(d // 4, 2)
        # U: [d, rank], V: [rank, d] — scale 0.1/sqrt(rank) gives ~0.02 relative
        # perturbation, matching Conv1dBlock signal level for probing discovery
        init_scale = 0.1 / (self.rank ** 0.5)
        self.U = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.V = nn.Parameter(torch.randn(self.rank, d) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, d] → correction: x @ V.T @ U.T → [batch, d]
        correction = x @ self.V.t() @ self.U.t()
        return x + correction

    def extra_repr(self) -> str:
        return f"d={self.d}, rank={self.rank}"


class MLPEmbedding(nn.Module):
    """Bottleneck MLP embedding with residual connection.

    Learns non-linear feature interactions: out = x + decode(relu(encode(x)))
    Hidden dimension = max(d//2, 4) creates an information bottleneck that
    forces the MLP to learn compressed feature representations.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.hidden = max(d // 2, 4)
        self.encode = nn.Linear(d, self.hidden)
        self.decode = nn.Linear(self.hidden, d)
        # Scale 0.1/sqrt(hidden) gives ~0.01 relative perturbation for probing discovery
        init_scale = 0.1 / (self.hidden ** 0.5)
        nn.init.normal_(self.encode.weight, std=init_scale)
        nn.init.zeros_(self.encode.bias)
        nn.init.normal_(self.decode.weight, std=init_scale)
        nn.init.zeros_(self.decode.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.decode(F.relu(self.encode(x)))
        return x + correction

    def extra_repr(self) -> str:
        return f"d={self.d}, hidden={self.hidden}"


class GeometricEmbedding(nn.Module):
    """Hyperbolic/Poincaré-inspired geometric embedding.

    Projects features through a Poincaré-like transform that maps to a unit ball,
    capturing hierarchical and distance-based feature structure.
    out = (1 - gate) * x + gate * poincare_transform(x)

    Poincaré transform: (radial_scale * x + bias) * tanh(norm) / (norm + eps)
    This normalizes feature vectors to unit ball while preserving direction.

    Init: gate ≈ 0.018 (sigmoid(-4.0)), radial_scale=1, bias=0 → near-identity
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.radial_scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        # Gate: sigmoid(-4.0) ≈ 0.018 → ~0.016 relative perturbation for probing
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # Poincaré-like transform
        scaled = self.radial_scale * x + self.bias
        norm = scaled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        poincare = scaled * (torch.tanh(norm) / norm)
        return (1 - gate) * x + gate * poincare

    def extra_repr(self) -> str:
        return f"d={self.d}"


class PositionalEmbedding(nn.Module):
    """Learned positional offset for feature ordering.

    Adds a learned per-feature bias: out = x + pos_emb
    Captures the importance of feature position/ordering in the input.
    Simplest embedding — small noise at init for ~0.01 probing discovery signal.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # Small noise gives ~0.01 relative perturbation for probing discovery
        self.pos_emb = nn.Parameter(torch.randn(d) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb

    def extra_repr(self) -> str:
        return f"d={self.d}"


# ==================== Attention Operations ====================
# Data-dependent attention for flat [batch, d] tabular data.
# Each feature's scalar value modulates learned Q/K/V embeddings, making
# attention weights input-dependent (unlike the static AttentionEmbedding).

class SelfAttentionOp(nn.Module):
    """Data-dependent feature self-attention for tabular data.

    Each feature i has learned embeddings q_i, k_i, v_i in R^r.
    Queries/keys/values are modulated by the feature's actual value:
        Q_i(x) = x_i * q_i,  K_j(x) = x_j * k_j,  V_j(x) = x_j * v_j
        attn(i,j) = softmax_j( Q_i · K_j / sqrt(r) )
        out_i = sum_j attn(i,j) * V_j · out_proj

    Attention patterns change based on actual input values — different samples
    get different feature mixing. This is the defining property of real attention.

    r = max(d//4, 4). Init: embeddings at 0.1/sqrt(r), gate=-4.0 → ~0.02 perturbation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rank = max(d // 4, 4)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.K_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.V_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.out_proj = nn.Parameter(torch.randn(self.rank) * init_scale)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # Q, K, V: [B, d, rank] — each feature's value modulates its embedding
        Q = x.unsqueeze(-1) * self.Q_emb.unsqueeze(0)
        K = x.unsqueeze(-1) * self.K_emb.unsqueeze(0)
        V = x.unsqueeze(-1) * self.V_emb.unsqueeze(0)
        # Attention: [B, d, d]
        scale = self.rank ** 0.5
        attn = torch.softmax(torch.bmm(Q, K.transpose(-2, -1)) / scale, dim=-1)
        # Weighted values -> project to scalar per feature: [B, d]
        weighted = torch.bmm(attn, V)  # [B, d, rank]
        out = (weighted * self.out_proj).sum(dim=-1)  # [B, d]
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, rank={self.rank}"


class MultiHeadAttentionOp(nn.Module):
    """Multi-head data-dependent feature attention.

    H independent attention heads, each with rank r_h, capturing different
    types of feature interactions simultaneously. All heads computed in
    parallel via vectorized einsum operations (no Python loops).

    Each head h has its own Q_h, K_h, V_h embeddings: [d, r_h].
    Heads combined via learned softmax(head_weights) mixing.

    H = max(d//8, 2), r_h = max(d//(H*2), 2).
    Init: embeddings at 0.1/sqrt(r_h), gate=-4.0 → ~0.02 perturbation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.num_heads = max(d // 8, 2)
        self.head_rank = max(d // (self.num_heads * 2), 2)
        init_scale = 0.1 / (self.head_rank ** 0.5)
        # Per-head Q/K/V embeddings: [H, d, r_h]
        self.Q_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        self.K_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        self.V_emb = nn.Parameter(torch.randn(self.num_heads, d, self.head_rank) * init_scale)
        # Per-head output projection: [H, r_h]
        self.out_proj = nn.Parameter(torch.randn(self.num_heads, self.head_rank) * init_scale)
        # Head mixing weights: [H], initialized uniform
        self.head_weights = nn.Parameter(torch.zeros(self.num_heads))
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # x: [B, d] -> [B, 1, d, 1] for broadcasting with [H, d, r_h]
        x_exp = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, d, 1]
        # Q, K, V per head: [B, H, d, r_h]
        Q = x_exp * self.Q_emb.unsqueeze(0)
        K = x_exp * self.K_emb.unsqueeze(0)
        V = x_exp * self.V_emb.unsqueeze(0)
        # Attention per head: [B, H, d, d]
        scale = self.head_rank ** 0.5
        attn = torch.softmax(
            torch.einsum('bhir,bhjr->bhij', Q, K) / scale, dim=-1
        )
        # Weighted values per head: [B, H, d, r_h]
        weighted = torch.einsum('bhij,bhjr->bhir', attn, V)
        # Project each head to scalar per feature: [B, H, d]
        head_out = (weighted * self.out_proj.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        # Mix heads: softmax(head_weights) -> [H], weighted sum -> [B, d]
        hw = torch.softmax(self.head_weights, dim=0)
        out = (head_out * hw.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, heads={self.num_heads}, head_rank={self.head_rank}"


class CrossAttentionOp(nn.Module):
    """Cross-attention between input features and a learned memory bank.

    Features attend not to each other but to M learned memory vectors.
    Queries come from input (data-dependent), keys/values come from memory
    (learned prototypes). Each memory slot represents a learned concept.

        Q_i(x) = x_i * q_i           (input-derived, [B, d, r])
        K_m = mem_keys[m]             (learned, [M, r])
        V_m = mem_values[m]           (learned, [M, d])
        attn(i, m) = softmax_m(Q_i · K_m / sqrt(r))
        out_i = sum_m attn(i,m) * V_m[i]

    This is genuinely cross-modal: queries from input, keys/values from memory.
    No d² cost — only d×M attention matrix.

    M = max(d//4, 4), r = max(d//4, 4).
    Init: embeddings at 0.1/sqrt(r), gate=-4.0 → ~0.02 perturbation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.num_memories = max(d // 4, 4)
        self.rank = max(d // 4, 4)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.mem_keys = nn.Parameter(torch.randn(self.num_memories, self.rank) * init_scale)
        self.mem_values = nn.Parameter(torch.randn(self.num_memories, d) * init_scale)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # Queries: [B, d, rank]
        Q = x.unsqueeze(-1) * self.Q_emb.unsqueeze(0)
        # Attention to memory: [B, d, M]
        scale = self.rank ** 0.5
        attn = torch.softmax(
            torch.einsum('bdr,mr->bdm', Q, self.mem_keys) / scale, dim=-1
        )
        # Retrieve from memory: out_i = sum_m attn(i,m) * mem_values(m,i)
        out = torch.einsum('bdm,md->bd', attn, self.mem_values)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, memories={self.num_memories}, rank={self.rank}"


class CausalAttentionOp(nn.Module):
    """Causal (autoregressive) data-dependent feature attention.

    Same Q/K/V mechanism as SelfAttentionOp, but with a causal mask:
    feature i can only attend to features j <= i (lower-indexed features).
    This creates directed, ordered information flow.

    Meaningful for tabular data because features are often ordered by
    importance, domain grouping, or engineering order. The model learns
    directed dependencies: feature A informs feature B, but not vice versa.

    r = max(d//4, 4). Init: embeddings at 0.1/sqrt(r), gate=-4.0.
    The causal_mask is a buffer (not a parameter), auto-recreated on resize.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.rank = max(d // 4, 4)
        init_scale = 0.1 / (self.rank ** 0.5)
        self.Q_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.K_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.V_emb = nn.Parameter(torch.randn(d, self.rank) * init_scale)
        self.out_proj = nn.Parameter(torch.randn(self.rank) * init_scale)
        # Causal mask: lower-triangular, True where j <= i
        self.register_buffer('causal_mask', torch.tril(torch.ones(d, d, dtype=torch.bool)))
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        Q = x.unsqueeze(-1) * self.Q_emb.unsqueeze(0)  # [B, d, rank]
        K = x.unsqueeze(-1) * self.K_emb.unsqueeze(0)
        V = x.unsqueeze(-1) * self.V_emb.unsqueeze(0)
        # Attention scores with causal mask
        scale = self.rank ** 0.5
        scores = torch.bmm(Q, K.transpose(-2, -1)) / scale  # [B, d, d]
        scores = scores.masked_fill(~self.causal_mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        # Weighted values -> project to scalar: [B, d]
        weighted = torch.bmm(attn, V)
        out = (weighted * self.out_proj).sum(dim=-1)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, rank={self.rank}, causal=True"


# ==================== Temporal / Sequence Operations ====================
# Operations designed for sequence/time-series data on flat [B, d] tensors.
# These are added to the flat candidate pool so surgery can discover them.


class DilatedConv1dBlock(nn.Module):
    """Dilated Conv1d on flat [batch, d] tensors for TCN/WaveNet-style processing.

    Like Conv1dBlock but with dilation, giving exponentially larger receptive
    field without extra parameters. Reshapes [B,d] → [B,1,d], applies dilated
    Conv1d with same-padding, reshapes back. Gated residual for near-identity.

    Different kernel sizes + dilation capture multi-scale temporal patterns:
      - kernel_size=3, dilation=2: effective receptive field = 5
      - kernel_size=5, dilation=2: effective receptive field = 9
    """

    def __init__(self, d: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        self.d = d
        self.kernel_size = kernel_size
        self.dilation = dilation
        # same-padding for dilated conv: padding = dilation * (kernel_size // 2)
        self.padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=kernel_size, padding=self.padding,
            dilation=dilation, bias=True,
        )
        # Near-identity init: center weight = 1.0, rest = small noise
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        self.conv.weight.data[0, 0, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        x_3d = x.unsqueeze(1)      # [B, 1, d]
        out = self.conv(x_3d)       # [B, 1, d]
        out = out.squeeze(1)        # [B, d]
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, kernel_size={self.kernel_size}, dilation={self.dilation}"


class EMASmooth(nn.Module):
    """Exponential Moving Average smoothing on flat [B, d] features.

    Treats the d features as a temporal sequence and applies EMA:
        y_0 = x_0
        y_i = alpha_i * x_i + (1 - alpha_i) * y_{i-1}

    Learnable per-channel alpha in (0, 1) via sigmoid(alpha_logit).
    Captures temporal smoothing / low-pass filtering of feature sequences.
    Gated residual: out = (1 - gate) * x + gate * ema(x).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # sigmoid(0.0) = 0.5 → alpha starts at 0.5 (moderate smoothing)
        self.alpha_logit = nn.Parameter(torch.zeros(d))
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        alpha = torch.sigmoid(self.alpha_logit)  # [d]

        # EMA scan over feature dimension (no in-place ops for autograd)
        B, d = x.shape
        cols = [x[:, 0]]  # y_0 = x_0
        for i in range(1, d):
            prev = cols[i - 1]
            cols.append(alpha[i] * x[:, i] + (1.0 - alpha[i]) * prev)
        y = torch.stack(cols, dim=1)  # [B, d]

        return (1 - gate) * x + gate * y

    def extra_repr(self) -> str:
        return f"d={self.d}"


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit: GLU(x) = sigmoid(Wx + b) * (Vx + c).

    Two parallel linear projections — one through sigmoid gate, one linear.
    Elementwise product creates a learned gating mechanism.
    Used in TCN, gated convolution, and modern sequence models (PaLM, LLaMA).

    Gated residual: out = (1 - outer_gate) * x + outer_gate * glu(x).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.gate_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d)
        # Small init so GLU output ≈ 0 initially → near-identity with residual
        init_scale = 0.1 / (d ** 0.5)
        nn.init.normal_(self.gate_proj.weight, std=init_scale)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.value_proj.weight, std=init_scale)
        nn.init.zeros_(self.value_proj.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer_gate = torch.sigmoid(self.gate_logit)
        gate = torch.sigmoid(self.gate_proj(x))
        value = self.value_proj(x)
        glu_out = gate * value
        return (1 - outer_gate) * x + outer_gate * glu_out

    def extra_repr(self) -> str:
        return f"d={self.d}"


class TemporalDiff(nn.Module):
    """First-order difference: y_i = x_i - x_{i-1}, y_0 = 0.

    Captures rate of change across the feature dimension.
    No learnable parameters except the gated residual gate.
    Useful for time-series where features represent temporal steps.

    Gated residual: out = (1 - gate) * x + gate * diff(x).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        diff = x[:, 1:] - x[:, :-1]           # [B, d-1]
        diff = F.pad(diff, (1, 0), value=0.0)  # [B, d] — zero-padded first element
        return (1 - gate) * x + gate * diff

    def extra_repr(self) -> str:
        return f"d={self.d}"


class ActivationNoise(nn.Module):
    """Additive Gaussian noise on activations during training.

    Unlike dropout (multiplicative binary mask), this adds continuous
    Gaussian noise that disrupts memorization patterns without completely
    zeroing out neurons. Parameter-free — no learnable weights.
    """
    def __init__(self, d: int, config=None):
        super().__init__()
        self._d = d
        self._config = config

    def forward(self, x, **kwargs):
        if self.training:
            std = getattr(self._config, 'activation_noise_std', 0.0) if self._config else 0.0
            if std > 0:
                noise = torch.randn_like(x) * std
                return x + noise
        return x

    def extra_repr(self) -> str:
        std = getattr(self._config, 'activation_noise_std', 0.0) if self._config else 0.0
        return f"d={self._d}, std={std}"


class GRUOp(nn.Module):
    """GRU-based sequential processing of feature groups.

    Splits features into chunks and processes them sequentially with a GRU,
    capturing dependencies across feature groups. Useful for:
    - Traffic: temporal patterns in projected time-step features
    - Tabular: sequential feature dependencies (lag structures, trend detection)
    - Any domain where feature ordering carries information

    Architecture: [B, d] → [B, num_chunks, chunk_size] → GRU → reshape → [B, d]
    Gated residual: out = (1-gate) * x + gate * gru(x)
    """

    def __init__(self, d: int, num_chunks: int = 8):
        super().__init__()
        self.d = d
        self.num_chunks = min(num_chunks, d)
        # Ensure d is divisible by num_chunks
        while d % self.num_chunks != 0 and self.num_chunks > 1:
            self.num_chunks -= 1
        self.chunk_size = d // self.num_chunks

        self.gru = nn.GRU(
            input_size=self.chunk_size,
            hidden_size=self.chunk_size,
            batch_first=True,
            num_layers=1,
        )
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

        # Near-identity init: small weights so initial output ≈ input
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # x: [B, d] → [B, num_chunks, chunk_size]
        x_seq = x.view(x.shape[0], self.num_chunks, self.chunk_size)
        out_seq, _ = self.gru(x_seq)  # [B, num_chunks, chunk_size]
        out = out_seq.reshape(x.shape[0], self.d)  # [B, d]
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, num_chunks={self.num_chunks}, chunk_size={self.chunk_size}"


class DerivativeConv1d(nn.Module):
    """Learnable derivative operator via Conv1d with finite-difference initialisation.

    Treats flat [B, d] features as a 1D signal and applies a convolution
    initialized to a discrete derivative stencil.  Supports order 1 and 2:
        Order 1 (central difference): [-0.5, 0, 0.5]
        Order 2 (second difference): [1, -2, 1]

    Gated residual with gate_logit = -2.0 → sigmoid ≈ 0.12 (~12% mixing).
    This is deliberately low ("aspirin dose"): the model learns whether
    derivative features are useful via gradient descent on the gate.

    Reshapes [B, d] → [B, 1, d], applies Conv1d(1, 1, 3, padding=1),
    reshapes back to [B, d].  The kernel has only 3 learnable elements
    so it's dimension-independent (survives add_neuron / remove_neuron).
    """

    def __init__(self, d: int, order: int = 1):
        super().__init__()
        self.d = d
        self.order = order
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=3, padding=1, bias=True,
        )
        # Finite-difference initialisation
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        if order == 1:
            # Central difference: [-0.5, 0, 0.5]
            self.conv.weight.data[0, 0, 0] = -0.5
            self.conv.weight.data[0, 0, 1] = 0.0
            self.conv.weight.data[0, 0, 2] = 0.5
        elif order == 2:
            # Second difference: [1, -2, 1]
            self.conv.weight.data[0, 0, 0] = 1.0
            self.conv.weight.data[0, 0, 1] = -2.0
            self.conv.weight.data[0, 0, 2] = 1.0
        else:
            raise ValueError(f"DerivativeConv1d only supports order 1 or 2, got {order}")

        # Aspirin dose: gate_logit=-2.0 → sigmoid≈0.12 (~12% mixing)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        x_3d = x.unsqueeze(1)      # [B, 1, d]
        out = self.conv(x_3d)       # [B, 1, d]
        out = out.squeeze(1)        # [B, d]
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, order={self.order}"


class DerivativeConv2d(nn.Module):
    """Learnable 2D derivative operator via depthwise Conv2d with finite-difference init.

    Operates on native [B, C, H, W] tensors. Each channel is processed independently
    (groups=C) with a 3×3 kernel initialized to a discrete spatial derivative stencil.

    Supported modes:
        dx  (order=1, axis='x'):  ∂/∂x   via [[ 0, 0, 0],[-0.5, 0, 0.5],[ 0, 0, 0]]
        dy  (order=1, axis='y'):  ∂/∂y   via [[ 0,-0.5, 0],[ 0, 0, 0],[ 0, 0.5, 0]]
        dxx (order=2, axis='x'):  ∂²/∂x² via [[ 0, 0, 0],[ 1,-2, 1],[ 0, 0, 0]]
        dyy (order=2, axis='y'):  ∂²/∂y² via [[ 0, 1, 0],[ 0,-2, 0],[ 0, 1, 0]]
        laplacian (order=2, axis='xy'): ∇²  via [[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]]

    Gated residual with gate_logit=-2.0 (sigmoid ≈ 0.12, aspirin dose).
    Kernel has 9 elements per channel group — dimension-independent.
    Survives add_channel/remove_channel by adjusting groups.
    """

    def __init__(self, C: int, H: int, W: int, order: int = 1, axis: str = 'x'):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.order = order
        self.axis = axis
        self.kernel_size = 3

        self.conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Build stencil kernel
        kernel = torch.zeros(3, 3)
        if order == 1 and axis == 'x':
            # Central difference ∂/∂x: [-0.5, 0, 0.5] along columns
            kernel[1, 0] = -0.5
            kernel[1, 2] = 0.5
        elif order == 1 and axis == 'y':
            # Central difference ∂/∂y: [-0.5, 0, 0.5] along rows
            kernel[0, 1] = -0.5
            kernel[2, 1] = 0.5
        elif order == 2 and axis == 'x':
            # Second difference ∂²/∂x²: [1, -2, 1] along columns
            kernel[1, 0] = 1.0
            kernel[1, 1] = -2.0
            kernel[1, 2] = 1.0
        elif order == 2 and axis == 'y':
            # Second difference ∂²/∂y²: [1, -2, 1] along rows
            kernel[0, 1] = 1.0
            kernel[1, 1] = -2.0
            kernel[2, 1] = 1.0
        elif order == 2 and axis == 'xy':
            # Laplacian ∇² = ∂²/∂x² + ∂²/∂y²
            kernel[0, 1] = 1.0
            kernel[1, 0] = 1.0
            kernel[1, 1] = -4.0
            kernel[1, 2] = 1.0
            kernel[2, 1] = 1.0
        else:
            raise ValueError(f"DerivativeConv2d: unsupported order={order}, axis={axis}")

        # Apply stencil to all channels
        for c in range(C):
            self.conv.weight.data[c, 0] = kernel.clone()

        # Aspirin dose: gate_logit=-2.0 → sigmoid≈0.12 (~12% mixing)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        out = self.conv(x)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, order={self.order}, axis={self.axis}"


class PolynomialOp(nn.Module):
    """Learnable polynomial interaction operation for flat [B, d] tensors.

    Computes element-wise polynomial features (x, x², optionally x³) and
    combines them via a low-rank projection back to [B, d].

    Internal computation:
        features = [x, x²]  (or [x, x², x³] for degree=3)
        expanded = cat(features, dim=-1)  → [B, degree*d]
        combined = Linear(degree*d, rank) → ReLU → Linear(rank, d) → [B, d]
        result = (1 - gate) * x + gate * combined

    The rank (= max(d//4, 4)) keeps parameter count bounded.
    Near-identity init: small random weights + gate_logit=-2.0.
    """

    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        self.d = d
        self.degree = degree
        self.rank = max(d // 4, 4)

        # Projection: [degree * d] → rank → d
        self.project_down = nn.Linear(degree * d, self.rank)
        self.project_up = nn.Linear(self.rank, d)

        # Small init for near-identity behaviour
        init_scale = 0.1 / (self.rank ** 0.5)
        nn.init.normal_(self.project_down.weight, std=init_scale)
        nn.init.zeros_(self.project_down.bias)
        nn.init.normal_(self.project_up.weight, std=init_scale)
        nn.init.zeros_(self.project_up.bias)

        # Aspirin dose gate
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        # Build polynomial features
        features = [x]
        x_power = x
        for _ in range(self.degree - 1):
            x_power = x_power * x  # x², x³, ...
            features.append(x_power)
        expanded = torch.cat(features, dim=-1)  # [B, degree*d]
        combined = F.relu(self.project_down(expanded))  # [B, rank]
        output = self.project_up(combined)  # [B, d]
        return (1 - gate) * x + gate * output

    def extra_repr(self) -> str:
        return f"d={self.d}, degree={self.degree}, rank={self.rank}"


class KANLinearOp(nn.Module):
    """Kolmogorov-Arnold Network operation for flat [B, d] tensors.

    Uses Radial Basis Function (RBF/FastKAN) approach for fast basis evaluation.
    Two additive paths:
        base_path:   Linear(d, d) applied to SiLU(x)  -- global smooth function
        spline_path: Linear(d*G, d) applied to RBF basis of LayerNorm(x)  -- local fine-grained

    The RBF basis evaluates Gaussian kernels centered on a fixed grid:
        rbf_j(x_i) = exp( -( (x_i - center_j) / denom )^2 )
    producing a [B, d, num_grids] tensor that is flattened and linearly projected.

    Parameters: d*d (base) + d*num_grids*d (spline) + 2*d (layernorm)
    For d=64, num_grids=8: ~37K params (vs PolynomialOp deg2: ~5K).

    Gated residual with gate_logit=-2.0 (sigmoid ~= 0.12) ensures
    near-identity at init, gradually contributing as training proceeds.
    """

    def __init__(self, d: int, num_grids: int = 8,
                 grid_range: tuple = (-2.0, 2.0)):
        super().__init__()
        self.d = d
        self.num_grids = num_grids

        # RBF grid centers (not trainable -- fixed spacing)
        grid = torch.linspace(grid_range[0], grid_range[1], num_grids)
        self.register_buffer('grid', grid)
        self.denominator = (grid_range[1] - grid_range[0]) / max(num_grids - 1, 1)

        # LayerNorm normalises input into grid range for stable RBF evaluation
        self.layernorm = nn.LayerNorm(d)

        # Spline path: [B, d * num_grids] -> [B, d]
        self.spline_weight = nn.Linear(d * num_grids, d, bias=False)
        nn.init.trunc_normal_(self.spline_weight.weight, std=0.1)

        # Base path: [B, d] -> [B, d]
        self.base_weight = nn.Linear(d, d, bias=False)
        nn.init.kaiming_uniform_(self.base_weight.weight)

        # Gate for gated residual (near-identity init)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)

        # RBF basis evaluation
        normed = self.layernorm(x)                  # [B, d]
        # normed.unsqueeze(-1): [B, d, 1], self.grid: [G]
        # Broadcasting: [B, d, G]
        rbf = torch.exp(
            -((normed.unsqueeze(-1) - self.grid) / self.denominator) ** 2
        )

        spline_out = self.spline_weight(
            rbf.reshape(x.size(0), -1)              # [B, d*G]
        )                                            # [B, d]
        base_out = self.base_weight(F.silu(x))       # [B, d]

        kan_output = spline_out + base_out
        return (1 - gate) * x + gate * kan_output

    def extra_repr(self) -> str:
        return f"d={self.d}, num_grids={self.num_grids}"


class SpatialPolynomialOp(nn.Module):
    """Learnable polynomial interaction for spatial [B, C, H, W] tensors.

    Computes element-wise channel powers and combines via pointwise (1×1) conv.
    Discovers nonlinear terms like u², u³ in PDE solutions.

    Internal computation:
        features = [x, x²]  → stack along channel dim: [B, degree*C, H, W]
        mixed = Conv2d(degree*C, C, 1×1) → [B, C, H, W]
        result = (1 - gate) * x + gate * mixed
    """

    def __init__(self, C: int, H: int, W: int, degree: int = 2):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.degree = degree

        self.mix = nn.Conv2d(degree * C, C, kernel_size=1, bias=True)
        init_scale = 0.1 / (C ** 0.5)
        nn.init.normal_(self.mix.weight, std=init_scale)
        nn.init.zeros_(self.mix.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        features = [x]
        x_power = x
        for _ in range(self.degree - 1):
            x_power = x_power * x
            features.append(x_power)
        expanded = torch.cat(features, dim=1)  # [B, degree*C, H, W]
        mixed = self.mix(expanded)  # [B, C, H, W]
        return (1 - gate) * x + gate * mixed

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, degree={self.degree}"


class BranchedOperationBlock(nn.Module):
    """Parallel branches merged by learned softmax weights for flat [B, d] tensors.

    Encapsulates N sub-operations that run in parallel on the same input,
    with outputs combined via a learned weighted sum (softmax over N branches).

    From the pipeline's perspective, this is a single [B,d]→[B,d] operation.
    Internally: x → [branch_0(x), branch_1(x), ...] → softmax_weighted_sum → output.

    For reaction-diffusion PDEs: branch_0 = derivative (diffusion),
    branch_1 = polynomial (reaction), discovering du/dt = D·∇²u + f(u).

    Gated residual on the combined output ensures near-identity init.
    """

    def __init__(self, d: int, branch_ops: list):
        super().__init__()
        self.d = d
        self.branches = nn.ModuleList(branch_ops)
        self.num_branches = len(branch_ops)
        # Uniform init → equal contribution initially
        self.branch_logits = nn.Parameter(torch.zeros(self.num_branches))
        # Gated residual
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        weights = F.softmax(self.branch_logits, dim=0)  # [N]
        combined = torch.zeros_like(x)
        for i, branch in enumerate(self.branches):
            combined = combined + weights[i] * branch(x)
        return (1 - gate) * x + gate * combined

    def extra_repr(self) -> str:
        return f"d={self.d}, branches={self.num_branches}"


class SpatialBranchedOperationBlock(nn.Module):
    """Parallel branches for spatial [B, C, H, W] tensors.

    Same design as BranchedOperationBlock but for 4D tensors.
    Typical use: DerivativeConv2d (diffusion) + SpatialPolynomialOp (reaction).
    """

    def __init__(self, C: int, H: int, W: int, branch_ops: list):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.branches = nn.ModuleList(branch_ops)
        self.num_branches = len(branch_ops)
        self.branch_logits = nn.Parameter(torch.zeros(self.num_branches))
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        weights = F.softmax(self.branch_logits, dim=0)
        combined = torch.zeros_like(x)
        for i, branch in enumerate(self.branches):
            combined = combined + weights[i] * branch(x)
        return (1 - gate) * x + gate * combined

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, branches={self.num_branches}"


# ==================== Spatial Operation Classes ====================
# These operate on native [B, C, H, W] tensors for spatial layers.


class SpatialConv2dOp(nn.Module):
    """Depthwise Conv2d on native [B, C, H, W] tensors with gated residual.

    Unlike Conv2dBlock which reshapes flat tensors, this operates on real
    spatial data. Depthwise (groups=C) with same-padding.
    """

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                              groups=C, bias=True)
        # Near-identity init
        nn.init.zeros_(self.conv.weight)
        center = kernel_size // 2
        for c in range(C):
            self.conv.weight.data[c, 0, center, center] = 1.0
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        # gate_logit=-1.0 → sigmoid≈0.27: ~27% contribution, enough to learn useful features
        # Matches flat ops (Conv1dBlock, embeddings, attention) for consistent probing.
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        out = self.conv(x)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, kernel_size={self.kernel_size}"


class PointwiseConv2dOp(nn.Module):
    """Pointwise (1×1) Conv2d for cross-channel mixing on [B, C, H, W] tensors.

    This is the "missing half" of depthwise separable convolutions.
    SpatialConv2dOp processes each channel independently (depthwise, groups=C);
    PointwiseConv2dOp mixes information ACROSS channels at each spatial position.
    Together they form the MobileNet-style depthwise separable pattern.

    Near-identity init: weight ≈ I + small noise, gated residual.
    """

    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        # Full cross-channel conv: groups=1 (default), kernel=1×1
        self.conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        # Near-identity: start as identity mapping + small noise
        nn.init.eye_(self.conv.weight.data.view(C, C))
        self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
        nn.init.zeros_(self.conv.bias)

        # gate_logit=-1.0 → sigmoid≈0.27: ~27% contribution, enough to learn useful features
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        out = self.conv(x)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}"


class ChannelAttentionOp(nn.Module):
    """Squeeze-and-Excitation channel attention on [B, C, H, W] tensors.

    global_avg_pool -> Linear(C, r) -> ReLU -> Linear(r, C) -> sigmoid -> rescale
    With gated residual for near-identity initialization.

    Args:
        C: Number of channels.
        reduction_ratio: Divisor for bottleneck width (r = C // reduction_ratio).
            Default 4 matches the original SE-Net paper. Variants:
            - reduction_ratio=4:  standard SE block (registered as "channel_attention")
            - reduction_ratio=8:  lighter SE (registered as "channel_attention_r8")
            - reduction_ratio=16: lightest SE (registered as "channel_attention_r16")
    """

    def __init__(self, C: int, reduction_ratio: int = 4):
        super().__init__()
        self.C = C
        self.reduction_ratio = reduction_ratio
        self.reduction = max(C // reduction_ratio, 2)
        self.fc1 = nn.Linear(C, self.reduction)
        self.fc2 = nn.Linear(self.reduction, C)
        # Small init for near-identity
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

        # gate_logit=-1.0 → sigmoid≈0.27: ~27% contribution, enough to learn useful features
        # Matches all other parametric ops for consistent near-identity init.
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        gate = torch.sigmoid(self.gate_logit)
        # Squeeze: global average pool -> [B, C]
        s = x.mean(dim=(2, 3))
        # Excitation: MLP
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        # Rescale: [B, C, 1, 1]
        s = s.unsqueeze(-1).unsqueeze(-1)
        return (1 - gate) * x + gate * (x * s)

    def extra_repr(self) -> str:
        return f"C={self.C}, reduction={self.reduction}, ratio={self.reduction_ratio}"


class DepthwiseSeparableConv2dOp(nn.Module):
    """Fused Depthwise Separable Conv2d on [B, C, H, W] tensors.

    MobileNet building block: depthwise(groups=C) → BN2d → ReLU → pointwise(1×1).
    More parameter-efficient than discovering depthwise + pointwise separately,
    and the model learns both spatial and cross-channel patterns as a single unit.

    Parameter count for C channels, kernel k:
        depthwise: C·k² | BN: 2C | pointwise: C²+C | total ≈ C²+C·k²+3C
    Compare full Conv2d: C²·k² (much larger for k>1).

    Gated residual: out = (1 - gate) * x + gate * dwsep(x)
    Near-identity init: depthwise center=1.0, pointwise=I, gate sigmoid(-1.0) ≈ 0.27.
    """

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise: each channel independently
        self.dw_conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                                 groups=C, bias=False)
        self.bn = nn.BatchNorm2d(C)
        # Pointwise: cross-channel mixing (1×1 conv)
        self.pw_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)

        # Near-identity init for depthwise: center weight = 1.0
        nn.init.zeros_(self.dw_conv.weight)
        center = kernel_size // 2
        for c in range(C):
            self.dw_conv.weight.data[c, 0, center, center] = 1.0
        self.dw_conv.weight.data += 0.01 * torch.randn_like(self.dw_conv.weight.data)

        # Near-identity init for pointwise: I + noise
        nn.init.eye_(self.pw_conv.weight.data.view(C, C))
        self.pw_conv.weight.data += 0.01 * torch.randn_like(self.pw_conv.weight.data)
        nn.init.zeros_(self.pw_conv.bias)

        # Gated residual: sigmoid(-4.0) ≈ 0.018
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        out = self.dw_conv(x)
        out = F.relu(self.bn(out))
        out = self.pw_conv(out)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}, kernel_size={self.kernel_size}"


class CapsuleConv2dOp(nn.Module):
    """Capsule-aware Depthwise Separable Conv2d on [B, C, H, W] tensors.

    Groups channels into capsule vectors of dimension `cap_dim`, applies
    depthwise conv -> squash activation (instead of ReLU) -> pointwise conv.
    Squash forces meaningful vector representations: length encodes entity
    existence probability, direction encodes instantiation parameters.

    Squash: v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)

    Handles edge case where C is not divisible by cap_dim by temporarily
    padding channels to the nearest multiple, then slicing back.

    Gated residual: out = (1 - gate) * x + gate * capsule_conv(x)
    Near-identity init: depthwise center=1.0, pointwise=I, gate sigmoid(-1.0) ~ 0.27.
    """

    def __init__(self, C: int, H: int, W: int, kernel_size: int = 3, cap_dim: int = 4):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        self.cap_dim = min(cap_dim, C) if C >= 2 else 1
        if self.cap_dim < 2:
            self.cap_dim = 1

        self.C_padded = ((C + self.cap_dim - 1) // self.cap_dim) * self.cap_dim
        self.needs_padding = (self.C_padded != C)
        self.num_capsules = self.C_padded // self.cap_dim

        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(C, C, kernel_size, padding=padding,
                                 groups=C, bias=False)
        self.pw_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)

        # Near-identity init for depthwise: center weight = 1.0
        nn.init.zeros_(self.dw_conv.weight)
        center = kernel_size // 2
        for c in range(C):
            self.dw_conv.weight.data[c, 0, center, center] = 1.0
        self.dw_conv.weight.data += 0.01 * torch.randn_like(self.dw_conv.weight.data)

        # Near-identity init for pointwise: I + noise
        nn.init.eye_(self.pw_conv.weight.data.view(C, C))
        self.pw_conv.weight.data += 0.01 * torch.randn_like(self.pw_conv.weight.data)
        nn.init.zeros_(self.pw_conv.bias)

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))
        self._squash_eps = 1e-8

    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        """Squash activation on capsule vectors. x: [B, num_caps, cap_dim, H, W]."""
        sq_norm = (x * x).sum(dim=2, keepdim=True)
        norm = torch.sqrt(sq_norm + self._squash_eps)
        scale = sq_norm / (1.0 + sq_norm) / norm
        return x * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gate = torch.sigmoid(self.gate_logit)

        out = self.dw_conv(x)

        # Squash activation (replaces ReLU)
        if self.needs_padding:
            pad_c = self.C_padded - C
            out = F.pad(out, (0, 0, 0, 0, 0, pad_c))
        out = out.view(B, self.num_capsules, self.cap_dim, H, W)
        out = self._squash(out)
        out = out.view(B, self.C_padded, H, W)
        if self.needs_padding:
            out = out[:, :C, :, :]

        out = self.pw_conv(out)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return (f"C={self.C}, H={self.H}, W={self.W}, "
                f"kernel_size={self.kernel_size}, cap_dim={self.cap_dim}")


class MultiScaleConv2dOp(nn.Module):
    """Inception-style multi-scale depthwise separable conv on [B, C, H, W].

    Parallel branches: depthwise 1x1 + depthwise 3x3 + depthwise 5x5,
    concatenated along channel dim → 1x1 pointwise projection back to C.

    Near-identity init: 3x3 branch dominant (center=1.0), 1x1/5x5 attenuated
    (center=0.1). Pointwise passes through the 3x3 block (weight=1.0) and
    attenuates 1x1/5x5 (0.1). Gated residual: out = (1-gate)*x + gate*proj.
    """

    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

        # Depthwise branches (each preserves spatial dims)
        self.dw_1x1 = nn.Conv2d(C, C, 1, groups=C, bias=False)
        self.dw_3x3 = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.dw_5x5 = nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False)

        # Pointwise projection: 3C → C
        self.pw_proj = nn.Conv2d(3 * C, C, 1, bias=True)

        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

        # --- Near-identity initialization ---
        # 1x1 branch: attenuated identity
        nn.init.zeros_(self.dw_1x1.weight)
        for c in range(C):
            self.dw_1x1.weight.data[c, 0, 0, 0] = 0.1

        # 3x3 branch: dominant identity (center=1.0)
        nn.init.zeros_(self.dw_3x3.weight)
        for c in range(C):
            self.dw_3x3.weight.data[c, 0, 1, 1] = 1.0
        self.dw_3x3.weight.data += 0.01 * torch.randn_like(self.dw_3x3.weight.data)

        # 5x5 branch: attenuated identity
        nn.init.zeros_(self.dw_5x5.weight)
        for c in range(C):
            self.dw_5x5.weight.data[c, 0, 2, 2] = 0.1

        # Pointwise: pass-through 3x3 block (middle C channels), attenuate others
        # Layout: [0:C] = 1x1 branch, [C:2C] = 3x3 branch, [2C:3C] = 5x5 branch
        nn.init.zeros_(self.pw_proj.weight)
        nn.init.zeros_(self.pw_proj.bias)
        for c in range(C):
            self.pw_proj.weight.data[c, C + c, 0, 0] = 1.0      # 3x3 pass-through
            self.pw_proj.weight.data[c, c, 0, 0] = 0.1           # 1x1 attenuated
            self.pw_proj.weight.data[c, 2 * C + c, 0, 0] = 0.1   # 5x5 attenuated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)

        b1 = self.dw_1x1(x)
        b3 = self.dw_3x3(x)
        b5 = self.dw_5x5(x)

        # Concat along channel dim: [B, 3C, H, W]
        cat = torch.cat([b1, b3, b5], dim=1)

        # Project back to [B, C, H, W]
        out = self.pw_proj(cat)

        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"C={self.C}, H={self.H}, W={self.W}"


# ==================== Pooling Operations (Spatial) ====================
# Stride=1, same-padding pooling ops for spatial layers.
# Channel-independent: work with any C, no dimension resize needed.
# All use gated residual for near-identity probing.


class MaxPool2dOp(nn.Module):
    """MaxPool2d with stride=1 same-padding and gated residual.

    Nonlinear spatial max filter: selects maximum value in each kernel window.
    Useful for edge/feature detection and noise robustness.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        return (1 - gate) * x + gate * self.pool(x)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


class AvgPool2dOp(nn.Module):
    """AvgPool2d with stride=1 same-padding and gated residual.

    Linear spatial smoothing filter: averages values in each kernel window.
    Useful for denoising and local feature aggregation.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        return (1 - gate) * x + gate * self.pool(x)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


class MinPool2dOp(nn.Module):
    """MinPool2d = -MaxPool(-x) with stride=1 same-padding and gated residual.

    Complement of MaxPool: detects minimum (darkest/lowest) features in each window.
    Useful for detecting valleys, shadows, and background-dominant patterns.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        return (1 - gate) * x + gate * (-self.pool(-x))

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


class MixedPool2dOp(nn.Module):
    """Learnable blend of MaxPool and AvgPool with gated residual.

    out = alpha * MaxPool(x) + (1-alpha) * AvgPool(x)
    where alpha = sigmoid(mix_logit) is learned.

    Initialized at alpha=0.5 (equal mix). The model discovers whether
    max pooling, avg pooling, or some blend works best for the task.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.max_pool = nn.MaxPool2d(kernel_size, stride=1, padding=padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5 → equal mix
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_logit)
        alpha = torch.sigmoid(self.mix_logit)
        pooled = alpha * self.max_pool(x) + (1 - alpha) * self.avg_pool(x)
        return (1 - gate) * x + gate * pooled

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"


# ==================== Graph Operations ====================
# Operations for graph-structured data. Access graph topology via
# auxiliary data (adjacency, edge_index, degree) stored on the model.
# All follow the gated residual pattern: (1-gate)*x + gate*op(x).
# Inserted by graph treatments, NOT in the base candidate pool.


def _graph_scatter_add(src: torch.Tensor, index: torch.Tensor,
                       dim: int = 0, dim_size: int = 0) -> torch.Tensor:
    """Scatter-add: sum src values grouped by index along dim.

    Falls back to manual implementation (no torch_scatter dependency).
    """
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    return out


def _graph_scatter_max(src: torch.Tensor, index: torch.Tensor,
                       dim_size: int = 0) -> torch.Tensor:
    """Scatter-max: per-group maximum of src values along dim 0."""
    out = torch.full((dim_size,), float('-inf'), device=src.device, dtype=src.dtype)
    out.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
    return out


def _build_symmetric_norm_sparse(adj_sparse: torch.Tensor, degree: torch.Tensor,
                                   add_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
    """Build D_hat^{-1/2} A_hat D_hat^{-1/2} symmetric normalized adjacency.

    A_hat = A + I (with self-loops), D_hat = degree(A_hat).
    Returns sparse tensor components: (indices, values, size, degree_hat).

    This is the standard GCN normalization (Kipf & Welling 2017).
    """
    adj_coalesced = adj_sparse.coalesce()
    indices = adj_coalesced.indices()
    values = adj_coalesced.values().float()
    N = adj_coalesced.size()[0]

    if add_self_loops:
        # Add self-loops: A_hat = A + I
        self_loop_indices = torch.arange(N, device=indices.device).unsqueeze(0).expand(2, -1)
        self_loop_values = torch.ones(N, device=values.device, dtype=values.dtype)
        indices = torch.cat([indices, self_loop_indices], dim=1)
        values = torch.cat([values, self_loop_values])

    # Recompute degree for A_hat
    degree_hat = torch.zeros(N, device=values.device, dtype=values.dtype)
    degree_hat.scatter_add_(0, indices[1], values)

    # D_hat^{-1/2}
    inv_sqrt_deg = 1.0 / degree_hat.clamp(min=1e-8).sqrt()

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    norm_values = values * inv_sqrt_deg[indices[0]] * inv_sqrt_deg[indices[1]]

    size = torch.Size([N, N])
    return indices, norm_values, size, degree_hat


def _update_graph_op_data(op: nn.Module, adj_sparse: torch.Tensor,
                          degree: torch.Tensor, num_nodes: int,
                          edge_index: torch.Tensor) -> None:
    """Update a graph op's stored graph data for inductive train/eval swapping.

    Generic function that detects which buffers the op has and updates them.
    Works for all graph op types (NeighborAggregation, SGConv, GraphDiffusion,
    DegreeScaling, GraphNorm, EdgeWeightedAggregation, etc.) because they all
    follow the same buffer naming patterns.

    Args:
        op: A graph operation module
        adj_sparse: [N, N] sparse adjacency tensor (new graph)
        degree: [N] degree tensor (new graph)
        num_nodes: number of nodes in the new graph
        edge_index: [2, E] edge index tensor (new graph)
    """
    device = next(op.parameters(), torch.tensor(0.0)).device
    adj_sparse = adj_sparse.to(device)
    degree = degree.to(device)
    edge_index = edge_index.to(device)

    # Update num_nodes -- some ops store it as a plain attribute (e.g.
    # GraphAttentionAggregation), others derive it from _adj_size via a
    # read-only @property (e.g. NeighborAggregation).  For the latter,
    # updating _adj_size below is sufficient; skip the property silently.
    try:
        op.num_nodes = num_nodes
    except AttributeError:
        pass  # derived from _adj_size -- updated below

    # Update degree buffers
    if hasattr(op, '_degree_raw'):
        op._degree_raw = degree.clone()
    if hasattr(op, '_log_degree'):
        op._log_degree = (degree + 1.0).log()

    # Update raw adjacency buffers
    adj_coalesced = adj_sparse.coalesce()
    if hasattr(op, '_adj_indices'):
        op._adj_indices = adj_coalesced.indices()
        op._adj_values = adj_coalesced.values()
        op._adj_size = adj_coalesced.size()
    if hasattr(op, '_adj_raw_indices'):
        op._adj_raw_indices = adj_coalesced.indices()
        op._adj_raw_values = adj_coalesced.values()
        op._adj_raw_size = adj_coalesced.size()

    # Rebuild symmetric normalized adjacency
    norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
        adj_sparse, degree, add_self_loops=True)
    norm_indices = norm_indices.to(device)
    norm_values = norm_values.to(device)

    if hasattr(op, '_norm_indices'):
        op._norm_indices = norm_indices
        op._norm_values = norm_values
        op._norm_size = norm_size

    # Row-normalized adjacency (for GraphSAGE-like ops)
    if hasattr(op, '_rownorm_indices'):
        # Row normalize: D^{-1} A (each row sums to 1)
        adj_hat = adj_sparse.coalesce()
        hat_indices = adj_hat.indices()
        hat_values = adj_hat.values().float()
        # Add self-loops
        N = num_nodes
        self_loops = torch.arange(N, device=device).unsqueeze(0).expand(2, -1)
        hat_indices = torch.cat([hat_indices, self_loops], dim=1)
        hat_values = torch.cat([hat_values, torch.ones(N, device=device)])
        # Row sums
        row_sum = torch.zeros(N, device=device)
        row_sum.scatter_add_(0, hat_indices[0], hat_values)
        rn_values = hat_values / row_sum[hat_indices[0]].clamp(min=1e-8)
        adj_rownorm = torch.sparse_coo_tensor(hat_indices, rn_values, (N, N)).coalesce()
        op._rownorm_indices = adj_rownorm.indices()
        op._rownorm_values = adj_rownorm.values()
        op._rownorm_size = adj_rownorm.size()

    # Update edge_index for attention-based ops (GAT, EdgeWeightedAgg)
    if hasattr(op, 'edge_index') and isinstance(getattr(op, 'edge_index', None), torch.Tensor):
        op.edge_index = edge_index

    # Recompute Laplacian eigenvectors for GraphPositionalEncoding
    if hasattr(op, '_eigvecs') and hasattr(op, '_compute_laplacian_pe'):
        eigvecs = op._compute_laplacian_pe(adj_sparse, degree, op.k)
        op._eigvecs = eigvecs.to(device)


def _batch_graph_mm(adj_sparse: torch.Tensor, x: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
    """Sparse matmul supporting batched [B*N, d] input.

    Since the adjacency A is identical for all snapshots in a batch,
    reshapes [B*N, d] -> [N, B*d], does one sparse.mm, reshapes back.
    Falls through to regular sparse.mm when B=1.
    """
    if x.shape[0] == num_nodes:
        return torch.sparse.mm(adj_sparse, x)
    B = x.shape[0] // num_nodes
    d = x.shape[1]
    # [B*N, d] -> [B, N, d] -> [N, B, d] -> [N, B*d]
    x_2d = x.view(B, num_nodes, d).permute(1, 0, 2).reshape(num_nodes, B * d)
    out_2d = torch.sparse.mm(adj_sparse, x_2d)
    # [N, B*d] -> [N, B, d] -> [B, N, d] -> [B*N, d]
    return out_2d.view(num_nodes, B, d).permute(1, 0, 2).reshape(B * num_nodes, d)


class NeighborAggregation(nn.Module):
    """GCN-style neighbor aggregation with gated residual and symmetric normalization.

    out = (1-gate) * x + gate * (D_hat^{-1/2} A_hat D_hat^{-1/2}) x W

    where A_hat = A + I (self-loops), D_hat = degree(A_hat).
    This is the standard GCN normalization (Kipf & Welling 2017).

    Works on [N, d] or [B*N, d] where N = num_nodes. Uses sparse adjacency.
    Gate initialized from config.graph_initial_gate (default sigmoid(-1)=0.27).

    Graph structure is stored as registered buffers for proper device handling
    and state_dict serialization.
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw adjacency/degree for dimension resizing
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()
        self.register_buffer('_degree_raw', degree.clone())

        # Precompute symmetric normalized adjacency with self-loops
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_norm_indices', norm_indices)
        self.register_buffer('_norm_values', norm_values)
        self._norm_size = norm_size

    @property
    def adj(self) -> torch.Tensor:
        """Raw adjacency (for rebuilding after add_neuron)."""
        return torch.sparse_coo_tensor(
            self._adj_indices, self._adj_values, self._adj_size
        )

    @property
    def adj_norm(self) -> torch.Tensor:
        """Symmetric normalized A_hat (for forward pass)."""
        return torch.sparse_coo_tensor(
            self._norm_indices, self._norm_values, self._norm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._adj_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        adj_n = self.adj_norm
        agg = _batch_graph_mm(adj_n, x, self.num_nodes)
        out = self.linear(agg)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._adj_size[0]}, gate_init={self.gate.item():.2f}"


class GraphAttentionAggregation(nn.Module):
    """GAT-style attention-weighted neighbor aggregation with gated residual.

    Computes attention weights per edge, then weighted sum of neighbor features.
    Single-head (multi-head via config.graph_attention_heads).

    Gate initialized from config.graph_initial_gate.
    """

    def __init__(self, d: int, edge_index: torch.Tensor, num_nodes: int,
                 heads: int = 1, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.linear = nn.Linear(d, d, bias=False)
        self.attn_src = nn.Parameter(torch.randn(d) * 0.01)
        self.attn_dst = nn.Parameter(torch.randn(d) * 0.01)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        self.register_buffer('edge_index', edge_index.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            return torch.cat([self._forward_single(x[b*N:(b+1)*N]) for b in range(B)], dim=0)
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        h = self.linear(x)

        src_idx = self.edge_index[0]
        dst_idx = self.edge_index[1]

        # Edge-level attention scores
        src_scores = (h[src_idx] * self.attn_src).sum(-1)   # [E]
        dst_scores = (h[dst_idx] * self.attn_dst).sum(-1)   # [E]
        edge_attn = F.leaky_relu(src_scores + dst_scores, 0.2)

        # Softmax per destination node (numerically stable)
        max_per_dst = _graph_scatter_max(edge_attn, dst_idx, dim_size=self.num_nodes)
        edge_attn = edge_attn - max_per_dst[dst_idx]
        edge_attn = edge_attn.exp()
        sum_per_dst = _graph_scatter_add(
            edge_attn.unsqueeze(-1), dst_idx, dim=0, dim_size=self.num_nodes
        ).squeeze(-1)
        edge_attn = edge_attn / (sum_per_dst[dst_idx] + 1e-8)

        # Weighted aggregation
        msg = h[src_idx] * edge_attn.unsqueeze(-1)    # [E, d]
        agg = _graph_scatter_add(msg, dst_idx, dim=0, dim_size=self.num_nodes)

        return (1 - gate) * x + gate * agg

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self.num_nodes}, edges={self.edge_index.shape[1]}"


class GraphDiffusion(nn.Module):
    """K-hop diffusion with learnable hop weights + gated residual.

    out = (1-gate) * x + gate * sum_k(alpha_k * A_norm^k * W * x)

    Hop weights alpha_k are softmax-normalized learnable parameters.
    Captures multi-scale neighborhood structure (1-hop, 2-hop, ..., K-hop).
    A_norm = D_hat^{-1/2} A_hat D_hat^{-1/2} (symmetric normalization with self-loops).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 max_hops: int = 3, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.max_hops = max_hops
        self.linear = nn.Linear(d, d, bias=False)
        self.hop_weights = nn.Parameter(torch.zeros(max_hops + 1))  # [K+1] incl. hop-0
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store degree for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())

        # Store raw adjacency for resizing
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Precompute D_hat^{-1/2} A_hat D_hat^{-1/2} with self-loops
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_adj_norm_indices', norm_indices)
        self.register_buffer('_adj_norm_values', norm_values)
        self._adj_norm_size = norm_size

    @property
    def adj_raw(self) -> torch.Tensor:
        """Raw adjacency (for rebuilding after add_neuron)."""
        return torch.sparse_coo_tensor(
            self._adj_raw_indices, self._adj_raw_values, self._adj_raw_size
        )

    @property
    def adj_norm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._adj_norm_indices, self._adj_norm_values, self._adj_norm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._adj_norm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        alphas = F.softmax(self.hop_weights, dim=0)
        h = self.linear(x)
        adj_n = self.adj_norm
        N = self.num_nodes

        out = alphas[0] * h        # hop-0: self
        power = h
        for k in range(1, self.max_hops + 1):
            power = _batch_graph_mm(adj_n, power, N)
            out = out + alphas[k] * power

        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, max_hops={self.max_hops}, nodes={self._adj_norm_size[0]}"


class DirectionalDiffusion(nn.Module):
    """Directional diffusion with forward and backward transition matrices.

    P_f = D_f^{-1} A   (forward: row-normalized — traffic flows OUT from node)
    P_b = D_b^{-1} A^T (backward: row-normalized on transpose — traffic flows IN)

    out = (1-gate)*x + gate * (W_f * sum_k(alpha_k * P_f^k * x)
                              + W_b * sum_k(alpha_k * P_b^k * x))

    K-hop powers with softmax-normalized learnable hop weights.
    Captures directional traffic flow patterns that symmetric GCN misses.
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 max_hops: int = 3, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.max_hops = max_hops
        self.linear_fwd = nn.Linear(d, d, bias=False)
        self.linear_bwd = nn.Linear(d, d, bias=False)
        self.hop_weights = nn.Parameter(torch.zeros(max_hops + 1))  # shared hop weights
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw adjacency/degree for dimension resizing
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()
        self.register_buffer('_degree_raw', degree.clone())

        # Build P_f = D_row^{-1} A (row-normalized, no self-loops)
        indices_f = adj_coalesced.indices()
        values_f = adj_coalesced.values().float()
        N = adj_coalesced.size()[0]
        # Row-sum for forward: D_row[i] = sum_j A[i,j]
        row_deg = torch.zeros(N, device=values_f.device, dtype=values_f.dtype)
        row_deg.scatter_add_(0, indices_f[0], values_f)
        inv_row_deg = 1.0 / row_deg.clamp(min=1e-8)
        pf_values = values_f * inv_row_deg[indices_f[0]]
        self.register_buffer('_pf_indices', indices_f)
        self.register_buffer('_pf_values', pf_values)

        # Build P_b = D_col^{-1} A^T  (transpose, then row-normalize)
        # A^T: swap src/dst indices
        indices_b = torch.stack([indices_f[1], indices_f[0]], dim=0)
        # Row-sum for backward (= column-sum of A): D_col[j] = sum_i A[i,j]
        col_deg = torch.zeros(N, device=values_f.device, dtype=values_f.dtype)
        col_deg.scatter_add_(0, indices_f[1], values_f)
        inv_col_deg = 1.0 / col_deg.clamp(min=1e-8)
        pb_values = values_f * inv_col_deg[indices_b[0]]
        self.register_buffer('_pb_indices', indices_b)
        self.register_buffer('_pb_values', pb_values)

        self._N = N

    @property
    def num_nodes(self) -> int:
        return self._N

    @property
    def P_f(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._pf_indices, self._pf_values, (self._N, self._N))

    @property
    def P_b(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._pb_indices, self._pb_values, (self._N, self._N))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        alphas = F.softmax(self.hop_weights, dim=0)
        N = self.num_nodes
        pf = self.P_f
        pb = self.P_b

        h_f = self.linear_fwd(x)
        h_b = self.linear_bwd(x)

        # Forward diffusion: sum_k alpha_k * P_f^k * h_f
        out_f = alphas[0] * h_f
        power_f = h_f
        for k in range(1, self.max_hops + 1):
            power_f = _batch_graph_mm(pf, power_f, N)
            out_f = out_f + alphas[k] * power_f

        # Backward diffusion: sum_k alpha_k * P_b^k * h_b
        out_b = alphas[0] * h_b
        power_b = h_b
        for k in range(1, self.max_hops + 1):
            power_b = _batch_graph_mm(pb, power_b, N)
            out_b = out_b + alphas[k] * power_b

        return (1 - gate) * x + gate * (out_f + out_b)

    def extra_repr(self) -> str:
        return f"d={self.d}, max_hops={self.max_hops}, nodes={self._N}"


class AdaptiveGraphConv(nn.Module):
    """Adaptive graph convolution with learned soft adjacency.

    A_adapt = softmax(ReLU(E_src @ E_dst^T))  — [N, N] dense
    out = (1-gate)*x + gate * (A_adapt @ W * x)

    E_src, E_dst are [N, d_emb] learnable node embeddings.
    Discovers hidden spatial correlations beyond the physical road network.
    Works on [N, d] or [B*N, d] batched input.
    """

    def __init__(self, d: int, num_nodes: int, d_emb: int = 10,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self._num_nodes = num_nodes
        self.d_emb = d_emb
        self.E_src = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.1)
        self.E_dst = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.1)
        self.linear = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        N = self._num_nodes
        # Learned adjacency: [N, N]
        A_adapt = F.softmax(F.relu(self.E_src @ self.E_dst.T), dim=1)

        h = self.linear(x)

        if x.shape[0] == N:
            # Single graph: [N, d]
            agg = A_adapt @ h
        else:
            # Batched: [B*N, d] -> single matmul with shared adjacency
            B = x.shape[0] // N
            d = h.shape[1]
            h_2d = h.view(B, N, d).permute(1, 0, 2).reshape(N, B * d)
            agg = (A_adapt @ h_2d).reshape(N, B, d).permute(1, 0, 2).reshape(B * N, d)

        return (1 - gate) * x + gate * agg

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._num_nodes}, d_emb={self.d_emb}"


class DegreeScaling(nn.Module):
    """Degree-aware feature scaling with learnable weights + gated residual.

    Nodes with many neighbors get different feature emphasis than isolated nodes.
    Uses log(degree + 1) as the conditioning signal.

    Very lightweight -- just element-wise scaling (no matrix multiplications).
    """

    def __init__(self, d: int, degree: torch.Tensor, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.scale = nn.Parameter(torch.zeros(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        self.register_buffer('_degree_raw', degree.clone())
        log_deg = torch.log(degree.float().clamp(min=1) + 1)  # [N]
        self.register_buffer('_log_degree', log_deg)

    @property
    def num_nodes(self) -> int:
        return self._log_degree.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        # [N, 1] * [d] + [d] -> [N, d]
        deg_feat = self._log_degree.unsqueeze(-1) * self.scale + self.bias
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            deg_feat = deg_feat.repeat(B, 1)  # [B*N, d]
        return (1 - gate) * x + gate * (x * torch.sigmoid(deg_feat))

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._log_degree.shape[0]}"


class GraphBranchedBlock(nn.Module):
    """Parallel branches of NeighborAggregation + GraphAttentionAggregation.

    Like BranchedOperationBlock for PDE (combines derivative + polynomial),
    this combines neighbor mean-aggregation + attention-based aggregation.
    A merge linear projects the concatenated branch outputs back to d.

    Gated residual on the merged output.
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 edge_index: torch.Tensor, num_nodes: int, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.branch_agg = NeighborAggregation(d, adj_sparse, degree, initial_gate=initial_gate)
        self.branch_attn = GraphAttentionAggregation(d, edge_index, num_nodes,
                                                      initial_gate=initial_gate)
        self.merge = nn.Linear(2 * d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        b1 = self.branch_agg(x)
        b2 = self.branch_attn(x)
        merged = self.merge(torch.cat([b1, b2], dim=-1))
        return (1 - gate) * x + gate * merged

    def extra_repr(self) -> str:
        return f"d={self.d}, branches=[NeighborAgg, GraphAttn]"


# ==================== New Graph Operations (Phase 3) ====================
# Advanced graph neural network operations for richer structural modeling.


class SpectralConv(nn.Module):
    """ChebNet-style Chebyshev polynomial spectral graph convolution.

    Uses Chebyshev polynomials of the scaled Laplacian for spectral filtering:
        L = I - D^{-1/2} A_hat D^{-1/2}  (normalized Laplacian)
        L_tilde = 2L / lambda_max - I     (scaled to [-1, 1])
        out = (1-gate)*x + gate * sum_k(theta_k * T_k(L_tilde) * x)

    Chebyshev recurrence: T_0=I, T_1=L_tilde, T_k = 2*L_tilde*T_{k-1} - T_{k-2}

    Reference: Defferrard et al., "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering" (NeurIPS 2016).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 K: int = 3, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.K = K
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Per-feature polynomial coefficients: [K+1, d]
        self.theta = nn.Parameter(torch.zeros(K + 1, d))
        nn.init.xavier_uniform_(self.theta)

        # Store raw degree/adjacency for rebuilding
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Compute normalized Laplacian: L = I - D_hat^{-1/2} A_hat D_hat^{-1/2}
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        N = norm_size[0]
        # L = I - norm_adj: negate norm values, add I on diagonal
        # Build as: L_indices = norm_indices + diag_indices
        #           L_values = -norm_values + diag_ones
        diag_indices = torch.arange(N, device=norm_indices.device).unsqueeze(0).expand(2, -1)
        diag_values = torch.ones(N, device=norm_values.device, dtype=norm_values.dtype)

        L_indices = torch.cat([norm_indices, diag_indices], dim=1)
        L_values = torch.cat([-norm_values, diag_values])

        # Coalesce to merge duplicate diagonal entries
        L_sparse = torch.sparse_coo_tensor(L_indices, L_values, norm_size).coalesce()

        # Estimate lambda_max (largest eigenvalue of L) -- for scaling
        # For normalized Laplacian, lambda_max <= 2. Use 2.0 as safe upper bound.
        lambda_max = 2.0

        # L_tilde = 2L/lambda_max - I
        L_tilde_indices = L_sparse.indices()
        L_tilde_values = L_sparse.values() * (2.0 / lambda_max)
        # Subtract I: add -1 on diagonal
        L_tilde_indices = torch.cat([L_tilde_indices, diag_indices], dim=1)
        L_tilde_values = torch.cat([L_tilde_values, -diag_values])
        L_tilde = torch.sparse_coo_tensor(L_tilde_indices, L_tilde_values, norm_size).coalesce()

        self.register_buffer('_L_tilde_indices', L_tilde.indices())
        self.register_buffer('_L_tilde_values', L_tilde.values())
        self._L_tilde_size = L_tilde.size()

    @property
    def L_tilde(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._L_tilde_indices, self._L_tilde_values, self._L_tilde_size
        )

    @property
    def num_nodes(self) -> int:
        return self._L_tilde_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        L_t = self.L_tilde
        N = self.num_nodes

        # Chebyshev polynomial recurrence
        T_prev = x                                     # T_0(L_tilde) * x = x
        out = self.theta[0].unsqueeze(0) * T_prev      # theta_0 * T_0

        if self.K >= 1:
            T_curr = _batch_graph_mm(L_t, x, N)        # T_1(L_tilde) * x = L_tilde * x
            out = out + self.theta[1].unsqueeze(0) * T_curr

        for k in range(2, self.K + 1):
            T_next = 2 * _batch_graph_mm(L_t, T_curr, N) - T_prev
            out = out + self.theta[k].unsqueeze(0) * T_next
            T_prev = T_curr
            T_curr = T_next

        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, K={self.K}, nodes={self._L_tilde_size[0]}"


class PairNorm(nn.Module):
    """PairNorm: anti-over-smoothing normalization for GNNs.

    Centers node features and rescales to prevent all features from converging
    after multiple graph convolution layers.

        x_centered = x - mean(x)
        out = s * x_centered / (||x_centered||_F / sqrt(N))

    Reference: Zhao & Akoglu, "PairNorm: Tackling Oversmoothing in GNNs" (ICLR 2020).

    Note: No gated residual -- this is a normalization layer.
    """

    def __init__(self, d: int, s_init: float = 1.0, num_nodes: int = 0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.s = nn.Parameter(torch.tensor(s_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes if self.num_nodes > 0 else x.shape[0]
        if x.shape[0] > N:
            # Batched: [B*N, d] -> per-graph normalization
            B = x.shape[0] // N
            x_3d = x.view(B, N, -1)                          # [B, N, d]
            mean = x_3d.mean(dim=1, keepdim=True)             # [B, 1, d]
            x_centered = x_3d - mean                          # [B, N, d]
            norm_factor = x_centered.norm(p='fro', dim=(1, 2), keepdim=False).clamp(min=1e-8) / (N ** 0.5)  # [B]
            out = self.s * x_centered / norm_factor.view(B, 1, 1)
            return out.view(x.shape)
        x_centered = x - x.mean(dim=0, keepdim=True)
        norm_factor = x_centered.norm(p='fro').clamp(min=1e-8) / (N ** 0.5)
        return self.s * x_centered / norm_factor

    def extra_repr(self) -> str:
        return f"d={self.d}"


class GraphNorm(nn.Module):
    """GraphNorm: graph-level learnable normalization.

    Controls how much graph-level mean information is preserved:
        out = gamma * (x - alpha * mean(x)) / std(x) + beta

    alpha=0: no centering (preserves graph-level info)
    alpha=1: full centering (like LayerNorm)

    Reference: Cai et al., "GraphNorm: A Principled Approach to Accelerating
    Graph Neural Network Training" (ICML 2021).

    Note: No gated residual -- this is a normalization layer.
    """

    def __init__(self, d: int, num_nodes: int = 0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable centering strength
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        N = self.num_nodes if self.num_nodes > 0 else x.shape[0]
        if x.shape[0] > N:
            # Batched: [B*N, d] -> per-graph normalization
            B = x.shape[0] // N
            x_3d = x.view(B, N, -1)                          # [B, N, d]
            mean = x_3d.mean(dim=1, keepdim=True)             # [B, 1, d]
            x_shifted = x_3d - alpha * mean
            std = x_shifted.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1, d]
            out = self.gamma * x_shifted / std + self.beta
            return out.view(x.shape)
        mean = x.mean(dim=0, keepdim=True)
        x_shifted = x - alpha * mean
        std = x_shifted.std(dim=0, keepdim=True).clamp(min=1e-6)
        return self.gamma * x_shifted / std + self.beta

    def extra_repr(self) -> str:
        return f"d={self.d}"


class MessagePassingGIN(nn.Module):
    """GIN-style isomorphism-aware message passing with gated residual.

    Uses sum aggregation (not mean) to preserve multiset structure:
        out = (1-gate)*x + gate * MLP((1+eps)*x + A*x)

    where A is the raw adjacency (sum of neighbors, no normalization).
    eps is a learnable self-weighting parameter.

    Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Store raw adjacency (no normalization -- GIN uses sum aggregation)
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()
        self.register_buffer('_degree_raw', degree.clone())

    @property
    def adj(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._adj_indices, self._adj_values, self._adj_size
        )

    @property
    def num_nodes(self) -> int:
        return self._adj_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        adj = self.adj
        # Sum aggregation (raw, no normalization)
        agg = _batch_graph_mm(adj, x, self.num_nodes)
        # (1 + eps) * x + sum(neighbors)
        combined = (1 + self.eps) * x + agg
        out = self.mlp(combined)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._adj_size[0]}"


class GraphPositionalEncoding(nn.Module):
    """Laplacian eigenvector positional encoding for graphs.

    Precomputes top-k eigenvectors of the graph Laplacian L = D - A,
    then projects them to feature dimension:
        out = x + gate * proj(eigvecs_k)

    This gives nodes positional awareness in graph space (analogous to
    positional encoding in Transformers).

    Reference: Dwivedi et al., "Benchmarking Graph Neural Networks" (JMLR 2023).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 k: int = 8, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.k = k
        self.gate = nn.Parameter(torch.tensor(initial_gate))
        self.proj = nn.Linear(k, d, bias=False)

        # Store for rebuilding
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Compute Laplacian eigenvectors
        eigvecs = self._compute_laplacian_pe(adj_sparse, degree, k)
        self.register_buffer('_eigvecs', eigvecs)  # [N, k]

    @staticmethod
    def _compute_laplacian_pe(adj_sparse: torch.Tensor, degree: torch.Tensor,
                               k: int) -> torch.Tensor:
        """Compute top-k smallest non-trivial eigenvectors of graph Laplacian."""
        N = degree.shape[0]
        k_actual = min(k, N - 1)  # Can't have more eigenvectors than N-1

        # Build dense Laplacian L = D - A (on CPU for eigendecomposition)
        adj_dense = adj_sparse.coalesce().to_dense().float().cpu()
        D = torch.diag(degree.float().cpu())
        L = D - adj_dense

        try:
            if N > 5000:
                # For large graphs, use scipy sparse eigsh for efficiency
                import scipy.sparse as sp
                import scipy.sparse.linalg as sla
                import numpy as np

                L_np = L.numpy()
                L_sp = sp.csr_matrix(L_np)
                # Smallest k+1 eigenvalues (skip the trivial 0 eigenvalue)
                eigenvalues, eigenvectors = sla.eigsh(
                    L_sp, k=k_actual + 1, which='SM', tol=1e-4
                )
                # Sort by eigenvalue and skip first (trivial eigenvalue = 0)
                idx = np.argsort(eigenvalues)
                eigvecs = torch.tensor(eigenvectors[:, idx[1:k_actual + 1]], dtype=torch.float32)
            else:
                # For small graphs, use dense eigendecomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(L)
                # Skip first eigenvector (constant, eigenvalue=0)
                eigvecs = eigenvectors[:, 1:k_actual + 1].float()
        except Exception:
            # Fallback: random orthogonal vectors if eigendecomposition fails
            eigvecs = torch.randn(N, k_actual)
            eigvecs, _ = torch.linalg.qr(eigvecs)

        # Pad if we got fewer eigenvectors than k
        if eigvecs.shape[1] < k:
            pad = torch.zeros(N, k - eigvecs.shape[1])
            eigvecs = torch.cat([eigvecs, pad], dim=1)

        # Sign ambiguity: fix sign so largest absolute value in each column is positive
        max_idx = eigvecs.abs().argmax(dim=0)
        signs = eigvecs[max_idx, torch.arange(eigvecs.shape[1])].sign()
        eigvecs = eigvecs * signs.unsqueeze(0)

        return eigvecs

    @property
    def num_nodes(self) -> int:
        return self._eigvecs.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        pe = self.proj(self._eigvecs)  # [N, d]
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            pe = pe.unsqueeze(0).expand(B, -1, -1).reshape(x.shape[0], -1)  # [B*N, d]
        return x + gate * pe

    def extra_repr(self) -> str:
        return f"d={self.d}, k={self.k}, nodes={self._eigvecs.shape[0]}"


# ==================== New Graph Operations (Phase 4 — Comprehensive Overhaul) ====================


class APPNPPropagation(nn.Module):
    """APPNP: Approximate Personalized Propagation of Neural Predictions.

    Decouples feature transformation from propagation using power iteration
    with teleport probability (Personalized PageRank):
        Z^(0) = Linear(x)
        Z^(k) = (1-alpha) * A_norm * Z^(k-1) + alpha * Z^(0)
        out = (1-gate)*x + gate * Z^(K)

    This allows adding more propagation hops WITHOUT adding parameters,
    making it more parameter-efficient than stacking GCN layers.

    Reference: Klicpera et al., "Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank" (ICLR 2019).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 alpha: float = 0.1, K: int = 10, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.alpha = alpha
        self.K = K
        self.linear = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Precompute symmetric normalized adjacency
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_norm_indices', norm_indices)
        self.register_buffer('_norm_values', norm_values)
        self._norm_size = norm_size

    @property
    def adj_norm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._norm_indices, self._norm_values, self._norm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._norm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        z0 = self.linear(x)  # Feature transformation
        z = z0
        adj_n = self.adj_norm
        N = self.num_nodes
        for _ in range(self.K):
            z = (1 - self.alpha) * _batch_graph_mm(adj_n, z, N) + self.alpha * z0
        return (1 - gate) * x + gate * z

    def extra_repr(self) -> str:
        return f"d={self.d}, alpha={self.alpha}, K={self.K}, nodes={self._norm_size[0]}"


class GraphSAGEMean(nn.Module):
    """GraphSAGE with mean aggregation + gated residual.

    Mean variant: out = (1-gate)*x + gate * Linear(concat(x, mean_neighbors(x)))
    Uses row-normalized adjacency (D^{-1}A) for mean aggregation.

    Reference: Hamilton et al., "Inductive Representation Learning on
    Large Graphs" (NeurIPS 2017).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(2 * d, d, bias=True)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()

        # Build row-normalized adjacency: D^{-1}A (mean aggregation)
        self._build_row_norm(adj_sparse, degree)

    def _build_row_norm(self, adj_sparse, degree):
        """Build D^{-1}A for mean neighbor aggregation."""
        adj_c = adj_sparse.coalesce()
        indices = adj_c.indices()
        values = adj_c.values().float()
        N = adj_c.size()[0]
        # D^{-1}: row normalization
        inv_deg = 1.0 / degree.float().clamp(min=1e-8)
        norm_values = values * inv_deg[indices[0]]
        self.register_buffer('_rownorm_indices', indices)
        self.register_buffer('_rownorm_values', norm_values)
        self._rownorm_size = torch.Size([N, N])

    @property
    def adj_rownorm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._rownorm_indices, self._rownorm_values, self._rownorm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._rownorm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        # Mean aggregation of neighbors
        neigh_mean = _batch_graph_mm(self.adj_rownorm, x, self.num_nodes)
        # Concatenate self + neighbor features, then project
        out = self.linear(torch.cat([x, neigh_mean], dim=-1))
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._rownorm_size[0]}"


class GraphSAGEGCN(nn.Module):
    """GraphSAGE with GCN-style aggregation + gated residual.

    GCN variant: out = (1-gate)*x + gate * Linear(D_hat^{-1} A_hat * x)
    Uses row-normalized A_hat = A + I (with self-loops).

    Reference: Hamilton et al., NeurIPS 2017.
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d, d, bias=True)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()

        # Build D_hat^{-1} * A_hat (row-normalized with self-loops)
        self._build_row_norm_selfloop(adj_sparse, degree)

    def _build_row_norm_selfloop(self, adj_sparse, degree):
        """Build D_hat^{-1} * A_hat (row-normalized adjacency with self-loops)."""
        adj_c = adj_sparse.coalesce()
        indices = adj_c.indices()
        values = adj_c.values().float()
        N = adj_c.size()[0]

        # Add self-loops
        self_loop_indices = torch.arange(N, device=indices.device).unsqueeze(0).expand(2, -1)
        self_loop_values = torch.ones(N, device=values.device, dtype=values.dtype)
        all_indices = torch.cat([indices, self_loop_indices], dim=1)
        all_values = torch.cat([values, self_loop_values])

        # Compute degree of A_hat
        degree_hat = torch.zeros(N, device=values.device, dtype=values.dtype)
        degree_hat.scatter_add_(0, all_indices[0], all_values)

        # Row normalization: D_hat^{-1}
        inv_deg = 1.0 / degree_hat.clamp(min=1e-8)
        norm_values = all_values * inv_deg[all_indices[0]]

        # Coalesce to merge self-loop duplicates
        adj_norm = torch.sparse_coo_tensor(all_indices, norm_values, torch.Size([N, N])).coalesce()
        self.register_buffer('_rownorm_indices', adj_norm.indices())
        self.register_buffer('_rownorm_values', adj_norm.values())
        self._rownorm_size = torch.Size([N, N])

    @property
    def adj_rownorm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._rownorm_indices, self._rownorm_values, self._rownorm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._rownorm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        agg = _batch_graph_mm(self.adj_rownorm, x, self.num_nodes)
        out = self.linear(agg)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._rownorm_size[0]}"


class GATv2Aggregation(nn.Module):
    """GATv2: Graph Attention Network v2 with dynamic attention + gated residual.

    Fixes the static attention problem of GATv1 by computing attention
    AFTER the shared linear transformation:
        e_ij = a^T * LeakyReLU(W_l * h_i + W_r * h_j)
        out = (1-gate)*x + gate * sum_j(alpha_ij * W_r * h_j)

    This makes attention truly dynamic — it can distinguish between nodes
    that GATv1 cannot.

    Reference: Brody et al., "How Attentive are Graph Attention Networks?"
    (ICLR 2022).
    """

    def __init__(self, d: int, edge_index: torch.Tensor, num_nodes: int,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.linear_l = nn.Linear(d, d, bias=False)
        self.linear_r = nn.Linear(d, d, bias=False)
        self.attn = nn.Parameter(torch.randn(d) * 0.01)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        self.register_buffer('edge_index', edge_index.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            return torch.cat([self._forward_single(x[b*N:(b+1)*N]) for b in range(B)], dim=0)
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        h_l = self.linear_l(x)  # [N, d]
        h_r = self.linear_r(x)  # [N, d]

        src_idx = self.edge_index[0]
        dst_idx = self.edge_index[1]

        # Dynamic attention: computed AFTER linear transformation
        edge_features = F.leaky_relu(h_l[src_idx] + h_r[dst_idx], 0.2)  # [E, d]
        edge_attn = (edge_features * self.attn).sum(-1)  # [E]

        # Softmax per destination node (numerically stable)
        max_per_dst = _graph_scatter_max(edge_attn, dst_idx, dim_size=self.num_nodes)
        edge_attn = edge_attn - max_per_dst[dst_idx]
        edge_attn = edge_attn.exp()
        sum_per_dst = _graph_scatter_add(
            edge_attn.unsqueeze(-1), dst_idx, dim=0, dim_size=self.num_nodes
        ).squeeze(-1)
        edge_attn = edge_attn / (sum_per_dst[dst_idx] + 1e-8)

        # Weighted aggregation using right-transformed features
        msg = h_r[src_idx] * edge_attn.unsqueeze(-1)  # [E, d]
        agg = _graph_scatter_add(msg, dst_idx, dim=0, dim_size=self.num_nodes)

        return (1 - gate) * x + gate * agg

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self.num_nodes}, edges={self.edge_index.shape[1]}"


class SGConv(nn.Module):
    """SGC: Simplified Graph Convolution + gated residual.

    Pre-computes multi-hop aggregation, then applies a single linear:
        out = (1-gate)*x + gate * Linear(A_norm^K * x)

    All graph propagation is in the pre-computation (the K sparse matmuls).
    Per-step cost is just the linear transformation.

    Reference: Wu et al., "Simplifying Graph Convolutional Networks" (ICML 2019).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 K: int = 2, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.K = K
        self.linear = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Precompute symmetric normalized adjacency
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_norm_indices', norm_indices)
        self.register_buffer('_norm_values', norm_values)
        self._norm_size = norm_size

    @property
    def adj_norm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._norm_indices, self._norm_values, self._norm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._norm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        adj_n = self.adj_norm
        N = self.num_nodes
        # K-hop propagation
        h = x
        for _ in range(self.K):
            h = _batch_graph_mm(adj_n, h, N)
        out = self.linear(h)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, K={self.K}, nodes={self._norm_size[0]}"


class DropEdgeAggregation(nn.Module):
    """DropEdge: Edge dropout regularization for graph convolution + gated residual.

    During training, randomly drops p% of edges before GCN-style aggregation.
    During eval, uses the full adjacency. This prevents oversmoothing by
    adding noise to the message passing process.

    out = (1-gate)*x + gate * Linear(dropped_A_norm * x)

    Reference: Rong et al., "DropEdge: Towards Deep Graph Convolutional Networks
    on Node Classification" (ICLR 2020).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 drop_rate: float = 0.2, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.drop_rate = drop_rate
        self.linear = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw adjacency
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()

        # Precompute full normalized adjacency for eval mode
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_norm_indices', norm_indices)
        self.register_buffer('_norm_values', norm_values)
        self._norm_size = norm_size

    @property
    def adj_norm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._norm_indices, self._norm_values, self._norm_size
        )

    def _drop_edges(self) -> torch.Tensor:
        """Create adjacency with randomly dropped edges during training."""
        indices = self._adj_indices
        values = self._adj_values
        N = self._adj_size[0]

        # Random edge mask
        mask = torch.rand(values.shape[0], device=values.device) > self.drop_rate
        kept_indices = indices[:, mask]
        kept_values = values[mask]

        # Add self-loops (never dropped)
        self_loop_indices = torch.arange(N, device=indices.device).unsqueeze(0).expand(2, -1)
        self_loop_values = torch.ones(N, device=values.device, dtype=values.dtype)
        all_indices = torch.cat([kept_indices, self_loop_indices], dim=1)
        all_values = torch.cat([kept_values, self_loop_values])

        # Symmetric normalization on dropped graph
        degree_hat = torch.zeros(N, device=values.device, dtype=values.dtype)
        degree_hat.scatter_add_(0, all_indices[1], all_values)
        inv_sqrt_deg = 1.0 / degree_hat.clamp(min=1e-8).sqrt()
        norm_values = all_values * inv_sqrt_deg[all_indices[0]] * inv_sqrt_deg[all_indices[1]]

        return torch.sparse_coo_tensor(all_indices, norm_values, torch.Size([N, N]))

    @property
    def num_nodes(self) -> int:
        return self._norm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        if self.training:
            adj = self._drop_edges()
        else:
            adj = self.adj_norm
        agg = _batch_graph_mm(adj, x, self.num_nodes)
        out = self.linear(agg)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, drop_rate={self.drop_rate}, nodes={self._norm_size[0]}"


class MixHopConv(nn.Module):
    """MixHop: Higher-order graph convolution via multi-hop feature mixing + gated residual.

    Concatenates features from different powers of the adjacency:
        [A^0*x, A^1*x, A^2*x, ..., A^K*x]
    Then projects the concatenation back to d dimensions.

    Captures multi-scale neighborhood patterns — local (1-hop) and
    global (K-hop) structure simultaneously.

    Reference: Abu-El-Haija et al., "MixHop: Higher-Order Graph Convolutional
    Architectures via Sparsified Neighborhood Mixing" (ICML 2019).
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 max_hops: int = 2, initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.max_hops = max_hops
        # (K+1) * d -> d projection
        self.proj = nn.Linear((max_hops + 1) * d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(initial_gate))

        # Store raw for dimension resizing
        self.register_buffer('_degree_raw', degree.clone())
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_raw_indices', adj_coalesced.indices())
        self.register_buffer('_adj_raw_values', adj_coalesced.values())
        self._adj_raw_size = adj_coalesced.size()

        # Precompute symmetric normalized adjacency
        norm_indices, norm_values, norm_size, _ = _build_symmetric_norm_sparse(
            adj_sparse, degree, add_self_loops=True
        )
        self.register_buffer('_norm_indices', norm_indices)
        self.register_buffer('_norm_values', norm_values)
        self._norm_size = norm_size

    @property
    def adj_norm(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._norm_indices, self._norm_values, self._norm_size
        )

    @property
    def num_nodes(self) -> int:
        return self._norm_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        adj_n = self.adj_norm
        N = self.num_nodes
        hop_features = [x]  # A^0 * x = x
        power = x
        for k in range(self.max_hops):
            power = _batch_graph_mm(adj_n, power, N)
            hop_features.append(power)
        # Concatenate multi-hop features and project
        out = self.proj(torch.cat(hop_features, dim=-1))
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, max_hops={self.max_hops}, nodes={self._norm_size[0]}"


class VirtualNodeOp(nn.Module):
    """Virtual Node: global graph context aggregation + gated residual.

    Adds a virtual node that:
    1. Aggregates all node features (global mean pool)
    2. Transforms them through a small MLP
    3. Broadcasts the global context back to all nodes

    Provides a global information pathway without explicit long-range edges.

    Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry"
    (ICML 2017); Li et al., "Training Graph Neural Networks with 1000 Layers"
    (ICML 2021).
    """

    def __init__(self, d: int, initial_gate: float = -1.0, num_nodes: int = 0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.gate = nn.Parameter(torch.tensor(initial_gate))
        # Small MLP for virtual node update
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        # Learnable virtual node embedding
        self.vn_embedding = nn.Parameter(torch.zeros(d))
        nn.init.xavier_uniform_(self.vn_embedding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        N = self.num_nodes if self.num_nodes > 0 else x.shape[0]
        if x.shape[0] > N:
            # Batched: [B*N, d] -> per-graph virtual node
            B = x.shape[0] // N
            x_3d = x.view(B, N, -1)                                    # [B, N, d]
            global_context = x_3d.mean(dim=1) + self.vn_embedding      # [B, d]
            vn_update = self.mlp(global_context)                        # [B, d]
            vn_broadcast = vn_update.unsqueeze(1).expand_as(x_3d)      # [B, N, d]
            out = (1 - gate) * x_3d + gate * vn_broadcast
            return out.view(x.shape)
        # Global mean pool + virtual node embedding
        global_context = x.mean(dim=0) + self.vn_embedding  # [d]
        # Transform
        vn_update = self.mlp(global_context)  # [d]
        # Broadcast back to all nodes
        return (1 - gate) * x + gate * vn_update.unsqueeze(0)

    def extra_repr(self) -> str:
        return f"d={self.d}"


class EdgeWeightedAggregation(nn.Module):
    """Learnable edge-weight aggregation + gated residual.

    Learns per-edge scalar weights via a small MLP on source-destination features:
        edge_weight_ij = sigma(MLP(h_i || h_j))
        out = (1-gate)*x + gate * sum_j(w_ij * h_j) / sum_j(w_ij)

    More expressive than uniform GCN aggregation (learns which edges matter),
    lighter than full attention (only scalar weights, not vector-valued).
    """

    def __init__(self, d: int, edge_index: torch.Tensor, num_nodes: int,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.num_nodes = num_nodes
        self.gate = nn.Parameter(torch.tensor(initial_gate))
        # Small MLP: 2d -> d -> 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.register_buffer('edge_index', edge_index.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            return torch.cat([self._forward_single(x[b*N:(b+1)*N]) for b in range(B)], dim=0)
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        src_idx = self.edge_index[0]
        dst_idx = self.edge_index[1]

        # Compute edge weights
        edge_features = torch.cat([x[src_idx], x[dst_idx]], dim=-1)  # [E, 2d]
        edge_weights = torch.sigmoid(self.edge_mlp(edge_features))  # [E, 1]

        # Weighted aggregation
        msg = x[src_idx] * edge_weights  # [E, d]
        agg = _graph_scatter_add(msg, dst_idx, dim=0, dim_size=self.num_nodes)

        # Normalize by weight sum
        weight_sum = _graph_scatter_add(edge_weights, dst_idx, dim=0, dim_size=self.num_nodes)
        agg = agg / (weight_sum + 1e-8)

        return (1 - gate) * x + gate * agg

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self.num_nodes}, edges={self.edge_index.shape[1]}"


class MessageBooster(nn.Module):
    """CMPNN-inspired sum*max message aggregation with gated residual.

    Computes enhanced node representations by combining sum and max
    neighbor aggregations via element-wise product:
        msg_sum = sum_j(h_j)
        msg_max = max_j(h_j)
        boosted = MLP(msg_sum * msg_max)
        out = (1-gate)*x + gate * boosted

    The sum*max product captures both total neighbor signal (sum) and
    the strongest neighbor signal (max). Their product amplifies features
    that are both globally present and locally dominant — a key insight
    from CMPNN (Song et al., 2020) that improves molecular property
    prediction without full communicative message passing overhead.

    Domain-agnostic: works on any graph, not just molecular graphs.
    """

    def __init__(self, d: int, adj_sparse: torch.Tensor, degree: torch.Tensor,
                 initial_gate: float = -1.0):
        super().__init__()
        self.d = d
        self.gate = nn.Parameter(torch.tensor(initial_gate))
        # MLP to transform the boosted signal
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Store raw adjacency for sum aggregation (no normalization)
        adj_coalesced = adj_sparse.coalesce()
        self.register_buffer('_adj_indices', adj_coalesced.indices())
        self.register_buffer('_adj_values', adj_coalesced.values())
        self._adj_size = adj_coalesced.size()
        self.register_buffer('_degree_raw', degree.clone())

    @property
    def adj(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._adj_indices, self._adj_values, self._adj_size
        )

    @property
    def num_nodes(self) -> int:
        return self._adj_size[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.num_nodes
        if x.shape[0] > N:
            B = x.shape[0] // N
            return torch.cat([self._forward_single(x[b*N:(b+1)*N]) for b in range(B)], dim=0)
        return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        adj = self.adj
        src_idx = self._adj_indices[0]
        dst_idx = self._adj_indices[1]
        N, d = x.shape

        # Sum aggregation (via sparse matmul)
        msg_sum = torch.sparse.mm(adj, x)  # [N, d]

        # Max aggregation (scatter_reduce over neighbors, 2D)
        neighbor_feats = x[src_idx]  # [E, d]
        msg_max = torch.full((N, d), float('-inf'), device=x.device, dtype=x.dtype)
        msg_max.scatter_reduce_(
            0, dst_idx.unsqueeze(-1).expand_as(neighbor_feats),
            neighbor_feats, reduce='amax', include_self=True
        )
        # Replace -inf with 0 for isolated nodes
        msg_max = msg_max.clamp(min=0.0)

        # Element-wise product: amplifies features present in both
        boosted = msg_sum * msg_max  # [N, d]

        out = self.mlp(boosted)
        return (1 - gate) * x + gate * out

    def extra_repr(self) -> str:
        return f"d={self.d}, nodes={self._adj_size[0]}"


# ==================== Operation Pool ====================
# These are the candidate operations that ASANN can discover and compose.
# Unlike DARTS, these are not averaged — they are inserted into sequential pipelines.

def _disable_op_gating(op: nn.Module) -> nn.Module:
    """Set gate_logit to +100 (sigmoid~1.0) so gated residual becomes pure op(x)."""
    if hasattr(op, 'gate_logit'):
        device = op.gate_logit.device
        op.gate_logit = nn.Parameter(torch.tensor(100.0, device=device),
                                     requires_grad=False)
    return op


def create_operation(name: str, d: int, device: str = "cpu", config=None,
                     spatial_shape=None, **kwargs) -> nn.Module:
    """Create an operation instance by name, parameterized by dimension d where needed.

    For actual insertion into the model. Conv1dBlock uses near-identity init
    for training stability. Conv2dBlock requires config.spatial_shape.

    When config.use_cuda_ops is True and CUDA ops are available, returns
    CUDA-accelerated variants that produce identical results.

    Args:
        name: Operation name from candidate pool.
        d: Flat dimension (used for flat ops).
        device: Target device.
        config: ASANNConfig (optional, used for old Conv2d spatial_shape).
        spatial_shape: (C, H, W) tuple for spatial operations.
        **kwargs: Extra args. graph_data={adj_sparse, edge_index, degree, num_nodes}
                  for graph operations.
    """
    cfg_spatial = config.spatial_shape if config else None
    use_cuda = (config is not None and getattr(config, 'use_cuda_ops', False)
                and _CUDA_OPS_AVAILABLE)
    if use_cuda:
        # CUDA-accelerated flat operations
        flat_ops = {
            "relu": lambda: nn.ReLU(),
            "gelu": lambda: nn.GELU(),
            "tanh": lambda: nn.Tanh(),
            "swish": lambda: nn.SiLU(),
            "softplus": lambda: nn.Softplus(),
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
            "prelu": lambda: nn.PReLU(num_parameters=1),
            "elu": lambda: nn.ELU(),
            "mish": lambda: nn.Mish(),
            "batchnorm": lambda: nn.BatchNorm1d(d),
            "layernorm": lambda: nn.LayerNorm(d),
            "dropout_01": lambda: nn.Dropout(0.1),
            "dropout_03": lambda: nn.Dropout(0.3),
            "activation_noise": lambda: ActivationNoise(d, config=config),
            "conv1d_k3": lambda: Conv1dBlockCUDA(d, kernel_size=3),
            "conv1d_k5": lambda: Conv1dBlockCUDA(d, kernel_size=5),
            "conv1d_k7": lambda: Conv1dBlockCUDA(d, kernel_size=7),
            "conv2d_k3": lambda: Conv2dBlockCUDA(d, kernel_size=3, spatial_shape=cfg_spatial),
            "conv2d_k5": lambda: Conv2dBlockCUDA(d, kernel_size=5, spatial_shape=cfg_spatial),
            "embed_factored": lambda: FactoredEmbeddingCUDA(d),
            "embed_mlp": lambda: MLPEmbeddingCUDA(d),
            "embed_geometric": lambda: GeometricEmbeddingCUDA(d),
            "embed_positional": lambda: PositionalEmbeddingCUDA(d),
            "attn_self": lambda: SelfAttentionOpCUDA(d),
            "attn_multihead": lambda: MultiHeadAttentionOpCUDA(d),
            "attn_cross": lambda: CrossAttentionOpCUDA(d),
            "attn_causal": lambda: CausalAttentionOpCUDA(d),
            # Temporal / sequence ops (CUDA-accelerated)
            "conv1d_dilated_k3": lambda: DilatedConv1dBlockCUDA(d, kernel_size=3, dilation=2),
            "conv1d_dilated_k5": lambda: DilatedConv1dBlockCUDA(d, kernel_size=5, dilation=2),
            "ema_smooth": lambda: EMASmoothCUDA(d),
            "gated_linear_unit": lambda: GatedLinearUnitCUDA(d),
            "temporal_diff": lambda: TemporalDiffCUDA(d),
            # PDE derivative ops (no CUDA variant — use Python class)
            "derivative_conv1d_o1": lambda: DerivativeConv1d(d, order=1),
            "derivative_conv1d_o2": lambda: DerivativeConv1d(d, order=2),
            # Polynomial / branching ops (no CUDA variant)
            "polynomial_deg2": lambda: PolynomialOp(d, degree=2),
            "polynomial_deg3": lambda: PolynomialOp(d, degree=3),
            "kan_linear": lambda: KANLinearOp(d, num_grids=config.kan_grid_size if config else 8),
            "branched_deriv_poly": lambda: BranchedOperationBlock(d, [
                DerivativeConv1d(d, order=1),
                PolynomialOp(d, degree=2),
            ]),
            # GRU op (no CUDA variant — pure PyTorch GRU is already fast)
            "gru": lambda: GRUOp(d),
        }
    else:
        # Standard Python flat operations
        flat_ops = {
            "relu": lambda: nn.ReLU(),
            "gelu": lambda: nn.GELU(),
            "tanh": lambda: nn.Tanh(),
            "swish": lambda: nn.SiLU(),
            "softplus": lambda: nn.Softplus(),
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
            "prelu": lambda: nn.PReLU(num_parameters=1),
            "elu": lambda: nn.ELU(),
            "mish": lambda: nn.Mish(),
            "batchnorm": lambda: nn.BatchNorm1d(d),
            "layernorm": lambda: nn.LayerNorm(d),
            "dropout_01": lambda: nn.Dropout(0.1),
            "dropout_03": lambda: nn.Dropout(0.3),
            "activation_noise": lambda: ActivationNoise(d, config=config),
            "conv1d_k3": lambda: Conv1dBlock(d, kernel_size=3),
            "conv1d_k5": lambda: Conv1dBlock(d, kernel_size=5),
            "conv1d_k7": lambda: Conv1dBlock(d, kernel_size=7),
            "conv2d_k3": lambda: Conv2dBlock(d, kernel_size=3, spatial_shape=cfg_spatial),
            "conv2d_k5": lambda: Conv2dBlock(d, kernel_size=5, spatial_shape=cfg_spatial),
            "embed_factored": lambda: FactoredEmbedding(d),
            "embed_mlp": lambda: MLPEmbedding(d),
            "embed_geometric": lambda: GeometricEmbedding(d),
            "embed_positional": lambda: PositionalEmbedding(d),
            "attn_self": lambda: SelfAttentionOp(d),
            "attn_multihead": lambda: MultiHeadAttentionOp(d),
            "attn_cross": lambda: CrossAttentionOp(d),
            "attn_causal": lambda: CausalAttentionOp(d),
            # Temporal / sequence ops
            "conv1d_dilated_k3": lambda: DilatedConv1dBlock(d, kernel_size=3, dilation=2),
            "conv1d_dilated_k5": lambda: DilatedConv1dBlock(d, kernel_size=5, dilation=2),
            "ema_smooth": lambda: EMASmooth(d),
            "gated_linear_unit": lambda: GatedLinearUnit(d),
            "temporal_diff": lambda: TemporalDiff(d),
            # PDE derivative ops
            "derivative_conv1d_o1": lambda: DerivativeConv1d(d, order=1),
            "derivative_conv1d_o2": lambda: DerivativeConv1d(d, order=2),
            # Polynomial / branching ops
            "polynomial_deg2": lambda: PolynomialOp(d, degree=2),
            "polynomial_deg3": lambda: PolynomialOp(d, degree=3),
            "kan_linear": lambda: KANLinearOp(d, num_grids=config.kan_grid_size if config else 8),
            "branched_deriv_poly": lambda: BranchedOperationBlock(d, [
                DerivativeConv1d(d, order=1),
                PolynomialOp(d, degree=2),
            ]),
            # GRU op (pure PyTorch)
            "gru": lambda: GRUOp(d),
        }

    if name in flat_ops:
        op = flat_ops[name]().to(device)
        if config and not getattr(config, 'op_gating_enabled', True):
            _disable_op_gating(op)
        return op

    # Spatial operations: require spatial_shape = (C, H, W)
    if spatial_shape is not None:
        C, H, W = spatial_shape
        if use_cuda:
            spatial_ops = {
                "spatial_conv2d_k3": lambda: SpatialConv2dOpCUDA(C, H, W, kernel_size=3),
                "spatial_conv2d_k5": lambda: SpatialConv2dOpCUDA(C, H, W, kernel_size=5),
                "spatial_pointwise_1x1": lambda: PointwiseConv2dOpCUDA(C, H, W),
                "spatial_dw_sep_k3": lambda: DepthwiseSeparableConv2dOpCUDA(C, H, W, kernel_size=3),
                "channel_attention": lambda: ChannelAttentionOpCUDA(C),
                "channel_attention_r8": lambda: ChannelAttentionOpCUDA(C, reduction_ratio=8),
                "channel_attention_r16": lambda: ChannelAttentionOpCUDA(C, reduction_ratio=16),
                "batchnorm2d": lambda: nn.BatchNorm2d(C),
                "groupnorm": lambda: nn.GroupNorm(_safe_num_groups(C), C),
                # 2D derivative ops (no CUDA variant — use Python class)
                "derivative_conv2d_dx": lambda: DerivativeConv2d(C, H, W, order=1, axis='x'),
                "derivative_conv2d_dy": lambda: DerivativeConv2d(C, H, W, order=1, axis='y'),
                "derivative_conv2d_dxx": lambda: DerivativeConv2d(C, H, W, order=2, axis='x'),
                "derivative_conv2d_dyy": lambda: DerivativeConv2d(C, H, W, order=2, axis='y'),
                "derivative_conv2d_laplacian": lambda: DerivativeConv2d(C, H, W, order=2, axis='xy'),
                # Spatial polynomial / branching ops
                "spatial_polynomial_deg2": lambda: SpatialPolynomialOp(C, H, W, degree=2),
                "spatial_polynomial_deg3": lambda: SpatialPolynomialOp(C, H, W, degree=3),
                "spatial_branched_diff_react": lambda: SpatialBranchedOperationBlock(C, H, W, [
                    DerivativeConv2d(C, H, W, order=2, axis='xy'),
                    SpatialPolynomialOp(C, H, W, degree=2),
                ]),
                # Capsule convolution (CUDA uses Python class — squash is cheap)
                "capsule_conv2d_d4": lambda: CapsuleConv2dOpCUDA(C, H, W, kernel_size=3, cap_dim=4) if _CUDA_OPS_AVAILABLE else CapsuleConv2dOp(C, H, W, kernel_size=3, cap_dim=4),
                "capsule_conv2d_d2": lambda: CapsuleConv2dOpCUDA(C, H, W, kernel_size=3, cap_dim=2) if _CUDA_OPS_AVAILABLE else CapsuleConv2dOp(C, H, W, kernel_size=3, cap_dim=2),
                # Multi-scale inception-style conv (pure Python — no CUDA variant needed)
                "multi_scale_conv": lambda: MultiScaleConv2dOp(C, H, W),
                # Per-channel PReLU (learned negative slope per channel)
                "spatial_prelu": lambda: nn.PReLU(num_parameters=C),
                # Pooling ops (stride=1, same-padding — channel-independent)
                "max_pool2d_k3": lambda: MaxPool2dOp(kernel_size=3),
                "max_pool2d_k5": lambda: MaxPool2dOp(kernel_size=5),
                "avg_pool2d_k3": lambda: AvgPool2dOp(kernel_size=3),
                "avg_pool2d_k5": lambda: AvgPool2dOp(kernel_size=5),
                "min_pool2d_k3": lambda: MinPool2dOp(kernel_size=3),
                "mixed_pool2d_k3": lambda: MixedPool2dOp(kernel_size=3),
            }
        else:
            spatial_ops = {
                "spatial_conv2d_k3": lambda: SpatialConv2dOp(C, H, W, kernel_size=3),
                "spatial_conv2d_k5": lambda: SpatialConv2dOp(C, H, W, kernel_size=5),
                "spatial_pointwise_1x1": lambda: PointwiseConv2dOp(C, H, W),
                "spatial_dw_sep_k3": lambda: DepthwiseSeparableConv2dOp(C, H, W, kernel_size=3),
                "channel_attention": lambda: ChannelAttentionOp(C),
                "channel_attention_r8": lambda: ChannelAttentionOp(C, reduction_ratio=8),
                "channel_attention_r16": lambda: ChannelAttentionOp(C, reduction_ratio=16),
                "batchnorm2d": lambda: nn.BatchNorm2d(C),
                "groupnorm": lambda: nn.GroupNorm(_safe_num_groups(C), C),
                # 2D derivative ops
                "derivative_conv2d_dx": lambda: DerivativeConv2d(C, H, W, order=1, axis='x'),
                "derivative_conv2d_dy": lambda: DerivativeConv2d(C, H, W, order=1, axis='y'),
                "derivative_conv2d_dxx": lambda: DerivativeConv2d(C, H, W, order=2, axis='x'),
                "derivative_conv2d_dyy": lambda: DerivativeConv2d(C, H, W, order=2, axis='y'),
                "derivative_conv2d_laplacian": lambda: DerivativeConv2d(C, H, W, order=2, axis='xy'),
                # Spatial polynomial / branching ops
                "spatial_polynomial_deg2": lambda: SpatialPolynomialOp(C, H, W, degree=2),
                "spatial_polynomial_deg3": lambda: SpatialPolynomialOp(C, H, W, degree=3),
                "spatial_branched_diff_react": lambda: SpatialBranchedOperationBlock(C, H, W, [
                    DerivativeConv2d(C, H, W, order=2, axis='xy'),
                    SpatialPolynomialOp(C, H, W, degree=2),
                ]),
                # Capsule convolution
                "capsule_conv2d_d4": lambda: CapsuleConv2dOp(C, H, W, kernel_size=3, cap_dim=4),
                "capsule_conv2d_d2": lambda: CapsuleConv2dOp(C, H, W, kernel_size=3, cap_dim=2),
                # Multi-scale inception-style conv
                "multi_scale_conv": lambda: MultiScaleConv2dOp(C, H, W),
                # Per-channel PReLU (learned negative slope per channel)
                "spatial_prelu": lambda: nn.PReLU(num_parameters=C),
                # Pooling ops (stride=1, same-padding — channel-independent)
                "max_pool2d_k3": lambda: MaxPool2dOp(kernel_size=3),
                "max_pool2d_k5": lambda: MaxPool2dOp(kernel_size=5),
                "avg_pool2d_k3": lambda: AvgPool2dOp(kernel_size=3),
                "avg_pool2d_k5": lambda: AvgPool2dOp(kernel_size=5),
                "min_pool2d_k3": lambda: MinPool2dOp(kernel_size=3),
                "mixed_pool2d_k3": lambda: MixedPool2dOp(kernel_size=3),
            }
        if name in spatial_ops:
            op = spatial_ops[name]().to(device)
            if config and not getattr(config, 'op_gating_enabled', True):
                _disable_op_gating(op)
            return op

    # Graph operations: require graph_data kwarg with adjacency/edge info
    if name.startswith("graph_"):
        graph_data = kwargs.get('graph_data', None)
        if graph_data is None:
            raise ValueError(f"Graph op '{name}' requires graph_data kwarg "
                             f"with keys: adj_sparse, edge_index, degree, num_nodes")
        max_hops = config.graph_diffusion_max_hops if config else 3
        initial_gate = getattr(config, 'graph_initial_gate', -1.0) if config else -1.0
        spectral_K = getattr(config, 'graph_spectral_K', 3) if config else 3
        pe_k = getattr(config, 'graph_pe_k', 8) if config else 8
        graph_ops = {
            "graph_neighbor_agg": lambda: NeighborAggregation(
                d, graph_data['adj_sparse'], graph_data['degree'],
                initial_gate=initial_gate),
            "graph_attention_agg": lambda: GraphAttentionAggregation(
                d, graph_data['edge_index'], graph_data['num_nodes'],
                initial_gate=initial_gate),
            "graph_diffusion_k3": lambda: GraphDiffusion(
                d, graph_data['adj_sparse'], graph_data['degree'],
                max_hops=max_hops, initial_gate=initial_gate),
            "graph_degree_scale": lambda: DegreeScaling(
                d, graph_data['degree'], initial_gate=initial_gate),
            "graph_branched_agg": lambda: GraphBranchedBlock(
                d, graph_data['adj_sparse'], graph_data['degree'],
                graph_data['edge_index'], graph_data['num_nodes'],
                initial_gate=initial_gate),
            # Phase 3 graph ops
            "graph_spectral_conv": lambda: SpectralConv(
                d, graph_data['adj_sparse'], graph_data['degree'],
                K=spectral_K, initial_gate=initial_gate),
            "graph_pairnorm": lambda: PairNorm(d, num_nodes=graph_data['num_nodes']),
            "graph_graphnorm": lambda: GraphNorm(d, num_nodes=graph_data['num_nodes']),
            "graph_gin": lambda: MessagePassingGIN(
                d, graph_data['adj_sparse'], graph_data['degree'],
                initial_gate=initial_gate),
            "graph_positional_enc": lambda: GraphPositionalEncoding(
                d, graph_data['adj_sparse'], graph_data['degree'],
                k=pe_k, initial_gate=initial_gate),
            # Phase 4 graph ops (comprehensive overhaul)
            "graph_appnp": lambda: APPNPPropagation(
                d, graph_data['adj_sparse'], graph_data['degree'],
                alpha=getattr(config, 'graph_appnp_alpha', 0.1) if config else 0.1,
                K=getattr(config, 'graph_appnp_K', 10) if config else 10,
                initial_gate=initial_gate),
            "graph_sage_mean": lambda: GraphSAGEMean(
                d, graph_data['adj_sparse'], graph_data['degree'],
                initial_gate=initial_gate),
            "graph_sage_gcn": lambda: GraphSAGEGCN(
                d, graph_data['adj_sparse'], graph_data['degree'],
                initial_gate=initial_gate),
            "graph_gatv2": lambda: GATv2Aggregation(
                d, graph_data['edge_index'], graph_data['num_nodes'],
                initial_gate=initial_gate),
            "graph_message_boost": lambda: MessageBooster(
                d, graph_data['adj_sparse'], graph_data['degree'],
                initial_gate=initial_gate),
            "graph_sgc": lambda: SGConv(
                d, graph_data['adj_sparse'], graph_data['degree'],
                K=getattr(config, 'graph_sgc_K', 2) if config else 2,
                initial_gate=initial_gate),
            "graph_dropedge": lambda: DropEdgeAggregation(
                d, graph_data['adj_sparse'], graph_data['degree'],
                drop_rate=getattr(config, 'graph_dropedge_rate', 0.2) if config else 0.2,
                initial_gate=initial_gate),
            "graph_mixhop": lambda: MixHopConv(
                d, graph_data['adj_sparse'], graph_data['degree'],
                max_hops=max_hops, initial_gate=initial_gate),
            "graph_virtual_node": lambda: VirtualNodeOp(d, initial_gate=initial_gate, num_nodes=graph_data['num_nodes']),
            "graph_edge_weighted_agg": lambda: EdgeWeightedAggregation(
                d, graph_data['edge_index'], graph_data['num_nodes'],
                initial_gate=initial_gate),
            # Traffic-specific ops (directional flow + learned adjacency)
            "graph_dir_diffusion": lambda: DirectionalDiffusion(
                d, graph_data['adj_sparse'], graph_data['degree'],
                max_hops=max_hops, initial_gate=initial_gate),
            "graph_adaptive": lambda: AdaptiveGraphConv(
                d, graph_data['num_nodes'],
                initial_gate=initial_gate),
        }
        if name in graph_ops:
            op = graph_ops[name]().to(device)
            if config and not getattr(config, 'op_gating_enabled', True):
                _disable_op_gating(op)
            return op

    raise ValueError(f"Unknown operation: {name} (spatial_shape={spatial_shape})")




CANDIDATE_OPERATIONS = [
    "relu", "gelu", "tanh", "swish", "softplus",
    "leaky_relu", "prelu", "elu", "mish",
    "batchnorm", "layernorm",
    "dropout_01", "dropout_03",
    "conv1d_k3", "conv1d_k5", "conv1d_k7",
    "conv1d_dilated_k3", "conv1d_dilated_k5",
    "embed_factored", "embed_mlp",
    "embed_geometric", "embed_positional",
    "attn_self", "attn_multihead", "attn_cross", "attn_causal",
    "ema_smooth", "gated_linear_unit", "temporal_diff",
    "derivative_conv1d_o1", "derivative_conv1d_o2",
    "polynomial_deg2", "polynomial_deg3",
    "kan_linear",
    "branched_deriv_poly",
    "gru",
]

# Spatial operations for layers operating on [B, C, H, W] tensors
SPATIAL_CANDIDATE_OPERATIONS = [
    "relu", "gelu", "swish",
    "leaky_relu", "elu", "mish", "spatial_prelu",
    "batchnorm2d", "groupnorm",
    "dropout_01", "dropout_03",
    "spatial_conv2d_k3", "spatial_conv2d_k5",
    "spatial_pointwise_1x1",
    "spatial_dw_sep_k3",
    "channel_attention", "channel_attention_r8", "channel_attention_r16",
    "derivative_conv2d_dx", "derivative_conv2d_dy",
    "derivative_conv2d_dxx", "derivative_conv2d_dyy",
    "derivative_conv2d_laplacian",
    "spatial_polynomial_deg2", "spatial_polynomial_deg3",
    "spatial_branched_diff_react",
    "capsule_conv2d_d4", "capsule_conv2d_d2",
    "multi_scale_conv",
    # Pooling ops (stride=1, same-padding)
    "max_pool2d_k3", "max_pool2d_k5",
    "avg_pool2d_k3", "avg_pool2d_k5",
    "min_pool2d_k3", "mixed_pool2d_k3",
]

# Graph operations for layers operating on [N, d] node feature tensors
GRAPH_CANDIDATE_OPERATIONS = [
    "relu", "gelu", "swish",
    "leaky_relu", "prelu", "elu", "mish",
    "batchnorm", "layernorm",
    "dropout_01", "dropout_03",
    "embed_factored", "embed_mlp",
    "attn_self",
    # Core message passing
    "graph_neighbor_agg", "graph_attention_agg", "graph_gin",
    # Advanced message passing
    "graph_gatv2", "graph_sage_mean", "graph_sage_gcn",
    "graph_appnp", "graph_sgc", "graph_message_boost",
    # Multi-scale / multi-hop
    "graph_diffusion_k3", "graph_mixhop",
    # Directional flow + adaptive (traffic-specific)
    "graph_dir_diffusion", "graph_adaptive",
    # Spectral
    "graph_spectral_conv",
    # Structural awareness
    "graph_degree_scale", "graph_edge_weighted_agg",
    # Regularization (graph-specific)
    "graph_dropedge", "graph_pairnorm", "graph_graphnorm",
    # Temporal / sequence ops — essential for spatio-temporal graphs (traffic)
    "conv1d_k3", "conv1d_k5",
    "conv1d_dilated_k3", "conv1d_dilated_k5",
    "ema_smooth", "gated_linear_unit", "temporal_diff",
    # PDE / derivative ops — GraphPDE: captures continuous dynamics on graphs
    "derivative_conv1d_o1", "derivative_conv1d_o2",
    "polynomial_deg2", "polynomial_deg3",
    "kan_linear",
    "branched_deriv_poly",
    # Recurrent — sequential feature-group processing
    "gru",
]


def get_candidate_operations(config=None):
    """Return the FLAT candidate operation pool, dynamically adjusted for spatial data.

    When spatial_shape is set (image data), Conv1d operations are replaced
    with Conv2d operations since Conv1d treats features as a 1D sequence
    which is meaningless for spatial (C, H, W) data.

    NOTE: For spatial layers, use get_candidate_operations_for_layer() instead.
    """
    candidates = list(CANDIDATE_OPERATIONS)
    if config and config.spatial_shape is not None:
        # Remove Conv1d (meaningless for spatial data) and add Conv2d
        candidates = [c for c in candidates if not c.startswith("conv1d_")]
        candidates.extend(["conv2d_k3", "conv2d_k5"])
    return candidates


_PHYSICS_OPS = {
    "spatial_polynomial_deg2", "spatial_polynomial_deg3",
    "spatial_branched_diff_react",
    "derivative_conv2d_dx", "derivative_conv2d_dy",
    "derivative_conv2d_dxx", "derivative_conv2d_dyy",
    "derivative_conv2d_laplacian",
    "polynomial_deg2", "polynomial_deg3",
    "branched_deriv_poly",
    "derivative_conv1d_o1", "derivative_conv1d_o2",
}


def get_candidate_operations_for_layer(config, layer, model=None):
    """Return the candidate operation pool appropriate for a specific layer.

    For spatial layers: returns SPATIAL_CANDIDATE_OPERATIONS (ops that work on [B,C,H,W]).
    For graph models: returns GRAPH_CANDIDATE_OPERATIONS (graph-aware ops for [N,d]).
    For flat layers: returns the standard flat pool via get_candidate_operations().

    When config.physics_ops_enabled is False (default), physics-oriented operations
    (derivatives, polynomials, branched) are excluded from the pool.

    Args:
        config: ASANNConfig instance.
        layer: The ASANNLayer to get candidates for.
        model: Optional ASANNModel instance. When provided and model._is_graph,
               returns graph candidate pool. Default None preserves backward compat.
    """
    if hasattr(layer, 'mode') and layer.mode == "spatial":
        candidates = list(SPATIAL_CANDIDATE_OPERATIONS)
    elif model is not None and getattr(model, '_is_graph', False):
        candidates = list(GRAPH_CANDIDATE_OPERATIONS)
    else:
        candidates = get_candidate_operations(config)

    # Filter out physics ops when not enabled
    if not getattr(config, 'physics_ops_enabled', False):
        candidates = [c for c in candidates if c not in _PHYSICS_OPS]

    # Filter out explicitly excluded operations (e.g. O(N^2) ops on large graphs)
    excluded = getattr(config, 'excluded_ops', ())
    if excluded:
        candidates = [c for c in candidates if c not in excluded]

    return candidates


def get_operation_name(op: nn.Module) -> str:
    """Get canonical name for an operation instance."""
    # Unwrap GatedOperation (immunosuppression wrapper)
    from .model import GatedOperation
    if isinstance(op, GatedOperation):
        op = op.operation
    # Check CUDA variants first (if available)
    if _CUDA_OPS_AVAILABLE:
        if isinstance(op, DepthwiseSeparableConv2dOpCUDA):
            return f"spatial_dw_sep_k{op.kernel_size}"
        if isinstance(op, PointwiseConv2dOpCUDA):
            return "spatial_pointwise_1x1"
        if isinstance(op, SpatialConv2dOpCUDA):
            return f"spatial_conv2d_k{op.kernel_size}"
        if isinstance(op, ChannelAttentionOpCUDA):
            if op.reduction_ratio == 8:
                return "channel_attention_r8"
            elif op.reduction_ratio == 16:
                return "channel_attention_r16"
            return "channel_attention"
        if isinstance(op, CapsuleConv2dOpCUDA):
            return f"capsule_conv2d_d{op.cap_dim}"
        if isinstance(op, Conv2dBlockCUDA):
            return f"conv2d_k{op.kernel_size}"
        if isinstance(op, Conv1dBlockCUDA):
            return f"conv1d_k{op.kernel_size}"
        if isinstance(op, FactoredEmbeddingCUDA):
            return "embed_factored"
        if isinstance(op, MLPEmbeddingCUDA):
            return "embed_mlp"
        if isinstance(op, GeometricEmbeddingCUDA):
            return "embed_geometric"
        if isinstance(op, PositionalEmbeddingCUDA):
            return "embed_positional"
        if isinstance(op, SelfAttentionOpCUDA):
            return "attn_self"
        if isinstance(op, MultiHeadAttentionOpCUDA):
            return "attn_multihead"
        if isinstance(op, CrossAttentionOpCUDA):
            return "attn_cross"
        if isinstance(op, CausalAttentionOpCUDA):
            return "attn_causal"
        if isinstance(op, DilatedConv1dBlockCUDA):
            return f"conv1d_dilated_k{op.kernel_size}"
        if isinstance(op, EMASmoothCUDA):
            return "ema_smooth"
        if isinstance(op, GatedLinearUnitCUDA):
            return "gated_linear_unit"
        if isinstance(op, TemporalDiffCUDA):
            return "temporal_diff"

    # PDE derivative ops (Python-only, no CUDA variant)
    if isinstance(op, DerivativeConv1d):
        return f"derivative_conv1d_o{op.order}"
    if isinstance(op, DerivativeConv2d):
        if op.order == 2 and op.axis == 'xy':
            return "derivative_conv2d_laplacian"
        suffix = op.axis if op.order == 1 else op.axis * 2
        return f"derivative_conv2d_d{suffix}"

    # Polynomial / branching ops
    if isinstance(op, KANLinearOp):
        return "kan_linear"
    if isinstance(op, PolynomialOp):
        return f"polynomial_deg{op.degree}"
    if isinstance(op, SpatialPolynomialOp):
        return f"spatial_polynomial_deg{op.degree}"
    if isinstance(op, BranchedOperationBlock):
        return "branched_deriv_poly"
    if isinstance(op, SpatialBranchedOperationBlock):
        return "spatial_branched_diff_react"

    # Graph ops
    if isinstance(op, GraphBranchedBlock):
        return "graph_branched_agg"
    if isinstance(op, NeighborAggregation):
        return "graph_neighbor_agg"
    if isinstance(op, GraphAttentionAggregation):
        return "graph_attention_agg"
    if isinstance(op, GraphDiffusion):
        return f"graph_diffusion_k{op.max_hops}"
    if isinstance(op, DegreeScaling):
        return "graph_degree_scale"
    if isinstance(op, SpectralConv):
        return "graph_spectral_conv"
    if isinstance(op, PairNorm):
        return "graph_pairnorm"
    if isinstance(op, GraphNorm):
        return "graph_graphnorm"
    if isinstance(op, MessagePassingGIN):
        return "graph_gin"
    if isinstance(op, GraphPositionalEncoding):
        return "graph_positional_enc"
    # Phase 4 graph ops
    if isinstance(op, APPNPPropagation):
        return "graph_appnp"
    if isinstance(op, GraphSAGEMean):
        return "graph_sage_mean"
    if isinstance(op, GraphSAGEGCN):
        return "graph_sage_gcn"
    if isinstance(op, GATv2Aggregation):
        return "graph_gatv2"
    if isinstance(op, SGConv):
        return "graph_sgc"
    if isinstance(op, DropEdgeAggregation):
        return "graph_dropedge"
    if isinstance(op, MixHopConv):
        return "graph_mixhop"
    if isinstance(op, VirtualNodeOp):
        return "graph_virtual_node"
    if isinstance(op, EdgeWeightedAggregation):
        return "graph_edge_weighted_agg"
    if isinstance(op, MessageBooster):
        return "graph_message_boost"
    if isinstance(op, DirectionalDiffusion):
        return "graph_dir_diffusion"
    if isinstance(op, AdaptiveGraphConv):
        return "graph_adaptive"

    # Temporal / sequence ops (Python fallback)
    if isinstance(op, DilatedConv1dBlock):
        return f"conv1d_dilated_k{op.kernel_size}"
    if isinstance(op, EMASmooth):
        return "ema_smooth"
    if isinstance(op, GatedLinearUnit):
        return "gated_linear_unit"
    if isinstance(op, TemporalDiff):
        return "temporal_diff"
    if isinstance(op, GRUOp):
        return "gru"
    if isinstance(op, ActivationNoise):
        return "activation_noise"

    # Standard Python variants
    if isinstance(op, DepthwiseSeparableConv2dOp):
        return f"spatial_dw_sep_k{op.kernel_size}"
    if isinstance(op, PointwiseConv2dOp):
        return "spatial_pointwise_1x1"
    if isinstance(op, SpatialConv2dOp):
        return f"spatial_conv2d_k{op.kernel_size}"
    if isinstance(op, CapsuleConv2dOp):
        return f"capsule_conv2d_d{op.cap_dim}"
    if isinstance(op, MultiScaleConv2dOp):
        return "multi_scale_conv"
    if isinstance(op, ChannelAttentionOp):
        if op.reduction_ratio == 8:
            return "channel_attention_r8"
        elif op.reduction_ratio == 16:
            return "channel_attention_r16"
        return "channel_attention"
    if isinstance(op, Conv2dBlock):
        return f"conv2d_k{op.kernel_size}"
    if isinstance(op, Conv1dBlock):
        return f"conv1d_k{op.kernel_size}"
    if isinstance(op, FactoredEmbedding):
        return "embed_factored"
    if isinstance(op, MLPEmbedding):
        return "embed_mlp"
    if isinstance(op, GeometricEmbedding):
        return "embed_geometric"
    if isinstance(op, PositionalEmbedding):
        return "embed_positional"
    if isinstance(op, SelfAttentionOp):
        return "attn_self"
    if isinstance(op, MultiHeadAttentionOp):
        return "attn_multihead"
    if isinstance(op, CrossAttentionOp):
        return "attn_cross"
    if isinstance(op, CausalAttentionOp):
        return "attn_causal"

    # Pooling ops (check before type_map — custom classes)
    if isinstance(op, MixedPool2dOp):
        return f"mixed_pool2d_k{op.kernel_size}"
    if isinstance(op, MinPool2dOp):
        return f"min_pool2d_k{op.kernel_size}"
    if isinstance(op, MaxPool2dOp):
        return f"max_pool2d_k{op.kernel_size}"
    if isinstance(op, AvgPool2dOp):
        return f"avg_pool2d_k{op.kernel_size}"

    # PReLU (check before type_map — needs special num_parameters handling)
    if isinstance(op, nn.PReLU):
        return "spatial_prelu" if op.num_parameters > 1 else "prelu"

    type_map = {
        nn.ReLU: "relu",
        nn.GELU: "gelu",
        nn.Tanh: "tanh",
        nn.SiLU: "swish",
        nn.Softplus: "softplus",
        nn.LeakyReLU: "leaky_relu",
        nn.ELU: "elu",
        nn.Mish: "mish",
        nn.BatchNorm1d: "batchnorm",
        nn.BatchNorm2d: "batchnorm2d",
        nn.GroupNorm: "groupnorm",
        nn.LayerNorm: "layernorm",
        nn.Dropout: None,  # need to check p
    }
    for cls, name in type_map.items():
        if isinstance(op, cls):
            if cls == nn.Dropout:
                p = op.p
                if abs(p - 0.1) < 0.01:
                    return "dropout_01"
                elif abs(p - 0.3) < 0.01:
                    return "dropout_03"
                else:
                    return f"dropout_{p}"
            return name
    return type(op).__name__.lower()


class SurgeryEngine:
    """Executes all four types of real structural surgery on a ASANN model.

    This is NOT a masked supernetwork approach. Every surgery actually modifies
    the model's tensors, computation graph, and memory footprint.

    Surgery Types:
    1. Neuron Surgery — tensors actually resize (Section 3.1)
    2. Layer Surgery  — computation graph changes depth (Section 3.2)
    3. Operation Surgery — sequential pipelines are composed (Section 3.3)
    4. Connection Surgery — skip connections are created/destroyed (Section 3.4)
    """

    def __init__(self, config: ASANNConfig, logger: Optional[SurgeryLogger] = None):
        self.config = config
        self.logger = logger
        self._stride_probe_data = None  # Cached (x_batch, y_batch, loss_fn) for stride probing
        # Consecutive-round tracking for op removal (Fix 3D)
        # Key: (layer_idx, op_name), Value: consecutive rounds flagged for removal
        self._op_removal_history: Dict[Tuple[int, str], int] = {}

    def shift_removal_history(self, at: int, direction: str = "remove"):
        """Shift _op_removal_history indices after layer insertion/removal.

        Same pattern as SurgeryScheduler._shift_identity_counts: when a layer
        is inserted or removed, layer indices in the history dict must be
        remapped so they stay in sync with the model's layer numbering.
        """
        old_history = dict(self._op_removal_history)
        self._op_removal_history.clear()
        for (layer_idx, op_name), count in old_history.items():
            if direction == "remove":
                if layer_idx < at:
                    self._op_removal_history[(layer_idx, op_name)] = count
                elif layer_idx > at:
                    self._op_removal_history[(layer_idx - 1, op_name)] = count
                # layer_idx == at: dropped (layer removed)
            elif direction == "insert":
                if layer_idx < at:
                    self._op_removal_history[(layer_idx, op_name)] = count
                else:
                    self._op_removal_history[(layer_idx + 1, op_name)] = count

    # ==================== NEURON SURGERY (Section 3.1) ====================

    @staticmethod
    def _get_linear(layer):
        """Get the underlying nn.Linear from nn.Linear or ASANNLayer(flat).

        ASANNLayer wraps its Linear in .linear; plain nn.Linear has
        weight/bias directly. This helper ensures neuron surgery works
        with both layer types.
        """
        if hasattr(layer, 'linear') and isinstance(layer.linear, nn.Linear):
            return layer.linear
        return layer

    def add_neuron(
        self,
        model: ASANNModel,
        layer_idx: int,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> bool:
        """Add a neuron to layer layer_idx via splitting (Net2Net style).

        For spatial layers, dispatches to add_channel() instead.
        """
        layer = model.layers[layer_idx]
        # Dispatch: spatial layers use channel surgery
        if hasattr(layer, 'mode') and layer.mode == "spatial":
            return self.add_channel(model, layer_idx, optimizer, step)

        layer = model.layers[layer_idx]
        lin = self._get_linear(layer)  # nn.Linear (or layer itself if already nn.Linear)
        d_old = lin.out_features
        d_in = lin.in_features
        d_new = d_old + 1
        device = lin.weight.device

        # --- Save old parameters ---
        old_weight = lin.weight
        old_bias = lin.bias

        # --- Expand layer output dimension ---
        W_new = torch.zeros(d_new, d_in, device=device)
        W_new[:d_old, :] = lin.weight.data

        # Split the highest-norm neuron
        neuron_norms = lin.weight.data.norm(dim=1)
        j = neuron_norms.argmax().item()
        W_new[d_old, :] = lin.weight.data[j, :] + self.config.split_noise_scale * torch.randn(d_in, device=device)
        # Original neuron stays as-is

        b_new = torch.zeros(d_new, device=device)
        b_new[:d_old] = lin.bias.data
        b_new[d_old] = lin.bias.data[j]

        new_weight = nn.Parameter(W_new)
        new_bias = nn.Parameter(b_new)
        lin.weight = new_weight
        lin.bias = new_bias
        lin.out_features = d_new

        # Register optimizer state transfer for this layer
        optimizer.register_neuron_surgery(old_weight, new_weight, layer_idx, 'neuron_add')
        optimizer.register_neuron_surgery(old_bias, new_bias, layer_idx, 'neuron_add')

        # --- Expand next layer's input dimension ---
        # When JK is enabled, the output_head is fed by JK projections (fixed dim),
        # not by the last layer directly. Skip output_head resize; _rebuild_jk()
        # handles the projection update lazily in forward().
        _is_last = (layer_idx == model.num_layers - 1)
        if _is_last and getattr(model, '_jk_enabled', False):
            pass  # JK projections handle this — no next-layer resize needed
        else:
            if _is_last:
                next_lin = model.output_head
            else:
                next_lin = self._get_linear(model.layers[layer_idx + 1])

            old_next_weight = next_lin.weight
            d_out_next = next_lin.out_features
            W_next_new = torch.zeros(d_out_next, d_new, device=device)
            W_next_new[:, :d_old] = next_lin.weight.data
            # Split incoming weights: halve the split neuron's contribution
            W_next_new[:, d_old] = next_lin.weight.data[:, j] / 2.0
            W_next_new[:, j] = next_lin.weight.data[:, j] / 2.0

            new_next_weight = nn.Parameter(W_next_new)
            next_lin.weight = new_next_weight
            next_lin.in_features = d_new

            optimizer.register_neuron_surgery(old_next_weight, new_next_weight, layer_idx, 'neuron_add')

        # --- Update skip connections affected by the dimension change ---
        # h[layer_idx+1] now has dim d_new (layer[layer_idx].out changed).
        # conn.source == layer_idx+1 reads h[layer_idx+1] → source dim changed
        # conn.target == T adds to h[T-1]. We need h[T-1] == h[layer_idx+1],
        #   so T == layer_idx + 2 → target dim changed
        for conn in model.connections:
            if conn.source == layer_idx + 1:
                self._update_connection_source_dim(model, conn, d_new, optimizer)
            if conn.target == layer_idx + 2:
                self._update_connection_target_dim(model, conn, d_new, optimizer)

        # --- Update BatchNorm/LayerNorm in the operation pipeline if present ---
        self._update_ops_dimensions(model.ops[layer_idx], d_new, device)

        if self.logger:
            self.logger.log_surgery(step, "add_neuron", {
                "layer": layer_idx,
                "split_neuron": j,
                "new_width": d_new,
            })

        return True

    def remove_neuron(
        self,
        model: ASANNModel,
        layer_idx: int,
        neuron_idx: int,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> bool:
        """Remove a neuron from layer layer_idx.

        For spatial layers, dispatches to remove_channel() instead.
        """
        layer = model.layers[layer_idx]
        # Dispatch: spatial layers use channel surgery
        if hasattr(layer, 'mode') and layer.mode == "spatial":
            return self.remove_channel(model, layer_idx, neuron_idx, optimizer, step)

        layer = model.layers[layer_idx]
        lin = self._get_linear(layer)
        d_old = lin.out_features

        # Enforce minimum width constraint
        if d_old <= self.config.d_min:
            return False

        device = lin.weight.device
        mask = torch.ones(d_old, dtype=torch.bool, device=device)
        mask[neuron_idx] = False
        d_new = d_old - 1

        # --- Save old parameters ---
        old_weight = lin.weight
        old_bias = lin.bias

        # --- Shrink layer output ---
        new_weight = nn.Parameter(lin.weight.data[mask, :])
        new_bias = nn.Parameter(lin.bias.data[mask])
        lin.weight = new_weight
        lin.bias = new_bias
        lin.out_features = d_new

        optimizer.register_neuron_surgery(old_weight, new_weight, layer_idx, 'neuron_remove')
        optimizer.register_neuron_surgery(old_bias, new_bias, layer_idx, 'neuron_remove')

        # --- Shrink next layer's input ---
        # When JK is enabled, skip output_head resize (JK projections handle it).
        _is_last = (layer_idx == model.num_layers - 1)
        if _is_last and getattr(model, '_jk_enabled', False):
            pass  # JK projections handle this — no next-layer resize needed
        else:
            if _is_last:
                next_lin = model.output_head
            else:
                next_lin = self._get_linear(model.layers[layer_idx + 1])

            old_next_weight = next_lin.weight
            new_next_weight = nn.Parameter(next_lin.weight.data[:, mask])
            next_lin.weight = new_next_weight
            next_lin.in_features = d_new

            optimizer.register_neuron_surgery(old_next_weight, new_next_weight, layer_idx, 'neuron_remove')

        # --- Update skip connections ---
        # h[layer_idx+1] now has dim d_new.
        # conn.source == layer_idx+1 reads h[layer_idx+1] → source dim changed
        # conn.target == layer_idx+2 adds to h[layer_idx+1] → target dim changed
        for conn in model.connections:
            if conn.source == layer_idx + 1:
                self._update_connection_source_dim(model, conn, d_new, optimizer)
            if conn.target == layer_idx + 2:
                self._update_connection_target_dim(model, conn, d_new, optimizer)

        # --- Update operations ---
        self._update_ops_dimensions(model.ops[layer_idx], d_new, device)

        if self.logger:
            self.logger.log_surgery(step, "remove_neuron", {
                "layer": layer_idx,
                "neuron": neuron_idx,
                "new_width": d_new,
            })

        return True

    # ==================== LAYER SURGERY (Section 3.2) ====================

    # --- Spatial shape helpers ---

    def _recompute_spatial_shapes(self, model: ASANNModel):
        """Recompute spatial_shape on all spatial layers based on actual strides.

        O(num_layers) with pure integer arithmetic — no tensor ops.
        Called after committing a stride-2 layer to fix downstream metadata.
        """
        if not model._is_spatial:
            return
        _, H, W = model.config.spatial_shape  # Start from input resolution
        for l in range(model.num_layers):
            layer = model.layers[l]
            if not (hasattr(layer, 'mode') and layer.mode == 'spatial'):
                break
            stride = getattr(layer, '_stride', 1) or 1
            H = H // stride
            W = W // stride
            C = layer.spatial_shape[0]  # Channels unchanged
            layer.spatial_shape = (C, H, W)

    def _count_stride2_layers(self, model: ASANNModel) -> int:
        """Count current number of stride-2 spatial layers."""
        return sum(
            1 for l in range(model.num_layers)
            if hasattr(model.layers[l], '_stride') and model.layers[l]._stride == 2
        )

    def _probe_stride(
        self,
        model: ASANNModel,
        position: int,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn,
    ) -> int:
        """Probe whether stride-2 is better than stride-1 for a new layer.

        Temporarily inserts candidate layers and compares forward-pass losses.
        Model state is fully restored after probing regardless of outcome.

        Returns 1 or 2 (the winning stride).
        """
        device = self.config.device
        prev_layer = model.layers[position]
        _, H, W = prev_layer.spatial_shape
        C = prev_layer.out_channels

        # Guards: check if stride-2 is even viable
        if H // 2 < self.config.min_spatial_resolution:
            return 1
        if W // 2 < self.config.min_spatial_resolution:
            return 1
        if self._count_stride2_layers(model) >= self.config.max_downsample_stages:
            return 1
        if not getattr(model, '_use_gap', False):
            return 1  # Non-GAP models can't handle stride changes

        # Save full model state for restoration
        saved_layers = list(model.layers)
        saved_ops = list(model.ops)
        saved_connections = model.connections[:]  # Shallow copy of list
        saved_conn_indices = [(c.source, c.target) for c in model.connections]
        saved_flatten = model._flatten_position

        # Drop-path rate for new layer
        _dp_rate = 0.0
        if self.config.drop_path_enabled and self.config.drop_path_rate > 0:
            new_depth = model.num_layers + 1
            new_pos = position + 1
            _dp_rate = self.config.drop_path_rate * new_pos / max(new_depth - 1, 1)

        # Create stride-1 candidate (same channels — no cascade)
        layer_s1 = ASANNLayer(
            mode="spatial", C_in=C, C_out=C, H=H, W=W,
            stride=1, drop_path_rate=_dp_rate,
        ).to(device)
        ops_new = SpatialOperationPipeline().to(device)
        ops_new.add_operation(nn.ReLU(), 0)

        try:
            # Temporarily disconnect all skip connections during probing.
            # Shifting indices alone isn't enough — projections have stale
            # dimensions after layer insertion. Both stride-1 and stride-2
            # probes are equally affected, so the comparison remains fair.
            model.connections = []

            # --- Probe stride=1 ---
            model.layers = nn.ModuleList(
                saved_layers[:position + 1] + [layer_s1] + saved_layers[position + 1:]
            )
            model.ops = nn.ModuleList(
                saved_ops[:position + 1] + [ops_new] + saved_ops[position + 1:]
            )
            model._recompute_flatten_position()

            model.eval()
            with torch.no_grad():
                out_s1 = model(x_batch)
                loss_s1 = loss_fn(out_s1, y_batch).item()

            # --- Probe stride=2 ---
            layer_s2 = ASANNLayer(
                mode="spatial", C_in=C, C_out=C, H=H, W=W,
                stride=2, drop_path_rate=_dp_rate,
            ).to(device)
            model.layers[position + 1] = layer_s2

            with torch.no_grad():
                out_s2 = model(x_batch)
                loss_s2 = loss_fn(out_s2, y_batch).item()

        finally:
            # Restore everything regardless of success/failure
            model.layers = nn.ModuleList(saved_layers)
            model.ops = nn.ModuleList(saved_ops)
            model.connections = saved_connections
            model._flatten_position = saved_flatten

        # Decision: stride-2 wins if within tolerance threshold
        threshold = self.config.stride_probe_threshold  # 1.02 = tolerate 2% worse
        if loss_s2 < loss_s1 * threshold:
            print(f"  [STRIDE PROBE] stride=2 wins at pos {position}: "
                  f"loss_s1={loss_s1:.4f}, loss_s2={loss_s2:.4f}")
            return 2
        else:
            print(f"  [STRIDE PROBE] stride=1 wins at pos {position}: "
                  f"loss_s1={loss_s1:.4f}, loss_s2={loss_s2:.4f}")
            return 1

    def add_layer(
        self,
        model: ASANNModel,
        position: int,
        optimizer,
        step: int,
        stride: Optional[int] = None,
    ):
        """Add a near-identity layer at the given position.

        For spatial models: adds a spatial layer (Conv2d + BN + residual) if
        the position is within the spatial region, otherwise adds a flat layer.

        Args:
            stride: Force a specific stride (1 or 2). If None, auto-detect:
                    uses stride probing when spatial_downsample_stages="auto"
                    and probe data is available, otherwise defaults to 1.

        Initialized as near-identity to minimize disruption.
        Optimizer is updated in-place via register_structural_surgery().
        """
        device = self.config.device
        prev_layer = model.layers[position]

        # Determine if we should add a spatial or flat layer
        if (model._is_spatial and hasattr(prev_layer, 'mode')
                and prev_layer.mode == "spatial"):
            # Add spatial layer: match channels and spatial dims of previous layer
            C = prev_layer.out_channels
            _, H, W = prev_layer.spatial_shape

            # Determine stride
            if stride is None:
                if (self.config.spatial_downsample_stages == "auto"
                        and self._stride_probe_data is not None):
                    x_batch, y_batch, loss_fn = self._stride_probe_data
                    stride = self._probe_stride(
                        model, position, x_batch, y_batch, loss_fn)
                else:
                    stride = 1

            # Compute drop_path_rate for the new layer based on its position
            _dp_rate = 0.0
            if self.config.drop_path_enabled and self.config.drop_path_rate > 0:
                new_depth = model.num_layers + 1  # after insertion
                new_pos = position + 1  # 0-indexed position of new layer
                _dp_rate = self.config.drop_path_rate * new_pos / max(new_depth - 1, 1)
            new_layer = ASANNLayer(
                mode="spatial", C_in=C, C_out=C, H=H, W=W,
                stride=stride, drop_path_rate=_dp_rate,
            ).to(device)
            new_ops = SpatialOperationPipeline().to(device)
            # Start with ReLU, matching initial spatial layers — a layer without
            # activation is just a linear transform and can't learn nonlinear patterns
            new_ops.add_operation(nn.ReLU(), 0)
            layer_type = "spatial"
        else:
            # Add flat layer (original behavior)
            d = prev_layer.out_features
            new_layer = ASANNLayer(mode="flat", d_in=d, d_out=d).to(device)
            # Near-identity init for flat layer
            nn.init.eye_(new_layer.linear.weight)
            new_layer.linear.weight.data += 0.01 * torch.randn_like(new_layer.linear.weight.data)
            nn.init.zeros_(new_layer.linear.bias)
            new_ops = OperationPipeline().to(device)
            # Start with ReLU, matching initial flat layers
            new_ops.add_operation(nn.ReLU(), 0)
            layer_type = "flat"

        # Insert into model
        layers_list = list(model.layers)
        layers_list.insert(position + 1, new_layer)
        model.layers = nn.ModuleList(layers_list)

        ops_list = list(model.ops)
        ops_list.insert(position + 1, new_ops)
        model.ops = nn.ModuleList(ops_list)

        # Update all connection indices that reference layers after the insertion point
        for conn in model.connections:
            if conn.source > position:
                conn.source += 1
            if conn.target > position:
                conn.target += 1

        # Recompute flatten position for spatial models
        if model._is_spatial:
            model._recompute_flatten_position()

        # If stride > 1, recompute spatial shapes on all downstream layers
        if layer_type == "spatial" and stride > 1:
            self._recompute_spatial_shapes(model)

        # After shifting indices, some connections may now span different spatial
        # resolutions (e.g., a stride-2 layer was inserted between source and target).
        # Validate and remove any that are now invalid.
        if model._is_spatial:
            self._remove_spatially_invalid_connections(model)

        # Repair stale skip connection projections: after index shifting,
        # a connection's projection may have wrong channel dimensions because
        # it now points at a different layer than when it was created.
        self._repair_connection_projections(model, optimizer)

        # Sync optimizer in-place (detects new/removed params, preserves state)
        optimizer.register_structural_surgery(model, surgery_type='layer_add')

        if self.logger:
            log_data = {
                "position": position + 1,
                "type": layer_type,
                "width": new_layer.out_features,
                "new_depth": model.num_layers,
            }
            if layer_type == "spatial":
                log_data["stride"] = stride
            self.logger.log_surgery(step, "add_layer", log_data)

        # Rebuild spinal projections after layer add (input re-splits among new layer count)
        if getattr(model, '_spinal_enabled', False):
            model._rebuild_spinal()

    def remove_layer(
        self,
        model: ASANNModel,
        layer_idx: int,
        optimizer,
        step: int,
    ) -> bool:
        """Remove a layer from the computation graph.

        Returns True if successful, False if minimum depth prevents removal.
        """
        # Enforce minimum depth
        if model.num_layers <= self.config.min_depth:
            return False

        layer = model.layers[layer_idx]

        # For spatial layers, enforce min_spatial_layers
        if (model._is_spatial and hasattr(layer, 'mode') and layer.mode == "spatial"):
            spatial_count = sum(1 for l in model.layers
                                if hasattr(l, 'mode') and l.mode == "spatial")
            if spatial_count <= self.config.min_spatial_layers:
                return False

        d_in = layer.in_features
        d_out = layer.out_features
        device = self.config.device

        # If dimensions don't match, adjust the next layer's input
        if d_in != d_out:
            if layer_idx < model.num_layers - 1:
                next_layer = model.layers[layer_idx + 1]
                if isinstance(next_layer, nn.Linear):
                    # Plain nn.Linear: resize input dimension
                    old_weight = next_layer.weight.data
                    new_weight_data = torch.zeros(
                        next_layer.out_features, d_in, device=device
                    )
                    copy_cols = min(d_out, d_in)
                    new_weight_data[:, :copy_cols] = old_weight[:, :copy_cols]
                    next_layer.weight = nn.Parameter(new_weight_data)
                    next_layer.in_features = d_in
                    if next_layer.bias is not None:
                        pass  # bias shape is [out_features], unchanged
                elif hasattr(next_layer, 'mode') and next_layer.mode == "flat":
                    old_weight = next_layer.linear.weight.data
                    new_weight_data = torch.zeros(
                        next_layer.linear.out_features, d_in, device=device
                    )
                    copy_cols = min(d_out, d_in)
                    new_weight_data[:, :copy_cols] = old_weight[:, :copy_cols]
                    next_layer.linear.weight = nn.Parameter(new_weight_data)
                    next_layer.linear.in_features = d_in
                elif hasattr(next_layer, 'mode') and next_layer.mode == "spatial":
                    # Spatial layers: channel mismatch after removal.
                    # Must resize the next layer's Conv2d input channels AND
                    # the BN + residual projection to match the new predecessor's
                    # output channels (= removed layer's input channels).
                    C_in_new = layer._C_in  # predecessor's output channels
                    C_out = next_layer.spatial_shape[0]
                    old_conv = next_layer.conv
                    stride = next_layer._stride or 1

                    # Rebuild Conv2d with new input channels
                    new_conv = nn.Conv2d(
                        C_in_new, C_out, kernel_size=3,
                        stride=stride, padding=1, bias=False,
                    ).to(device)
                    # Copy overlapping weights: [C_out, C_in, 3, 3]
                    copy_c = min(old_conv.in_channels, C_in_new)
                    new_conv.weight.data[:, :copy_c] = old_conv.weight.data[:, :copy_c]
                    next_layer.conv = new_conv
                    next_layer._C_in = C_in_new

                    # Update residual projection if it exists
                    if next_layer.residual_proj is not None:
                        old_res = next_layer.residual_proj
                        new_res = nn.Conv2d(
                            C_in_new, C_out, kernel_size=1,
                            stride=stride, bias=False,
                        ).to(device)
                        copy_c_res = min(old_res.in_channels, C_in_new)
                        new_res.weight.data[:, :copy_c_res] = old_res.weight.data[:, :copy_c_res]
                        next_layer.residual_proj = new_res
                    elif C_in_new != C_out or stride != 1:
                        # Need a new residual projection
                        new_res = nn.Conv2d(
                            C_in_new, C_out, kernel_size=1,
                            stride=stride, bias=False,
                        ).to(device)
                        nn.init.dirac_(new_res.weight)
                        next_layer.residual_proj = new_res
            else:
                # Removing the last layer — adjust output head
                use_gap = getattr(model, '_use_gap', False)
                if use_gap and hasattr(layer, 'mode') and layer.mode == "spatial":
                    # GAP mode: output head is [d_output, C], resize to match
                    # the preceding layer's channel count
                    prev_layer = model.layers[layer_idx - 1] if layer_idx > 0 else None
                    if prev_layer and hasattr(prev_layer, 'out_channels'):
                        new_C = prev_layer.out_channels
                    else:
                        new_C = d_in  # fallback
                    old_weight = model.output_head.weight.data
                    new_weight_data = torch.zeros(
                        model.d_output, new_C, device=device
                    )
                    copy_cols = min(old_weight.shape[1], new_C)
                    new_weight_data[:, :copy_cols] = old_weight[:, :copy_cols]
                    model.output_head.weight = nn.Parameter(new_weight_data)
                    model.output_head.in_features = new_C
                else:
                    old_weight = model.output_head.weight.data
                    new_weight_data = torch.zeros(
                        model.d_output, d_in, device=device
                    )
                    copy_cols = min(d_out, d_in)
                    new_weight_data[:, :copy_cols] = old_weight[:, :copy_cols]
                    model.output_head.weight = nn.Parameter(new_weight_data)
                    model.output_head.in_features = d_in

        # Remove the layer and its operation pipeline
        layers_list = list(model.layers)
        layers_list.pop(layer_idx)
        model.layers = nn.ModuleList(layers_list)

        ops_list = list(model.ops)
        ops_list.pop(layer_idx)
        model.ops = nn.ModuleList(ops_list)

        # Remove connections involving this layer and shift indices
        self._remove_connections_for_layer(model, layer_idx)
        for conn in model.connections:
            if conn.source > layer_idx:
                conn.source -= 1
            if conn.target > layer_idx:
                conn.target -= 1

        # After shifting indices, validate spatial resolution consistency
        if model._is_spatial:
            self._remove_spatially_invalid_connections(model)

        # Repair stale skip connection projections (same issue as add_layer:
        # shifted indices may cause projection channel mismatches).
        self._repair_connection_projections(model, optimizer)

        # Recompute flatten position for spatial models
        if model._is_spatial:
            model._recompute_flatten_position()

        # Sync optimizer in-place
        optimizer.register_structural_surgery(model, surgery_type='layer_remove')

        if self.logger:
            self.logger.log_surgery(step, "remove_layer", {
                "layer_idx": layer_idx,
                "new_depth": model.num_layers,
            })

        # Rebuild spinal projections after layer remove (input re-splits among new layer count)
        if getattr(model, '_spinal_enabled', False):
            model._rebuild_spinal()

        return True

    # ==================== CHANNEL SURGERY (for spatial layers) ====================

    def add_channel(
        self,
        model: ASANNModel,
        layer_idx: int,
        optimizer,
        step: int,
    ) -> bool:
        """Add a channel to a spatial layer via filter splitting (Net2Net style).

        Analogous to add_neuron for flat layers. Splits the highest-norm filter,
        halves downstream incoming weights to preserve function.
        """
        layer = model.layers[layer_idx]
        assert hasattr(layer, 'mode') and layer.mode == "spatial"

        # Enforce max channels
        C_old = layer.out_channels
        if C_old >= self.config.max_channels:
            return False

        # Enforce max channel ratio between adjacent spatial layers
        max_ratio = self.config.max_channel_ratio
        if max_ratio > 0:
            # Check against previous layer's output channels
            if layer_idx > 0:
                prev_layer = model.layers[layer_idx - 1]
                if hasattr(prev_layer, 'mode') and prev_layer.mode == "spatial":
                    prev_C = prev_layer.out_channels
                    if (C_old + 1) / max(prev_C, 1) > max_ratio:
                        return False
            # Check against next layer's output channels
            if layer_idx < model.num_layers - 1:
                next_layer = model.layers[layer_idx + 1]
                if hasattr(next_layer, 'mode') and next_layer.mode == "spatial":
                    next_C = next_layer.out_channels
                    if (C_old + 1) / max(next_C, 1) > max_ratio:
                        return False

        C_in = layer._C_in
        C_new = C_old + 1
        device = layer.conv.weight.device
        k = layer.conv.kernel_size[0]
        _, H, W = layer.spatial_shape

        # --- Save old parameters ---
        old_conv_weight = layer.conv.weight  # [C_old, C_in, k, k]

        # --- Expand conv output filters ---
        W_new = torch.zeros(C_new, C_in, k, k, device=device)
        W_new[:C_old] = layer.conv.weight.data

        # Split highest-norm filter
        filter_norms = layer.conv.weight.data.flatten(1).norm(dim=1)  # [C_old]
        j = filter_norms.argmax().item()
        noise = self.config.split_noise_scale * torch.randn(C_in, k, k, device=device)
        W_new[C_old] = layer.conv.weight.data[j] + noise

        new_conv_weight = nn.Parameter(W_new)
        layer.conv.weight = new_conv_weight
        layer.conv.out_channels = C_new

        optimizer.register_neuron_surgery(old_conv_weight, new_conv_weight,
                                          layer_idx, 'neuron_add')

        # --- Expand BatchNorm ---
        old_bn = layer.bn
        old_bn_weight = old_bn.weight
        old_bn_bias = old_bn.bias
        new_bn = nn.BatchNorm2d(C_new).to(device)
        # Copy stats
        new_bn.weight.data[:C_old] = old_bn.weight.data
        new_bn.weight.data[C_old] = old_bn.weight.data[j]
        new_bn.bias.data[:C_old] = old_bn.bias.data
        new_bn.bias.data[C_old] = old_bn.bias.data[j]
        new_bn.running_mean[:C_old] = old_bn.running_mean
        new_bn.running_mean[C_old] = old_bn.running_mean[j]
        new_bn.running_var[:C_old] = old_bn.running_var
        new_bn.running_var[C_old] = old_bn.running_var[j]
        layer.bn = new_bn

        # Register BN params with optimizer (swap old -> new references)
        optimizer.register_neuron_surgery(old_bn_weight, new_bn.weight,
                                          layer_idx, 'neuron_add')
        optimizer.register_neuron_surgery(old_bn_bias, new_bn.bias,
                                          layer_idx, 'neuron_add')

        # Update spatial shape
        layer.spatial_shape = (C_new, H, W)

        # --- Update structural residual projection ---
        # Projection is needed when C_in != C_out OR stride > 1.
        # A stride > 1 layer always needs projection for spatial downsampling,
        # even when channel counts match.
        needs_proj = (layer._C_in != C_new) or (layer._stride > 1)
        if needs_proj:
            if layer.residual_proj is not None:
                # Resize existing projection — use actual tensor shape for copy
                # (projection may be stale from a prior add_channel in same round)
                old_proj = layer.residual_proj
                new_proj = nn.Conv2d(layer._C_in, C_new, 1,
                                     stride=layer._stride, bias=False).to(device)
                copy_c = min(old_proj.weight.shape[0], C_new)
                new_proj.weight.data[:copy_c] = old_proj.weight.data[:copy_c]
                layer.residual_proj = new_proj
                optimizer.register_neuron_surgery(old_proj.weight, new_proj.weight,
                                                  layer_idx, 'neuron_add')
            else:
                # Create new projection
                layer.residual_proj = nn.Conv2d(
                    layer._C_in, C_new, 1, stride=layer._stride, bias=False
                ).to(device)
                nn.init.dirac_(layer.residual_proj.weight)
                optimizer.register_new_parameters(
                    [layer.residual_proj.weight],
                    group_name='spatial_residual',
                    source_layer=layer_idx,
                    surgery_type='neuron_add')
        else:
            # C_in == C_out and stride == 1 — remove stale projection
            layer.residual_proj = None

        # --- Update next layer's input channels ---
        if layer_idx < model.num_layers - 1:
            next_layer = model.layers[layer_idx + 1]
            if hasattr(next_layer, 'mode') and next_layer.mode == "spatial":
                # Next spatial layer: expand conv input channels
                old_next_w = next_layer.conv.weight  # [C_next_out, C_old, k2, k2]
                k2 = next_layer.conv.kernel_size[0]
                C_next_out = next_layer.conv.out_channels
                W_next = torch.zeros(C_next_out, C_new, k2, k2, device=device)
                W_next[:, :C_old] = next_layer.conv.weight.data
                # Split: halve incoming from split channel
                W_next[:, C_old] = next_layer.conv.weight.data[:, j] / 2.0
                W_next[:, j] = next_layer.conv.weight.data[:, j] / 2.0
                new_next_w = nn.Parameter(W_next)
                next_layer.conv.weight = new_next_w
                next_layer.conv.in_channels = C_new
                next_layer._C_in = C_new
                # Update next layer's residual projection:
                # If C_in now differs from C_out, we need a projection.
                # If they match, no projection needed.
                if next_layer._C_in != next_layer.out_channels or next_layer._stride != 1:
                    # Need a residual projection
                    old_proj = next_layer.residual_proj
                    new_proj = nn.Conv2d(
                        C_new, next_layer.out_channels, 1,
                        stride=next_layer._stride, bias=False
                    ).to(device)
                    if old_proj is not None:
                        # Copy existing weights
                        copy_out = min(old_proj.out_channels, next_layer.out_channels)
                        copy_in = min(old_proj.in_channels, C_new)
                        new_proj.weight.data[:copy_out, :copy_in] = old_proj.weight.data[:copy_out, :copy_in]
                        optimizer.register_neuron_surgery(
                            old_proj.weight, new_proj.weight,
                            layer_idx + 1, 'neuron_add')
                    else:
                        nn.init.dirac_(new_proj.weight)
                        optimizer.register_new_parameters(
                            [new_proj.weight],
                            group_name='spatial_residual',
                            source_layer=layer_idx + 1,
                            surgery_type='neuron_add')
                    next_layer.residual_proj = new_proj
                else:
                    if next_layer.residual_proj is not None:
                        optimizer.register_removed_parameters(
                            [next_layer.residual_proj.weight])
                    next_layer.residual_proj = None  # Dimensions match, no proj needed

                optimizer.register_neuron_surgery(old_next_w, new_next_w,
                                                  layer_idx, 'neuron_add')
            else:
                # Next is flat (across flatten boundary) or output_head
                if layer_idx == model.num_layers - 1:
                    target = model.output_head
                else:
                    target = next_layer.linear if hasattr(next_layer, 'linear') else next_layer

                old_w = target.weight
                d_out = target.out_features
                d_old_in = C_old * H * W
                d_new_in = C_new * H * W
                W_flat = torch.zeros(d_out, d_new_in, device=device)
                W_flat[:, :d_old_in] = target.weight.data
                # The new channel's columns: copy from split channel j, halve
                j_start = j * H * W
                j_end = (j + 1) * H * W
                new_start = C_old * H * W
                new_end = new_start + H * W
                W_flat[:, new_start:new_end] = target.weight.data[:, j_start:j_end] / 2.0
                W_flat[:, j_start:j_end] = target.weight.data[:, j_start:j_end] / 2.0
                new_w = nn.Parameter(W_flat)
                target.weight = new_w
                target.in_features = d_new_in
                optimizer.register_neuron_surgery(old_w, new_w, layer_idx, 'neuron_add')
        else:
            # Last layer -> output head
            old_w = model.output_head.weight
            d_out = model.output_head.out_features
            use_gap = getattr(model, '_use_gap', False)
            if use_gap:
                # GAP mode: head is [d_out, C], add one column for new channel
                d_new_in = C_new
                W_new = torch.zeros(d_out, d_new_in, device=device)
                W_new[:, :C_old] = model.output_head.weight.data
                # Net2Net split: halve parent column, copy to new
                W_new[:, j] = model.output_head.weight.data[:, j] / 2.0
                W_new[:, C_old] = model.output_head.weight.data[:, j] / 2.0
            else:
                # Flatten mode: head is [d_out, C*H*W]
                d_old_in = C_old * H * W
                d_new_in = C_new * H * W
                W_new = torch.zeros(d_out, d_new_in, device=device)
                W_new[:, :d_old_in] = model.output_head.weight.data
                j_start = j * H * W
                j_end = (j + 1) * H * W
                new_start = C_old * H * W
                new_end = new_start + H * W
                W_new[:, new_start:new_end] = model.output_head.weight.data[:, j_start:j_end] / 2.0
                W_new[:, j_start:j_end] = model.output_head.weight.data[:, j_start:j_end] / 2.0
            new_w = nn.Parameter(W_new)
            model.output_head.weight = new_w
            model.output_head.in_features = d_new_in
            optimizer.register_neuron_surgery(old_w, new_w, layer_idx, 'neuron_add')

        # --- Update skip connections affected by the channel change ---
        # h[layer_idx+1] is the output of layer[layer_idx], now has C_new channels.
        # conn.source == layer_idx+1 → source channels changed
        # conn.target == layer_idx+2 → target channels changed (adds to h[layer_idx+1])
        d_new_flat = C_new * H * W  # For cross-type (spatial→flat) connections
        for conn in model.connections:
            if conn.source == layer_idx + 1:
                dim = C_new if self._is_spatial_connection(conn) else d_new_flat
                self._update_connection_source_dim(model, conn, dim, optimizer)
            if conn.target == layer_idx + 2:
                dim = C_new if self._is_spatial_connection(conn) else d_new_flat
                self._update_connection_target_dim(model, conn, dim, optimizer)

        # --- Update spatial ops dimensions ---
        self._update_spatial_ops_dimensions(model.ops[layer_idx], C_new, H, W, device)

        # Rebuild spinal projections after channel resize
        if getattr(model, '_spinal_enabled', False):
            model._rebuild_spinal()

        if self.logger:
            self.logger.log_surgery(step, "add_channel", {
                "layer": layer_idx,
                "split_channel": j,
                "new_channels": C_new,
            })

        return True

    def remove_channel(
        self,
        model: ASANNModel,
        layer_idx: int,
        channel_idx: int,
        optimizer,
        step: int,
    ) -> bool:
        """Remove a channel from a spatial layer.

        Analogous to remove_neuron for flat layers. Deletes filter and
        corresponding input channel in the next layer.
        """
        layer = model.layers[layer_idx]
        assert hasattr(layer, 'mode') and layer.mode == "spatial"

        C_old = layer.out_channels
        # Use spatial-specific minimum (higher than flat d_min)
        min_channels = max(self.config.d_min,
                           getattr(self.config, 'min_spatial_channels', 8))
        if C_old <= min_channels:
            return False

        C_new = C_old - 1
        device = layer.conv.weight.device
        _, H, W = layer.spatial_shape
        mask = torch.ones(C_old, dtype=torch.bool, device=device)
        mask[channel_idx] = False

        # --- Shrink conv output filters ---
        old_conv_weight = layer.conv.weight
        new_conv_weight = nn.Parameter(layer.conv.weight.data[mask])
        layer.conv.weight = new_conv_weight
        layer.conv.out_channels = C_new
        optimizer.register_neuron_surgery(old_conv_weight, new_conv_weight,
                                          layer_idx, 'neuron_remove')

        # --- Shrink BatchNorm ---
        old_bn = layer.bn
        old_bn_weight = old_bn.weight
        old_bn_bias = old_bn.bias
        new_bn = nn.BatchNorm2d(C_new).to(device)
        new_bn.weight.data = old_bn.weight.data[mask]
        new_bn.bias.data = old_bn.bias.data[mask]
        new_bn.running_mean = old_bn.running_mean[mask]
        new_bn.running_var = old_bn.running_var[mask]
        layer.bn = new_bn

        # Register BN params with optimizer (swap old -> new references)
        optimizer.register_neuron_surgery(old_bn_weight, new_bn.weight,
                                          layer_idx, 'neuron_remove')
        optimizer.register_neuron_surgery(old_bn_bias, new_bn.bias,
                                          layer_idx, 'neuron_remove')

        # Update spatial shape
        layer.spatial_shape = (C_new, H, W)

        # --- Update structural residual projection ---
        old_res_proj = layer.residual_proj
        if layer._C_in != C_new:
            new_res_proj = nn.Conv2d(
                layer._C_in, C_new, 1, stride=layer._stride, bias=False
            ).to(device)
            nn.init.dirac_(new_res_proj.weight)
            layer.residual_proj = new_res_proj
            if old_res_proj is not None:
                optimizer.register_neuron_surgery(
                    old_res_proj.weight, new_res_proj.weight,
                    layer_idx, 'neuron_remove')
            else:
                optimizer.register_new_parameters(
                    [new_res_proj.weight],
                    group_name='spatial_residual',
                    source_layer=layer_idx,
                    surgery_type='neuron_remove')
        elif layer._stride == 1:
            layer.residual_proj = None  # Dims match, no projection
            if old_res_proj is not None:
                optimizer.register_removed_parameters([old_res_proj.weight])

        # --- Shrink next layer's input channels ---
        if layer_idx < model.num_layers - 1:
            next_layer = model.layers[layer_idx + 1]
            if hasattr(next_layer, 'mode') and next_layer.mode == "spatial":
                old_next_w = next_layer.conv.weight
                new_next_w = nn.Parameter(next_layer.conv.weight.data[:, mask])
                next_layer.conv.weight = new_next_w
                next_layer.conv.in_channels = C_new
                next_layer._C_in = C_new
                # Update residual projection: create/resize/remove as needed
                old_next_proj = next_layer.residual_proj
                if next_layer._C_in != next_layer.out_channels or next_layer._stride != 1:
                    new_proj = nn.Conv2d(
                        C_new, next_layer.out_channels, 1,
                        stride=next_layer._stride, bias=False
                    ).to(device)
                    nn.init.dirac_(new_proj.weight)
                    next_layer.residual_proj = new_proj
                    if old_next_proj is not None:
                        optimizer.register_neuron_surgery(
                            old_next_proj.weight, new_proj.weight,
                            layer_idx + 1, 'neuron_remove')
                    else:
                        optimizer.register_new_parameters(
                            [new_proj.weight],
                            group_name='spatial_residual',
                            source_layer=layer_idx + 1,
                            surgery_type='neuron_remove')
                else:
                    next_layer.residual_proj = None
                    if old_next_proj is not None:
                        optimizer.register_removed_parameters([old_next_proj.weight])
                optimizer.register_neuron_surgery(old_next_w, new_next_w,
                                                  layer_idx, 'neuron_remove')
            else:
                # Flat next layer or output head
                if layer_idx == model.num_layers - 1:
                    target = model.output_head
                else:
                    target = next_layer.linear if hasattr(next_layer, 'linear') else next_layer

                old_w = target.weight
                # Remove columns corresponding to removed channel
                d_new_in = C_new * H * W
                keep_cols = []
                for c in range(C_old):
                    if c != channel_idx:
                        keep_cols.extend(range(c * H * W, (c + 1) * H * W))
                keep_cols = torch.tensor(keep_cols, device=device)
                new_w = nn.Parameter(target.weight.data[:, keep_cols])
                target.weight = new_w
                target.in_features = d_new_in
                optimizer.register_neuron_surgery(old_w, new_w, layer_idx, 'neuron_remove')
        else:
            # Last layer -> output head
            old_w = model.output_head.weight
            use_gap = getattr(model, '_use_gap', False)
            if use_gap:
                # GAP mode: head is [d_out, C], just remove one column
                mask = torch.ones(C_old, dtype=torch.bool, device=device)
                mask[channel_idx] = False
                new_w = nn.Parameter(model.output_head.weight.data[:, mask])
                model.output_head.weight = new_w
                model.output_head.in_features = C_new
            else:
                # Flatten mode: head is [d_out, C*H*W], remove H*W columns
                d_new_in = C_new * H * W
                keep_cols = []
                for c in range(C_old):
                    if c != channel_idx:
                        keep_cols.extend(range(c * H * W, (c + 1) * H * W))
                keep_cols = torch.tensor(keep_cols, device=device)
                new_w = nn.Parameter(model.output_head.weight.data[:, keep_cols])
                model.output_head.weight = new_w
                model.output_head.in_features = d_new_in
            optimizer.register_neuron_surgery(old_w, new_w, layer_idx, 'neuron_remove')

        # --- Update skip connections affected by the channel change ---
        # h[layer_idx+1] is the output of layer[layer_idx], now has C_new channels.
        # conn.source == layer_idx+1 → source channels changed
        # conn.target == layer_idx+2 → target channels changed (adds to h[layer_idx+1])
        d_new_flat = C_new * H * W  # For cross-type (spatial→flat) connections
        for conn in model.connections:
            if conn.source == layer_idx + 1:
                dim = C_new if self._is_spatial_connection(conn) else d_new_flat
                self._update_connection_source_dim(model, conn, dim, optimizer)
            if conn.target == layer_idx + 2:
                dim = C_new if self._is_spatial_connection(conn) else d_new_flat
                self._update_connection_target_dim(model, conn, dim, optimizer)

        # --- Update spatial ops dimensions ---
        self._update_spatial_ops_dimensions(model.ops[layer_idx], C_new, H, W, device)

        # Rebuild spinal projections after channel resize
        if getattr(model, '_spinal_enabled', False):
            model._rebuild_spinal()

        if self.logger:
            self.logger.log_surgery(step, "remove_channel", {
                "layer": layer_idx,
                "channel": channel_idx,
                "new_channels": C_new,
            })

        return True

    def _update_spatial_ops_dimensions(self, pipeline, C_new, H, W, device):
        """Update spatial operations when channel count changes.

        Handles both Python and CUDA op variants. CUDA variants have the same
        internal structure (dw_conv, bn, pw_conv, gate_logit, etc.) so the
        parameter copy logic is shared — only the class used for construction differs.
        """
        def _set_op(idx, new_op, gate_wrap):
            """Replace op in pipeline, re-wrapping in GatedOperation if needed."""
            if gate_wrap is not None:
                gate_wrap.operation = new_op
                pipeline.operations[idx] = gate_wrap
            else:
                pipeline.operations[idx] = new_op
        # Build CUDA type checks
        _cuda = _CUDA_OPS_AVAILABLE
        _SpatialConv2dCUDA = SpatialConv2dOpCUDA if _cuda else None
        _PointwiseCUDA = PointwiseConv2dOpCUDA if _cuda else None
        _DWSepCUDA = DepthwiseSeparableConv2dOpCUDA if _cuda else None
        _ChanAttnCUDA = ChannelAttentionOpCUDA if _cuda else None
        _CapsuleConv2dCUDA = CapsuleConv2dOpCUDA if _cuda else None

        from .model import GatedOperation

        for i, raw_op in enumerate(list(pipeline.operations)):
            # Unwrap GatedOperation to check the inner op type
            gate_wrapper = None
            op = raw_op
            if isinstance(raw_op, GatedOperation):
                gate_wrapper = raw_op
                op = raw_op.operation

            # --- SpatialConv2d ---
            if isinstance(op, SpatialConv2dOp) or (_cuda and isinstance(op, _SpatialConv2dCUDA)):
                use_cuda = _cuda and isinstance(op, _SpatialConv2dCUDA)
                cls = SpatialConv2dOpCUDA if use_cuda else SpatialConv2dOp
                new_op = cls(C_new, H, W, kernel_size=op.kernel_size).to(device)
                copy_c = min(op.C, C_new)
                new_op.conv.weight.data[:copy_c, :min(1, 1)] = op.conv.weight.data[:copy_c, :min(1, 1)]
                new_op.conv.bias.data[:copy_c] = op.conv.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- PointwiseConv2d ---
            elif isinstance(op, PointwiseConv2dOp) or (_cuda and isinstance(op, _PointwiseCUDA)):
                use_cuda = _cuda and isinstance(op, _PointwiseCUDA)
                cls = PointwiseConv2dOpCUDA if use_cuda else PointwiseConv2dOp
                new_op = cls(C_new, H, W).to(device)
                copy_c = min(op.C, C_new)
                new_op.conv.weight.data[:copy_c, :copy_c] = op.conv.weight.data[:copy_c, :copy_c]
                new_op.conv.bias.data[:copy_c] = op.conv.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- DepthwiseSeparableConv2d ---
            elif isinstance(op, DepthwiseSeparableConv2dOp) or (_cuda and isinstance(op, _DWSepCUDA)):
                use_cuda = _cuda and isinstance(op, _DWSepCUDA)
                cls = DepthwiseSeparableConv2dOpCUDA if use_cuda else DepthwiseSeparableConv2dOp
                new_op = cls(C_new, H, W, kernel_size=op.kernel_size).to(device)
                copy_c = min(op.C, C_new)
                new_op.dw_conv.weight.data[:copy_c] = op.dw_conv.weight.data[:copy_c]
                new_op.bn.weight.data[:copy_c] = op.bn.weight.data[:copy_c]
                new_op.bn.bias.data[:copy_c] = op.bn.bias.data[:copy_c]
                new_op.bn.running_mean[:copy_c] = op.bn.running_mean[:copy_c]
                new_op.bn.running_var[:copy_c] = op.bn.running_var[:copy_c]
                new_op.pw_conv.weight.data[:copy_c, :copy_c] = op.pw_conv.weight.data[:copy_c, :copy_c]
                new_op.pw_conv.bias.data[:copy_c] = op.pw_conv.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- ChannelAttention ---
            elif isinstance(op, ChannelAttentionOp) or (_cuda and isinstance(op, _ChanAttnCUDA)):
                use_cuda = _cuda and isinstance(op, _ChanAttnCUDA)
                cls = ChannelAttentionOpCUDA if use_cuda else ChannelAttentionOp
                new_op = cls(C_new, reduction_ratio=op.reduction_ratio).to(device)
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- CapsuleConv2d ---
            elif isinstance(op, CapsuleConv2dOp) or (_cuda and isinstance(op, _CapsuleConv2dCUDA)):
                use_cuda = _cuda and isinstance(op, _CapsuleConv2dCUDA)
                cls = CapsuleConv2dOpCUDA if use_cuda else CapsuleConv2dOp
                new_op = cls(C_new, H, W, kernel_size=op.kernel_size,
                             cap_dim=op.cap_dim).to(device)
                copy_c = min(op.C, C_new)
                new_op.dw_conv.weight.data[:copy_c] = op.dw_conv.weight.data[:copy_c]
                new_op.pw_conv.weight.data[:copy_c, :copy_c] = \
                    op.pw_conv.weight.data[:copy_c, :copy_c]
                new_op.pw_conv.bias.data[:copy_c] = op.pw_conv.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- MultiScaleConv2dOp ---
            elif isinstance(op, MultiScaleConv2dOp):
                new_op = MultiScaleConv2dOp(C_new, H, W).to(device)
                copy_c = min(op.C, C_new)
                try:
                    # dw convs: [C, 1, k, k] → copy first copy_c filters
                    new_op.dw_1x1.weight.data[:copy_c] = op.dw_1x1.weight.data[:copy_c]
                    new_op.dw_3x3.weight.data[:copy_c] = op.dw_3x3.weight.data[:copy_c]
                    new_op.dw_5x5.weight.data[:copy_c] = op.dw_5x5.weight.data[:copy_c]
                    # pw_proj: [C_new, 3*C_new, 1, 1] ← copy overlapping region
                    copy_3c = min(3 * op.C, 3 * C_new)
                    new_op.pw_proj.weight.data[:copy_c, :copy_3c] = \
                        op.pw_proj.weight.data[:copy_c, :copy_3c]
                    new_op.pw_proj.bias.data[:copy_c] = op.pw_proj.bias.data[:copy_c]
                except (RuntimeError, IndexError):
                    pass  # Fall back to fresh near-identity init from constructor
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- GroupNorm ---
            elif isinstance(op, nn.GroupNorm):
                new_op = nn.GroupNorm(_safe_num_groups(C_new), C_new).to(device)
                copy_c = min(op.num_channels, C_new)
                new_op.weight.data[:copy_c] = op.weight.data[:copy_c]
                new_op.bias.data[:copy_c] = op.bias.data[:copy_c]
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- BatchNorm2d ---
            elif isinstance(op, nn.BatchNorm2d):
                new_bn = nn.BatchNorm2d(C_new).to(device)
                copy_c = min(op.num_features, C_new)
                new_bn.weight.data[:copy_c] = op.weight.data[:copy_c]
                new_bn.bias.data[:copy_c] = op.bias.data[:copy_c]
                new_bn.running_mean[:copy_c] = op.running_mean[:copy_c]
                new_bn.running_var[:copy_c] = op.running_var[:copy_c]
                if hasattr(op, '_asann_added_step'):
                    new_bn._asann_added_step = op._asann_added_step
                _set_op(i, new_bn, gate_wrapper)

            # --- PReLU (per-channel) ---
            elif isinstance(op, nn.PReLU) and op.num_parameters > 1:
                new_op = nn.PReLU(num_parameters=C_new).to(device)
                copy_c = min(op.num_parameters, C_new)
                new_op.weight.data[:copy_c] = op.weight.data[:copy_c]
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- DerivativeConv2d ---
            elif isinstance(op, DerivativeConv2d):
                new_op = DerivativeConv2d(C_new, H, W, order=op.order, axis=op.axis).to(device)
                copy_c = min(op.C, C_new)
                new_op.conv.weight.data[:copy_c] = op.conv.weight.data[:copy_c]
                new_op.conv.bias.data[:copy_c] = op.conv.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- SpatialPolynomialOp ---
            elif isinstance(op, SpatialPolynomialOp):
                new_op = SpatialPolynomialOp(C_new, H, W, degree=op.degree).to(device)
                copy_c = min(op.C, C_new)
                copy_deg_c = min(op.C * op.degree, C_new * op.degree)
                new_op.mix.weight.data[:copy_c, :copy_deg_c] = \
                    op.mix.weight.data[:copy_c, :copy_deg_c]
                new_op.mix.bias.data[:copy_c] = op.mix.bias.data[:copy_c]
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

            # --- SpatialBranchedOperationBlock ---
            elif isinstance(op, SpatialBranchedOperationBlock):
                new_branches = []
                for branch in op.branches:
                    if isinstance(branch, DerivativeConv2d):
                        new_b = DerivativeConv2d(C_new, H, W,
                                                 order=branch.order, axis=branch.axis).to(device)
                        copy_c_b = min(branch.C, C_new)
                        new_b.conv.weight.data[:copy_c_b] = branch.conv.weight.data[:copy_c_b]
                        new_b.conv.bias.data[:copy_c_b] = branch.conv.bias.data[:copy_c_b]
                        new_b.gate_logit.data = branch.gate_logit.data.clone()
                    elif isinstance(branch, SpatialPolynomialOp):
                        new_b = SpatialPolynomialOp(C_new, H, W,
                                                    degree=branch.degree).to(device)
                        copy_c_b = min(branch.C, C_new)
                        copy_deg_c_b = min(branch.C * branch.degree,
                                           C_new * branch.degree)
                        new_b.mix.weight.data[:copy_c_b, :copy_deg_c_b] = \
                            branch.mix.weight.data[:copy_c_b, :copy_deg_c_b]
                        new_b.mix.bias.data[:copy_c_b] = branch.mix.bias.data[:copy_c_b]
                        new_b.gate_logit.data = branch.gate_logit.data.clone()
                    else:
                        new_b = type(branch)(C_new, H, W).to(device)
                    new_branches.append(new_b)
                new_op = SpatialBranchedOperationBlock(C_new, H, W, new_branches).to(device)
                new_op.branch_logits.data.copy_(op.branch_logits.data)
                new_op.gate_logit.data = op.gate_logit.data.clone()
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                _set_op(i, new_op, gate_wrapper)

    # ==================== OPERATION SURGERY (Section 3.3) ====================

    def probe_operations(
        self,
        model: ASANNModel,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn,
        current_loss: float,
        current_step: int = 0,
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        """Run operation benefit probes for all layers.

        For each candidate operation not already in a layer's pipeline,
        and for each possible position, compute:
            Benefit(l, o, pos) = L_current - L_probe(l, o, pos)

        An operation is recommended if its benefit exceeds EITHER:
          - The absolute threshold (benefit_threshold), OR
          - 5% of the current loss (relative threshold)
        This ensures operations can be discovered even when loss is very low.

        This costs |O_candidates| x |positions| forward passes per layer per surgery.
        NOT per training step. This is far cheaper than DARTS.

        Also probes removal of existing operations.

        RC5b: Dropout operations are evaluated on val_batch (if available) since
        dropout's benefit is on unseen data, not training data.

        RC6: If validation_gated_surgery is enabled and val_batch is provided,
        operation additions are gated by validation loss improvement.

        RC7: max_ops_per_layer caps how many ops a layer can accumulate.

        Returns a list of recommended add/remove actions sorted by benefit.
        """
        model.eval()

        # Ensure FP32 precision for probing — AMP autocast must not be active.
        # Probing measures tiny loss differences (e.g., 0.001) that FP16 would corrupt.
        _amp_guard = torch.amp.autocast('cuda', enabled=False)
        _amp_guard.__enter__()

        recommendations = []
        # Use both absolute and relative thresholds so operations can be
        # discovered even when loss is already very low.
        # The effective threshold is the LOWER of:
        #   - The absolute threshold (benefit_threshold from config)
        #   - 10% of current loss (relative, with floor to avoid near-zero threshold)
        abs_threshold = self.config.benefit_threshold
        rel_threshold = max(0.10 * current_loss, 1e-6)
        effective_threshold = min(abs_threshold, rel_threshold)

        # RC7: max ops per layer
        max_ops = self.config.max_ops_per_layer

        # Candidate pool is now per-layer (spatial vs flat vs graph)

        # Build graph_data kwargs once (empty dict for non-graph models)
        graph_data_kwargs = {}
        if getattr(model, '_is_graph', False):
            graph_data_kwargs['graph_data'] = {
                'adj_sparse': model._graph_adj_sparse,
                'edge_index': model._graph_edge_index,
                'degree': model._graph_degree,
                'num_nodes': model._graph_num_nodes,
            }

        with torch.no_grad():
            # RC6: Pre-compute base validation loss for gating
            base_val_loss = None
            if val_batch is not None and self.config.validation_gated_surgery:
                val_x, val_y = val_batch
                try:
                    val_output = model(val_x)
                    base_val_loss = loss_fn(val_output, val_y).item()
                except Exception:
                    pass

            for l in range(model.num_layers):
                layer = model.layers[l]
                d = layer.out_features
                pipeline = model.ops[l]

                # Per-layer candidate pool (spatial ops for spatial layers,
                # graph ops for graph models, flat ops for flat)
                candidates = get_candidate_operations_for_layer(self.config, layer, model=model)

                # Spatial shape for creating spatial ops
                layer_spatial_shape = getattr(layer, 'spatial_shape', None)
                is_spatial_layer = hasattr(layer, 'mode') and layer.mode == "spatial"

                # Determine which operations are already in this pipeline
                existing_ops = set()
                has_dropout = False
                has_activation = False
                for op in pipeline.operations:
                    name = get_operation_name(op)
                    existing_ops.add(name)
                    if name.startswith("dropout"):
                        has_dropout = True
                    if name in ("relu", "gelu", "swish"):
                        has_activation = True

                # --- Probe additions ---
                # RC7: Skip additions if layer already has max ops
                skip_additions = (max_ops > 0 and pipeline.num_operations >= max_ops)

                if not skip_additions:
                    for op_name in candidates:
                        if op_name in existing_ops:
                            continue

                        # Prevent stacking multiple dropout or activation variants
                        is_dropout = op_name.startswith("dropout")
                        is_activation = op_name in ("relu", "gelu", "swish")
                        if is_dropout and has_dropout:
                            continue  # Only one dropout per layer
                        if is_activation and has_activation:
                            continue  # Only one activation per layer
                        if is_dropout and val_batch is not None:
                            eval_x, eval_y = val_batch
                            # For dropout, use val data as base loss too
                            eval_current_loss = base_val_loss if base_val_loss is not None else current_loss
                        else:
                            eval_x, eval_y = x_batch, y_batch
                            eval_current_loss = current_loss

                        # Determine possible positions
                        num_positions = pipeline.num_operations + 1
                        for pos in range(num_positions):
                            # Temporarily insert the operation
                            try:
                                trial_op = create_operation(
                                    op_name, d, device=self.config.device,
                                    config=self.config,
                                    spatial_shape=layer_spatial_shape,
                                    **graph_data_kwargs,
                                )
                            except Exception:
                                continue

                            # Save original ops
                            original_ops = model.ops[l]

                            # Create trial pipeline (spatial or flat)
                            if is_spatial_layer:
                                trial_pipeline = SpatialOperationPipeline().to(self.config.device)
                            else:
                                trial_pipeline = OperationPipeline().to(self.config.device)
                            for i, existing_op in enumerate(original_ops.operations):
                                if i == pos:
                                    trial_pipeline.operations.append(trial_op)
                                trial_pipeline.operations.append(existing_op)
                            if pos == original_ops.num_operations:
                                trial_pipeline.operations.append(trial_op)

                            model.ops[l] = trial_pipeline

                            try:
                                # For dropout, set model to train mode briefly for dropout to take effect
                                if is_dropout:
                                    model.train()
                                output = model(eval_x)
                                probe_loss = loss_fn(output, eval_y).item()
                                benefit = eval_current_loss - probe_loss
                                if is_dropout:
                                    model.eval()

                                if benefit > effective_threshold:
                                    # RC6: Validation gate — check if op hurts val loss
                                    gated_out = False
                                    if (val_batch is not None
                                            and self.config.validation_gated_surgery
                                            and base_val_loss is not None
                                            and not is_dropout):  # dropout already uses val
                                        val_x, val_y = val_batch
                                        try:
                                            val_output = model(val_x)
                                            probe_val_loss = loss_fn(val_output, val_y).item()
                                            tolerance = self.config.validation_gate_tolerance
                                            if probe_val_loss - base_val_loss > tolerance * base_val_loss:
                                                gated_out = True
                                        except Exception:
                                            pass

                                    if not gated_out:
                                        recommendations.append({
                                            "type": "add_op",
                                            "layer": l,
                                            "operation": op_name,
                                            "position": pos,
                                            "benefit": benefit,
                                        })
                            except Exception:
                                if is_dropout:
                                    model.eval()

                            # Restore original pipeline
                            model.ops[l] = original_ops

                # --- Probe removals ---
                # Parametric ops (conv, batchnorm, layernorm) get a protection
                # period of 3 surgery intervals after being added, giving their
                # parameters time to train before being evaluated for removal.
                #
                # For graph (semi-supervised) models, removal probes use validation
                # data instead of training data.  With only ~120 labeled nodes the
                # model memorises the training set quickly; a training-loss probe
                # then favours removing graph-aggregation ops (they add "noise" to
                # already-memorised labels) even though those ops are essential for
                # generalisation to unseen nodes.
                #
                # Asymmetric safeguards (Fix 3):
                #   - Removal threshold is op_removal_threshold_mult x addition threshold
                #   - Last non-trivial op in a layer is protected from removal
                is_graph_model = getattr(model, '_is_graph', False)
                if is_graph_model and val_batch is not None and base_val_loss is not None:
                    remove_eval_x, remove_eval_y = val_batch
                    remove_eval_base_loss = base_val_loss
                else:
                    remove_eval_x, remove_eval_y = x_batch, y_batch
                    remove_eval_base_loss = current_loss

                # Asymmetric removal threshold: harder to remove than to add
                removal_threshold = effective_threshold * self.config.op_removal_threshold_mult

                # Count non-trivial ops for last-nontrivial protection
                from .model import GatedOperation as _GatedOp
                nontrivial_count = sum(
                    1 for _op in pipeline.operations
                    if type(_op.operation if isinstance(_op, _GatedOp) else _op).__name__
                    not in _TRIVIAL_OPS
                )

                protection_steps = 3 * self.config.surgery_interval_init
                for pos in range(pipeline.num_operations):
                    op = pipeline.operations[pos]
                    op_name = get_operation_name(op)

                    # Skip recently-added parametric ops
                    added_step = getattr(op, '_asann_added_step', 0)
                    if added_step > 0 and (current_step - added_step) < protection_steps:
                        continue

                    # Protect the last non-trivial op in this layer
                    if self.config.op_protect_last_nontrivial and nontrivial_count <= 1:
                        unwrapped = op.operation if isinstance(op, _GatedOp) else op
                        if type(unwrapped).__name__ not in _TRIVIAL_OPS:
                            continue  # This is the last real op -- skip removal probe

                    # Save original
                    original_ops = model.ops[l]

                    # Create pipeline without this operation (spatial or flat)
                    if is_spatial_layer:
                        trial_pipeline = SpatialOperationPipeline().to(self.config.device)
                    else:
                        trial_pipeline = OperationPipeline().to(self.config.device)
                    for i, existing_op in enumerate(original_ops.operations):
                        if i != pos:
                            trial_pipeline.operations.append(existing_op)

                    model.ops[l] = trial_pipeline

                    try:
                        output = model(remove_eval_x)
                        probe_loss = loss_fn(output, remove_eval_y).item()
                        remove_cost = probe_loss - remove_eval_base_loss

                        if remove_cost < -removal_threshold:
                            # Removing this operation HELPS beyond the asymmetric threshold
                            recommendations.append({
                                "type": "remove_op",
                                "layer": l,
                                "operation": op_name,
                                "position": pos,
                                "benefit": -remove_cost,
                            })
                    except Exception:
                        pass

                    model.ops[l] = original_ops

                # --- Probe replacements ---
                # Try replacing each existing non-protected op with each candidate.
                # This discovers if a different op would work better than what's
                # currently in the pipeline. Only probed for ops past their
                # protection period (same as removal probing).
                _PROTECTED_FROM_REPLACE = {
                    'relu', 'gelu', 'swish', 'leaky_relu', 'elu', 'mish',
                    'tanh', 'softplus', 'prelu', 'spatial_prelu',
                    'batchnorm', 'batchnorm2d', 'groupnorm', 'layernorm',
                    'dropout_01', 'dropout_03',
                }
                is_graph_model = getattr(model, '_is_graph', False)

                for pos in range(pipeline.num_operations):
                    op = pipeline.operations[pos]
                    old_op_name = get_operation_name(op)

                    # Skip protected ops (activations, normalization, dropout)
                    if old_op_name in _PROTECTED_FROM_REPLACE:
                        continue
                    if is_graph_model and old_op_name.startswith("graph_"):
                        continue
                    if old_op_name.startswith("dropout"):
                        continue

                    # Skip recently-added ops (same protection period as removals)
                    added_step = getattr(op, '_asann_added_step', 0)
                    if added_step > 0 and (current_step - added_step) < protection_steps:
                        continue

                    # Build set of other ops in pipeline (excluding the one at pos)
                    other_ops_in_pipeline = set()
                    for i, other_op in enumerate(pipeline.operations):
                        if i != pos:
                            other_ops_in_pipeline.add(get_operation_name(other_op))

                    for candidate_name in candidates:
                        # Skip if same as existing
                        if candidate_name == old_op_name:
                            continue
                        # Skip if candidate already in pipeline at another position
                        if candidate_name in other_ops_in_pipeline:
                            continue
                        # Skip activations/dropout candidates (handled separately)
                        if candidate_name in _PROTECTED_FROM_REPLACE:
                            continue
                        if candidate_name.startswith("dropout"):
                            continue

                        try:
                            trial_op = create_operation(
                                candidate_name, d, device=self.config.device,
                                config=self.config,
                                spatial_shape=layer_spatial_shape,
                                **graph_data_kwargs,
                            )
                        except Exception:
                            continue

                        # Build trial pipeline with replacement
                        original_ops = model.ops[l]
                        if is_spatial_layer:
                            trial_pipeline = SpatialOperationPipeline().to(self.config.device)
                        else:
                            trial_pipeline = OperationPipeline().to(self.config.device)
                        for i, existing_op in enumerate(original_ops.operations):
                            if i == pos:
                                trial_pipeline.operations.append(trial_op)
                            else:
                                trial_pipeline.operations.append(existing_op)

                        model.ops[l] = trial_pipeline

                        try:
                            output = model(x_batch)
                            probe_loss = loss_fn(output, y_batch).item()
                            benefit = current_loss - probe_loss

                            if benefit > effective_threshold:
                                # Validate on val_batch if available
                                gated_out = False
                                if (val_batch is not None
                                        and self.config.validation_gated_surgery
                                        and base_val_loss is not None):
                                    val_x, val_y = val_batch
                                    try:
                                        val_output = model(val_x)
                                        probe_val_loss = loss_fn(val_output, val_y).item()
                                        tolerance = self.config.validation_gate_tolerance
                                        if probe_val_loss - base_val_loss > tolerance * base_val_loss:
                                            gated_out = True
                                    except Exception:
                                        pass

                                if not gated_out:
                                    recommendations.append({
                                        "type": "replace_op",
                                        "layer": l,
                                        "operation": candidate_name,
                                        "old_operation": old_op_name,
                                        "position": pos,
                                        "benefit": benefit,
                                    })
                        except Exception:
                            pass

                        model.ops[l] = original_ops

        model.train()
        _amp_guard.__exit__(None, None, None)

        # Sort by benefit (highest first)
        recommendations.sort(key=lambda r: r["benefit"], reverse=True)
        return recommendations

    def execute_operation_surgery(
        self,
        model: ASANNModel,
        recommendations: List[Dict[str, Any]],
        optimizer,
        step: int,
        skip_exploration: bool = False,
    ):
        """Execute the top operation surgery recommendations within the budget.

        Optimizer is updated in-place via register_structural_surgery().
        If skip_exploration=True, the stochastic exploration step is suppressed
        (used when the model is architecturally stable).
        """
        ops_changed = 0
        max_changes = self.config.max_ops_change_per_surgery

        # Build graph_data kwargs once (empty dict for non-graph models)
        graph_data_kwargs = {}
        if getattr(model, '_is_graph', False):
            graph_data_kwargs['graph_data'] = {
                'adj_sparse': model._graph_adj_sparse,
                'edge_index': model._graph_edge_index,
                'degree': model._graph_degree,
                'num_nodes': model._graph_num_nodes,
            }

        # Track which layers have been modified to avoid conflicting changes
        modified_layers = set()

        # All parametric op types that need a protection period before
        # removal probing (their parameters need training time)
        _PARAMETRIC_OP_TYPES = (
            Conv1dBlock, Conv2dBlock, FactoredEmbedding, MLPEmbedding,
            GeometricEmbedding, PositionalEmbedding,
            SelfAttentionOp, MultiHeadAttentionOp,
            CrossAttentionOp, CausalAttentionOp,
            SpatialConv2dOp, PointwiseConv2dOp, ChannelAttentionOp,
            DepthwiseSeparableConv2dOp, CapsuleConv2dOp,
            MultiScaleConv2dOp,
            MaxPool2dOp, AvgPool2dOp, MinPool2dOp, MixedPool2dOp,
            NeighborAggregation, GraphAttentionAggregation,
            GraphDiffusion, SpectralConv, MessagePassingGIN,
            DegreeScaling, GraphBranchedBlock, GraphPositionalEncoding,
            DirectionalDiffusion, AdaptiveGraphConv,
            GRUOp,
            KANLinearOp,
        )

        for rec in recommendations:
            if ops_changed >= max_changes:
                break

            l = rec["layer"]
            if l in modified_layers:
                continue

            if rec["type"] == "add_op":
                layer = model.layers[l]
                d = layer.out_features
                layer_spatial_shape = getattr(layer, 'spatial_shape', None)
                try:
                    new_op = create_operation(
                        rec["operation"], d, device=self.config.device,
                        config=self.config, spatial_shape=layer_spatial_shape,
                        **graph_data_kwargs,
                    )
                    if hasattr(new_op, 'weight') or isinstance(new_op, _PARAMETRIC_OP_TYPES):
                        new_op._asann_added_step = step
                    warmup = self.config.surgery_warmup_epochs
                    model.ops[l].add_operation(new_op, rec["position"],
                                               gated=True, warmup_epochs=warmup)
                    modified_layers.add(l)
                    ops_changed += 1

                    if self.logger:
                        self.logger.log_surgery(step, "add_operation", {
                            "layer": l,
                            "operation": rec["operation"],
                            "position": rec["position"],
                            "benefit": rec["benefit"],
                            "pipeline": model.ops[l].describe(),
                        })
                except Exception:
                    continue

            elif rec["type"] == "remove_op":
                if model.ops[l].num_operations > 0:
                    removed_name = rec["operation"]

                    # Consecutive-round gate: op must be flagged for removal
                    # in op_removal_consecutive successive rounds before
                    # actually being removed.  This filters single-batch noise.
                    key = (l, removed_name)
                    streak = self._op_removal_history.get(key, 0) + 1
                    self._op_removal_history[key] = streak

                    if streak < self.config.op_removal_consecutive:
                        if self.logger:
                            self.logger.log_surgery(step, "remove_op_deferred", {
                                "layer": l,
                                "operation": removed_name,
                                "streak": streak,
                                "required": self.config.op_removal_consecutive,
                                "benefit": rec["benefit"],
                            })
                        continue  # Not enough consecutive rounds yet

                    # Passed consecutive gate -- execute removal
                    model.ops[l].remove_operation(rec["position"])
                    modified_layers.add(l)
                    ops_changed += 1
                    # Clear history for this op after successful removal
                    self._op_removal_history.pop(key, None)

                    if self.logger:
                        self.logger.log_surgery(step, "remove_operation", {
                            "layer": l,
                            "operation": removed_name,
                            "position": rec["position"],
                            "benefit": rec["benefit"],
                            "pipeline": model.ops[l].describe(),
                        })

            elif rec["type"] == "replace_op":
                pipeline = model.ops[l]
                pos = rec["position"]
                if pos < pipeline.num_operations:
                    try:
                        layer = model.layers[l]
                        d = layer.out_features
                        layer_spatial_shape = getattr(layer, 'spatial_shape', None)
                        new_op = create_operation(
                            rec["operation"], d, device=self.config.device,
                            config=self.config, spatial_shape=layer_spatial_shape,
                            **graph_data_kwargs,
                        )
                        # Tag parametric ops with step for protection period
                        if hasattr(new_op, 'weight') or isinstance(new_op, _PARAMETRIC_OP_TYPES):
                            new_op._asann_added_step = step
                        # Remove old, insert new at same position with gating
                        pipeline.remove_operation(pos)
                        warmup = self.config.surgery_warmup_epochs
                        pipeline.add_operation(new_op, pos, gated=True, warmup_epochs=warmup)
                        modified_layers.add(l)
                        ops_changed += 1

                        if self.logger:
                            self.logger.log_surgery(step, "replace_operation", {
                                "layer": l,
                                "old_operation": rec.get("old_operation", "unknown"),
                                "new_operation": rec["operation"],
                                "position": pos,
                                "benefit": rec["benefit"],
                                "pipeline": model.ops[l].describe(),
                            })
                    except Exception:
                        continue

        # Clear removal history for ops NOT flagged in this round
        # (resets non-consecutive streaks so ops must be flagged in
        # consecutive rounds, not intermittently).
        current_removal_keys = {
            (r["layer"], r["operation"])
            for r in recommendations if r["type"] == "remove_op"
        }
        stale_keys = [k for k in self._op_removal_history if k not in current_removal_keys]
        for k in stale_keys:
            del self._op_removal_history[k]

        # --- Stochastic op exploration (mutation + natural selection) ---
        # With probability op_exploration_prob, randomly insert an untried op
        # into a layer. The op gets a protection period to train, then removal
        # probes decide whether to keep or discard it. This solves the
        # chicken-and-egg problem for complex parametric ops
        # that can't show benefit on a single cold forward pass.
        # Exploration is independent of the regular ops budget -- it only picks
        # layers not already modified this round, so there's no conflict.
        if (self.config.op_exploration_enabled
                and not skip_exploration
                and random.random() < self.config.op_exploration_prob):
            if self._explore_random_op(model, optimizer, step,
                                       modified_layers, graph_data_kwargs,
                                       _PARAMETRIC_OP_TYPES):
                ops_changed += 1  # Count for optimizer sync

        # Sync optimizer in-place if ops with parameters were added/removed
        if ops_changed > 0:
            optimizer.register_structural_surgery(model, surgery_type='op_change')

    def _explore_random_op(
        self,
        model: ASANNModel,
        optimizer,
        step: int,
        modified_layers: set,
        graph_data_kwargs: dict,
        parametric_types: tuple,
    ) -> bool:
        """Randomly insert an untried op into a layer for exploration.

        The op gets the standard protection period (3 surgery intervals) to
        train. After protection expires, removal probes evaluate whether to
        keep or discard it. This is biological mutation + natural selection.

        Returns True if an op was inserted, False otherwise.
        """
        # Collect eligible (layer, op_name) pairs: untried ops in layers
        # that have room. Unlike regular probing, exploration CAN modify
        # layers already touched this round -- the regular "one op per
        # layer" constraint prevents conflicting greedy picks, but
        # exploration is a single random insertion that's safe to add.
        candidates = []
        max_ops = self.config.max_ops_per_layer

        for l in range(model.num_layers):
            layer = model.layers[l]
            pipeline = model.ops[l]

            # Check capacity
            if max_ops > 0 and pipeline.num_operations >= max_ops:
                continue

            # Get candidate ops for this layer type
            layer_candidates = get_candidate_operations_for_layer(
                self.config, layer, model=model
            )

            # Find which ops are already in this pipeline
            existing_ops = set()
            has_dropout = False
            has_activation = False
            for op in pipeline.operations:
                name = get_operation_name(op)
                existing_ops.add(name)
                if name.startswith("dropout"):
                    has_dropout = True
                if name in ("relu", "gelu", "swish"):
                    has_activation = True

            # Filter to untried ops (excluding constraint violations)
            for op_name in layer_candidates:
                if op_name in existing_ops:
                    continue
                if op_name.startswith("dropout") and has_dropout:
                    continue
                if op_name in ("relu", "gelu", "swish") and has_activation:
                    continue
                candidates.append((l, op_name))

        if not candidates:
            return False

        # Physics-biased exploration: when physics_ops_enabled, 50% of
        # exploration attempts focus exclusively on physics ops. This
        # overcomes the dilution problem (6 physics ops among ~32 total).
        if getattr(self.config, 'physics_ops_enabled', False) and random.random() < 0.5:
            physics_candidates = [
                (l, op_name) for l, op_name in candidates
                if op_name in _PHYSICS_OPS
            ]
            if physics_candidates:
                candidates = physics_candidates

        # Pick a random (layer, op) pair
        l, op_name = random.choice(candidates)
        layer = model.layers[l]
        d = layer.out_features
        layer_spatial_shape = getattr(layer, 'spatial_shape', None)

        try:
            new_op = create_operation(
                op_name, d, device=self.config.device,
                config=self.config, spatial_shape=layer_spatial_shape,
                **graph_data_kwargs,
            )
        except Exception:
            return False

        # Mark parametric ops with step for protection period
        if hasattr(new_op, 'weight') or isinstance(new_op, parametric_types):
            new_op._asann_added_step = step

        # Insert at a random position in the pipeline
        num_positions = model.ops[l].num_operations + 1
        pos = random.randint(0, num_positions - 1)

        warmup = self.config.surgery_warmup_epochs
        model.ops[l].add_operation(new_op, pos, gated=True,
                                    warmup_epochs=warmup)
        modified_layers.add(l)

        print(f"  [EXPLORE] Randomly inserted '{op_name}' at layer {l} "
              f"pos {pos} (will be evaluated after protection period)")

        if self.logger:
            self.logger.log_surgery(step, "explore_operation", {
                "layer": l,
                "operation": op_name,
                "position": pos,
                "benefit": 0.0,  # No probing benefit -- exploration
                "pipeline": model.ops[l].describe(),
            })
        return True

    # ==================== CONNECTION SURGERY (Section 3.4) ====================

    def create_connection(
        self,
        model: ASANNModel,
        source_idx: int,
        target_idx: int,
        optimizer,
        step: int,
    ):
        """Create a real skip connection from layer source_idx to layer target_idx.

        This creates an actual tensor operation with a real projection matrix (if needed).
        When dimensions differ, a projection Linear(d_source, d_target) is created.
        The connection starts at near-zero scale to avoid disrupting training.
        Optimizer is updated in-place via register_structural_surgery().
        """
        # Check if connection already exists
        for conn in model.connections:
            if conn.source == source_idx and conn.target == target_idx:
                return

        # Get dimensions and spatial info
        if source_idx == 0:
            d_source = model.input_projection.out_features
            src_spatial = getattr(model.input_projection, 'spatial_shape', None)
        else:
            src_layer = model.layers[source_idx - 1]
            d_source = src_layer.out_features
            src_spatial = getattr(src_layer, 'spatial_shape', None)

        # Connection adds to h[target_idx - 1], which is the OUTPUT of layer target_idx - 2
        # (or stem output if target_idx - 1 == 0).
        tgt_h_idx = target_idx - 1
        if tgt_h_idx == 0:
            d_target = model.input_projection.out_features
            tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
        else:
            tgt_h_layer = model.layers[tgt_h_idx - 1]
            d_target = tgt_h_layer.out_features
            tgt_spatial = getattr(tgt_h_layer, 'spatial_shape', None)

        # Create the connection (spatial-aware)
        conn = SkipConnection(
            source_idx=source_idx,
            target_idx=target_idx,
            d_source=d_source,
            d_target=d_target,
            init_scale=self.config.connection_init_scale,
            device=self.config.device,
            spatial_source_shape=src_spatial,
            spatial_target_shape=tgt_spatial,
        )
        conn._asann_created_step = step
        model.connections.append(conn)

        # Sync optimizer in-place to include new connection parameters
        optimizer.register_structural_surgery(model, surgery_type='conn_add')

        if self.logger:
            self.logger.log_surgery(step, "create_connection", {
                "source": source_idx,
                "target": target_idx,
                "d_source": d_source,
                "d_target": d_target,
                "has_projection": d_source != d_target,
            })

    def remove_connection(
        self,
        model: ASANNModel,
        conn_idx: int,
        optimizer,
        step: int,
    ):
        """Remove a skip connection — delete the projection matrix and free memory.

        The tensor operation is deleted. Memory is freed. This is real removal,
        not zeroing out a gate. Optimizer is updated in-place.
        """
        conn = model.connections[conn_idx]

        if self.logger:
            self.logger.log_surgery(step, "remove_connection", {
                "source": conn.source,
                "target": conn.target,
                "utility": conn.utility(),
            })

        # Collect params to remove before deleting
        removed_params = list(conn.parameters())

        # Delete the connection
        model.connections.pop(conn_idx)
        if conn.projection is not None:
            del conn.projection
        del conn.scale

        # Remove stale params from optimizer
        optimizer.register_removed_parameters(removed_params)

    # ==================== Helper Methods ====================

    def _remove_spatially_invalid_connections(self, model: ASANNModel):
        """Remove connections that span different spatial resolutions.

        Called after add_layer/remove_layer to clean up connections whose
        source and target now reside at different H×W due to stride-2 layers.
        Also updates spatial shape metadata on remaining connections.
        """
        invalid_indices = []
        for i, conn in enumerate(model.connections):
            # Get the actual spatial shape of h[source]
            if conn.source == 0:
                src_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif conn.source - 1 < len(model.layers):
                src_spatial = getattr(model.layers[conn.source - 1], 'spatial_shape', None)
            else:
                invalid_indices.append(i)
                continue

            # Get the actual spatial shape of h[target-1] (the tensor being added to)
            tgt_h_idx = conn.target - 1
            if tgt_h_idx == 0:
                tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif tgt_h_idx - 1 < len(model.layers):
                tgt_spatial = getattr(model.layers[tgt_h_idx - 1], 'spatial_shape', None)
            else:
                invalid_indices.append(i)
                continue

            # Check spatial resolution match
            if (src_spatial is not None and tgt_spatial is not None
                    and (src_spatial[1] != tgt_spatial[1]
                         or src_spatial[2] != tgt_spatial[2])):
                print(f"  [SURGERY] Removing spatially invalid connection "
                      f"{conn.source}->{conn.target} "
                      f"(src={src_spatial}, tgt={tgt_spatial})")
                invalid_indices.append(i)
            else:
                # Update spatial shape metadata on valid connections
                conn.spatial_source_shape = src_spatial
                conn.spatial_target_shape = tgt_spatial

        # Remove invalid connections in reverse order to preserve indices
        for idx in reversed(invalid_indices):
            model.connections.pop(idx)

    def _repair_connection_projections(self, model: ASANNModel, optimizer=None):
        """Validate and repair all skip connection projections after structural surgery.

        After add_layer/remove_layer, connection indices are shifted and spatial
        shapes are validated, but the *projection dimensions* can become stale:
        a connection's Conv2d(C_old_src, C_old_tgt) may no longer match the
        actual channel counts at its (now shifted) source/target layers.

        This method re-reads the actual dimensions from the model and updates
        any mismatched projections using the existing _update_connection_*_dim
        helpers (the same ones used by add_channel).
        """
        for conn in model.connections:
            # --- Determine actual source dimension ---
            if conn.source == 0:
                src_spatial = getattr(model.input_projection, 'spatial_shape', None)
                d_source = model.input_projection.out_features
            elif conn.source - 1 < len(model.layers):
                src_layer = model.layers[conn.source - 1]
                src_spatial = getattr(src_layer, 'spatial_shape', None)
                d_source = src_layer.out_features
            else:
                continue  # stale connection, will be cleaned up elsewhere

            # --- Determine actual target dimension ---
            # h[target] is the input to layer[target] = output of layer[target-1]
            # Skip connection adds to h[target], which has the shape of
            # the output of layer[target-1].
            tgt_h_idx = conn.target - 1  # layer whose output forms h[target]
            if tgt_h_idx == 0:
                tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
                d_target = model.input_projection.out_features
            elif tgt_h_idx - 1 < len(model.layers):
                tgt_layer = model.layers[tgt_h_idx - 1]
                tgt_spatial = getattr(tgt_layer, 'spatial_shape', None)
                d_target = tgt_layer.out_features
            else:
                continue

            is_spatial = (src_spatial is not None and tgt_spatial is not None)

            if is_spatial:
                actual_C_src = src_spatial[0]
                actual_C_tgt = tgt_spatial[0]

                # Check source channel mismatch
                if conn.projection is not None:
                    proj_C_in = conn.projection.in_channels
                    proj_C_out = conn.projection.out_channels
                    if proj_C_in != actual_C_src:
                        self._update_spatial_connection_source(
                            model, conn, actual_C_src,
                            self.config.device, optimizer)
                    # Re-read projection after potential source update
                    if conn.projection is not None:
                        proj_C_out = conn.projection.out_channels
                    if proj_C_out != actual_C_tgt:
                        self._update_spatial_connection_target(
                            model, conn, actual_C_tgt,
                            self.config.device, optimizer)
                else:
                    # No projection — check if one is needed now
                    if actual_C_src != actual_C_tgt:
                        new_proj = nn.Conv2d(
                            actual_C_src, actual_C_tgt,
                            kernel_size=1, bias=False,
                        ).to(self.config.device)
                        nn.init.zeros_(new_proj.weight)
                        conn.projection = new_proj
                        if optimizer is not None:
                            optimizer.register_new_parameters(
                                [new_proj.weight],
                                group_name='skip_connections',
                                source_layer=-1,
                                surgery_type='conn_repair')
            else:
                # Flat connections
                if conn.projection is not None:
                    proj_d_in = conn.projection.in_features
                    proj_d_out = conn.projection.out_features
                    if proj_d_in != d_source:
                        self._update_flat_connection_source(
                            model, conn, d_source,
                            self.config.device, optimizer)
                    if conn.projection is not None:
                        proj_d_out = conn.projection.out_features
                    if proj_d_out != d_target:
                        self._update_flat_connection_target(
                            model, conn, d_target,
                            self.config.device, optimizer)
                else:
                    if d_source != d_target:
                        new_proj = nn.Linear(
                            d_source, d_target, bias=False,
                        ).to(self.config.device)
                        nn.init.zeros_(new_proj.weight)
                        conn.projection = new_proj
                        if optimizer is not None:
                            optimizer.register_new_parameters(
                                [new_proj.weight],
                                group_name='skip_connections',
                                source_layer=-1,
                                surgery_type='conn_repair')

    def _is_spatial_connection(self, conn: SkipConnection) -> bool:
        """Check if a connection uses spatial (Conv2d) projection or operates on spatial tensors."""
        return (conn.spatial_source_shape is not None
                and conn.spatial_target_shape is not None)

    def _update_connection_source_dim(
        self, model: ASANNModel, conn: SkipConnection, new_d_source: int,
        optimizer=None
    ):
        """Update a connection's projection when its source dimension changes.

        For flat connections: new_d_source is the full flat dimension (d_new).
        For spatial connections: new_d_source is the new channel count (C_new).
        """
        device = self.config.device
        is_spatial = self._is_spatial_connection(conn)

        if is_spatial:
            self._update_spatial_connection_source(model, conn, new_d_source, device, optimizer)
        else:
            self._update_flat_connection_source(model, conn, new_d_source, device, optimizer)

    def _update_flat_connection_source(self, model, conn, new_d_source, device, optimizer=None):
        """Update a flat (Linear) connection when source dimension changes."""
        if conn.projection is not None:
            old_proj = conn.projection
            d_target = old_proj.out_features
            new_proj = nn.Linear(new_d_source, d_target, bias=False).to(device)
            # Copy overlapping weights
            copy_cols = min(old_proj.in_features, new_d_source)
            new_proj.weight.data[:, :copy_cols] = old_proj.weight.data[:, :copy_cols]
            if optimizer is not None:
                optimizer.register_neuron_surgery(
                    old_proj.weight, new_proj.weight, -1, 'conn_update')
            conn.projection = new_proj
        else:
            # Dimensions might now differ — need to add projection
            if conn.target <= model.num_layers:
                if conn.target == model.num_layers:
                    d_target = model.output_head.in_features
                else:
                    d_target = model.layers[conn.target - 1].in_features

                if new_d_source != d_target:
                    new_proj = nn.Linear(new_d_source, d_target, bias=False).to(device)
                    nn.init.zeros_(new_proj.weight)
                    conn.projection = new_proj
                    if optimizer is not None:
                        optimizer.register_new_parameters(
                            [new_proj.weight], group_name='skip_connections',
                            source_layer=-1, surgery_type='conn_update')

    def _update_spatial_connection_source(self, model, conn, new_C_source, device, optimizer=None):
        """Update a spatial (Conv2d) connection when source channels change."""
        # Update tracked spatial shape
        _, H_src, W_src = conn.spatial_source_shape
        conn.spatial_source_shape = (new_C_source, H_src, W_src)

        C_target = conn.spatial_target_shape[0]

        if conn.projection is not None:
            old_proj = conn.projection  # Conv2d(C_old_src, C_tgt, 1)
            old_C_src = old_proj.in_channels
            new_proj = nn.Conv2d(new_C_source, C_target, kernel_size=1, bias=False).to(device)
            # Copy overlapping weights: [C_tgt, C_src, 1, 1]
            copy_in = min(old_C_src, new_C_source)
            copy_out = min(old_proj.out_channels, C_target)
            new_proj.weight.data[:copy_out, :copy_in] = old_proj.weight.data[:copy_out, :copy_in]
            if optimizer is not None:
                optimizer.register_neuron_surgery(
                    old_proj.weight, new_proj.weight, -1, 'conn_update')
            conn.projection = new_proj
        else:
            # No projection existed (channels matched), but now they may differ
            if new_C_source != C_target:
                new_proj = nn.Conv2d(new_C_source, C_target, kernel_size=1, bias=False).to(device)
                nn.init.zeros_(new_proj.weight)
                conn.projection = new_proj
                if optimizer is not None:
                    optimizer.register_new_parameters(
                        [new_proj.weight], group_name='skip_connections',
                        source_layer=-1, surgery_type='conn_update')

    def _update_connection_target_dim(
        self, model: ASANNModel, conn: SkipConnection, new_d_target: int,
        optimizer=None
    ):
        """Update a connection's projection when its target dimension changes.

        This happens when neurons are added/removed from a layer, changing the
        width that skip connections must output to match h[layer_idx].

        For flat connections: new_d_target is the full flat dimension (d_new).
        For spatial connections: new_d_target is the new channel count (C_new).
        """
        device = self.config.device
        is_spatial = self._is_spatial_connection(conn)

        if is_spatial:
            self._update_spatial_connection_target(model, conn, new_d_target, device, optimizer)
        else:
            self._update_flat_connection_target(model, conn, new_d_target, device, optimizer)

    def _update_flat_connection_target(self, model, conn, new_d_target, device, optimizer=None):
        """Update a flat (Linear) connection when target dimension changes."""
        if conn.projection is not None:
            old_proj = conn.projection
            d_source = old_proj.in_features
            new_proj = nn.Linear(d_source, new_d_target, bias=False).to(device)
            # Copy overlapping weights
            copy_rows = min(old_proj.out_features, new_d_target)
            new_proj.weight.data[:copy_rows, :] = old_proj.weight.data[:copy_rows, :]
            if optimizer is not None:
                optimizer.register_neuron_surgery(
                    old_proj.weight, new_proj.weight, -1, 'conn_update')
            conn.projection = new_proj
        else:
            # No projection existed (source == target dim), but now they differ
            if conn.source == 0:
                d_source = model.input_projection.out_features
            else:
                d_source = model.layers[conn.source - 1].out_features

            if d_source != new_d_target:
                new_proj = nn.Linear(d_source, new_d_target, bias=False).to(device)
                # Initialize as near-identity (copy what we can, zero the rest)
                copy_d = min(d_source, new_d_target)
                new_proj.weight.data[:copy_d, :copy_d] = torch.eye(copy_d, device=device)
                conn.projection = new_proj
                if optimizer is not None:
                    optimizer.register_new_parameters(
                        [new_proj.weight], group_name='skip_connections',
                        source_layer=-1, surgery_type='conn_update')

    def _update_spatial_connection_target(self, model, conn, new_C_target, device, optimizer=None):
        """Update a spatial (Conv2d) connection when target channels change."""
        # Update tracked spatial shape
        _, H_tgt, W_tgt = conn.spatial_target_shape
        conn.spatial_target_shape = (new_C_target, H_tgt, W_tgt)

        C_source = conn.spatial_source_shape[0]

        if conn.projection is not None:
            old_proj = conn.projection  # Conv2d(C_src, C_old_tgt, 1)
            old_C_tgt = old_proj.out_channels
            new_proj = nn.Conv2d(C_source, new_C_target, kernel_size=1, bias=False).to(device)
            # Copy overlapping weights: [C_tgt, C_src, 1, 1]
            copy_out = min(old_C_tgt, new_C_target)
            new_proj.weight.data[:copy_out] = old_proj.weight.data[:copy_out]
            if optimizer is not None:
                optimizer.register_neuron_surgery(
                    old_proj.weight, new_proj.weight, -1, 'conn_update')
            conn.projection = new_proj
        else:
            # No projection existed (channels matched), but now they may differ
            if C_source != new_C_target:
                new_proj = nn.Conv2d(C_source, new_C_target, kernel_size=1, bias=False).to(device)
                # Initialize as near-identity: copy_c channels get identity mapping
                copy_c = min(C_source, new_C_target)
                for c in range(copy_c):
                    new_proj.weight.data[c, c, 0, 0] = 1.0
                conn.projection = new_proj
                if optimizer is not None:
                    optimizer.register_new_parameters(
                        [new_proj.weight], group_name='skip_connections',
                        source_layer=-1, surgery_type='conn_update')

    def _update_ops_dimensions(self, pipeline: OperationPipeline, new_d: int, device: str):
        """Update dimension-dependent operations in a pipeline.

        Handles both Python and CUDA op variants. CUDA ops have the same
        internal parameter structure, so copy logic is shared — only the
        class used for construction differs.
        """
        _cuda = _CUDA_OPS_AVAILABLE
        from .model import GatedOperation

        for i, raw_op in enumerate(pipeline.operations):
            new_op = None
            # Unwrap GatedOperation to check the inner op type
            gate_wrapper = None
            op = raw_op
            if isinstance(raw_op, GatedOperation):
                gate_wrapper = raw_op
                op = raw_op.operation

            if isinstance(op, nn.BatchNorm1d):
                new_op = nn.BatchNorm1d(new_d).to(device)
                copy_d = min(op.num_features, new_d)
                new_op.weight.data[:copy_d] = op.weight.data[:copy_d]
                new_op.bias.data[:copy_d] = op.bias.data[:copy_d]
                if op.running_mean is not None:
                    new_op.running_mean[:copy_d] = op.running_mean[:copy_d]
                    new_op.running_var[:copy_d] = op.running_var[:copy_d]

            elif isinstance(op, nn.LayerNorm):
                new_op = nn.LayerNorm(new_d).to(device)
                copy_d = min(op.normalized_shape[0], new_d)
                new_op.weight.data[:copy_d] = op.weight.data[:copy_d]
                new_op.bias.data[:copy_d] = op.bias.data[:copy_d]

            # --- ActivationNoise (parameter-free) ---
            elif isinstance(op, ActivationNoise):
                new_op = ActivationNoise(new_d, config=op._config).to(device)

            # --- Conv1dBlock ---
            elif isinstance(op, Conv1dBlock) or (_cuda and isinstance(op, Conv1dBlockCUDA)):
                use_cuda = _cuda and isinstance(op, Conv1dBlockCUDA)
                cls = Conv1dBlockCUDA if use_cuda else Conv1dBlock
                new_op = cls(new_d, kernel_size=op.kernel_size).to(device)
                new_op.conv.weight.data.copy_(op.conv.weight.data)
                new_op.conv.bias.data.copy_(op.conv.bias.data)

            # --- Conv2dBlock ---
            elif isinstance(op, Conv2dBlock) or (_cuda and isinstance(op, Conv2dBlockCUDA)):
                use_cuda = _cuda and isinstance(op, Conv2dBlockCUDA)
                spatial = self.config.spatial_shape if hasattr(self.config, 'spatial_shape') else None
                cls = Conv2dBlockCUDA if use_cuda else Conv2dBlock
                new_op = cls(new_d, kernel_size=op.kernel_size, spatial_shape=spatial).to(device)
                old_C = op.C
                new_C = new_op.C
                copy_C = min(old_C, new_C)
                new_op.conv.weight.data[:copy_C, :, :, :] = op.conv.weight.data[:copy_C, :, :, :]
                new_op.conv.bias.data[:copy_C] = op.conv.bias.data[:copy_C]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- FactoredEmbedding ---
            elif isinstance(op, FactoredEmbedding) or (_cuda and isinstance(op, FactoredEmbeddingCUDA)):
                use_cuda = _cuda and isinstance(op, FactoredEmbeddingCUDA)
                cls = FactoredEmbeddingCUDA if use_cuda else FactoredEmbedding
                old_d = op.d
                old_rank = op.rank
                new_op = cls(new_d).to(device)
                new_rank = new_op.rank
                copy_d = min(old_d, new_d)
                copy_rank = min(old_rank, new_rank)
                new_op.U.data[:copy_d, :copy_rank] = op.U.data[:copy_d, :copy_rank]
                new_op.V.data[:copy_rank, :copy_d] = op.V.data[:copy_rank, :copy_d]

            # --- MLPEmbedding ---
            elif isinstance(op, MLPEmbedding) or (_cuda and isinstance(op, MLPEmbeddingCUDA)):
                use_cuda = _cuda and isinstance(op, MLPEmbeddingCUDA)
                cls = MLPEmbeddingCUDA if use_cuda else MLPEmbedding
                old_d = op.d
                old_hidden = op.hidden
                new_op = cls(new_d).to(device)
                new_hidden = new_op.hidden
                copy_d = min(old_d, new_d)
                copy_h = min(old_hidden, new_hidden)
                new_op.encode.weight.data[:copy_h, :copy_d] = op.encode.weight.data[:copy_h, :copy_d]
                new_op.encode.bias.data[:copy_h] = op.encode.bias.data[:copy_h]
                new_op.decode.weight.data[:copy_d, :copy_h] = op.decode.weight.data[:copy_d, :copy_h]
                new_op.decode.bias.data[:copy_d] = op.decode.bias.data[:copy_d]

            # --- SelfAttention ---
            elif isinstance(op, SelfAttentionOp) or (_cuda and isinstance(op, SelfAttentionOpCUDA)):
                use_cuda = _cuda and isinstance(op, SelfAttentionOpCUDA)
                cls = SelfAttentionOpCUDA if use_cuda else SelfAttentionOp
                old_d, old_rank = op.d, op.rank
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                copy_r = min(old_rank, new_op.rank)
                new_op.Q_emb.data[:copy_d, :copy_r] = op.Q_emb.data[:copy_d, :copy_r]
                new_op.K_emb.data[:copy_d, :copy_r] = op.K_emb.data[:copy_d, :copy_r]
                new_op.V_emb.data[:copy_d, :copy_r] = op.V_emb.data[:copy_d, :copy_r]
                new_op.out_proj.data[:copy_r] = op.out_proj.data[:copy_r]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- MultiHeadAttention ---
            elif isinstance(op, MultiHeadAttentionOp) or (_cuda and isinstance(op, MultiHeadAttentionOpCUDA)):
                use_cuda = _cuda and isinstance(op, MultiHeadAttentionOpCUDA)
                cls = MultiHeadAttentionOpCUDA if use_cuda else MultiHeadAttentionOp
                old_d = op.d
                old_H, old_r = op.num_heads, op.head_rank
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                copy_H = min(old_H, new_op.num_heads)
                copy_r = min(old_r, new_op.head_rank)
                new_op.Q_emb.data[:copy_H, :copy_d, :copy_r] = op.Q_emb.data[:copy_H, :copy_d, :copy_r]
                new_op.K_emb.data[:copy_H, :copy_d, :copy_r] = op.K_emb.data[:copy_H, :copy_d, :copy_r]
                new_op.V_emb.data[:copy_H, :copy_d, :copy_r] = op.V_emb.data[:copy_H, :copy_d, :copy_r]
                new_op.out_proj.data[:copy_H, :copy_r] = op.out_proj.data[:copy_H, :copy_r]
                new_op.head_weights.data[:copy_H] = op.head_weights.data[:copy_H]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- CrossAttention ---
            elif isinstance(op, CrossAttentionOp) or (_cuda and isinstance(op, CrossAttentionOpCUDA)):
                use_cuda = _cuda and isinstance(op, CrossAttentionOpCUDA)
                cls = CrossAttentionOpCUDA if use_cuda else CrossAttentionOp
                old_d = op.d
                old_M, old_r = op.num_memories, op.rank
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                copy_M = min(old_M, new_op.num_memories)
                copy_r = min(old_r, new_op.rank)
                new_op.Q_emb.data[:copy_d, :copy_r] = op.Q_emb.data[:copy_d, :copy_r]
                new_op.mem_keys.data[:copy_M, :copy_r] = op.mem_keys.data[:copy_M, :copy_r]
                new_op.mem_values.data[:copy_M, :copy_d] = op.mem_values.data[:copy_M, :copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- CausalAttention ---
            elif isinstance(op, CausalAttentionOp) or (_cuda and isinstance(op, CausalAttentionOpCUDA)):
                use_cuda = _cuda and isinstance(op, CausalAttentionOpCUDA)
                cls = CausalAttentionOpCUDA if use_cuda else CausalAttentionOp
                old_d, old_rank = op.d, op.rank
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                copy_r = min(old_rank, new_op.rank)
                new_op.Q_emb.data[:copy_d, :copy_r] = op.Q_emb.data[:copy_d, :copy_r]
                new_op.K_emb.data[:copy_d, :copy_r] = op.K_emb.data[:copy_d, :copy_r]
                new_op.V_emb.data[:copy_d, :copy_r] = op.V_emb.data[:copy_d, :copy_r]
                new_op.out_proj.data[:copy_r] = op.out_proj.data[:copy_r]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- GeometricEmbedding ---
            elif isinstance(op, GeometricEmbedding) or (_cuda and isinstance(op, GeometricEmbeddingCUDA)):
                use_cuda = _cuda and isinstance(op, GeometricEmbeddingCUDA)
                cls = GeometricEmbeddingCUDA if use_cuda else GeometricEmbedding
                old_d = op.d
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                new_op.radial_scale.data[:copy_d] = op.radial_scale.data[:copy_d]
                new_op.bias.data[:copy_d] = op.bias.data[:copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- PositionalEmbedding ---
            elif isinstance(op, PositionalEmbedding) or (_cuda and isinstance(op, PositionalEmbeddingCUDA)):
                use_cuda = _cuda and isinstance(op, PositionalEmbeddingCUDA)
                cls = PositionalEmbeddingCUDA if use_cuda else PositionalEmbedding
                old_d = op.d
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                new_op.pos_emb.data[:copy_d] = op.pos_emb.data[:copy_d]

            # --- EMASmooth ---
            elif isinstance(op, EMASmooth) or (_cuda and isinstance(op, EMASmoothCUDA)):
                use_cuda = _cuda and isinstance(op, EMASmoothCUDA)
                cls = EMASmoothCUDA if use_cuda else EMASmooth
                old_d = op.d
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                new_op.alpha_logit.data[:copy_d] = op.alpha_logit.data[:copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- GatedLinearUnit ---
            elif isinstance(op, GatedLinearUnit) or (_cuda and isinstance(op, GatedLinearUnitCUDA)):
                use_cuda = _cuda and isinstance(op, GatedLinearUnitCUDA)
                cls = GatedLinearUnitCUDA if use_cuda else GatedLinearUnit
                old_d = op.d
                new_op = cls(new_d).to(device)
                copy_d = min(old_d, new_d)
                new_op.gate_proj.weight.data[:copy_d, :copy_d] = op.gate_proj.weight.data[:copy_d, :copy_d]
                new_op.gate_proj.bias.data[:copy_d] = op.gate_proj.bias.data[:copy_d]
                new_op.value_proj.weight.data[:copy_d, :copy_d] = op.value_proj.weight.data[:copy_d, :copy_d]
                new_op.value_proj.bias.data[:copy_d] = op.value_proj.bias.data[:copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- TemporalDiff ---
            elif isinstance(op, TemporalDiff) or (_cuda and isinstance(op, TemporalDiffCUDA)):
                use_cuda = _cuda and isinstance(op, TemporalDiffCUDA)
                cls = TemporalDiffCUDA if use_cuda else TemporalDiff
                new_op = cls(new_d).to(device)
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- DerivativeConv1d (PDE discovery) ---
            elif isinstance(op, DerivativeConv1d):
                new_op = DerivativeConv1d(new_d, order=op.order).to(device)
                # Conv1d(1,1,3) kernel is dimension-independent — copy weights
                new_op.conv.weight.data.copy_(op.conv.weight.data)
                new_op.conv.bias.data.copy_(op.conv.bias.data)
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- PolynomialOp ---
            elif isinstance(op, PolynomialOp):
                old_d = op.d
                new_op = PolynomialOp(new_d, degree=op.degree).to(device)
                old_rank = op.rank
                new_rank = new_op.rank
                copy_d = min(old_d, new_d)
                copy_rank = min(old_rank, new_rank)
                copy_deg_d = min(old_d * op.degree, new_d * op.degree)
                new_op.project_down.weight.data[:copy_rank, :copy_deg_d] = \
                    op.project_down.weight.data[:copy_rank, :copy_deg_d]
                new_op.project_down.bias.data[:copy_rank] = \
                    op.project_down.bias.data[:copy_rank]
                new_op.project_up.weight.data[:copy_d, :copy_rank] = \
                    op.project_up.weight.data[:copy_d, :copy_rank]
                new_op.project_up.bias.data[:copy_d] = \
                    op.project_up.bias.data[:copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- KANLinearOp ---
            elif isinstance(op, KANLinearOp):
                old_d = op.d
                num_grids = op.num_grids
                new_op = KANLinearOp(new_d, num_grids=num_grids).to(device)
                copy_d = min(old_d, new_d)
                copy_grid_d = min(old_d * num_grids, new_d * num_grids)
                # LayerNorm
                new_op.layernorm.weight.data[:copy_d] = op.layernorm.weight.data[:copy_d]
                new_op.layernorm.bias.data[:copy_d] = op.layernorm.bias.data[:copy_d]
                # Spline weight: Linear(d*G, d, bias=False)
                new_op.spline_weight.weight.data[:copy_d, :copy_grid_d] = \
                    op.spline_weight.weight.data[:copy_d, :copy_grid_d]
                # Base weight: Linear(d, d, bias=False)
                new_op.base_weight.weight.data[:copy_d, :copy_d] = \
                    op.base_weight.weight.data[:copy_d, :copy_d]
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- BranchedOperationBlock ---
            elif isinstance(op, BranchedOperationBlock):
                new_branches = []
                for branch in op.branches:
                    if isinstance(branch, DerivativeConv1d):
                        new_b = DerivativeConv1d(new_d, order=branch.order).to(device)
                        new_b.conv.weight.data.copy_(branch.conv.weight.data)
                        new_b.conv.bias.data.copy_(branch.conv.bias.data)
                        new_b.gate_logit.data.copy_(branch.gate_logit.data)
                    elif isinstance(branch, PolynomialOp):
                        old_d_b = branch.d
                        new_b = PolynomialOp(new_d, degree=branch.degree).to(device)
                        copy_d_b = min(old_d_b, new_d)
                        copy_rank_b = min(branch.rank, new_b.rank)
                        copy_deg_d_b = min(old_d_b * branch.degree,
                                           new_d * branch.degree)
                        new_b.project_down.weight.data[:copy_rank_b, :copy_deg_d_b] = \
                            branch.project_down.weight.data[:copy_rank_b, :copy_deg_d_b]
                        new_b.project_down.bias.data[:copy_rank_b] = \
                            branch.project_down.bias.data[:copy_rank_b]
                        new_b.project_up.weight.data[:copy_d_b, :copy_rank_b] = \
                            branch.project_up.weight.data[:copy_d_b, :copy_rank_b]
                        new_b.project_up.bias.data[:copy_d_b] = \
                            branch.project_up.bias.data[:copy_d_b]
                        new_b.gate_logit.data.copy_(branch.gate_logit.data)
                    else:
                        # Generic fallback: recreate with new dimension
                        new_b = type(branch)(new_d).to(device)
                    new_branches.append(new_b)
                new_op = BranchedOperationBlock(new_d, new_branches).to(device)
                new_op.branch_logits.data.copy_(op.branch_logits.data)
                new_op.gate_logit.data.copy_(op.gate_logit.data)

            # --- Graph Operations (dimension-dependent: contain Linear(d,d)) ---

            elif isinstance(op, GraphBranchedBlock):
                # Rebuild both branches with new dimension, copy gate
                new_op = GraphBranchedBlock(
                    new_d, op.branch_agg.adj, op.branch_agg._degree_raw,
                    op.branch_attn.edge_index, op.branch_attn.num_nodes,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                # Copy merge layer
                copy_2d = min(op.merge.in_features, new_op.merge.in_features)
                new_op.merge.weight.data[:copy_d, :copy_2d] = \
                    op.merge.weight.data[:copy_d, :copy_2d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, NeighborAggregation):
                new_op = NeighborAggregation(
                    new_d, op.adj, op._degree_raw,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, GraphAttentionAggregation):
                new_op = GraphAttentionAggregation(
                    new_d, op.edge_index, op.num_nodes,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]
                new_op.attn_src.data[:copy_d] = op.attn_src.data[:copy_d]
                new_op.attn_dst.data[:copy_d] = op.attn_dst.data[:copy_d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, GraphDiffusion):
                new_op = GraphDiffusion(
                    new_d, op.adj_raw, op._degree_raw, op.max_hops,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]
                new_op.hop_weights.data.copy_(op.hop_weights.data)
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, DegreeScaling):
                new_op = DegreeScaling(
                    new_d, op._degree_raw, initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.scale.data[:copy_d] = op.scale.data[:copy_d]
                new_op.bias.data[:copy_d] = op.bias.data[:copy_d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, SpectralConv):
                # Rebuild SpectralConv: graph structure unchanged, only d changes
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = SpectralConv(
                    new_d, adj_raw, op._degree_raw, K=op.K,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                copy_K = min(op.K + 1, new_op.K + 1)
                new_op.theta.data[:copy_K, :copy_d] = op.theta.data[:copy_K, :copy_d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, PairNorm):
                new_op = PairNorm(new_d, s_init=op.s.item(), num_nodes=op.num_nodes).to(device)

            elif isinstance(op, GraphNorm):
                new_op = GraphNorm(new_d, num_nodes=op.num_nodes).to(device)
                copy_d = min(op.d, new_d)
                new_op.gamma.data[:copy_d] = op.gamma.data[:copy_d]
                new_op.beta.data[:copy_d] = op.beta.data[:copy_d]
                new_op.alpha.data.copy_(op.alpha.data)

            elif isinstance(op, MessagePassingGIN):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_indices, op._adj_values, op._adj_size
                )
                new_op = MessagePassingGIN(
                    new_d, adj_raw, op._degree_raw,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                # Copy MLP weights (layer 0 and layer 2)
                new_op.mlp[0].weight.data[:copy_d, :copy_d] = \
                    op.mlp[0].weight.data[:copy_d, :copy_d]
                new_op.mlp[0].bias.data[:copy_d] = op.mlp[0].bias.data[:copy_d]
                new_op.mlp[2].weight.data[:copy_d, :copy_d] = \
                    op.mlp[2].weight.data[:copy_d, :copy_d]
                new_op.mlp[2].bias.data[:copy_d] = op.mlp[2].bias.data[:copy_d]
                new_op.eps.data.copy_(op.eps.data)
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, MessageBooster):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_indices, op._adj_values, op._adj_size
                )
                new_op = MessageBooster(
                    new_d, adj_raw, op._degree_raw,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                # Copy MLP weights (layer 0 and layer 2)
                new_op.mlp[0].weight.data[:copy_d, :copy_d] = \
                    op.mlp[0].weight.data[:copy_d, :copy_d]
                new_op.mlp[0].bias.data[:copy_d] = op.mlp[0].bias.data[:copy_d]
                new_op.mlp[2].weight.data[:copy_d, :copy_d] = \
                    op.mlp[2].weight.data[:copy_d, :copy_d]
                new_op.mlp[2].bias.data[:copy_d] = op.mlp[2].bias.data[:copy_d]
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, GraphPositionalEncoding):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = GraphPositionalEncoding(
                    new_d, adj_raw, op._degree_raw, k=op.k,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_k = min(op.k, new_op.k)
                copy_d = min(op.d, new_d)
                new_op.proj.weight.data[:copy_d, :copy_k] = \
                    op.proj.weight.data[:copy_d, :copy_k]
                new_op.gate.data.copy_(op.gate.data)

            # --- GRUOp ---
            elif isinstance(op, GRUOp):
                new_op = GRUOp(new_d, num_chunks=op.num_chunks).to(device)
                new_op.gate_logit.data.copy_(op.gate_logit.data)
                if new_op.chunk_size == op.chunk_size:
                    for (name_new, p_new), (name_old, p_old) in zip(
                        new_op.gru.named_parameters(), op.gru.named_parameters()
                    ):
                        p_new.data.copy_(p_old.data)

            # --- Phase 4 Graph Ops ---

            elif isinstance(op, APPNPPropagation):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = APPNPPropagation(
                    new_d, adj_raw, op._degree_raw,
                    alpha=op.alpha, K=op.K,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]

            elif isinstance(op, GraphSAGEMean):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_indices, op._adj_values, op._adj_size
                )
                new_op = GraphSAGEMean(
                    new_d, adj_raw, op._degree_raw,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                copy_2d = min(2 * op.d, 2 * new_d)
                new_op.linear.weight.data[:copy_d, :copy_2d] = \
                    op.linear.weight.data[:copy_d, :copy_2d]
                new_op.linear.bias.data[:copy_d] = op.linear.bias.data[:copy_d]

            elif isinstance(op, GraphSAGEGCN):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_indices, op._adj_values, op._adj_size
                )
                new_op = GraphSAGEGCN(
                    new_d, adj_raw, op._degree_raw,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]
                new_op.linear.bias.data[:copy_d] = op.linear.bias.data[:copy_d]

            elif isinstance(op, GATv2Aggregation):
                new_op = GATv2Aggregation(
                    new_d, op.edge_index, op.num_nodes,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear_l.weight.data[:copy_d, :copy_d] = \
                    op.linear_l.weight.data[:copy_d, :copy_d]
                new_op.linear_r.weight.data[:copy_d, :copy_d] = \
                    op.linear_r.weight.data[:copy_d, :copy_d]
                new_op.attn.data[:copy_d] = op.attn.data[:copy_d]

            elif isinstance(op, SGConv):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = SGConv(
                    new_d, adj_raw, op._degree_raw,
                    K=op.K, initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]

            elif isinstance(op, DropEdgeAggregation):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_indices, op._adj_values, op._adj_size
                )
                new_op = DropEdgeAggregation(
                    new_d, adj_raw, op._degree_raw,
                    drop_rate=op.drop_rate,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]

            elif isinstance(op, MixHopConv):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = MixHopConv(
                    new_d, adj_raw, op._degree_raw,
                    max_hops=op.max_hops,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                copy_in = min((op.max_hops + 1) * op.d,
                              (op.max_hops + 1) * new_d)
                new_op.proj.weight.data[:copy_d, :copy_in] = \
                    op.proj.weight.data[:copy_d, :copy_in]

            elif isinstance(op, VirtualNodeOp):
                new_op = VirtualNodeOp(
                    new_d, initial_gate=op.gate.item(), num_nodes=op.num_nodes
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.vn_embedding.data[:copy_d] = op.vn_embedding.data[:copy_d]
                new_op.mlp[0].weight.data[:copy_d, :copy_d] = \
                    op.mlp[0].weight.data[:copy_d, :copy_d]
                new_op.mlp[0].bias.data[:copy_d] = op.mlp[0].bias.data[:copy_d]
                new_op.mlp[2].weight.data[:copy_d, :copy_d] = \
                    op.mlp[2].weight.data[:copy_d, :copy_d]
                new_op.mlp[2].bias.data[:copy_d] = op.mlp[2].bias.data[:copy_d]

            elif isinstance(op, EdgeWeightedAggregation):
                new_op = EdgeWeightedAggregation(
                    new_d, op.edge_index, op.num_nodes,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                copy_2d = min(2 * op.d, 2 * new_d)
                new_op.edge_mlp[0].weight.data[:copy_d, :copy_2d] = \
                    op.edge_mlp[0].weight.data[:copy_d, :copy_2d]
                new_op.edge_mlp[0].bias.data[:copy_d] = op.edge_mlp[0].bias.data[:copy_d]
                new_op.edge_mlp[2].weight.data[:1, :copy_d] = \
                    op.edge_mlp[2].weight.data[:1, :copy_d]

            elif isinstance(op, DirectionalDiffusion):
                adj_raw = torch.sparse_coo_tensor(
                    op._adj_raw_indices, op._adj_raw_values, op._adj_raw_size
                )
                new_op = DirectionalDiffusion(
                    new_d, adj_raw, op._degree_raw,
                    max_hops=op.max_hops,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear_fwd.weight.data[:copy_d, :copy_d] = \
                    op.linear_fwd.weight.data[:copy_d, :copy_d]
                new_op.linear_bwd.weight.data[:copy_d, :copy_d] = \
                    op.linear_bwd.weight.data[:copy_d, :copy_d]
                new_op.hop_weights.data.copy_(op.hop_weights.data)
                new_op.gate.data.copy_(op.gate.data)

            elif isinstance(op, AdaptiveGraphConv):
                new_op = AdaptiveGraphConv(
                    new_d, op.num_nodes, d_emb=op.d_emb,
                    initial_gate=op.gate.item()
                ).to(device)
                copy_d = min(op.d, new_d)
                new_op.linear.weight.data[:copy_d, :copy_d] = \
                    op.linear.weight.data[:copy_d, :copy_d]
                new_op.E_src.data.copy_(op.E_src.data)
                new_op.E_dst.data.copy_(op.E_dst.data)
                new_op.gate.data.copy_(op.gate.data)

            if new_op is not None:
                # Preserve protection period metadata
                if hasattr(op, '_asann_added_step'):
                    new_op._asann_added_step = op._asann_added_step
                # Re-wrap in GatedOperation if the original was gated
                if gate_wrapper is not None:
                    gate_wrapper.operation = new_op
                    new_op = gate_wrapper
                ops_list = list(pipeline.operations)
                ops_list[i] = new_op
                pipeline.operations = nn.ModuleList(ops_list)

    def _remove_connections_for_layer(self, model: ASANNModel, layer_idx: int):
        """Remove all connections involving a layer that is being deleted."""
        model.connections = [
            conn for conn in model.connections
            if conn.source != layer_idx and conn.target != layer_idx
        ]
