"""
ASANN Encoder Framework: Domain-aware input encoders for self-architecting discovery.

Provides a general encoder interface that ASANN's treatment/surgery system can
use to discover the best input representation for each task. Experiments provide
a list of candidate encoders; ASANN starts with the simplest (LinearEncoder) and
escalates to more powerful encoders when underfitting persists.

Architecture:
    Raw Input -> [Encoder] -> [N, d_enc] -> ASANN self-architecting body -> output_head
                    ↑
              Discoverable via treatment system

Encoder Types:
    - LinearEncoder: Simple linear projection (default, backward compatible)
    - ConvEncoder: ConvStem wrapper for spatial/vision data
    - FourierEncoder: Random Fourier features for coordinate inputs (PDE)
    - PatchEmbedEncoder: ViT-style patch embedding for vision
    - TemporalEncoder: PatchTST-style patching for time series
    - GraphNodeEncoder: GNN pre-encoder for node classification
    - MolecularGraphEncoder: GNN on atom/bond molecular graphs (Phase 4)
    - TransformerEncoder: Chunked self-attention for high-dim tabular/bio features
    - AutoencoderEncoder: Bottleneck encoder with reconstruction loss

Usage:
    from asann.encoders import create_encoder, ENCODER_REGISTRY
    encoder = create_encoder("linear", d_input=524, d_output=32)
    h = encoder(x)  # [N, 32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type, Optional, Tuple
from abc import ABC, abstractmethod


# =============================================================================
# Base Encoder
# =============================================================================

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all ASANN encoders.

    Contract:
        - forward(x) -> [N, d_output] tensor (or [N, C, H, W] for spatial encoders)
        - d_output: int property (output dimension, flat equivalent)
        - d_input: int property (expected raw input dimension)
        - encoder_type: str class attribute (for registry/serialization)
        - is_spatial: bool property (True if output is [N, C, H, W])
        - describe() -> str (human-readable description)
        - cost() -> float (approximate FLOPs for architecture cost computation)
    """
    encoder_type: str = "base"

    @property
    @abstractmethod
    def d_output(self) -> int:
        """Output dimension (flat equivalent: C*H*W for spatial)."""
        ...

    @property
    @abstractmethod
    def d_input(self) -> int:
        """Expected raw input dimension."""
        ...

    @property
    def out_features(self) -> int:
        """Alias for d_output — compatibility with nn.Linear interface.

        Many parts of the codebase (surgery.py, scheduler.py, model.py) access
        model.input_projection.out_features. This ensures all encoders support it.
        """
        return self.d_output

    @property
    def in_features(self) -> int:
        """Alias for d_input — compatibility with nn.Linear interface."""
        return self.d_input

    @property
    def is_spatial(self) -> bool:
        """Whether this encoder outputs [N, C, H, W] for spatial layers."""
        return False

    def describe(self) -> str:
        """Human-readable description of the encoder."""
        n_params = sum(p.numel() for p in self.parameters())
        return (f"{self.encoder_type} encoder: "
                f"{self.d_input} -> {self.d_output} "
                f"({n_params:,} params)")

    def cost(self) -> float:
        """Approximate FLOPs for architecture cost computation."""
        return sum(p.numel() for p in self.parameters()) * 2.0  # 2 FLOPs per param (mul + add)


# =============================================================================
# LinearEncoder (Default — Backward Compatible)
# =============================================================================

class LinearEncoder(BaseEncoder):
    """Simple linear projection encoder.

    Wraps nn.Linear(d_input, d_output). This is the default encoder and
    matches the current ASANNModel.input_projection behavior exactly.
    """
    encoder_type = "linear"

    def __init__(self, d_input: int, d_output: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle spatial input [B, C, H, W] from spatial models
        # (e.g., when ENCODER_DOWNGRADE swaps conv -> linear)
        if x.dim() == 4:
            x = x.flatten(1)
        return self.linear(x)

    @property
    def d_output(self) -> int:
        return self.linear.out_features

    @property
    def d_input(self) -> int:
        return self.linear.in_features

    # Backward compatibility aliases
    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# =============================================================================
# ConvEncoder (Current ConvStem — Spatial Models)
# =============================================================================

class ConvEncoder(BaseEncoder):
    """Convolutional stem encoder for spatial/vision data.

    Wraps the existing ConvStem: Conv2d(C_in, C_stem, 3, pad=1) + BN + ReLU.
    Output is [N, C_stem, H, W] (spatial format for spatial ASANN layers).

    The d_output property returns C_stem * H * W for flat-equivalent compatibility,
    but the actual output tensor is 4D.
    """
    encoder_type = "conv"

    def __init__(self, C_in: int, C_stem: int, H: int, W: int, **kwargs):
        super().__init__()
        self.C_in = C_in
        self.C_stem = C_stem
        self.H = H
        self.W = W
        self.conv = nn.Conv2d(C_in, C_stem, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_stem)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W] (already reshaped by model.forward())
        return self.act(self.bn(self.conv(x)))

    @property
    def d_output(self) -> int:
        return self.C_stem * self.H * self.W

    @property
    def d_input(self) -> int:
        return self.C_in * self.H * self.W

    @property
    def is_spatial(self) -> bool:
        return True

    # Backward compatibility with ConvStem interface
    @property
    def out_features(self) -> int:
        return self.C_stem * self.H * self.W

    @property
    def in_features(self) -> int:
        return self.C_in * self.H * self.W

    @property
    def out_channels(self) -> int:
        return self.C_stem

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return (self.C_stem, self.H, self.W)

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"conv encoder: "
                f"[{self.C_in}, {self.H}, {self.W}] -> "
                f"[{self.C_stem}, {self.H}, {self.W}] "
                f"({n_params:,} params)")


# =============================================================================
# FourierEncoder (Tier 5 PDE — Phase 3)
# =============================================================================

class FourierEncoder(BaseEncoder):
    """Random Fourier feature encoder for coordinate inputs.

    Maps low-dimensional coordinates (e.g., [x, y] for 2D PDE) to a rich
    feature representation via random Fourier features:
        z = [sin(Bx), cos(Bx)] where B is a fixed random matrix

    This helps neural networks capture high-frequency functions (Tancik et al.,
    2020 — "Fourier Features Let Networks Learn High Frequency Functions").

    The B matrix is NOT learned — it's a fixed random projection that provides
    diverse frequency coverage. Only the final projection is learned.
    """
    encoder_type = "fourier"

    def __init__(self, d_input: int, d_output: int,
                 num_frequencies: int = 64, sigma: float = 10.0, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        # Random Fourier frequencies (fixed, not learned)
        self.register_buffer('B_matrix',
                             torch.randn(d_input, num_frequencies) * sigma)
        # Learned projection from 2*num_freq -> d_output
        self.proj = nn.Linear(2 * num_frequencies, d_output)
        self.norm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle spatial input [B, C, H, W] from spatial models
        if x.dim() == 4:
            x = x.flatten(1)
        # x: [N, d_input] — coordinates
        proj = x @ self.B_matrix  # [N, num_frequencies]
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.norm(self.proj(features))

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        num_freq = self.B_matrix.shape[1]
        n_params = sum(p.numel() for p in self.parameters())
        return (f"fourier encoder: {self._d_input}d coords -> "
                f"{num_freq} frequencies -> {self._d_output}d "
                f"({n_params:,} params)")


# =============================================================================
# PatchEmbedEncoder (Tier 3 Vision — Phase 3)
# =============================================================================

class PatchEmbedEncoder(BaseEncoder):
    """ViT-style patch embedding encoder for vision data.

    Divides input image into non-overlapping patches, projects each patch
    to d_embed dimensions, adds learned positional embeddings, then
    mean-pools across patches to produce [N, d_embed].

    Input: [N, C*H*W] flat tensor (internally reshaped using spatial_shape config)
    Output: [N, d_embed] flat tensor (NOT spatial — feeds into flat ASANN layers)
    """
    encoder_type = "patch_embed"

    def __init__(self, d_input: int, d_output: int,
                 C_in: int = 1, H: int = 28, W: int = 28,
                 patch_size: int = 4, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        self.C_in = C_in
        self.H = H
        self.W = W
        self.patch_size = patch_size

        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image size ({H}x{W}) must be divisible by patch_size ({patch_size})"

        self.num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = C_in * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, d_output)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_output) * 0.02
        )
        self.norm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Reshape flat input to image: [B, C*H*W] -> [B, C, H, W]
        x = x.view(B, self.C_in, self.H, self.W)

        # Extract non-overlapping patches
        p = self.patch_size
        # [B, C, H, W] -> [B, C, H//p, p, W//p, p] -> [B, num_patches, C*p*p]
        x = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, H//p, W//p, p, p]
        x = x.contiguous().view(B, self.C_in, -1, p, p)  # [B, C, num_patches, p, p]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_patches, C, p, p]
        x = x.view(B, self.num_patches, -1)  # [B, num_patches, C*p*p]

        # Project patches + positional embedding
        tokens = self.proj(x) + self.pos_embed  # [B, num_patches, d_output]
        tokens = self.norm(tokens)

        # Mean pool across patches -> [B, d_output]
        return tokens.mean(dim=1)

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"patch_embed encoder: [{self.C_in}, {self.H}, {self.W}] -> "
                f"{self.num_patches} patches (size {self.patch_size}) -> "
                f"{self._d_output}d ({n_params:,} params)")


# =============================================================================
# TemporalEncoder (Tier 4 Time Series — Phase 3)
# =============================================================================

class TemporalEncoder(BaseEncoder):
    """PatchTST-style temporal encoder for time series data.

    Divides flattened time series windows into patches along the time axis,
    projects each patch, adds positional embeddings, then mean-pools.

    Input: [N, window_size * n_features] flat tensor
    Output: [N, d_output] flat tensor
    """
    encoder_type = "temporal"

    def __init__(self, d_input: int, d_output: int,
                 n_features: int = 1, window_size: int = 24,
                 patch_size: int = 8, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        self.n_features = n_features
        self.window_size = window_size
        self.patch_size = patch_size

        # Handle case where window_size not divisible by patch_size
        self.num_patches = max(1, window_size // patch_size)
        actual_patch_size = window_size // self.num_patches
        self.actual_patch_size = actual_patch_size
        patch_dim = n_features * actual_patch_size

        self.patch_proj = nn.Linear(patch_dim, d_output)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_output) * 0.02
        )
        self.norm = nn.LayerNorm(d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Reshape: [B, window*features] -> [B, window, features]
        x = x.view(B, self.window_size, self.n_features)

        # Truncate to fit patches evenly
        usable = self.num_patches * self.actual_patch_size
        x = x[:, :usable, :]  # [B, usable, n_features]

        # Reshape into patches: [B, num_patches, patch_size * n_features]
        x = x.view(B, self.num_patches, self.actual_patch_size * self.n_features)

        # Project + positional embedding
        tokens = self.patch_proj(x) + self.pos_embed
        tokens = self.norm(tokens)

        # Mean pool -> [B, d_output]
        return tokens.mean(dim=1)

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"temporal encoder: {self.window_size} steps × "
                f"{self.n_features} features -> "
                f"{self.num_patches} patches -> "
                f"{self._d_output}d ({n_params:,} params)")


# =============================================================================
# GraphNodeEncoder (Tier 6 Graph Nodes — Phase 3)
# =============================================================================

class GraphNodeEncoder(BaseEncoder):
    """GNN pre-encoder for node-level tasks.

    Performs message passing using the model's stored graph adjacency
    (from model.set_graph_data()) BEFORE the ASANN body processes features.

    Uses simple GCN-style aggregation: h' = ReLU(D^{-1/2} A_hat D^{-1/2} h W)
    with residual connections.

    The graph adjacency is passed in at construction time (from model attributes).
    """
    encoder_type = "graph_node"

    def __init__(self, d_input: int, d_output: int,
                 num_gnn_layers: int = 2,
                 adj_sparse: Optional[torch.Tensor] = None,
                 degree: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        d = d_input
        for i in range(num_gnn_layers):
            d_next = d_output
            self.gnn_layers.append(nn.Linear(d, d_next))
            d = d_next

        self.norm = nn.LayerNorm(d_output)

        # Graph structure (buffers — not parameters, move with model.to())
        # Keep sparse for large graphs (N > 10K) to avoid O(N^2) memory;
        # the forward code already has a sparse path via _batch_graph_mm.
        if adj_sparse is not None:
            if adj_sparse.is_sparse and adj_sparse.shape[0] > 10000:
                self.register_buffer('_adj', adj_sparse.coalesce())
            elif adj_sparse.is_sparse:
                self.register_buffer('_adj', adj_sparse.to_dense())
            else:
                self.register_buffer('_adj', adj_sparse)
        else:
            self._adj = None

        if degree is not None:
            self.register_buffer('_degree', degree)
        else:
            self._degree = None

        # Residual projection if d_input != d_output
        if d_input != d_output:
            self.residual_proj = nn.Linear(d_input, d_output, bias=False)
        else:
            self.residual_proj = None

    def set_graph_data(self, adj_sparse, degree):
        """Update graph data (called when model.set_graph_data() is called)."""
        if adj_sparse.is_sparse and adj_sparse.shape[0] > 10000:
            self._adj = adj_sparse.coalesce()
        elif adj_sparse.is_sparse:
            self._adj = adj_sparse.to_dense()
        else:
            self._adj = adj_sparse
        self._degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_input] or [B*N, d_input] for batched graph input
        if self._adj is None:
            # No graph data — just linear projection
            return self.norm(self.gnn_layers[0](x))

        N = self._adj.shape[0]  # num_nodes
        D_inv_sqrt = (self._degree + 1).pow(-0.5).unsqueeze(1)  # [N, 1]
        # Tile degree normalization for batched input
        if x.shape[0] > N:
            B = x.shape[0] // N
            D_inv_sqrt = D_inv_sqrt.repeat(B, 1)  # [B*N, 1]

        h = x
        x_res = x
        for i, layer in enumerate(self.gnn_layers):
            # GCN aggregation: D^{-1/2} A D^{-1/2} h
            h_scaled = h * D_inv_sqrt

            if self._adj.is_sparse:
                from .surgery import _batch_graph_mm
                h_agg = _batch_graph_mm(self._adj, h_scaled, N)
            else:
                # Dense matmul: handle batched [B*N, d] -> [B, N, d]
                if h_scaled.shape[0] > N:
                    B = h_scaled.shape[0] // N
                    d = h_scaled.shape[1]
                    h_2d = h_scaled.view(B, N, d).permute(1, 0, 2).reshape(N, B * d)
                    h_agg = (self._adj @ h_2d).reshape(N, B, d).permute(1, 0, 2).reshape(B * N, d)
                else:
                    h_agg = self._adj @ h_scaled

            h_agg = h_agg * D_inv_sqrt

            # Transform
            h = layer(h_agg)

            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)

        # Residual
        if self.residual_proj is not None:
            h = h + self.residual_proj(x_res)
        elif h.shape == x_res.shape:
            h = h + x_res

        return self.norm(h)

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        n_layers = len(self.gnn_layers)
        has_graph = self._adj is not None
        return (f"graph_node encoder: {self._d_input} -> "
                f"{n_layers}-layer GCN -> {self._d_output}d "
                f"({'graph loaded' if has_graph else 'no graph'}) "
                f"({n_params:,} params)")


# =============================================================================
# TemporalGraphEncoder (Tier 6 Traffic — Spatio-Temporal)
# =============================================================================

class TemporalGraphEncoder(BaseEncoder):
    """Temporal-aware GNN encoder: dilated causal Conv1d → GCN aggregation.

    Processes time series features with dilated causal convolutions to capture
    temporal patterns (trends, periodicity), then applies GCN aggregation to
    capture spatial correlations across the graph.

    Architecture:
        Input [B*N, T] → reshape [B*N, 1, T]
        → DilatedCausalConv1d(dilation=1) → GELU → LayerNorm
        → DilatedCausalConv1d(dilation=2) → GELU → LayerNorm
        → DilatedCausalConv1d(dilation=4) → GELU → LayerNorm
        → AdaptiveAvgPool1d(1) → squeeze → [B*N, d_output]
        → GCN aggregation (D^{-1/2} A D^{-1/2} h W)
        → LayerNorm + residual → [B*N, d_output]

    Causal padding ensures no future information leaks into predictions.
    Dilations [1, 2, 4] give receptive field of 15 time steps with kernel_size=3.
    """
    encoder_type = "temporal_graph"

    def __init__(self, d_input: int, d_output: int,
                 num_gnn_layers: int = 1,
                 adj_sparse: Optional[torch.Tensor] = None,
                 degree: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output

        # Temporal conv stack: dilated causal Conv1d
        kernel_size = 3
        channels = [1, 32, 64, d_output]
        dilations = [1, 2, 4]

        self.conv_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.conv_paddings = []  # left-padding amounts

        for i in range(len(dilations)):
            dilation = dilations[i]
            c_in = channels[i]
            c_out = channels[i + 1]
            pad = (kernel_size - 1) * dilation  # causal: all padding on left
            self.conv_paddings.append(pad)
            self.conv_layers.append(
                nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, bias=False)
            )
            self.conv_norms.append(nn.LayerNorm(c_out))

        self.pool = nn.AdaptiveAvgPool1d(1)

        # GCN aggregation layer(s)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(nn.Linear(d_output, d_output))
        self.norm = nn.LayerNorm(d_output)

        # Graph structure — keep sparse for large graphs (N > 10K)
        if adj_sparse is not None:
            if adj_sparse.is_sparse and adj_sparse.shape[0] > 10000:
                self.register_buffer('_adj', adj_sparse.coalesce())
            elif adj_sparse.is_sparse:
                self.register_buffer('_adj', adj_sparse.to_dense())
            else:
                self.register_buffer('_adj', adj_sparse)
        else:
            self._adj = None

        if degree is not None:
            self.register_buffer('_degree', degree)
        else:
            self._degree = None

        # Residual projection if d_input != d_output
        if d_input != d_output:
            self.residual_proj = nn.Linear(d_input, d_output, bias=False)
        else:
            self.residual_proj = None

    def set_graph_data(self, adj_sparse, degree):
        """Update graph data (called when model.set_graph_data() is called)."""
        if adj_sparse.is_sparse and adj_sparse.shape[0] > 10000:
            self._adj = adj_sparse.coalesce()
        elif adj_sparse.is_sparse:
            self._adj = adj_sparse.to_dense()
        else:
            self._adj = adj_sparse
        self._degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, T] where T = d_input (e.g. 16 = 12 window + 4 time features)
        x_res = x

        # Temporal convolution: [B*N, T] -> [B*N, 1, T] -> Conv1d stack -> [B*N, d_output]
        h = x.unsqueeze(1)  # [B*N, 1, T]

        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.conv_norms)):
            # Causal padding: pad left only
            h = F.pad(h, (self.conv_paddings[i], 0))
            h = conv(h)                     # [B*N, c_out, T]
            # LayerNorm on channel dim: transpose to [B*N, T, c_out]
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = F.gelu(h)

        # Pool over time: [B*N, d_output, T'] -> [B*N, d_output]
        h = self.pool(h).squeeze(-1)

        # GCN aggregation
        if self._adj is not None:
            N = self._adj.shape[0]
            D_inv_sqrt = (self._degree + 1).pow(-0.5).unsqueeze(1)  # [N, 1]
            if h.shape[0] > N:
                B = h.shape[0] // N
                D_inv_sqrt_tiled = D_inv_sqrt.repeat(B, 1)  # [B*N, 1]
            else:
                D_inv_sqrt_tiled = D_inv_sqrt

            for j, gnn in enumerate(self.gnn_layers):
                h_scaled = h * D_inv_sqrt_tiled

                if h_scaled.shape[0] > N:
                    # Single matmul: reshape [B*N, d] -> [N, B*d], multiply
                    # once with adj [N, N], reshape back. Avoids expanding
                    # adj B times (saves B*N*N memory, better cache use).
                    B = h_scaled.shape[0] // N
                    d = h_scaled.shape[1]
                    h_2d = h_scaled.view(B, N, d).permute(1, 0, 2).reshape(N, B * d)
                    h_agg = (self._adj @ h_2d).reshape(N, B, d).permute(1, 0, 2).reshape(B * N, d)
                else:
                    h_agg = self._adj @ h_scaled

                h_agg = h_agg * D_inv_sqrt_tiled
                h = gnn(h_agg)
                if j < len(self.gnn_layers) - 1:
                    h = F.gelu(h)
        else:
            # No graph — just linear transform
            for j, gnn in enumerate(self.gnn_layers):
                h = gnn(h)
                if j < len(self.gnn_layers) - 1:
                    h = F.gelu(h)

        # Residual
        if self.residual_proj is not None:
            h = h + self.residual_proj(x_res)
        elif h.shape[-1] == x_res.shape[-1]:
            h = h + x_res

        return self.norm(h)

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        n_conv = len(self.conv_layers)
        n_gnn = len(self.gnn_layers)
        has_graph = self._adj is not None
        return (f"temporal_graph encoder: {self._d_input} -> "
                f"{n_conv}-layer dilated Conv1d + {n_gnn}-layer GCN -> "
                f"{self._d_output}d "
                f"({'graph loaded' if has_graph else 'no graph'}) "
                f"({n_params:,} params)")


# =============================================================================
# MolecularGraphEncoder (Tier 7 Molecular — Phase 4)
# =============================================================================

class MolecularGraphEncoder(BaseEncoder):
    """GNN encoder for molecular graphs (atom/bond level).

    Processes pre-computed molecular graphs (atoms as nodes, bonds as edges)
    using GNN message passing (GAT/GCN/GIN/GINE), then pools per-molecule
    representations via mean + sum pooling.

    GNN types:
    - GAT: Graph attention (multi-head, concatenated)
    - GCN: Graph convolution (symmetric normalization)
    - GIN: Graph isomorphism (sum aggregation, WL-equivalent)
    - GINE: GIN with edge/bond features (additive mechanism, WL-equivalent)
      Requires bond_feature_dim > 0 and edge_attr in graph data.

    The molecular graph data (PyG Batch) must be pre-computed from SMILES
    and stored via set_molecular_batch() before forward passes.

    Input x: [N, d_fp] fingerprints -- IGNORED (molecular graph data used)
    Output: [N_molecules, d_output] molecule-level embeddings

    NOTE: Currently supports full-batch mode only (all molecules processed
    at once). Mini-batch support would require index tracking to select
    the correct molecular subgraphs per batch.

    Requires: pip install torch_geometric rdkit
    """
    encoder_type = "molecular_graph"

    def __init__(self, d_input: int, d_output: int,
                 gnn_type: str = "gat", hidden_dim: int = 64,
                 num_layers: int = 2, atom_feature_dim: int = 32,
                 bond_feature_dim: int = 0, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        self._gnn_type = gnn_type
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._atom_feature_dim = atom_feature_dim
        self._bond_feature_dim = bond_feature_dim

        # Lazy import PyG — only needed when this encoder is actually created
        try:
            from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv
        except ImportError:
            raise ImportError(
                "MolecularGraphEncoder requires torch_geometric. "
                "Install with: pip install torch_geometric"
            )

        # Build GNN message-passing layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        d_in = atom_feature_dim
        for i in range(num_layers):
            d_out = hidden_dim
            if gnn_type == "gat":
                # GAT with 4 heads, concatenated -> 4*d_out, then project
                n_heads = 4
                self.gnn_layers.append(GATConv(
                    in_channels=d_in, out_channels=d_out // n_heads,
                    heads=n_heads, concat=True, dropout=dropout,
                ))
            elif gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(d_in, d_out))
            elif gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.ReLU(),
                    nn.Linear(d_out, d_out),
                )
                self.gnn_layers.append(GINConv(mlp))
            elif gnn_type == "gine":
                # GINEConv: GIN with edge/bond features added into messages.
                # Unlike GAT (attention bias), GINE adds edge features directly
                # into the message aggregation — proper WL-equivalent mechanism.
                # Requires bond_feature_dim > 0 and edge_attr in graph data.
                mlp = nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.ReLU(),
                    nn.Linear(d_out, d_out),
                )
                self.gnn_layers.append(GINEConv(mlp, edge_dim=bond_feature_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.norms.append(nn.LayerNorm(d_out))
            d_in = d_out

        # Pooling: mean + sum concatenation
        pool_output_dim = 2 * hidden_dim

        # Readout: pooling output -> d_output
        self.readout_proj = nn.Sequential(
            nn.Linear(pool_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_output),
        )
        self.output_norm = nn.LayerNorm(d_output)

        # Molecular batch data (set externally via set_molecular_batch)
        self._molecular_batch = None
        # Individual graphs for mini-batch sub-batching
        self._all_graphs = None
        # Current mini-batch molecule indices (set per step by trainer)
        self._current_mol_indices = None

    def set_molecular_batch(self, pyg_batch):
        """Store pre-batched molecular graph data (PyG Batch object).

        Used for full-batch mode (backward compatible). For mini-batch,
        also call set_molecular_graphs() with the individual graph list.

        Args:
            pyg_batch: torch_geometric.data.Batch with:
                - x: [total_atoms, atom_feature_dim] node features
                - edge_index: [2, total_edges] bond connectivity
                - batch: [total_atoms] mapping atoms -> molecule index
        """
        self._molecular_batch = pyg_batch

    def set_molecular_graphs(self, graphs_list):
        """Store individual molecular graphs for mini-batch sub-batching.

        When set, forward() can process only the molecules in the current
        mini-batch (via set_current_mol_indices) instead of all molecules.

        Args:
            graphs_list: List[torch_geometric.data.Data], one per molecule,
                         ordered to match [train, val, test] concatenation.
        """
        self._all_graphs = graphs_list

    def set_current_mol_indices(self, indices):
        """Set which molecule indices are in the current mini-batch.

        Called by the trainer before each forward pass when using
        mini-batch dataloaders that yield (x, y, mol_idx).

        Args:
            indices: Tensor of integer indices into the stored graphs list.
        """
        self._current_mol_indices = indices

    def _get_batch_for_forward(self, n_samples: int):
        """Get the PyG Batch to process in forward().

        Returns the mini-batch sub-batch if indices are set, otherwise
        falls back to the full molecular batch.
        """
        from torch_geometric.data import Batch as PyGBatch

        # Mini-batch mode: sub-batch from individual graphs
        if (self._current_mol_indices is not None
                and self._all_graphs is not None):
            idx_list = self._current_mol_indices.cpu().tolist()
            sub_graphs = [self._all_graphs[i] for i in idx_list]
            return PyGBatch.from_data_list(sub_graphs)

        # Full-batch mode (backward compatible)
        return self._molecular_batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process molecular graphs through GNN.

        Supports both full-batch and mini-batch modes:
        - Mini-batch: uses _current_mol_indices to extract sub-batch from
          _all_graphs. Returns [batch_size, d_output].
        - Full-batch: processes all molecules in _molecular_batch.
          Returns [N_total, d_output].

        Args:
            x: [N, d_fp] fingerprint features (IGNORED — graph data used)

        Returns:
            [N_molecules, d_output] molecule-level embeddings
        """
        batch = self._get_batch_for_forward(x.shape[0])
        if batch is None:
            # Fallback: return zeros (will be blended with old encoder in bridge)
            return torch.zeros(x.shape[0], self._d_output,
                               device=x.device, dtype=x.dtype)

        from torch_geometric.nn import global_mean_pool, global_add_pool

        # Ensure data is on the correct device
        device = next(self.parameters()).device
        h = batch.x.to(device)                         # [total_atoms, atom_feat_dim]
        edge_index = batch.edge_index.to(device)       # [2, total_edges]
        batch_idx = batch.batch.to(device)             # [total_atoms]

        # Extract edge_attr for GINEConv (bond features)
        edge_attr = getattr(batch, 'edge_attr', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        # GNN message passing
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            if self._gnn_type == "gine" and edge_attr is not None:
                h = gnn(h, edge_index, edge_attr=edge_attr)
            else:
                h = gnn(h, edge_index)
            h = norm(h)
            if i < len(self.gnn_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        # Per-molecule pooling: mean + sum concatenation
        h_mean = global_mean_pool(h, batch_idx)
        h_sum = global_add_pool(h, batch_idx)
        h_pool = torch.cat([h_mean, h_sum], dim=-1)

        # Project to d_output
        out = self.output_norm(self.readout_proj(h_pool))

        return out

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        has_batch = self._molecular_batch is not None or self._all_graphs is not None
        if self._molecular_batch is not None:
            n_mols = self._molecular_batch.num_graphs
        elif self._all_graphs is not None:
            n_mols = len(self._all_graphs)
        else:
            n_mols = 0
        mode = "mini-batch" if self._all_graphs is not None else "full-batch"
        info = f"loaded {n_mols} molecules, {mode}" if has_batch else "no data"
        bond_info = f", bond_dim={self._bond_feature_dim}" if self._bond_feature_dim > 0 else ""
        return (f"molecular_graph encoder: {self._gnn_type.upper()} "
                f"{self._num_layers}L x {self._hidden_dim}d -> {self._d_output}d "
                f"({info}{bond_info}) ({n_params:,} params)")

    def cost(self) -> float:
        c = sum(p.numel() for p in self.parameters()) * 2.0
        if self._molecular_batch is not None:
            # Add message passing cost estimate
            n_edges = self._molecular_batch.edge_index.shape[1]
            c += n_edges * self._hidden_dim * 2.0 * self._num_layers
        return c


class DualDrugCellEncoder(BaseEncoder):
    """Dual-encoder for drug response prediction: cell MLP + drug GNN.

    Combines cell line gene expression (passed as X) with molecular graph
    features (from stored PyG data) via concatenation and learned fusion.

    Input x: [N, d_cell] cell line features (e.g. PCA-reduced gene expression)
    Drug info: molecular graphs set via set_molecular_batch/graphs/indices
    Output: [N, d_output] fused drug-cell embeddings
    """
    encoder_type = "dual_drug_cell"

    def __init__(self, d_input: int, d_output: int,
                 cell_hidden: int = 256, cell_out: int = 64,
                 drug_out: int = 64,
                 gnn_type: str = "gine", gnn_hidden_dim: int = 64,
                 gnn_layers: int = 3, atom_feature_dim: int = 32,
                 bond_feature_dim: int = 12, dropout: float = 0.2,
                 **kwargs):
        super().__init__()
        self._d_input = d_input  # d_cell
        self._d_output = d_output

        # Cell line branch: MLP
        self.cell_mlp = nn.Sequential(
            nn.Linear(d_input, cell_hidden),
            nn.BatchNorm1d(cell_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cell_hidden, cell_out),
        )

        # Drug branch: reuse MolecularGraphEncoder
        self.drug_encoder = MolecularGraphEncoder(
            d_input=d_input,  # not used by MolecularGraphEncoder
            d_output=drug_out,
            gnn_type=gnn_type,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            dropout=dropout,
        )

        # Fusion: concat -> project
        fusion_in = cell_out + drug_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_output),
            nn.LayerNorm(d_output),
            nn.ReLU(),
        )

    # --- Delegate molecular data methods to drug_encoder ---
    def set_molecular_batch(self, pyg_batch):
        self.drug_encoder.set_molecular_batch(pyg_batch)

    def set_molecular_graphs(self, graphs_list):
        self.drug_encoder.set_molecular_graphs(graphs_list)

    def set_current_mol_indices(self, indices):
        self.drug_encoder.set_current_mol_indices(indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining cell features and drug graph.

        Args:
            x: [N, d_cell] cell line features

        Returns:
            [N, d_output] fused embeddings
        """
        # Cell branch
        cell_emb = self.cell_mlp(x)  # [N, cell_out]

        # Drug branch (ignores x, uses molecular graph data)
        drug_emb = self.drug_encoder(x)  # [N, drug_out]

        # Fuse
        combined = torch.cat([cell_emb, drug_emb], dim=-1)  # [N, cell_out+drug_out]
        return self.fusion(combined)  # [N, d_output]

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        drug_desc = self.drug_encoder.describe()
        return (f"dual_drug_cell encoder: cell MLP({self._d_input}->cell_out) + "
                f"{drug_desc} -> {self._d_output}d ({n_params:,} params)")

    def cost(self) -> float:
        return (sum(p.numel() for p in self.parameters()) * 2.0
                + self.drug_encoder.cost())


# =============================================================================
# GatedEncoderBridge (Phase 2: Encoder Switching)
# =============================================================================

class GatedEncoderBridge(BaseEncoder):
    """Blends old and new encoder during encoder switching.

    Same immunosuppression pattern as GatedOperation:
        output = (1 - alpha) * old_encoder(x) + alpha * proj(new_encoder(x))
    where alpha ramps from 0 -> 1 over warmup_epochs.

    If the new encoder has a different d_output than the old, a learned
    projection (nn.Linear) maps new_d -> old_d so the ASANN body layers
    continue to receive the expected input dimension.

    Handles spatial <-> flat encoder transitions: when one encoder outputs
    [B, C, H, W] (spatial) and the other outputs [B, D] (flat), the bridge
    flattens spatial outputs before projection and reshapes back to spatial
    format after projection if the old encoder was spatial.

    After warmup (alpha=1), the bridge is "absorbed" — replaced by the
    new encoder (plus projection if needed).
    """
    encoder_type = "bridge"

    def __init__(self, old_encoder: BaseEncoder, new_encoder: BaseEncoder,
                 warmup_epochs: int = 15):
        super().__init__()
        self.old_encoder = old_encoder
        self.new_encoder = new_encoder
        self.warmup_epochs = max(1, warmup_epochs)
        self._current_epoch: int = 0
        self._absorbed: bool = False

        # Spatial format handling for blending spatial <-> flat encoders.
        # When old encoder is spatial, we need to reshape projected flat output
        # back to [B, C, H, W] for compatible blending and body layer input.
        self._old_is_spatial = old_encoder.is_spatial
        self._new_is_spatial = getattr(new_encoder, 'is_spatial', False)
        self._spatial_shape: Optional[Tuple[int, ...]] = None
        if self._old_is_spatial and hasattr(old_encoder, 'spatial_shape'):
            self._spatial_shape = old_encoder.spatial_shape  # (C, H, W)

        # Dimension alignment: if new encoder outputs different dim, project.
        # d_output is the flat-equivalent dimension (C*H*W for spatial).
        old_d = old_encoder.d_output
        new_d = new_encoder.d_output
        if old_d != new_d:
            self.proj = nn.Linear(new_d, old_d)
        else:
            self.proj = None

    def _normalize_new_output(self, new_out: torch.Tensor,
                              old_out: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """Flatten spatial new_out, apply projection, reshape to match old format.

        This handles the spatial <-> flat transition:
        1. If new_out is 4D (spatial), flatten to [B, C*H*W] for projection
        2. Apply projection (nn.Linear) if dimensions differ
        3. If old encoder was spatial, reshape back to [B, C, H, W]
        """
        # Flatten 4D spatial output to 2D for projection
        if new_out.dim() == 4:
            new_out = new_out.flatten(1)  # [B, C*H*W]

        if self.proj is not None:
            new_out = self.proj(new_out)  # [B, old_d]

        # Reshape to match old_out's spatial format if needed
        if old_out is not None and old_out.dim() == 4 and new_out.dim() == 2:
            new_out = new_out.view_as(old_out)
        elif self._old_is_spatial and self._spatial_shape and new_out.dim() == 2:
            B = new_out.shape[0]
            new_out = new_out.view(B, *self._spatial_shape)

        return new_out

    @property
    def alpha(self) -> float:
        """Current blending factor: 0 = pure old, 1 = pure new."""
        if self._absorbed:
            return 1.0
        return min(1.0, self._current_epoch / self.warmup_epochs)

    def advance_epoch(self):
        """Call once per epoch to advance the gate."""
        self._current_epoch += 1

    @property
    def is_ready_to_absorb(self) -> bool:
        """True when warmup complete and bridge can be replaced."""
        return self._current_epoch >= self.warmup_epochs and not self._absorbed

    def absorb(self) -> BaseEncoder:
        """Return the new encoder (with projection wrapper if needed).

        After absorption, the model should replace self.encoder with the
        returned encoder. If projection exists, it's wrapped into the
        new encoder as a post-projection. If old encoder was spatial,
        the spatial shape is preserved in the ProjectedEncoder so the
        model body continues to receive [B, C, H, W] tensors.
        """
        self._absorbed = True
        # Determine spatial output shape for ProjectedEncoder
        spatial_shape = self._spatial_shape if self._old_is_spatial else None

        if self.proj is not None or (self._old_is_spatial != self._new_is_spatial):
            # Need projection and/or spatial reshape
            return ProjectedEncoder(self.new_encoder, self.proj,
                                    output_spatial_shape=spatial_shape)
        return self.new_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._absorbed:
            return self._normalize_new_output(self.new_encoder(x))

        a = self.alpha
        old_out = self.old_encoder(x)

        if a <= 0.0:
            return old_out  # First epoch: pure old encoder

        new_out = self._normalize_new_output(self.new_encoder(x), old_out)

        if a >= 1.0:
            return new_out

        # Blend: shapes are guaranteed to match after normalization
        return (1.0 - a) * old_out + a * new_out

    @property
    def d_output(self) -> int:
        # Bridge always outputs old encoder's dimension (projection handles mismatch)
        return self.old_encoder.d_output

    @property
    def d_input(self) -> int:
        return self.old_encoder.d_input

    # Backward compatibility: delegate to old encoder during bridge
    @property
    def out_features(self) -> int:
        return self.old_encoder.d_output

    @property
    def in_features(self) -> int:
        return self.old_encoder.d_input

    @property
    def is_spatial(self) -> bool:
        return self.old_encoder.is_spatial

    @property
    def out_channels(self) -> int:
        """Channel count — delegates to old encoder for spatial models."""
        if hasattr(self.old_encoder, 'out_channels'):
            return self.old_encoder.out_channels
        return self.d_output

    @property
    def spatial_shape(self) -> Optional[Tuple[int, ...]]:
        """Spatial output shape (C, H, W) if spatial, else None."""
        return self._spatial_shape

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"bridge encoder: {self.old_encoder.encoder_type} -> "
                f"{self.new_encoder.encoder_type} "
                f"(alpha={self.alpha:.2f}, {n_params:,} params)")

    def cost(self) -> float:
        """Cost = old + new + projection (during bridge, both are active)."""
        c = self.old_encoder.cost() + self.new_encoder.cost()
        if self.proj is not None:
            c += self.proj.in_features * self.proj.out_features * 2.0
        return c


class ProjectedEncoder(BaseEncoder):
    """Wrapper that applies a learned projection after an encoder.

    Used after GatedEncoderBridge absorption when the new encoder has
    a different d_output than the old encoder. The projection maps
    new_d -> old_d permanently.

    Handles spatial <-> flat transitions: if the wrapped encoder is spatial
    (outputs 4D), its output is flattened before projection. If
    output_spatial_shape is set, the final output is reshaped to
    [B, C, H, W] to preserve spatial format for downstream body layers.
    """
    encoder_type = "projected"

    def __init__(self, encoder: BaseEncoder, proj: Optional[nn.Linear],
                 output_spatial_shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self._encoder = encoder
        self._proj = proj
        # (C, H, W) tuple — if set, reshape output to [B, C, H, W]
        self._output_spatial_shape = output_spatial_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._encoder(x)

        # Flatten 4D spatial output before projection
        if out.dim() == 4:
            out = out.flatten(1)  # [B, C*H*W]

        if self._proj is not None:
            out = self._proj(out)

        # Reshape to spatial format if needed
        if self._output_spatial_shape is not None and out.dim() == 2:
            B = out.shape[0]
            out = out.view(B, *self._output_spatial_shape)

        return out

    @property
    def d_output(self) -> int:
        if self._proj is not None:
            return self._proj.out_features
        return self._encoder.d_output

    @property
    def d_input(self) -> int:
        return self._encoder.d_input

    @property
    def is_spatial(self) -> bool:
        if self._output_spatial_shape is not None:
            return True
        return self._encoder.is_spatial

    # Delegate common attributes
    @property
    def out_features(self) -> int:
        if self._proj is not None:
            return self._proj.out_features
        return self._encoder.d_output

    @property
    def in_features(self) -> int:
        return self._encoder.d_input

    @property
    def out_channels(self) -> int:
        """Channel count for spatial mode."""
        if self._output_spatial_shape is not None:
            return self._output_spatial_shape[0]
        if hasattr(self._encoder, 'out_channels'):
            return self._encoder.out_channels
        return self.d_output

    @property
    def spatial_shape(self) -> Optional[Tuple[int, ...]]:
        """Spatial output shape (C, H, W) if spatial, else None."""
        if self._output_spatial_shape is not None:
            return self._output_spatial_shape
        if hasattr(self._encoder, 'spatial_shape'):
            return self._encoder.spatial_shape
        return None

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        proj_info = ""
        if self._proj is not None:
            proj_info = f" -> {self._proj.out_features}"
        spatial_info = ""
        if self._output_spatial_shape is not None:
            C, H, W = self._output_spatial_shape
            spatial_info = f" -> [{C},{H},{W}]"
        return (f"projected({self._encoder.encoder_type}): "
                f"{self._encoder.d_output}{proj_info}{spatial_info} "
                f"({n_params:,} params)")

    def cost(self) -> float:
        c = self._encoder.cost()
        if self._proj is not None:
            c += self._proj.in_features * self._proj.out_features * 2.0
        return c


# =============================================================================
# TransformerEncoder — Chunked self-attention for high-dim features
# =============================================================================

class TransformerEncoder(BaseEncoder):
    """Transformer encoder: chunks flat features into tokens, applies MHA+FFN.

    Architecture:
        [N, d_input] -> chunk into [N, n_patches, chunk_size]
        -> Linear(chunk_size, d_model) projection
        -> + learnable positional embeddings
        -> L x TransformerEncoderLayer (pre-norm, GELU, batch_first)
        -> LayerNorm -> mean pool over tokens -> Linear(d_model, d_output)

    Works on ANY flat input. For small d_input, uses fewer/smaller chunks.
    Pads d_input to the nearest multiple of chunk_size if needed.

    Args:
        d_input: Raw input feature dimension
        d_output: Output dimension (for ASANN body)
        d_model: Transformer model dimension (default: 64)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of TransformerEncoderLayer stacks (default: 2)
        dim_feedforward: FFN inner dimension (default: 4 * d_model)
        dropout: Dropout rate (default: 0.1)
        chunk_size: Number of raw features per token (default: 16)
    """
    encoder_type = "transformer"

    def __init__(self, d_input: int, d_output: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = None, dropout: float = 0.1,
                 chunk_size: int = 16):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        self._d_model = d_model
        self._chunk_size = chunk_size
        self._nhead = nhead
        self._num_layers = num_layers

        # Compute number of patches (pad d_input to multiple of chunk_size)
        self._n_patches = math.ceil(d_input / chunk_size)
        self._padded_dim = self._n_patches * chunk_size

        # Patch projection: chunk_size -> d_model
        self.patch_proj = nn.Linear(chunk_size, d_model)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self._n_patches, d_model) * 0.02
        )

        # Transformer encoder layers (pre-norm for stability)
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm: more stable for mid-training switching
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final: LayerNorm + projection to d_output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_output)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for safe encoder switching."""
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: chunk -> project -> attend -> pool -> output.

        Args:
            x: [N, d_input] flat feature tensor

        Returns:
            [N, d_output] encoded features
        """
        N = x.shape[0]

        # Pad if d_input is not a multiple of chunk_size
        if self._padded_dim > self._d_input:
            x = F.pad(x, (0, self._padded_dim - self._d_input))

        # Reshape into patches: [N, n_patches, chunk_size]
        x = x.view(N, self._n_patches, self._chunk_size)

        # Project patches to d_model and add positional embeddings
        x = self.patch_proj(x) + self.pos_embed  # [N, n_patches, d_model]

        # Apply transformer layers
        x = self.transformer(x)  # [N, n_patches, d_model]

        # Final norm + mean pool over tokens
        x = self.final_norm(x)  # [N, n_patches, d_model]
        x = x.mean(dim=1)       # [N, d_model]

        # Project to output dimension
        return self.output_proj(x)  # [N, d_output]

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"transformer encoder: {self._d_input} -> {self._d_output} "
                f"(chunks={self._n_patches}x{self._chunk_size}, "
                f"d_model={self._d_model}, heads={self._nhead}, "
                f"layers={self._num_layers}, {n_params:,} params)")

    def cost(self) -> float:
        """Approximate FLOPs: patch_proj + self-attention + FFN + output_proj."""
        S = self._n_patches
        D = self._d_model
        L = self._num_layers
        # Patch projection
        c = S * self._chunk_size * D * 2.0
        # Per layer: attention (Q,K,V projections + attn matmul + output proj) + FFN
        attn_cost = S * D * D * 4 * 2.0 + S * S * D * 2.0  # QKV+out proj + attn
        ffn_cost = S * D * (4 * D) * 2.0 * 2  # 2 linear layers in FFN
        c += L * (attn_cost + ffn_cost)
        # Output projection
        c += D * self._d_output * 2.0
        return c


# =============================================================================
# AutoencoderEncoder (Bottleneck feature learning with reconstruction)
# =============================================================================

class AutoencoderEncoder(BaseEncoder):
    """Autoencoder encoder: learns compressed representations via reconstruction.

    Architecture:
        Encoder: d_input -> d_hidden -> d_output (bottleneck)
        Decoder: d_output -> d_hidden -> d_input (reconstruction)

    During forward(), returns the bottleneck representation [N, d_output].
    Simultaneously computes reconstruction loss and stores it in
    `self._recon_loss` for the trainer to pick up as auxiliary loss.

    Inspired by Dong et al. (2025) Adaptive Toeplitz Convolutional Autoencoder
    which achieves 99.6% on ECG5000 by learning compressed representations.

    Args:
        d_input: Raw input feature dimension
        d_output: Bottleneck dimension (output for ASANN body)
        d_hidden: Hidden layer dimension (default: 4 * d_output)
        num_layers: Number of encoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        recon_weight: Weight for reconstruction loss (default: 0.1)
    """
    encoder_type = "autoencoder"

    def __init__(self, d_input: int, d_output: int, d_hidden: int = None,
                 num_layers: int = 2, dropout: float = 0.1,
                 recon_weight: float = 0.1, **kwargs):
        super().__init__()
        self._d_input = d_input
        self._d_output = d_output
        self._recon_weight = recon_weight

        if d_hidden is None:
            d_hidden = max(4 * d_output, d_input // 2)
        self._d_hidden = d_hidden

        # Encoder: d_input -> d_hidden -> ... -> d_output
        enc_layers = []
        d_in = d_input
        for i in range(num_layers):
            d_out = d_hidden if i < num_layers - 1 else d_output
            enc_layers.append(nn.Linear(d_in, d_out))
            enc_layers.append(nn.LayerNorm(d_out))
            enc_layers.append(nn.GELU())
            if dropout > 0 and i < num_layers - 1:
                enc_layers.append(nn.Dropout(dropout))
            d_in = d_out
        self.enc = nn.Sequential(*enc_layers)

        # Decoder: d_output -> d_hidden -> ... -> d_input (symmetric)
        dec_layers = []
        d_in = d_output
        for i in range(num_layers):
            d_out = d_hidden if i < num_layers - 1 else d_input
            dec_layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                dec_layers.append(nn.LayerNorm(d_out))
                dec_layers.append(nn.GELU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
            d_in = d_out
        self.dec = nn.Sequential(*dec_layers)

        # Stored reconstruction loss for trainer to pick up
        self._recon_loss = None

        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier for stable encoder switching."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input and compute reconstruction loss.

        Args:
            x: [N, d_input] flat feature tensor

        Returns:
            [N, d_output] bottleneck features
        """
        if x.dim() == 4:
            x = x.flatten(1)

        # Encode to bottleneck
        z = self.enc(x)

        # Decode and compute reconstruction loss (only during training)
        if self.training:
            x_recon = self.dec(z)
            self._recon_loss = F.mse_loss(x_recon, x) * self._recon_weight
        else:
            self._recon_loss = None

        return z

    def get_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Return stored reconstruction loss (called by trainer after forward)."""
        loss = self._recon_loss
        self._recon_loss = None  # Clear after reading
        return loss

    @property
    def d_output(self) -> int:
        return self._d_output

    @property
    def d_input(self) -> int:
        return self._d_input

    def describe(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"autoencoder encoder: {self._d_input} -> "
                f"{self._d_hidden} -> {self._d_output} "
                f"(recon_w={self._recon_weight}, {n_params:,} params)")

    def cost(self) -> float:
        """Approximate FLOPs: encoder + decoder."""
        return sum(p.numel() for p in self.parameters()) * 2.0


# =============================================================================
# Encoder Registry & Factory
# =============================================================================

ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    "linear": LinearEncoder,
    "conv": ConvEncoder,
    "fourier": FourierEncoder,
    "patch_embed": PatchEmbedEncoder,
    "temporal": TemporalEncoder,
    "graph_node": GraphNodeEncoder,
    "temporal_graph": TemporalGraphEncoder,
    "molecular_graph": MolecularGraphEncoder,
    "dual_drug_cell": DualDrugCellEncoder,
    "transformer": TransformerEncoder,
    "autoencoder": AutoencoderEncoder,
}


def create_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    """Factory function for creating encoders by name.

    Args:
        encoder_type: Encoder type string (must be in ENCODER_REGISTRY)
        **kwargs: Encoder-specific keyword arguments (d_input, d_output, etc.)

    Returns:
        Instantiated encoder module
    """
    if encoder_type not in ENCODER_REGISTRY:
        available = ', '.join(ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder type '{encoder_type}'. "
            f"Available: {available}"
        )
    cls = ENCODER_REGISTRY[encoder_type]
    return cls(**kwargs)


def build_encoder_kwargs(encoder_type: str, d_input: int, d_output: int,
                         config=None, model=None) -> dict:
    """Build keyword arguments for encoder construction from config.

    Each encoder type needs different kwargs. This function extracts the
    relevant config fields and model attributes.

    Args:
        encoder_type: Encoder type string
        d_input: Raw input dimension
        d_output: Desired output dimension (usually d_init)
        config: ASANNConfig instance (for encoder-specific settings)
        model: ASANNModel instance (for graph data, etc.)

    Returns:
        dict of kwargs for create_encoder()
    """
    kwargs = {'d_input': d_input, 'd_output': d_output}

    if encoder_type == "conv" and config is not None:
        if config.spatial_shape is not None:
            C_in, H, W = config.spatial_shape
            kwargs = {
                'C_in': C_in,
                'C_stem': config.c_stem_init,
                'H': H,
                'W': W,
            }

    elif encoder_type == "fourier" and config is not None:
        kwargs['num_frequencies'] = getattr(config, 'encoder_fourier_frequencies', 64)
        kwargs['sigma'] = getattr(config, 'encoder_fourier_sigma', 10.0)

    elif encoder_type == "patch_embed" and config is not None:
        if config.spatial_shape is not None:
            C_in, H, W = config.spatial_shape
            kwargs['C_in'] = C_in
            kwargs['H'] = H
            kwargs['W'] = W
            kwargs['patch_size'] = getattr(config, 'encoder_patch_size', 4)

    elif encoder_type == "temporal" and config is not None:
        n_feat = getattr(config, 'encoder_temporal_n_features', None)
        win_size = getattr(config, 'encoder_temporal_window_size', None)

        # Infer window_size and n_features from d_input when not explicitly set.
        # d_input = window_size * n_features, so if one is known we derive the other.
        if n_feat is None and win_size is None:
            # Neither set — assume univariate: n_features=1, window_size=d_input
            n_feat = 1
            win_size = d_input
        elif n_feat is not None and win_size is None:
            win_size = d_input // n_feat
        elif win_size is not None and n_feat is None:
            n_feat = d_input // win_size

        kwargs['n_features'] = n_feat
        kwargs['window_size'] = win_size
        kwargs['patch_size'] = getattr(config, 'encoder_temporal_patch_size', 8)

    elif encoder_type == "graph_node" and config is not None:
        kwargs['num_gnn_layers'] = getattr(config, 'encoder_gnn_layers', 2)
        if model is not None:
            adj = getattr(model, '_graph_adj_sparse', None)
            degree = getattr(model, '_graph_degree', None)
            if adj is not None:
                kwargs['adj_sparse'] = adj
            if degree is not None:
                kwargs['degree'] = degree

    elif encoder_type == "temporal_graph" and config is not None:
        kwargs['num_gnn_layers'] = getattr(config, 'encoder_gnn_layers', 1)
        if model is not None:
            adj = getattr(model, '_graph_adj_sparse', None)
            degree = getattr(model, '_graph_degree', None)
            if adj is not None:
                kwargs['adj_sparse'] = adj
            if degree is not None:
                kwargs['degree'] = degree

    elif encoder_type == "molecular_graph" and config is not None:
        kwargs['gnn_type'] = getattr(config, 'encoder_mol_gnn_type', 'gat')
        kwargs['hidden_dim'] = getattr(config, 'encoder_mol_hidden_dim', 64)
        kwargs['num_layers'] = getattr(config, 'encoder_gnn_layers', 2)
        # Get atom feature dim: prefer live data from model, fall back to config
        atom_dim_found = False
        bond_dim_found = False
        if model is not None:
            mol_batch = getattr(model, '_molecular_batch', None)
            if mol_batch is not None and mol_batch.x is not None:
                kwargs['atom_feature_dim'] = mol_batch.x.shape[1]
                atom_dim_found = True
                # Detect bond feature dim from live data
                edge_attr = getattr(mol_batch, 'edge_attr', None)
                if edge_attr is not None and edge_attr.shape[0] > 0:
                    kwargs['bond_feature_dim'] = edge_attr.shape[1]
                    bond_dim_found = True
            elif not atom_dim_found:
                all_graphs = getattr(model, '_all_molecular_graphs', None)
                if all_graphs is not None and len(all_graphs) > 0:
                    kwargs['atom_feature_dim'] = all_graphs[0].x.shape[1]
                    atom_dim_found = True
                    # Detect bond feature dim from first graph
                    edge_attr = getattr(all_graphs[0], 'edge_attr', None)
                    if edge_attr is not None and edge_attr.shape[0] > 0:
                        kwargs['bond_feature_dim'] = edge_attr.shape[1]
                        bond_dim_found = True
        if not atom_dim_found:
            kwargs['atom_feature_dim'] = getattr(config, 'encoder_mol_atom_feature_dim', 32)
        if not bond_dim_found:
            kwargs['bond_feature_dim'] = getattr(config, 'encoder_mol_bond_feature_dim', 0)

    elif encoder_type == "dual_drug_cell" and config is not None:
        kwargs['cell_hidden'] = getattr(config, 'dual_encoder_cell_hidden', 256)
        kwargs['cell_out'] = getattr(config, 'dual_encoder_cell_out', 64)
        kwargs['drug_out'] = getattr(config, 'dual_encoder_drug_out', 64)
        kwargs['gnn_type'] = getattr(config, 'encoder_mol_gnn_type', 'gine')
        kwargs['gnn_hidden_dim'] = getattr(config, 'encoder_mol_hidden_dim', 64)
        kwargs['gnn_layers'] = getattr(config, 'encoder_gnn_layers', 3)
        # Get atom/bond feature dims from model data
        atom_dim_found = False
        bond_dim_found = False
        if model is not None:
            mol_batch = getattr(model, '_molecular_batch', None)
            if mol_batch is not None and mol_batch.x is not None:
                kwargs['atom_feature_dim'] = mol_batch.x.shape[1]
                atom_dim_found = True
                edge_attr = getattr(mol_batch, 'edge_attr', None)
                if edge_attr is not None and edge_attr.shape[0] > 0:
                    kwargs['bond_feature_dim'] = edge_attr.shape[1]
                    bond_dim_found = True
            elif not atom_dim_found:
                all_graphs = getattr(model, '_all_molecular_graphs', None)
                if all_graphs is not None and len(all_graphs) > 0:
                    kwargs['atom_feature_dim'] = all_graphs[0].x.shape[1]
                    atom_dim_found = True
                    edge_attr = getattr(all_graphs[0], 'edge_attr', None)
                    if edge_attr is not None and edge_attr.shape[0] > 0:
                        kwargs['bond_feature_dim'] = edge_attr.shape[1]
                        bond_dim_found = True
        if not atom_dim_found:
            kwargs['atom_feature_dim'] = getattr(config, 'encoder_mol_atom_feature_dim', 32)
        if not bond_dim_found:
            kwargs['bond_feature_dim'] = getattr(config, 'encoder_mol_bond_feature_dim', 12)

    elif encoder_type == "transformer" and config is not None:
        kwargs['d_model'] = getattr(config, 'encoder_transformer_d_model', 64)
        kwargs['nhead'] = getattr(config, 'encoder_transformer_nhead', 4)
        kwargs['num_layers'] = getattr(config, 'encoder_transformer_layers', 2)
        kwargs['dim_feedforward'] = getattr(config, 'encoder_transformer_ff_dim', None)
        kwargs['dropout'] = getattr(config, 'encoder_transformer_dropout', 0.1)
        kwargs['chunk_size'] = getattr(config, 'encoder_transformer_chunk_size', 16)

    elif encoder_type == "autoencoder" and config is not None:
        kwargs['d_hidden'] = getattr(config, 'encoder_ae_d_hidden', None)
        kwargs['num_layers'] = getattr(config, 'encoder_ae_layers', 2)
        kwargs['dropout'] = getattr(config, 'encoder_ae_dropout', 0.1)
        kwargs['recon_weight'] = getattr(config, 'encoder_ae_recon_weight', 0.1)

    return kwargs
