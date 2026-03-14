import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from .config import ASANNConfig
from .encoders import (BaseEncoder, LinearEncoder, ConvEncoder,
                       GatedEncoderBridge, ProjectedEncoder,
                       create_encoder, build_encoder_kwargs, ENCODER_REGISTRY)


# Operations that do NOT contribute real computational capacity to a layer.
# A layer with only trivial ops is "unenriched" -- it has depth but no
# discovered intelligence.  Used by count_unenriched_layers(),
# _prune_useless_layers(), and op-removal safeguards.
_TRIVIAL_OPS = frozenset({
    # Activations
    'Identity', 'ReLU', 'LeakyReLU', 'GELU', 'ELU', 'Mish',
    'SiLU', 'PReLU', 'Tanh', 'Softplus',
    # Normalization
    'BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'GroupNorm',
    # Regularization
    'Dropout', 'ActivationNoise',
})


class DropPath(nn.Module):
    """Stochastic Depth / DropPath regularization (Huang et al., 2016).

    During training, randomly drops the entire residual branch with probability
    `drop_prob`, leaving only the identity shortcut. During eval, this is a no-op.

    Applied per-sample: each sample in the batch independently drops/keeps.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask: [B, 1, 1, 1] for 4D or [B, 1] for 2D
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        # Scale by 1/keep_prob so expected value is unchanged
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


if TYPE_CHECKING:
    from .surgery import (Conv1dBlock, Conv2dBlock, FactoredEmbedding, MLPEmbedding,
                          GeometricEmbedding, PositionalEmbedding,
                          SelfAttentionOp, MultiHeadAttentionOp,
                          CrossAttentionOp, CausalAttentionOp,
                          SpatialConv2dOp, ChannelAttentionOp,
                          DilatedConv1dBlock, EMASmooth,
                          GatedLinearUnit, TemporalDiff,
                          GRUOp, ActivationNoise)


class GatedOperation(nn.Module):
    """Immunosuppression wrapper: gradually introduces a new operation.

    When ASANN performs mid-training surgery (inserting BatchNorm, JK projections,
    skip connections, etc.), inserting immediately disrupts the learned activation
    distribution and causes model collapse — analogous to organ transplant rejection.

    GatedOperation wraps the new operation with a blending gate:
        output = (1 - alpha) * x + alpha * op(x)

    The gate starts at alpha=0 (pure identity, operation has NO effect) and linearly
    ramps to alpha=1 over `warmup_epochs` epochs.  This gives the model time to
    adapt its weights to the new activation distribution gradually.

    After warmup completes (alpha=1), the wrapper can be "absorbed" — replaced
    by the raw operation — to eliminate any runtime overhead.
    """

    def __init__(self, operation: nn.Module, warmup_epochs: int = 10):
        super().__init__()
        self.operation = operation
        self.warmup_epochs = max(1, warmup_epochs)
        self._current_epoch: int = 0  # Epochs since insertion
        self._absorbed: bool = False   # True after gate reaches 1.0

    @property
    def alpha(self) -> float:
        """Current blending factor: 0 = pure identity, 1 = pure operation."""
        if self._absorbed:
            return 1.0
        return min(1.0, self._current_epoch / self.warmup_epochs)

    def advance_epoch(self):
        """Call once per epoch to advance the gate."""
        self._current_epoch += 1

    @property
    def is_ready_to_absorb(self) -> bool:
        """True when warmup is complete and wrapper can be replaced by raw op."""
        return self._current_epoch >= self.warmup_epochs and not self._absorbed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._absorbed:
            return self.operation(x)
        a = self.alpha
        if a <= 0.0:
            return x  # First epoch: pure identity, op has zero effect
        if a >= 1.0:
            return self.operation(x)
        # Blend: smooth transition from identity to operation
        return (1.0 - a) * x + a * self.operation(x)

    def absorb(self) -> nn.Module:
        """Return the raw operation for permanent replacement."""
        self._absorbed = True
        return self.operation

    def extra_repr(self) -> str:
        return (f"alpha={self.alpha:.2f}, epoch={self._current_epoch}/"
                f"{self.warmup_epochs}, absorbed={self._absorbed}")


class OperationPipeline(nn.Module):
    """A sequential pipeline of operations for a single layer.

    Unlike DARTS which averages operations with softmax weights,
    this is a real sequential composition: op1 -> op2 -> op3.
    The pipeline can be modified by adding/removing operations at any position.
    """

    def __init__(self):
        super().__init__()
        self.operations = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.operations:
            x = op(x)
        return x

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    def add_operation(self, operation: nn.Module, position: int,
                      gated: bool = False, warmup_epochs: int = 10):
        """Insert an operation at a specific position in the pipeline.

        Args:
            operation: The nn.Module to insert.
            position: Index in the pipeline (0 = first).
            gated: If True, wrap in GatedOperation for gradual introduction
                   (immunosuppression). The operation starts with zero effect
                   and linearly ramps to full effect over warmup_epochs.
            warmup_epochs: Number of epochs for the gate to ramp from 0→1.
        """
        if gated:
            operation = GatedOperation(operation, warmup_epochs=warmup_epochs)
        ops_list = list(self.operations)
        ops_list.insert(position, operation)
        self.operations = nn.ModuleList(ops_list)

    def remove_operation(self, position: int):
        """Remove an operation at a specific position."""
        ops_list = list(self.operations)
        ops_list.pop(position)
        self.operations = nn.ModuleList(ops_list)

    def advance_gates(self):
        """Advance all GatedOperation gates by one epoch."""
        for op in self.operations:
            if isinstance(op, GatedOperation):
                op.advance_epoch()

    def absorb_ready_gates(self) -> bool:
        """Replace completed GatedOperations with their raw operations.

        Returns:
            True if any gates were absorbed (architecture changed).
        """
        ops_list = list(self.operations)
        changed = False
        for i, op in enumerate(ops_list):
            if isinstance(op, GatedOperation) and op.is_ready_to_absorb:
                ops_list[i] = op.absorb()
                changed = True
        if changed:
            self.operations = nn.ModuleList(ops_list)
        return changed

    def describe(self) -> List[str]:
        """Return a human-readable description of the pipeline."""
        if self.num_operations == 0:
            return ["Identity"]
        from .surgery import get_operation_name
        result = []
        for op in self.operations:
            # Unwrap GatedOperation to describe the inner operation
            gate_prefix = ""
            if isinstance(op, GatedOperation):
                gate_prefix = f"[g{op.alpha:.0%}]"
                op = op.operation  # Describe the inner op
            # Use descriptive names where attributes are available
            op_name = get_operation_name(op)
            cls_name = type(op).__name__
            if 'conv1d_dilated' in op_name:
                result.append(f"DilConv1d(k={op.kernel_size},d={op.dilation})")
            elif hasattr(op, 'kernel_size') and 'conv1d' in op_name:
                result.append(f"Conv1d(k={op.kernel_size})")
            elif hasattr(op, 'kernel_size') and 'conv2d_k' in op_name:
                result.append(f"Conv2d(k={op.kernel_size},{op.C}x{op.H}x{op.W})")
            elif op_name == "embed_factored":
                result.append(f"FactEmbed(r={op.rank})")
            elif op_name == "embed_mlp":
                result.append(f"MLPEmbed(h={op.hidden})")
            elif op_name == "embed_geometric":
                result.append("GeoEmbed")
            elif op_name == "embed_positional":
                result.append("PosEmbed")
            elif op_name == "attn_self":
                result.append(f"SelfAttn(r={op.rank})")
            elif op_name == "attn_multihead":
                result.append(f"MHAttn(h={op.num_heads})")
            elif op_name == "attn_cross":
                result.append(f"CrossAttn(m={op.num_memories})")
            elif op_name == "attn_causal":
                result.append(f"CausalAttn(r={op.rank})")
            elif op_name == "ema_smooth":
                result.append(f"EMA(d={op.d})")
            elif op_name == "gated_linear_unit":
                result.append(f"GLU(d={op.d})")
            elif op_name == "temporal_diff":
                result.append(f"TempDiff(d={op.d})")
            elif 'derivative_conv1d' in op_name:
                result.append(f"DerConv1d(o{op.order})")
            elif 'polynomial' in op_name:
                result.append(f"Poly(deg={op.degree})")
            elif 'branched' in op_name and 'graph' not in op_name:
                result.append(f"Branch({op.num_branches})")
            # Graph operations
            elif op_name == "graph_neighbor_agg":
                result.append(f"NeighborAgg(d={op.d})")
            elif op_name == "graph_attention_agg":
                result.append(f"GraphAttn(d={op.d})")
            elif op_name == "graph_diffusion_k3":
                result.append(f"GDiffusion(d={op.d})")
            elif op_name == "graph_spectral_conv":
                result.append(f"SpectConv(d={op.d})")
            elif op_name == "graph_gin":
                result.append(f"GIN(d={op.d})")
            elif op_name == "graph_degree_scale":
                result.append(f"DegScale(d={op.d})")
            elif op_name == "graph_pairnorm":
                result.append("PairNorm")
            elif op_name == "graph_graphnorm":
                result.append("GraphNorm")
            elif op_name == "graph_positional_enc":
                result.append(f"GraphPE(d={op.d})")
            elif op_name == "graph_branched_agg":
                result.append(f"GraphBranch(d={op.d})")
            elif op_name == "graph_appnp":
                result.append(f"APPNP(d={op.d})")
            elif op_name == "graph_sage_mean":
                result.append(f"SAGEmean(d={op.d})")
            elif op_name == "graph_sage_gcn":
                result.append(f"SAGEgcn(d={op.d})")
            elif op_name == "graph_gatv2":
                result.append(f"GATv2(d={op.d})")
            elif op_name == "graph_sgc":
                result.append(f"SGC(d={op.d})")
            elif op_name == "graph_dropedge":
                result.append(f"DropEdge(d={op.d})")
            elif op_name == "graph_mixhop":
                result.append(f"MixHop(d={op.d})")
            elif op_name == "graph_virtual_node":
                result.append(f"VirtNode(d={op.d})")
            elif op_name == "graph_edge_weighted_agg":
                result.append(f"EdgeWtAgg(d={op.d})")
            elif op_name == "gru":
                result.append(f"GRU({op.num_chunks}x{op.chunk_size})")
            elif op_name == "activation_noise":
                result.append("ActNoise")
            else:
                result.append(cls_name)
            # Prepend gate indicator if this op is being gradually introduced
            if gate_prefix and result:
                result[-1] = gate_prefix + result[-1]
        return result


class SpatialOperationPipeline(nn.Module):
    """A sequential pipeline of operations for spatial layers.

    Operates on [B, C, H, W] tensors natively. Same interface as OperationPipeline
    but for 4D spatial tensors.
    """

    def __init__(self):
        super().__init__()
        self.operations = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.operations:
            x = op(x)
        return x

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    def add_operation(self, operation: nn.Module, position: int,
                      gated: bool = False, warmup_epochs: int = 10):
        """Insert an operation at a specific position in the pipeline."""
        if gated:
            operation = GatedOperation(operation, warmup_epochs=warmup_epochs)
        ops_list = list(self.operations)
        ops_list.insert(position, operation)
        self.operations = nn.ModuleList(ops_list)

    def remove_operation(self, position: int):
        """Remove an operation at a specific position."""
        ops_list = list(self.operations)
        ops_list.pop(position)
        self.operations = nn.ModuleList(ops_list)

    def advance_gates(self):
        """Advance all GatedOperation gates by one epoch."""
        for op in self.operations:
            if isinstance(op, GatedOperation):
                op.advance_epoch()

    def absorb_ready_gates(self) -> bool:
        """Replace completed GatedOperations with their raw operations.

        Returns:
            True if any gates were absorbed (architecture changed).
        """
        ops_list = list(self.operations)
        changed = False
        for i, op in enumerate(ops_list):
            if isinstance(op, GatedOperation) and op.is_ready_to_absorb:
                ops_list[i] = op.absorb()
                changed = True
        if changed:
            self.operations = nn.ModuleList(ops_list)
        return changed

    def describe(self) -> List[str]:
        """Return a human-readable description of the pipeline."""
        if self.num_operations == 0:
            return ["Identity"]
        from .surgery import get_operation_name
        result = []
        for op in self.operations:
            # Unwrap GatedOperation for describe
            gate_prefix = ""
            if isinstance(op, GatedOperation):
                gate_prefix = f"[g{op.alpha:.0%}]"
                op = op.operation
            op_name = get_operation_name(op)
            desc = None
            if 'spatial_dw_sep' in op_name:
                desc = f"DWSep(k={op.kernel_size})"
            elif 'spatial_conv2d' in op_name:
                desc = f"SpatConv2d(k={op.kernel_size})"
            elif op_name == 'spatial_pointwise_1x1':
                desc = f"PW1x1(C={op.C})"
            elif 'capsule_conv2d' in op_name:
                desc = f"CapConv(k={op.kernel_size},d={op.cap_dim})"
            elif 'channel_attention' in op_name:
                desc = f"SE(C={op.C},r={op.reduction_ratio})"
            elif isinstance(op, nn.BatchNorm2d):
                desc = f"BN2d({op.num_features})"
            elif 'derivative_conv2d' in op_name:
                if 'laplacian' in op_name:
                    desc = "DerConv2d(lap)"
                else:
                    suffix = op_name.split('_d')[-1] if '_d' in op_name else op_name
                    desc = f"DerConv2d({suffix})"
            elif 'polynomial' in op_name:
                desc = f"SpatPoly(deg={op.degree})"
            elif 'branched' in op_name:
                desc = f"SpatBranch({op.num_branches})"
            else:
                desc = type(op).__name__
            result.append(gate_prefix + desc)
        return result


class ConvStem(nn.Module):
    """Minimal convolutional stem that preserves spatial structure.

    Replaces nn.Linear(d_input, d_init) when spatial_shape is set.
    Maps [B, C_in, H, W] -> [B, C_stem, H, W].

    Architecture: Conv2d(C_in, C_stem, 3, padding=1) -> BN -> ReLU
    """

    def __init__(self, C_in: int, C_stem: int, H: int, W: int):
        super().__init__()
        self.C_in = C_in
        self.C_stem = C_stem
        self.H = H
        self.W = W
        self.conv = nn.Conv2d(C_in, C_stem, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_stem)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        return self.act(self.bn(self.conv(x)))

    @property
    def out_features(self) -> int:
        """For compatibility with code that reads input_projection.out_features."""
        return self.C_stem * self.H * self.W

    @property
    def in_features(self) -> int:
        """For compatibility."""
        return self.C_in * self.H * self.W

    @property
    def out_channels(self) -> int:
        return self.C_stem

    @property
    def spatial_shape(self):
        """Spatial shape of the stem output: (C_stem, H, W)."""
        return (self.C_stem, self.H, self.W)


class ASANNLayer(nn.Module):
    """Polymorphic layer: either flat (nn.Linear) or spatial (Conv2d + BN + residual).

    Provides a uniform interface for surgery code so it can work with both types.

    Spatial mode:
        - Conv2d(C_in, C_out, 3, stride, padding=1) + BN2d + structural residual
        - Tracks spatial_shape = (C_out, H_out, W_out)
        - Structural residual is always-on (no learnable scale)

    Flat mode:
        - nn.Linear(d_in, d_out) — identical to current behavior
    """

    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode  # "spatial" or "flat"

        if mode == "spatial":
            C_in = kwargs["C_in"]
            C_out = kwargs["C_out"]
            H = kwargs["H"]
            W = kwargs["W"]
            stride = kwargs.get("stride", 1)
            drop_path_rate = kwargs.get("drop_path_rate", 0.0)
            self._C_in = C_in
            self._stride = stride

            self.conv = nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
            self.bn = nn.BatchNorm2d(C_out)

            H_out = H // stride
            W_out = W // stride
            self.spatial_shape: Optional[Tuple[int, int, int]] = (C_out, H_out, W_out)

            # Structural residual (always-on, no learnable scale)
            self.has_structural_residual = True
            if C_in != C_out or stride != 1:
                # 1x1 conv to match channels + avg pool for stride
                self.residual_proj = nn.Conv2d(C_in, C_out, kernel_size=1,
                                               stride=stride, bias=False)
            else:
                self.residual_proj = None

            # DropPath: stochastic depth for the residual branch
            self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

            # Near-identity init: identity convolution + small noise
            nn.init.dirac_(self.conv.weight)
            self.conv.weight.data += 0.01 * torch.randn_like(self.conv.weight.data)
            if self.residual_proj is not None:
                nn.init.dirac_(self.residual_proj.weight)

        elif mode == "flat":
            d_in = kwargs["d_in"]
            d_out = kwargs["d_out"]
            self.linear = nn.Linear(d_in, d_out)
            self.spatial_shape = None
            self._C_in = None
            self._stride = None
            self.has_structural_residual = False
            self.residual_proj = None
            self.drop_path = nn.Identity()

        else:
            raise ValueError(f"Unknown layer mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "spatial":
            # x: [B, C_in, H, W]
            out = self.bn(self.conv(x))
            if self.has_structural_residual:
                if self.residual_proj is not None:
                    identity = self.residual_proj(x)
                elif x.shape[1] == out.shape[1]:
                    identity = x
                else:
                    # Safety: shapes mismatch but no proj (can happen after
                    # surgery add/remove channel sequences). Skip residual.
                    return out
                # DropPath: randomly drop the conv branch, keeping identity only.
                # This is the standard stochastic depth formulation.
                out = identity + self.drop_path(out)
            return out
        else:
            # x: [B, d]
            return self.linear(x)

    # --- Uniform interface for surgery code ---

    @property
    def out_features(self) -> int:
        """Flat width: d for Linear, C*H*W for spatial."""
        if self.mode == "flat":
            return self.linear.out_features
        else:
            C, H, W = self.spatial_shape
            return C * H * W

    @out_features.setter
    def out_features(self, value):
        """Setter for backward compat with code that sets layer.out_features directly."""
        if self.mode == "flat":
            self.linear.out_features = value

    @property
    def in_features(self) -> int:
        if self.mode == "flat":
            return self.linear.in_features
        else:
            return self._C_in * self.spatial_shape[1] * (self._stride if self._stride else 1) * \
                   self.spatial_shape[2] * (self._stride if self._stride else 1)

    @in_features.setter
    def in_features(self, value):
        """Setter for backward compat with code that sets layer.in_features directly."""
        if self.mode == "flat":
            self.linear.in_features = value

    @property
    def out_channels(self) -> int:
        """Channel count for spatial layers."""
        if self.mode != "spatial":
            raise AttributeError("out_channels only valid for spatial layers")
        return self.spatial_shape[0]

    @property
    def in_channels(self) -> int:
        if self.mode != "spatial":
            raise AttributeError("in_channels only valid for spatial layers")
        return self._C_in

    @property
    def weight(self):
        """Access underlying weight for NUS computation and surgery."""
        if self.mode == "flat":
            return self.linear.weight
        else:
            return self.conv.weight

    @weight.setter
    def weight(self, value):
        if self.mode == "flat":
            self.linear.weight = value
        else:
            self.conv.weight = value

    @property
    def bias(self):
        if self.mode == "flat":
            return self.linear.bias
        else:
            return self.bn.bias  # BN bias serves this role

    @bias.setter
    def bias(self, value):
        if self.mode == "flat":
            self.linear.bias = value

    # Properties that delegate to inner modules and must bypass Module.__setattr__
    _DELEGATED_PROPERTIES = frozenset({'weight', 'bias', 'out_features', 'in_features'})

    def __setattr__(self, name: str, value):
        """Override to route delegated properties through their setters.

        PyTorch's nn.Module.__setattr__ intercepts nn.Parameter assignments
        and tries to register_parameter() on THIS module, which conflicts
        with the property descriptors that delegate to self.linear/self.conv.
        It also intercepts plain attribute assignments when a property exists.
        We intercept delegated names here and route them correctly.
        """
        if name in self._DELEGATED_PROPERTIES and hasattr(self, 'mode'):
            prop = type(self).__dict__.get(name)
            if prop is not None and isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return
        super().__setattr__(name, value)


class SkipConnection:
    """A real skip connection with an optional projection.

    When created, the projection starts at near-zero to avoid disrupting training.
    When removed, the projection is deleted and memory is freed.

    For spatial connections (both source/target are [B,C,H,W]):
      - Uses Conv2d(1x1) projection when channels differ
      - Uses adaptive_avg_pool2d when spatial dims differ (stride mismatch)
    For flat connections: uses nn.Linear projection (original behavior).
    """

    def __init__(
        self,
        source_idx: int,
        target_idx: int,
        d_source: int,
        d_target: int,
        init_scale: float = 0.01,
        device: str = "cpu",
        spatial_source_shape=None,  # (C, H, W) or None
        spatial_target_shape=None,  # (C, H, W) or None
    ):
        self.source = source_idx
        self.target = target_idx
        self.spatial_source_shape = spatial_source_shape
        self.spatial_target_shape = spatial_target_shape

        both_spatial = (spatial_source_shape is not None
                        and spatial_target_shape is not None)

        if both_spatial:
            C_src = spatial_source_shape[0]
            C_tgt = spatial_target_shape[0]
            if C_src != C_tgt:
                self.projection = nn.Conv2d(C_src, C_tgt, kernel_size=1, bias=False).to(device)
                nn.init.zeros_(self.projection.weight)
            else:
                self.projection = None
        else:
            if d_source != d_target:
                self.projection = nn.Linear(d_source, d_target, bias=False).to(device)
                nn.init.zeros_(self.projection.weight)
            else:
                self.projection = None

        self.scale = nn.Parameter(torch.tensor(init_scale, device=device))

        # Tracking for removal: how many consecutive intervals with low utility
        self.low_utility_count = 0

    def forward(self, source_h: torch.Tensor) -> torch.Tensor:
        x = source_h
        if self.projection is not None:
            x = self.projection(x)
        # Handle spatial dim mismatch via adaptive pooling
        if (self.spatial_target_shape is not None
                and x.dim() == 4):
            _, H_tgt, W_tgt = self.spatial_target_shape
            if x.shape[2] != H_tgt or x.shape[3] != W_tgt:
                x = nn.functional.adaptive_avg_pool2d(x, (H_tgt, W_tgt))
        return self.scale * x

    def utility(self) -> float:
        """Connection Utility: scale * (||projection.weight||_2 if projection else 1.0).

        Negative scale means the connection is inverting its source signal,
        which is actively harmful. Returns 0.0 for negative scales so
        these connections are flagged for removal.
        """
        scale_val = self.scale.item()  # raw value, NOT abs()
        if scale_val < 0:
            return 0.0  # Negative scale = zero utility -> will be removed
        if self.projection is not None:
            proj_norm = self.projection.weight.data.norm().item()
        else:
            proj_norm = 1.0
        return scale_val * proj_norm

    def parameters(self):
        """Yield all trainable parameters of this connection."""
        yield self.scale
        if self.projection is not None:
            yield from self.projection.parameters()

    def to(self, device):
        self.scale = nn.Parameter(self.scale.data.to(device))
        if self.projection is not None:
            self.projection = self.projection.to(device)
        return self


class ASANNModel(nn.Module):
    """Continuously Self-Architecting Neural Network.

    At any point during training, this is a REAL, compact neural network.
    - self.num_layers is the ACTUAL current depth
    - self.layers[l].weight has shape [d_out_actual, d_in_actual] - actual dimensions, not d_max
    - self.ops[l] is an actual OperationPipeline, not a softmax mixture
    - self.connections is a list of actual tensor operations, not a dense weighted matrix
    - Memory usage and compute cost reflect the actual current architecture, always
    """

    def __init__(self, d_input: int, d_output: int, config: ASANNConfig):
        super().__init__()
        self.config = config
        self.d_input = d_input
        self.d_output = d_output

        # Determine if this is a spatial (image) model
        self._is_spatial = config.spatial_shape is not None
        self._is_graph = False  # Set True by set_graph_data()

        # Hidden layers and ops (shared ModuleLists for both modes)
        self.layers = nn.ModuleList()
        self.ops = nn.ModuleList()

        if self._is_spatial:
            # ===== SPATIAL MODE: Convolutional backbone with downsampling =====
            C_in, H, W = config.spatial_shape
            C_stem = config.c_stem_init
            self._effective_d_init = C_stem * H * W  # For compatibility

            # ConvEncoder replaces old ConvStem input_projection
            self.encoder = ConvEncoder(C_in, C_stem, H, W)

            # Build spatial layers with progressive downsampling:
            # Stage pattern: [same-res block, stride-2 downsample] repeated
            # Channels double at each stride-2 stage (standard CNN design)
            C_cur = C_stem
            H_cur, W_cur = H, W
            if config.spatial_downsample_stages == "auto":
                n_downsample = 0  # Start all stride-1; discover downsampling via surgery
            else:
                n_downsample = min(config.spatial_downsample_stages,
                                   config.initial_num_layers // 2)

            # Calculate total layers: at least initial_num_layers
            layer_idx = 0
            total_layers = max(config.initial_num_layers,
                               n_downsample * 2)  # 2 layers per stage

            # DropPath: linearly increasing drop rates across depth
            # Layer 0 gets 0, last layer gets drop_path_rate
            _dp_enabled = config.drop_path_enabled and config.drop_path_rate > 0
            _dp_rates = [
                config.drop_path_rate * i / max(total_layers - 1, 1)
                for i in range(total_layers)
            ] if _dp_enabled else [0.0] * total_layers

            for stage in range(n_downsample):
                # Same-resolution conv block
                layer = ASANNLayer(mode="spatial", C_in=C_cur, C_out=C_cur,
                                   H=H_cur, W=W_cur,
                                   drop_path_rate=_dp_rates[layer_idx])
                self.layers.append(layer)
                pipeline = SpatialOperationPipeline()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)
                layer_idx += 1

                # Stride-2 downsample (double channels, halve spatial dims)
                C_next = min(C_cur * 2, config.max_channels)
                H_next = H_cur // 2
                W_next = W_cur // 2
                if H_next >= config.min_spatial_resolution:
                    layer = ASANNLayer(mode="spatial", C_in=C_cur, C_out=C_next,
                                       H=H_cur, W=W_cur, stride=2,
                                       drop_path_rate=_dp_rates[layer_idx])
                    self.layers.append(layer)
                    pipeline = SpatialOperationPipeline()
                    pipeline.add_operation(nn.ReLU(), 0)
                    self.ops.append(pipeline)
                    C_cur = C_next
                    H_cur, W_cur = H_next, W_next
                    layer_idx += 1

            # Fill remaining layers at final resolution (if initial_num_layers > stages*2)
            while layer_idx < total_layers:
                layer = ASANNLayer(mode="spatial", C_in=C_cur, C_out=C_cur,
                                   H=H_cur, W=W_cur,
                                   drop_path_rate=_dp_rates[layer_idx])
                self.layers.append(layer)
                pipeline = SpatialOperationPipeline()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)
                layer_idx += 1

            # Flatten position: index of last spatial layer (all layers start spatial)
            self._flatten_position = len(self.layers) - 1

            # Output head: GAP → linear (Global Average Pooling eliminates H,W)
            # This forces the backbone to learn per-channel semantic features
            # instead of memorizing spatial positions through a huge Linear head.
            self._use_gap = True
            gap_d = C_cur  # GAP reduces [B, C, H, W] → [B, C]
            self.output_head = nn.Linear(gap_d, d_output)
            self._d_output = d_output  # Saved for JK/spinal rebuild
            print(f"  Spatial mode: stem={C_in}->{C_stem}ch, "
                  f"stages={n_downsample} downsample, "
                  f"final={C_cur}ch@{H_cur}x{W_cur}, "
                  f"GAP_d={gap_d}")

        else:
            # ===== FLAT MODE: Original MLP backbone (unchanged) =====
            # RC1: Auto-scale d_init for high-dimensional inputs
            d_init = config.d_init
            if config.d_init_auto and d_input > config.d_init * config.d_init_ratio:
                auto_d = int(d_input / config.d_init_ratio)
                d_init = max(config.d_init_min, min(config.d_init_max, auto_d))
                d_init = ((d_init + 7) // 8) * 8  # Align to 8 for GPU efficiency
                print(f"  Auto d_init: {config.d_init} -> {d_init} (d_input={d_input}, ratio={config.d_init_ratio})")
            self._effective_d_init = d_init
            self._flatten_position = -1  # No spatial layers

            # Encoder: maps raw features to initial hidden dimension.
            # If encoder_candidates is configured, start with the first candidate
            # instead of always defaulting to LinearEncoder. This lets experiments
            # specify e.g. encoder_candidates=["molecular_graph"] to use a GNN
            # encoder from the start.
            initial_encoder_type = None
            if config.encoder_candidates and len(config.encoder_candidates) > 0:
                initial_encoder_type = config.encoder_candidates[0]

            if initial_encoder_type and initial_encoder_type != "linear":
                try:
                    enc_kwargs = build_encoder_kwargs(
                        initial_encoder_type,
                        d_input=d_input,
                        d_output=d_init,
                        config=config,
                        model=self,
                    )
                    self.encoder = create_encoder(initial_encoder_type, **enc_kwargs)
                    print(f"  Initial encoder: {initial_encoder_type} "
                          f"(d_input={d_input}, d_output={d_init})")
                except Exception as e:
                    print(f"  WARNING: Failed to create {initial_encoder_type} encoder: {e}")
                    print(f"  Falling back to LinearEncoder")
                    self.encoder = LinearEncoder(d_input, d_init)
            else:
                self.encoder = LinearEncoder(d_input, d_init)

            # Hidden layers: start minimal (Section 7)
            for _ in range(config.initial_num_layers):
                self.layers.append(nn.Linear(d_init, d_init))

            # Operation pipelines: one per layer, start with just ReLU
            for _ in range(config.initial_num_layers):
                pipeline = OperationPipeline()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)

            # Output head: maps final hidden to output
            self.output_head = nn.Linear(d_init, d_output)
            self._d_output = d_output  # Saved for JK rebuild (output_head may be replaced)

        # Skip connections: none initially
        self.connections: List[SkipConnection] = []

        # Jumping Knowledge (Xu et al., 2018): attention-weighted layer aggregation.
        # Always OFF initially — enabled/disabled by the treatment system dynamically.
        # When enabled, all layer outputs are projected to a fixed d_jk dimension,
        # weighted by learned attention, and fed to the output_head.
        self._jk_enabled = False
        self._jk_d: int = 0           # Fixed projection dim (set on enable, never changes)
        self._jk_projections = nn.ModuleList()
        self._jk_attn_logits = nn.ParameterList()
        self._jk_widths: List[int] = []  # Cached input widths for lazy rebuild

        # SpinalNet-style architecture: input split + concat output head
        self._spinal_enabled = False
        self._spinal_input_projs = nn.ModuleList()
        self._spinal_concat_head: Optional[nn.Linear] = None  # Linear(sum_C_outs, d_output)
        self._spinal_slices: List[slice] = []
        self._spinal_num_layers: int = 0  # Cached for lazy rebuild
        self._spinal_layer_widths: List[int] = []  # Cached channel dims
        self._spinal_C_stem: int = 0  # Cached at enable time (survives encoder swaps)

        # Architecture stabilized flag
        self.architecture_stable = False

        # ----- Graph auxiliary data (populated by experiment if graph-structured) -----
        # These are NOT flags/toggles — they are structural data analogous to spatial_shape.
        # Graph-aware operations access these when constructed by treatments.
        self._graph_edge_index: Optional[torch.Tensor] = None   # [2, E] edge list
        self._graph_adj_sparse: Optional[torch.Tensor] = None   # [N, N] sparse adjacency
        self._graph_degree: Optional[torch.Tensor] = None        # [N] node degrees
        self._graph_num_nodes: int = 0
        # Learnable gate for input pre-aggregation: sigmoid(gate) controls how
        # much neighbor-aggregated features vs raw features reach input_projection.
        # The model learns whether pre-aggregation helps (node classification)
        # or hurts (traffic forecasting) rather than requiring a manual toggle.
        self._graph_pre_agg_gate = nn.Parameter(torch.tensor(0.0))

        # ----- Statistics tracking for surgery signals -----
        # These are populated by the SurgeryScheduler during training
        self._activation_history: Dict[int, List[torch.Tensor]] = {}
        self._gradient_history: Dict[int, List[torch.Tensor]] = {}
        self._layer_input_history: Dict[int, List[torch.Tensor]] = {}
        self._layer_output_history: Dict[int, List[torch.Tensor]] = {}

        # Config-driven feature toggles (before to(device) for proper device placement)
        if config.jk_enabled:
            self.enable_jk()
        if config.spinal_enabled and self._is_spatial:
            self._enable_spinal()

        self.to(config.device)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def input_projection(self):
        """Backward compatibility alias: returns self.encoder.

        All external code (surgery.py, treatments.py, scheduler.py, trainer.py)
        accesses model.input_projection for:
          - .out_features, .in_features, .weight, .bias (LinearEncoder)
          - .spatial_shape, .out_channels, .C_in, .C_stem, .H, .W (ConvEncoder)
        The encoder classes provide all these attributes.
        """
        return self.encoder

    @input_projection.setter
    def input_projection(self, value):
        """Allow setting input_projection (e.g., in reevaluate scripts)."""
        self.encoder = value

    def encoder_switch(self, new_encoder: 'BaseEncoder',
                       warmup_epochs: Optional[int] = None):
        """Switch to a new encoder using a gated bridge for gradual transition.

        Creates a GatedEncoderBridge that blends old and new encoder output:
            output = (1 - alpha) * old_encoder(x) + alpha * proj(new_encoder(x))
        where alpha ramps 0 -> 1 over warmup_epochs.

        After warmup completes, advance_surgery_gates() absorbs the bridge,
        replacing self.encoder with the new encoder permanently.

        Args:
            new_encoder: The new encoder to switch to.
            warmup_epochs: Blending warmup epochs (default: config.encoder_switch_warmup_epochs).
        """
        if warmup_epochs is None:
            warmup_epochs = self.config.encoder_switch_warmup_epochs

        old_encoder = self.encoder
        # Unwrap existing bridge if switching again mid-bridge
        if isinstance(old_encoder, GatedEncoderBridge):
            old_encoder = old_encoder.old_encoder

        bridge = GatedEncoderBridge(old_encoder, new_encoder,
                                    warmup_epochs=warmup_epochs)
        self.encoder = bridge.to(self.config.device)

        # Propagate molecular batch and individual graphs to new encoder
        self._propagate_molecular_batch()
        self._propagate_molecular_graphs()

        print(f"  [ENCODER] Switching: {old_encoder.encoder_type} -> "
              f"{new_encoder.encoder_type} "
              f"(warmup={warmup_epochs} epochs, "
              f"d_out: {old_encoder.d_output} -> {new_encoder.d_output})")

    def set_graph_data(self, edge_index: torch.Tensor, num_nodes: int,
                       degree: Optional[torch.Tensor] = None):
        """Store graph structure as auxiliary data for graph-aware operations.

        This is analogous to spatial_shape for image models. Graph-aware operations
        (NeighborAggregation, GraphAttention, etc.) access this data when constructed
        by the treatment engine.

        Args:
            edge_index: [2, E] tensor of directed edges (src, dst).
            num_nodes: Total number of nodes in the graph.
            degree: [N] tensor of node in-degrees. Computed from edge_index if None.
        """
        self._graph_edge_index = edge_index
        self._graph_num_nodes = num_nodes

        # Build sparse adjacency matrix [N, N]
        values = torch.ones(edge_index.shape[1], device=edge_index.device)
        self._graph_adj_sparse = torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_nodes)
        ).coalesce()

        # Compute in-degree if not provided
        if degree is None:
            degree = torch.zeros(num_nodes, device=edge_index.device)
            degree.scatter_add_(0, edge_index[1], values)
        self._graph_degree = degree
        self._is_graph = True

        # Propagate graph data to encoder if it supports it
        encoder = self.encoder
        if hasattr(encoder, 'set_graph_data'):
            encoder.set_graph_data(self._graph_adj_sparse, degree)

    def swap_graph_data(self, edge_index, num_nodes, degree):
        """Swap graph data on model AND all graph ops for inductive train/eval.

        Unlike set_graph_data() which only updates model-level attributes,
        this also updates every graph operation's stored adjacency buffers.
        Used to switch between train subgraph (for training) and full graph
        (for evaluation) in inductive graph learning.

        Args:
            edge_index: [2, E] edge index tensor
            num_nodes: number of nodes
            degree: [N] degree tensor
        """
        self.set_graph_data(edge_index, num_nodes, degree)
        from .surgery import (
            _update_graph_op_data,
            NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
            SpectralConv, MessagePassingGIN, DegreeScaling, GraphBranchedBlock,
            APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
            SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
            DirectionalDiffusion, AdaptiveGraphConv, GraphNorm,
            MessageBooster, VirtualNodeOp, PairNorm, GraphPositionalEncoding,
        )
        _GRAPH_OP_TYPES = (
            NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
            SpectralConv, MessagePassingGIN, GraphBranchedBlock,
            APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
            SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
            DirectionalDiffusion, AdaptiveGraphConv, DegreeScaling, GraphNorm,
            MessageBooster, VirtualNodeOp, PairNorm, GraphPositionalEncoding,
        )
        for l in range(self.num_layers):
            for op in self.ops[l].operations:
                # Unwrap GatedOp to get the actual operation
                inner = getattr(op, 'operation', op)
                if isinstance(inner, _GRAPH_OP_TYPES):
                    _update_graph_op_data(
                        inner, self._graph_adj_sparse, degree,
                        num_nodes, edge_index)

    def set_molecular_batch(self, pyg_batch):
        """Store pre-batched molecular graph data for MolecularGraphEncoder.

        The molecular batch is a PyG Batch object containing all molecule graphs
        (atoms as nodes, bonds as edges). It is used by MolecularGraphEncoder
        during forward passes to process molecular structure instead of fingerprints.

        Also propagates to the current encoder if it is (or wraps) a
        MolecularGraphEncoder.

        Args:
            pyg_batch: torch_geometric.data.Batch with:
                - x: [total_atoms, atom_feature_dim] node features
                - edge_index: [2, total_edges] bond connectivity
                - batch: [total_atoms] mapping atoms -> molecule index
        """
        self._molecular_batch = pyg_batch

        # Propagate to current encoder if applicable
        self._propagate_molecular_batch()

    def _propagate_molecular_batch(self):
        """Forward stored molecular batch to the current encoder if applicable."""
        batch = getattr(self, '_molecular_batch', None)
        if batch is None:
            return

        encoder = self.encoder
        # Direct MolecularGraphEncoder
        if hasattr(encoder, 'set_molecular_batch'):
            encoder.set_molecular_batch(batch)
        # Inside a GatedEncoderBridge
        elif isinstance(encoder, GatedEncoderBridge):
            if hasattr(encoder.new_encoder, 'set_molecular_batch'):
                encoder.new_encoder.set_molecular_batch(batch)
            if hasattr(encoder.old_encoder, 'set_molecular_batch'):
                encoder.old_encoder.set_molecular_batch(batch)
        # Inside a ProjectedEncoder
        elif isinstance(encoder, ProjectedEncoder):
            inner = getattr(encoder, '_encoder', None)
            if inner is not None and hasattr(inner, 'set_molecular_batch'):
                inner.set_molecular_batch(batch)

        # Also propagate individual graphs list for mini-batch mode
        self._propagate_molecular_graphs()

    def set_molecular_graphs(self, graphs_list):
        """Store individual molecular graphs for mini-batch sub-batching.

        Each element in graphs_list is a torch_geometric.data.Data object
        for one molecule, ordered to match [train, val, test] concatenation.
        The trainer yields (x, y, mol_idx) batches; mol_idx maps into this list.

        Args:
            graphs_list: List[torch_geometric.data.Data], one per molecule.
        """
        self._all_molecular_graphs = graphs_list
        self._propagate_molecular_graphs()

    def set_current_mol_indices(self, indices):
        """Set which molecule indices are in the current mini-batch.

        Called by the trainer before each forward pass when using
        molecular mini-batch dataloaders that yield (x, y, mol_idx).

        Args:
            indices: Tensor of integer indices into the stored graphs list.
        """
        self._current_mol_indices = indices
        # Propagate to current encoder
        encoder = self.encoder
        if hasattr(encoder, 'set_current_mol_indices'):
            encoder.set_current_mol_indices(indices)
        elif isinstance(encoder, GatedEncoderBridge):
            if hasattr(encoder.new_encoder, 'set_current_mol_indices'):
                encoder.new_encoder.set_current_mol_indices(indices)
            if hasattr(encoder.old_encoder, 'set_current_mol_indices'):
                encoder.old_encoder.set_current_mol_indices(indices)
        elif isinstance(encoder, ProjectedEncoder):
            inner = getattr(encoder, '_encoder', None)
            if inner is not None and hasattr(inner, 'set_current_mol_indices'):
                inner.set_current_mol_indices(indices)

    def _propagate_molecular_graphs(self):
        """Forward stored individual molecular graphs to the current encoder."""
        graphs = getattr(self, '_all_molecular_graphs', None)
        if graphs is None:
            return

        encoder = self.encoder
        if hasattr(encoder, 'set_molecular_graphs'):
            encoder.set_molecular_graphs(graphs)
        elif isinstance(encoder, GatedEncoderBridge):
            if hasattr(encoder.new_encoder, 'set_molecular_graphs'):
                encoder.new_encoder.set_molecular_graphs(graphs)
            if hasattr(encoder.old_encoder, 'set_molecular_graphs'):
                encoder.old_encoder.set_molecular_graphs(graphs)
        elif isinstance(encoder, ProjectedEncoder):
            inner = getattr(encoder, '_encoder', None)
            if inner is not None and hasattr(inner, 'set_molecular_graphs'):
                inner.set_molecular_graphs(graphs)

    def seed_graph_ops(self):
        """Seed each layer with a NeighborAggregation op for GCN-equivalent from step 0.

        Makes the model equivalent to a multi-layer GCN immediately after
        set_graph_data() is called, instead of waiting for the treatment system
        to discover graph ops (which requires the model to get "sick" first).

        Idempotent: skips layers that already have a graph aggregation op.
        Safe for resume from checkpoint.
        """
        if not self._is_graph:
            raise RuntimeError("seed_graph_ops() requires set_graph_data() to be called first")

        from .surgery import (create_operation, NeighborAggregation, GraphAttentionAggregation,
            GraphDiffusion, SpectralConv, MessagePassingGIN, DegreeScaling, GraphBranchedBlock,
            APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation, SGConv,
            DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
            DirectionalDiffusion, AdaptiveGraphConv)

        _GRAPH_AGG_TYPES = (
            NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
            SpectralConv, MessagePassingGIN, GraphBranchedBlock,
            APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
            SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
            DirectionalDiffusion, AdaptiveGraphConv,
        )

        graph_data = {
            'adj_sparse': self._graph_adj_sparse,
            'edge_index': self._graph_edge_index,
            'degree': self._graph_degree,
            'num_nodes': self._graph_num_nodes,
        }

        for l in range(self.num_layers):
            # Check if layer already has a graph aggregation op
            has_graph_op = any(
                isinstance(op, _GRAPH_AGG_TYPES)
                for op in self.ops[l].operations
            )
            if has_graph_op:
                continue

            layer = self.layers[l]
            d = layer.out_features
            try:
                agg_op = create_operation(
                    "graph_neighbor_agg", d,
                    device=self.config.device,
                    config=self.config,
                    graph_data=graph_data,
                )
                # Mark as added at step 1 so surgery gives it the standard
                # protection period before considering it for removal.
                agg_op._asann_added_step = 1
                self.ops[l].add_operation(agg_op, position=0)
            except Exception as e:
                print(f"  [WARN] seed_graph_ops: failed to add NeighborAgg to layer {l}: {e}")

    def enable_jk(self):
        """Enable Jumping Knowledge — called by the treatment system or config.

        Builds per-layer projections (Linear(d_l, d_jk)) and attention logits.
        Also rebuilds the output_head to accept d_jk input (fixed dimension).
        Idempotent: safe to call if already enabled.

        For spatial models: JK applies GAP to each layer's 4D output before
        projecting to d_jk. The projection is Linear(C_layer, d_jk).
        """
        if self._jk_enabled:
            return

        # Determine fixed d_jk (never changes after this, even if layers resize)
        # For spatial models with GAP, use channel count (not C*H*W)
        if self.config.jk_d > 0:
            d_jk = self.config.jk_d
        elif self._is_spatial:
            d_jk = self.layers[-1].spatial_shape[0]  # Channel count (C) after GAP
        else:
            d_jk = self._effective_d_init
        self._jk_d = d_jk
        self._jk_enabled = True

        # Rebuild output_head to accept d_jk input
        # Use saved _d_output (not output_head.out_features) because output_head
        # may be an MLPHead whose .out_features proxies to the hidden layer dim.
        d_output = getattr(self, '_d_output', self.output_head.out_features)
        device = self.config.device
        self.output_head = nn.Linear(d_jk, d_output, device=device)

        # Build projections
        self._rebuild_jk()
        print(f"  [JK] Enabled: d_jk={d_jk}, {self.num_layers} layer projections")

    def disable_jk(self):
        """Disable Jumping Knowledge — reverts to using last layer output.

        Rebuilds the output_head to accept the last layer's width.
        """
        if not self._jk_enabled:
            return

        self._jk_enabled = False
        device = self.config.device

        # Rebuild output_head to match last layer width
        d_last = self.layers[-1].out_features if self.num_layers > 0 else self._effective_d_init
        d_output = getattr(self, '_d_output', self.output_head.out_features)
        self.output_head = nn.Linear(d_last, d_output, device=device)

        # Clear JK components
        self._jk_projections = nn.ModuleList()
        self._jk_attn_logits = nn.ParameterList()
        self._jk_widths = []
        self._jk_d = 0
        print(f"  [JK] Disabled: output_head now {d_last} -> {d_output}")

    def verify_output_head(self):
        """Verify output_head input dimension matches the last layer's output.

        Called after surgery as a safety net. If JK or Spinal is enabled, the
        output_head dimension is managed by those subsystems and this is a no-op.
        When a mismatch is detected (from any surgery bug), auto-repairs it
        and logs a warning so the root cause can be investigated.
        """
        if self._jk_enabled or self._spinal_enabled:
            return  # Managed by JK/Spinal subsystems

        if self.num_layers == 0:
            return

        last_layer = self.layers[-1]
        if hasattr(last_layer, 'mode') and last_layer.mode == "spatial":
            if getattr(self, '_use_gap', False):
                expected_in = last_layer.spatial_shape[0]  # C after GAP
            else:
                C, H, W = last_layer.spatial_shape
                expected_in = C * H * W
        else:
            expected_in = last_layer.out_features

        actual_in = self.output_head.in_features
        if expected_in != actual_in:
            print(f"  [SAFETY] output_head dim mismatch: last layer outputs "
                  f"{expected_in}, but output_head expects {actual_in}. Auto-repairing.")
            d_output = getattr(self, '_d_output', self.output_head.out_features)
            device = next(self.parameters()).device
            old_weight = self.output_head.weight.data
            new_weight = torch.zeros(d_output, expected_in, device=device)
            copy_cols = min(old_weight.shape[1], expected_in)
            new_weight[:, :copy_cols] = old_weight[:, :copy_cols]
            self.output_head = nn.Linear(expected_in, d_output, device=device)
            self.output_head.weight = nn.Parameter(new_weight)
            if hasattr(self, '_d_output'):
                pass  # _d_output stays unchanged

    def _rebuild_jk(self):
        """Rebuild JK projections when layer count or widths change.

        Uses the fixed self._jk_d as output dimension (never tracks output_head).
        Called lazily in forward() when widths mismatch.
        """
        if not self._jk_enabled or self._jk_d == 0:
            return

        d_jk = self._jk_d
        # For spatial models, use channel count (C) not C*H*W
        if self._is_spatial:
            new_widths = [self.layers[l].spatial_shape[0] for l in range(self.num_layers)]
        else:
            new_widths = [self.layers[l].out_features for l in range(self.num_layers)]

        # Check if anything changed
        if (new_widths == self._jk_widths
                and len(self._jk_projections) == self.num_layers):
            return

        device = self.config.device
        old_projs = list(self._jk_projections)
        old_attn = list(self._jk_attn_logits)

        self._jk_projections = nn.ModuleList()
        self._jk_attn_logits = nn.ParameterList()

        for l in range(self.num_layers):
            d_l = new_widths[l]

            # Reuse existing projection if input width unchanged and already existed
            if (l < len(old_projs) and l < len(self._jk_widths)
                    and self._jk_widths[l] == d_l
                    and old_projs[l].out_features == d_jk):
                self._jk_projections.append(old_projs[l])
                self._jk_attn_logits.append(old_attn[l])
            else:
                self._jk_projections.append(nn.Linear(d_l, d_jk, device=device))
                self._jk_attn_logits.append(nn.Parameter(torch.zeros(1, device=device)))

        self._jk_widths = new_widths

    # ---- SpinalNet-style input/output split ----

    def _enable_spinal(self):
        """Enable SpinalNet-style architecture: additive injection + concat output.

        Splits encoder output channels into K groups (K = num_layers).
        Each layer receives normal input + projected channel group injection.
        Concat head aggregates all layer outputs into final prediction.
        """
        if self._spinal_enabled:
            return
        if not self._is_spatial:
            print("  [Spinal] Only supported for spatial models")
            return
        self._spinal_enabled = True
        self._spinal_C_stem = self.encoder.C_stem  # Cache (survives encoder swaps)
        self._rebuild_spinal()
        print(f"  [Spinal] Enabled: {self.num_layers} groups, "
              f"C_stem={self._spinal_C_stem}")

    def _rebuild_spinal(self):
        """Rebuild spinal projections when layer count or channel dims change.

        Key design: channel splits are LOCKED at initialization. When surgery
        adds layers beyond the initial count, new layers get zero-init
        projections (all C_stem channels) so they start as normal ASANN layers
        and gradually learn to use encoder features. This prevents catastrophic
        accuracy drops from re-splitting learned projections.
        """
        if not self._spinal_enabled:
            return

        K = self.num_layers
        if K == 0:
            return

        C_stem = self._spinal_C_stem
        d_output = getattr(self, '_d_output', self.d_output)
        device = self.config.device

        # Get current layer channel widths (C only, not C*H*W)
        new_widths = [self.layers[l].spatial_shape[0] for l in range(K)]

        # Check if anything changed
        if (K == self._spinal_num_layers
                and new_widths == self._spinal_layer_widths):
            return

        # Slices: only compute at initial enable (when no slices exist yet)
        # After that, keep existing slices locked — don't re-split
        if len(self._spinal_slices) == 0:
            initial_K = K
            group_size = C_stem // initial_K
            for l in range(initial_K):
                start = l * group_size
                end = (l + 1) * group_size if l < initial_K - 1 else C_stem
                self._spinal_slices.append(slice(start, end))

        # Build/update input projections
        # - Existing layers: keep current projections if dims match,
        #   rebuild only if layer C_in changed (surgery channel resize)
        # - New layers (beyond existing slices): zero-init projection
        #   from ALL C_stem channels (doesn't fragment input)
        old_projs = list(self._spinal_input_projs)
        self._spinal_input_projs = nn.ModuleList()

        for l in range(K):
            C_layer_in = self.layers[l]._C_in

            if l < len(self._spinal_slices):
                # Existing layer with a channel group slice
                group_channels = (self._spinal_slices[l].stop
                                  - self._spinal_slices[l].start)
            else:
                # New layer beyond initial split — uses all C_stem channels
                group_channels = C_stem
                # Add a slice that covers all channels for this layer
                if l >= len(self._spinal_slices):
                    self._spinal_slices.append(slice(0, C_stem))

            # Reuse existing projection if dims match
            if (l < len(old_projs)
                    and old_projs[l].in_channels == group_channels
                    and old_projs[l].out_channels == C_layer_in):
                self._spinal_input_projs.append(old_projs[l])
            else:
                proj = nn.Conv2d(group_channels, C_layer_in, 1, device=device)
                if l < len(old_projs) and old_projs[l].out_channels == C_layer_in:
                    # Layer C_in unchanged but group size changed (shouldn't
                    # happen with locked splits, but handle gracefully):
                    # copy overlapping weights
                    old_in = old_projs[l].in_channels
                    copy_in = min(old_in, group_channels)
                    proj.weight.data[:, :copy_in] = \
                        old_projs[l].weight.data[:, :copy_in]
                    if copy_in < group_channels:
                        nn.init.zeros_(proj.weight.data[:, copy_in:])
                    proj.bias.data.copy_(old_projs[l].bias.data)
                else:
                    # Completely new projection: zero-init so it doesn't
                    # disrupt the layer's normal input at first
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                self._spinal_input_projs.append(proj)

        # Build concat output head with weight preservation
        total_features = sum(new_widths)
        old_head = self._spinal_concat_head
        if (old_head is not None
                and old_head.out_features == d_output
                and old_head.in_features != total_features):
            new_head = nn.Linear(total_features, d_output, device=device)
            nn.init.zeros_(new_head.weight)
            nn.init.zeros_(new_head.bias)
            # Copy old weights per-layer column blocks
            old_offset, new_offset = 0, 0
            old_widths = self._spinal_layer_widths
            for l_idx in range(min(len(old_widths), K)):
                old_w = old_widths[l_idx]
                new_w = new_widths[l_idx]
                copy_w = min(old_w, new_w)
                new_head.weight.data[:, new_offset:new_offset + copy_w] = \
                    old_head.weight.data[:, old_offset:old_offset + copy_w]
                old_offset += old_w
                new_offset += new_w
            new_head.bias.data.copy_(old_head.bias.data)
            self._spinal_concat_head = new_head
        elif old_head is None or old_head.in_features != total_features:
            self._spinal_concat_head = nn.Linear(
                total_features, d_output, device=device)

        self._spinal_num_layers = K
        self._spinal_layer_widths = new_widths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actual current architecture.

        This is a real forward pass through a real network. The number of layers,
        their widths, the operations applied, and the connections used are all
        the actual current state of the architecture.

        For spatial models, x is [B, C*H*W] flat (from dataloaders) and gets
        reshaped to [B, C, H, W] before passing through ConvStem.
        """
        h: Dict[int, torch.Tensor] = {}

        if self._is_spatial:
            B = x.shape[0]
            C_in, H_in, W_in = self.config.spatial_shape
            x_spatial = x.view(B, C_in, H_in, W_in)
            h[0] = self.encoder(x_spatial)  # [B, C_stem, H, W]
        else:
            if self._is_graph:
                from .surgery import _batch_graph_mm
                N = self._graph_num_nodes
                alpha = torch.sigmoid(self._graph_pre_agg_gate)
                x_agg = _batch_graph_mm(self._graph_adj_sparse, x, N)
                x_agg = _batch_graph_mm(self._graph_adj_sparse, x_agg, N)
                x = alpha * x_agg + (1.0 - alpha) * x
            h[0] = self.encoder(x)  # [B*N, d_init] or [N, d_init]

        # SpinalNet mode: additive injection + concat output
        # Each layer gets normal input PLUS a projection of its assigned
        # encoder channel group. Requires spatial encoder with C_stem channels.
        if self._spinal_enabled and h[0].dim() == 4 and h[0].shape[1] >= self._spinal_C_stem:
            self._rebuild_spinal()  # Lazy rebuild if architecture changed
            for l in range(self.num_layers):
                # Normal input: encoder output (layer 0) or previous layer output
                h_in = h[l]

                # Spinal injection: add projected channel group from encoder
                group = h[0][:, self._spinal_slices[l]]
                h_in = h_in + self._spinal_input_projs[l](group)

                # Skip connections still apply
                for conn in self.connections:
                    if conn.target == l + 1:
                        h_in = h_in + conn.forward(h[conn.source])

                z = self.layers[l](h_in)
                h[l + 1] = self.ops[l](z)

            # Concat all layer outputs (after GAP) → single classifier
            features = []
            for l in range(self.num_layers):
                h_flat = nn.functional.adaptive_avg_pool2d(
                    h[l + 1], 1).flatten(1)
                features.append(h_flat)
            concat = torch.cat(features, dim=1)
            return self._spinal_concat_head(concat)

        for l in range(self.num_layers):
            # Start with direct predecessor output
            h_in = h[l]

            # Add contributions from skip connections targeting this layer
            for conn in self.connections:
                if conn.target == l + 1:
                    source_h = h[conn.source]
                    h_in = h_in + conn.forward(source_h)

            # Layer transformation (ASANNLayer handles spatial/flat internally)
            z = self.layers[l](h_in)

            # Operation pipeline: actual composed operations
            h[l + 1] = self.ops[l](z)

            # Flatten transition: after the last spatial layer
            if self._is_spatial and l == self._flatten_position:
                if getattr(self, '_use_gap', False):
                    # Global Average Pooling: [B, C, H, W] → [B, C]
                    h[l + 1] = nn.functional.adaptive_avg_pool2d(
                        h[l + 1], 1).flatten(start_dim=1)
                else:
                    h[l + 1] = h[l + 1].flatten(start_dim=1)

        # Jumping Knowledge: attention-weighted aggregation of all layer outputs
        if self._jk_enabled:
            self._rebuild_jk()  # Lazy rebuild if widths changed
            projs = []
            for l in range(self.num_layers):
                h_l = h[l + 1]
                # For spatial models: GAP before projection [B,C,H,W] → [B,C]
                if self._is_spatial and h_l.dim() == 4:
                    h_l = nn.functional.adaptive_avg_pool2d(h_l, 1).flatten(1)
                projs.append(self._jk_projections[l](h_l))
            logits = torch.cat([a for a in self._jk_attn_logits])  # [num_layers]
            attn_weights = torch.softmax(logits, dim=0)  # [num_layers]
            stacked = torch.stack(projs, dim=0)  # [num_layers, B, d_jk]
            jk_out = (attn_weights.view(-1, 1, 1) * stacked).sum(dim=0)  # [B, d_jk]
            return self.output_head(jk_out)

        return self.output_head(h[self.num_layers])

    def forward_with_intermediates(self, x: torch.Tensor) -> tuple:
        """Forward pass that also returns all intermediate activations.

        Used during surgery signal computation to collect gradient/activation stats.
        """
        h: Dict[int, torch.Tensor] = {}

        if self._is_spatial:
            B = x.shape[0]
            C_in, H_in, W_in = self.config.spatial_shape
            x_spatial = x.view(B, C_in, H_in, W_in)
            h[0] = self.encoder(x_spatial)
        else:
            if self._is_graph:
                from .surgery import _batch_graph_mm
                N = self._graph_num_nodes
                alpha = torch.sigmoid(self._graph_pre_agg_gate)
                x_agg = _batch_graph_mm(self._graph_adj_sparse, x, N)
                x_agg = _batch_graph_mm(self._graph_adj_sparse, x_agg, N)
                x = alpha * x_agg + (1.0 - alpha) * x
            h[0] = self.encoder(x)

        layer_inputs: Dict[int, torch.Tensor] = {}
        layer_outputs: Dict[int, torch.Tensor] = {}

        # SpinalNet mode: additive injection + concat output
        if self._spinal_enabled and h[0].dim() == 4 and h[0].shape[1] >= self._spinal_C_stem:
            self._rebuild_spinal()
            for l in range(self.num_layers):
                h_in = h[l]

                group = h[0][:, self._spinal_slices[l]]
                h_in = h_in + self._spinal_input_projs[l](group)

                for conn in self.connections:
                    if conn.target == l + 1:
                        h_in = h_in + conn.forward(h[conn.source])

                layer_inputs[l] = h_in
                z = self.layers[l](h_in)
                layer_outputs[l] = z
                h[l + 1] = self.ops[l](z)

            features = []
            for l in range(self.num_layers):
                h_flat = nn.functional.adaptive_avg_pool2d(
                    h[l + 1], 1).flatten(1)
                features.append(h_flat)
            concat = torch.cat(features, dim=1)
            output = self._spinal_concat_head(concat)
            return output, h, layer_inputs, layer_outputs

        for l in range(self.num_layers):
            h_in = h[l]
            for conn in self.connections:
                if conn.target == l + 1:
                    source_h = h[conn.source]
                    conn_out = conn.forward(source_h)
                    # Dimension check: compare last dim for flat, channel dim for spatial
                    if h_in.shape != conn_out.shape:
                        widths_in = [layer.in_features for layer in self.layers]
                        widths_out = [layer.out_features for layer in self.layers]
                        conns_info = []
                        for c in self.connections:
                            p = 'None' if c.projection is None else f'proj'
                            conns_info.append(f'{c.source}->{c.target}({p})')
                        raise RuntimeError(
                            f"Skip connection dim mismatch at l={l}: "
                            f"h_in={h_in.shape}, conn_out={conn_out.shape}, "
                            f"conn={conn.source}->{conn.target}, "
                            f"widths_out={widths_out}, conns={conns_info}"
                        )
                    h_in = h_in + conn_out

            layer_inputs[l] = h_in
            z = self.layers[l](h_in)
            layer_outputs[l] = z
            h[l + 1] = self.ops[l](z)

            # Flatten transition
            if self._is_spatial and l == self._flatten_position:
                if getattr(self, '_use_gap', False):
                    h[l + 1] = nn.functional.adaptive_avg_pool2d(
                        h[l + 1], 1).flatten(start_dim=1)
                else:
                    h[l + 1] = h[l + 1].flatten(start_dim=1)

        # Jumping Knowledge: attention-weighted aggregation of all layer outputs
        if self._jk_enabled:
            self._rebuild_jk()
            projs = []
            for l in range(self.num_layers):
                h_l = h[l + 1]
                if self._is_spatial and h_l.dim() == 4:
                    h_l = nn.functional.adaptive_avg_pool2d(h_l, 1).flatten(1)
                projs.append(self._jk_projections[l](h_l))
            logits = torch.cat([a for a in self._jk_attn_logits])
            attn_weights = torch.softmax(logits, dim=0)
            stacked = torch.stack(projs, dim=0)
            jk_out = (attn_weights.view(-1, 1, 1) * stacked).sum(dim=0)
            output = self.output_head(jk_out)
        else:
            output = self.output_head(h[self.num_layers])

        return output, h, layer_inputs, layer_outputs

    def _recompute_flatten_position(self):
        """Recompute the flatten boundary after layer add/remove.

        The flatten position is the index of the last spatial layer.
        Everything after is flat. Called after any layer surgery.
        """
        if not self._is_spatial:
            self._flatten_position = -1
            return
        self._flatten_position = -1
        for l in range(self.num_layers - 1, -1, -1):
            layer = self.layers[l]
            if hasattr(layer, 'mode') and layer.mode == "spatial":
                self._flatten_position = l
                return
        # No spatial layers found — edge case, shouldn't happen with min_spatial_layers

    def get_layer_width(self, layer_idx: int) -> int:
        """Get the actual output width of a layer."""
        return self.layers[layer_idx].out_features

    def get_layer_in_width(self, layer_idx: int) -> int:
        """Get the actual input width of a layer."""
        return self.layers[layer_idx].in_features

    def count_unenriched_layers(self) -> int:
        """Count layers whose ops pipeline has only trivial operations.

        A layer is 'unenriched' if its ops pipeline contains only trivial ops
        (activations, norms, regularization) -- meaning operation surgery hasn't
        discovered any useful computational operations for it yet.
        """
        trivial_ops = _TRIVIAL_OPS
        count = 0
        for l in range(self.num_layers):
            if l >= len(self.ops):
                break
            pipeline = self.ops[l]
            if hasattr(pipeline, 'operations'):
                if len(pipeline.operations) == 0:
                    count += 1
                elif all(
                    type(op.operation if isinstance(op, GatedOperation) else op).__name__
                    in trivial_ops
                    for op in pipeline.operations
                ):
                    count += 1
        return count

    def compute_architecture_cost(self) -> float:
        """Compute the REAL computational cost of the current architecture.

        For flat layers:
          C = sum_l (d_in(l) * d_out(l))  # matmul FLOPs
        For spatial layers:
          C = sum_l (C_out * C_in * k² * H_out * W_out)  # conv FLOPs
        Plus connection and operation costs.
        """
        cost = 0.0

        # Encoder cost
        if isinstance(self.encoder, BaseEncoder):
            cost += self.encoder.cost()
        elif isinstance(self.encoder, ConvStem):
            stem = self.encoder
            cost += stem.C_stem * stem.C_in * 9 * stem.H * stem.W  # 3x3 conv
        else:
            cost += self.encoder.in_features * self.encoder.out_features

        # Layer costs
        for layer in self.layers:
            if hasattr(layer, 'mode') and layer.mode == "spatial":
                C_out, H_out, W_out = layer.spatial_shape
                C_in = layer._C_in
                k = 3  # kernel size
                cost += C_out * C_in * k * k * H_out * W_out
                # Structural residual projection cost
                if layer.residual_proj is not None:
                    cost += C_out * C_in * H_out * W_out  # 1x1 conv
            else:
                cost += layer.in_features * layer.out_features

        # Connection costs
        for conn in self.connections:
            if conn.projection is not None:
                if isinstance(conn.projection, nn.Conv2d):
                    # Spatial projection: 1x1 conv
                    p = conn.projection
                    cost += p.out_channels * p.in_channels
                else:
                    cost += conn.projection.in_features * conn.projection.out_features

        # Operation costs — import Python types; try CUDA types too
        from .surgery import (Conv1dBlock, Conv2dBlock, FactoredEmbedding, MLPEmbedding,
                              GeometricEmbedding, PositionalEmbedding,
                              SelfAttentionOp, MultiHeadAttentionOp,
                              CrossAttentionOp, CausalAttentionOp,
                              SpatialConv2dOp, PointwiseConv2dOp, ChannelAttentionOp,
                              DepthwiseSeparableConv2dOp,
                              DilatedConv1dBlock, EMASmooth,
                              GatedLinearUnit, TemporalDiff, GRUOp,
                              ActivationNoise,
                              NeighborAggregation, GraphAttentionAggregation,
                              GraphDiffusion, SpectralConv, MessagePassingGIN,
                              DegreeScaling, GraphBranchedBlock,
                              PairNorm, GraphNorm, GraphPositionalEncoding,
                              APPNPPropagation, GraphSAGEMean, GraphSAGEGCN,
                              GATv2Aggregation, SGConv, DropEdgeAggregation,
                              MixHopConv, VirtualNodeOp, EdgeWeightedAggregation,
                              DirectionalDiffusion, AdaptiveGraphConv)
        try:
            from asann_cuda.ops import (
                Conv1dBlockCUDA, Conv2dBlockCUDA, FactoredEmbeddingCUDA,
                MLPEmbeddingCUDA, GeometricEmbeddingCUDA, PositionalEmbeddingCUDA,
                SelfAttentionOpCUDA, MultiHeadAttentionOpCUDA,
                CrossAttentionOpCUDA, CausalAttentionOpCUDA,
                SpatialConv2dOpCUDA, PointwiseConv2dOpCUDA,
                DepthwiseSeparableConv2dOpCUDA, ChannelAttentionOpCUDA,
                DilatedConv1dBlockCUDA, EMASmoothCUDA,
                GatedLinearUnitCUDA, TemporalDiffCUDA,
            )
            _cost_cuda = True
        except ImportError:
            _cost_cuda = False

        # Build type tuples that cover both Python and CUDA variants
        _DilConv1d = (DilatedConv1dBlock, DilatedConv1dBlockCUDA) if _cost_cuda else (DilatedConv1dBlock,)
        _EMA = (EMASmooth, EMASmoothCUDA) if _cost_cuda else (EMASmooth,)
        _GLU = (GatedLinearUnit, GatedLinearUnitCUDA) if _cost_cuda else (GatedLinearUnit,)
        _TDiff = (TemporalDiff, TemporalDiffCUDA) if _cost_cuda else (TemporalDiff,)
        _Conv1d = (Conv1dBlock, Conv1dBlockCUDA) if _cost_cuda else (Conv1dBlock,)
        _Conv2d = (Conv2dBlock, Conv2dBlockCUDA) if _cost_cuda else (Conv2dBlock,)
        _FacEmb = (FactoredEmbedding, FactoredEmbeddingCUDA) if _cost_cuda else (FactoredEmbedding,)
        _MLPEmb = (MLPEmbedding, MLPEmbeddingCUDA) if _cost_cuda else (MLPEmbedding,)
        _GeoEmb = (GeometricEmbedding, GeometricEmbeddingCUDA) if _cost_cuda else (GeometricEmbedding,)
        _PosEmb = (PositionalEmbedding, PositionalEmbeddingCUDA) if _cost_cuda else (PositionalEmbedding,)
        _SelfAttn = (SelfAttentionOp, SelfAttentionOpCUDA) if _cost_cuda else (SelfAttentionOp,)
        _MHAttn = (MultiHeadAttentionOp, MultiHeadAttentionOpCUDA) if _cost_cuda else (MultiHeadAttentionOp,)
        _CrossAttn = (CrossAttentionOp, CrossAttentionOpCUDA) if _cost_cuda else (CrossAttentionOp,)
        _CausalAttn = (CausalAttentionOp, CausalAttentionOpCUDA) if _cost_cuda else (CausalAttentionOp,)
        _SpConv2d = (SpatialConv2dOp, SpatialConv2dOpCUDA) if _cost_cuda else (SpatialConv2dOp,)
        _PwConv2d = (PointwiseConv2dOp, PointwiseConv2dOpCUDA) if _cost_cuda else (PointwiseConv2dOp,)
        _ChAttn = (ChannelAttentionOp, ChannelAttentionOpCUDA) if _cost_cuda else (ChannelAttentionOp,)
        _DWSep = (DepthwiseSeparableConv2dOp, DepthwiseSeparableConv2dOpCUDA) if _cost_cuda else (DepthwiseSeparableConv2dOp,)

        for l in range(self.num_layers):
            layer = self.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            if is_spatial:
                C, H, W = layer.spatial_shape
                for op in self.ops[l].operations:
                    # Unwrap GatedOperation for cost computation
                    if isinstance(op, GatedOperation):
                        op = op.operation
                    if isinstance(op, _DWSep):
                        k = op.kernel_size
                        cost += C * k * k * H * W + 4 * C * H * W + C * C * H * W
                    elif isinstance(op, _SpConv2d):
                        cost += op.kernel_size * op.kernel_size * C * H * W
                    elif isinstance(op, _PwConv2d):
                        cost += C * C * H * W
                    elif isinstance(op, _ChAttn):
                        cost += 2 * C * op.reduction + C
                    elif isinstance(op, nn.BatchNorm2d):
                        cost += 4 * C * H * W
                    else:
                        cost += C * H * W
            else:
                d = layer.out_features
                for op in self.ops[l].operations:
                    # Unwrap GatedOperation for cost computation
                    if isinstance(op, GatedOperation):
                        op = op.operation
                    if isinstance(op, _DilConv1d):
                        cost += op.kernel_size * d
                    elif isinstance(op, _EMA):
                        cost += 3 * d
                    elif isinstance(op, _GLU):
                        cost += 2 * d * d + d
                    elif isinstance(op, _TDiff):
                        cost += d
                    elif isinstance(op, _Conv1d):
                        cost += op.kernel_size * d
                    elif isinstance(op, _Conv2d):
                        cost += op.kernel_size * op.kernel_size * op.C * d
                    elif isinstance(op, _FacEmb):
                        cost += 2 * d * op.rank
                    elif isinstance(op, _MLPEmb):
                        cost += 2 * d * op.hidden
                    elif isinstance(op, _GeoEmb):
                        cost += 4 * d
                    elif isinstance(op, _PosEmb):
                        cost += d
                    elif isinstance(op, _SelfAttn):
                        cost += d * d + 3 * d * op.rank
                    elif isinstance(op, _MHAttn):
                        cost += op.num_heads * (d * d + 3 * d * op.head_rank)
                    elif isinstance(op, _CrossAttn):
                        cost += d * (op.rank + op.num_memories)
                    elif isinstance(op, _CausalAttn):
                        cost += d * d + 3 * d * op.rank
                    elif isinstance(op, (nn.BatchNorm1d, nn.LayerNorm)):
                        cost += 4 * d
                    # Graph operation costs
                    elif isinstance(op, NeighborAggregation):
                        cost += d * d + d  # Linear + sparse matmul
                    elif isinstance(op, GraphAttentionAggregation):
                        cost += d * d + 2 * d  # Linear + attention params
                    elif isinstance(op, MessagePassingGIN):
                        cost += 2 * d * d + d  # 2-layer MLP
                    elif isinstance(op, SpectralConv):
                        cost += d * (op.K + 1)  # Chebyshev coefficients
                    elif isinstance(op, GraphDiffusion):
                        cost += d * d + op.max_hops + 1  # Linear + hop weights
                    elif isinstance(op, GraphBranchedBlock):
                        cost += 2 * (d * d + 2 * d) + 2 * d * d  # Two branches + merge
                    elif isinstance(op, GraphPositionalEncoding):
                        cost += op.k * d  # Projection from k eigenvecs to d
                    elif isinstance(op, APPNPPropagation):
                        cost += d * d + d  # Linear + sparse iterations
                    elif isinstance(op, (GraphSAGEMean,)):
                        cost += 2 * d * d + d  # Linear(2d->d) concat projection
                    elif isinstance(op, (GraphSAGEGCN, SGConv)):
                        cost += d * d  # Single linear
                    elif isinstance(op, GATv2Aggregation):
                        cost += d * d + 2 * d  # Two linears + attention vector
                    elif isinstance(op, DropEdgeAggregation):
                        cost += d * d + d  # Linear + sparse matmul
                    elif isinstance(op, MixHopConv):
                        cost += d * d * (op.max_hops + 1)  # Projection from concatenated hops
                    elif isinstance(op, VirtualNodeOp):
                        cost += 2 * d * d + d  # 2-layer MLP
                    elif isinstance(op, EdgeWeightedAggregation):
                        cost += 2 * d * d + d  # Edge MLP(2d->d->1)
                    elif isinstance(op, DirectionalDiffusion):
                        cost += 2 * d * d + (op.max_hops + 1)  # 2 Linears + hop weights
                    elif isinstance(op, AdaptiveGraphConv):
                        N = op.num_nodes
                        cost += d * d + 2 * N * op.d_emb  # Linear + E_src + E_dst
                    elif isinstance(op, (DegreeScaling, PairNorm, GraphNorm)):
                        cost += d  # Lightweight
                    elif isinstance(op, GRUOp):
                        # GRU: 3 gates × (input_size × hidden_size + hidden_size × hidden_size + 2 × hidden_size)
                        cs = op.chunk_size
                        cost += 3 * (cs * cs + cs * cs + 2 * cs)  # 6*cs^2 + 6*cs
                    elif isinstance(op, ActivationNoise):
                        cost += 0  # Parameter-free, zero trainable cost
                    else:
                        cost += d

        # Output head cost
        cost += self.output_head.in_features * self.output_head.out_features

        # JK projection costs
        if self._jk_enabled:
            for proj in self._jk_projections:
                cost += proj.in_features * proj.out_features

        return cost

    # ==================== Immunosuppression Gate Management ====================

    def advance_surgery_gates(self) -> bool:
        """Advance all GatedOperation gates by one epoch.

        Called once per epoch by the trainer. Gradually increases the blending
        factor of recently inserted operations (immunosuppression ramp-up).
        After warmup completes, absorbs the gate wrapper for zero overhead.

        Also advances encoder bridge gates during encoder switching.

        Returns:
            True if any architecture change occurred (bridge absorbed, gates
            absorbed) that would invalidate a saved state_dict snapshot.
        """
        architecture_changed = False

        # Advance operation pipeline gates
        for l in range(self.num_layers):
            self.ops[l].advance_gates()
            absorbed_any = self.ops[l].absorb_ready_gates()
            if absorbed_any:
                architecture_changed = True

        # Advance encoder bridge gate
        if isinstance(self.encoder, GatedEncoderBridge):
            self.encoder.advance_epoch()
            if self.encoder.is_ready_to_absorb:
                absorbed = self.encoder.absorb()
                self.encoder = absorbed
                architecture_changed = True
                print(f"  [ENCODER] Bridge absorbed -> {absorbed.describe()}")

        return architecture_changed

    def has_active_gates(self) -> bool:
        """True if any GatedOperation or encoder bridge is still in warmup."""
        # Check encoder bridge
        if isinstance(self.encoder, GatedEncoderBridge) and not self.encoder._absorbed:
            return True
        # Check operation gates
        for l in range(self.num_layers):
            for op in self.ops[l].operations:
                if isinstance(op, GatedOperation) and not op._absorbed:
                    return True
        return False

    def describe_architecture(self) -> Dict[str, Any]:
        """Return a complete description of the current architecture."""
        if isinstance(self.encoder, BaseEncoder):
            ip_desc = self.encoder.describe()
        elif isinstance(self.encoder, ConvStem):
            ip_desc = f"ConvStem({self.encoder.C_in}->{self.encoder.C_stem}ch, {self.encoder.H}x{self.encoder.W})"
        else:
            ip_desc = f"{self.encoder.in_features} -> {self.encoder.out_features}"

        desc = {
            "num_layers": self.num_layers,
            "input_dim": self.d_input,
            "output_dim": self.d_output,
            "encoder": ip_desc,
            "input_projection": ip_desc,  # backward compat key
            "is_spatial": self._is_spatial,
            "flatten_position": self._flatten_position,
            "layers": [],
            "connections": [],
            "total_parameters": sum(p.numel() for p in self.all_parameters()),
            "architecture_cost": self.compute_architecture_cost(),
        }

        for l in range(self.num_layers):
            layer = self.layers[l]
            layer_desc = {
                "index": l,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "operations": self.ops[l].describe(),
            }
            if hasattr(layer, 'mode'):
                layer_desc["mode"] = layer.mode
                if layer.mode == "spatial":
                    layer_desc["channels"] = layer.out_channels
                    layer_desc["spatial_shape"] = layer.spatial_shape
                    layer_desc["stride"] = layer._stride
            desc["layers"].append(layer_desc)

        for conn in self.connections:
            conn_desc = {
                "source": conn.source,
                "target": conn.target,
                "has_projection": conn.projection is not None,
                "scale": conn.scale.item(),
                "utility": conn.utility(),
            }
            desc["connections"].append(conn_desc)

        desc["output_head"] = f"{self.output_head.in_features} -> {self.output_head.out_features}"
        desc["architecture_stable"] = self.architecture_stable

        # Jumping Knowledge info
        if self._jk_enabled and len(self._jk_attn_logits) > 0:
            jk_weights = torch.softmax(
                torch.cat([a.data for a in self._jk_attn_logits]), dim=0
            ).tolist()
            desc["jk_enabled"] = True
            desc["jk_weights"] = jk_weights
        else:
            desc["jk_enabled"] = False

        return desc

    def all_parameters(self):
        """Yield all parameters including connection parameters."""
        yield from self.parameters()
        for conn in self.connections:
            yield from conn.parameters()

    def connection_parameters(self):
        """Yield only connection parameters (for separate optimizer groups)."""
        for conn in self.connections:
            yield from conn.parameters()

    def clear_stats(self):
        """Clear all accumulated statistics."""
        self._activation_history.clear()
        self._gradient_history.clear()
        self._layer_input_history.clear()
        self._layer_output_history.clear()
