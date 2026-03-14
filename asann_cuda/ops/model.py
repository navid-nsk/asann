import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple

from .asann_layer import ASANNLayerCUDA
from .conv_stem import ConvStemCUDA
from .skip_connection import SkipConnectionCUDA
from .gap_flatten import GAPFlattenCUDA


class OperationPipelineCUDA(nn.Module):
    """Sequential pipeline of operations (same as original)."""

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

    def add_operation(self, operation: nn.Module, position: int):
        ops_list = list(self.operations)
        ops_list.insert(position, operation)
        self.operations = nn.ModuleList(ops_list)

    def remove_operation(self, position: int):
        ops_list = list(self.operations)
        ops_list.pop(position)
        self.operations = nn.ModuleList(ops_list)

    def describe(self) -> List[str]:
        if self.num_operations == 0:
            return ["Identity"]
        from .conv1d_block import Conv1dBlockCUDA
        from .conv2d_block import Conv2dBlockCUDA
        from .factored_embedding import FactoredEmbeddingCUDA
        from .mlp_embedding import MLPEmbeddingCUDA
        from .geometric_embedding import GeometricEmbeddingCUDA
        from .positional_embedding import PositionalEmbeddingCUDA
        from .self_attention import SelfAttentionOpCUDA
        from .multihead_attention import MultiHeadAttentionOpCUDA
        from .cross_attention import CrossAttentionOpCUDA
        from .causal_attention import CausalAttentionOpCUDA
        result = []
        for op in self.operations:
            if isinstance(op, Conv1dBlockCUDA):
                result.append(f"Conv1d(k={op.kernel_size})")
            elif isinstance(op, Conv2dBlockCUDA):
                result.append(f"Conv2d(k={op.kernel_size},{op.C}x{op.H}x{op.W})")
            elif isinstance(op, FactoredEmbeddingCUDA):
                result.append(f"FactEmbed(r={op.rank})")
            elif isinstance(op, MLPEmbeddingCUDA):
                result.append(f"MLPEmbed(h={op.hidden})")
            elif isinstance(op, GeometricEmbeddingCUDA):
                result.append("GeoEmbed")
            elif isinstance(op, PositionalEmbeddingCUDA):
                result.append("PosEmbed")
            elif isinstance(op, SelfAttentionOpCUDA):
                result.append(f"SelfAttn(r={op.rank})")
            elif isinstance(op, MultiHeadAttentionOpCUDA):
                result.append(f"MHAttn(h={op.num_heads})")
            elif isinstance(op, CrossAttentionOpCUDA):
                result.append(f"CrossAttn(m={op.num_memories})")
            elif isinstance(op, CausalAttentionOpCUDA):
                result.append(f"CausalAttn(r={op.rank})")
            else:
                result.append(type(op).__name__)
        return result


class SpatialOperationPipelineCUDA(nn.Module):
    """Sequential pipeline for spatial operations (same as original)."""

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

    def add_operation(self, operation: nn.Module, position: int):
        ops_list = list(self.operations)
        ops_list.insert(position, operation)
        self.operations = nn.ModuleList(ops_list)

    def remove_operation(self, position: int):
        ops_list = list(self.operations)
        ops_list.pop(position)
        self.operations = nn.ModuleList(ops_list)

    def describe(self) -> List[str]:
        if self.num_operations == 0:
            return ["Identity"]
        from .spatial_conv2d import SpatialConv2dOpCUDA
        from .pointwise_conv2d import PointwiseConv2dOpCUDA
        from .channel_attention import ChannelAttentionOpCUDA
        from .dw_separable_conv2d import DepthwiseSeparableConv2dOpCUDA
        result = []
        for op in self.operations:
            if isinstance(op, DepthwiseSeparableConv2dOpCUDA):
                result.append(f"DWSep(k={op.kernel_size})")
            elif isinstance(op, SpatialConv2dOpCUDA):
                result.append(f"SpatConv2d(k={op.kernel_size})")
            elif isinstance(op, PointwiseConv2dOpCUDA):
                result.append(f"PW1x1(C={op.C})")
            elif isinstance(op, ChannelAttentionOpCUDA):
                result.append(f"SE(C={op.C},r={op.reduction_ratio})")
            elif isinstance(op, nn.BatchNorm2d):
                result.append(f"BN2d({op.num_features})")
            else:
                result.append(type(op).__name__)
        return result


class ASANNModelCUDA(nn.Module):
    """Drop-in CUDA replacement for ASANNModel.

    Same interface, same behavior, using CUDA-accelerated operations.
    """

    def __init__(self, d_input: int, d_output: int, config):
        super().__init__()
        self.config = config
        self.d_input = d_input
        self.d_output = d_output

        self._is_spatial = config.spatial_shape is not None

        self.layers = nn.ModuleList()
        self.ops = nn.ModuleList()

        if self._is_spatial:
            C_in, H, W = config.spatial_shape
            C_stem = config.c_stem_init
            self._effective_d_init = C_stem * H * W

            self.input_projection = ConvStemCUDA(C_in, C_stem, H, W)

            C_cur = C_stem
            H_cur, W_cur = H, W
            n_downsample = min(config.spatial_downsample_stages,
                               config.initial_num_layers // 2)

            layer_idx = 0
            total_layers = max(config.initial_num_layers, n_downsample * 2)

            for stage in range(n_downsample):
                layer = ASANNLayerCUDA(mode="spatial", C_in=C_cur, C_out=C_cur,
                                       H=H_cur, W=W_cur)
                self.layers.append(layer)
                pipeline = SpatialOperationPipelineCUDA()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)
                layer_idx += 1

                C_next = min(C_cur * 2, config.max_channels)
                H_next = H_cur // 2
                W_next = W_cur // 2
                if H_next >= config.min_spatial_resolution:
                    layer = ASANNLayerCUDA(mode="spatial", C_in=C_cur, C_out=C_next,
                                           H=H_cur, W=W_cur, stride=2)
                    self.layers.append(layer)
                    pipeline = SpatialOperationPipelineCUDA()
                    pipeline.add_operation(nn.ReLU(), 0)
                    self.ops.append(pipeline)
                    C_cur = C_next
                    H_cur, W_cur = H_next, W_next
                    layer_idx += 1

            while layer_idx < total_layers:
                layer = ASANNLayerCUDA(mode="spatial", C_in=C_cur, C_out=C_cur,
                                       H=H_cur, W=W_cur)
                self.layers.append(layer)
                pipeline = SpatialOperationPipelineCUDA()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)
                layer_idx += 1

            self._flatten_position = len(self.layers) - 1
            self._use_gap = True
            gap_d = C_cur
            self.output_head = nn.Linear(gap_d, d_output)
            self._gap_flatten = GAPFlattenCUDA()
            print(f"  Spatial mode (CUDA): stem={C_in}->{C_stem}ch, "
                  f"stages={n_downsample} downsample, "
                  f"final={C_cur}ch@{H_cur}x{W_cur}, "
                  f"GAP_d={gap_d}")

        else:
            d_init = config.d_init
            if config.d_init_auto and d_input > config.d_init * config.d_init_ratio:
                auto_d = int(d_input / config.d_init_ratio)
                d_init = max(config.d_init_min, min(config.d_init_max, auto_d))
                d_init = ((d_init + 7) // 8) * 8
                print(f"  Auto d_init: {config.d_init} -> {d_init} (d_input={d_input})")
            self._effective_d_init = d_init
            self._flatten_position = -1

            self.input_projection = nn.Linear(d_input, d_init)

            for _ in range(config.initial_num_layers):
                self.layers.append(nn.Linear(d_init, d_init))

            for _ in range(config.initial_num_layers):
                pipeline = OperationPipelineCUDA()
                pipeline.add_operation(nn.ReLU(), 0)
                self.ops.append(pipeline)

            self.output_head = nn.Linear(d_init, d_output)

        self.connections: List[SkipConnectionCUDA] = []
        self.architecture_stable = False

        self._activation_history: Dict[int, List[torch.Tensor]] = {}
        self._gradient_history: Dict[int, List[torch.Tensor]] = {}
        self._layer_input_history: Dict[int, List[torch.Tensor]] = {}
        self._layer_output_history: Dict[int, List[torch.Tensor]] = {}

        self.to(config.device)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: Dict[int, torch.Tensor] = {}

        if self._is_spatial:
            B = x.shape[0]
            C_in, H_in, W_in = self.config.spatial_shape
            x_spatial = x.view(B, C_in, H_in, W_in)
            h[0] = self.input_projection(x_spatial)
        else:
            h[0] = self.input_projection(x)

        for l in range(self.num_layers):
            h_in = h[l]

            for conn in self.connections:
                if conn.target == l + 1:
                    source_h = h[conn.source]
                    h_in = h_in + conn.forward(source_h)

            z = self.layers[l](h_in)
            h[l + 1] = self.ops[l](z)

            if self._is_spatial and l == self._flatten_position:
                if getattr(self, '_use_gap', False):
                    h[l + 1] = self._gap_flatten(h[l + 1])
                else:
                    h[l + 1] = h[l + 1].flatten(start_dim=1)

        return self.output_head(h[self.num_layers])

    def forward_with_intermediates(self, x: torch.Tensor) -> tuple:
        h: Dict[int, torch.Tensor] = {}

        if self._is_spatial:
            B = x.shape[0]
            C_in, H_in, W_in = self.config.spatial_shape
            x_spatial = x.view(B, C_in, H_in, W_in)
            h[0] = self.input_projection(x_spatial)
        else:
            h[0] = self.input_projection(x)

        layer_inputs: Dict[int, torch.Tensor] = {}
        layer_outputs: Dict[int, torch.Tensor] = {}

        for l in range(self.num_layers):
            h_in = h[l]
            for conn in self.connections:
                if conn.target == l + 1:
                    source_h = h[conn.source]
                    conn_out = conn.forward(source_h)
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

            if self._is_spatial and l == self._flatten_position:
                if getattr(self, '_use_gap', False):
                    h[l + 1] = self._gap_flatten(h[l + 1])
                else:
                    h[l + 1] = h[l + 1].flatten(start_dim=1)

        output = self.output_head(h[self.num_layers])
        return output, h, layer_inputs, layer_outputs

    def _recompute_flatten_position(self):
        if not self._is_spatial:
            self._flatten_position = -1
            return
        self._flatten_position = -1
        for l in range(self.num_layers - 1, -1, -1):
            layer = self.layers[l]
            if hasattr(layer, 'mode') and layer.mode == "spatial":
                self._flatten_position = l
                return

    def get_layer_width(self, layer_idx: int) -> int:
        return self.layers[layer_idx].out_features

    def get_layer_in_width(self, layer_idx: int) -> int:
        return self.layers[layer_idx].in_features

    def compute_architecture_cost(self) -> float:
        cost = 0.0

        if isinstance(self.input_projection, ConvStemCUDA):
            stem = self.input_projection
            cost += stem.C_stem * stem.C_in * 9 * stem.H * stem.W
        else:
            cost += self.input_projection.in_features * self.input_projection.out_features

        for layer in self.layers:
            if hasattr(layer, 'mode') and layer.mode == "spatial":
                C_out, H_out, W_out = layer.spatial_shape
                C_in = layer._C_in
                k = 3
                cost += C_out * C_in * k * k * H_out * W_out
                if layer.residual_proj is not None:
                    cost += C_out * C_in * H_out * W_out
            else:
                cost += layer.in_features * layer.out_features

        for conn in self.connections:
            if conn.projection is not None:
                if isinstance(conn.projection, nn.Conv2d):
                    p = conn.projection
                    cost += p.out_channels * p.in_channels
                else:
                    cost += conn.projection.in_features * conn.projection.out_features

        from .conv1d_block import Conv1dBlockCUDA
        from .conv2d_block import Conv2dBlockCUDA
        from .factored_embedding import FactoredEmbeddingCUDA
        from .mlp_embedding import MLPEmbeddingCUDA
        from .geometric_embedding import GeometricEmbeddingCUDA
        from .positional_embedding import PositionalEmbeddingCUDA
        from .self_attention import SelfAttentionOpCUDA
        from .multihead_attention import MultiHeadAttentionOpCUDA
        from .cross_attention import CrossAttentionOpCUDA
        from .causal_attention import CausalAttentionOpCUDA
        from .spatial_conv2d import SpatialConv2dOpCUDA
        from .pointwise_conv2d import PointwiseConv2dOpCUDA
        from .channel_attention import ChannelAttentionOpCUDA
        from .dw_separable_conv2d import DepthwiseSeparableConv2dOpCUDA

        for l in range(self.num_layers):
            layer = self.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            if is_spatial:
                C, H, W = layer.spatial_shape
                for op in self.ops[l].operations:
                    if isinstance(op, DepthwiseSeparableConv2dOpCUDA):
                        k = op.kernel_size
                        cost += C * k * k * H * W + 4 * C * H * W + C * C * H * W
                    elif isinstance(op, SpatialConv2dOpCUDA):
                        cost += op.kernel_size * op.kernel_size * C * H * W
                    elif isinstance(op, PointwiseConv2dOpCUDA):
                        cost += C * C * H * W
                    elif isinstance(op, ChannelAttentionOpCUDA):
                        cost += 2 * C * op.reduction + C
                    elif isinstance(op, nn.BatchNorm2d):
                        cost += 4 * C * H * W
                    else:
                        cost += C * H * W
            else:
                d = layer.out_features
                for op in self.ops[l].operations:
                    if isinstance(op, Conv1dBlockCUDA):
                        cost += op.kernel_size * d
                    elif isinstance(op, Conv2dBlockCUDA):
                        cost += op.kernel_size * op.kernel_size * op.C * d
                    elif isinstance(op, FactoredEmbeddingCUDA):
                        cost += 2 * d * op.rank
                    elif isinstance(op, MLPEmbeddingCUDA):
                        cost += 2 * d * op.hidden
                    elif isinstance(op, GeometricEmbeddingCUDA):
                        cost += 4 * d
                    elif isinstance(op, PositionalEmbeddingCUDA):
                        cost += d
                    elif isinstance(op, SelfAttentionOpCUDA):
                        cost += d * d + 3 * d * op.rank
                    elif isinstance(op, MultiHeadAttentionOpCUDA):
                        cost += op.num_heads * (d * d + 3 * d * op.head_rank)
                    elif isinstance(op, CrossAttentionOpCUDA):
                        cost += d * (op.rank + op.num_memories)
                    elif isinstance(op, CausalAttentionOpCUDA):
                        cost += d * d + 3 * d * op.rank
                    elif isinstance(op, (nn.BatchNorm1d, nn.LayerNorm)):
                        cost += 4 * d
                    else:
                        cost += d

        cost += self.output_head.in_features * self.output_head.out_features
        return cost

    def describe_architecture(self) -> Dict[str, Any]:
        if isinstance(self.input_projection, ConvStemCUDA):
            ip_desc = f"ConvStem({self.input_projection.C_in}->{self.input_projection.C_stem}ch, {self.input_projection.H}x{self.input_projection.W})"
        else:
            ip_desc = f"{self.input_projection.in_features} -> {self.input_projection.out_features}"

        desc = {
            "num_layers": self.num_layers,
            "input_dim": self.d_input,
            "output_dim": self.d_output,
            "input_projection": ip_desc,
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
        return desc

    def all_parameters(self):
        yield from self.parameters()
        for conn in self.connections:
            yield from conn.parameters()

    def connection_parameters(self):
        for conn in self.connections:
            yield from conn.parameters()

    def clear_stats(self):
        self._activation_history.clear()
        self._gradient_history.clear()
        self._layer_input_history.clear()
        self._layer_output_history.clear()
