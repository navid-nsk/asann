import torch
import torch.nn as nn
import asann_cuda_ops


class SkipScaleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, scale):
        scale_val = scale.item()
        output = asann_cuda_ops.skip_connection_forward(x, scale_val)
        ctx.save_for_backward(x, scale)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        scale_val = scale.item()
        grads = asann_cuda_ops.skip_connection_backward(grad_output, x, scale_val)
        return grads[0], grads[1]


class SkipConnectionCUDA:
    """Drop-in CUDA replacement for SkipConnection.

    Mirrors the original SkipConnection interface exactly.
    """

    def __init__(
        self,
        source_idx: int,
        target_idx: int,
        d_source: int,
        d_target: int,
        init_scale: float = 0.01,
        device: str = "cpu",
        spatial_source_shape=None,
        spatial_target_shape=None,
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
        self.low_utility_count = 0

    def forward(self, source_h: torch.Tensor) -> torch.Tensor:
        x = source_h
        if self.projection is not None:
            x = self.projection(x)
        # Handle spatial dim mismatch via adaptive pooling
        if (self.spatial_target_shape is not None and x.dim() == 4):
            _, H_tgt, W_tgt = self.spatial_target_shape
            if x.shape[2] != H_tgt or x.shape[3] != W_tgt:
                x = nn.functional.adaptive_avg_pool2d(x, (H_tgt, W_tgt))
        return SkipScaleFunction.apply(x, self.scale)

    def utility(self) -> float:
        scale_val = self.scale.item()
        if scale_val < 0:
            return 0.0
        if self.projection is not None:
            proj_norm = self.projection.weight.data.norm().item()
        else:
            proj_norm = 1.0
        return scale_val * proj_norm

    def parameters(self):
        yield self.scale
        if self.projection is not None:
            yield from self.projection.parameters()

    def to(self, device):
        self.scale = nn.Parameter(self.scale.data.to(device))
        if self.projection is not None:
            self.projection = self.projection.to(device)
        return self
