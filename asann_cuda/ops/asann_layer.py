import torch
import torch.nn as nn
from typing import Optional, Tuple


class ASANNLayerCUDA(nn.Module):
    """Drop-in CUDA replacement for ASANNLayer.

    Polymorphic layer: either flat (nn.Linear) or spatial (Conv2d + BN + residual).
    Uses standard PyTorch ops since Conv2d and BN already have optimized CUDA implementations.
    """

    def __init__(self, mode: str, **kwargs):
        super().__init__()
        self.mode = mode

        if mode == "spatial":
            C_in = kwargs["C_in"]
            C_out = kwargs["C_out"]
            H = kwargs["H"]
            W = kwargs["W"]
            stride = kwargs.get("stride", 1)
            self._C_in = C_in
            self._stride = stride

            self.conv = nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
            self.bn = nn.BatchNorm2d(C_out)

            H_out = H // stride
            W_out = W // stride
            self.spatial_shape: Optional[Tuple[int, int, int]] = (C_out, H_out, W_out)

            self.has_structural_residual = True
            if C_in != C_out or stride != 1:
                self.residual_proj = nn.Conv2d(C_in, C_out, kernel_size=1,
                                               stride=stride, bias=False)
            else:
                self.residual_proj = None

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

        else:
            raise ValueError(f"Unknown layer mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "spatial":
            out = self.bn(self.conv(x))
            if self.has_structural_residual:
                identity = x if self.residual_proj is None else self.residual_proj(x)
                out = out + identity
            return out
        else:
            return self.linear(x)

    @property
    def out_features(self) -> int:
        if self.mode == "flat":
            return self.linear.out_features
        else:
            C, H, W = self.spatial_shape
            return C * H * W

    @out_features.setter
    def out_features(self, value):
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
        if self.mode == "flat":
            self.linear.in_features = value

    @property
    def out_channels(self) -> int:
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
            return self.bn.bias

    @bias.setter
    def bias(self, value):
        if self.mode == "flat":
            self.linear.bias = value

    _DELEGATED_PROPERTIES = frozenset({'weight', 'bias', 'out_features', 'in_features'})

    def __setattr__(self, name: str, value):
        if name in self._DELEGATED_PROPERTIES and hasattr(self, 'mode'):
            prop = type(self).__dict__.get(name)
            if prop is not None and isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return
        super().__setattr__(name, value)
