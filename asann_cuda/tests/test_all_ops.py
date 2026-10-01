"""Comprehensive numerical equivalence tests for CUDA vs Python ops.

Tests every operation forward + backward, comparing CUDA implementations
against the original Python implementations from asann/surgery.py.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from asann.surgery import (
    Conv1dBlock, Conv2dBlock, FactoredEmbedding, MLPEmbedding,
    GeometricEmbedding, PositionalEmbedding, SelfAttentionOp,
    MultiHeadAttentionOp, CrossAttentionOp, CausalAttentionOp,
    SpatialConv2dOp, PointwiseConv2dOp, DepthwiseSeparableConv2dOp,
    ChannelAttentionOp,
    DilatedConv1dBlock, EMASmooth, GatedLinearUnit, TemporalDiff,
)
from asann_cuda.ops import (
    PositionalEmbeddingCUDA, FactoredEmbeddingCUDA, MLPEmbeddingCUDA,
    GeometricEmbeddingCUDA, GAPFlattenCUDA, Conv1dBlockCUDA, Conv2dBlockCUDA,
    SpatialConv2dOpCUDA, PointwiseConv2dOpCUDA, DepthwiseSeparableConv2dOpCUDA,
    ChannelAttentionOpCUDA, SelfAttentionOpCUDA, MultiHeadAttentionOpCUDA,
    CrossAttentionOpCUDA, CausalAttentionOpCUDA, SkipConnectionCUDA,
    DilatedConv1dBlockCUDA, EMASmoothCUDA, GatedLinearUnitCUDA, TemporalDiffCUDA,
)

device = 'cuda'
ATOL = 1e-5
RTOL = 1e-5
GRAD_ATOL = 1e-4
GRAD_RTOL = 1e-4


def copy_params(src, dst):
    """Copy all parameters from src module to dst module by name."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd:
        if key in src_sd:
            dst_sd[key] = src_sd[key].clone()
    dst.load_state_dict(dst_sd)


def check_forward(name, py_out, cuda_out, atol=ATOL, rtol=RTOL):
    """Check forward pass outputs match."""
    if not torch.allclose(py_out, cuda_out, atol=atol, rtol=rtol):
        max_diff = (py_out - cuda_out).abs().max().item()
        mean_diff = (py_out - cuda_out).abs().mean().item()
        print(f"  FAIL {name} forward: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        return False
    print(f"  PASS {name} forward")
    return True


def check_grad(name, param_name, py_grad, cuda_grad, atol=GRAD_ATOL, rtol=GRAD_RTOL):
    """Check backward pass gradients match."""
    if py_grad is None and cuda_grad is None:
        return True
    if py_grad is None or cuda_grad is None:
        print(f"  FAIL {name} grad {param_name}: one is None")
        return False
    if not torch.allclose(py_grad, cuda_grad, atol=atol, rtol=rtol):
        max_diff = (py_grad - cuda_grad).abs().max().item()
        mean_diff = (py_grad - cuda_grad).abs().mean().item()
        print(f"  FAIL {name} grad {param_name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        return False
    return True


def test_flat_op(name, py_cls, cuda_cls, d=16, B=4, extra_kwargs=None):
    """Test a flat [B, d] -> [B, d] operation."""
    print(f"\nTesting {name}...")
    kwargs = extra_kwargs or {}

    torch.manual_seed(42)
    py_op = py_cls(d, **kwargs).to(device)

    torch.manual_seed(42)
    cuda_op = cuda_cls(d, **kwargs).to(device)

    # Copy params from python to cuda to ensure identical weights
    copy_params(py_op, cuda_op)

    # Forward
    torch.manual_seed(123)
    x = torch.randn(B, d, device=device, requires_grad=True)
    x_cuda = x.clone().detach().requires_grad_(True)

    py_out = py_op(x)
    cuda_out = cuda_op(x_cuda)

    passed = check_forward(name, py_out, cuda_out)

    # Backward
    grad_out = torch.randn_like(py_out)
    py_out.backward(grad_out)
    cuda_out.backward(grad_out.clone())

    # Check input grad
    if not check_grad(name, "x", x.grad, x_cuda.grad):
        passed = False

    # Check parameter grads
    for (pname, pp), (_, cp) in zip(py_op.named_parameters(), cuda_op.named_parameters()):
        if not check_grad(name, pname, pp.grad, cp.grad):
            passed = False

    if passed:
        print(f"  PASS {name} backward (all grads match)")
    return passed


def test_spatial_op(name, py_cls, cuda_cls, C=4, H=8, W=8, B=2, extra_kwargs=None):
    """Test a spatial [B, C, H, W] -> [B, C, H, W] operation."""
    print(f"\nTesting {name}...")
    kwargs = extra_kwargs or {}

    torch.manual_seed(42)
    py_op = py_cls(C, H, W, **kwargs).to(device)

    torch.manual_seed(42)
    cuda_op = cuda_cls(C, H, W, **kwargs).to(device)

    copy_params(py_op, cuda_op)

    torch.manual_seed(123)
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    x_cuda = x.clone().detach().requires_grad_(True)

    py_op.train()
    cuda_op.train()

    py_out = py_op(x)
    cuda_out = cuda_op(x_cuda)

    passed = check_forward(name, py_out, cuda_out)

    grad_out = torch.randn_like(py_out)
    py_out.backward(grad_out)
    cuda_out.backward(grad_out.clone())

    if not check_grad(name, "x", x.grad, x_cuda.grad):
        passed = False

    for (pname, pp), (_, cp) in zip(py_op.named_parameters(), cuda_op.named_parameters()):
        if not check_grad(name, pname, pp.grad, cp.grad):
            passed = False

    if passed:
        print(f"  PASS {name} backward (all grads match)")
    return passed


def test_channel_attention(C=8, B=2, H=8, W=8):
    """Special test for channel attention (different constructor signature)."""
    name = "ChannelAttentionOp"
    print(f"\nTesting {name}...")

    torch.manual_seed(42)
    py_op = ChannelAttentionOp(C).to(device)

    torch.manual_seed(42)
    cuda_op = ChannelAttentionOpCUDA(C).to(device)

    copy_params(py_op, cuda_op)

    torch.manual_seed(123)
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    x_cuda = x.clone().detach().requires_grad_(True)

    py_out = py_op(x)
    cuda_out = cuda_op(x_cuda)

    passed = check_forward(name, py_out, cuda_out)

    grad_out = torch.randn_like(py_out)
    py_out.backward(grad_out)
    cuda_out.backward(grad_out.clone())

    if not check_grad(name, "x", x.grad, x_cuda.grad):
        passed = False

    for (pname, pp), (_, cp) in zip(py_op.named_parameters(), cuda_op.named_parameters()):
        if not check_grad(name, pname, pp.grad, cp.grad):
            passed = False

    if passed:
        print(f"  PASS {name} backward (all grads match)")
    return passed


def test_gap_flatten(C=4, H=8, W=8, B=2):
    """Test GAP Flatten (no Python equivalent in surgery.py - compare with PyTorch)."""
    name = "GAPFlatten"
    print(f"\nTesting {name}...")

    cuda_op = GAPFlattenCUDA().to(device)

    torch.manual_seed(123)
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    x_py = x.clone().detach().requires_grad_(True)

    # Python reference: adaptive_avg_pool2d -> flatten
    py_out = F.adaptive_avg_pool2d(x_py, (1, 1)).flatten(1)
    cuda_out = cuda_op(x)

    passed = check_forward(name, py_out, cuda_out)

    grad_out = torch.randn(B, C, device=device)
    py_out.backward(grad_out)
    cuda_out.backward(grad_out.clone())

    if not check_grad(name, "x", x_py.grad, x.grad):
        passed = False

    if passed:
        print(f"  PASS {name} backward (all grads match)")
    return passed


def test_skip_connection():
    """Test SkipConnection CUDA wrapper."""
    name = "SkipConnection"
    print(f"\nTesting {name}...")

    # Test flat case (same dim, no projection)
    d = 16
    B = 4

    cuda_skip = SkipConnectionCUDA(
        source_idx=0, target_idx=1, d_source=d, d_target=d,
        init_scale=0.01, device=device
    )

    torch.manual_seed(123)
    x = torch.randn(B, d, device=device, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    cuda_out = cuda_skip.forward(x)
    # Reference: just scale * x (when d_source == d_target, no projection)
    ref_out = cuda_skip.scale * x_ref

    passed = check_forward(name, ref_out, cuda_out)

    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    cuda_out.backward(grad_out.clone())

    if not check_grad(name, "x", x_ref.grad, x.grad):
        passed = False

    if passed:
        print(f"  PASS {name} backward (all grads match)")
    return passed


def main():
    print("=" * 60)
    print("ASANN CUDA Numerical Equivalence Tests")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Forward tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"Gradient tolerance: atol={GRAD_ATOL}, rtol={GRAD_RTOL}")

    results = {}

    # ==================== Flat Operations ====================
    results["PositionalEmbedding"] = test_flat_op(
        "PositionalEmbedding", PositionalEmbedding, PositionalEmbeddingCUDA)

    results["FactoredEmbedding"] = test_flat_op(
        "FactoredEmbedding", FactoredEmbedding, FactoredEmbeddingCUDA)

    results["MLPEmbedding"] = test_flat_op(
        "MLPEmbedding", MLPEmbedding, MLPEmbeddingCUDA)

    results["GeometricEmbedding"] = test_flat_op(
        "GeometricEmbedding", GeometricEmbedding, GeometricEmbeddingCUDA)

    results["Conv1dBlock_k3"] = test_flat_op(
        "Conv1dBlock_k3", Conv1dBlock, Conv1dBlockCUDA, extra_kwargs={"kernel_size": 3})

    results["Conv1dBlock_k5"] = test_flat_op(
        "Conv1dBlock_k5", Conv1dBlock, Conv1dBlockCUDA, extra_kwargs={"kernel_size": 5})

    results["Conv2dBlock_k3"] = test_flat_op(
        "Conv2dBlock_k3", Conv2dBlock, Conv2dBlockCUDA, extra_kwargs={"kernel_size": 3})

    results["SelfAttentionOp"] = test_flat_op(
        "SelfAttentionOp", SelfAttentionOp, SelfAttentionOpCUDA)

    results["MultiHeadAttentionOp"] = test_flat_op(
        "MultiHeadAttentionOp", MultiHeadAttentionOp, MultiHeadAttentionOpCUDA)

    results["CrossAttentionOp"] = test_flat_op(
        "CrossAttentionOp", CrossAttentionOp, CrossAttentionOpCUDA)

    results["CausalAttentionOp"] = test_flat_op(
        "CausalAttentionOp", CausalAttentionOp, CausalAttentionOpCUDA)

    # ==================== Temporal / Sequence Operations ====================
    results["DilatedConv1dBlock_k3"] = test_flat_op(
        "DilatedConv1dBlock_k3", DilatedConv1dBlock, DilatedConv1dBlockCUDA,
        extra_kwargs={"kernel_size": 3, "dilation": 2})

    results["DilatedConv1dBlock_k5"] = test_flat_op(
        "DilatedConv1dBlock_k5", DilatedConv1dBlock, DilatedConv1dBlockCUDA,
        extra_kwargs={"kernel_size": 5, "dilation": 2})

    results["EMASmooth"] = test_flat_op(
        "EMASmooth", EMASmooth, EMASmoothCUDA)

    results["GatedLinearUnit"] = test_flat_op(
        "GatedLinearUnit", GatedLinearUnit, GatedLinearUnitCUDA)

    results["TemporalDiff"] = test_flat_op(
        "TemporalDiff", TemporalDiff, TemporalDiffCUDA)

    # ==================== Spatial Operations ====================
    results["SpatialConv2dOp_k3"] = test_spatial_op(
        "SpatialConv2dOp_k3", SpatialConv2dOp, SpatialConv2dOpCUDA,
        extra_kwargs={"kernel_size": 3})

    results["SpatialConv2dOp_k5"] = test_spatial_op(
        "SpatialConv2dOp_k5", SpatialConv2dOp, SpatialConv2dOpCUDA,
        extra_kwargs={"kernel_size": 5})

    results["PointwiseConv2dOp"] = test_spatial_op(
        "PointwiseConv2dOp", PointwiseConv2dOp, PointwiseConv2dOpCUDA)

    results["DepthwiseSeparableConv2dOp"] = test_spatial_op(
        "DepthwiseSeparableConv2dOp", DepthwiseSeparableConv2dOp,
        DepthwiseSeparableConv2dOpCUDA, extra_kwargs={"kernel_size": 3})

    # ==================== Special Operations ====================
    results["ChannelAttentionOp"] = test_channel_attention()
    results["GAPFlatten"] = test_gap_flatten()
    results["SkipConnection"] = test_skip_connection()

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{total} tests passed, {failed} failed")
    if failed > 0:
        print("SOME TESTS FAILED!")
        return 1
    else:
        print("ALL TESTS PASSED!")
        return 0


if __name__ == "__main__":
    exit(main())
