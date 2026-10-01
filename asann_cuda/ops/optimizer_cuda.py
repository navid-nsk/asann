"""Python wrappers for CUDA-accelerated optimizer kernels."""

import math
import asann_cuda_ops


def normuon_normalize_cuda(grad, eps=1e-6):
    """Per-neuron gradient normalization (NorMuon) via CUDA.

    Supports 4D [C_out, C_in, k, k] and 2D [d_out, d_in] tensors.
    Modifies grad in-place.
    """
    asann_cuda_ops.normuon_normalize(grad, eps)


def fused_optimizer_step_cuda(
    p_data, grad, m_fast, m_medium, m_slow, v, buf,
    step, beta1_fast, beta1_med, beta1_slow, beta2,
    lr, weight_decay, newborn_mult, eps, needs_ns
):
    """Fused optimizer step: momentum updates + adaptive step in one kernel.

    Precomputes bias correction scalars in Python (float64 precision) then
    calls the CUDA kernel.

    Args:
        p_data: Parameter data tensor (modified in-place)
        grad: Gradient tensor
        m_fast, m_medium, m_slow: Momentum buffers (modified in-place)
        v: Second moment buffer (modified in-place)
        buf: Scratch buffer (modified in-place)
        step: Current step count (int)
        beta1_fast, beta1_med, beta1_slow, beta2: Momentum/variance betas
        lr: Learning rate
        weight_decay: Decoupled weight decay coefficient
        newborn_mult: Newborn warmup multiplier (1.0 for non-newborn)
        eps: Epsilon for numerical stability
        needs_ns: If True, write combined momentum to buf (Newton-Schulz path)
    """
    # Compute bias correction in Python at float64 precision (matches original)
    bc_fast = 1.0 - beta1_fast ** step
    bc_med = 1.0 - beta1_med ** step
    bc_slow = 1.0 - beta1_slow ** step
    bc_v = 1.0 - beta2 ** step

    # Precompute the fused coefficients
    inv_bc_fast = 0.5 / bc_fast
    inv_bc_med = 0.35 / bc_med
    inv_bc_slow = 0.15 / bc_slow
    inv_sqrt_bc_v = 1.0 / math.sqrt(bc_v)

    asann_cuda_ops.fused_optimizer_step(
        p_data, grad, m_fast, m_medium, m_slow, v, buf,
        float(beta1_fast), float(beta1_med), float(beta1_slow), float(beta2),
        float(inv_bc_fast), float(inv_bc_med), float(inv_bc_slow),
        float(inv_sqrt_bc_v),
        float(lr), float(weight_decay), float(newborn_mult), float(eps),
        bool(needs_ns)
    )


def optimizer_apply_update_cuda(
    p_data, buf, v,
    step, beta2, lr, weight_decay, newborn_mult, eps
):
    """Apply parameter update after Newton-Schulz orthogonalization.

    Called after the Newton-Schulz step has been applied to buf in Python.

    Args:
        p_data: Parameter data tensor (modified in-place)
        buf: Buffer containing orthogonalized combined momentum
        v: Second moment buffer
        step: Current step count
        beta2: Variance beta
        lr: Learning rate
        weight_decay: Decoupled weight decay coefficient
        newborn_mult: Newborn warmup multiplier
        eps: Epsilon for numerical stability
    """
    bc_v = 1.0 - beta2 ** step
    inv_sqrt_bc_v = 1.0 / math.sqrt(bc_v)

    asann_cuda_ops.optimizer_apply_update(
        p_data, buf, v,
        float(inv_sqrt_bc_v),
        float(lr), float(weight_decay), float(newborn_mult), float(eps)
    )
