"""
ASANNOptimizer: Nested Learning Optimizer for ASANN v4.

Replaces SurgeryAwareOptimizer with a more capable system modeled after
the nl_framework's NestedLearningOptimizer. Key features:

1. Multi-scale momentum (M3-style): Three momentum buffers at different
   time scales with adaptive variance-based weighting.
2. Per-module parameter groups: Different LR scales and update frequencies
   for different architectural components.
3. Per-group gradient clipping.
4. Newton-Schulz orthogonalization for small 2D weight matrices.
5. Surgery-aware state management (register/unregister params, state transfer).
6. Newborn parameter warmup and graduation.
7. Per-neuron gradient normalization (NorMuon-inspired).

Architecture:
  ASANNTrainer
  ├── ASANNOptimizer        ← THIS FILE
  ├── ASANNLRController     (hypergradient-based LR adaptation)
  ├── ASANNWarmupScheduler  (warmup + cosine annealing)
  ├── SurgeryScheduler      (unchanged)
  └── MetaLearner           (simplified — only thresholds + interval)

Usage:
    param_groups = create_asann_parameter_groups(model, config)
    optimizer = ASANNOptimizer(param_groups, lr=config.base_lr, ...)
    # Normal training step:
    optimizer.step()
    # After neuron surgery:
    optimizer.register_neuron_surgery(old_param, new_param, layer_idx, 'neuron_add')
    # After layer surgery:
    optimizer.register_structural_surgery(model, 'layer_add')
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass
from torch.optim import Optimizer

from .config import ASANNOptimizerConfig


# =============================================================================
# Newton-Schulz Orthogonalization (from nl_framework)
# =============================================================================

def newton_schulz_orthogonalize(M: torch.Tensor, num_iterations: int = 3) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iterations.

    Given a matrix M, converges to the nearest orthogonal matrix Q
    such that Q^T Q ≈ I.

    This provides implicit spectral regularization: the update direction
    has bounded spectral norm, preventing any single direction from
    dominating the weight update.
    """
    if M.dim() != 2:
        return M

    m, n = M.shape
    if m == 0 or n == 0:
        return M

    # Normalize to unit Frobenius norm for numerical stability
    norm = M.norm()
    if norm < 1e-8:
        return M
    Y = M / norm

    # Newton-Schulz iteration: Y_{k+1} = Y_k * (3I - Y_k^T Y_k) / 2
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    for _ in range(num_iterations):
        YtY = Y.t() @ Y
        Y = Y @ (3 * I - YtY) / 2

    return Y


# =============================================================================
# Newborn Parameter Registry Entry
# =============================================================================

@dataclass
class NewbornEntry:
    """Metadata for a newborn parameter undergoing warmup."""
    param: nn.Parameter
    birth_step: int
    warmup_remaining: int
    source_group_name: str     # Which group it belongs to after graduation
    source_layer: int = -1
    surgery_type: str = ""


# =============================================================================
# ASANNOptimizer
# =============================================================================

class ASANNOptimizer(Optimizer):
    """ASANN Nested Learning Optimizer with multi-scale momentum and surgery awareness.

    Extends torch.optim.Optimizer with:
    1. Multi-scale momentum (M3-style) with bias correction
    2. Multi-frequency parameter group updates
    3. Per-group gradient clipping
    4. Newton-Schulz orthogonalization for small 2D matrices
    5. Surgery-aware state management (register/unregister params)
    6. Newborn parameter warmup and graduation
    7. Per-neuron gradient normalization (NorMuon)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float, float] = (0.9, 0.95, 0.99, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        neuron_norm_enabled: bool = True,
        neuron_norm_eps: float = 1e-6,
        variance_transfer_enabled: bool = True,
        newborn_warmup_steps: int = 100,
        newborn_graduation_steps: int = 200,
        newborn_lr_scale: float = 0.1,
        use_cuda_ops: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

        self.global_step = 0
        self._neuron_norm_enabled = neuron_norm_enabled
        self._neuron_norm_eps = neuron_norm_eps
        self._variance_transfer_enabled = variance_transfer_enabled
        self._newborn_warmup_steps = newborn_warmup_steps
        self._newborn_graduation_steps = newborn_graduation_steps
        self._newborn_lr_scale = newborn_lr_scale

        # Newborn parameter registry
        self._newborn_registry: Dict[int, NewbornEntry] = {}

        # Track all param ids for fast lookup
        self._all_param_ids: Set[int] = set()
        for group in self.param_groups:
            for p in group['params']:
                self._all_param_ids.add(id(p))

        # Try importing CUDA optimizer kernels (only if use_cuda_ops=True)
        self._use_cuda_kernels = False
        if use_cuda_ops:
            try:
                from asann_cuda.ops.optimizer_cuda import (
                    normuon_normalize_cuda,
                    fused_optimizer_step_cuda,
                    optimizer_apply_update_cuda,
                )
                self._normuon_normalize_cuda = normuon_normalize_cuda
                self._fused_optimizer_step_cuda = fused_optimizer_step_cuda
                self._optimizer_apply_update_cuda = optimizer_apply_update_cuda
                self._use_cuda_kernels = True
            except ImportError:
                pass

    # =========================================================================
    # Core Optimizer Step
    # =========================================================================

    def _get_effective_lr(self, group: dict) -> float:
        """Compute effective LR for a group: base * scale."""
        return group['lr'] * group.get('lr_scale', 1.0)

    def _should_update_group(self, group: dict) -> bool:
        """Multi-frequency: check if group should update at current step."""
        freq = group.get('update_freq', 1)
        return self.global_step % freq == 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform one optimization step.

        Pipeline:
        1. Check multi-frequency update schedule for each group
        2. Apply per-neuron gradient normalization (NorMuon)
        3. Per-group gradient clipping
        4. Update multi-scale momentum buffers with bias correction
        5. Adaptive momentum weighting based on gradient variance
        6. Newton-Schulz orthogonalization for small 2D matrices
        7. Compute adaptive step (combined_momentum / sqrt(v))
        8. Apply weight decay (decoupled)
        9. Apply update
        10. Graduate newborn parameters
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            # Multi-frequency: skip groups not scheduled for this step
            if not self._should_update_group(group):
                continue

            lr = self._get_effective_lr(group)
            betas = group.get('betas', (0.9, 0.95, 0.99, 0.999))
            # Handle both 2-tuple (legacy) and 4-tuple betas
            if len(betas) == 2:
                beta1_fast = betas[0]
                beta1_med = (betas[0] + betas[1]) / 2  # interpolate
                beta1_slow = betas[1]
                beta2 = betas[1]
            else:
                beta1_fast, beta1_med, beta1_slow, beta2 = betas

            eps = group.get('eps', 1e-8)
            weight_decay = group.get('weight_decay', 0.0)
            grad_clip = group.get('grad_clip', group.get('max_grad_norm', 1.0))

            # --- Per-neuron gradient normalization (NorMuon) ---
            if self._neuron_norm_enabled:
                self._apply_neuron_norm_to_group(group)

            # --- Per-group gradient clipping ---
            if grad_clip > 0:
                params_with_grad = [p for p in group['params'] if p.grad is not None]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, grad_clip)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get or initialize state
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Multi-scale momentum buffers
                    state['m_fast'] = torch.zeros_like(p.data)
                    state['m_medium'] = torch.zeros_like(p.data)
                    state['m_slow'] = torch.zeros_like(p.data)
                    # Second moment (variance)
                    state['v'] = torch.zeros_like(p.data)
                    # Reusable scratch buffer (avoids temporary allocations)
                    state['_buf'] = torch.empty_like(p.data)

                state['step'] += 1
                step = state['step']

                # Multi-scale momentum (M3-style)
                m_fast = state['m_fast']
                m_medium = state['m_medium']
                m_slow = state['m_slow']
                v = state['v']
                buf = state['_buf']

                # Check if Newton-Schulz applies to this parameter
                needs_ns = (p.dim() == 2 and p.shape[0] >= 128
                            and p.shape[1] >= 128 and p.shape[0] <= 256
                            and p.shape[1] <= 256)

                newborn_mult = self._get_newborn_multiplier(p)

                # --- CUDA fused path ---
                if self._use_cuda_kernels and p.data.is_cuda:
                    self._fused_optimizer_step_cuda(
                        p.data, grad, m_fast, m_medium, m_slow, v, buf,
                        step, beta1_fast, beta1_med, beta1_slow, beta2,
                        lr, weight_decay, newborn_mult, eps, needs_ns
                    )

                    if needs_ns:
                        # Newton-Schulz on buf (same Python code, unchanged)
                        try:
                            orig_norm = buf.norm()
                            if orig_norm > 1e-8:
                                ns_result = newton_schulz_orthogonalize(
                                    buf, num_iterations=3)
                                buf.copy_(ns_result).mul_(
                                    orig_norm / (ns_result.norm() + 1e-8))
                        except Exception:
                            pass
                        # Apply update with orthogonalized buf
                        self._optimizer_apply_update_cuda(
                            p.data, buf, v,
                            step, beta2, lr, weight_decay, newborn_mult, eps
                        )
                    continue

                # --- Python fallback path (unchanged) ---
                # Update momentum buffers (in-place)
                m_fast.mul_(beta1_fast).add_(grad, alpha=1 - beta1_fast)
                m_medium.mul_(beta1_med).add_(grad, alpha=1 - beta1_med)
                m_slow.mul_(beta1_slow).add_(grad, alpha=1 - beta1_slow)

                # Second moment (in-place)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction factors (scalars)
                bc_fast = 1 - beta1_fast ** step
                bc_med = 1 - beta1_med ** step
                bc_slow = 1 - beta1_slow ** step
                bc_v = 1 - beta2 ** step

                # Fused combined bias-corrected momentum into buf
                torch.mul(m_fast, 0.5 / bc_fast, out=buf)
                buf.add_(m_medium, alpha=0.35 / bc_med)
                buf.add_(m_slow, alpha=0.15 / bc_slow)

                # Newton-Schulz orthogonalization for large 2D matrices
                if needs_ns:
                    try:
                        orig_norm = buf.norm()
                        if orig_norm > 1e-8:
                            ns_result = newton_schulz_orthogonalize(buf, num_iterations=3)
                            buf.copy_(ns_result).mul_(orig_norm / (ns_result.norm() + 1e-8))
                    except Exception:
                        pass

                # Weight decay (decoupled, before update)
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Apply newborn warmup scaling
                step_size = -lr * newborn_mult

                # Fused adaptive step: p += step_size * buf / (sqrt(v/bc_v) + eps)
                inv_sqrt_bc_v = 1.0 / math.sqrt(bc_v)
                p.data.addcdiv_(buf, v.sqrt().mul_(inv_sqrt_bc_v).add_(eps),
                                value=step_size)

        # Graduate newborn parameters
        self._graduate_newborns()

        return loss

    # =========================================================================
    # Per-Neuron Gradient Normalization (NorMuon)
    # =========================================================================

    def _apply_neuron_norm_to_group(self, group: dict):
        """Apply per-neuron gradient normalization within a group.

        Only applies to 4D weight matrices (Conv2d in spatial models) and
        large 2D weight matrices (≥128 rows). For small tabular weight matrices,
        equalizing row norms can prevent the network from learning important
        magnitude differences between neurons.
        """
        group_name = group.get('name', '')
        # Skip normalization for 1D params (biases, norms, scales)
        if group_name in ('spatial_bn', 'other_1d'):
            return

        eps = self._neuron_norm_eps
        use_cuda = self._use_cuda_kernels
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data

            if p.ndim == 4:
                if use_cuda and grad.is_cuda:
                    self._normuon_normalize_cuda(grad, eps)
                else:
                    # Conv2d [C_out, C_in, k, k]: per-filter normalization (always useful)
                    filter_norms = grad.flatten(1).norm(dim=1)  # [C_out]
                    mean_norm = filter_norms.mean()
                    filter_norms = filter_norms.view(-1, 1, 1, 1).clamp(min=eps)
                    grad.div_(filter_norms).mul_(mean_norm)
            elif p.ndim == 2 and p.shape[0] >= 128:
                if use_cuda and grad.is_cuda:
                    self._normuon_normalize_cuda(grad, eps)
                else:
                    # Linear [d_out, d_in]: per-row normalization only for large matrices.
                    # For small tabular models (d<128), skip — the old optimizer didn't
                    # do per-neuron normalization and achieved R²=0.991.
                    row_norms = grad.norm(dim=1, keepdim=True)  # [d_out, 1]
                    mean_norm = row_norms.mean()
                    grad.div_(row_norms.clamp(min=eps)).mul_(mean_norm)

    # =========================================================================
    # Newborn Parameter Management
    # =========================================================================

    def _get_newborn_multiplier(self, p: nn.Parameter) -> float:
        """Get warmup multiplier for a parameter. 1.0 if not newborn."""
        entry = self._newborn_registry.get(id(p))
        if entry is None:
            return 1.0

        total_warmup = self._newborn_warmup_steps
        elapsed = total_warmup - entry.warmup_remaining
        progress = max(0.0, min(1.0, elapsed / max(total_warmup, 1)))
        # Ramp from newborn_lr_scale to 1.0
        return self._newborn_lr_scale + (1.0 - self._newborn_lr_scale) * progress

    def _graduate_newborns(self):
        """Check and graduate newborn parameters that have completed their warmup."""
        graduated_ids = []

        for pid, entry in list(self._newborn_registry.items()):
            entry.warmup_remaining = max(0, entry.warmup_remaining - 1)
            age = self.global_step - entry.birth_step

            if age >= self._newborn_graduation_steps:
                graduated_ids.append(pid)

        for pid in graduated_ids:
            self._newborn_registry.pop(pid, None)

    @property
    def num_newborn_params(self) -> int:
        """Number of parameters currently in newborn status."""
        return len(self._newborn_registry)

    # =========================================================================
    # Surgery Registration — Called by SurgeryEngine AFTER surgery
    # =========================================================================

    def register_neuron_surgery(
        self,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        layer_idx: int,
        surgery_type: str = 'neuron_add',
    ):
        """Register a parameter that was resized during neuron surgery.

        Handles:
        1. Optimizer state transfer (momentum, variance) with proper overlap copy
        2. Variance-calibrated initialization for new dimensions
        3. Registration as newborn with warmup schedule
        4. Step counter preservation (no bias correction reset)
        """
        old_state = self.state.get(old_param, {})

        # --- Transfer optimizer state ---
        if old_state:
            new_state = {}
            for key, val in old_state.items():
                if isinstance(val, torch.Tensor) and val.shape == old_param.shape:
                    new_val = torch.zeros_like(new_param.data)
                    # Copy overlapping region
                    slices = tuple(
                        slice(0, min(s1, s2))
                        for s1, s2 in zip(val.shape, new_val.shape)
                    )
                    new_val[slices] = val[slices]

                    # Variance transfer for new dimensions
                    if self._variance_transfer_enabled and surgery_type == 'neuron_add':
                        new_val = self._variance_calibrate_state(new_val, val, slices, key)

                    new_state[key] = new_val
                elif isinstance(val, torch.Tensor) and val.numel() == 1:
                    new_state[key] = val.clone()
                elif key == 'step':
                    new_state[key] = val  # Preserve step counter
                else:
                    new_state[key] = val

            if old_param in self.state:
                del self.state[old_param]
            self.state[new_param] = new_state

        # --- Swap parameter reference in param_groups ---
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p is old_param:
                    group['params'][i] = new_param
                    break

        # --- Update tracking ---
        self._all_param_ids.discard(id(old_param))
        self._all_param_ids.add(id(new_param))

        # Remove old newborn entry if any
        self._newborn_registry.pop(id(old_param), None)

        # Register as newborn if adding
        if surgery_type == 'neuron_add':
            # Find which group this param is in
            group_name = 'other'
            for group in self.param_groups:
                for p in group['params']:
                    if p is new_param:
                        group_name = group.get('name', 'other')
                        break

            self._newborn_registry[id(new_param)] = NewbornEntry(
                param=new_param,
                birth_step=self.global_step,
                warmup_remaining=self._newborn_warmup_steps,
                source_group_name=group_name,
                source_layer=layer_idx,
                surgery_type=surgery_type,
            )

    def register_new_parameters(
        self,
        params: List[nn.Parameter],
        group_name: str = 'other',
        source_layer: int = -1,
        surgery_type: str = 'structural',
    ):
        """Register entirely new parameters (from layer/connection/op addition).

        Adds them to the correct parameter group in the optimizer.
        For per-layer groups (layer_N_weights, layer_N_ops), creates a new
        param group if one doesn't exist yet (e.g., after layer addition).
        """
        for p in params:
            if id(p) in self._all_param_ids:
                continue  # Already tracked

            # Find the right group or create catch-all
            added = False
            for group in self.param_groups:
                if group.get('name') == group_name:
                    if not any(pp is p for pp in group['params']):
                        group['params'].append(p)
                    added = True
                    break

            if not added:
                # For per-layer groups, create a new param group with proper config
                if (group_name.startswith('layer_') and
                        (group_name.endswith('_weights') or group_name.endswith('_ops'))):
                    # Inherit config from an existing similar group or use defaults
                    template_group = None
                    for group in self.param_groups:
                        gn = group.get('name', '')
                        if group_name.endswith('_weights') and gn.endswith('_weights'):
                            template_group = group
                            break
                        if group_name.endswith('_ops') and gn.endswith('_ops'):
                            template_group = group
                            break

                    if template_group is None:
                        # Fallback: use first group as template
                        template_group = self.param_groups[0] if self.param_groups else None

                    if template_group is not None:
                        new_group = {
                            'params': [p],
                            'lr': template_group['lr'],
                            'lr_scale': template_group.get('lr_scale', 1.0),
                            'update_freq': template_group.get('update_freq', 1),
                            'weight_decay': template_group.get('weight_decay', 0.01),
                            'grad_clip': template_group.get('grad_clip', 10.0),
                            'betas': template_group.get('betas', (0.9, 0.9, 0.9, 0.999)),
                            'eps': template_group.get('eps', 1e-8),
                            'name': group_name,
                        }
                        self.param_groups.append(new_group)
                        added = True

                if not added:
                    # Add to 'other' group or first group
                    for group in self.param_groups:
                        if group.get('name') == 'other':
                            group['params'].append(p)
                            added = True
                            break
                    if not added and self.param_groups:
                        self.param_groups[-1]['params'].append(p)

            self._all_param_ids.add(id(p))

            # Register as newborn
            self._newborn_registry[id(p)] = NewbornEntry(
                param=p,
                birth_step=self.global_step,
                warmup_remaining=self._newborn_warmup_steps,
                source_group_name=group_name,
                source_layer=source_layer,
                surgery_type=surgery_type,
            )

    def register_removed_parameters(self, params: List[nn.Parameter]):
        """Unregister parameters that were removed (layer/connection/op removal)."""
        for p in params:
            pid = id(p)
            self._all_param_ids.discard(pid)
            self._newborn_registry.pop(pid, None)

            # Remove from optimizer state
            if p in self.state:
                del self.state[p]

            # Remove from param_groups (identity check)
            for group in self.param_groups:
                for i, pp in enumerate(group['params']):
                    if pp is p:
                        group['params'].pop(i)
                        break

    def register_structural_surgery(self, model: nn.Module, surgery_type: str = 'layer_add'):
        """Full re-sync after major structural surgery (layer add/remove).

        1. Identify which parameters are new vs. surviving
        2. Register new params with warmup
        3. Remove stale params
        4. Preserve all optimizer state for surviving params
        """
        current_params = set()
        for p in model.parameters():
            current_params.add(id(p))
        if hasattr(model, 'connections'):
            for conn in model.connections:
                for p in conn.parameters():
                    current_params.add(id(p))

        # Remove stale params
        stale_ids = self._all_param_ids - current_params
        for pid in list(stale_ids):
            # Find the actual param object
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if id(p) == pid:
                        if p in self.state:
                            del self.state[p]
                        group['params'].pop(i)
                        break
            self._all_param_ids.discard(pid)
            self._newborn_registry.pop(pid, None)

        # Find new params (in model but not tracked)
        new_params = []
        for p in model.parameters():
            if id(p) not in self._all_param_ids:
                new_params.append(p)
        if hasattr(model, 'connections'):
            for conn in model.connections:
                for p in conn.parameters():
                    if id(p) not in self._all_param_ids:
                        new_params.append(p)

        # Classify and register new params
        if new_params:
            classified = _classify_new_params(model, new_params)
            for group_name, params in classified.items():
                self.register_new_parameters(
                    params=params,
                    group_name=group_name,
                    surgery_type=surgery_type,
                )

    def enter_post_surgery_warmup(self):
        """Signal for post-surgery warmup. The actual warmup is handled by
        ASANNWarmupScheduler. This method exists for interface compatibility."""
        pass  # Warmup scheduler handles this

    # =========================================================================
    # Variance Transfer (Yuan et al. 2023)
    # =========================================================================

    def _variance_calibrate_state(
        self,
        new_state_tensor: torch.Tensor,
        old_state_tensor: torch.Tensor,
        overlap_slices: tuple,
        state_key: str,
    ) -> torch.Tensor:
        """Calibrate optimizer state for new dimensions after neuron addition.

        Warm-starts the new neuron's optimizer state to the mean of existing
        neurons' state, preventing bias correction spikes.
        """
        if state_key in ('m_fast', 'm_medium', 'm_slow', 'v'):
            old_region = old_state_tensor
            if old_region.ndim >= 2 and old_region.shape[0] > 0:
                mean_row = old_region.mean(dim=0, keepdim=True)
                old_rows = old_region.shape[0]
                num_new = new_state_tensor.shape[0] - old_rows
                if num_new > 0:
                    expand_shape = (num_new,) + (-1,) * (mean_row.ndim - 1)
                    expanded = mean_row.expand(*expand_shape)
                    slices = tuple(
                        slice(0, min(expanded.shape[d], new_state_tensor.shape[d]))
                        for d in range(1, expanded.ndim)
                    )
                    target_slices = (slice(old_rows, old_rows + num_new),) + slices
                    source_slices = (slice(None),) + slices
                    new_state_tensor[target_slices] = expanded[source_slices]

        return new_state_tensor

    # =========================================================================
    # Accessors
    # =========================================================================

    @property
    def current_lr(self) -> float:
        """Return the effective LR of the first major weight group."""
        for group in self.param_groups:
            name = group.get('name', '')
            if name in ('spatial_weights', 'flat_weights', 'encoder'):
                return self._get_effective_lr(group)
            # Per-layer groups: layer_0_weights is the first major weight group
            if name == 'layer_0_weights':
                return self._get_effective_lr(group)
        # Fallback: first group
        if self.param_groups:
            return self._get_effective_lr(self.param_groups[0])
        return 0.0

    @property
    def phase(self) -> str:
        """Phase is now managed by ASANNWarmupScheduler. Return 'stable'."""
        return "stable"

    @property
    def complexity_multiplier(self) -> float:
        """Complexity modulation is now handled by LR controller. Return 1.0."""
        return 1.0

    def get_group_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics about each parameter group (vectorized norms)."""
        stats = {}
        for group in self.param_groups:
            name = group.get('name', 'unknown')
            n_params = len(group['params'])
            total_elements = sum(p.numel() for p in group['params'])

            # Collect tensors for vectorized norm computation
            m_fast_list, m_medium_list, m_slow_list, v_list, grad_list = [], [], [], [], []
            for p in group['params']:
                state = self.state.get(p)
                if state and 'm_fast' in state:
                    m_fast_list.append(state['m_fast'])
                    m_medium_list.append(state['m_medium'])
                    m_slow_list.append(state['m_slow'])
                    v_list.append(state['v'])
                if p.grad is not None:
                    grad_list.append(p.grad)

            m_fast_norm = m_medium_norm = m_slow_norm = v_norm = grad_norm = 0.0
            count = len(m_fast_list)

            if count > 0:
                # Batched norm: one fused kernel per buffer type, single CPU transfer
                all_tensors = m_fast_list + m_medium_list + m_slow_list + v_list
                all_norms = torch._foreach_norm(all_tensors, 2)
                all_norms_cpu = torch.stack(all_norms).cpu()
                m_fast_norm = all_norms_cpu[:count].sum().item() / count
                m_medium_norm = all_norms_cpu[count:2*count].sum().item() / count
                m_slow_norm = all_norms_cpu[2*count:3*count].sum().item() / count
                v_norm = all_norms_cpu[3*count:].sum().item() / count

            if grad_list:
                grad_norms = torch._foreach_norm(grad_list, 2)
                grad_norm = torch.stack(grad_norms).sum().cpu().item()

            stats[name] = {
                'n_params': n_params,
                'total_elements': total_elements,
                'lr': group['lr'],
                'effective_lr': self._get_effective_lr(group),
                'lr_scale': group.get('lr_scale', 1.0),
                'update_freq': group.get('update_freq', 1),
                'grad_clip': group.get('grad_clip', group.get('max_grad_norm', 1.0)),
                'm_fast_norm': m_fast_norm,
                'm_medium_norm': m_medium_norm,
                'm_slow_norm': m_slow_norm,
                'v_norm': v_norm,
                'grad_norm': grad_norm,
            }
        return stats

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Save optimizer state for checkpointing."""
        base_state = super().state_dict()
        return {
            'optimizer_state': base_state,
            'global_step': self.global_step,
            'newborn_registry': {
                str(pid): {
                    'birth_step': entry.birth_step,
                    'warmup_remaining': entry.warmup_remaining,
                    'source_group_name': entry.source_group_name,
                    'source_layer': entry.source_layer,
                    'surgery_type': entry.surgery_type,
                }
                for pid, entry in self._newborn_registry.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load optimizer state from checkpoint."""
        if 'optimizer_state' in state:
            super().load_state_dict(state['optimizer_state'])
            self.global_step = state.get('global_step', 0)
        else:
            # Backward compatibility: state IS the base optimizer state
            super().load_state_dict(state)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def describe(self) -> str:
        """Human-readable summary of optimizer state."""
        lines = [
            f"ASANNOptimizer (step {self.global_step})",
            f"  Newborn params: {self.num_newborn_params}",
            f"  Parameter groups:",
        ]
        for group in self.param_groups:
            name = group.get('name', '?')
            n = len(group['params'])
            lr = self._get_effective_lr(group)
            freq = group.get('update_freq', 1)
            lines.append(f"    {name}: {n} params, lr={lr:.6f}, freq={freq}")
        return '\n'.join(lines)


# =============================================================================
# Parameter Group Creation
# =============================================================================

def create_asann_parameter_groups(
    model: nn.Module,
    config: ASANNOptimizerConfig,
    verbose: bool = True,
) -> List[Dict]:
    """Create parameter groups for ASANNOptimizer.

    Classifies model parameters into groups with different LR scales,
    update frequencies, weight decay, and gradient clipping.

    Groups:
    - encoder: Input encoder (lr_scale=0.5 spatial / 1.0 tabular)
    - spatial_weights: Spatial layer conv weights (lr_scale=1.0, freq=1)
    - spatial_bn: Spatial layer batch norm (lr_scale=0.5, freq=1, wd=0)
    - spatial_residual: Spatial residual projections (lr_scale=0.5, freq=1)
    - flat_weights: Flat layer linear weights (lr_scale=1.0, freq=1)
    - operations: Operation pipeline params (lr_scale=0.5, freq=1)
    - connections: Skip connection params (lr_scale=0.5, freq=1)
    - output_head: Output head (lr_scale=0.5, freq=1)
    - other: Catch-all (lr_scale=1.0, freq=1)
    """
    # Group configuration: name -> (lr_scale, update_freq, weight_decay, grad_clip)
    # Determine if model is spatial (conv layers) or tabular (flat layers only)
    is_spatial = getattr(model, '_is_spatial', False) or (
        hasattr(model, 'layers') and any(
            getattr(layer, 'mode', 'flat') == 'spatial'
            for layer in model.layers
        )
    )

    if is_spatial:
        # Spatial models: moderate LR differences, all groups update every step.
        # Previously spatial_residual and connections had lr_scale=0.3, update_freq=5,
        # giving them only 6% effective LR — too starved to learn useful projections.
        group_configs = {
            'encoder':          {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 1.0},
            'spatial_weights':  {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 1.0},
            'spatial_bn':       {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 1.0},
            'spatial_residual': {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 1.0},
            'flat_weights':     {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 1.0},
            'operations':       {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.005, 'grad_clip': 1.0},
            'connections':      {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 1.0},
            'output_head':      {'lr_scale': 0.5,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 1.0},
            'other':            {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 1.0},
        }
    else:
        # Tabular models: all groups get full LR, connections update every step.
        # Use high grad_clip (10.0) to match old SurgeryAwareOptimizer behavior
        # which used a global grad_clip=1.0 across ALL params, not per-group.
        # Per-group clipping at 1.0 is too restrictive for small models.
        group_configs = {
            'encoder':          {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 10.0},
            'spatial_weights':  {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 10.0},
            'spatial_bn':       {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 10.0},
            'spatial_residual': {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 10.0},
            'flat_weights':     {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 10.0},
            'operations':       {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 10.0},
            'connections':      {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.0,  'grad_clip': 10.0},
            'output_head':      {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 10.0},
            'other':            {'lr_scale': 1.0,  'update_freq': 1,  'weight_decay': 0.01, 'grad_clip': 10.0},
        }

    groups: Dict[str, List[nn.Parameter]] = defaultdict(list)
    seen: Set[int] = set()

    # --- Classify parameters ---

    # 1. Encoder (input encoder — LinearEncoder, ConvEncoder, etc.)
    if hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            if id(p) not in seen and p.requires_grad:
                groups['encoder'].append(p)
                seen.add(id(p))
    elif hasattr(model, 'input_projection'):
        # Legacy fallback for old models without encoder attribute
        for p in model.input_projection.parameters():
            if id(p) not in seen and p.requires_grad:
                groups['encoder'].append(p)
                seen.add(id(p))

    # 2. Output head
    if hasattr(model, 'output_head'):
        for p in model.output_head.parameters():
            if id(p) not in seen and p.requires_grad:
                groups['output_head'].append(p)
                seen.add(id(p))

    # 3. Layer weights (spatial vs flat)
    # Per-layer groups for tabular models: each layer gets its own group
    # so the LR controller can independently adjust LR per-layer.
    # e.g., the input layer (d_input→d_init, most params) may overfit
    # faster than narrow inner layers — per-layer LR lets the controller
    # slow it down while keeping inner layers at full speed.
    if hasattr(model, 'layers'):
        for l_idx, layer in enumerate(model.layers):
            mode = getattr(layer, 'mode', 'flat')
            for name, p in layer.named_parameters():
                if id(p) not in seen and p.requires_grad:
                    if mode == 'spatial':
                        if 'bn' in name or 'norm' in name:
                            groups['spatial_bn'].append(p)
                        elif 'residual' in name or 'proj' in name:
                            groups['spatial_residual'].append(p)
                        else:
                            groups['spatial_weights'].append(p)
                    else:
                        # Per-layer group for tabular: layer_0_weights, layer_1_weights, etc.
                        groups[f'layer_{l_idx}_weights'].append(p)
                    seen.add(id(p))

    # 4. Operation pipeline parameters (per-layer for tabular)
    if hasattr(model, 'ops'):
        for l_idx, pipeline in enumerate(model.ops):
            for p in pipeline.parameters():
                if id(p) not in seen and p.requires_grad:
                    if is_spatial:
                        groups['operations'].append(p)
                    else:
                        # Per-layer ops group for tabular
                        groups[f'layer_{l_idx}_ops'].append(p)
                    seen.add(id(p))

    # 5. Connection parameters
    if hasattr(model, 'connections'):
        for conn in model.connections:
            for p in conn.parameters():
                if id(p) not in seen and p.requires_grad:
                    groups['connections'].append(p)
                    seen.add(id(p))

    # 6. Catch remaining
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            groups['other'].append(p)
            seen.add(id(p))

    # --- Build param_groups list ---
    param_groups = []
    for name, params in groups.items():
        if not params:
            continue
        # Resolve per-layer group names to their parent config:
        #   layer_N_weights -> flat_weights config
        #   layer_N_ops     -> operations config
        if name in group_configs:
            gc = group_configs[name]
        elif name.startswith('layer_') and name.endswith('_weights'):
            gc = group_configs['flat_weights']
        elif name.startswith('layer_') and name.endswith('_ops'):
            gc = group_configs['operations']
        else:
            gc = group_configs['other']
        group_dict = {
            'params': params,
            'lr': config.base_lr,
            'lr_scale': gc['lr_scale'],
            'update_freq': gc['update_freq'],
            'weight_decay': gc['weight_decay'],
            'grad_clip': gc['grad_clip'],
            'betas': config.betas,
            'eps': config.eps,
            'name': name,
        }
        param_groups.append(group_dict)

        if verbose:
            effective_lr = config.base_lr * gc['lr_scale']
            total_elems = sum(p.numel() for p in params)
            print(f"  {name:20s}: {len(params):4d} params ({total_elems:>8d} elements), "
                  f"lr={effective_lr:.6f}, freq={gc['update_freq']:2d}")

    # Ensure at least one group exists
    if not param_groups:
        param_groups = [{
            'params': [],
            'lr': config.base_lr,
            'lr_scale': 1.0,
            'update_freq': 1,
            'weight_decay': config.weight_decay,
            'grad_clip': config.max_grad_norm,
            'betas': config.betas,
            'eps': config.eps,
            'name': 'empty',
        }]

    return param_groups


def _classify_new_params(
    model: nn.Module,
    new_params: List[nn.Parameter],
) -> Dict[str, List[nn.Parameter]]:
    """Classify newly added parameters by examining model structure.

    For tabular models, uses per-layer group names (layer_N_weights,
    layer_N_ops) to match the grouping from create_asann_parameter_groups.
    """
    groups: Dict[str, List[nn.Parameter]] = defaultdict(list)
    new_set = set(id(p) for p in new_params)

    # Detect spatial vs tabular
    is_spatial = getattr(model, '_is_spatial', False) or (
        hasattr(model, 'layers') and any(
            getattr(layer, 'mode', 'flat') == 'spatial'
            for layer in model.layers
        )
    )

    # Check operations (per-layer for tabular)
    if hasattr(model, 'ops'):
        for l_idx, pipeline in enumerate(model.ops):
            for p in pipeline.parameters():
                if id(p) in new_set:
                    if is_spatial:
                        groups['operations'].append(p)
                    else:
                        groups[f'layer_{l_idx}_ops'].append(p)
                    new_set.discard(id(p))

    # Check connections
    if hasattr(model, 'connections'):
        for conn in model.connections:
            for p in conn.parameters():
                if id(p) in new_set:
                    groups['connections'].append(p)
                    new_set.discard(id(p))

    # Check encoder
    if hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            if id(p) in new_set:
                groups['encoder'].append(p)
                new_set.discard(id(p))
    elif hasattr(model, 'input_projection'):
        for p in model.input_projection.parameters():
            if id(p) in new_set:
                groups['encoder'].append(p)
                new_set.discard(id(p))
    if hasattr(model, 'output_head'):
        for p in model.output_head.parameters():
            if id(p) in new_set:
                groups['output_head'].append(p)
                new_set.discard(id(p))

    # Check layers (per-layer for tabular)
    if hasattr(model, 'layers'):
        for l_idx, layer in enumerate(model.layers):
            mode = getattr(layer, 'mode', 'flat')
            for name, p in layer.named_parameters():
                if id(p) in new_set:
                    if mode == 'spatial':
                        if 'bn' in name or 'norm' in name:
                            groups['spatial_bn'].append(p)
                        elif 'residual' in name or 'proj' in name:
                            groups['spatial_residual'].append(p)
                        else:
                            groups['spatial_weights'].append(p)
                    else:
                        groups[f'layer_{l_idx}_weights'].append(p)
                    new_set.discard(id(p))

    # Remaining
    for p in new_params:
        if id(p) in new_set:
            groups['other'].append(p)

    return groups
