"""
ASANNLRController: Hypergradient-based Learnable LR Controller for ASANN.

Adapted from nl_framework's LearnableLRController for ASANN's architecture.

Manages per-group adaptive learning rates via hypergradient descent:
    eta_{t+1} = eta_t * exp(beta * <g_t, g_{t-1}>)

If consecutive gradients align → increase LR (good direction, go faster)
If consecutive gradients oppose → decrease LR (oscillating, slow down)

Key features:
- Per-group hypergradient tracking (each parameter group adapts independently)
- Dead zone: no LR change if hypergradient is in [-0.2, 0.1]
- Loss trend modulation: conservative when loss increasing, aggressive when decreasing
- Plateau detection: reduce all LRs when no improvement for N intervals
- Warmup gate: no LR adaptation during warmup period
- Surgery reset: clear hypergradient state after structural surgery

Integration:
    # In _training_step(), AFTER backward, BEFORE gradient clipping:
    loss.backward()
    lr_controller.step(loss.item())  # Compute hypergradients from raw gradients
    optimizer.step()
    scheduler.step()
"""

import math
from typing import Dict, List, Optional, Any
import torch
from torch import Tensor
from torch.optim import Optimizer


class ASANNLRController:
    """Hypergradient-based learnable LR controller for ASANN.

    Manages LRs for optimizer parameter groups via per-group
    hypergradient tracking with EMA smoothing.

    Update rule:
        log_scale_{t+1} = log_scale_t + meta_lr * adjustment_factor * effective_hg

    where effective_hg is the EMA-smoothed cosine similarity of consecutive
    gradients, offset by the dead zone boundary.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        update_interval: int = 50,
        warmup_steps: int = 500,
        meta_lr: float = 0.01,
        momentum: float = 0.9,
        dead_zone: tuple = (-0.03, 0.03),
        scale_min: float = 0.1,
        scale_max: float = 10.0,
        plateau_patience: int = 40,
        plateau_factor: float = 0.90,
        plateau_cooldown: int = 20,
        plateau_min_scale: float = 0.1,
        plateau_max_reductions: int = 5,
        target_lrs: Optional[List[float]] = None,
    ):
        self.optimizer = optimizer
        self.update_interval = update_interval
        self.warmup_steps = warmup_steps
        self.meta_lr = meta_lr
        self.momentum = momentum
        self.dead_zone_low, self.dead_zone_high = dead_zone
        self.scale_min = scale_min
        self.scale_max = scale_max

        # Plateau detection
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.plateau_cooldown = plateau_cooldown
        self.plateau_min_scale = plateau_min_scale
        self.plateau_max_reductions = plateau_max_reductions

        self.step_count = 0

        # Per-group state
        self.group_names: List[str] = [
            g.get('name', f'group_{i}')
            for i, g in enumerate(optimizer.param_groups)
        ]

        # Capture base LRs (the target effective LRs after warmup completes).
        # IMPORTANT: If target_lrs is provided (from warmup scheduler's captured
        # target LRs), use those. The warmup scheduler zeroes optimizer group['lr']
        # at init, so reading from optimizer directly would give 0.0.
        self.base_lrs: Dict[str, float] = {}
        if target_lrs is not None and len(target_lrs) == len(self.group_names):
            for name, target_lr in zip(self.group_names, target_lrs):
                self.base_lrs[name] = target_lr
        else:
            for name, g in zip(self.group_names, optimizer.param_groups):
                self.base_lrs[name] = g['lr'] * g.get('lr_scale', 1.0)

        # Log-scale learned adjustments (0.0 = no change)
        self.log_scales: Dict[str, float] = {name: 0.0 for name in self.group_names}

        # EMA-smoothed hypergradients per group
        self.hypergradient_ema: Dict[str, float] = {name: 0.0 for name in self.group_names}

        # Previous gradients for cosine similarity
        self._prev_grads: Dict[str, Optional[Tensor]] = {name: None for name in self.group_names}

        # Loss history for trend analysis
        self.loss_history: List[float] = []
        self.loss_window = 50

        # Plateau tracking
        self.best_loss: float = float('inf')
        self.intervals_without_improvement: int = 0
        self.intervals_since_reduction: int = 999
        self.plateau_reductions: int = 0

    def resync_groups(self, optimizer: Optional[Optimizer] = None):
        """Re-synchronize group tracking with current optimizer param groups.

        Call this after surgery changes the optimizer's parameter groups
        (e.g., when operations are added/removed and the optimizer is rebuilt).
        Preserves state for groups that still exist, initializes new groups.

        Args:
            optimizer: If provided, update self.optimizer reference (needed when
                      the optimizer is rebuilt after structural surgery).
        """
        if optimizer is not None:
            self.optimizer = optimizer

        new_names = [
            g.get('name', f'group_{i}')
            for i, g in enumerate(self.optimizer.param_groups)
        ]

        if set(new_names) == set(self.group_names):
            return  # No change needed

        # Build new state dicts preserving existing values
        new_base_lrs = {}
        new_log_scales = {}
        new_hg_ema = {}
        new_prev_grads = {}

        for name, g in zip(new_names, self.optimizer.param_groups):
            if name in self.base_lrs:
                new_base_lrs[name] = self.base_lrs[name]
                new_log_scales[name] = self.log_scales.get(name, 0.0)
                new_hg_ema[name] = self.hypergradient_ema.get(name, 0.0)
                new_prev_grads[name] = self._prev_grads.get(name, None)
            else:
                # New group — initialize fresh
                new_base_lrs[name] = g['lr'] * g.get('lr_scale', 1.0)
                new_log_scales[name] = 0.0
                new_hg_ema[name] = 0.0
                new_prev_grads[name] = None

        self.group_names = new_names
        self.base_lrs = new_base_lrs
        self.log_scales = new_log_scales
        self.hypergradient_ema = new_hg_ema
        self._prev_grads = new_prev_grads

    # =========================================================================
    # Core Step
    # =========================================================================

    def step(self, loss):
        """Update learning rates based on gradient alignment.

        Call this AFTER loss.backward() and BEFORE optimizer.step().
        Accepts float or torch.Tensor (defers .item() to avoid sync stalls).
        """
        import torch as _torch
        loss_f = loss.item() if isinstance(loss, _torch.Tensor) else float(loss)
        self.step_count += 1
        self.loss_history.append(loss_f)

        # Auto-resync if optimizer groups changed (e.g., after surgery)
        current_names = [
            g.get('name', f'group_{i}')
            for i, g in enumerate(self.optimizer.param_groups)
        ]
        if current_names != self.group_names:
            self.resync_groups()

        # Keep history bounded
        if len(self.loss_history) > self.loss_window * 2:
            self.loss_history = self.loss_history[-self.loss_window:]

        # Skip during warmup
        if self.step_count < self.warmup_steps:
            return

        # Only update at intervals
        if self.step_count % self.update_interval != 0:
            return

        self._update_optimizer_lrs()
        self._check_plateau(loss_f)

    # =========================================================================
    # Hypergradient Update
    # =========================================================================

    def _update_optimizer_lrs(self):
        """Update optimizer group LRs using hypergradient descent."""
        loss_trend = self._compute_loss_trend()

        for name in self.group_names:
            # Get current gradient (flattened and concatenated for the group)
            current_grad = self._get_group_gradient(name)
            if current_grad is None:
                continue

            # Compute hypergradient (cosine similarity with previous gradient)
            prev_grad = self._prev_grads.get(name)
            hypergradient = self._compute_hypergradient(current_grad, prev_grad)

            # Store for next step
            self._prev_grads[name] = current_grad.detach().clone()

            # EMA smoothing
            self.hypergradient_ema[name] = (
                self.momentum * self.hypergradient_ema[name] +
                (1 - self.momentum) * hypergradient
            )

            smoothed_hg = self.hypergradient_ema[name]

            # Dead zone: don't change LR for normal small oscillations
            if self.dead_zone_low < smoothed_hg < self.dead_zone_high:
                continue

            # Adjust based on loss trend
            adjustment_factor = 1.0
            if loss_trend > 0.05:   # Loss increasing
                adjustment_factor = 0.5
            elif loss_trend < -0.05:  # Loss decreasing
                adjustment_factor = 1.5

            # Offset by dead zone boundary
            if smoothed_hg >= self.dead_zone_high:
                effective_hg = smoothed_hg - self.dead_zone_high
            else:
                effective_hg = smoothed_hg - self.dead_zone_low

            # Update log-scale
            self.log_scales[name] += self.meta_lr * adjustment_factor * effective_hg

            # Clamp
            log_min = math.log(self.scale_min)
            log_max = math.log(self.scale_max)
            self.log_scales[name] = max(log_min, min(log_max, self.log_scales[name]))

        # Apply to optimizer
        self._apply_lrs()

    def _apply_lrs(self):
        """Apply learned LR scales to optimizer param groups.

        Computes effective LR from base_lr and learned scale, then writes
        to g['lr'] accounting for the group's lr_scale factor.

        After warmup, the cosine scheduler stops stepping and this becomes
        the sole LR setter. The base_lrs captured at init represent the
        target effective LRs (base_lr * lr_scale).
        """
        for i, g in enumerate(self.optimizer.param_groups):
            name = g.get('name', f'group_{i}')
            if name in self.log_scales:
                scale = math.exp(self.log_scales[name])
                base_lr = self.base_lrs.get(name, g['lr'] * g.get('lr_scale', 1.0))
                lr_scale = g.get('lr_scale', 1.0)
                if lr_scale > 0:
                    g['lr'] = base_lr * scale / lr_scale
                else:
                    g['lr'] = 0.0

    # =========================================================================
    # Hypergradient Computation
    # =========================================================================

    def _compute_hypergradient(
        self,
        current_grad: Tensor,
        prev_grad: Optional[Tensor],
    ) -> float:
        """Cosine similarity between current and previous gradients."""
        if prev_grad is None:
            return 0.0
        if current_grad.shape != prev_grad.shape:
            return 0.0

        g_curr = current_grad.flatten()
        g_prev = prev_grad.flatten()

        # Compute on GPU, single sync point
        dot_t = torch.dot(g_curr, g_prev)
        norm_product_t = g_curr.norm() * g_prev.norm()
        # Batch both .item() calls into one (they share the same CUDA stream)
        dot = dot_t.item()
        norm_product = norm_product_t.item() + 1e-8

        return dot / norm_product

    def _get_group_gradient(self, group_name: str) -> Optional[Tensor]:
        """Get flattened concatenated gradient for a parameter group."""
        for i, g in enumerate(self.optimizer.param_groups):
            if g.get('name', f'group_{i}') == group_name:
                grads = []
                for p in g['params']:
                    if p.grad is not None:
                        grads.append(p.grad.flatten())
                if grads:
                    return torch.cat(grads)
        return None

    # =========================================================================
    # Loss Trend
    # =========================================================================

    def _compute_loss_trend(self) -> float:
        """Compute loss trend. Negative = decreasing (good)."""
        if len(self.loss_history) < 10:
            return 0.0

        recent = self.loss_history[-10:]
        older = self.loss_history[-min(len(self.loss_history), self.loss_window):-10]

        if not older:
            return 0.0

        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        return (recent_mean - older_mean) / (older_mean + 1e-8)

    # =========================================================================
    # Plateau Detection
    # =========================================================================

    def _check_plateau(self, loss: float):
        """Check for plateau and reduce LRs if detected.

        Cap at max 5 reductions total to prevent LR from decaying to zero.
        The old code allowed unlimited reductions (13+ in practice), which
        caused LR to decay to ~0.0001 — starving the optimizer.
        """
        self.intervals_since_reduction += 1

        if loss < self.best_loss * 0.995:  # 0.5% improvement threshold (was 0.1%, too strict)
            self.best_loss = loss
            self.intervals_without_improvement = 0
        else:
            self.intervals_without_improvement += 1

        if (self.intervals_without_improvement >= self.plateau_patience
                and self.intervals_since_reduction >= self.plateau_cooldown
                and self.plateau_reductions < self.plateau_max_reductions):
            self._reduce_all_lrs()
            self.intervals_since_reduction = 0
            self.intervals_without_improvement = 0
            self.plateau_reductions += 1

    def _reduce_all_lrs(self):
        """Reduce all LR scales by plateau_factor."""
        log_factor = math.log(self.plateau_factor)
        log_min = math.log(self.plateau_min_scale)

        for name in self.log_scales:
            self.log_scales[name] = max(log_min, self.log_scales[name] + log_factor)

        self._apply_lrs()

    def reduce_base_lrs(self, factor: float):
        """Permanently reduce base learning rates by a factor.

        Fix 9: Called by the trainer when the treatment system prescribes
        LR_REDUCE. This reduces the *target* LRs that the hypergradient
        controller modulates around, rather than brute-forcing g['lr']
        which gets overwritten on the next controller step.

        The controller keeps its learned log_scales (relative adjustments),
        but the base they adjust FROM is now lower. The controller can still
        learn to increase LR if the new regime benefits from it.

        Args:
            factor: Multiplicative factor (e.g., 0.5 = halve base LRs)
        """
        for name in self.base_lrs:
            self.base_lrs[name] *= factor
        self._apply_lrs()
        print(f"  [LR_CTRL] Base LRs reduced by {factor:.2f}x - "
              f"controller will modulate from new baseline")

    # =========================================================================
    # Surgery Reset
    # =========================================================================

    def on_surgery(self):
        """Reset hypergradient state after structural surgery.

        After architecture changes, gradient alignment is meaningless
        (different parameters, different loss landscape).
        """
        for name in self.group_names:
            self.hypergradient_ema[name] = 0.0
            self._prev_grads[name] = None

    # resync_groups(optimizer=...) is defined above — unified single method.
    # External callers pass the optimizer arg to update self.optimizer reference.

    # =========================================================================
    # Accessors
    # =========================================================================

    def get_effective_lr(self, group_name: str) -> float:
        """Get current effective LR for a group."""
        scale = math.exp(self.log_scales.get(group_name, 0.0))
        base_lr = self.base_lrs.get(group_name, 1e-3)
        return base_lr * scale

    def get_all_effective_lrs(self) -> Dict[str, float]:
        """Get all effective LRs."""
        return {
            name: self.get_effective_lr(name)
            for name in self.group_names
        }

    def get_csv_log_data(self, loss: float) -> Dict[str, Any]:
        """Get data formatted for CSV logging."""
        loss_trend = self._compute_loss_trend()

        per_group = {}
        for name in self.group_names:
            base_lr = self.base_lrs.get(name, 0)
            scale = math.exp(self.log_scales.get(name, 0))
            effective_lr = base_lr * scale
            hypergradient = self.hypergradient_ema.get(name, 0)

            per_group[name] = {
                'base_lr': base_lr,
                'log_scale': self.log_scales.get(name, 0),
                'scale': scale,
                'effective_lr': effective_lr,
                'hypergradient': hypergradient,
            }

        return {
            'step': self.step_count,
            'loss': loss,
            'loss_trend': loss_trend,
            'plateau_reductions': self.plateau_reductions,
            'intervals_without_improvement': self.intervals_without_improvement,
            'per_group': per_group,
        }

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'step_count': self.step_count,
            'log_scales': dict(self.log_scales),
            'hypergradient_ema': dict(self.hypergradient_ema),
            'base_lrs': dict(self.base_lrs),
            'loss_history': list(self.loss_history),
            'best_loss': self.best_loss,
            'intervals_without_improvement': self.intervals_without_improvement,
            'intervals_since_reduction': self.intervals_since_reduction,
            'plateau_reductions': self.plateau_reductions,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.step_count = state['step_count']
        self.log_scales = dict(state['log_scales'])
        self.hypergradient_ema = dict(state['hypergradient_ema'])
        self.base_lrs = dict(state.get('base_lrs', {}))
        self.loss_history = list(state.get('loss_history', []))
        self.best_loss = state.get('best_loss', float('inf'))
        self.intervals_without_improvement = state.get('intervals_without_improvement', 0)
        self.intervals_since_reduction = state.get('intervals_since_reduction', 999)
        self.plateau_reductions = state.get('plateau_reductions', 0)

        # Apply loaded LRs
        self._apply_lrs()
