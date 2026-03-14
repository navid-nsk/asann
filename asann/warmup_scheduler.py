"""
ASANNWarmupScheduler: LR scheduler with warmup and cosine annealing for ASANN.

Adapted from nl_framework's NLWarmupScheduler with additions for
post-surgery re-warmup.

Schedule phases:
1. Linear warmup: LR ramps from 0 → base_lr over warmup_steps
2. Cosine annealing with warm restarts (period grows by restart_mult)
3. Post-surgery re-warmup: brief 50-step warmup from 30% → 100% LR

Interaction with ASANNLRController:
- Scheduler manages LRs during warmup (critical for stable start)
- After warmup, ASANNLRController takes over via hypergradient
- Scheduler stops stepping after warmup, LRController adapts from that point
- On surgery: scheduler does brief re-warmup, then hands back

Usage:
    scheduler = ASANNWarmupScheduler(optimizer, warmup_steps=500, ...)

    for step in training:
        loss.backward()
        optimizer.step()
        if should_step_scheduler:
            scheduler.step()  # Call AFTER optimizer.step()
"""

import math
from typing import List, Optional, Dict, Any
from torch.optim import Optimizer


class ASANNWarmupScheduler:
    """LR scheduler with linear warmup, cosine annealing, and post-surgery re-warmup.

    Key properties:
    - Captures target LRs at creation (before zeroing for warmup)
    - Respects per-group lr_scale
    - Cosine restart period with multiplier
    - Post-surgery re-warmup: 50 steps from 30% to 100% LR
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.01,
        restart_period: int = 20000,
        restart_mult: float = 1.5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_period = restart_period
        self.restart_mult = restart_mult

        # Capture the target effective LRs for each group
        # (group['lr'] * lr_scale = the LR we want after warmup completes)
        self.target_lrs: List[float] = []
        for g in optimizer.param_groups:
            target_lr = g['lr'] * g.get('lr_scale', 1.0)
            self.target_lrs.append(target_lr)

        # Phase tracking
        self._phase = "warmup"  # 'warmup', 'cosine', 'post_surgery_warmup'
        self.current_step = 0
        self.current_restart = 0
        self.steps_since_restart = 0
        self.current_period = restart_period

        # Post-surgery warmup state
        self._post_surgery_warmup_len = 50
        self._post_surgery_step = 0

        # Initialize to near-zero LR (warmup starts from 0)
        self._set_lr(0.0)

    def _set_lr(self, factor: float):
        """Set learning rates with given factor (0 to 1).

        During warmup: factor goes 0 → 1
        During cosine: factor follows cosine schedule
        """
        for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
            lr_scale = group.get('lr_scale', 1.0)
            if lr_scale > 0:
                group['lr'] = target_lr * factor / lr_scale
            else:
                group['lr'] = 0.0

    def step(self):
        """Update learning rates. Call after optimizer.step()."""
        self.current_step += 1

        if self._phase == "warmup":
            # Linear warmup: 0 → 1
            factor = self.current_step / max(self.warmup_steps, 1)
            if factor >= 1.0:
                factor = 1.0
                self._phase = "cosine"
                self.steps_since_restart = 0

        elif self._phase == "post_surgery_warmup":
            # Brief re-warmup: 30% → 100%
            self._post_surgery_step += 1
            progress = min(self._post_surgery_step / self._post_surgery_warmup_len, 1.0)
            factor = 0.3 + 0.7 * progress
            if progress >= 1.0:
                self._phase = "cosine"
                self.steps_since_restart = 0

        elif self._phase == "cosine":
            # Cosine annealing with warm restarts
            self.steps_since_restart += 1

            # Check for restart
            if self.steps_since_restart >= self.current_period:
                self.current_restart += 1
                self.steps_since_restart = 0
                self.current_period = int(self.current_period * self.restart_mult)

            # Cosine factor
            progress = self.steps_since_restart / self.current_period
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        else:
            factor = 1.0

        self._set_lr(factor)

    def trigger_warm_restart(self):
        """Reset cosine annealing cycle to escape local minima.

        Called by the trainer when LR_WARMUP_RESTART treatment is prescribed.
        Resets the cosine cycle so LR jumps back to base_lr (factor=1.0),
        giving the optimizer a fresh start without changing model architecture.
        """
        self._phase = "cosine"
        self.steps_since_restart = 0
        self.current_restart += 1
        # Keep current_period (don't reset it — each restart should use
        # the multiplied period for progressively longer cycles)
        self._set_lr(1.0)  # Immediately set LR to full target

    def enter_post_surgery_warmup(self):
        """Enter brief post-surgery re-warmup phase.

        Called after major structural surgery (layer add/remove).
        Ramps LR from 30% to 100% over 50 steps.
        """
        self._phase = "post_surgery_warmup"
        self._post_surgery_step = 0

    def get_phase(self) -> str:
        """Get current scheduler phase."""
        return self._phase

    def get_lr_factor(self) -> float:
        """Get current LR multiplier (0 to 1)."""
        if self._phase == "warmup":
            return min(self.current_step / max(self.warmup_steps, 1), 1.0)
        elif self._phase == "post_surgery_warmup":
            progress = min(self._post_surgery_step / self._post_surgery_warmup_len, 1.0)
            return 0.3 + 0.7 * progress
        elif self._phase == "cosine":
            if self.current_period == 0:
                return 1.0
            progress = self.steps_since_restart / self.current_period
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        return 1.0

    def get_last_lr(self) -> List[float]:
        """Get current effective learning rates."""
        return [
            g['lr'] * g.get('lr_scale', 1.0)
            for g in self.optimizer.param_groups
        ]

    def is_in_warmup(self) -> bool:
        """Check if scheduler is in any warmup phase."""
        return self._phase in ("warmup", "post_surgery_warmup")

    def resync_target_lrs(self, optimizer: Optimizer):
        """Re-sync target LRs after optimizer groups change (surgery).

        Call this after register_structural_surgery() when groups may
        have been added or removed.
        """
        self.optimizer = optimizer
        # Extend target_lrs for new groups
        while len(self.target_lrs) < len(optimizer.param_groups):
            g = optimizer.param_groups[len(self.target_lrs)]
            self.target_lrs.append(g['lr'] * g.get('lr_scale', 1.0))
        # Trim if groups removed
        self.target_lrs = self.target_lrs[:len(optimizer.param_groups)]

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'current_restart': self.current_restart,
            'steps_since_restart': self.steps_since_restart,
            'current_period': self.current_period,
            'phase': self._phase,
            'post_surgery_step': self._post_surgery_step,
            'target_lrs': list(self.target_lrs),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state['current_step']
        self.current_restart = state['current_restart']
        self.steps_since_restart = state['steps_since_restart']
        self.current_period = state['current_period']
        self._phase = state.get('phase', 'cosine')
        self._post_surgery_step = state.get('post_surgery_step', 0)
        if 'target_lrs' in state:
            self.target_lrs = state['target_lrs']
