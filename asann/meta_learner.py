"""
MetaLearner for ASANN v4.

LR management is handled by ASANNOptimizer + ASANNLRController + ASANNWarmupScheduler.

This MetaLearner retains only:
  - Adaptive surgery threshold management (Level 2)
  - Surgery interval adaptation
  - Training dynamics tracking for threshold decisions
"""

import torch
from typing import Dict, Any
from collections import deque


class MetaLearner:
    """Adaptive thresholds + surgery interval only.

    Level 2 (slowest) optimization:
    - Adjusts surgery thresholds based on training dynamics
    - Adapts surgery interval S based on loss trends
    - No more MLP_eta -- LR is handled by ASANNOptimizer + ASANNLRController

    The key insight: the original MLP_eta was a learned function
    mapping 7 features to a scalar LR multiplier. This has been replaced by:
    1. Complexity-aware LR modulation (zero parameters, uses Frobenius norm)
    2. WSD phase scheduling (zero parameters, uses step counter)
    3. Per-parameter-group differentiated LR (configuration, not learned)

    These three mechanisms together provide more principled LR control than
    a learned network with indirect meta-gradients.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Adaptive surgery thresholds (Level 2)
        self.adaptive_thresholds = {
            "gds_k": config.gds_k,
            "nus_percentile": config.nus_percentile,
            "saturation_threshold": config.saturation_threshold,
            "identity_threshold": config.identity_threshold,
            "benefit_threshold": config.benefit_threshold,
            "connection_threshold": config.connection_threshold,
            "connection_remove_threshold": config.connection_remove_threshold,
        }

        # Store initial thresholds as FLOOR — thresholds can only go UP from here.
        # This prevents the meta-learner from making surgery ever easier than the
        # starting configuration, which was the root cause of infinite growth.
        self._initial_thresholds = {
            "gds_k": config.gds_k,
            "benefit_threshold": config.benefit_threshold,
            "connection_threshold": config.connection_threshold,
        }

        # Surgery interval adaptation
        self.current_surgery_interval = config.surgery_interval_init

        # Training dynamics tracking
        self.loss_history = deque(maxlen=config.meta_update_interval)
        self.grad_norm_history = deque(maxlen=200)
        self.surgery_counts = deque(maxlen=10)  # Track recent surgery activity

    def record_step(self, task_loss: float, grad_norm: float):
        """Record per-step training metrics for threshold adaptation."""
        self.loss_history.append(task_loss)
        self.grad_norm_history.append(grad_norm)

    def record_surgery(self, num_surgeries: int):
        """Record how many surgeries occurred in the last interval."""
        self.surgery_counts.append(num_surgeries)

    def meta_update(self, recent_losses: list):
        """Level 2 meta-parameter update (every C steps).

        Updates adaptive surgery thresholds and surgery interval.
        No more MLP_eta update -- that's handled by the optimizer.
        """
        if len(recent_losses) < 10:
            return

        self._update_surgery_thresholds(recent_losses)
        self._adjust_surgery_interval(recent_losses)

    def _update_surgery_thresholds(self, recent_losses: list):
        """Adapt surgery thresholds based on training dynamics.

        CRITICAL DESIGN RULE: Thresholds can only get TIGHTER (making surgery
        harder), never LOOSER. The initial config values are the FLOOR.

        This prevents the meta-learner from creating an infinite growth loop
        where lowered thresholds → more surgery → disrupted training → meta-learner
        lowers thresholds further.

        Strategy:
        - If loss is decreasing well -> tighten thresholds (surgery not needed)
        - If loss is stalling/worsening -> keep thresholds where they are (DON'T loosen)
        """
        losses = recent_losses[-20:]
        half = len(losses) // 2
        if half == 0:
            return

        first_half = sum(losses[:half]) / half
        second_half = sum(losses[half:]) / (len(losses) - half)
        improvement_rate = (first_half - second_half) / max(abs(first_half), 1e-8)

        if improvement_rate > 0.01:
            # Good progress -> tighten thresholds (make surgery harder)
            # Spatial models use gentler tightening since image loss improves
            # more continuously than tabular (which plateaus and needs less surgery).
            rate = self.config.meta_tighten_rate_spatial if self.config.spatial_shape else self.config.meta_tighten_rate
            conn_rate = min(rate, 1.02)  # connection threshold tightens slower
            self.adaptive_thresholds["gds_k"] *= rate
            self.adaptive_thresholds["benefit_threshold"] *= rate
            self.adaptive_thresholds["connection_threshold"] *= conn_rate
        # If loss is stalling or worsening: DO NOTHING.
        # The old code would loosen thresholds here (×0.93 or ×0.97),
        # which caused the infinite growth loop.

        # Clamp: initial value is the FLOOR, not 0.5
        floor_gds_k = self._initial_thresholds["gds_k"]
        floor_benefit = self._initial_thresholds["benefit_threshold"]
        floor_conn = self._initial_thresholds["connection_threshold"]

        self.adaptive_thresholds["gds_k"] = max(floor_gds_k, min(5.0, self.adaptive_thresholds["gds_k"]))
        self.adaptive_thresholds["benefit_threshold"] = max(
            floor_benefit, min(0.1, self.adaptive_thresholds["benefit_threshold"])
        )
        self.adaptive_thresholds["connection_threshold"] = max(
            floor_conn, min(0.9, self.adaptive_thresholds["connection_threshold"])
        )

    def _adjust_surgery_interval(self, recent_losses: list):
        """Adapt surgery interval based on training dynamics.

        CRITICAL DESIGN RULE: The interval can only INCREASE (making surgery
        rarer), never decrease. The scheduler's exponential growth (Fix 3)
        handles the primary interval expansion. This meta-learner just adds
        extra increases when training is going well.

        This prevents the doom loop where meta-learner shortens interval →
        more frequent surgery → disrupted training → meta-learner shortens more.
        """
        if len(recent_losses) < 20:
            return

        losses = recent_losses[-20:]
        trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)

        if trend < -0.01:
            # Good progress -> increase interval (fewer surgeries)
            self.current_surgery_interval = min(
                self.current_surgery_interval + 25, 3000
            )
        # If loss is stalling/worsening: DO NOT decrease interval.
        # The old code would decrease here, causing more frequent surgery
        # and more disruptions.

    def get_current_thresholds(self) -> Dict[str, float]:
        """Return the current adaptive thresholds."""
        return dict(self.adaptive_thresholds)

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "adaptive_thresholds": dict(self.adaptive_thresholds),
            "current_surgery_interval": self.current_surgery_interval,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.adaptive_thresholds = state["adaptive_thresholds"]
        self.current_surgery_interval = state["current_surgery_interval"]
