"""
ASANN Diagnosis Engine — Medical-inspired architecture health monitoring.

Diagnoses the model's health by comparing training vs validation metrics trends.
Detects 7 disease types ranging from mild underfitting to severe overfitting.
Used by SurgeryScheduler to decide what treatments to apply.
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from .config import ASANNConfig


# ==================== Health States ====================

class HealthState(Enum):
    HEALTHY = auto()    # Metrics improving or acceptably plateaued together
    SICK = auto()       # One or more diseases detected
    RECOVERING = auto() # Treatment applied, waiting for recovery period


# ==================== Warmup Phases ====================

class WarmupPhase(Enum):
    HARD = auto()    # No diagnosis at all - model still learning basics
    SOFT = auto()    # Only severe diseases (severity >= 3) diagnosed
    ACTIVE = auto()  # Full diagnosis active


# ==================== Disease Types ====================

class DiseaseType(Enum):
    UNDERFITTING_MILD = auto()       # Train improving slowly, val flat
    UNDERFITTING_SEVERE = auto()     # Train loss high, not improving
    OVERFITTING_EARLY = auto()       # Gap growing but small, val still improving
    OVERFITTING_MODERATE = auto()    # Gap significant, val plateaued
    OVERFITTING_SEVERE = auto()      # Gap large, val declining
    CAPACITY_EXHAUSTION = auto()     # All layers saturated, loss plateaued
    TRAINING_STAGNATION = auto()     # Both train and val flat
    TRAINING_INSTABILITY = auto()    # Val loss exploding, model near chance — not overfitting
    MODEL_COLLAPSED = auto()         # Model predicting mean — loss ≈ 1.0 for standardized data
    OVERSMOOTHING = auto()           # Graph node features converging — all nodes look alike
    MEMORIZATION = auto()            # Neurons storing individual samples — train loss ~0 but val loss high
    STALLED_CONVERGENCE = auto()     # Val metric hasn't improved for many epochs despite healthy status
    GRADIENT_DEATH = auto()          # All layer grad norms collapsed to ~0 — model is effectively dead
    CLASS_IMBALANCE = auto()         # Model ignores minority classes — accuracy >> balanced_accuracy


# Severity levels for treatment selection
DISEASE_SEVERITY = {
    DiseaseType.UNDERFITTING_MILD: 1,
    DiseaseType.UNDERFITTING_SEVERE: 2,
    DiseaseType.OVERFITTING_EARLY: 1,
    DiseaseType.OVERFITTING_MODERATE: 2,
    DiseaseType.OVERFITTING_SEVERE: 3,
    DiseaseType.CAPACITY_EXHAUSTION: 2,
    DiseaseType.TRAINING_STAGNATION: 1,
    DiseaseType.TRAINING_INSTABILITY: 3,
    DiseaseType.MODEL_COLLAPSED: 4,
    DiseaseType.OVERSMOOTHING: 2,
    DiseaseType.MEMORIZATION: 2,
    DiseaseType.STALLED_CONVERGENCE: 2,  # Moderate — needs structural treatments (norm, JK, skip)
    DiseaseType.GRADIENT_DEATH: 4,       # Emergency — model is dead, needs weight reinit
    DiseaseType.CLASS_IMBALANCE: 2,      # Moderate — needs focal loss or class weight adjustment
}


@dataclass
class Disease:
    """A detected disease with its type, severity, and diagnostic evidence."""
    disease_type: DiseaseType
    severity: int
    evidence: Dict[str, float] = field(default_factory=dict)

    def __repr__(self):
        return f"Disease({self.disease_type.name}, severity={self.severity})"


@dataclass
class Diagnosis:
    """Result of a diagnostic evaluation."""
    state: HealthState
    diseases: List[Disease] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)
    epoch: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.state == HealthState.HEALTHY

    @property
    def is_sick(self) -> bool:
        return self.state == HealthState.SICK

    @property
    def worst_severity(self) -> int:
        if not self.diseases:
            return 0
        return max(d.severity for d in self.diseases)

    def __repr__(self):
        if self.is_healthy:
            return f"Diagnosis(HEALTHY, epoch={self.epoch})"
        names = [d.disease_type.name for d in self.diseases]
        return f"Diagnosis(SICK: {names}, epoch={self.epoch})"


# ==================== Diagnostic Signals ====================

@dataclass
class DiagnosticSnapshot:
    """Metrics captured at a single diagnostic evaluation point."""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_acc: Optional[float] = None           # Classification only
    val_acc: Optional[float] = None             # Classification only
    val_balanced_acc: Optional[float] = None    # Classification only — for imbalance detection
    mean_lss: float = 0.0                       # Mean Layer Saturation Score
    num_params: int = 0


class DiagnosisEngine:
    """Medical-inspired architecture health diagnosis.

    Monitors train vs val metrics trends over a sliding window
    and detects diseases (underfitting, overfitting, stagnation, etc.).

    Usage:
        engine = DiagnosisEngine(config, n_classes=10)
        # After each evaluation epoch:
        engine.record_snapshot(epoch, step, train_loss, val_loss, ...)
        diagnosis = engine.diagnose(epoch)
    """

    def __init__(self, config: ASANNConfig, n_classes: int = 0):
        self.config = config
        self.window_size = config.diagnosis_window
        self.n_classes = n_classes  # 0 for regression, >0 for classification

        # History of diagnostic snapshots
        self.history: deque = deque(maxlen=max(config.diagnosis_window * 3, 20))

        # Health tracking
        self.consecutive_healthy: int = 0
        self.last_treatment_epoch: int = -999  # When was last treatment applied
        self._recovery_until_epoch: int = -1   # Recovery period end epoch

        # Track the task type for correct signal computation
        self._task_type: str = "classification"

        # Performance gate: track best val loss for convergence detection
        self._best_val_loss: float = float('inf')

        # Two-phase warmup: prevent premature diagnosis during early learning
        self._warmup_phase: WarmupPhase = WarmupPhase.HARD
        self._soft_warmup_end_epoch: int = -1

        # Expose last computed signals for scheduler access (e.g. val_acc_trend)
        self.last_signals: Dict[str, float] = {}

        # Diseases to skip (from config)
        self._disabled_diseases: tuple = getattr(config, 'disabled_diseases', ())

    def set_task_type(self, task_type: str):
        """Set the task type ('classification' or 'regression')."""
        self._task_type = task_type

    def record_snapshot(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        val_balanced_acc: Optional[float] = None,
        mean_lss: float = 0.0,
        num_params: int = 0,
    ):
        """Record a diagnostic snapshot from evaluation."""
        self.history.append(DiagnosticSnapshot(
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            val_balanced_acc=val_balanced_acc,
            mean_lss=mean_lss,
            num_params=num_params,
        ))

        # Track best validation loss for performance-aware diagnosis gate
        # Guard: NaN/Inf must never corrupt _best_val_loss
        import math
        if (isinstance(val_loss, (int, float))
                and not math.isnan(val_loss) and not math.isinf(val_loss)
                and val_loss < self._best_val_loss):
            self._best_val_loss = val_loss

        # Two-phase warmup: detect when model first shows meaningful learning
        if self._warmup_phase == WarmupPhase.HARD:
            crossed_threshold = False
            hard_warmup_max = getattr(self.config, 'hard_warmup_epochs', 20)

            if self.n_classes > 0 and train_acc is not None:
                # Classification: accuracy > 2x random chance
                chance_level = 1.0 / max(self.n_classes, 2)
                crossed_threshold = train_acc > chance_level * 2.0
            elif self._task_type == "regression":
                # Regression: R^2 > ~0.3 (for standardized data, MSE < 0.7)
                crossed_threshold = train_loss < 0.7

            if crossed_threshold or epoch >= hard_warmup_max:
                self._warmup_phase = WarmupPhase.SOFT
                soft_duration = getattr(self.config, 'soft_warmup_epochs', 10)
                self._soft_warmup_end_epoch = epoch + soft_duration

        elif self._warmup_phase == WarmupPhase.SOFT:
            if epoch >= self._soft_warmup_end_epoch:
                self._warmup_phase = WarmupPhase.ACTIVE

    def notify_treatment_applied(self, epoch: int, recovery_duration: int = None):
        """Called when a treatment is applied — starts recovery period.

        Args:
            epoch: Current epoch when treatment was applied.
            recovery_duration: Optional custom recovery duration (epochs).
                If None, uses min_recovery_epochs from config.
                Structural treatments pass longer durations.
        """
        self.last_treatment_epoch = epoch
        if recovery_duration is not None:
            self._recovery_until_epoch = epoch + recovery_duration
        else:
            min_rec = getattr(self.config, 'min_recovery_epochs',
                              getattr(self.config, 'recovery_epochs', 3))
            self._recovery_until_epoch = epoch + min_rec
        self.consecutive_healthy = 0  # Reset healthy streak

    def diagnose(self, current_epoch: int) -> Diagnosis:
        """Run full diagnostic evaluation.

        Returns a Diagnosis with detected diseases (if any).
        During recovery period, returns RECOVERING state.
        """
        # Not enough data yet — too early to diagnose
        if len(self.history) < 2:
            return Diagnosis(state=HealthState.HEALTHY, epoch=current_epoch)

        # Hard warmup: no diagnosis at all — model still learning basics
        if self._warmup_phase == WarmupPhase.HARD:
            return Diagnosis(state=HealthState.HEALTHY, epoch=current_epoch)

        # During recovery period after treatment — skip diagnosis
        if current_epoch <= self._recovery_until_epoch:
            return Diagnosis(state=HealthState.RECOVERING, epoch=current_epoch)

        # Compute diagnostic signals
        signals = self._compute_signals()
        self.last_signals = signals  # Expose for scheduler (stall trajectory guard)

        # Detect diseases
        diseases = self._detect_diseases(signals)

        # Filter out disabled diseases (from config.disabled_diseases)
        if self._disabled_diseases:
            diseases = [d for d in diseases
                        if d.disease_type.name not in self._disabled_diseases]

        # Soft warmup: only severe diseases (severity >= 3) pass through
        if self._warmup_phase == WarmupPhase.SOFT:
            diseases = [d for d in diseases if d.severity >= 3]

        if diseases:
            self.consecutive_healthy = 0
            return Diagnosis(
                state=HealthState.SICK,
                diseases=diseases,
                signals=signals,
                epoch=current_epoch,
            )
        else:
            self.consecutive_healthy += 1
            return Diagnosis(
                state=HealthState.HEALTHY,
                diseases=[],
                signals=signals,
                epoch=current_epoch,
            )

    def is_architecture_stable(self) -> bool:
        """Check if architecture should be considered stable.

        Stable = HEALTHY for stability_healthy_epochs consecutive evaluations.
        """
        return self.consecutive_healthy >= self.config.stability_healthy_epochs

    def break_stability(self):
        """Break stability (called when disease detected post-stable)."""
        self.consecutive_healthy = 0

    # ==================== Signal Computation ====================

    def _compute_signals(self) -> Dict[str, float]:
        """Compute all diagnostic signals from history."""
        signals = {}
        snapshots = list(self.history)

        # Use last W snapshots for trend computation
        W = min(self.window_size, len(snapshots))
        recent = snapshots[-W:]

        if len(recent) < 2:
            return signals

        # --- Loss trends (linear regression slope) ---
        epochs = np.array([s.epoch for s in recent], dtype=float)
        train_losses = np.array([s.train_loss for s in recent])
        val_losses = np.array([s.val_loss for s in recent])

        signals["train_loss_trend"] = self._compute_slope(epochs, train_losses)
        signals["val_loss_trend"] = self._compute_slope(epochs, val_losses)

        # --- Current values ---
        signals["train_loss"] = recent[-1].train_loss
        signals["val_loss"] = recent[-1].val_loss

        # --- Train-val gap ---
        # Gap = (val_loss - train_loss) / max(val_loss, eps)
        # Positive gap means val is worse than train (overfitting)
        val_l = recent[-1].val_loss
        train_l = recent[-1].train_loss
        signals["train_val_gap"] = (val_l - train_l) / max(abs(val_l), 1e-8)

        # --- Gap trend ---
        gaps = np.array([
            (s.val_loss - s.train_loss) / max(abs(s.val_loss), 1e-8)
            for s in recent
        ])
        signals["gap_trend"] = self._compute_slope(epochs, gaps)

        # --- Accuracy trends (classification only) ---
        if recent[-1].train_acc is not None:
            train_accs = np.array([s.train_acc for s in recent if s.train_acc is not None])
            val_accs = np.array([s.val_acc for s in recent if s.val_acc is not None])
            if len(train_accs) >= 2:
                acc_epochs = epochs[-len(train_accs):]
                signals["train_acc_trend"] = self._compute_slope(acc_epochs, train_accs)
                signals["train_acc"] = train_accs[-1]
            if len(val_accs) >= 2:
                acc_epochs = epochs[-len(val_accs):]
                signals["val_acc_trend"] = self._compute_slope(acc_epochs, val_accs)
                signals["val_acc"] = val_accs[-1]

        # --- Balanced accuracy (class imbalance detection) ---
        if recent[-1].val_balanced_acc is not None:
            signals["val_balanced_acc"] = recent[-1].val_balanced_acc

        # --- Saturation ---
        signals["mean_lss"] = recent[-1].mean_lss

        # --- Relative improvement rates ---
        # How much did train_loss improve relative to its value?
        if len(recent) >= 2 and abs(recent[0].train_loss) > 1e-8:
            signals["train_loss_rel_change"] = (
                (recent[0].train_loss - recent[-1].train_loss) / abs(recent[0].train_loss)
            )
        else:
            signals["train_loss_rel_change"] = 0.0

        if len(recent) >= 2 and abs(recent[0].val_loss) > 1e-8:
            signals["val_loss_rel_change"] = (
                (recent[0].val_loss - recent[-1].val_loss) / abs(recent[0].val_loss)
            )
        else:
            signals["val_loss_rel_change"] = 0.0

        return signals

    def _detect_diseases(self, signals: Dict[str, float]) -> List[Disease]:
        """Detect diseases from diagnostic signals."""
        diseases = []
        cfg = self.config

        train_loss_trend = signals.get("train_loss_trend", 0)
        val_loss_trend = signals.get("val_loss_trend", 0)
        gap = signals.get("train_val_gap", 0)
        gap_trend = signals.get("gap_trend", 0)
        train_rel_change = signals.get("train_loss_rel_change", 0)
        val_rel_change = signals.get("val_loss_rel_change", 0)
        mean_lss = signals.get("mean_lss", 0)
        train_loss = signals.get("train_loss", 0)
        val_loss = signals.get("val_loss", 0)

        # --- NaN / Inf guard ---
        # NaN losses mean the model has diverged. All subsequent comparisons
        # (NaN > threshold) return False, which causes a false HEALTHY diagnosis.
        # Detect this early and return TRAINING_INSTABILITY immediately.
        import math
        _has_nan = (
            (isinstance(train_loss, float) and (math.isnan(train_loss) or math.isinf(train_loss)))
            or (isinstance(val_loss, float) and (math.isnan(val_loss) or math.isinf(val_loss)))
            or (isinstance(train_loss_trend, float) and math.isnan(train_loss_trend))
        )
        if _has_nan:
            diseases.append(Disease(
                disease_type=DiseaseType.TRAINING_INSTABILITY,
                severity=5,  # Maximum severity — model is diverged
                evidence={"train_loss": train_loss, "val_loss": val_loss,
                          "reason": "NaN/Inf detected in losses"},
            ))
            return diseases

        # For classification, also use accuracy trends
        val_acc_trend = signals.get("val_acc_trend", None)
        train_acc = signals.get("train_acc", None)
        val_acc = signals.get("val_acc", None)

        # --- MODEL_COLLAPSED check (highest priority — model is dead) ---
        # For standardized regression data, MSE ≈ 1.0 means predicting the mean.
        # For classification, accuracy near chance means collapsed.
        # The model must also NOT be improving (flat or rising loss for 3+ checks).
        collapse_threshold = getattr(cfg, 'collapse_loss_threshold', 0.9)
        _is_collapsed = False
        if train_acc is not None:
            # Classification: collapsed if accuracy near random chance
            chance_level = 1.0 / max(self.n_classes, 2)
            _is_collapsed = (
                train_acc < chance_level * 1.5
                and train_loss_trend >= 0  # Not improving
            )
        else:
            # Regression: collapsed if loss is near 1.0 (predicting mean of
            # standardized data) AND not improving
            _is_collapsed = (
                train_loss > collapse_threshold
                and val_loss > collapse_threshold
                and train_loss_trend >= -cfg.stagnation_threshold  # Not meaningfully improving
            )

        if _is_collapsed and len(self.history) >= 3:
            # Verify: check that loss has been high for at least 3 consecutive snapshots
            recent_3 = list(self.history)[-3:]
            all_high = all(s.train_loss > collapse_threshold * 0.8 for s in recent_3)
            if all_high:
                diseases.append(Disease(
                    disease_type=DiseaseType.MODEL_COLLAPSED,
                    severity=DISEASE_SEVERITY[DiseaseType.MODEL_COLLAPSED],
                    evidence={
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_loss_trend": train_loss_trend,
                    },
                ))
                # MODEL_COLLAPSED is terminal — skip all other disease checks.
                # Growing capacity or adding regularization on a dead model is futile.
                return diseases

        # --- OVERFITTING checks (highest priority — can cause real damage) ---

        # Guard: Don't diagnose overfitting unless the model has actually learned
        # something. A model at near-chance accuracy with a large gap is UNSTABLE,
        # not overfitting (val loss explosion ≠ memorization).
        # For classification: train_acc must be at least 2× random chance.
        # For regression (no acc available): always eligible.
        _min_overfit_acc = 2.0 / max(self.n_classes, 1) if self.n_classes > 0 else 0.0
        _can_diagnose_overfitting = (
            train_acc is None  # Regression: no accuracy, rely on gap alone
            or train_acc > _min_overfit_acc
        )

        # OVERFITTING_SEVERE: Large gap AND val getting worse
        if (_can_diagnose_overfitting
                and gap > cfg.overfitting_gap_severe and val_loss_trend > 0):
            diseases.append(Disease(
                disease_type=DiseaseType.OVERFITTING_SEVERE,
                severity=DISEASE_SEVERITY[DiseaseType.OVERFITTING_SEVERE],
                evidence={"gap": gap, "val_loss_trend": val_loss_trend},
            ))
        # OVERFITTING_MODERATE: Significant gap AND val plateaued
        elif (_can_diagnose_overfitting
                and gap > cfg.overfitting_gap_moderate
                and abs(val_loss_trend) < cfg.stagnation_threshold):
            diseases.append(Disease(
                disease_type=DiseaseType.OVERFITTING_MODERATE,
                severity=DISEASE_SEVERITY[DiseaseType.OVERFITTING_MODERATE],
                evidence={"gap": gap, "val_loss_trend": val_loss_trend},
            ))
        # OVERFITTING_EARLY: Gap growing but small, val NOT improving.
        # If val_loss is actively decreasing, the gap growth is just healthy
        # learning (train converges faster than val) — not overfitting.
        elif (_can_diagnose_overfitting
                and gap > cfg.overfitting_gap_early and gap_trend > 0
                and val_loss_trend >= 0):
            diseases.append(Disease(
                disease_type=DiseaseType.OVERFITTING_EARLY,
                severity=DISEASE_SEVERITY[DiseaseType.OVERFITTING_EARLY],
                evidence={"gap": gap, "gap_trend": gap_trend,
                          "val_loss_trend": val_loss_trend},
            ))

        # --- TRAINING_INSTABILITY check ---
        # Fires when: large gap + val loss exploding BUT model accuracy is near
        # random chance. This is NOT overfitting (model hasn't memorized anything);
        # it's logit-magnitude instability causing val loss to explode.
        # Treatment: LR reduction, not dropout.
        if (not _can_diagnose_overfitting
                and gap > cfg.overfitting_gap_severe
                and val_loss_trend > 0
                and train_acc is not None):
            diseases.append(Disease(
                disease_type=DiseaseType.TRAINING_INSTABILITY,
                severity=DISEASE_SEVERITY[DiseaseType.TRAINING_INSTABILITY],
                evidence={"gap": gap, "val_loss_trend": val_loss_trend,
                          "train_acc": train_acc},
            ))

        # --- UNDERFITTING checks ---

        # UNDERFITTING_SEVERE: Train not improving, still has high loss
        # (We detect this if train loss is not going down meaningfully)
        if (train_rel_change < cfg.underfitting_loss_threshold
                and val_rel_change < cfg.underfitting_loss_threshold
                and gap < cfg.overfitting_gap_early):
            # Both losses barely changing AND not overfitting
            # Check if train accuracy is low (for classification)
            if train_acc is not None and train_acc < 0.5:
                diseases.append(Disease(
                    disease_type=DiseaseType.UNDERFITTING_SEVERE,
                    severity=DISEASE_SEVERITY[DiseaseType.UNDERFITTING_SEVERE],
                    evidence={"train_rel_change": train_rel_change, "train_acc": train_acc},
                ))
            elif train_acc is None:
                # Regression: check if train loss is still high (above initial moving avg)
                if train_loss_trend >= 0:  # Not improving at all
                    diseases.append(Disease(
                        disease_type=DiseaseType.UNDERFITTING_SEVERE,
                        severity=DISEASE_SEVERITY[DiseaseType.UNDERFITTING_SEVERE],
                        evidence={"train_loss_trend": train_loss_trend, "train_rel_change": train_rel_change},
                    ))

        # UNDERFITTING_MILD: Train improving slowly but val is flat
        if (train_loss_trend < 0  # Train loss going down
                and abs(val_loss_trend) < cfg.stagnation_threshold  # Val flat
                and gap < cfg.overfitting_gap_early  # Not overfitting
                and DiseaseType.UNDERFITTING_SEVERE not in [d.disease_type for d in diseases]):
            diseases.append(Disease(
                disease_type=DiseaseType.UNDERFITTING_MILD,
                severity=DISEASE_SEVERITY[DiseaseType.UNDERFITTING_MILD],
                evidence={"train_loss_trend": train_loss_trend, "val_loss_trend": val_loss_trend},
            ))

        # --- CAPACITY_EXHAUSTION ---
        # All layers saturated (high mean LSS) AND BOTH train and val loss plateaued.
        # This is a strict check: the model must truly be stuck, not just slowing down.
        # Also require val to be stagnating (not just train), otherwise the model is
        # still learning to generalize even if train loss is flat.
        # AND: do NOT diagnose capacity exhaustion when overfitting is present.
        # AND: val loss must NOT be increasing — if val loss is going up while train
        # loss is flat/decreasing, that's overfitting (even if the overfitting gap
        # thresholds are disabled, e.g., for semi-supervised learning). Growing the
        # model in that scenario would make overfitting worse, not better.
        has_overfitting = any(
            d.disease_type in (DiseaseType.OVERFITTING_EARLY,
                               DiseaseType.OVERFITTING_MODERATE,
                               DiseaseType.OVERFITTING_SEVERE,
                               DiseaseType.TRAINING_INSTABILITY)
            for d in diseases
        )
        # Need at least diagnosis_window evaluations to have reliable trends
        has_enough_history = len(self.history) >= cfg.diagnosis_window
        val_also_flat = abs(val_loss_trend) < cfg.stagnation_threshold * 2
        # Guard: val loss increasing while train loss is flat/decreasing = overfitting,
        # NOT capacity exhaustion. This catches cases where overfitting gap thresholds
        # are disabled (e.g., semi-supervised learning with tiny train sets).
        val_is_worsening = (val_loss_trend > cfg.stagnation_threshold
                            and train_loss_trend <= 0)
        if (has_enough_history
                and mean_lss > cfg.saturation_threshold
                and abs(train_loss_trend) < cfg.stagnation_threshold
                and val_also_flat
                and not has_overfitting
                and not val_is_worsening):
            diseases.append(Disease(
                disease_type=DiseaseType.CAPACITY_EXHAUSTION,
                severity=DISEASE_SEVERITY[DiseaseType.CAPACITY_EXHAUSTION],
                evidence={"mean_lss": mean_lss, "train_loss_trend": train_loss_trend,
                          "val_loss_trend": val_loss_trend},
            ))

        # --- TRAINING_STAGNATION ---
        # Both train AND val flat, no overfitting
        if (abs(train_loss_trend) < cfg.stagnation_threshold
                and abs(val_loss_trend) < cfg.stagnation_threshold
                and gap < cfg.overfitting_gap_early
                and not any(d.disease_type in (DiseaseType.UNDERFITTING_SEVERE,
                                                DiseaseType.CAPACITY_EXHAUSTION)
                           for d in diseases)):
            diseases.append(Disease(
                disease_type=DiseaseType.TRAINING_STAGNATION,
                severity=DISEASE_SEVERITY[DiseaseType.TRAINING_STAGNATION],
                evidence={"train_loss_trend": train_loss_trend, "val_loss_trend": val_loss_trend},
            ))

        # --- CLASS_IMBALANCE ---
        # Model ignores minority classes: accuracy >> balanced_accuracy.
        # Only diagnose for classification tasks (val_acc is not None) and when
        # both accuracy and balanced accuracy are available.
        val_acc = signals.get("val_acc", None)
        val_bal_acc = signals.get("val_balanced_acc", None)
        if val_acc is not None and val_bal_acc is not None and val_acc > 0.5:
            imbalance_ratio = val_acc / max(val_bal_acc, 1e-6)
            imbalance_threshold = getattr(cfg, 'class_imbalance_threshold', 1.3)
            if imbalance_ratio > imbalance_threshold:
                diseases.append(Disease(
                    disease_type=DiseaseType.CLASS_IMBALANCE,
                    severity=DISEASE_SEVERITY[DiseaseType.CLASS_IMBALANCE],
                    evidence={"accuracy": val_acc, "balanced_accuracy": val_bal_acc,
                              "imbalance_ratio": imbalance_ratio},
                ))

        # === PERFORMANCE-AWARE GATE ===
        # Gate underfitting/capacity/stagnation when near best val loss.
        # Also gate overfitting when val loss is very close to best (tight gate).
        _GATABLE = {
            DiseaseType.UNDERFITTING_MILD, DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION, DiseaseType.TRAINING_STAGNATION,
        }

        if (self._best_val_loss > 0
                and self._best_val_loss < float('inf')
                and val_loss > 0):
            perf_ratio = val_loss / self._best_val_loss

            has_overfit = any(d.disease_type in (
                DiseaseType.OVERFITTING_EARLY, DiseaseType.OVERFITTING_MODERATE,
                DiseaseType.OVERFITTING_SEVERE, DiseaseType.TRAINING_INSTABILITY,
            ) for d in diseases)

            val_worsening = val_loss_trend > cfg.stagnation_threshold

            # Near-best + no overfitting + val not worsening → convergence, not disease
            if not has_overfit and not val_worsening:
                gate_tight = getattr(cfg, 'perf_gate_tight', 1.5)
                if perf_ratio < gate_tight:
                    diseases = [d for d in diseases if d.disease_type not in _GATABLE]
            # Moderately near-best → suppress stagnation only (it's just slow convergence)
            gate_loose = getattr(cfg, 'perf_gate_loose', 2.0)
            if perf_ratio < gate_loose:
                diseases = [d for d in diseases
                            if d.disease_type != DiseaseType.TRAINING_STAGNATION]

        return diseases

    # ==================== Utilities ====================

    @staticmethod
    def _compute_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Compute linear regression slope (normalized by x range)."""
        if len(x) < 2:
            return 0.0
        x_range = x[-1] - x[0]
        if x_range < 1e-8:
            return 0.0
        # Normalized slope: change per unit epoch
        x_norm = (x - x[0]) / x_range
        # Simple least-squares slope
        x_mean = x_norm.mean()
        y_mean = y.mean()
        num = ((x_norm - x_mean) * (y - y_mean)).sum()
        den = ((x_norm - x_mean) ** 2).sum()
        if abs(den) < 1e-12:
            return 0.0
        # Scale back to per-epoch units
        return float(num / den) / max(x_range, 1.0)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoint."""
        return {
            "history": [
                {
                    "epoch": s.epoch, "step": s.step,
                    "train_loss": s.train_loss, "val_loss": s.val_loss,
                    "train_acc": s.train_acc, "val_acc": s.val_acc,
                    "mean_lss": s.mean_lss, "num_params": s.num_params,
                }
                for s in self.history
            ],
            "consecutive_healthy": self.consecutive_healthy,
            "last_treatment_epoch": self.last_treatment_epoch,
            "_recovery_until_epoch": self._recovery_until_epoch,
            "_task_type": self._task_type,
            # Performance gate state
            "_best_val_loss": self._best_val_loss,
            # Warmup phase state
            "_warmup_phase": self._warmup_phase.name,
            "_soft_warmup_end_epoch": self._soft_warmup_end_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore from checkpoint."""
        self.history.clear()
        for s in state.get("history", []):
            self.history.append(DiagnosticSnapshot(**s))
        self.consecutive_healthy = state.get("consecutive_healthy", 0)
        self.last_treatment_epoch = state.get("last_treatment_epoch", -999)
        self._recovery_until_epoch = state.get("_recovery_until_epoch", -1)
        self._task_type = state.get("_task_type", "classification")
        # Performance gate state
        self._best_val_loss = state.get("_best_val_loss", float('inf'))
        # Warmup phase state
        warmup_name = state.get("_warmup_phase", "ACTIVE")  # Default ACTIVE for old checkpoints
        self._warmup_phase = WarmupPhase[warmup_name]
        self._soft_warmup_end_epoch = state.get("_soft_warmup_end_epoch", -1)
