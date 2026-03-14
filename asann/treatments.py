"""
ASANN Treatment Planner — Evidence-based architectural treatments.

Pre-defined treatment packages from decades of ML research.
Treatments range from Level 1 (aspirin — light regularization)
to Level 4 (chemotherapy — major restructuring).

Each treatment is applied as a package (multiple changes at once),
not a single random probe.
"""

import torch
import torch.nn as nn
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from .config import ASANNConfig, snap_add_for_alignment
from .model import ASANNModel, OperationPipeline, SpatialOperationPipeline, GatedOperation
from .surgery import (
    SurgeryEngine, create_operation, get_operation_name,
    CANDIDATE_OPERATIONS, SPATIAL_CANDIDATE_OPERATIONS,
    ActivationNoise,
)
from .diagnosis import Disease, DiseaseType, Diagnosis, HealthState
import torch.nn.functional as F


# ==================== Focal Loss ====================

class _FocalCrossEntropyLoss(nn.CrossEntropyLoss):
    """Focal loss variant of CrossEntropyLoss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples so the model focuses on hard (minority) examples.
    Inherits from CrossEntropyLoss for isinstance() compatibility.
    """

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__(weight=weight, reduction='none', label_smoothing=label_smoothing)
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = super().forward(input, target)  # per-sample CE loss (no reduction)
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE) = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==================== Treatment Types ====================

class TreatmentType(Enum):
    """All available treatments, organized by level."""
    # Level 1: Aspirin (light regularization)
    DROPOUT_LIGHT = auto()
    DROPOUT_HEAVY = auto()
    WEIGHT_DECAY_BOOST = auto()
    LR_REDUCE = auto()
    LR_WARMUP_RESTART = auto()      # Reset cosine LR schedule (gentle unstalling for flat models)
    TARGET_NOISE = auto()           # Add Gaussian noise to regression targets
    ACTIVATION_NOISE = auto()       # Gaussian noise on layer activations (disrupts memorization)
    MEMORIZATION_PRUNE = auto()     # Remove low-entropy neurons that memorize

    # Level 2: Prescription (learned ops on existing layers)
    BATCHNORM_PACKAGE = auto()
    SPATIAL_CONV_PACKAGE = auto()
    ATTENTION_PACKAGE = auto()
    LABEL_SMOOTHING = auto()
    DERIVATIVE_PACKAGE = auto()
    POLYNOMIAL_PACKAGE = auto()
    BRANCHED_DIFFUSION_REACTION = auto()
    GRAPH_CONV_PACKAGE = auto()
    GRAPH_ATTENTION_PACKAGE = auto()
    GRAPH_DIFFUSION_PACKAGE = auto()
    GRAPH_BRANCHED_AGG = auto()
    GRAPH_SPECTRAL_PACKAGE = auto()
    GRAPH_GIN_PACKAGE = auto()
    GRAPH_MESSAGE_BOOST_PACKAGE = auto()   # sum*max message booster (CMPNN-inspired)
    GRAPH_NORM_PACKAGE = auto()
    GRAPH_POSITIONAL_PACKAGE = auto()
    JUMPING_KNOWLEDGE = auto()         # Attn-weighted aggregation of all layer outputs
    GNN_TYPE_SWITCH = auto()           # Switch molecular GNN type (e.g., gat -> gine)

    # Level 2.5: Operation-aware (swap existing ops based on lab findings)
    OPERATION_SWAP = auto()

    # Level 3: Procedure (structural changes)
    RESNET_BLOCK = auto()
    CAPACITY_GROWTH = auto()
    LAYER_ADDITION = auto()
    PROGRESSIVE_CHANNELS = auto()
    WIDEN_LAYERS = auto()               # Wide-ResNet style: multiply spatial widths by 1.5x
    ENCODER_UPGRADE = auto()            # Switch to a more powerful input encoder
    WIDTH_REDUCTION = auto()            # Shrink layer widths (capacity reduction for overfitting)
    LAYER_REMOVAL = auto()              # Remove a layer (depth reduction for overfitting)
    ENCODER_DOWNGRADE = auto()          # Switch to a simpler encoder (overfitting)

    # Level 4: Chemotherapy (last resort)
    AGGRESSIVE_REGULARIZATION = auto()
    ARCHITECTURE_RESET_SOFT = auto()
    DUAL_DEPTH_WIDTH = auto()

    # Class imbalance treatments
    FOCAL_LOSS = auto()                 # Switch to focal loss (down-weights easy examples)
    CLASS_WEIGHT_INCREASE = auto()      # Increase weights for underperforming classes
    BALANCED_SAMPLER = auto()           # Switch to class-balanced batch sampling

    # Level 5: Emergency — model is dead
    WEIGHT_REINITIALIZE = auto()



# Treatment level mapping
TREATMENT_LEVEL = {
    TreatmentType.DROPOUT_LIGHT: 1,
    TreatmentType.DROPOUT_HEAVY: 1,
    TreatmentType.WEIGHT_DECAY_BOOST: 1,
    TreatmentType.LR_REDUCE: 1,
    TreatmentType.LR_WARMUP_RESTART: 1,
    TreatmentType.TARGET_NOISE: 1,
    TreatmentType.ACTIVATION_NOISE: 1,
    TreatmentType.MEMORIZATION_PRUNE: 2,
    TreatmentType.BATCHNORM_PACKAGE: 2,
    TreatmentType.SPATIAL_CONV_PACKAGE: 2,
    TreatmentType.ATTENTION_PACKAGE: 2,
    TreatmentType.LABEL_SMOOTHING: 2,
    TreatmentType.DERIVATIVE_PACKAGE: 2,
    TreatmentType.POLYNOMIAL_PACKAGE: 2,
    TreatmentType.BRANCHED_DIFFUSION_REACTION: 2,
    TreatmentType.GRAPH_CONV_PACKAGE: 2,
    TreatmentType.GRAPH_ATTENTION_PACKAGE: 2,
    TreatmentType.GRAPH_DIFFUSION_PACKAGE: 2,
    TreatmentType.GRAPH_BRANCHED_AGG: 2,
    TreatmentType.GRAPH_SPECTRAL_PACKAGE: 2,
    TreatmentType.GRAPH_GIN_PACKAGE: 2,
    TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE: 2,
    TreatmentType.GRAPH_NORM_PACKAGE: 2,
    TreatmentType.GRAPH_POSITIONAL_PACKAGE: 2,
    TreatmentType.JUMPING_KNOWLEDGE: 2,
    TreatmentType.GNN_TYPE_SWITCH: 2,
    TreatmentType.OPERATION_SWAP: 2,
    TreatmentType.RESNET_BLOCK: 3,
    TreatmentType.CAPACITY_GROWTH: 3,
    TreatmentType.LAYER_ADDITION: 3,
    TreatmentType.PROGRESSIVE_CHANNELS: 3,
    TreatmentType.WIDEN_LAYERS: 3,
    TreatmentType.ENCODER_UPGRADE: 3,
    TreatmentType.WIDTH_REDUCTION: 3,
    TreatmentType.LAYER_REMOVAL: 3,
    TreatmentType.ENCODER_DOWNGRADE: 3,
    TreatmentType.AGGRESSIVE_REGULARIZATION: 4,
    TreatmentType.ARCHITECTURE_RESET_SOFT: 4,
    TreatmentType.DUAL_DEPTH_WIDTH: 4,
    TreatmentType.FOCAL_LOSS: 1,
    TreatmentType.CLASS_WEIGHT_INCREASE: 1,
    TreatmentType.BALANCED_SAMPLER: 1,
    TreatmentType.WEIGHT_REINITIALIZE: 5,
}

# Which treatments are appropriate for which diseases
# Maps disease → list of treatments in escalation order (Level 1 first)
DISEASE_TREATMENTS: Dict[DiseaseType, List[TreatmentType]] = {
    DiseaseType.OVERFITTING_EARLY: [
        TreatmentType.DROPOUT_LIGHT,
        TreatmentType.WEIGHT_DECAY_BOOST,
        TreatmentType.LABEL_SMOOTHING,
        TreatmentType.TARGET_NOISE,            # Gaussian noise on regression targets
        TreatmentType.DROPOUT_HEAVY,
        TreatmentType.ACTIVATION_NOISE,        # Fix 8: noise on hidden activations (decades-old technique)
        TreatmentType.LR_REDUCE,               # Fix 8: slower learning = less noise fitting
        TreatmentType.GRAPH_CONV_PACKAGE,      # Graph agg = structural regularization
        TreatmentType.GRAPH_NORM_PACKAGE,      # Normalization prevents memorization
    ],
    DiseaseType.OVERFITTING_MODERATE: [
        TreatmentType.DROPOUT_HEAVY,
        TreatmentType.LABEL_SMOOTHING,
        TreatmentType.TARGET_NOISE,            # Gaussian noise on regression targets
        TreatmentType.WEIGHT_DECAY_BOOST,
        TreatmentType.ACTIVATION_NOISE,        # Fix 8: noise on hidden activations
        TreatmentType.LR_REDUCE,               # Fix 8: slower learning
        TreatmentType.MEMORIZATION_PRUNE,      # Fix 8: REDUCE model capacity (textbook #1 overfitting fix!)
        TreatmentType.WIDTH_REDUCTION,         # Structural: shrink ALL layer widths by 25%
        TreatmentType.ENCODER_DOWNGRADE,       # Switch to simpler encoder if current is complex
        TreatmentType.GRAPH_CONV_PACKAGE,      # Graph agg = structural regularization
        TreatmentType.GRAPH_NORM_PACKAGE,      # Normalization prevents memorization
        TreatmentType.LAYER_REMOVAL,           # Remove a layer to reduce depth
        TreatmentType.AGGRESSIVE_REGULARIZATION,
    ],
    DiseaseType.OVERFITTING_SEVERE: [
        TreatmentType.DROPOUT_HEAVY,           # Start with dropout (lightest)
        TreatmentType.LR_REDUCE,               # Slow down fitting
        TreatmentType.AGGRESSIVE_REGULARIZATION,  # Then aggressive reg
        TreatmentType.WIDTH_REDUCTION,         # Structural: reduce model capacity
        TreatmentType.MEMORIZATION_PRUNE,      # Shrink model
        TreatmentType.ENCODER_DOWNGRADE,       # Switch to simpler encoder
        TreatmentType.LAYER_REMOVAL,           # Remove a layer
        TreatmentType.GRAPH_CONV_PACKAGE,      # Graph agg = structural regularization
        TreatmentType.GRAPH_NORM_PACKAGE,
        TreatmentType.ARCHITECTURE_RESET_SOFT,
    ],
    DiseaseType.UNDERFITTING_MILD: [
        TreatmentType.BATCHNORM_PACKAGE,
        TreatmentType.SPATIAL_CONV_PACKAGE,
        TreatmentType.ENCODER_UPGRADE,            # Try a more powerful input encoder (early!)
        TreatmentType.DERIVATIVE_PACKAGE,
        TreatmentType.POLYNOMIAL_PACKAGE,
        TreatmentType.BRANCHED_DIFFUSION_REACTION,
        TreatmentType.GRAPH_CONV_PACKAGE,
        TreatmentType.GRAPH_SPECTRAL_PACKAGE,
        TreatmentType.GRAPH_GIN_PACKAGE,
        TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE,  # sum*max message enhancement
        TreatmentType.GNN_TYPE_SWITCH,              # Switch GNN type (e.g., gat -> gine)
        TreatmentType.GRAPH_POSITIONAL_PACKAGE,
        TreatmentType.JUMPING_KNOWLEDGE,       # Combine all layer representations

        TreatmentType.PROGRESSIVE_CHANNELS,
        TreatmentType.ATTENTION_PACKAGE,
    ],
    DiseaseType.UNDERFITTING_SEVERE: [
        TreatmentType.SPATIAL_CONV_PACKAGE,
        TreatmentType.ENCODER_UPGRADE,            # Try a more powerful input encoder (early!)
        TreatmentType.DERIVATIVE_PACKAGE,
        TreatmentType.POLYNOMIAL_PACKAGE,
        TreatmentType.BRANCHED_DIFFUSION_REACTION,
        TreatmentType.GRAPH_CONV_PACKAGE,
        TreatmentType.GRAPH_SPECTRAL_PACKAGE,
        TreatmentType.GRAPH_GIN_PACKAGE,
        TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE,  # sum*max message enhancement
        TreatmentType.GRAPH_BRANCHED_AGG,
        TreatmentType.GRAPH_POSITIONAL_PACKAGE,

        TreatmentType.CAPACITY_GROWTH,
        TreatmentType.RESNET_BLOCK,
        TreatmentType.DUAL_DEPTH_WIDTH,
    ],
    DiseaseType.CAPACITY_EXHAUSTION: [
        TreatmentType.OPERATION_SWAP,
        TreatmentType.CAPACITY_GROWTH,
        TreatmentType.POLYNOMIAL_PACKAGE,
        TreatmentType.GRAPH_DIFFUSION_PACKAGE,
        TreatmentType.GRAPH_SPECTRAL_PACKAGE,
        TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE,  # sum*max message enhancement

        TreatmentType.LAYER_ADDITION,
        TreatmentType.RESNET_BLOCK,
        TreatmentType.DUAL_DEPTH_WIDTH,
    ],
    DiseaseType.TRAINING_STAGNATION: [
        TreatmentType.LR_REDUCE,
        TreatmentType.OPERATION_SWAP,
        TreatmentType.ENCODER_UPGRADE,            # Try a more powerful input encoder (early!)
        TreatmentType.BATCHNORM_PACKAGE,
        TreatmentType.DERIVATIVE_PACKAGE,
        TreatmentType.POLYNOMIAL_PACKAGE,
        TreatmentType.SPATIAL_CONV_PACKAGE,
        TreatmentType.GRAPH_CONV_PACKAGE,
        TreatmentType.GRAPH_ATTENTION_PACKAGE,
        TreatmentType.GRAPH_SPECTRAL_PACKAGE,
        TreatmentType.GRAPH_POSITIONAL_PACKAGE,
        TreatmentType.JUMPING_KNOWLEDGE,       # Combine all layer representations
        TreatmentType.PROGRESSIVE_CHANNELS,
    ],
    DiseaseType.TRAINING_INSTABILITY: [
        # Val loss exploding with near-chance accuracy: logit instability.
        # LR reduction is the primary fix, then weight decay to tame logits.
        TreatmentType.LR_REDUCE,
        TreatmentType.WEIGHT_DECAY_BOOST,
        TreatmentType.AGGRESSIVE_REGULARIZATION,
    ],
    DiseaseType.MODEL_COLLAPSED: [
        # Model is dead (predicting mean). Only option: reinitialize weights
        # to restart optimization from the current architecture.
        TreatmentType.WEIGHT_REINITIALIZE,
    ],
    DiseaseType.OVERSMOOTHING: [
        # Node features converging — all nodes look alike after graph propagation.
        # Level 1: PairNorm/GraphNorm to prevent feature collapse
        TreatmentType.GRAPH_NORM_PACKAGE,
        # JK directly combats oversmoothing by using pre-smoothed layer outputs
        TreatmentType.JUMPING_KNOWLEDGE,
        # Level 1: Light dropout on nodes acts as regularization
        TreatmentType.DROPOUT_LIGHT,
        # Level 2: Heavier dropout
        TreatmentType.DROPOUT_HEAVY,
    ],
    DiseaseType.MEMORIZATION: [
        # Neurons storing individual samples — targeted treatments
        TreatmentType.ACTIVATION_NOISE,         # Level 1: Disrupt memorization patterns
        TreatmentType.TARGET_NOISE,             # Level 1: Noise on targets (regression only)
        TreatmentType.DROPOUT_HEAVY,            # Level 1: Random neuron dropout
        TreatmentType.MEMORIZATION_PRUNE,       # Level 2: Remove memorizing neurons
        TreatmentType.WEIGHT_DECAY_BOOST,       # Level 1: Penalize large weights
        TreatmentType.WIDTH_REDUCTION,          # Level 3: Structural capacity reduction
        TreatmentType.LAYER_REMOVAL,            # Level 3: Depth reduction
    ],
    DiseaseType.STALLED_CONVERGENCE: [
        # Val metric stuck for many epochs — model needs structural improvements.
        # These are the exact components that close the gap to SOTA (tuned GCN):
        # normalization, skip connections, JK, wider capacity, and depth.
        # Order follows treatment level escalation.
        TreatmentType.LR_WARMUP_RESTART,        # Level 1: Reset cosine LR (gentle, flat-only)
        TreatmentType.BATCHNORM_PACKAGE,        # Level 2: Normalization stabilizes training
        TreatmentType.WIDEN_LAYERS,              # Level 3: Wide-ResNet style 1.5x channel widening
        TreatmentType.ENCODER_UPGRADE,           # Level 3: Try a more powerful encoder
        TreatmentType.DERIVATIVE_PACKAGE,        # Physics: derivative ops for PDE discovery
        TreatmentType.POLYNOMIAL_PACKAGE,        # Physics: polynomial ops for PDE terms
        TreatmentType.BRANCHED_DIFFUSION_REACTION,  # Physics: branched derivative+polynomial
        TreatmentType.GRAPH_NORM_PACKAGE,        # Level 2: Graph-specific normalization (PairNorm/GraphNorm)
        TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE,  # Level 2: sum*max message enhancement
        TreatmentType.GNN_TYPE_SWITCH,            # Level 2: Switch GNN type (e.g., gat -> gine)
        TreatmentType.JUMPING_KNOWLEDGE,         # Level 2: Attn-weighted multi-layer aggregation
        TreatmentType.RESNET_BLOCK,              # Level 3: Skip connections for gradient flow
        TreatmentType.LAYER_ADDITION,            # Level 3: Add depth (more layers for hierarchy)
        TreatmentType.CAPACITY_GROWTH,           # Level 1: More neurons per layer
        TreatmentType.PROGRESSIVE_CHANNELS,      # Level 2: Progressive width increase
    ],
    DiseaseType.GRADIENT_DEATH: [
        # All gradients collapsed to ~0 — model is effectively dead.
        # Caused by weight norm explosion + activation saturation (e.g., Tanh).
        # Only treatment is to reinitialize weights and restart learning.
        TreatmentType.WEIGHT_REINITIALIZE,       # Level 5: Re-initialize all weights
    ],
    DiseaseType.CLASS_IMBALANCE: [
        # Model ignores minority classes — accuracy >> balanced_accuracy.
        # Balanced sampler is the most effective treatment: ensures each batch
        # has equal class representation, so minority classes get sufficient
        # gradient signal. Focal loss down-weights easy (majority) examples.
        TreatmentType.BALANCED_SAMPLER,          # Oversample minority classes in batches
        TreatmentType.FOCAL_LOSS,                # Switch to focal loss (gamma=2.0)
        TreatmentType.CLASS_WEIGHT_INCREASE,     # Increase class weights by 2x
        TreatmentType.LABEL_SMOOTHING,           # Also helps with overconfident majority
    ],
}

# Conflict groups: treatments in the same group cannot be applied together
_CONFLICT_GROUPS = [
    {TreatmentType.DROPOUT_LIGHT, TreatmentType.DROPOUT_HEAVY,
     TreatmentType.AGGRESSIVE_REGULARIZATION},
    {TreatmentType.CAPACITY_GROWTH, TreatmentType.ARCHITECTURE_RESET_SOFT,
     TreatmentType.WIDTH_REDUCTION, TreatmentType.WIDEN_LAYERS},  # Don't grow and shrink at same time
    {TreatmentType.LAYER_ADDITION, TreatmentType.DUAL_DEPTH_WIDTH,
     TreatmentType.LAYER_REMOVAL},    # Don't add and remove layers at same time
    {TreatmentType.ENCODER_UPGRADE, TreatmentType.ENCODER_DOWNGRADE,
     TreatmentType.GNN_TYPE_SWITCH},
]


@dataclass
class Treatment:
    """A prescribed treatment with metadata."""
    treatment_type: TreatmentType
    target_disease: DiseaseType
    level: int
    details: Dict[str, Any] = field(default_factory=dict)
    target_layers: Optional[List[int]] = None  # If set, apply only to these layers

    def __repr__(self):
        layers_str = f", layers={self.target_layers}" if self.target_layers else ""
        return f"Treatment({self.treatment_type.name}, level={self.level}, for={self.target_disease.name}{layers_str})"


@dataclass
class TreatmentRecord:
    """Record of a treatment that was applied, for tracking outcomes."""
    treatment: Treatment
    epoch_applied: int
    pre_val_metric: float
    post_val_metric: Optional[float] = None  # Filled after recovery
    outcome: str = "pending"  # "improved", "no_change", "worsened"


class TreatmentPlanner:
    """Prescribes and applies evidence-based treatments for detected diseases.

    Key features:
    - Escalation: starts with lightest treatment, escalates if disease persists
    - No conflicts: won't prescribe conflicting treatments in same round
    - Treatment history: tracks what was tried and outcomes
    - Recovery awareness: waits for recovery period before re-prescribing

    Usage:
        planner = TreatmentPlanner(config, surgery_engine)
        treatments = planner.prescribe(diagnosis, model)
        planner.apply_treatments(treatments, model, optimizer, step)
    """

    def __init__(self, config: ASANNConfig, surgery_engine: SurgeryEngine):
        self.config = config
        self.engine = surgery_engine
        self.n_train_samples: int = 0  # Set by scheduler; 0 = unknown (no scaling)

        # Per-disease escalation tracking: how many times each disease was treated
        self.disease_escalation: Dict[DiseaseType, int] = defaultdict(int)

        # Treatment history
        self.treatment_history: List[TreatmentRecord] = []

        # Set of treatments already tried (to avoid repeats)
        self._tried_treatments: Dict[DiseaseType, Set[TreatmentType]] = defaultdict(set)

        # Weight decay tracking for boost/restore
        self._original_weight_decay: Optional[float] = None

        # Label smoothing state
        self._label_smoothing_active: bool = False

        # Focal loss state: prevents re-selection when already active
        self._focal_loss_active: bool = False

        # Class weight increase count: caps how many times weights can be squared
        self._class_weight_increase_count: int = 0

        # Balanced sampler: pending signal for trainer to switch DataLoader sampler
        self._balanced_sampler_active: bool = False
        self._pending_balanced_sampler: bool = False

        # Fix 9: Pending LR reduction signal for the trainer.
        # _apply_lr_reduce sets this; the trainer reads + clears it after
        # applying the reduction through the LR controller & warmup scheduler.
        # This avoids the brute-force g['lr'] *= factor that gets immediately
        # overwritten by the LR controller on the next step.
        self._pending_lr_factor: Optional[float] = None

        # Pending warm restart signal: _apply_lr_warmup_restart sets this;
        # the trainer reads + clears it and calls warmup_scheduler.trigger_warm_restart()
        self._pending_lr_restart: bool = False

        # Encoder upgrade tracking: set of encoder types already tried
        self._tried_encoders: Set[str] = set()

        # Stackable treatment count: how many times each stackable treatment
        # has been applied (LR_REDUCE, WEIGHT_DECAY_BOOST). Capped at 3.
        self._stackable_treatment_count: Dict[TreatmentType, int] = defaultdict(int)

    def prescribe(
        self,
        diagnosis: Diagnosis,
        model: ASANNModel,
        lab_report: Optional[Any] = None,
        patient_history: Optional[Any] = None,
    ) -> List[Treatment]:
        """Prescribe treatments based on diagnosis and optional lab report.

        When a lab report is provided with high confidence, its recommended
        treatment takes priority over the standard escalation ladder.

        Returns a list of treatments to apply, respecting:
        - Escalation order (lightest first, then heavier if already tried)
        - Conflict avoidance (no conflicting treatments in same round)
        - Max escalation limit
        - Semantic conflicts (never grow capacity while overfitting)
        """
        if diagnosis.is_healthy:
            return []

        # Detect if any overfitting or instability disease is present — used
        # to block capacity-growing treatments that would make things worse.
        has_overfitting = any(
            d.disease_type in (DiseaseType.OVERFITTING_EARLY,
                               DiseaseType.OVERFITTING_MODERATE,
                               DiseaseType.OVERFITTING_SEVERE,
                               DiseaseType.TRAINING_INSTABILITY)
            for d in diagnosis.diseases
        )

        # When CLASS_IMBALANCE coexists with overfitting, the "overfitting"
        # is often the model memorizing majority class patterns. Capacity-
        # reducing treatments (width reduction, layer removal) make this
        # WORSE by removing the model's ability to learn minority patterns.
        # Block destructive overfitting treatments when imbalance is present.
        has_class_imbalance = any(
            d.disease_type == DiseaseType.CLASS_IMBALANCE
            for d in diagnosis.diseases
        )
        _CAPACITY_REDUCING = {
            TreatmentType.WIDTH_REDUCTION,
            TreatmentType.LAYER_REMOVAL,
            TreatmentType.ENCODER_DOWNGRADE,
            TreatmentType.MEMORIZATION_PRUNE,
            TreatmentType.ARCHITECTURE_RESET_SOFT,
        }

        # When balanced sampling is active, the train/val gap is expected
        # (train on balanced distribution, validate on original). The model
        # needs capacity to learn minority class patterns — capacity-reducing
        # overfitting treatments (width reduction, layer removal) make
        # class imbalance WORSE. Treat balanced sampling as implicit
        # class imbalance: block destructive overfitting treatments.
        if self._balanced_sampler_active:
            has_class_imbalance = True

        # Treatments that increase model capacity — never apply during overfitting
        _CAPACITY_GROWING = {
            TreatmentType.CAPACITY_GROWTH,
            TreatmentType.LAYER_ADDITION,
            TreatmentType.DUAL_DEPTH_WIDTH,
        }

        treatments = []
        applied_types = set()

        # If lab report recommends a specific treatment with high confidence,
        # try to use it directly (bypass escalation ladder)
        if lab_report is not None and hasattr(lab_report, 'recommended_treatments'):
            for rec_type in lab_report.recommended_treatments:
                if rec_type in applied_types:
                    continue
                if has_overfitting and rec_type in _CAPACITY_GROWING:
                    continue
                # Skip treatments already tried for this disease — prevents
                # oscillation where the lab keeps recommending the same treatment
                # (e.g. DROPOUT_LIGHT for UNDERFITTING_MILD every 2 epochs)
                target_d = diagnosis.diseases[0].disease_type if diagnosis.diseases else None
                if target_d is not None and rec_type in self._tried_treatments[target_d]:
                    continue
                # Outcome-based filtering: skip treatments that failed before
                if patient_history is not None:
                    if target_d is not None:
                        effectiveness = patient_history.was_treatment_effective(target_d, rec_type)
                        if effectiveness is False:
                            continue  # Failed before — skip
                if not self._is_applicable(rec_type, model):
                    continue
                # Find the disease this treatment would target
                target_disease = diagnosis.diseases[0] if diagnosis.diseases else None
                if target_disease is not None:
                    level = TREATMENT_LEVEL.get(rec_type, 1)
                    treatment_details = {
                        "source": "lab_report",
                        "confidence": getattr(lab_report, 'confidence', 0),
                    }
                    treatment_target_layers = None

                    # Extract lab findings for operation-aware treatment
                    if hasattr(lab_report, 'results'):
                        for result in lab_report.results:
                            if hasattr(result, 'test_name') and result.test_name == "Operation Sensitivity Probe":
                                if hasattr(result, 'findings'):
                                    treatment_details["lab_findings"] = result.findings
                                    if "best_layer" in result.findings:
                                        treatment_target_layers = [result.findings["best_layer"]]

                    treatments.append(Treatment(
                        treatment_type=rec_type,
                        target_disease=target_disease.disease_type,
                        level=level,
                        details=treatment_details,
                        target_layers=treatment_target_layers,
                    ))
                    applied_types.add(rec_type)
                    break  # Apply at most one lab-recommended treatment

        # If no lab-recommended treatment was applied, fall back to standard escalation
        if not treatments:
            # Sort diseases by severity (highest first)
            sorted_diseases = sorted(
                diagnosis.diseases,
                key=lambda d: d.severity,
                reverse=True,
            )

            for disease in sorted_diseases:
                treatment = self._select_treatment_for_disease(
                    disease, model, applied_types, patient_history
                )
                if treatment is not None:
                    # Block capacity-growing treatments when overfitting
                    if has_overfitting and treatment.treatment_type in _CAPACITY_GROWING:
                        continue
                    # Block capacity-reducing treatments when class imbalance
                    # is present — the model needs capacity to learn minorities
                    if has_class_imbalance and treatment.treatment_type in _CAPACITY_REDUCING:
                        continue
                    treatments.append(treatment)
                    applied_types.add(treatment.treatment_type)

        return treatments

    def apply_treatments(
        self,
        treatments: List[Treatment],
        model: ASANNModel,
        optimizer,
        step: int,
        epoch: int,
        current_val_metric: float = 0.0,
        loss_fn=None,
    ) -> Tuple[int, List[Treatment]]:
        """Apply prescribed treatments to the model.

        Returns (count, applied_list): number of treatments successfully
        applied and the list of treatments that actually took effect.
        """
        applied = 0
        applied_treatments: List[Treatment] = []

        for treatment in treatments:
            success = self._apply_single_treatment(
                treatment, model, optimizer, step, loss_fn
            )
            # ALWAYS mark as tried so selection escalates to next candidate.
            # Without this, a failing treatment (e.g. dropout on layers that
            # already have dropout) gets re-selected every round forever.
            self._tried_treatments[treatment.target_disease].add(
                treatment.treatment_type
            )
            if success:
                applied += 1
                applied_treatments.append(treatment)
                self.treatment_history.append(TreatmentRecord(
                    treatment=treatment,
                    epoch_applied=epoch,
                    pre_val_metric=current_val_metric,
                ))
                self.disease_escalation[treatment.target_disease] += 1

        return applied, applied_treatments

    def reset_escalation_on_healthy(self):
        """Reset escalation counters when the model returns to healthy state.

        Resets the escalation *counter* so the system can attempt higher-level
        treatments if a disease recurs.  However, keeps the memory of which
        specific treatments were already tried (_tried_treatments) so we never
        re-apply a treatment that already failed — retrying Level 1 dropout
        when the model just recovered from Level 1 dropout is the root cause
        of treatment-oscillation loops.
        """
        # Fix 7: Preserve escalation for diseases that oscillate with HEALTHY.
        # Without this, the sequence: overfit → treat → recovery → HEALTHY →
        # reset escalation to 0 → overfit again → fallback loop re-selects
        # tried treatments → escalation 0→1 → repeat... means the
        # max_treatment_escalations cap (3) NEVER fires. The tried_treatments
        # set blocks the first loop, but the fallback loop bypasses it, and
        # the reset counter never reaches the cap.
        #
        # Preserve: STALLED_CONVERGENCE (fires during HEALTHY),
        #           OVERFITTING_* (oscillates with HEALTHY after treatment),
        #           MEMORIZATION (same oscillation pattern).
        # Reset: UNDERFITTING_*, TRAINING_*, MODEL_COLLAPSED, CAPACITY_EXHAUSTION
        #        (these genuinely change character between episodes).
        _preserve_diseases = {
            DiseaseType.STALLED_CONVERGENCE,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
            DiseaseType.MEMORIZATION,
        }
        preserved = {
            k: v for k, v in self.disease_escalation.items()
            if k in _preserve_diseases and v > 0
        }
        self.disease_escalation.clear()
        self.disease_escalation.update(preserved)
        # NOTE: intentionally do NOT clear _tried_treatments — this prevents
        # the infinite cycle: treat → recover → healthy → reset → same disease
        # → same treatment → treat → ... (93 treatments in 300 epochs)

    def evaluate_past_treatments(self, current_val_metric: float, current_epoch: int):
        """Update treatment outcomes based on current metrics.

        Called after recovery period to determine if treatments helped.
        Uses surgery_warmup_epochs as the evaluation delay instead of
        recovery_epochs, since GatedOperation needs the full ramp period
        (alpha → 100%) before a treatment's effect can be judged. At
        recovery_epochs=3, the gate is only at 30% — far too early.
        """
        # Fix 2: Evaluate after gate ramp completes, not after recovery_epochs
        eval_delay = max(
            self.config.recovery_epochs,
            getattr(self.config, 'surgery_warmup_epochs', 10),
        )
        for record in self.treatment_history:
            if record.outcome != "pending":
                continue
            # Check if evaluation delay has passed
            recovery_done = (
                current_epoch >= record.epoch_applied + eval_delay
            )
            if not recovery_done:
                continue

            record.post_val_metric = current_val_metric
            improvement = current_val_metric - record.pre_val_metric
            # For classification: higher metric = better (accuracy)
            # For regression with loss: lower = better (but we track primary metric)
            # The caller should ensure the metric is "higher is better"
            if improvement > 0.005:  # 0.5% improvement
                record.outcome = "improved"
            elif improvement < -0.005:
                record.outcome = "worsened"
            else:
                record.outcome = "no_change"

    # ==================== Treatment Selection ====================

    def _select_treatment_for_disease(
        self,
        disease: Disease,
        model: ASANNModel,
        already_selected: Set[TreatmentType],
        patient_history: Optional[Any] = None,
    ) -> Optional[Treatment]:
        """Select the best treatment for a specific disease.

        Uses escalation: tries lightest untried treatment first.
        Consults patient_history to skip treatments that have failed 2+ times.
        """
        candidates = list(DISEASE_TREATMENTS.get(disease.disease_type, []))
        if not candidates:
            return None

        # Smart reordering: for narrow models (d_init < 32), prioritize capacity
        # growth BEFORE structural treatments (JK, ResNet, norms). Structural ops
        # need sufficient hidden dimensions to be effective — applying them to a
        # 16-dim model is counterproductive. Grow first, then add structure.
        if disease.disease_type == DiseaseType.STALLED_CONVERGENCE:
            max_width = max((l.out_features for l in model.layers), default=16)
            if max_width < 32:
                # Move CAPACITY_GROWTH and PROGRESSIVE_CHANNELS to front
                _grow_types = {TreatmentType.CAPACITY_GROWTH, TreatmentType.PROGRESSIVE_CHANNELS}
                grow_first = [t for t in candidates if t in _grow_types]
                rest = [t for t in candidates if t not in _grow_types]
                candidates = grow_first + rest

        # Physics-first reordering: for PDE tasks, physics treatments
        # (derivative, polynomial, branched) should fire before generic
        # structural treatments. Without this, BATCHNORM/ENCODER_UPGRADE
        # consume the escalation budget before physics ops are ever tried.
        if getattr(self.config, 'physics_ops_enabled', False):
            _physics_diseases = {
                DiseaseType.UNDERFITTING_MILD,
                DiseaseType.UNDERFITTING_SEVERE,
                DiseaseType.TRAINING_STAGNATION,
                DiseaseType.STALLED_CONVERGENCE,
                DiseaseType.CAPACITY_EXHAUSTION,
            }
            if disease.disease_type in _physics_diseases:
                _physics_types = {
                    TreatmentType.DERIVATIVE_PACKAGE,
                    TreatmentType.POLYNOMIAL_PACKAGE,
                    TreatmentType.BRANCHED_DIFFUSION_REACTION,
                }
                phys_first = [t for t in candidates if t in _physics_types]
                rest = [t for t in candidates if t not in _physics_types]
                candidates = phys_first + rest

        escalation_count = self.disease_escalation[disease.disease_type]
        tried = self._tried_treatments[disease.disease_type]

        # Fix 8b: The escalation cap should ONLY block the fallback loop
        # (re-trying previously applied treatments), NOT the first-pass loop
        # that tries NEW treatments. With the cap placed before both loops,
        # max_treatment_escalations=3 means we only ever try 3 treatments
        # from any list, even if the list has 9+ treatments — making 6 of
        # them permanently unreachable. Now: always try new treatments first,
        # only cap re-tries.

        # Stackable treatments: can be re-applied even if already tried.
        # These have cumulative effects (LR halves again, weight decay doubles).
        # Max stacks configurable via config (defaults: 3 each).
        _STACKABLE = {TreatmentType.LR_REDUCE, TreatmentType.WEIGHT_DECAY_BOOST,
                       TreatmentType.LR_WARMUP_RESTART}
        _STACKABLE_MAX = {
            TreatmentType.WEIGHT_DECAY_BOOST: getattr(self.config, 'wd_boost_max_stacks', 3),
            TreatmentType.LR_REDUCE: getattr(self.config, 'lr_reduce_max_stacks', 3),
            TreatmentType.LR_WARMUP_RESTART: 2,  # At most 2 restarts per disease
        }

        # Find first untried treatment that doesn't conflict
        for treatment_type in candidates:
            if treatment_type in tried:
                # Allow stackable treatments to re-fire (up to max_stacks times)
                if treatment_type in _STACKABLE:
                    count = self._stackable_treatment_count.get(treatment_type, 0)
                    max_stacks = _STACKABLE_MAX.get(treatment_type, 3)
                    if count < max_stacks:
                        if not self._conflicts_with(treatment_type, already_selected):
                            if self._is_applicable(treatment_type, model):
                                return Treatment(
                                    treatment_type=treatment_type,
                                    target_disease=disease.disease_type,
                                    level=TREATMENT_LEVEL[treatment_type],
                                    details={"disease_evidence": disease.evidence,
                                             "severity": disease.severity,
                                             "stack_count": count + 1},
                                )
                continue  # Already tried — escalate to next in ladder

            if self._conflicts_with(treatment_type, already_selected):
                continue

            # Outcome-based filtering: skip if failed 2+ times for this disease
            if patient_history is not None:
                effectiveness = patient_history.was_treatment_effective(
                    disease.disease_type, treatment_type)
                if effectiveness is False:
                    ineff_count = patient_history.get_ineffective_treatments(
                        disease.disease_type).count(treatment_type.name)
                    if ineff_count >= 2:
                        continue  # Hard skip — failed 2+ times

            # Check if this treatment is applicable to the model
            if not self._is_applicable(treatment_type, model):
                continue

            return Treatment(
                treatment_type=treatment_type,
                target_disease=disease.disease_type,
                level=TREATMENT_LEVEL[treatment_type],
                details={"disease_evidence": disease.evidence,
                         "severity": disease.severity},
            )

        # All untried candidates were inapplicable or conflicting — try
        # re-using a tried treatment as a fallback (escalation).
        # Fix 8b: Escalation cap guards ONLY this fallback loop. Without it,
        # the fallback would keep re-selecting the first applicable tried
        # treatment forever (BATCHNORM_PACKAGE runaway: 339 in one experiment).
        if escalation_count >= self.config.max_treatment_escalations:
            return None  # All new treatments tried + re-try cap reached → give up

        for treatment_type in candidates:
            if self._conflicts_with(treatment_type, already_selected):
                continue
            if not self._is_applicable(treatment_type, model):
                continue
            # Outcome-based filtering in fallback loop too
            if patient_history is not None:
                ineff_count = patient_history.get_ineffective_treatments(
                    disease.disease_type).count(treatment_type.name)
                if ineff_count >= 2:
                    continue
            return Treatment(
                treatment_type=treatment_type,
                target_disease=disease.disease_type,
                level=TREATMENT_LEVEL[treatment_type],
                details={"escalation": escalation_count + 1,
                         "severity": disease.severity},
            )

        return None

    def _conflicts_with(
        self,
        treatment_type: TreatmentType,
        already_selected: Set[TreatmentType],
    ) -> bool:
        """Check if a treatment conflicts with already-selected treatments."""
        for conflict_group in _CONFLICT_GROUPS:
            if treatment_type in conflict_group:
                if already_selected & conflict_group:
                    return True
        return False

    def _is_applicable(self, treatment_type: TreatmentType, model: ASANNModel) -> bool:
        """Check if a treatment can be applied to the current model."""
        is_spatial = model._is_spatial

        if treatment_type == TreatmentType.LR_WARMUP_RESTART:
            # LR restart is the gentle alternative to BATCHNORM_PACKAGE for flat models.
            # Spatial/graph models benefit from BN; flat tabular models don't.
            return not is_spatial

        if treatment_type == TreatmentType.TARGET_NOISE:
            # Target noise only for regression — classification already has label smoothing
            return self.config.target_noise_scale == 0.0  # Only if not already active

        if treatment_type == TreatmentType.ACTIVATION_NOISE:
            return self.config.activation_noise_std == 0.0  # Only if not already active

        if treatment_type == TreatmentType.MEMORIZATION_PRUNE:
            # Only if model has neurons above d_min to prune
            return any(layer.out_features > self.config.d_min
                       for layer in model.layers)

        if treatment_type == TreatmentType.JUMPING_KNOWLEDGE:
            # JK only for non-spatial models with 2+ layers, not already enabled.
            # CRITICAL: JK requires sufficient hidden dimension for meaningful
            # projections. With d_jk < 32, the attention-weighted averaging
            # destroys discriminative information and causes model collapse
            # (observed on CiteSeer with d_init=16).
            d_jk = model.config.jk_d if model.config.jk_d > 0 else getattr(model, '_effective_d_init', 16)
            return (not is_spatial
                    and model.num_layers >= 2
                    and d_jk >= 32  # Minimum safe dimension for JK projections
                    and not getattr(model, '_jk_enabled', False))

        if treatment_type == TreatmentType.SPATIAL_CONV_PACKAGE:
            return is_spatial  # Only for spatial models

        if treatment_type == TreatmentType.ATTENTION_PACKAGE:
            return is_spatial  # Channel attention only for spatial models

        # Graph treatments: only for models with graph auxiliary data
        if treatment_type in (TreatmentType.GRAPH_CONV_PACKAGE,
                              TreatmentType.GRAPH_ATTENTION_PACKAGE,
                              TreatmentType.GRAPH_DIFFUSION_PACKAGE,
                              TreatmentType.GRAPH_BRANCHED_AGG,
                              TreatmentType.GRAPH_SPECTRAL_PACKAGE,
                              TreatmentType.GRAPH_GIN_PACKAGE,
                              TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE,
                              TreatmentType.GRAPH_NORM_PACKAGE,
                              TreatmentType.GRAPH_POSITIONAL_PACKAGE):
            return (getattr(model, '_graph_edge_index', None) is not None
                    and getattr(model, '_graph_num_nodes', 0) > 0)

        if treatment_type == TreatmentType.GNN_TYPE_SWITCH:
            # Only applicable for molecular graph encoders not already behind a bridge
            from .encoders import MolecularGraphEncoder, GatedEncoderBridge
            encoder = model.encoder
            if isinstance(encoder, GatedEncoderBridge):
                return False  # Already mid-switch
            if not isinstance(encoder, MolecularGraphEncoder):
                return False
            # Only switch if current type can be upgraded (gat/gcn/gin -> gine)
            return encoder._gnn_type in ("gat", "gcn", "gin")

        # Physics-oriented packages: gated by physics_ops_enabled
        if treatment_type in (TreatmentType.DERIVATIVE_PACKAGE,
                              TreatmentType.POLYNOMIAL_PACKAGE,
                              TreatmentType.BRANCHED_DIFFUSION_REACTION):
            return getattr(self.config, 'physics_ops_enabled', False)

        if treatment_type == TreatmentType.FOCAL_LOSS:
            return not self._focal_loss_active  # Already active -> skip

        if treatment_type == TreatmentType.CLASS_WEIGHT_INCREASE:
            # Allow at most 2 weight squarings (weights grow exponentially)
            return self._class_weight_increase_count < 2

        if treatment_type == TreatmentType.BALANCED_SAMPLER:
            return not self._balanced_sampler_active

        if treatment_type == TreatmentType.LAYER_ADDITION:
            return model.num_layers < 20  # Safety limit

        if treatment_type == TreatmentType.ARCHITECTURE_RESET_SOFT:
            # Only applicable if model has ops beyond basic ReLU
            for ops in model.ops:
                if ops.num_operations > 1:
                    return True
            return False

        if treatment_type == TreatmentType.OPERATION_SWAP:
            # Only applicable if model has non-protected ops to swap
            _PROTECTED = (nn.ReLU, nn.GELU, nn.SiLU,
                          nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)
            # For graph models, graph aggregation ops are also protected
            if getattr(model, '_is_graph', False):
                from .surgery import (NeighborAggregation, GraphAttentionAggregation,
                    GraphDiffusion, SpectralConv, MessagePassingGIN, DegreeScaling,
                    APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                    SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                    MessageBooster, DirectionalDiffusion, AdaptiveGraphConv)
                _PROTECTED = _PROTECTED + (
                    NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
                    SpectralConv, MessagePassingGIN, DegreeScaling,
                    APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                    SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                    MessageBooster, DirectionalDiffusion, AdaptiveGraphConv,
                )
            for l in range(model.num_layers):
                for op in model.ops[l].operations:
                    inner = op.operation if isinstance(op, GatedOperation) else op
                    if not isinstance(inner, _PROTECTED):
                        return True
            return False

        # Physical limit: CAPACITY_GROWTH blocked only when model is at max width
        if treatment_type == TreatmentType.CAPACITY_GROWTH:
            for l in range(model.num_layers):
                layer = model.layers[l]
                is_spatial_layer = hasattr(layer, 'mode') and layer.mode == "spatial"
                current_width = layer.out_channels if is_spatial_layer else layer.out_features
                if current_width >= self.config.max_channels:
                    return False

        # WIDEN_LAYERS: applicable if any spatial layer has out_channels < max_channels
        if treatment_type == TreatmentType.WIDEN_LAYERS:
            has_widenable = False
            for l in range(model.num_layers):
                layer = model.layers[l]
                if hasattr(layer, 'mode') and layer.mode == "spatial":
                    if layer.out_channels < self.config.max_channels:
                        has_widenable = True
                        break
            return has_widenable

        # ENCODER_UPGRADE: only if encoder_candidates has untried candidates
        if treatment_type == TreatmentType.ENCODER_UPGRADE:
            candidates = model.config.encoder_candidates
            if candidates is None or len(candidates) == 0:
                return False  # No candidates configured
            # Check if there are untried candidates
            from .encoders import GatedEncoderBridge
            # Don't switch while a bridge is active
            if isinstance(model.encoder, GatedEncoderBridge):
                return False
            # Get current encoder type
            current_type = getattr(model.encoder, 'encoder_type', 'linear')
            untried = [c for c in candidates
                       if c != current_type and c not in self._tried_encoders]
            return len(untried) > 0

        # WIDTH_REDUCTION: only if at least one layer is wider than d_min
        if treatment_type == TreatmentType.WIDTH_REDUCTION:
            return any(layer.out_features > self.config.d_min
                       for layer in model.layers
                       if not (hasattr(layer, 'mode') and layer.mode == "spatial"))

        # LAYER_REMOVAL: only if model has more than 1 layer
        if treatment_type == TreatmentType.LAYER_REMOVAL:
            return model.num_layers > 1

        # ENCODER_DOWNGRADE: only if current encoder is NOT the simplest (linear)
        if treatment_type == TreatmentType.ENCODER_DOWNGRADE:
            from .encoders import GatedEncoderBridge
            # Don't switch while a bridge is active
            if isinstance(model.encoder, GatedEncoderBridge):
                return False
            current_type = getattr(model.encoder, 'encoder_type', 'linear')
            return current_type != 'linear'

        return True

    # ==================== Treatment Application ====================

    def _apply_single_treatment(
        self,
        treatment: Treatment,
        model: ASANNModel,
        optimizer,
        step: int,
        loss_fn=None,
    ) -> bool:
        """Apply a single treatment. Returns True if successful."""
        tt = treatment.treatment_type

        if tt == TreatmentType.DROPOUT_LIGHT:
            return self._apply_dropout(model, optimizer, step,
                                       p=getattr(self.config, 'dropout_light_p', 0.1))
        elif tt == TreatmentType.DROPOUT_HEAVY:
            return self._apply_dropout(model, optimizer, step,
                                       p=getattr(self.config, 'dropout_heavy_p', 0.3))
        elif tt == TreatmentType.WEIGHT_DECAY_BOOST:
            return self._apply_weight_decay_boost(
                optimizer, factor=getattr(self.config, 'wd_boost_factor', 2.0))
        elif tt == TreatmentType.LR_REDUCE:
            return self._apply_lr_reduce(
                optimizer, factor=getattr(self.config, 'lr_reduce_factor', 0.5))
        elif tt == TreatmentType.LR_WARMUP_RESTART:
            return self._apply_lr_warmup_restart(optimizer)
        elif tt == TreatmentType.TARGET_NOISE:
            return self._apply_target_noise(model)
        elif tt == TreatmentType.ACTIVATION_NOISE:
            return self._apply_activation_noise(model, optimizer, step, treatment.target_layers)
        elif tt == TreatmentType.MEMORIZATION_PRUNE:
            return self._apply_memorization_prune(model, optimizer, step, treatment.target_layers)
        elif tt == TreatmentType.JUMPING_KNOWLEDGE:
            return self._apply_jumping_knowledge(model, optimizer, step)
        elif tt == TreatmentType.BATCHNORM_PACKAGE:
            return self._apply_batchnorm_package(model, optimizer, step)
        elif tt == TreatmentType.SPATIAL_CONV_PACKAGE:
            return self._apply_spatial_conv_package(model, optimizer, step)
        elif tt == TreatmentType.ATTENTION_PACKAGE:
            return self._apply_attention_package(model, optimizer, step)
        elif tt == TreatmentType.LABEL_SMOOTHING:
            return self._apply_label_smoothing(loss_fn)
        elif tt == TreatmentType.RESNET_BLOCK:
            return self._apply_resnet_block(model, optimizer, step)
        elif tt == TreatmentType.OPERATION_SWAP:
            return self._apply_operation_swap(model, optimizer, step, treatment)
        elif tt == TreatmentType.CAPACITY_GROWTH:
            severity = treatment.details.get("severity",
                                             treatment.details.get("disease_evidence", {}).get("severity", 2))
            return self._apply_capacity_growth(model, optimizer, step, severity=severity)
        elif tt == TreatmentType.LAYER_ADDITION:
            return self._apply_layer_addition(model, optimizer, step)
        elif tt == TreatmentType.PROGRESSIVE_CHANNELS:
            return self._apply_progressive_channels(model, optimizer, step)
        elif tt == TreatmentType.AGGRESSIVE_REGULARIZATION:
            return self._apply_aggressive_regularization(model, optimizer, step)
        elif tt == TreatmentType.ARCHITECTURE_RESET_SOFT:
            return self._apply_architecture_reset_soft(model, optimizer, step)
        elif tt == TreatmentType.DUAL_DEPTH_WIDTH:
            return self._apply_dual_depth_width(model, optimizer, step)
        elif tt == TreatmentType.DERIVATIVE_PACKAGE:
            return self._apply_derivative_package(model, optimizer, step)
        elif tt == TreatmentType.POLYNOMIAL_PACKAGE:
            return self._apply_polynomial_package(model, optimizer, step)
        elif tt == TreatmentType.BRANCHED_DIFFUSION_REACTION:
            return self._apply_branched_diffusion_reaction(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_CONV_PACKAGE:
            return self._apply_graph_conv_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_ATTENTION_PACKAGE:
            return self._apply_graph_attention_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_DIFFUSION_PACKAGE:
            return self._apply_graph_diffusion_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_BRANCHED_AGG:
            return self._apply_graph_branched_agg(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_SPECTRAL_PACKAGE:
            return self._apply_graph_spectral_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_GIN_PACKAGE:
            return self._apply_graph_gin_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_MESSAGE_BOOST_PACKAGE:
            return self._apply_graph_message_boost_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_NORM_PACKAGE:
            return self._apply_graph_norm_package(model, optimizer, step)
        elif tt == TreatmentType.GRAPH_POSITIONAL_PACKAGE:
            return self._apply_graph_positional_package(model, optimizer, step)
        elif tt == TreatmentType.WEIGHT_REINITIALIZE:
            return self._apply_weight_reinitialize(model, optimizer, step)
        elif tt == TreatmentType.GNN_TYPE_SWITCH:
            return self._apply_gnn_type_switch(model, optimizer, step)
        elif tt == TreatmentType.WIDEN_LAYERS:
            return self._apply_widen_layers(model, optimizer, step)
        elif tt == TreatmentType.ENCODER_UPGRADE:
            return self._apply_encoder_upgrade(model, optimizer, step)
        elif tt == TreatmentType.WIDTH_REDUCTION:
            return self._apply_width_reduction(model, optimizer, step)
        elif tt == TreatmentType.LAYER_REMOVAL:
            return self._apply_layer_removal(model, optimizer, step)
        elif tt == TreatmentType.ENCODER_DOWNGRADE:
            return self._apply_encoder_downgrade(model, optimizer, step)
        elif tt == TreatmentType.FOCAL_LOSS:
            return self._apply_focal_loss(loss_fn)
        elif tt == TreatmentType.CLASS_WEIGHT_INCREASE:
            return self._apply_class_weight_increase(loss_fn)
        elif tt == TreatmentType.BALANCED_SAMPLER:
            return self._apply_balanced_sampler()

        return False

    # ===== Level 1: Aspirin =====

    def _apply_dropout(self, model: ASANNModel, optimizer, step: int, p: float) -> bool:
        """Add or increase Dropout to all layers.

        If a layer has no dropout: add Dropout(p).
        If a layer already has dropout with rate < p: increase rate to p.
        If a layer already has dropout with rate >= p: skip (already strong enough).
        Maximum dropout rate is capped at 0.5 to prevent killing the layer.
        """
        device = self.config.device
        max_dropout = 0.5
        p = min(p, max_dropout)
        changed = 0

        for l in range(model.num_layers):
            ops = model.ops[l]
            # Find existing dropout (raw or inside GatedOperation)
            existing_dropout = None
            for op in ops.operations:
                if isinstance(op, nn.Dropout):
                    existing_dropout = op
                    break
                if hasattr(op, 'operation') and isinstance(op.operation, nn.Dropout):
                    existing_dropout = op.operation
                    break

            if existing_dropout is not None:
                if existing_dropout.p < p:
                    old_p = existing_dropout.p
                    existing_dropout.p = p
                    changed += 1
                    print(f"  [TREATMENT] Layer {l}: dropout {old_p:.2f} -> {p:.2f}")
            else:
                new_op = nn.Dropout(p).to(device)
                ops.add_operation(new_op, ops.num_operations)
                changed += 1
                print(f"  [TREATMENT] Layer {l}: added Dropout({p:.2f})")

        if changed > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_dropout')
        return changed > 0

    def _apply_weight_decay_boost(self, optimizer, factor: float) -> bool:
        """Increase weight decay by factor. Stackable up to max_stacks times."""
        if self._original_weight_decay is None:
            self._original_weight_decay = self.config.optimizer.weight_decay

        new_wd = self.config.optimizer.weight_decay * factor
        self.config.optimizer.weight_decay = new_wd

        # Update optimizer param groups
        for group in optimizer.param_groups:
            if group.get('weight_decay', 0) > 0:
                group['weight_decay'] = new_wd

        # Track stackable count
        max_stacks = getattr(self.config, 'wd_boost_max_stacks', 3)
        self._stackable_treatment_count[TreatmentType.WEIGHT_DECAY_BOOST] += 1
        count = self._stackable_treatment_count[TreatmentType.WEIGHT_DECAY_BOOST]
        print(f"  [TREATMENT] Weight decay boosted {factor}x -> {new_wd:.6f} "
              f"(stack {count}/{max_stacks})")
        return True

    def _apply_lr_reduce(self, optimizer, factor: float) -> bool:
        """Reduce learning rate by factor. Stackable up to 3 times.

        Fix 9: Instead of brute-forcing g['lr'] *= factor (which the LR
        controller overwrites on the very next step), we set a pending signal.
        The trainer reads this after treatment application and reduces:
          1. LR controller's base_lrs (permanent target LRs)
          2. Warmup scheduler's target_lrs (for consistency)
        This way the reduction is permanent and the controller learns from
        the new, lower baseline.

        Also applies immediately to optimizer groups as a fallback (in case
        the trainer doesn't have an LR controller).
        """
        if self._pending_lr_factor is not None:
            # Already pending — compound the factors so both take effect
            # when the trainer reads the signal.
            self._pending_lr_factor *= factor
        else:
            self._pending_lr_factor = factor

        # Immediate fallback: reduce optimizer groups directly.
        # Will be properly overridden by the trainer via LR controller.
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * factor

        # Track stackable count
        max_stacks = getattr(self.config, 'lr_reduce_max_stacks', 3)
        self._stackable_treatment_count[TreatmentType.LR_REDUCE] += 1
        count = self._stackable_treatment_count[TreatmentType.LR_REDUCE]
        print(f"  [TREATMENT] LR reduced by {factor:.2f}x "
              f"(pending signal for LR controller, stack {count}/{max_stacks})")
        return True

    def _apply_lr_warmup_restart(self, optimizer) -> bool:
        """Reset the cosine LR schedule to escape local minima.

        Gentle alternative to BATCHNORM_PACKAGE for flat/tabular models.
        Resets the cosine annealing cycle so LR jumps back to base_lr,
        giving the optimizer a fresh chance to find better parameters
        without altering model architecture.

        Uses a pending signal (like LR_REDUCE) because the warmup scheduler
        is owned by the trainer, not the treatment planner.
        """
        self._pending_lr_restart = True

        # Track stackable count
        self._stackable_treatment_count[TreatmentType.LR_WARMUP_RESTART] += 1
        count = self._stackable_treatment_count[TreatmentType.LR_WARMUP_RESTART]
        print(f"  [TREATMENT] LR warm restart scheduled "
              f"(pending signal for warmup scheduler, stack {count}/2)")
        return True

    def _apply_jumping_knowledge(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Enable Jumping Knowledge on the model.

        JK creates per-layer projections and attention weights so the output
        is a learned combination of ALL layer representations, not just the last.
        This is especially powerful for graph models where oversmoothing causes
        later layers to lose discriminative information.
        """
        if getattr(model, '_jk_enabled', False):
            return False  # Already enabled

        model.enable_jk()

        # Register new JK parameters with optimizer (auto-detects all new params)
        optimizer.register_structural_surgery(model, surgery_type='treatment_jk')

        print(f"  [TREATMENT] Jumping Knowledge enabled: "
              f"d_jk={model._jk_d}, {model.num_layers} projections")
        return True

    def _apply_target_noise(self, model: ASANNModel) -> bool:
        """Enable Gaussian noise injection on regression targets.

        Sets config.target_noise_scale which the trainer reads during loss
        computation. The noise std is relative to a scale of 0.01 (1% of
        normalized target range, since targets are typically standardized).
        """
        if self.config.target_noise_scale > 0:
            return False  # Already active
        self.config.target_noise_scale = 0.01  # 1% noise
        print(f"  [TREATMENT] Target noise enabled: scale={self.config.target_noise_scale}")
        return True

    def _apply_activation_noise(self, model: ASANNModel, optimizer, step: int,
                                target_layers=None) -> bool:
        """Insert ActivationNoise ops into memorizing layers.

        ActivationNoise is parameter-free (no learnable weights), so it doesn't
        need optimizer registration — just insert the op into the pipeline.
        """
        if self.config.activation_noise_std > 0:
            return False  # Already active

        self.config.activation_noise_std = 0.05  # 5% Gaussian noise

        # Insert ActivationNoise op into target layers (or all layers if no target)
        layers_to_treat = target_layers or list(range(model.num_layers))
        inserted = 0
        for l_idx in layers_to_treat:
            pipeline = model.ops[l_idx]
            # Check if already has ActivationNoise
            has_noise = any(isinstance(op, ActivationNoise) for op in pipeline.operations)
            if not has_noise:
                d = model.layers[l_idx].out_features
                noise_op = create_operation(
                    "activation_noise", d,
                    device=self.config.device, config=self.config)
                pipeline.operations.append(noise_op)
                inserted += 1

        if inserted > 0:
            # No optimizer registration needed (parameter-free op)
            print(f"  [TREATMENT] ActivationNoise inserted on {inserted} layers "
                  f"(std={self.config.activation_noise_std})")
        return inserted > 0

    def _apply_memorization_prune(self, model: ASANNModel, optimizer, step: int,
                                   target_layers=None) -> bool:
        """Remove lowest-activation neurons to disrupt memorization.

        Removes ~10% of neurons from target layers, picking the neuron
        with the smallest weight magnitude (likely the most specialized /
        memorizing neuron).
        """
        surgery_engine = SurgeryEngine(self.config)

        pruned = 0
        layers_to_treat = target_layers or list(range(model.num_layers))
        for l_idx in layers_to_treat:
            layer = model.layers[l_idx]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            min_width = max(self.config.d_min,
                            getattr(self.config, 'min_spatial_channels', 8)) if is_spatial else self.config.d_min
            width = layer.out_channels if is_spatial else layer.out_features
            n_prune = max(1, width // 10)  # Remove ~10% of neurons
            for _ in range(n_prune):
                layer = model.layers[l_idx]  # refresh after surgery
                current_width = layer.out_channels if is_spatial else layer.out_features
                if current_width <= min_width:
                    break
                # Pick neuron with smallest weight magnitude (most specialized)
                if is_spatial:
                    # Conv weight: [C_out, C_in, kH, kW] → sum over C_in, kH, kW
                    w = layer.conv.weight.data
                    neuron_mag = w.abs().sum(dim=(1, 2, 3))  # [C_out]
                else:
                    # Linear weight: [out, in] → sum over in
                    w = layer.weight.data
                    neuron_mag = w.abs().sum(dim=1)  # [out_features]
                neuron_idx = neuron_mag.argmin().item()
                surgery_engine.remove_neuron(
                    model, l_idx, neuron_idx, optimizer, step)
                pruned += 1

        if pruned > 0:
            print(f"  [TREATMENT] Memorization prune: removed {pruned} neurons "
                  f"from {len(layers_to_treat)} layers")
        return pruned > 0

    # ===== Level 2: Prescription =====

    def _apply_batchnorm_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add BatchNorm to all layers that don't have it."""
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            ops = model.ops[l]
            layer = model.layers[l]

            # Check if layer already has BatchNorm (unwrap GatedOperation)
            has_bn = any(
                isinstance(op.operation if isinstance(op, GatedOperation) else op,
                           (nn.BatchNorm1d, nn.BatchNorm2d))
                for op in ops.operations
            )
            if has_bn:
                continue

            # Determine correct BN type
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            if is_spatial:
                C = layer.out_channels
                new_op = nn.BatchNorm2d(C).to(device)
            else:
                d = layer.out_features
                new_op = nn.BatchNorm1d(d).to(device)

            # Insert before activation (position 0 in pipeline).
            # Gated: immunosuppression prevents distribution shock from mid-training insertion.
            warmup = self.config.surgery_warmup_epochs
            ops.add_operation(new_op, 0, gated=True, warmup_epochs=warmup)
            added += 1

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_batchnorm')
        return added > 0

    def _apply_spatial_conv_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add DWSep conv to ALL spatial layers that lack spatial ops."""
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            if not (hasattr(layer, 'mode') and layer.mode == "spatial"):
                continue

            ops = model.ops[l]
            # Check if layer already has learned spatial ops
            has_spatial_op = any(
                'conv' in get_operation_name(op).lower()
                or 'dw_sep' in get_operation_name(op).lower()
                or 'pointwise' in get_operation_name(op).lower()
                for op in ops.operations
            )
            if has_spatial_op:
                continue

            # Add DWSep conv — the workhorse of efficient spatial models
            C = layer.out_channels
            _, H, W = layer.spatial_shape
            try:
                new_op = create_operation(
                    "spatial_dw_sep_k3", d=C * H * W, device=device,
                    config=self.config, spatial_shape=(C, H, W),
                )
                new_op._asann_added_step = step
                # Insert after activation (find ReLU/GELU position)
                insert_pos = ops.num_operations  # Default: end
                for i, op in enumerate(ops.operations):
                    if isinstance(op, (nn.ReLU, nn.GELU, nn.SiLU)):
                        insert_pos = i + 1
                        break
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(new_op, insert_pos, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception:
                continue

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_spatial_conv')
        return added > 0

    def _apply_attention_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add channel attention (SE block) to spatial layers that lack it."""
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            if not (hasattr(layer, 'mode') and layer.mode == "spatial"):
                continue

            ops = model.ops[l]
            # Check if layer already has attention
            has_attention = any(
                'attention' in get_operation_name(op).lower()
                for op in ops.operations
            )
            if has_attention:
                continue

            # Don't exceed max ops per layer
            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            C = layer.out_channels
            _, H, W = layer.spatial_shape
            try:
                new_op = create_operation(
                    "channel_attention", d=C * H * W, device=device,
                    config=self.config, spatial_shape=(C, H, W),
                )
                new_op._asann_added_step = step
                # Insert at end
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(new_op, ops.num_operations, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception:
                continue

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_attention')
        return added > 0

    def _apply_label_smoothing(self, loss_fn) -> bool:
        """Enable label smoothing on the loss function."""
        if self._label_smoothing_active:
            return False

        if loss_fn is not None and hasattr(loss_fn, 'task_loss_fn'):
            # ASANNLoss wraps the task loss
            task_loss = loss_fn.task_loss_fn
            if isinstance(task_loss, nn.CrossEntropyLoss):
                # Preserve existing class weights when adding label smoothing
                existing_weight = getattr(task_loss, 'weight', None)
                loss_fn.task_loss_fn = nn.CrossEntropyLoss(
                    weight=existing_weight,
                    label_smoothing=self.config.label_smoothing_alpha,
                )
                self._label_smoothing_active = True
                return True
        return False

    # ===== Class Imbalance Treatments =====

    def _apply_focal_loss(self, loss_fn) -> bool:
        """Switch from CrossEntropyLoss to FocalLoss.

        Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Down-weights easy (majority) examples so the model focuses on
        hard (minority) examples. gamma=2.0 is the standard default.
        """
        if loss_fn is None or not hasattr(loss_fn, 'task_loss_fn'):
            return False
        task_loss = loss_fn.task_loss_fn
        if not isinstance(task_loss, nn.CrossEntropyLoss):
            return False
        # Already focal
        if isinstance(task_loss, _FocalCrossEntropyLoss):
            return False

        gamma = getattr(self.config, 'focal_loss_gamma', 2.0)
        # Preserve existing class weights if any
        weight = getattr(task_loss, 'weight', None)
        label_smoothing = getattr(task_loss, 'label_smoothing', 0.0)
        loss_fn.task_loss_fn = _FocalCrossEntropyLoss(
            gamma=gamma, weight=weight, label_smoothing=label_smoothing)
        self._focal_loss_active = True
        print(f"  [TREATMENT] Switched to focal loss (gamma={gamma})")
        return True

    def _apply_class_weight_increase(self, loss_fn) -> bool:
        """Increase class weights by 2x for minority classes.

        If the loss function has class weights, doubles the max/min weight ratio.
        If no weights exist, creates inverse-frequency-style weights.
        """
        if loss_fn is None or not hasattr(loss_fn, 'task_loss_fn'):
            return False
        task_loss = loss_fn.task_loss_fn
        if not isinstance(task_loss, (nn.CrossEntropyLoss, _FocalCrossEntropyLoss)):
            return False

        weight = getattr(task_loss, 'weight', None)
        if weight is not None:
            # Square the weights: amplifies the imbalance correction
            new_weight = weight ** 2
            # Re-normalize so mean is 1.0
            new_weight = new_weight / new_weight.mean()
            task_loss.register_buffer('weight', new_weight)
            self._class_weight_increase_count += 1
            print(f"  [TREATMENT] Class weights squared (range: "
                  f"{new_weight.min().item():.3f}-{new_weight.max().item():.3f})")
            return True
        return False

    def _apply_balanced_sampler(self) -> bool:
        """Signal trainer to switch to class-balanced batch sampling.

        Sets a pending flag that the trainer reads at the start of the next
        epoch. The trainer then creates a WeightedRandomSampler from the
        training labels, ensuring each class is equally represented in batches.
        This is the most effective treatment for extreme class imbalance.
        """
        if self._balanced_sampler_active:
            return False
        self._pending_balanced_sampler = True
        self._balanced_sampler_active = True
        print(f"  [TREATMENT] Balanced sampler requested (will activate next epoch)")
        return True

    # ===== Level 3: Procedure =====

    def _apply_resnet_block(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add skip connection between layers that lack one."""
        # Find pairs of spatial layers without skip connections
        existing_connections = set()
        for conn in model.connections:
            existing_connections.add((conn.source, conn.target))

        added = False
        for l in range(model.num_layers - 1):
            # Connection semantics: create_connection(model, source, target+1)
            # connects h[source] → adds to h[target] (input of layer target).
            # h[source] = output of layer source-1 (or stem if source==0).
            # h[target] = output of layer target-1.
            conn_source = l
            conn_target = l + 2  # The actual target arg to create_connection
            if (conn_source, conn_target) in existing_connections:
                continue

            # Get spatial shape of h[conn_source] — the actual tensor being read
            if conn_source == 0:
                src_spatial = getattr(model.input_projection, 'spatial_shape', None)
            else:
                src_spatial = getattr(model.layers[conn_source - 1], 'spatial_shape', None)

            # Get spatial shape of h[conn_target - 1] — the tensor being added to
            # conn_target - 1 is the index into h[], which is output of layer conn_target - 2
            tgt_h_idx = conn_target - 1  # index in h dict
            if tgt_h_idx == 0:
                tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif tgt_h_idx - 1 < len(model.layers):
                tgt_spatial = getattr(model.layers[tgt_h_idx - 1], 'spatial_shape', None)
            else:
                continue

            # Guard: skip connections across spatial resolution boundaries
            if (src_spatial is not None and tgt_spatial is not None
                    and (src_spatial[1] != tgt_spatial[1]
                         or src_spatial[2] != tgt_spatial[2])):
                continue  # Different H×W — cannot connect

            self.engine.create_connection(model, conn_source, conn_target, optimizer, step)
            added = True
            break  # One skip connection per treatment

        return added

    def _apply_capacity_growth(self, model: ASANNModel, optimizer, step: int,
                               severity: int = 2) -> bool:
        """Add channels/neurons to layers, scaled by severity.

        Severity-proportional growth:
          - severity 1: +1 neuron to only the narrowest layer
          - severity 2: +2 per layer (half the base growth)
          - severity 3+: full base growth (+4 per layer, original behavior)
        """
        base_growth = getattr(self.config, 'min_channels_add_per_qualifying_layer', 4)
        added = 0

        # Determine growth amount and target layers based on severity
        if severity <= 1:
            channels_per_layer = 1
            # Only grow the narrowest layer
            widths = []
            for l in range(model.num_layers):
                layer = model.layers[l]
                is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
                w = layer.out_channels if is_spatial else layer.out_features
                widths.append((l, w))
            if widths:
                target_layer_indices = {min(widths, key=lambda x: x[1])[0]}
            else:
                target_layer_indices = set()
            print(f"    [CAPACITY] Severity {severity}: growing narrowest layer only (+{channels_per_layer})")
        elif severity <= 2:
            channels_per_layer = max(1, base_growth // 2)
            target_layer_indices = None  # All layers
            print(f"    [CAPACITY] Severity {severity}: moderate growth (+{channels_per_layer}/layer)")
        else:
            channels_per_layer = base_growth
            target_layer_indices = None  # All layers
            print(f"    [CAPACITY] Severity {severity}: full growth (+{channels_per_layer}/layer)")

        # Scale growth by dataset size (small datasets get lighter growth)
        if (getattr(self.config, 'dose_dataset_scaling_enabled', True)
                and self.n_train_samples > 0):
            ref_n = getattr(self.config, 'dose_reference_n_samples', 5000.0)
            min_ds = getattr(self.config, 'dose_dataset_min_factor', 0.10)
            ds_scale = max(min_ds, min(1.0, self.n_train_samples / ref_n))
            scaled = max(1, int(channels_per_layer * ds_scale))
            if scaled != channels_per_layer:
                print(f"    [CAPACITY] Dataset-size scaling: {channels_per_layer} -> {scaled} channels "
                      f"(n={self.n_train_samples}, factor={ds_scale:.3f})")
                channels_per_layer = scaled

        for l in range(model.num_layers):
            # Skip layers not in target set (for severity-targeted growth)
            if target_layer_indices is not None and l not in target_layer_indices:
                continue

            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # For spatial layers, use channel count; for flat, use neuron count
            if is_spatial:
                current_width = layer.out_channels
                max_w = self.config.max_channels
            else:
                current_width = layer.out_features
                # Use max_channels as absolute cap for flat layers too
                max_w = min(current_width + 8, self.config.max_channels)

            # Respect max width cap
            room = max(0, max_w - current_width)
            target_add = min(channels_per_layer, room)

            # Snap to alignment for Tensor Core efficiency (spatial only)
            if is_spatial and self.config.channel_alignment > 1:
                target_add = snap_add_for_alignment(
                    current_width, target_add, self.config.channel_alignment)
                target_add = min(target_add, room)  # Re-cap after snap

            if target_add == 0:
                continue

            layer_added = 0
            for _ in range(target_add):
                success = self.engine.add_neuron(model, l, optimizer, step)
                if success:
                    added += 1
                    layer_added += 1
                else:
                    break
            if layer_added > 0:
                new_width = layer.out_channels if is_spatial else layer.out_features
                print(f"    [CAPACITY] Layer {l}: {current_width}->{new_width} (+{layer_added})")

        if added > 0:
            print(f"    [CAPACITY] Total added: {added} neurons/channels")
        else:
            print(f"    [CAPACITY] WARNING: Could not add any neurons/channels (all at max?)")

        return added > 0

    def _apply_layer_addition(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add a new layer at the position with highest saturation."""
        # Guard: don't add layers if too many are unenriched
        unenriched = model.count_unenriched_layers()
        if unenriched > self.config.max_unenriched_layers:
            print(f"    [LAYER_ADD] Blocked: {unenriched} unenriched layers exist "
                  f"(max={self.config.max_unenriched_layers})")
            return False
        # Use the middle position as a heuristic if no saturation data
        position = model.num_layers // 2
        self.engine.add_layer(model, position, optimizer, step)
        return True

    def _apply_progressive_channels(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add channels to each spatial layer (progressive growing, aligned to 8)."""
        added = 0
        for l in range(model.num_layers):
            layer = model.layers[l]
            if not (hasattr(layer, 'mode') and layer.mode == "spatial"):
                continue
            if layer.out_channels >= self.config.max_channels:
                continue

            n_add = 4
            # Snap to alignment for Tensor Core efficiency
            if self.config.channel_alignment > 1:
                n_add = snap_add_for_alignment(
                    layer.out_channels, n_add, self.config.channel_alignment)
                room = max(0, self.config.max_channels - layer.out_channels)
                n_add = min(n_add, room)

            for _ in range(n_add):
                success = self.engine.add_neuron(model, l, optimizer, step)
                if success:
                    added += 1
                else:
                    break

        return added > 0

    def _apply_widen_layers(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Wide-ResNet style: multiply all spatial layer widths by 1.5x.

        More aggressive than PROGRESSIVE_CHANNELS (+4/layer) but far safer
        than ENCODER_UPGRADE. Uses add_neuron() which dispatches to
        add_channel() for spatial layers (Net2Net filter splitting).

        After all widening, calls register_structural_surgery() to pick up
        new spatial op params created by _update_spatial_ops_dimensions().
        """
        total_added = 0
        for l in range(model.num_layers):
            layer = model.layers[l]
            if not (hasattr(layer, 'mode') and layer.mode == "spatial"):
                continue
            current_c = layer.out_channels
            if current_c >= self.config.max_channels:
                continue

            target_c = min(int(current_c * 1.5), self.config.max_channels)
            n_to_add = target_c - current_c
            # Snap to alignment for Tensor Core efficiency
            if self.config.channel_alignment > 1:
                n_to_add = snap_add_for_alignment(
                    current_c, n_to_add, self.config.channel_alignment)
                n_to_add = min(n_to_add, self.config.max_channels - current_c)
            added = 0
            for _ in range(n_to_add):
                if not self.engine.add_neuron(model, l, optimizer, step):
                    break  # Hit max_channels or ratio constraint
                added += 1
            if added > 0:
                total_added += added
                print(f"  [WIDEN_LAYERS] Layer {l}: {current_c} -> {current_c + added} channels")

        if total_added > 0:
            # Full re-sync: picks up new spatial op params, drops stale old ones
            optimizer.register_structural_surgery(model, surgery_type='widen_layers')

        return total_added > 0

    # ===== Level 3: Capacity Reduction (overfitting) =====

    def _apply_width_reduction(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Shrink all layer widths by ~25% to reduce model capacity.

        The textbook #1 fix for overfitting: reduce model capacity. This
        removes neurons from each layer using SurgeryEngine.remove_neuron(),
        targeting the smallest-magnitude neurons first (least useful).
        """
        surgery_engine = SurgeryEngine(self.config)
        total_removed = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            if is_spatial:
                continue  # Only flat layers for now

            width = layer.out_features
            # Remove 25% of neurons, but keep at least d_min
            n_remove = max(1, width // 4)
            removable = width - self.config.d_min
            n_remove = min(n_remove, max(0, removable))

            if n_remove <= 0:
                continue

            removed_this_layer = 0
            for _ in range(n_remove):
                if model.layers[l].out_features <= self.config.d_min:
                    break
                # Pick neuron with smallest weight magnitude
                w = model.layers[l].weight.data
                neuron_mag = w.abs().sum(dim=1)
                neuron_idx = neuron_mag.argmin().item()
                surgery_engine.remove_neuron(model, l, neuron_idx, optimizer, step)
                removed_this_layer += 1

            if removed_this_layer > 0:
                new_width = model.layers[l].out_features
                print(f"    [WIDTH_REDUCTION] Layer {l}: {width}->{new_width} "
                      f"(-{removed_this_layer} neurons)")
                total_removed += removed_this_layer

        if total_removed > 0:
            print(f"    [WIDTH_REDUCTION] Total: removed {total_removed} neurons")
        return total_removed > 0

    def _apply_layer_removal(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Remove the least useful layer to reduce model depth.

        Picks the layer with the smallest average weight magnitude
        (contributing least to the output) and removes it.
        Requires model.num_layers > 1.
        """
        if model.num_layers <= 1:
            return False

        surgery_engine = SurgeryEngine(self.config)

        # Find the layer with smallest average weight magnitude
        best_layer = None
        best_mag = float('inf')
        for l in range(model.num_layers):
            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            if is_spatial:
                continue
            w = layer.weight.data
            mag = w.abs().mean().item()
            if mag < best_mag:
                best_mag = mag
                best_layer = l

        if best_layer is None:
            return False

        old_width = model.layers[best_layer].out_features
        old_num_layers = model.num_layers
        surgery_engine.remove_layer(model, best_layer, optimizer, step)

        print(f"    [LAYER_REMOVAL] Removed layer {best_layer} "
              f"(width={old_width}, avg_mag={best_mag:.6f}), "
              f"layers: {old_num_layers}->{model.num_layers}")
        return True

    def _apply_encoder_downgrade(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Switch to a simpler encoder (LinearEncoder) to reduce complexity.

        This is the reverse of ENCODER_UPGRADE: when the model is overfitting
        and using a complex encoder (transformer, molecular_graph, etc.),
        downgrading to LinearEncoder reduces capacity.

        Uses the same gated bridge mechanism as ENCODER_UPGRADE for smooth
        transition.
        """
        from .encoders import create_encoder, build_encoder_kwargs, GatedEncoderBridge

        # Don't switch while a bridge is active
        if isinstance(model.encoder, GatedEncoderBridge):
            return False

        current_type = getattr(model.encoder, 'encoder_type', 'linear')
        if current_type == 'linear':
            return False  # Already at simplest

        # Downgrade to LinearEncoder
        target_type = 'linear'

        d_enc = model.config.encoder_d_output or model._effective_d_init

        kwargs = build_encoder_kwargs(
            target_type,
            d_input=model.d_input,
            d_output=d_enc,
            config=model.config,
            model=model,
        )

        try:
            new_encoder = create_encoder(target_type, **kwargs)
            new_encoder = new_encoder.to(model.config.device)

            # Use gated bridge for gradual transition
            model.encoder_switch(new_encoder,
                                 warmup_epochs=model.config.encoder_switch_warmup_epochs)

            if optimizer is not None and hasattr(optimizer, 'register_structural_surgery'):
                optimizer.register_structural_surgery(model, surgery_type='encoder_downgrade')

            print(f"  [TREATMENT] ENCODER_DOWNGRADE: {current_type} -> {target_type} "
                  f"(d_enc={d_enc}, warmup={model.config.encoder_switch_warmup_epochs})")
            return True

        except Exception as e:
            print(f"  [TREATMENT] ENCODER_DOWNGRADE failed: {e}")
            return False

    # ===== Level 4: Chemotherapy =====

    def _compute_dose_factor(self, model: ASANNModel) -> float:
        """Compute capacity-based dose scaling factor.

        Small model (few layers, narrow) -> light dose (kid).
        Large model (many layers, wide) -> full dose (adult).
        Additionally scales by dataset size: small datasets get lighter doses.
        """
        # --- Capacity scaling (existing) ---
        widths = []
        for l in range(model.num_layers):
            layer = model.layers[l]
            w = getattr(layer, 'out_channels', None) or layer.out_features
            widths.append(w)
        median_width = sorted(widths)[len(widths) // 2] if widths else 64
        capacity = model.num_layers * median_width
        ref = getattr(self.config, 'dose_reference_capacity', 1024.0)
        min_factor = getattr(self.config, 'dose_min_factor', 0.15)
        capacity_factor = max(min_factor, min(1.0, capacity / ref))

        # --- Dataset-size scaling (new) ---
        dataset_factor = 1.0
        if (getattr(self.config, 'dose_dataset_scaling_enabled', True)
                and self.n_train_samples > 0):
            ref_n = getattr(self.config, 'dose_reference_n_samples', 5000.0)
            min_ds = getattr(self.config, 'dose_dataset_min_factor', 0.10)
            dataset_factor = max(min_ds, min(1.0, self.n_train_samples / ref_n))

        return capacity_factor * dataset_factor

    def _apply_aggressive_regularization(
        self, model: ASANNModel, optimizer, step: int,
        dose_level: float = 1.0,
    ) -> bool:
        """Dose-adaptive strong regularization: dropout + WD boost + LR reduce.

        Dose scales with model capacity (small model → light dose) and can be
        titrated down via dose_level (1.0 = full, 0.5 = half, etc.).
        """
        if getattr(self.config, 'dose_adaptive_enabled', True):
            dose_factor = self._compute_dose_factor(model)
        else:
            dose_factor = 1.0

        effective_dose = dose_factor * dose_level

        # Scale parameters by effective dose (interpolate toward neutral)
        base_dropout = getattr(self.config, 'aggressive_reg_dropout_p', 0.3)
        base_wd = getattr(self.config, 'aggressive_reg_wd_factor', 3.0)
        base_lr = getattr(self.config, 'aggressive_reg_lr_factor', 0.5)

        dropout_p = base_dropout * effective_dose
        wd_factor = 1.0 + (base_wd - 1.0) * effective_dose  # Interpolate toward 1.0
        lr_factor = 1.0 - (1.0 - base_lr) * effective_dose  # Interpolate toward 1.0

        print(f"  [DOSE] Aggressive reg: dose_factor={dose_factor:.3f}, "
              f"dose_level={dose_level:.3f}, effective={effective_dose:.3f}")
        print(f"  [DOSE] dropout_p={dropout_p:.4f}, wd_factor={wd_factor:.3f}, "
              f"lr_factor={lr_factor:.3f}")

        success = False
        success |= self._apply_dropout(model, optimizer, step, p=dropout_p)
        success |= self._apply_weight_decay_boost(optimizer, factor=wd_factor)
        success |= self._apply_lr_reduce(optimizer, factor=lr_factor)

        # Store dose info for titration
        self._last_dose_factor = dose_factor
        self._last_dose_level = dose_level
        return success

    def adjust_aggressive_reg_dose(
        self, model: ASANNModel, optimizer, new_dose_level: float,
        dose_factor: float,
    ) -> None:
        """Lower regularization in-place (no rollback to overfitting state).

        Adjusts dropout probability and weight decay on the live model.
        LR is left for the LR controller to manage naturally.
        """
        effective = dose_factor * new_dose_level

        # Adjust dropout.p on all existing dropout layers
        base_dropout = getattr(self.config, 'aggressive_reg_dropout_p', 0.3)
        new_p = base_dropout * effective
        import torch.nn as nn
        for l in range(model.num_layers):
            if l < len(model.ops):
                for op in model.ops[l].operations:
                    if isinstance(op, nn.Dropout):
                        op.p = new_p

        # Adjust weight decay in optimizer
        base_wd = getattr(self.config, 'aggressive_reg_wd_factor', 3.0)
        base_weight_decay = self.config.optimizer.weight_decay
        new_wd = base_weight_decay * (1.0 + (base_wd - 1.0) * effective)
        for group in optimizer.param_groups:
            if 'weight_decay' in group:
                group['weight_decay'] = new_wd

        print(f"  [TITRATION] Dose lowered: level={new_dose_level:.3f}, "
              f"effective={effective:.3f}, dropout_p={new_p:.4f}, wd={new_wd:.6f}")

    def _apply_architecture_reset_soft(
        self, model: ASANNModel, optimizer, step: int
    ) -> bool:
        """Remove all non-essential ops, keep only BatchNorm + ReLU.

        This is a soft reset — preserves the layer structure and weights,
        but strips away complex ops that might be causing overfitting.
        """
        device = self.config.device
        reset_count = 0

        for l in range(model.num_layers):
            ops = model.ops[l]
            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # Identify non-essential ops
            to_keep = []
            for i, op in enumerate(ops.operations):
                op_name = type(op).__name__
                # Keep: ReLU, BatchNorm, Dropout
                if isinstance(op, (nn.ReLU, nn.GELU, nn.SiLU,
                                   nn.BatchNorm1d, nn.BatchNorm2d,
                                   nn.Dropout)):
                    to_keep.append(op)
                # Remove everything else (convs, attention, embeddings, etc.)
                else:
                    reset_count += 1

            if reset_count > 0:
                # Rebuild operations list
                ops.operations = nn.ModuleList(to_keep)

        if reset_count > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_reset_soft')
        return reset_count > 0

    def _apply_dual_depth_width(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add layer AND channels simultaneously (compound scaling)."""
        success = False

        # Add layer at the middle
        position = model.num_layers // 2
        self.engine.add_layer(model, position, optimizer, step)
        success = True

        # Add channels to each layer (aligned to 8 for Tensor Core efficiency)
        for l in range(model.num_layers):
            layer = model.layers[l]
            if hasattr(layer, 'mode') and layer.mode == "spatial":
                if layer.out_channels >= self.config.max_channels:
                    continue
                n_add = 4
                if self.config.channel_alignment > 1:
                    n_add = snap_add_for_alignment(
                        layer.out_channels, n_add, self.config.channel_alignment)
                    room = max(0, self.config.max_channels - layer.out_channels)
                    n_add = min(n_add, room)
                for _ in range(n_add):
                    self.engine.add_neuron(model, l, optimizer, step)

        return success

    # ===== Level 3: Encoder Upgrade =====

    def _apply_encoder_upgrade(
        self, model: ASANNModel, optimizer, step: int
    ) -> bool:
        """Switch to a more powerful input encoder.

        Picks the next untried candidate from config.encoder_candidates,
        creates the encoder, and initiates a gated bridge transition.
        The bridge blends old and new encoder output over warmup_epochs.

        This is how ASANN discovers that e.g. Fourier features help PDE tasks,
        or that GNN pre-encoding helps molecular tasks — through the treatment
        system, not by pre-baking the choice.
        """
        from .encoders import (create_encoder, build_encoder_kwargs,
                               GatedEncoderBridge)

        candidates = model.config.encoder_candidates
        if candidates is None:
            return False

        # Get current encoder type
        current_type = getattr(model.encoder, 'encoder_type', 'linear')

        # Mark current encoder as tried
        self._tried_encoders.add(current_type)

        # Find next untried candidate
        untried = [c for c in candidates
                   if c != current_type and c not in self._tried_encoders]
        if not untried:
            return False

        next_type = untried[0]
        self._tried_encoders.add(next_type)

        # Determine encoder output dimension
        d_enc = model.config.encoder_d_output or model._effective_d_init

        # Build kwargs for the new encoder
        kwargs = build_encoder_kwargs(
            next_type,
            d_input=model.d_input,
            d_output=d_enc,
            config=model.config,
            model=model,
        )

        try:
            new_encoder = create_encoder(next_type, **kwargs)
            new_encoder = new_encoder.to(model.config.device)

            # Check spatial compatibility: switching from spatial to flat encoder
            # is problematic because the model body (Conv2d layers) expects [B, C, H, W].
            # The bridge can handle the blending via flatten + reshape, but log a warning.
            if model._is_spatial and not new_encoder.is_spatial:
                print(f"  [TREATMENT] ENCODER_UPGRADE: {current_type} -> {next_type} "
                      f"(spatial -> flat bridge, output will be reshaped)")
            elif not model._is_spatial and new_encoder.is_spatial:
                print(f"  [TREATMENT] ENCODER_UPGRADE: {current_type} -> {next_type} "
                      f"(flat -> spatial bridge, output will be flattened)")

            # Initiate gated bridge transition
            model.encoder_switch(new_encoder,
                                 warmup_epochs=model.config.encoder_switch_warmup_epochs)

            # Register new encoder parameters with optimizer
            if optimizer is not None and hasattr(optimizer, 'register_structural_surgery'):
                optimizer.register_structural_surgery(model, surgery_type='encoder_upgrade')

            print(f"  [TREATMENT] ENCODER_UPGRADE: {current_type} -> {next_type} "
                  f"(d_enc={d_enc}, warmup={model.config.encoder_switch_warmup_epochs})")
            return True

        except Exception as e:
            print(f"  [TREATMENT] ENCODER_UPGRADE failed: {e}")
            return False

    # ===== Level 5: Emergency — Model Collapsed =====

    def _apply_weight_reinitialize(
        self, model: ASANNModel, optimizer, step: int
    ) -> bool:
        """Emergency treatment: reinitialize all weights from the current architecture.

        When the model has collapsed (predicting the mean), growing capacity or
        adding regularization is futile. The only recovery is to restart
        optimization from the current architecture.

        This treatment:
        1. Reinitializes all layer weights using Kaiming/Xavier
        2. Resets dropout to 0.0 (removes all dropout layers)
        3. Restores weight decay to original config value
        4. Resets LR to initial value across all param groups
        5. Keeps the architecture intact (widths, layers, operations)
        """
        print(f"    [WEIGHT_REINITIALIZE] Model collapsed detected - reinitializing weights")

        # 1. Reinitialize layer weights
        for l in range(model.num_layers):
            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            if is_spatial:
                # Spatial layer: reinitialize conv weights
                if hasattr(layer, 'conv') and hasattr(layer.conv, 'weight'):
                    nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                    if layer.conv.bias is not None:
                        nn.init.zeros_(layer.conv.bias)
            else:
                # Flat layer: reinitialize linear weights
                if hasattr(layer, 'linear'):
                    nn.init.kaiming_normal_(layer.linear.weight, nonlinearity='relu')
                    if layer.linear.bias is not None:
                        nn.init.zeros_(layer.linear.bias)
                elif hasattr(layer, 'weight'):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            # Reinitialize operations in this layer's pipeline
            ops = model.ops[l]
            for op in ops.operations:
                # Unwrap GatedOperation for type checks
                inner = op.operation if isinstance(op, GatedOperation) else op
                if isinstance(inner, nn.Dropout):
                    continue  # Will be handled below
                if isinstance(inner, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    inner.reset_parameters()
                elif hasattr(inner, 'gate_logit'):
                    # Gated residual ops: reset gate to near-identity
                    inner.gate_logit.data.fill_(-2.0)
                # Reinitialize any learnable weights in the op
                for name, param in op.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        # Reinitialize input projection
        if hasattr(model.input_projection, 'weight'):
            nn.init.kaiming_normal_(model.input_projection.weight, nonlinearity='relu')
            if model.input_projection.bias is not None:
                nn.init.zeros_(model.input_projection.bias)
        elif hasattr(model.input_projection, 'linear'):
            nn.init.kaiming_normal_(model.input_projection.linear.weight, nonlinearity='relu')
            if model.input_projection.linear.bias is not None:
                nn.init.zeros_(model.input_projection.linear.bias)

        # Reinitialize output head
        if hasattr(model, 'output_head') and hasattr(model.output_head, 'weight'):
            nn.init.xavier_normal_(model.output_head.weight)
            if model.output_head.bias is not None:
                nn.init.zeros_(model.output_head.bias)

        # 2. Remove all dropout layers
        for l in range(model.num_layers):
            ops = model.ops[l]
            to_keep = [op for op in ops.operations if not isinstance(op, nn.Dropout)]
            if len(to_keep) != len(list(ops.operations)):
                ops.operations = nn.ModuleList(to_keep)
                print(f"    [WEIGHT_REINITIALIZE] Layer {l}: removed dropout")

        # 3. Restore weight decay to original value
        original_wd = self._original_weight_decay or self.config.optimizer.weight_decay
        self.config.optimizer.weight_decay = original_wd
        for group in optimizer.param_groups:
            if group.get('weight_decay', 0) > 0:
                group['weight_decay'] = original_wd
        self._original_weight_decay = None  # Reset tracking

        # 4. Reset LR to initial value
        initial_lr = self.config.optimizer.base_lr
        for group in optimizer.param_groups:
            group['lr'] = initial_lr
        print(f"    [WEIGHT_REINITIALIZE] LR reset to {initial_lr}, "
              f"WD reset to {original_wd}")

        # 5. Reset counters — give the model a fresh start
        self._tried_treatments.clear()
        self.disease_escalation.clear()
        self._label_smoothing_active = False

        # Register the structural change with optimizer
        optimizer.register_structural_surgery(model, surgery_type='treatment_weight_reinitialize')

        print(f"    [WEIGHT_REINITIALIZE] All weights reinitialized. Architecture preserved: "
              f"{model.num_layers} layers, {sum(p.numel() for p in model.parameters())} params")
        return True

    # ===== Level 2: PDE Discovery Operations =====

    def _apply_derivative_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add derivative ops to layers that lack them.

        Inserts DerivativeConv2d (spatial) or DerivativeConv1d (flat) after the
        activation function in each layer's pipeline.
        """
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            ops = model.ops[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # Check if layer already has derivative ops
            has_deriv = any(
                'derivative' in get_operation_name(op).lower()
                for op in ops.operations
            )
            if has_deriv:
                continue

            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            try:
                if is_spatial:
                    sp = layer.spatial_shape
                    new_op = create_operation(
                        "derivative_conv2d_laplacian", d=sp[0],
                        device=device, config=self.config,
                        spatial_shape=sp,
                    )
                else:
                    d = layer.out_features
                    new_op = create_operation(
                        "derivative_conv1d_o1", d=d,
                        device=device, config=self.config,
                    )
                new_op._asann_added_step = step

                # Insert after activation
                insert_pos = ops.num_operations
                for i, op in enumerate(ops.operations):
                    if isinstance(op, (nn.ReLU, nn.GELU, nn.SiLU)):
                        insert_pos = i + 1
                        break
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(new_op, insert_pos, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception:
                continue

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_derivative')
        return added > 0

    def _apply_polynomial_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add polynomial interaction ops to layers that lack them.

        Inserts SpatialPolynomialOp (spatial) or PolynomialOp (flat) to discover
        nonlinear terms like u², uv², etc.
        """
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            ops = model.ops[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # Check if layer already has polynomial ops
            has_poly = any(
                'polynomial' in get_operation_name(op).lower()
                for op in ops.operations
            )
            if has_poly:
                continue

            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            try:
                if is_spatial:
                    sp = layer.spatial_shape
                    new_op = create_operation(
                        "spatial_polynomial_deg2", d=sp[0],
                        device=device, config=self.config,
                        spatial_shape=sp,
                    )
                else:
                    d = layer.out_features
                    new_op = create_operation(
                        "polynomial_deg2", d=d,
                        device=device, config=self.config,
                    )
                new_op._asann_added_step = step

                # Insert at end of pipeline
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(new_op, ops.num_operations, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception:
                continue

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_polynomial')
        return added > 0

    def _apply_branched_diffusion_reaction(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add branched diffusion-reaction block to layers lacking both derivative and polynomial ops.

        Inserts a single BranchedOperationBlock (flat) or SpatialBranchedOperationBlock (spatial)
        that combines derivative (diffusion) and polynomial (reaction) branches.
        Skips layers that already have derivative, polynomial, or branched ops.
        """
        device = self.config.device
        added = 0

        for l in range(model.num_layers):
            layer = model.layers[l]
            ops = model.ops[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # Skip if already has derivative, polynomial, or branched ops
            has_relevant = any(
                any(kw in get_operation_name(op).lower()
                    for kw in ('derivative', 'polynomial', 'branched'))
                for op in ops.operations
            )
            if has_relevant:
                continue

            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            try:
                if is_spatial:
                    sp = layer.spatial_shape
                    new_op = create_operation(
                        "spatial_branched_diff_react", d=sp[0],
                        device=device, config=self.config,
                        spatial_shape=sp,
                    )
                else:
                    d = layer.out_features
                    new_op = create_operation(
                        "branched_deriv_poly", d=d,
                        device=device, config=self.config,
                    )
                new_op._asann_added_step = step

                # Insert after activation
                insert_pos = ops.num_operations
                for i, op in enumerate(ops.operations):
                    if isinstance(op, (nn.ReLU, nn.GELU, nn.SiLU)):
                        insert_pos = i + 1
                        break
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(new_op, insert_pos, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception:
                continue

        if added > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_branched')
        return added > 0

    # ===== Level 2: Graph Discovery Operations =====

    def _apply_graph_conv_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add NeighborAggregation to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_neighbor_agg")

    def _apply_graph_attention_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add GraphAttentionAggregation to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_attention_agg")

    def _apply_graph_diffusion_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add GraphDiffusion to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_diffusion_k3")

    def _apply_graph_branched_agg(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add GraphBranchedBlock to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_branched_agg")

    def _apply_graph_spectral_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add SpectralConv (ChebNet) to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_spectral_conv")

    def _apply_graph_gin_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add MessagePassingGIN to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_gin")

    def _apply_graph_message_boost_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add MessageBooster (sum*max, CMPNN-inspired) to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_message_boost")

    def _apply_gnn_type_switch(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Switch molecular GNN encoder type (e.g., gat -> gine).

        Creates a new MolecularGraphEncoder with GINE type (GIN + edge features)
        and initiates a gated bridge transition. The bridge blends old and new
        encoder output over warmup_epochs to prevent catastrophic representation
        change.

        The switch order is: current_type -> gine (uses bond features additively).
        If already gine, no switch is possible.
        """
        from .encoders import (MolecularGraphEncoder, GatedEncoderBridge,
                               build_encoder_kwargs, create_encoder)

        encoder = model.encoder
        if isinstance(encoder, GatedEncoderBridge):
            return False  # Already mid-switch
        if not isinstance(encoder, MolecularGraphEncoder):
            return False

        old_type = encoder._gnn_type
        # Determine target GNN type
        if old_type in ("gat", "gcn", "gin"):
            new_type = "gine"
        else:
            return False  # Already gine or unknown

        # Determine encoder output dimension
        d_enc = model.config.encoder_d_output or model._effective_d_init

        # Build kwargs with gine type
        # Temporarily override config
        orig_gnn_type = model.config.encoder_mol_gnn_type
        model.config.encoder_mol_gnn_type = new_type
        try:
            kwargs = build_encoder_kwargs(
                "molecular_graph",
                d_input=model.d_input,
                d_output=d_enc,
                config=model.config,
                model=model,
            )

            new_encoder = create_encoder("molecular_graph", **kwargs)
            new_encoder = new_encoder.to(model.config.device)

            # Transfer molecular graph data to new encoder
            if encoder._all_graphs is not None:
                new_encoder.set_molecular_graphs(encoder._all_graphs)
            if encoder._molecular_batch is not None:
                new_encoder.set_molecular_batch(encoder._molecular_batch)

            # Initiate gated bridge transition
            model.encoder_switch(new_encoder,
                                 warmup_epochs=model.config.encoder_switch_warmup_epochs)

            # Register new encoder parameters with optimizer
            if optimizer is not None and hasattr(optimizer, 'register_structural_surgery'):
                optimizer.register_structural_surgery(model, surgery_type='gnn_type_switch')

            print(f"  [TREATMENT] GNN_TYPE_SWITCH: {old_type} -> {new_type} "
                  f"(d_enc={d_enc}, bond_dim={kwargs.get('bond_feature_dim', 0)}, "
                  f"warmup={model.config.encoder_switch_warmup_epochs})")
            return True

        except Exception as e:
            print(f"  [TREATMENT] GNN_TYPE_SWITCH failed: {e}")
            return False
        finally:
            # Restore original config
            model.config.encoder_mol_gnn_type = orig_gnn_type

    def _apply_graph_positional_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add GraphPositionalEncoding to layers that lack graph ops."""
        return self._apply_graph_op_package(model, optimizer, step, "graph_positional_enc")

    def _apply_graph_norm_package(self, model: ASANNModel, optimizer, step: int) -> bool:
        """Add PairNorm + GraphNorm to layers for anti-oversmoothing.

        Unlike other graph packages, this adds normalization layers (no gate)
        to ALL layers (even those that already have graph ops, since norms
        complement aggregation ops).
        """
        device = self.config.device
        added = 0

        graph_data = {
            'adj_sparse': model._graph_adj_sparse,
            'edge_index': model._graph_edge_index,
            'degree': model._graph_degree,
            'num_nodes': model._graph_num_nodes,
        }

        for l in range(model.num_layers):
            layer = model.layers[l]
            ops = model.ops[l]

            # Skip spatial layers
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            if is_spatial:
                continue

            # Check if layer already has graph norms
            has_pairnorm = any(
                get_operation_name(op) == 'graph_pairnorm'
                for op in ops.operations
            )
            if has_pairnorm:
                continue

            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            try:
                d = layer.out_features
                # Add PairNorm at end of pipeline (after graph aggregation ops)
                pairnorm = create_operation(
                    "graph_pairnorm", d=d, device=device, config=self.config,
                    graph_data=graph_data,
                )
                pairnorm._asann_added_step = step
                warmup = self.config.surgery_warmup_epochs
                ops.add_operation(pairnorm, ops.num_operations, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception as e:
                print(f"    [GRAPH_NORM] Failed to add PairNorm to layer {l}: {e}")
                continue

        if added > 0:
            optimizer.register_structural_surgery(
                model, surgery_type='treatment_graph_norm'
            )
            print(f"    [GRAPH_NORM] Added PairNorm to {added}/{model.num_layers} layers")
        return added > 0

    def _apply_graph_op_package(
        self, model: ASANNModel, optimizer, step: int, op_name: str
    ) -> bool:
        """Shared implementation for all graph operation packages.

        Follows the same pattern as _apply_derivative_package():
        - Iterates layers
        - Skips layers that already have graph ops
        - Creates op via create_operation() with graph_data kwarg
        - Inserts after activation
        - Registers structural surgery with optimizer for param registration
        """
        device = self.config.device
        added = 0

        # Collect graph auxiliary data from model
        graph_data = {
            'adj_sparse': model._graph_adj_sparse,
            'edge_index': model._graph_edge_index,
            'degree': model._graph_degree,
            'num_nodes': model._graph_num_nodes,
        }

        for l in range(model.num_layers):
            layer = model.layers[l]
            ops = model.ops[l]

            # Skip spatial layers (graph ops work on flat [N, d] only)
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"
            if is_spatial:
                continue

            # Check if layer already has graph ops
            has_graph = any(
                'graph_' in get_operation_name(op).lower()
                for op in ops.operations
            )
            if has_graph:
                continue

            if ops.num_operations >= self.config.max_ops_per_layer:
                continue

            try:
                d = layer.out_features
                new_op = create_operation(
                    op_name, d=d, device=device, config=self.config,
                    graph_data=graph_data,
                )
                new_op._asann_added_step = step

                # Insert after activation (same pattern as derivative package)
                insert_pos = ops.num_operations
                warmup = self.config.surgery_warmup_epochs
                for i, op in enumerate(ops.operations):
                    if isinstance(op, (nn.ReLU, nn.GELU, nn.SiLU)):
                        insert_pos = i + 1
                        break
                ops.add_operation(new_op, insert_pos, gated=True, warmup_epochs=warmup)
                added += 1
            except Exception as e:
                print(f"    [GRAPH_PACKAGE] Failed to add {op_name} to layer {l}: {e}")
                continue

        if added > 0:
            # Register with optimizer so new params get proper param groups + LR warmup
            optimizer.register_structural_surgery(
                model, surgery_type=f'treatment_{op_name}'
            )
            print(f"    [GRAPH_PACKAGE] Added {op_name} to {added}/{model.num_layers} layers")
        return added > 0

    # ===== Level 2.5: Operation-Aware Swap =====

    def _apply_operation_swap(
        self, model: ASANNModel, optimizer, step: int, treatment: Treatment
    ) -> bool:
        """Replace the least beneficial operation with the most beneficial one.

        Uses lab findings (OperationSensitivityTest) when available to identify
        which operation to swap and what to replace it with. Falls back to
        inserting a new operation if no swappable op is found.

        Protected operations (ReLU, BatchNorm, Dropout) cannot be swapped out.
        """
        device = self.config.device
        lab_findings = treatment.details.get("lab_findings", {})
        target_layers = treatment.target_layers

        # Protected ops that should never be swapped out
        _PROTECTED_TYPES = (nn.ReLU, nn.GELU, nn.SiLU,
                            nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.Dropout)
        # For graph models, graph aggregation ops are also protected —
        # swapping out NeighborAgg destroys the GCN structure
        is_graph = getattr(model, '_is_graph', False)
        if is_graph:
            from .surgery import (NeighborAggregation, GraphAttentionAggregation,
                GraphDiffusion, SpectralConv, MessagePassingGIN, DegreeScaling,
                APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                DirectionalDiffusion, AdaptiveGraphConv)
            _PROTECTED_TYPES = _PROTECTED_TYPES + (
                NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
                SpectralConv, MessagePassingGIN, DegreeScaling,
                APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                DirectionalDiffusion, AdaptiveGraphConv,
            )

        best_operation = lab_findings.get("best_operation", None)
        best_layer = lab_findings.get("best_layer", None)
        operation_ranking = lab_findings.get("operation_ranking", [])

        # Determine which layers to process
        if target_layers is not None:
            layers_to_process = [l for l in target_layers if l < model.num_layers]
        elif best_layer is not None and best_layer < model.num_layers:
            layers_to_process = [best_layer]
        else:
            # Default: try the layer with most operations (most to swap from)
            layer_op_counts = [(l, model.ops[l].num_operations) for l in range(model.num_layers)]
            layer_op_counts.sort(key=lambda x: x[1], reverse=True)
            layers_to_process = [l for l, c in layer_op_counts[:2] if c > 1]

        if not layers_to_process:
            print(f"    [OP_SWAP] No eligible layers for operation swap")
            return False

        swapped = 0
        for l in layers_to_process:
            layer = model.layers[l]
            ops = model.ops[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            # Find the least beneficial non-protected operation
            swap_idx = None
            for i, op in enumerate(ops.operations):
                # Unwrap GatedOperation for type check
                inner = op.operation if isinstance(op, GatedOperation) else op
                if isinstance(inner, _PROTECTED_TYPES):
                    continue
                # Prefer swapping operations that are NOT in the top-ranked from lab
                op_name = get_operation_name(op).lower()
                if swap_idx is None:
                    swap_idx = i  # First non-protected op is default swap candidate
                # If we have ranking info, prefer to swap the lowest-ranked op
                if operation_ranking:
                    # Check if this op is lower-ranked than current candidate
                    pass  # Keep the first non-protected as default

            if swap_idx is None:
                continue  # All ops are protected

            # Determine what to swap in
            swap_in_name = best_operation
            if swap_in_name is None:
                # No lab findings — pick a standard complementary op
                physics_enabled = getattr(self.config, 'physics_ops_enabled', False)
                has_dw_sep = any('dw_sep' in get_operation_name(op).lower()
                                for op in ops.operations)
                has_pw = any('pointwise' in get_operation_name(op).lower() or 'pw1x1' in get_operation_name(op).lower()
                             for op in ops.operations)
                has_chan_attn = any('channel_attention' in get_operation_name(op).lower()
                                   for op in ops.operations)
                if is_spatial:
                    if not has_dw_sep:
                        swap_in_name = "spatial_dw_sep_k3"
                    elif not has_pw:
                        swap_in_name = "spatial_pointwise_1x1"
                    elif not has_chan_attn:
                        swap_in_name = "channel_attention"
                    elif physics_enabled:
                        swap_in_name = "derivative_conv2d_laplacian"
                    else:
                        swap_in_name = "groupnorm"
                else:
                    has_deriv = any('derivative' in get_operation_name(op).lower()
                                    for op in ops.operations)
                    if not has_deriv and physics_enabled:
                        swap_in_name = "derivative_conv1d_o1"
                    else:
                        swap_in_name = "gated_linear_unit"

            # Create the new operation
            try:
                if is_spatial:
                    sp = layer.spatial_shape
                    new_op = create_operation(
                        swap_in_name, d=sp[0],
                        device=device, config=self.config,
                        spatial_shape=sp,
                    )
                else:
                    d = layer.out_features
                    # For graph models, pass graph_data so new graph ops get adjacency
                    graph_kwargs = {}
                    if is_graph:
                        graph_kwargs['graph_data'] = {
                            'adj_sparse': model._graph_adj_sparse,
                            'edge_index': model._graph_edge_index,
                            'degree': model._graph_degree,
                            'num_nodes': model._graph_num_nodes,
                        }
                    new_op = create_operation(
                        swap_in_name, d=d,
                        device=device, config=self.config,
                        **graph_kwargs,
                    )
                new_op._asann_added_step = step
            except Exception as e:
                print(f"    [OP_SWAP] Could not create {swap_in_name} for layer {l}: {e}")
                continue

            # Perform the swap: remove old, insert new at same position
            old_op = ops.operations[swap_idx]
            old_name = get_operation_name(old_op)
            ops.remove_operation(swap_idx)
            warmup = self.config.surgery_warmup_epochs
            ops.add_operation(new_op, min(swap_idx, ops.num_operations),
                              gated=True, warmup_epochs=warmup)
            swapped += 1
            print(f"    [OP_SWAP] Layer {l}: {old_name} -> {swap_in_name}")

        if swapped > 0:
            optimizer.register_structural_surgery(model, surgery_type='treatment_op_swap')
        else:
            print(f"    [OP_SWAP] No operations were swapped")

        return swapped > 0

    # ==================== State Management ====================

    def state_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoint."""
        return {
            "disease_escalation": dict(self.disease_escalation),
            "_tried_treatments": {
                k.name: [v.name for v in vals]
                for k, vals in self._tried_treatments.items()
            },
            "_original_weight_decay": self._original_weight_decay,
            "_label_smoothing_active": self._label_smoothing_active,
            "_focal_loss_active": self._focal_loss_active,
            "_class_weight_increase_count": self._class_weight_increase_count,
            "_balanced_sampler_active": self._balanced_sampler_active,
            "_tried_encoders": list(self._tried_encoders),
            "_stackable_treatment_count": {
                k.name: v for k, v in self._stackable_treatment_count.items()
            },
            "treatment_history": [
                {
                    "treatment_type": r.treatment.treatment_type.name,
                    "target_disease": r.treatment.target_disease.name,
                    "level": r.treatment.level,
                    "epoch_applied": r.epoch_applied,
                    "pre_val_metric": r.pre_val_metric,
                    "post_val_metric": r.post_val_metric,
                    "outcome": r.outcome,
                }
                for r in self.treatment_history
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore from checkpoint."""
        # Restore escalation counts
        self.disease_escalation = defaultdict(int)
        for k, v in state.get("disease_escalation", {}).items():
            if isinstance(k, str):
                self.disease_escalation[DiseaseType[k]] = v
            else:
                self.disease_escalation[k] = v

        # Restore tried treatments
        self._tried_treatments = defaultdict(set)
        for k, vals in state.get("_tried_treatments", {}).items():
            disease = DiseaseType[k] if isinstance(k, str) else k
            self._tried_treatments[disease] = {
                TreatmentType[v] for v in vals
            }

        self._original_weight_decay = state.get("_original_weight_decay")
        self._label_smoothing_active = state.get("_label_smoothing_active", False)
        self._focal_loss_active = state.get("_focal_loss_active", False)
        self._class_weight_increase_count = state.get("_class_weight_increase_count", 0)
        self._balanced_sampler_active = state.get("_balanced_sampler_active", False)

        # Restore tried encoders
        self._tried_encoders = set(state.get("_tried_encoders", []))

        # Restore stackable treatment counts
        self._stackable_treatment_count = defaultdict(int)
        for k, v in state.get("_stackable_treatment_count", {}).items():
            try:
                self._stackable_treatment_count[TreatmentType[k]] = v
            except KeyError:
                pass  # Skip unknown treatment types from older checkpoints
