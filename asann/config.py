import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class SurgeryOptimizerConfig:
    """DEPRECATED: Configuration for the old SurgeryAwareOptimizer.

    Kept for backward compatibility with existing tier-1/tier-2 experiment configs.
    New code should use ASANNOptimizerConfig instead.
    """

    # --- Base learning rates per group ---
    lr_mature_weights: float = 1e-3       # Base LR for trained 2D weights
    lr_mature_1d: float = 1e-3            # Base LR for biases, norms, scales
    lr_connection: float = 2e-3           # Connections start near-zero, need higher LR
    lr_operation: float = 1.5e-3          # Operation params (conv kernels, embeddings, attn)

    # --- AdamW hyperparameters ---
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # --- Newborn parameter handling ---
    newborn_lr_multiplier: float = 3.0    # Newborn params get higher LR initially
    newborn_warmup_steps: int = 100       # Linear warmup steps for newborn params
    newborn_graduation_steps: int = 200   # Steps until newborn -> mature

    # --- Complexity-aware LR modulation (AMC-inspired) ---
    complexity_lr_enabled: bool = True
    complexity_lr_sensitivity: float = 0.5  # How much complexity changes affect LR
    complexity_ema_decay: float = 0.99      # EMA smoothing for complexity tracking

    # --- Per-neuron normalization (NorMuon-inspired) ---
    neuron_norm_enabled: bool = True
    neuron_norm_eps: float = 1e-6

    # --- Variance transfer for splits ---
    variance_transfer_enabled: bool = True

    # --- Gradient clipping ---
    max_grad_norm: float = 1.0

    # --- WSD-style phase tracking ---
    warmup_steps: int = 500               # Global warmup at training start
    stable_lr_fraction: float = 1.0       # During stable phase, use full LR


@dataclass
class ASANNOptimizerConfig:
    """Configuration for the ASANN Nested Learning Optimizer.

    Replaces SurgeryOptimizerConfig with:
    - Multi-scale momentum (M3-style, 3 time scales + second moment)
    - Per-module parameter groups with different LR scales and update frequencies
    - Hypergradient-based learnable LR controller
    - Warmup + cosine annealing scheduler with post-surgery re-warmup
    - Surgery-aware state management (register/unregister params)
    """

    # --- Base learning rate ---
    base_lr: float = 1e-3

    # --- Multi-scale momentum betas (fast, medium, slow, second_moment) ---
    # For tabular models, use (0.9, 0.9, 0.9, 0.999) to match standard Adam behavior.
    # For spatial models, consider (0.9, 0.95, 0.99, 0.999) for multi-scale smoothing.
    betas: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- Per-neuron normalization (NorMuon-inspired, kept from old optimizer) ---
    neuron_norm_enabled: bool = True
    neuron_norm_eps: float = 1e-6

    # --- Variance transfer for splits (kept from old optimizer) ---
    variance_transfer_enabled: bool = True

    # --- Newborn parameter handling ---
    newborn_warmup_steps: int = 100
    newborn_graduation_steps: int = 200
    newborn_lr_scale: float = 0.1   # Start at 10% LR, ramp to 100%

    # --- Warmup / scheduler ---
    warmup_steps: int = 500
    cosine_restart_period: int = 20000
    cosine_restart_mult: float = 1.5
    cosine_min_lr_ratio: float = 0.01

    # --- Learnable LR controller ---
    lr_controller_enabled: bool = True
    lr_controller_update_interval: int = 50
    lr_controller_meta_lr: float = 0.01
    lr_controller_momentum: float = 0.9
    lr_controller_dead_zone: Tuple[float, float] = (-0.03, 0.03)  # Empirical: signals are ±0.01-0.15
    lr_controller_scale_min: float = 0.1
    lr_controller_scale_max: float = 5.0

    # --- Plateau detection ---
    plateau_patience: int = 40    # intervals without improvement before reducing
    plateau_factor: float = 0.90  # reduce LR by this factor on plateau
    plateau_cooldown: int = 20    # intervals after reduction before next
    plateau_min_scale: float = 0.1  # minimum LR scale
    plateau_max_reductions: int = 5  # max plateau reductions (0.90^5 ≈ 0.59x)


def snap_add_for_alignment(C_old: int, num_to_add: int, alignment: int = 8) -> int:
    """Adjust num_to_add so (C_old + result) is a multiple of alignment.

    Rounds target UP to next aligned value. Returns adjusted count (>= 0).
    """
    if num_to_add <= 0 or alignment <= 1:
        return num_to_add
    target = C_old + num_to_add
    aligned = ((target + alignment - 1) // alignment) * alignment
    return max(0, aligned - C_old)


def snap_remove_for_alignment(C_old: int, num_to_remove: int, alignment: int = 8,
                              min_channels: int = 8) -> int:
    """Adjust num_to_remove so (C_old - result) is a multiple of alignment.

    Rounds result UP (removes more to reach alignment). Never goes below min_channels.
    Returns adjusted count (>= 0).
    """
    if num_to_remove <= 0 or alignment <= 1:
        return num_to_remove
    target = C_old - num_to_remove
    aligned = (target // alignment) * alignment
    aligned = max(aligned, min_channels)
    return max(0, C_old - aligned)


@dataclass
class ASANNConfig:
    # ----- Architecture initialization -----
    d_init: int = 32
    initial_num_layers: int = 2
    d_min: int = 8
    min_depth: int = 2

    # ----- Auto-scaling for high-dimensional inputs (RC1) -----
    d_init_auto: bool = False          # If True, auto-compute d_init from d_input
    d_init_ratio: float = 4.0          # d_init = d_input / ratio (clamped)
    d_init_min: int = 32               # Floor
    d_init_max: int = 512              # Ceiling (GPU-friendly)

    # ----- Epoch-based timing (v2: replaces step-based intervals) -----
    warmup_epochs: int = 3                    # Warmup epochs before surgery begins
    surgery_epoch_interval: int = 2           # Surgery every N epochs
    eval_epoch_interval: int = 1              # Validation / diagnosis every N epochs
    meta_update_epoch_interval: int = 5       # Meta-learner threshold updates every N epochs

    # ----- Legacy step-based timing (used as fallback when steps_per_epoch unknown) -----
    surgery_interval_init: int = 500
    warmup_steps: int = 1000
    convergence_intervals: int = 5   # K consecutive "stable" intervals (legacy)

    # ----- Architecture stability controls (RC3) -----
    surgery_cooldown_multiplier: float = 1.1   # Multiply interval after each surgery round (was 1.0 = off)
    max_total_surgeries_per_interval: int = 99  # Global soft cap (99 = rely on per-type caps)
    min_post_surgery_steps: int = 200          # Min steps between surgery rounds (was 0 = off)

    # ----- Per-type surgery budgets -----
    # Each surgery type gets its own independent budget so they don't starve each other.
    # The global max_total_surgeries_per_interval still applies as an outer cap.
    max_neuron_surgeries_per_interval: int = 128    # channel/neuron adds + removes (was 12)
    max_operation_surgeries_per_interval: int = 2    # op adds + removes
    max_layer_surgeries_per_interval: int = 1        # layer adds + removes
    max_connection_surgeries_per_interval: int = 2   # connection adds + removes

    # ----- Loss-gated surgery (Fix 1, legacy — replaced by diagnosis engine) -----
    loss_gate_enabled: bool = True
    loss_gate_window: int = 300          # Window size for improvement detection
    loss_gate_min_improvement: float = 0.01  # Relative improvement threshold (1%)

    # ----- Stability budget for convergence (Fix 2, legacy — replaced by health-based) -----
    stability_cost_threshold: float = 0.02  # 2% cost change = still "stable"

    # ----- Exponential surgery interval (Fix 3) -----
    surgery_interval_growth: float = 1.1   # Multiply interval after EACH surgery round
    surgery_interval_max: int = 3000       # Hard cap on surgery interval

    # ----- Diagnosis engine (v2) -----
    diagnosis_enabled: bool = True            # Use diagnosis-based surgery (disable for legacy mode)
    diagnosis_window: int = 5                 # Number of eval windows for trend computation
    overfitting_gap_early: float = 0.15       # train-val gap threshold for OVERFITTING_EARLY
    overfitting_gap_moderate: float = 0.30    # Gap threshold for OVERFITTING_MODERATE
    overfitting_gap_severe: float = 0.50      # Gap threshold for OVERFITTING_SEVERE
    underfitting_loss_threshold: float = 0.01 # Min relative train loss improvement rate for "improving"
    stagnation_threshold: float = 0.005       # Both losses flat below this slope → stagnation

    # ----- Augmentation -----
    dataset_augmented: bool = False           # True = dataset applies augmentation (AutoAugment etc),
                                              # so GPU augment should only do cutout. False = GPU does
                                              # full augmentation (flip, crop, jitter, cutout).
    cutout_size: Optional[int] = None         # Cutout patch size (None = H // 2). Set to 2 for MNIST.
    physics_ops_enabled: bool = False        # Include physics ops (SpatPoly, SpatBranch, derivatives) in surgery candidates

    # ----- GPU elastic deformation (replaces CPU ElasticTransform) -----
    elastic_enabled: bool = False             # Enable GPU elastic deformation in trainer
    elastic_alpha: float = 30.0               # Displacement magnitude (pixels)
    elastic_sigma: float = 4.0                # Gaussian smoothing sigma (kernel = 8*sigma+1)

    # ----- Mixup augmentation (applied in trainer for classification) -----
    mixup_alpha: float = 0.2                  # Beta distribution parameter. 0 = disabled.
    mixup_enabled: bool = True                # Enable mixup for spatial classification models

    # ----- CutMix augmentation (applied in trainer for classification, paired with Mixup) -----
    # When both mixup_enabled and cutmix_enabled are True, the trainer picks one per batch:
    # CutMix with probability cutmix_prob, otherwise Mixup. They are NOT applied together
    # to the same image (that would erase too much content).
    cutmix_alpha: float = 1.0                 # Beta distribution parameter (paper default). 0 = disabled.
    cutmix_enabled: bool = False              # Enable CutMix for spatial classification models
    cutmix_prob: float = 0.5                  # P(CutMix | augmenting) when both Mixup and CutMix are on

    # ----- DropPath / Stochastic Depth (Huang et al., 2016) -----
    drop_path_rate: float = 0.1               # Max drop rate (linearly increases over depth)
    drop_path_enabled: bool = True            # Enable DropPath for spatial models

    # ----- EMA of model weights (Polyak averaging) -----
    ema_enabled: bool = True                  # Use EMA weights for validation/evaluation
    ema_decay: float = 0.999                  # EMA decay factor (higher = slower update)

    # ----- Treatment engine (v2) -----
    recovery_epochs: int = 3                  # Epochs to wait after treatment before re-diagnosis (legacy alias)
    min_recovery_epochs: int = 3              # Minimum recovery epochs before evaluation
    max_recovery_epochs: int = 15             # Maximum wait before forced rollback/accept
    recovery_catastrophic_ratio: float = 3.0  # Rollback immediately if loss > 3x pre-treatment
    recovery_harmful_ratio: float = 2.0       # Rollback after min_recovery if loss > 2x pre-treatment
    recovery_acceptable_ratio: float = 1.0    # Accept treatment if loss ≤ pre-treatment level
    structural_recovery_multiplier: float = 2.5  # min_recovery multiplier for structural treatments (level >= 2)
    recovery_best_metric_target: float = 0.90    # Accept if target metric reaches this fraction of best
    max_treatment_escalations: int = 3        # Max escalation attempts per disease type
    treatment_exhaustion_patience: int = 3    # Consecutive 'no applicable treatment' before early stop
    stop_on_treatment_exhaustion: bool = True  # If False, training continues even when treatments are exhausted
    disabled_diseases: tuple = ()             # Disease types to skip (e.g. ("CLASS_IMBALANCE",))
    label_smoothing_alpha: float = 0.1        # Label smoothing factor when prescribed

    # ----- Treatment intensity controls (previously hardcoded) -----
    # These allow per-experiment tuning of treatment aggressiveness.
    # Defaults match original hardcoded values for backward compatibility.
    dropout_light_p: float = 0.1              # Dropout probability for DROPOUT_LIGHT treatment
    dropout_heavy_p: float = 0.3              # Dropout probability for DROPOUT_HEAVY treatment
    wd_boost_factor: float = 2.0              # Weight decay multiplier per WEIGHT_DECAY_BOOST
    wd_boost_max_stacks: int = 3              # Max cumulative WD boost applications
    lr_reduce_factor: float = 0.5             # LR multiplier per LR_REDUCE application
    lr_reduce_max_stacks: int = 3             # Max cumulative LR reduce applications
    aggressive_reg_dropout_p: float = 0.3     # Dropout p in AGGRESSIVE_REGULARIZATION
    aggressive_reg_wd_factor: float = 3.0     # WD boost factor in AGGRESSIVE_REGULARIZATION
    aggressive_reg_lr_factor: float = 0.5     # LR reduce factor in AGGRESSIVE_REGULARIZATION

    # ----- Dose-adaptive treatment -----
    dose_adaptive_enabled: bool = True        # Scale treatment dose by model capacity
    dose_reference_capacity: float = 1024.0   # Reference "adult" capacity (layers × median_width)
    dose_min_factor: float = 0.15             # Minimum dose factor (tiny models still get some)
    dose_titration_levels: int = 4            # Discrete dose levels before open heart surgery
    dose_titration_window: int = 5            # Epochs per titration level before lowering dose
    open_heart_max_remove_frac: float = 0.15  # Max fraction of neurons to remove in open heart surgery
    open_heart_memorization_threshold: float = 3.0  # train/val activation ratio → memorization

    # ----- Dose-adaptive treatment (dataset-size scaling) -----
    dose_dataset_scaling_enabled: bool = True     # Scale treatment dose by dataset size
    dose_reference_n_samples: float = 5000.0      # Reference "large" dataset size
    dose_dataset_min_factor: float = 0.10         # Floor: tiny datasets get at least 10% dose

    # ----- Performance-aware diagnosis gate (previously hardcoded) -----
    # Suppresses false-positive diseases when model is near its best val loss.
    perf_gate_tight: float = 1.5              # Suppress UNDERFIT/CAPACITY/STAGNATION when val_loss < best * this
    perf_gate_loose: float = 2.0              # Suppress STAGNATION only when val_loss < best * this

    # ----- Two-phase warmup (diagnosis engine) -----
    hard_warmup_epochs: int = 20              # Max epochs for hard warmup (no diagnosis at all)
    soft_warmup_epochs: int = 10              # Soft warmup duration after hard ends (only severe diseases)

    # ----- Lab Diagnostic System -----
    lab_enabled: bool = True                    # Enable lab referral system
    lab_referral_recurrence: int = 2            # Refer after N occurrences of same disease
    lab_referral_on_ambiguous: bool = True      # Refer when multiple diseases detected
    lab_max_tier: int = 2                       # Maximum lab test tier (1-3)
    lab_probe_steps: int = 50                   # Steps for Tier 3 architecture comparison
    lab_confidence_threshold: float = 0.7       # Stop escalating tests when confidence >= this

    # ----- PDE Discovery Operations -----
    polynomial_max_degree: int = 3    # Maximum polynomial degree to discover (2 or 3)
    branch_count: int = 2             # Number of branches in BranchedOperationBlock

    # ----- KAN (Kolmogorov-Arnold Network) Operations -----
    kan_grid_size: int = 8                # Number of RBF grid centers for KAN operations

    # ----- Graph Discovery Operations -----
    graph_diffusion_max_hops: int = 3   # Max hops for GraphDiffusion (K in A^K)
    graph_attention_heads: int = 1      # Attention heads for GraphAttentionAggregation
    graph_initial_gate: float = -1.0    # Initial gate logit for graph ops (sigmoid(-1)=0.27)
    graph_spectral_K: int = 3           # Chebyshev polynomial order for SpectralConv
    graph_pe_k: int = 8                 # Number of Laplacian eigenvectors for positional encoding
    graph_appnp_alpha: float = 0.1      # Teleport probability for APPNP propagation
    graph_appnp_K: int = 10             # Power iteration steps for APPNP
    graph_sgc_K: int = 2                # Pre-propagation hops for SGC
    graph_dropedge_rate: float = 0.2    # Edge dropout rate for DropEdge regularization

    # ----- Encoder Framework -----
    # General encoder framework: ASANN can self-discover which input encoder
    # helps each task through its treatment/surgery system. Experiments provide
    # candidate encoders; ASANN starts simple and escalates when underfitting.
    encoder_type: str = "linear"                      # Default encoder (backward compatible)
    encoder_candidates: Optional[List[str]] = None    # None = no switching allowed
    encoder_d_output: Optional[int] = None            # None = use d_init
    encoder_switch_warmup_epochs: int = 15            # Blending warmup for encoder switch

    # Encoder-specific hyperparameters (used when that encoder type is created)
    encoder_patch_size: int = 4                        # PatchEmbed: patch size
    encoder_patch_d_embed: int = 64                    # PatchEmbed: embedding dimension
    encoder_fourier_frequencies: int = 64              # Fourier: number of random frequencies
    encoder_fourier_sigma: float = 10.0                # Fourier: frequency scale (higher = sharper features)
    encoder_temporal_patch_size: int = 8               # Temporal: patch size along time axis
    encoder_temporal_n_features: Optional[int] = None  # Temporal: number of input features (inferred if None)
    encoder_temporal_window_size: Optional[int] = None # Temporal: window size (inferred if None)
    encoder_gnn_layers: int = 2                        # Graph/Molecular: number of GNN layers
    encoder_mol_gnn_type: str = "gine"                  # Molecular: GNN variant (gat, gcn, gin, gine)
    encoder_mol_hidden_dim: int = 64                   # Molecular: GNN hidden dimension
    encoder_mol_atom_feature_dim: int = 32             # Molecular: atom feature vector dim (from bio_utils)
    encoder_mol_bond_feature_dim: int = 12             # Molecular: bond feature dim (0=disabled, 12=full; needed for gine)
    # Dual drug-cell encoder
    dual_encoder_cell_hidden: int = 256                # Dual: cell line MLP hidden dim
    dual_encoder_cell_out: int = 64                    # Dual: cell line MLP output dim
    dual_encoder_drug_out: int = 64                    # Dual: drug GNN output dim
    encoder_transformer_d_model: int = 64              # Transformer: model dimension
    encoder_transformer_nhead: int = 4                 # Transformer: number of attention heads
    encoder_transformer_layers: int = 2                # Transformer: number of encoder layers
    encoder_transformer_ff_dim: Optional[int] = None   # Transformer: feedforward dim (None = 4*d_model)
    encoder_transformer_dropout: float = 0.1           # Transformer: dropout rate
    encoder_transformer_chunk_size: int = 16           # Transformer: raw features per token

    # ----- Jumping Knowledge (Xu et al., 2018) -----
    # JK is NOT a static toggle — it's added/removed by the treatment system.
    # These control the JK projection dimension when the treatment activates JK.
    jk_d: int = 0                        # JK projection dimension (0 = auto: use d_init)
    jk_enabled: bool = False             # True = enable JK from init (spatial + flat)

    # ----- Op gating toggle -----
    op_gating_enabled: bool = True       # False = ops compute pure op(x) without gated residual

    # ----- SpinalNet-style architecture -----
    spinal_enabled: bool = False         # SpinalNet-style input/output split across layers

    # ----- Target noise regularization -----
    target_noise_scale: float = 0.0      # Gaussian noise std added to regression targets (0 = off)
                                         # Applied by treatment system to prevent memorization

    # ----- Memorization detection & treatment -----
    memorization_train_loss_thresh: float = 0.05   # Train loss below this = suspiciously low
    memorization_gap_thresh: float = 0.15           # Train-val gap above this + low train loss = memorization
    memorization_entropy_thresh: float = 1.0         # Per-neuron activation entropy below this = memorizing
    activation_noise_std: float = 0.0                # Gaussian noise std on activations (0 = off, set by treatment)
    memorization_min_train_samples: int = 500        # Skip memorization detection if training set < this (tiny sets naturally memorize)
    memorization_max_consecutive: int = 5            # After N consecutive memorization diagnoses without improvement, stop diagnosing

    # ----- Stalled convergence detection -----
    # Detects when val metric hasn't improved for many epochs despite model being "healthy".
    # Unlike UNDERFITTING/STAGNATION, this is NOT suppressed by the performance gate —
    # it fires based on absolute lack of improvement, triggering structural treatments
    # (normalization, JK, skip connections) that the performance gate would otherwise block.
    stalled_convergence_patience: int = 25     # Epochs without val improvement -> diagnose
    stalled_convergence_min_epochs: int = 40   # Minimum total epochs before checking
    stalled_convergence_min_improvement: float = 0.005  # Minimum relative improvement to reset patience

    # ----- Immunosuppression (surgery warmup) -----
    # When structural operations are inserted mid-training (BatchNorm, JK, ResNet,
    # graph ops, etc.), they can disrupt the learned activation distribution and
    # cause model collapse. Immunosuppression wraps new ops in a blending gate:
    #   output = (1 - alpha) * x + alpha * op(x)
    # where alpha ramps from 0 → 1 over surgery_warmup_epochs.
    # This gives the model time to adapt its weights to the new component.
    surgery_warmup_epochs: int = 10  # Epochs for new ops to ramp from 0% to 100% effect

    # ----- Health-based stability (v2) -----
    stability_healthy_epochs: int = 8         # Consecutive HEALTHY evals → architecture_stable
    stability_can_break: bool = True          # Allow re-opening surgery if sick post-stable

    # ----- Post-stable auto-stop (v2, epoch-based) -----
    post_stable_patience_epochs: int = 30     # Eval epochs without val improvement → stop

    # ----- Hard complexity ceiling (Fix 4) -----
    # If cost > ceiling_mult * target, BLOCK all add-surgeries
    complexity_ceiling_mult: float = 1.0   # 1.0 = hard cap at target

    # ----- Per-layer width budget (Fix 6) -----
    max_layer_width_ratio: float = 3.0     # Max width / median width ratio

    # ----- Neuron surgery thresholds -----
    gds_k: float = 1.5  # add threshold = mean + k * std of GDS across layers
    nus_percentile: float = 5.0  # remove bottom k-th percentile neurons
    activation_history_window: int = 100  # T steps for average activation tracking

    # ----- Layer surgery thresholds -----
    saturation_threshold: float = 0.8
    identity_threshold: float = 0.05
    loss_plateau_steps: int = 200  # P steps for plateau detection
    loss_plateau_epsilon: float = 1e-4
    layer_identity_consecutive: int = 3  # K consecutive intervals for layer removal
    layer_add_cooldown_steps: int = 3000  # Min steps between layer insertions (was 0 -- too fast)
    max_unenriched_layers: int = 0        # ALL layers must be enriched before adding new ones
    unenriched_removal_intervals: int = 5  # Remove unenriched layer after N consecutive intervals
    min_healthy_for_layer_add: int = 3    # Consecutive HEALTHY diagnoses before allowing layer addition
    surgery_interval_stable: int = 5000   # Step-based surgery interval when architecture is stable (legacy path)
    stability_review_epochs: int = 10     # Epoch interval for architecture reviews when stable

    # ----- Operation surgery -----
    benefit_threshold: float = 0.001  # minimum loss improvement to add an operation
    max_ops_per_layer: int = 16        # Cap operations per layer (was 0 = unlimited)
    op_removal_threshold_mult: float = 3.0   # Removal needs 3x the benefit of addition
    op_removal_consecutive: int = 2          # Must be flagged 2 consecutive rounds before removal
    op_protect_last_nontrivial: bool = True  # Never remove the last non-trivial op from a layer

    # Stochastic op exploration: randomly insert untried ops so they get
    # trained and evaluated by removal probes later (mutation + selection).
    # This solves the chicken-and-egg problem where complex parametric ops
    # can't show benefit on a single forward pass but would be valuable
    # after training.
    op_exploration_enabled: bool = True
    op_exploration_prob: float = 0.15  # probability of exploring per surgery
    excluded_ops: tuple = ()  # operation names to exclude from exploration

    # ----- Connection surgery -----
    connection_threshold: float = 0.5  # CLGC threshold to create connection
    connection_remove_threshold: float = 0.01  # CU threshold to remove connection
    connection_remove_consecutive: int = 3  # K consecutive intervals

    # ----- Surgery budgets -----
    max_neurons_add_per_surgery: int = 64      # Global ceiling across all layers (was 4)
    max_neurons_remove_per_surgery: int = 32   # Global ceiling across all layers (was 2)
    max_layers_add_per_surgery: int = 1
    max_layers_remove_per_surgery: int = 1
    max_ops_change_per_surgery: int = 2
    max_connections_change_per_surgery: int = 2

    # ----- Complexity cost -----
    complexity_target: float = 10000.0  # C_target in FLOPs
    complexity_lambda_init: float = 1e-6
    complexity_dual_step: float = 1e-7  # rho for dual ascent

    # ----- Auto complexity target (RC2) -----
    complexity_target_auto: bool = False       # Auto-compute from initial cost
    complexity_target_multiplier: float = 5.0  # target = multiplier × initial_cost (was 1.5 — too tight)
    complexity_lambda_max: float = 5e-5        # Lambda ceiling (was inf — unbounded can crush architecture)

    # ----- Scale-free auto complexity target -----
    complexity_target_base_per_class: float = 50000.0   # Base FLOPs per output class
    complexity_target_per_spatial_dim: float = 2.0       # Multiplier per spatial input dim

    # ----- Spatial-aware operations (RC4) -----
    spatial_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W) or None for tabular

    # ----- Spatial/Conv backbone -----
    c_stem_init: int = 16             # ConvStem output channels (small — surgery grows it)
    min_spatial_layers: int = 1        # Minimum spatial layers before flatten allowed
    min_spatial_resolution: int = 4    # Don't downsample below 4x4
    max_channels: int = 256            # Max channels per spatial layer
    max_channels_add_per_surgery: int = 64   # Global ceiling for channel adds (was 4)
    max_channels_remove_per_surgery: int = 32  # Global ceiling for channel removes (was 2)
    min_spatial_channels: int = 16      # Min channels per spatial layer (NUS cannot breach)
    min_channels_add_per_qualifying_layer: int = 4  # Minimum channels to add per qualifying spatial layer
    max_channel_ratio: float = 2.0     # Max channel ratio between adjacent spatial layers
    spatial_downsample_stages: Union[int, str] = 2 # int for fixed stages, "auto" for self-discovery
    max_downsample_stages: int = 4          # Max stride-2 layers allowed (guard for auto mode)
    stride_probe_threshold: float = 0.95    # stride-2 wins if loss < stride-1 * this

    # ----- Gradient-proportional channel growth -----
    channel_growth_fraction: float = 0.125     # Base fraction of width to add per surgery (64ch → +8)
    channel_growth_max_fraction: float = 0.25  # Hard cap fraction (64ch → max +16)
    channel_growth_gds_scale: bool = True      # Scale growth by GDS/threshold ratio (XGBoost-style)
    channel_removal_fraction: float = 0.10     # Fraction of below-NUS channels to remove per layer
    channel_removal_max_fraction: float = 0.20 # Hard cap fraction for removal per layer
    per_layer_channel_budget: bool = True      # Each layer gets its own budget (vs shared global)
    channel_alignment: int = 8                 # Align spatial channel widths to multiples of this (Tensor Core efficiency; set 1 to disable)

    # ----- Validation-gated surgery (RC6) -----
    validation_gated_surgery: bool = True      # Gate op additions on validation loss (was False)
    validation_gate_tolerance: float = 0.02    # Allow up to 2% val loss increase

    # ----- Meta-learning (adaptive thresholds + surgery interval only) -----
    meta_update_interval: int = 1000  # C steps for meta updates (legacy, used if epoch mode off)
    meta_tighten_rate: float = 1.05           # Threshold tightening rate (tabular)
    meta_tighten_rate_spatial: float = 1.01   # Gentler tightening for spatial models

    # ----- Neuron splitting noise -----
    split_noise_scale: float = 0.01

    # ----- Training input noise (data augmentation for small datasets) -----
    train_noise_std: float = 0.0  # Gaussian noise added to inputs during training only

    # ----- Connection init scale -----
    connection_init_scale: float = 0.1  # Was 0.01 — too small for connections to develop utility

    # ----- Architecture-Aware Stopping -----
    auto_stop_enabled: bool = True           # Extend training past max_steps if arch not stable
    post_stable_patience_intervals: int = 6  # Eval intervals without val improvement → stop
    post_stable_improvement_threshold: float = 0.001  # 0.1% relative val loss improvement
    post_stable_min_steps: int = 1000        # Min steps after arch_stable before stop check
    hard_max_multiplier: float = 1.0         # Hard cap = max_steps × multiplier (safety net)

    # ----- Optimizer -----
    optimizer: ASANNOptimizerConfig = field(default_factory=ASANNOptimizerConfig)

    # ----- Device -----
    device: str = "cpu"

    # ----- Performance -----
    amp_enabled: bool = False              # Automatic Mixed Precision (FP16 on CUDA)
    torch_compile_enabled: bool = False    # torch.compile() for eval/validation forward passes
    torch_compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    train_eval_max_batches: Optional[int] = None  # Subsample train eval (None = full)
    csv_logging_enabled: bool = False              # Per-step CSV logging (disable for speed)

    # ----- CUDA custom ops -----
    use_cuda_ops: bool = True             # Use custom CUDA kernels for operations

    def __post_init__(self):
        # Normalize device to plain string (torch.device objects break == "cuda")
        if isinstance(self.device, torch.device):
            self.device = self.device.type
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # torch.compile is incompatible with custom CUDA C++ ops (graph breaks)
        # and Triton is unavailable on Windows
        if self.use_cuda_ops and self.torch_compile_enabled:
            self.torch_compile_enabled = False
        # Auto multi-scale momentum for spatial models:
        # Default betas (0.9, 0.9, 0.9, 0.999) collapse 3 momentum scales into one.
        # Spatial models benefit from true multi-scale smoothing: fast/medium/slow.
        if self.spatial_shape is not None and self.optimizer.betas == (0.9, 0.9, 0.9, 0.999):
            self.optimizer.betas = (0.9, 0.95, 0.99, 0.999)

    # ------------------------------------------------------------------
    #  Auto-configuration from task descriptors
    # ------------------------------------------------------------------
    _VALID_MODALITIES = {
        "tabular", "image", "pde", "graph", "temporal_graph",
        "molecular", "molecular_classification",
        "pharmacogenomic", "leukemia",
    }

    @classmethod
    def from_task(
        cls,
        task_type: str,
        modality: str,
        d_input: int,
        d_output: int,
        n_samples: int,
        spatial_shape: Optional[Tuple[int, int, int]] = None,
        n_features: Optional[int] = None,
        *,
        device: str = "auto",
    ) -> "ASANNConfig":
        """Create a fully configured ASANNConfig from task descriptors only.

        This method deterministically derives every hyperparameter from the
        task description so that no per-experiment tuning is required.

        Parameters
        ----------
        task_type : str
            ``"regression"`` or ``"classification"``.
        modality : str
            One of: ``"tabular"``, ``"image"``, ``"pde"``, ``"graph"``,
            ``"temporal_graph"``, ``"molecular"``,
            ``"molecular_classification"``, ``"pharmacogenomic"``,
            ``"leukemia"``.
        d_input : int
            Input dimensionality (number of features for tabular, flattened
            size for images, number of input coordinates for PDEs).
        d_output : int
            Output dimensionality (1 for scalar regression, number of
            classes for classification).
        n_samples : int
            Number of training samples.
        spatial_shape : tuple of int, optional
            ``(C, H, W)`` for image modalities.
        n_features : tuple of int, optional
            Number of node features for graph modalities.
        device : str
            Device string (default ``"auto"``).

        Returns
        -------
        ASANNConfig
            A fully configured instance.  Two extra attributes are set on
            the returned object: ``recommended_max_epochs`` and
            ``recommended_batch_size``.
        """
        # --- validation ---
        assert task_type in ("regression", "classification"), \
            f"task_type must be 'regression' or 'classification', got '{task_type}'"
        assert modality in cls._VALID_MODALITIES, \
            f"modality must be one of {cls._VALID_MODALITIES}, got '{modality}'"
        if modality == "image":
            assert spatial_shape is not None, "spatial_shape required for image modality"

        H = spatial_shape[1] if spatial_shape is not None else 0

        # ================================================================
        #  1. Architecture: d_init, initial_num_layers
        # ================================================================
        if modality == "graph":
            d_init = 16
        elif modality == "temporal_graph":
            d_init = 64
        elif modality == "tabular":
            if d_input < 30:
                d_init = 32 if n_samples < 2000 else 48
            else:
                d_init = 64
        elif modality == "image":
            if H <= 28:
                d_init = 32
            elif H <= 32:
                d_init = 96 if d_output <= 10 else 128
            else:
                d_init = 32
        elif modality == "pde":
            d_init = 48 if n_samples < 10000 else 64
        elif modality in ("molecular", "molecular_classification"):
            d_init = 64
        elif modality == "pharmacogenomic":
            d_init = 128
        elif modality == "leukemia":
            d_init = 64
        else:
            d_init = 32

        if modality == "tabular":
            initial_num_layers = 1 if task_type == "classification" else 2
        elif modality == "image":
            if H <= 28:
                initial_num_layers = 2
            elif H <= 32:
                initial_num_layers = 1 if d_output <= 10 else 4
            else:
                initial_num_layers = 4
        elif modality == "pde":
            initial_num_layers = 3 if n_samples < 10000 else 4
        elif modality == "graph":
            initial_num_layers = 1
        elif modality == "temporal_graph":
            initial_num_layers = 3
        elif modality in ("molecular", "molecular_classification"):
            initial_num_layers = 2
        elif modality == "pharmacogenomic":
            initial_num_layers = 3
        elif modality == "leukemia":
            initial_num_layers = 3
        else:
            initial_num_layers = 2

        # ================================================================
        #  2. Complexity target & ceiling
        # ================================================================
        if modality in ("image", "graph", "temporal_graph", "molecular",
                        "molecular_classification", "pharmacogenomic", "leukemia"):
            complexity_target_auto = True
            complexity_target = 10000.0
        elif modality == "tabular":
            complexity_target_auto = False
            scale = max(1.0, min(d_input / 8.0, 10.0))
            complexity_target = float(max(10000, min(int(n_samples * scale), 200000)))
        elif modality == "pde":
            complexity_target_auto = False
            complexity_target = float(max(50000, min(n_samples * 4, 250000)))
        else:
            complexity_target_auto = True
            complexity_target = 50000.0

        if modality == "graph":
            complexity_ceiling_mult = 2.8
            hard_max_multiplier = 1.0
        elif modality == "temporal_graph":
            complexity_ceiling_mult = 5.0
            hard_max_multiplier = 1.0
        elif modality in ("molecular", "molecular_classification", "pharmacogenomic"):
            complexity_ceiling_mult = 5.0
            hard_max_multiplier = 2.0
        elif modality == "image" and d_output >= 100:
            complexity_ceiling_mult = 5.0
            hard_max_multiplier = 1.0
        elif modality == "image" and H > 28:
            complexity_ceiling_mult = 3.0
            hard_max_multiplier = 1.0
        else:
            complexity_ceiling_mult = 1.0
            hard_max_multiplier = 1.0

        complexity_target_multiplier = 8.0 if modality == "pharmacogenomic" else 5.0

        # ================================================================
        #  3. Encoder
        # ================================================================
        _ENCODER_MAP = {
            "tabular": None,
            "image": ["conv", "patch_embed"],
            "pde": ["linear", "fourier"],
            "graph": ["graph_node"],
            "temporal_graph": ["temporal_graph"],
            "molecular": ["molecular_graph"],
            "molecular_classification": ["molecular_graph"],
            "pharmacogenomic": ["dual_drug_cell"],
            "leukemia": None,
        }
        encoder_candidates = _ENCODER_MAP.get(modality)

        # Spatial / conv backbone
        c_stem_init = 16
        spatial_downsample_stages: Union[int, str] = 2
        max_downsample_stages = 4
        max_channels = 256
        if spatial_shape is not None:
            if H <= 28:
                c_stem_init = 32
                spatial_downsample_stages = 0
            elif H <= 32:
                c_stem_init = 128 if d_output >= 100 else 96
                spatial_downsample_stages = "auto" if d_output <= 10 else 3
                max_downsample_stages = 3
                max_channels = 512 if d_output <= 10 else 256
            else:
                c_stem_init = 32
                spatial_downsample_stages = 3
                max_downsample_stages = 4

        # Molecular encoder
        encoder_gnn_layers = 3 if modality in ("molecular", "molecular_classification",
                                                "pharmacogenomic") else 2
        encoder_switch_warmup_epochs = 10 if modality in ("molecular", "molecular_classification",
                                                           "pharmacogenomic") else 15

        # ================================================================
        #  4. Timing (epoch-based)
        # ================================================================
        if modality == "tabular":
            warmup_epochs = 5
            surgery_epoch_interval = 3
            eval_epoch_interval = 2
            meta_update_epoch_interval = 10
        elif modality == "image":
            warmup_epochs = 3 if H <= 28 else 5
            surgery_epoch_interval = 2 if H <= 28 else (4 if d_output <= 10 else 5)
            eval_epoch_interval = 1
            meta_update_epoch_interval = 5 if H <= 28 else 8
        elif modality == "pde":
            warmup_epochs = 5
            surgery_epoch_interval = 3
            eval_epoch_interval = 2
            meta_update_epoch_interval = 10
        elif modality == "graph":
            warmup_epochs = 5
            surgery_epoch_interval = 11
            eval_epoch_interval = 1
            meta_update_epoch_interval = 10
        elif modality == "temporal_graph":
            warmup_epochs = 3
            surgery_epoch_interval = 2
            eval_epoch_interval = 5
            meta_update_epoch_interval = 10
        elif modality in ("molecular", "molecular_classification"):
            warmup_epochs = 20
            surgery_epoch_interval = 5
            eval_epoch_interval = 2
            meta_update_epoch_interval = 10
        elif modality == "pharmacogenomic":
            warmup_epochs = 5
            surgery_epoch_interval = 3
            eval_epoch_interval = 2
            meta_update_epoch_interval = 10
        elif modality == "leukemia":
            warmup_epochs = 5
            surgery_epoch_interval = 3
            eval_epoch_interval = 2
            meta_update_epoch_interval = 10
        else:
            warmup_epochs = 3
            surgery_epoch_interval = 2
            eval_epoch_interval = 1
            meta_update_epoch_interval = 5

        # Step-based fallbacks (estimated from sample count)
        _batch_est = 256
        _steps_per_epoch = max(1, n_samples // _batch_est)
        surgery_interval_init = max(150, surgery_epoch_interval * _steps_per_epoch)
        warmup_steps = max(300, warmup_epochs * _steps_per_epoch)
        meta_update_interval = max(500, meta_update_epoch_interval * _steps_per_epoch)

        # ================================================================
        #  5. Diagnosis thresholds
        # ================================================================
        if modality in ("molecular", "molecular_classification"):
            overfitting_gap_early = 1.0
            overfitting_gap_moderate = 2.0
            overfitting_gap_severe = 5.0
        elif modality == "temporal_graph" and task_type == "regression":
            overfitting_gap_early = 0.40
            overfitting_gap_moderate = 0.60
            overfitting_gap_severe = 0.80
        else:
            overfitting_gap_early = 0.30
            overfitting_gap_moderate = 0.50
            overfitting_gap_severe = 0.70

        if modality == "image" and H >= 32:
            stagnation_threshold = 0.003
            saturation_threshold = 0.90
            diagnosis_window = 8
            hard_warmup_epochs = 30
            soft_warmup_epochs = 15
            stalled_convergence_patience = 40
            perf_gate_tight = 1.2
            perf_gate_loose = 1.5
        elif modality in ("graph", "temporal_graph"):
            stagnation_threshold = 0.005
            saturation_threshold = 0.80
            diagnosis_window = 5
            hard_warmup_epochs = 3
            soft_warmup_epochs = 2
            stalled_convergence_patience = 25
            perf_gate_tight = 1.5
            perf_gate_loose = 2.0
        elif modality in ("molecular", "molecular_classification"):
            stagnation_threshold = 0.005
            saturation_threshold = 0.80
            diagnosis_window = 5
            hard_warmup_epochs = 20
            soft_warmup_epochs = 10
            stalled_convergence_patience = 80
            perf_gate_tight = 1.5
            perf_gate_loose = 2.0
        elif modality == "tabular":
            stagnation_threshold = 0.005
            saturation_threshold = 0.80
            diagnosis_window = 5
            hard_warmup_epochs = 20
            soft_warmup_epochs = 10
            stalled_convergence_patience = 50
            perf_gate_tight = 1.5
            perf_gate_loose = 2.0
        else:
            stagnation_threshold = 0.005
            saturation_threshold = 0.80
            diagnosis_window = 5
            hard_warmup_epochs = 20
            soft_warmup_epochs = 10
            stalled_convergence_patience = 25
            perf_gate_tight = 1.5
            perf_gate_loose = 2.0

        # ================================================================
        #  6. Treatment
        # ================================================================
        # Unified treatment-intensity defaults across all modalities, per
        # paper §4.4 (dropout p in {0.1, 0.3}, WD boost 2x up to 3 stacks,
        # LR reduce 0.5x up to 3 stacks, aggressive regularisation triple
        # of 0.3 dropout + 3.0x WD + 0.5x LR).
        dropout_light_p = 0.1
        dropout_heavy_p = 0.3
        wd_boost_factor = 2.0
        wd_boost_max_stacks = 3
        lr_reduce_factor = 0.5
        lr_reduce_max_stacks = 3
        aggressive_reg_dropout_p = 0.3
        aggressive_reg_wd_factor = 3.0
        aggressive_reg_lr_factor = 0.5

        _stability_map = {
            "tabular": 10, "image": 10 if H >= 32 else 6,
            "pde": 10, "graph": 17, "temporal_graph": 15,
            "molecular": 15, "molecular_classification": 15,
            "pharmacogenomic": 10, "leukemia": 10,
        }
        stability_healthy_epochs = _stability_map.get(modality, 8)

        _recovery_map = {
            "tabular": 4, "image": 5 if H >= 32 else 3,
            "pde": 4, "graph": 5, "temporal_graph": 5,
            "molecular": 6, "molecular_classification": 6,
            "pharmacogenomic": 4, "leukemia": 4,
        }
        recovery_epochs = _recovery_map.get(modality, 3)
        min_recovery_epochs = max(3, recovery_epochs)
        max_recovery_epochs = 20 if (modality == "image" and H >= 32) else 15
        structural_recovery_multiplier = 3.0 if (modality == "image" and H >= 32) else 2.5

        _escalation_map = {
            "graph": 4, "temporal_graph": 4,
            "molecular": 10, "molecular_classification": 10,
            "pharmacogenomic": 4, "leukemia": 4,
        }
        max_treatment_escalations = _escalation_map.get(modality, 3)

        _exhaustion_map = {
            "molecular": 10, "molecular_classification": 10,
        }
        treatment_exhaustion_patience = _exhaustion_map.get(modality, 3)

        post_stable_patience_epochs = 80 if modality == "tabular" else 30

        # Molecular: disable auto-stop, longer min epochs
        auto_stop_enabled = modality not in ("molecular", "molecular_classification")
        stalled_convergence_min_epochs = 200 if modality in (
            "molecular", "molecular_classification") else 40

        # ================================================================
        #  7. Augmentation
        # ================================================================
        if modality in ("graph", "temporal_graph", "molecular", "molecular_classification"):
            mixup_enabled = False
            drop_path_enabled = False
        elif modality == "pde":
            mixup_enabled = False
            drop_path_enabled = True
        else:
            mixup_enabled = True
            drop_path_enabled = True

        mixup_alpha = 0.2  # Unified across modalities, per paper §4.7.
        # CutMix is on for image classification only (paired with Mixup at the batch level).
        # task_type filtering happens in the trainer; we just supply the modality default here.
        cutmix_enabled = (modality == "image")
        cutmix_alpha = 1.0
        cutmix_prob = 0.5
        physics_ops_enabled = modality == "pde"
        amp_enabled = modality == "image"

        # Image augmentation hints
        dataset_augmented = modality == "image"

        # Cutout and elastic deformation (data-level augmentation, auto-derived)
        cutout_size = None
        elastic_enabled = False
        elastic_alpha = 30.0
        elastic_sigma = 4.0
        if modality == "image" and spatial_shape is not None:
            if H <= 28:
                cutout_size = 2
                elastic_enabled = True  # handwriting-like deformation for digits
            elif H <= 32:
                cutout_size = 8
            # 96x96+ (STL-10): no cutout needed

        # Op gating: disabled for image (destabilises spatial pipelines)
        op_gating_enabled = modality != "image"

        # ================================================================
        #  8. Optimizer
        # ================================================================
        # Unified base learning rate and decoupled weight decay across all
        # modalities, per paper §4.7.
        base_lr = 1e-3
        weight_decay = 0.01

        lr_controller_scale_max = 3.0 if modality in ("image", "graph", "temporal_graph") else 5.0

        # ================================================================
        #  9. Misc
        # ================================================================
        max_ops_per_layer = 4 if modality in (
            "graph", "temporal_graph", "molecular",
            "molecular_classification", "pharmacogenomic") else 16

        # Graph-specific
        if modality == "graph":
            graph_initial_gate = 1.23
            graph_diffusion_max_hops = 1
        elif modality == "temporal_graph":
            graph_initial_gate = 0.0
            graph_diffusion_max_hops = 3
        else:
            graph_initial_gate = -1.0
            graph_diffusion_max_hops = 3

        # PDE surgery budgets
        if modality == "pde" and n_samples >= 10000:
            max_neuron_surgeries_per_interval = 16
            max_operation_surgeries_per_interval = 3
        else:
            max_neuron_surgeries_per_interval = 128
            max_operation_surgeries_per_interval = 2

        # Label smoothing hint
        label_smoothing_alpha = 0.1 if modality == "image" and d_output >= 10 else 0.1

        # Graph label smoothing
        if modality in ("graph",):
            label_smoothing_alpha = 0.19

        # ================================================================
        #  10. Recommended batch_size and max_epochs
        # ================================================================
        if modality == "temporal_graph":
            rec_batch_size = 32
        elif modality == "image" and H >= 96:
            rec_batch_size = 64
        elif n_samples < 200:
            rec_batch_size = 16
        elif n_samples < 1000:
            rec_batch_size = 32
        elif n_samples < 5000:
            rec_batch_size = 64
        elif n_samples < 20000:
            rec_batch_size = 128
        elif n_samples >= 50000 and modality == "tabular":
            rec_batch_size = 512
        else:
            rec_batch_size = 256

        _epoch_map = {
            "tabular": 400, "pde": 600, "graph": 300,
            "temporal_graph": 300, "pharmacogenomic": 100,
            "leukemia": 200, "molecular": 500,
            "molecular_classification": 500,
        }
        if modality == "image":
            rec_max_epochs = 150 if H <= 28 else 300
        else:
            rec_max_epochs = _epoch_map.get(modality, 300)

        # ================================================================
        #  Assemble
        # ================================================================
        opt = ASANNOptimizerConfig(
            base_lr=base_lr,
            weight_decay=weight_decay,
            lr_controller_scale_max=lr_controller_scale_max,
        )

        cfg = cls(
            # Architecture
            d_init=d_init,
            initial_num_layers=initial_num_layers,
            # Complexity
            complexity_target=complexity_target,
            complexity_target_auto=complexity_target_auto,
            complexity_target_multiplier=complexity_target_multiplier,
            complexity_ceiling_mult=complexity_ceiling_mult,
            hard_max_multiplier=hard_max_multiplier,
            # Encoder
            encoder_candidates=encoder_candidates,
            spatial_shape=spatial_shape,
            c_stem_init=c_stem_init,
            spatial_downsample_stages=spatial_downsample_stages,
            max_downsample_stages=max_downsample_stages,
            max_channels=max_channels,
            encoder_gnn_layers=encoder_gnn_layers,
            encoder_switch_warmup_epochs=encoder_switch_warmup_epochs,
            # Timing (epoch)
            warmup_epochs=warmup_epochs,
            surgery_epoch_interval=surgery_epoch_interval,
            eval_epoch_interval=eval_epoch_interval,
            meta_update_epoch_interval=meta_update_epoch_interval,
            # Timing (step fallbacks)
            surgery_interval_init=surgery_interval_init,
            warmup_steps=warmup_steps,
            meta_update_interval=meta_update_interval,
            # Diagnosis
            overfitting_gap_early=overfitting_gap_early,
            overfitting_gap_moderate=overfitting_gap_moderate,
            overfitting_gap_severe=overfitting_gap_severe,
            stagnation_threshold=stagnation_threshold,
            saturation_threshold=saturation_threshold,
            diagnosis_window=diagnosis_window,
            hard_warmup_epochs=hard_warmup_epochs,
            soft_warmup_epochs=soft_warmup_epochs,
            stalled_convergence_patience=stalled_convergence_patience,
            stalled_convergence_min_epochs=stalled_convergence_min_epochs,
            perf_gate_tight=perf_gate_tight,
            perf_gate_loose=perf_gate_loose,
            # Treatment
            stability_healthy_epochs=stability_healthy_epochs,
            recovery_epochs=recovery_epochs,
            min_recovery_epochs=min_recovery_epochs,
            max_recovery_epochs=max_recovery_epochs,
            structural_recovery_multiplier=structural_recovery_multiplier,
            max_treatment_escalations=max_treatment_escalations,
            treatment_exhaustion_patience=treatment_exhaustion_patience,
            post_stable_patience_epochs=post_stable_patience_epochs,
            auto_stop_enabled=auto_stop_enabled,
            dropout_light_p=dropout_light_p,
            dropout_heavy_p=dropout_heavy_p,
            wd_boost_factor=wd_boost_factor,
            wd_boost_max_stacks=wd_boost_max_stacks,
            lr_reduce_factor=lr_reduce_factor,
            lr_reduce_max_stacks=lr_reduce_max_stacks,
            aggressive_reg_dropout_p=aggressive_reg_dropout_p,
            aggressive_reg_wd_factor=aggressive_reg_wd_factor,
            aggressive_reg_lr_factor=aggressive_reg_lr_factor,
            label_smoothing_alpha=label_smoothing_alpha,
            # Augmentation
            mixup_enabled=mixup_enabled,
            mixup_alpha=mixup_alpha,
            cutmix_enabled=cutmix_enabled,
            cutmix_alpha=cutmix_alpha,
            cutmix_prob=cutmix_prob,
            drop_path_enabled=drop_path_enabled,
            physics_ops_enabled=physics_ops_enabled,
            amp_enabled=amp_enabled,
            dataset_augmented=dataset_augmented,
            cutout_size=cutout_size,
            elastic_enabled=elastic_enabled,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            op_gating_enabled=op_gating_enabled,
            # Surgery budgets
            max_neuron_surgeries_per_interval=max_neuron_surgeries_per_interval,
            max_operation_surgeries_per_interval=max_operation_surgeries_per_interval,
            max_ops_per_layer=max_ops_per_layer,
            # Graph
            graph_initial_gate=graph_initial_gate,
            graph_diffusion_max_hops=graph_diffusion_max_hops,
            # Optimizer
            optimizer=opt,
            # Device
            device=device,
        )

        # Extra attributes (not dataclass fields)
        cfg.recommended_max_epochs = rec_max_epochs
        cfg.recommended_batch_size = rec_batch_size
        return cfg


# Backward-compatibility aliases for checkpoints saved under the old class names
CSANNConfig = ASANNConfig
CSANNOptimizerConfig = ASANNOptimizerConfig
