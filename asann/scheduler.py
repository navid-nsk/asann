import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from .config import ASANNConfig
from .model import ASANNModel, GatedOperation, _TRIVIAL_OPS
from .surgery import SurgeryEngine
from .logger import SurgeryLogger
from .diagnosis import DiagnosisEngine, Diagnosis, HealthState, DiseaseType
from .treatments import TreatmentPlanner
from .lab import PatientHistory, LabDiagnostics, LabReport
from .lab_tests import create_default_lab


class GradientStatsAccumulator:
    """Accumulates gradient-derived statistics between surgery intervals.

    Between surgeries, the network trains normally. These statistics are
    accumulated over S steps and used to make surgery decisions.
    All signals are free byproducts of normal training — they come from
    the same gradients computed for weight optimization.
    """

    def __init__(self, config: ASANNConfig):
        self.config = config

        # Per-layer gradient demand scores: ||dL/dh(l)||_2 accumulated
        self.gradient_norms: Dict[int, List[float]] = {}

        # Per-neuron activation magnitudes for utility scoring
        self.activation_magnitudes: Dict[int, List[torch.Tensor]] = {}

        # Layer input/output pairs for saturation/contribution scoring
        self.layer_input_norms: Dict[int, List[float]] = {}
        self.layer_output_norms: Dict[int, List[float]] = {}
        self.layer_transform_norms: Dict[int, List[float]] = {}

        # Cross-layer gradient correlations for connection discovery
        self.layer_gradients: Dict[int, List[torch.Tensor]] = {}

        # Loss history for plateau detection
        self.loss_history: deque = deque(maxlen=config.loss_plateau_steps)

        # Graph-specific: oversmoothing detection (node feature similarity per layer)
        self.node_similarity: Dict[int, List[float]] = {}

        self.num_accumulated = 0

    def accumulate(
        self,
        model: ASANNModel,
        loss: torch.Tensor,
        layer_activations: Dict[int, torch.Tensor],
        layer_inputs: Dict[int, torch.Tensor],
        layer_outputs: Dict[int, torch.Tensor],
        hook_results: Optional[Dict] = None,
        loss_val_f: Optional[float] = None,
    ):
        """Accumulate statistics from one training step — GPU-only, zero sync.

        ALL computations stay on GPU. No .cpu() or .item() calls.
        GPU->CPU transfer is deferred to compute_signals(), which runs only
        at surgery epochs (~every 730 steps). This eliminates ~183ms of GPU
        pipeline stalls per training step.

        Args:
            hook_results: Dict from backward hooks containing pre-computed
                gradient summaries. Keys: ('gds_mean', l) -> scalar tensor,
                ('clgc', l) -> [C] tensor. When provided, .grad access on
                layer_activations/layer_outputs is skipped entirely.
            loss_val_f: Pre-computed loss float from trainer's deferred scalar
                extraction. Appended to loss_history for plateau detection.
        """
        self.num_accumulated += 1

        n_layers = model.num_layers
        _dev = loss.device  # Cache device for zero-tensor creation

        # Loss value (float from trainer's deferred scalar extraction)
        if loss_val_f is not None:
            self.loss_history.append(loss_val_f)

        for l in range(n_layers):
            # --- Gradient Demand Score (GDS) ---
            # Directly append GPU tensors (no .cpu() — deferred to compute_signals)
            if hook_results is not None and ('gds_mean', l) in hook_results:
                if l not in self.gradient_norms:
                    self.gradient_norms[l] = []
                self.gradient_norms[l].append(hook_results[('gds_mean', l)].detach())
            else:
                # Fallback: access .grad directly (requires retain_grad)
                grad_tensor = None
                if l + 1 in layer_activations and layer_activations[l + 1].grad is not None:
                    grad_tensor = layer_activations[l + 1].grad.detach()
                elif l in layer_outputs and layer_outputs[l].grad is not None:
                    grad_tensor = layer_outputs[l].grad.detach()

                if grad_tensor is not None:
                    if grad_tensor.dim() == 4:
                        per_neuron = grad_tensor.abs().mean(dim=(0, 2, 3))
                        H, W = grad_tensor.shape[2], grad_tensor.shape[3]
                        spatial_factor = (H * W) ** 0.5
                        gds_val = per_neuron.mean() * spatial_factor
                    elif grad_tensor.dim() >= 2:
                        gds_val = grad_tensor.abs().mean(dim=0).mean()
                    else:
                        gds_val = grad_tensor.abs().mean()
                    if l not in self.gradient_norms:
                        self.gradient_norms[l] = []
                    self.gradient_norms[l].append(gds_val)
                else:
                    if l not in self.gradient_norms:
                        self.gradient_norms[l] = []
                    self.gradient_norms[l].append(torch.tensor(0.0, device=_dev))

            # --- Neuron activation magnitudes (no gradients needed) ---
            # Keep on GPU (no .cpu() — deferred to compute_signals)
            if l + 1 in layer_activations:
                act = layer_activations[l + 1].detach()
                if act.dim() == 4:
                    neuron_magnitudes = act.abs().mean(dim=(0, 2, 3))
                elif act.dim() >= 2:
                    neuron_magnitudes = act.abs().mean(dim=0)
                else:
                    neuron_magnitudes = act.abs()
                if l not in self.activation_magnitudes:
                    self.activation_magnitudes[l] = []
                self.activation_magnitudes[l].append(neuron_magnitudes)

            # --- Layer Saturation Score ingredients ---
            # Keep on GPU (no .cpu() — deferred to compute_signals)
            if l in layer_inputs and l in layer_outputs:
                inp = layer_inputs[l].detach()
                out = layer_outputs[l].detach()
                inp_norm = inp.norm()
                out_norm = out.norm()
                transform_norm = (out - inp).norm() if inp.shape == out.shape else out.norm()
                if l not in self.layer_input_norms:
                    self.layer_input_norms[l] = []
                    self.layer_output_norms[l] = []
                    self.layer_transform_norms[l] = []
                self.layer_input_norms[l].append(inp_norm)
                self.layer_output_norms[l].append(out_norm)
                self.layer_transform_norms[l].append(transform_norm)

            # --- Cross-layer gradient correlations (CLGC) ---
            # Keep on GPU (no .cpu() — deferred to compute_signals)
            if hook_results is not None and ('clgc', l) in hook_results:
                if l not in self.layer_gradients:
                    self.layer_gradients[l] = []
                self.layer_gradients[l].append(hook_results[('clgc', l)].detach())
            elif l in layer_outputs and layer_outputs[l].grad is not None:
                # Fallback: access .grad directly (requires retain_grad)
                grad = layer_outputs[l].grad.detach()
                if grad.dim() == 4:
                    mean_grad = grad.mean(dim=(0, 2, 3))
                elif grad.dim() >= 2:
                    mean_grad = grad.mean(dim=0)
                else:
                    mean_grad = grad
                if l not in self.layer_gradients:
                    self.layer_gradients[l] = []
                self.layer_gradients[l].append(mean_grad)

        # --- Graph-specific: Oversmoothing detection (OSD) ---
        # Compute pairwise cosine similarity on a random sample of node pairs.
        # Only for graph models -- skip for flat/spatial.
        if getattr(model, '_is_graph', False):
            N_SAMPLE = 100  # Sample size for efficiency
            graph_N = getattr(model, '_graph_num_nodes', 0)
            for l in range(n_layers):
                if l + 1 in layer_activations:
                    act = layer_activations[l + 1].detach()
                    if act.dim() == 2 and act.shape[0] > 2:
                        # For batched graph input [B*N, d], use first graph only
                        if graph_N > 0 and act.shape[0] > graph_N:
                            act = act[:graph_N]
                        N = act.shape[0]
                        n_pairs = min(N_SAMPLE, N * (N - 1) // 2)
                        # Random pairs
                        idx_i = torch.randint(0, N, (n_pairs,), device=act.device)
                        idx_j = torch.randint(0, N, (n_pairs,), device=act.device)
                        # Cosine similarity
                        act_norm = act / (act.norm(dim=1, keepdim=True).clamp(min=1e-8))
                        cos_sim = (act_norm[idx_i] * act_norm[idx_j]).sum(dim=1)
                        mean_sim = cos_sim.mean()  # Keep on GPU (no .item())
                        if l not in self.node_similarity:
                            self.node_similarity[l] = []
                        self.node_similarity[l].append(mean_sim)

    def compute_signals(self, model: ASANNModel) -> Dict[str, Any]:
        """Compute all surgery signals from accumulated statistics.

        Returns a dictionary of signals for each surgery type:
        - GDS: Gradient Demand Score per layer (for neuron addition)
        - NUS: Neuron Utility Score per neuron (for neuron removal)
        - LSS: Layer Saturation Score per layer (for layer addition)
        - LCS: Layer Contribution Score per layer (for layer removal)
        - CLGC: Cross-Layer Gradient Correlation (for connection creation)
        """
        if self.num_accumulated == 0:
            return {}

        signals = {}

        # ==================== GDS: Gradient Demand Score ====================
        # GDS(l) = mean per-neuron |dL/dh(l)|
        # Already width-independent because accumulate() computes the mean
        # per-neuron gradient magnitude (averaged over batch and spatial dims).
        # High GDS means the layer is under-capacity: each neuron carries a
        # heavy gradient load, signaling that more neurons would help.
        # Note: gradient_norms values are GPU tensors (deferred from accumulate).
        gds = {}
        for l in range(model.num_layers):
            if l in self.gradient_norms and self.gradient_norms[l]:
                vals = self.gradient_norms[l]
                if isinstance(vals[0], torch.Tensor):
                    gds[l] = torch.stack(vals).mean().item()
                else:
                    gds[l] = float(np.mean(vals))
            else:
                gds[l] = 0.0

        signals["GDS"] = gds

        # Adaptive threshold: mean + k * std across layers
        gds_values = list(gds.values())
        if gds_values:
            gds_mean = np.mean(gds_values)
            gds_std = np.std(gds_values) if len(gds_values) > 1 else 0.0
            signals["GDS_threshold"] = gds_mean + self.config.gds_k * gds_std
        else:
            signals["GDS_threshold"] = float("inf")

        # ==================== NUS: Neuron Utility Score ====================
        # NUS(l, i) = |w_out(i)|_2 * |w_in(i)|_2 * mean(|h_i(l,t)|)
        # For spatial layers, channels are "neurons": filter norms replace weight row norms
        nus = {}
        for l in range(model.num_layers):
            layer = model.layers[l]
            is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

            if is_spatial:
                # Spatial layer: d_l = number of output channels
                d_l = layer.out_channels

                # Outgoing weight norms: per-filter norm [C_out]
                w_out_norms = layer.conv.weight.data.flatten(1).norm(dim=1)  # [C_out]

                # Incoming weight norms from next layer
                if l < model.num_layers - 1:
                    next_layer = model.layers[l + 1]
                    if hasattr(next_layer, 'mode') and next_layer.mode == "spatial":
                        # Next is spatial: per-input-channel norm [C_in]
                        w_in_norms = next_layer.conv.weight.data.permute(1, 0, 2, 3).flatten(1).norm(dim=1)
                    else:
                        # Next is flat (across flatten boundary):
                        # one channel → H*W columns in the linear weight
                        C = layer.out_channels
                        _, H, W = layer.spatial_shape
                        hw = H * W
                        w_in_norms = torch.stack([
                            next_layer.linear.weight.data[:, c * hw:(c + 1) * hw].norm()
                            for c in range(C)
                        ])
                else:
                    # Last layer → output_head
                    C = layer.out_channels
                    use_gap = getattr(model, '_use_gap', False)
                    if use_gap:
                        # GAP mode: head is [d_out, C], per-column norm
                        w_in_norms = model.output_head.weight.data.norm(dim=0)  # [C]
                    else:
                        # Flatten mode: head is [d_out, C*H*W]
                        _, H, W = layer.spatial_shape
                        hw = H * W
                        w_in_norms = torch.stack([
                            model.output_head.weight.data[:, c * hw:(c + 1) * hw].norm()
                            for c in range(C)
                        ])
            else:
                # Flat layer: original behavior
                d_l = layer.out_features

                # Get outgoing weight norms
                w_out_norms = layer.weight.data.norm(dim=1)  # [d_l]

                # Get incoming weight norms (from next layer)
                if l < model.num_layers - 1:
                    next_layer = model.layers[l + 1]
                    w_in_norms = next_layer.weight.data.norm(dim=0)  # [d_l]
                else:
                    w_in_norms = model.output_head.weight.data.norm(dim=0)  # [d_l]

            # Get mean activation magnitudes
            if l in self.activation_magnitudes and self.activation_magnitudes[l]:
                stacked = torch.stack(self.activation_magnitudes[l])  # [T, d_l]
                mean_activations = stacked.mean(dim=0)
                # Handle dimension mismatch (can happen if surgery occurred mid-accumulation)
                if mean_activations.shape[0] != d_l:
                    mean_activations = torch.ones(d_l)
            else:
                mean_activations = torch.ones(d_l)

            # Compute NUS per neuron vectorized (avoid per-neuron .item() stalls)
            # Ensure all tensors have consistent size, pad with zeros if needed
            n_out = min(d_l, w_out_norms.shape[0])
            n_in = min(d_l, w_in_norms.shape[0])
            n_act = min(d_l, mean_activations.shape[0])
            nus_tensor = torch.zeros(d_l)
            nus_tensor[:n_out] = w_out_norms[:n_out].cpu().float()
            nus_in = torch.zeros(d_l)
            nus_in[:n_in] = w_in_norms[:n_in].cpu().float()
            nus_act = torch.zeros(d_l)
            nus_act[:n_act] = mean_activations[:n_act].cpu().float()
            nus_values = (nus_tensor * nus_in * nus_act).tolist()
            layer_nus = {i: nus_values[i] for i in range(d_l)}

            nus[l] = layer_nus

        signals["NUS"] = nus

        # NUS removal threshold: bottom k-th percentile across all neurons
        all_nus_values = []
        for l_nus in nus.values():
            all_nus_values.extend(l_nus.values())
        if all_nus_values:
            signals["NUS_threshold"] = np.percentile(
                all_nus_values, self.config.nus_percentile
            )
        else:
            signals["NUS_threshold"] = 0.0

        # ==================== LSS: Layer Saturation Score ====================
        # LSS(l) = ||h(l) - h(l-1)||_2 / ||h(l-1)||_2
        # Note: layer norms are GPU tensors (deferred from accumulate).
        lss = {}
        for l in range(model.num_layers):
            if (
                l in self.layer_transform_norms
                and l in self.layer_input_norms
                and self.layer_input_norms[l]
            ):
                t_vals = self.layer_transform_norms[l]
                i_vals = self.layer_input_norms[l]
                if isinstance(t_vals[0], torch.Tensor):
                    mean_transform = torch.stack(t_vals).mean().item()
                    mean_input = torch.stack(i_vals).mean().item()
                else:
                    mean_transform = np.mean(t_vals)
                    mean_input = np.mean(i_vals)
                lss[l] = mean_transform / max(mean_input, 1e-8)
            else:
                lss[l] = 0.0

        signals["LSS"] = lss
        signals["mean_LSS"] = np.mean(list(lss.values())) if lss else 0.0

        # ==================== LCS: Layer Contribution Score ====================
        # LCS(l) = ||f(l)(h(l-1)) - h(l-1)||_2 / ||h(l-1)||_2
        # Same as LSS for layers with matching input/output dimensions
        # Note: layer norms are GPU tensors (deferred from accumulate).
        lcs = {}
        for l in range(model.num_layers):
            if (
                l in self.layer_transform_norms
                and l in self.layer_input_norms
                and self.layer_input_norms[l]
            ):
                t_vals = self.layer_transform_norms[l]
                i_vals = self.layer_input_norms[l]
                if isinstance(t_vals[0], torch.Tensor):
                    mean_transform = torch.stack(t_vals).mean().item()
                    mean_input = torch.stack(i_vals).mean().item()
                else:
                    mean_transform = np.mean(t_vals)
                    mean_input = np.mean(i_vals)
                lcs[l] = mean_transform / max(mean_input, 1e-8)
            else:
                lcs[l] = 1.0  # Assume contributing if no data

        signals["LCS"] = lcs

        # ==================== CLGC: Cross-Layer Gradient Correlation ====================
        # CLGC(i, j) = |corr(dL/dh(i), dL/dh(j))| averaged over last T steps
        clgc = {}
        layer_indices = sorted(self.layer_gradients.keys())
        for i_idx, i in enumerate(layer_indices):
            for j in layer_indices[i_idx + 1:]:
                if self.layer_gradients[i] and self.layer_gradients[j]:
                    grads_i = self.layer_gradients[i]
                    grads_j = self.layer_gradients[j]
                    min_len = min(len(grads_i), len(grads_j))
                    if min_len == 0:
                        continue

                    correlations = []
                    for t in range(min_len):
                        gi = grads_i[t].flatten().float()
                        gj = grads_j[t].flatten().float()
                        # Pad or truncate to same length for correlation
                        min_d = min(gi.shape[0], gj.shape[0])
                        if min_d > 0:
                            gi = gi[:min_d]
                            gj = gj[:min_d]
                            gi_centered = gi - gi.mean()
                            gj_centered = gj - gj.mean()
                            denom = gi_centered.norm() * gj_centered.norm()
                            if denom > 1e-8:
                                corr = float((gi_centered @ gj_centered / denom).abs())
                                correlations.append(corr)

                    if correlations:
                        clgc[(i, j)] = np.mean(correlations)

        signals["CLGC"] = clgc

        # ==================== Graph-Specific Signals ====================
        # Only computed for graph models — zero cost for flat/spatial.

        if getattr(model, '_is_graph', False):
            # --- OSD: Oversmoothing Detector ---
            # Mean pairwise cosine similarity per layer. Values approaching 1.0
            # indicate that node features are collapsing to the same representation.
            # Note: node_similarity values are GPU tensors (deferred from accumulate).
            osd = {}
            for l in range(model.num_layers):
                if l in self.node_similarity and self.node_similarity[l]:
                    vals = self.node_similarity[l]
                    if isinstance(vals[0], torch.Tensor):
                        osd[l] = torch.stack(vals).mean().item()
                    else:
                        osd[l] = float(np.mean(vals))
                else:
                    osd[l] = 0.0
            signals["OSD"] = osd

            # --- AUS: Adjacency Utilization Score ---
            # Read gate values from graph ops to measure how much the model
            # relies on graph structure. No accumulation needed.
            from .surgery import (NeighborAggregation, GraphAttentionAggregation,
                GraphDiffusion, SpectralConv, MessagePassingGIN, GraphBranchedBlock,
                APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                DirectionalDiffusion, AdaptiveGraphConv)
            _GATED_GRAPH_OPS = (
                NeighborAggregation, GraphAttentionAggregation, GraphDiffusion,
                SpectralConv, MessagePassingGIN, GraphBranchedBlock,
                APPNPPropagation, GraphSAGEMean, GraphSAGEGCN, GATv2Aggregation,
                SGConv, DropEdgeAggregation, MixHopConv, EdgeWeightedAggregation,
                DirectionalDiffusion, AdaptiveGraphConv,
            )
            aus = {}
            for l in range(model.num_layers):
                for op in model.ops[l].operations:
                    if isinstance(op, _GATED_GRAPH_OPS) and hasattr(op, 'gate'):
                        aus[l] = float(torch.sigmoid(op.gate).item())
                        break
            signals["AUS"] = aus

            # --- HDS: Hop Demand Signal ---
            # Ratio of mean GDS on layers with graph ops vs all layers.
            # HDS > 1.5 means graph layers are under more gradient pressure.
            graph_layer_gds = [gds[l] for l in aus.keys() if l in gds]
            all_gds_vals = list(gds.values())
            if graph_layer_gds and all_gds_vals:
                mean_graph_gds = float(np.mean(graph_layer_gds))
                mean_all_gds = float(np.mean(all_gds_vals))
                hds = mean_graph_gds / max(mean_all_gds, 1e-8)
            else:
                hds = 1.0
            signals["HDS"] = hds

        # ==================== Loss plateau detection ====================
        signals["loss_plateaued"] = self._detect_loss_plateau()

        return signals

    def _detect_loss_plateau(self) -> bool:
        """Detect whether the training loss has plateaued.

        Compares the mean loss in the first half vs second half of the history.
        If the improvement is below epsilon, we consider it plateaued.
        """
        if len(self.loss_history) < self.config.loss_plateau_steps:
            return False

        losses = list(self.loss_history)
        half = len(losses) // 2
        first_half_mean = np.mean(losses[:half])
        second_half_mean = np.mean(losses[half:])

        improvement = first_half_mean - second_half_mean
        return improvement < self.config.loss_plateau_epsilon

    def reset(self):
        """Reset all accumulated statistics after a surgery step."""
        self.gradient_norms.clear()
        self.activation_magnitudes.clear()
        self.layer_input_norms.clear()
        self.layer_output_norms.clear()
        self.layer_transform_norms.clear()
        self.layer_gradients.clear()
        self.node_similarity.clear()
        self.num_accumulated = 0
        # Note: loss_history is NOT reset — it's a rolling window


class SurgeryScheduler:
    """Orchestrates the surgery decision process.

    Manages the surgery interval, warm-up period, budget enforcement,
    and convergence detection. Delegates actual surgery execution to SurgeryEngine.

    Cooldown is no longer managed here — the SurgeryAwareOptimizer handles
    post-surgery warmup internally via WSD phase management.

    Between surgeries, the network trains normally — it IS a standard neural network
    with standard backpropagation, standard compute cost, and standard memory usage.
    """

    def __init__(
        self,
        config: ASANNConfig,
        surgery_engine: SurgeryEngine,
        logger: Optional[SurgeryLogger] = None,
        n_classes: int = 0,
    ):
        self.config = config
        self.engine = surgery_engine
        self.logger = logger
        self.stats = GradientStatsAccumulator(config)

        # Current surgery interval (can be adjusted by meta-learning)
        self.surgery_interval = config.surgery_interval_init

        # Layer identity tracking for removal (consecutive intervals with low LCS)
        self.layer_identity_counts: Dict[int, int] = {}

        # Unenriched layer tracking (consecutive intervals with only trivial ops)
        self._unenriched_counts: Dict[int, int] = {}

        # Connection low-utility tracking (consecutive intervals)
        # (stored on SkipConnection objects directly)

        # Convergence detection (Fix 2: stability budget replaces zero-surgery check)
        self.consecutive_stable_intervals = 0
        self.architecture_stable = False

        # Track total surgeries for this interval
        self.interval_surgery_count = 0

        # RC3: Track last surgery step for min_post_surgery_steps
        self.last_surgery_step = 0
        self.last_layer_add_step = 0  # Layer-specific cooldown tracking

        # Fix 1: Loss-gated surgery — track recent losses
        self._loss_gate_history: deque = deque(maxlen=config.loss_gate_window)

        # Fix 2: Previous cost for stability budget
        self._prev_cost: Optional[float] = None

        # Fix 3: Track total active surgery rounds for interval growth
        self._total_active_rounds: int = 0

        # Performance guard: track EMA of loss (more robust than instantaneous min).
        self._loss_ema: float = float('inf')
        self._loss_ema_decay: float = 0.995  # Slow EMA — tracks trend, not noise
        self._best_loss_ema: float = float('inf')  # Best EMA value seen

        # ===== v2: Diagnosis + Treatment Engine =====
        self.diagnosis_engine = DiagnosisEngine(config, n_classes=n_classes)
        self.treatment_planner = TreatmentPlanner(config, surgery_engine)
        self._last_diagnosis: Optional[Diagnosis] = None

        # ===== v3: Lab Diagnostic System + Patient History =====
        self.patient_history = PatientHistory()
        self.lab: Optional[LabDiagnostics] = None
        if getattr(config, 'lab_enabled', True):
            self.lab = create_default_lab(
                max_tier=getattr(config, 'lab_max_tier', 2),
                confidence_threshold=getattr(config, 'lab_confidence_threshold', 0.7),
            )
        self._last_lab_report: Optional[LabReport] = None

        # Treatment rollback — save full model copy before Level 2+ treatments.
        # Uses deepcopy (not state_dict) so rollback works even when treatments
        # change the architecture (BatchNorm, JK, ResNet, etc. add new parameters).
        self._pre_treatment_model: Optional[ASANNModel] = None  # deepcopy of model
        self._pre_treatment_val_loss: float = float('inf')
        self._pre_treatment_epoch: int = -1
        self._rollback_treatment_type: Optional[str] = None
        self._pre_treatment_is_structural: bool = False  # Level >= 2 treatment (needs longer recovery)

        # Dose titration state — tracks dose-adaptive treatment protocol
        self._titration_active: bool = False       # True = in titration protocol
        self._titration_dose_level: int = 0        # Current dose level index (0 = full)
        self._titration_dose_factor: float = 1.0   # Capacity-based factor from compute_dose_factor
        self._titration_epoch: int = -1            # Epoch when current dose level started
        self._open_heart_attempted: bool = False   # True = open heart surgery already tried

        # Memorization tracking — prevent false positives on tiny datasets
        # and stop endless re-diagnosis loops
        self.n_train_samples: int = 0  # Set by trainer before training starts
        self._consecutive_memorization: int = 0  # Consecutive memorization diagnoses
        self._memorization_best_val: float = float('inf')  # Best val_loss when memorization active
        self._memorization_suppressed: bool = False  # True = stop diagnosing memorization

        # Stalled convergence tracking — detect when val metric stops improving
        # despite the model being "healthy" (no diseases detected).
        # This bypasses the performance gate to trigger structural treatments.
        # Uses the experiment's target metric (auroc, accuracy, r2, etc.)
        # instead of hardcoded val_acc / val_loss.
        self._stall_best_val_metric: Optional[float] = None  # Best target metric value seen
        self._stall_higher_is_better: bool = True  # Direction of target metric
        self._stall_best_epoch: int = 0          # Epoch when best val metric was achieved
        self._stall_diagnosed_count: int = 0     # How many times STALLED_CONVERGENCE was diagnosed

        # Gradient death tracking — detect when all layer grad norms collapse to ~0
        self._consecutive_grad_death: int = 0    # Consecutive evaluations with dead gradients

        # Fix 5: Iatrogenic damage tracking — detect when diseases are caused by treatments
        self._last_treatment_epoch: int = -999
        self._last_treatment_pre_metric: Optional[float] = None

        # Treatment exhaustion tracking — stop training when nothing more can help.
        # When prescribe() returns empty for N consecutive sick diagnoses, signal
        # the trainer to stop (continuing only degrades the model).
        self._consecutive_no_treatment: int = 0
        self._treatments_exhausted: bool = False

    def accumulate_step(
        self,
        model: ASANNModel,
        loss: torch.Tensor,
        layer_activations: Dict[int, torch.Tensor],
        layer_inputs: Dict[int, torch.Tensor],
        layer_outputs: Dict[int, torch.Tensor],
        hook_results: Optional[Dict] = None,
        loss_val_f: Optional[float] = None,
    ):
        """Accumulate statistics from one training step — GPU-only, zero sync.

        The GPU tensor accumulation (accumulate()) has NO .cpu() calls.
        Loss EMA tracking is done separately via update_loss_tracking()
        which the trainer calls AFTER its single GPU->CPU sync point.

        Args:
            hook_results: Pre-computed gradient summaries from backward hooks.
                When provided, avoids retain_grad entirely.
            loss_val_f: Pre-computed loss float. When provided, appended to
                loss_history and used for loss EMA. When None (deferred sync),
                the trainer will call update_loss_tracking() later.
        """
        self.stats.accumulate(
            model, loss, layer_activations, layer_inputs, layer_outputs,
            hook_results=hook_results, loss_val_f=loss_val_f,
        )
        # Loss EMA tracking is handled by update_loss_tracking() when
        # loss_val_f is deferred (called after trainer's GPU->CPU sync).
        if loss_val_f is not None:
            self._update_loss_ema(loss_val_f)

    def update_loss_tracking(self, loss_val_f: float):
        """Update loss tracking AFTER GPU->CPU sync (deferred from accumulate_step).

        Called by the trainer after its single .cpu() sync point, so the loss
        float is available without forcing an extra GPU sync.
        """
        # Append to loss_history for plateau detection
        self.stats.loss_history.append(loss_val_f)
        self._update_loss_ema(loss_val_f)

    def _update_loss_ema(self, loss_val: float):
        """Internal: update loss gate history and EMA."""
        self._loss_gate_history.append(loss_val)

        # Performance guard: track EMA of loss
        if self._loss_ema == float('inf'):
            self._loss_ema = loss_val
        else:
            self._loss_ema = self._loss_ema_decay * self._loss_ema + (1 - self._loss_ema_decay) * loss_val
        if self._loss_ema < self._best_loss_ema:
            self._best_loss_ema = self._loss_ema

    def update_stall_tracking(
        self,
        epoch: int,
        target_metric_value: Optional[float],
        target_metric_higher_is_better: bool = True,
    ):
        """Update stall convergence tracking with the latest target metric.

        Called at EVERY eval epoch by the trainer (not just surgery epochs)
        so the stall tracker sees every metric sample and doesn't miss peaks.
        The actual STALLED_CONVERGENCE diagnosis happens separately in
        execute_diagnosis_surgery() at surgery epochs.
        """
        stall_min_epoch = self.config.stalled_convergence_min_epochs
        stall_min_improve = self.config.stalled_convergence_min_improvement

        if target_metric_value is None or epoch < stall_min_epoch:
            return

        self._stall_higher_is_better = target_metric_higher_is_better

        if self._stall_best_val_metric is None:
            # First time tracking
            self._stall_best_val_metric = target_metric_value
            self._stall_best_epoch = epoch
            return

        improved = False
        if target_metric_higher_is_better:
            if target_metric_value > self._stall_best_val_metric * (1 + stall_min_improve):
                improved = True
        else:
            if target_metric_value < self._stall_best_val_metric * (1 - stall_min_improve):
                improved = True

        if improved:
            self._stall_best_val_metric = target_metric_value
            self._stall_best_epoch = epoch
            if self._stall_diagnosed_count > 0:
                self.treatment_planner.disease_escalation.pop(
                    DiseaseType.STALLED_CONVERGENCE, None)
                self.treatment_planner._tried_treatments.pop(
                    DiseaseType.STALLED_CONVERGENCE, None)
                self._stall_diagnosed_count = 0

    def _quick_screen_input(self, model, val_data, signals, diagnosis, task_type):
        """Run InputStructureTest as a fast pre-treatment screen.

        Returns a mini LabReport if the test finds a high-confidence
        recommendation (e.g. DERIVATIVE_PACKAGE for PDE-like inputs),
        or None otherwise.
        """
        from .lab_tests import InputStructureTest
        from .lab import LabReport
        test = InputStructureTest()
        if not test.is_applicable(diagnosis, task_type):
            return None
        try:
            result = test.run(model, val_data, signals, self.config,
                              task_type=task_type)
            if (result.confidence >= getattr(self.config, 'lab_confidence_threshold', 0.7)
                    and result.suggested_treatment is not None):
                report = LabReport()
                report.add_result(result)
                print(f"  [SCREEN] {result.findings.get('summary', '')}")
                return report
        except Exception:
            pass
        return None

    def _quick_screen_graph(self, model, val_data, signals, diagnosis, task_type):
        """Run GraphStructureTest as a fast pre-treatment screen.

        Analogous to _quick_screen_input for PDE detection: runs every
        diagnosis cycle to detect graph structure and recommend graph ops
        before the model wastes epochs on purely flat treatments.

        More aggressive than other screens:
        - Fires even during HEALTHY phase if model has graph data but no graph ops
        - Uses lower confidence threshold (0.6) for graph screening
        - Graph structure is deterministic knowledge, not a disease heuristic

        Returns a mini LabReport if graph structure is detected and graph
        ops are not yet present, or None otherwise.
        """
        # Quick check: does model even have graph data?
        if (getattr(model, '_graph_edge_index', None) is None
                or getattr(model, '_graph_num_nodes', 0) == 0):
            return None

        # Check if model already has graph ops
        from .surgery import get_operation_name
        has_graph_ops = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                if 'graph_' in get_operation_name(op).lower():
                    has_graph_ops = True
                    break
            if has_graph_ops:
                break

        # If already has graph ops, skip (don't keep re-screening)
        if has_graph_ops:
            return None

        from .lab_tests import GraphStructureTest
        from .lab import LabReport
        test = GraphStructureTest()

        # Override applicability check: graph data present + no graph ops = always screen
        # (don't wait for a disease diagnosis when we know graph structure exists)
        try:
            result = test.run(model, val_data, signals, self.config,
                              task_type=task_type,
                              patient_history=self.patient_history)
            # Lower confidence threshold for graph screening (0.6 instead of 0.7)
            graph_threshold = 0.6
            if (result.confidence >= graph_threshold
                    and result.suggested_treatment is not None):
                report = LabReport()
                report.add_result(result)
                print(f"  [SCREEN] {result.findings.get('summary', '')}")
                return report
        except Exception:
            pass
        return None

    # ============================= v2: Diagnosis-Based Surgery =============================

    def execute_diagnosis_surgery(
        self,
        model: ASANNModel,
        optimizer,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        val_balanced_acc: Optional[float] = None,
        x_batch: Optional[torch.Tensor] = None,
        y_batch: Optional[torch.Tensor] = None,
        loss_fn=None,
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        val_data: Optional[Any] = None,
        task_type: str = "regression",
        target_metric_name: Optional[str] = None,
        target_metric_value: Optional[float] = None,
        target_metric_higher_is_better: bool = True,
    ):
        """v3: Diagnosis-based surgery with lab referral — the 3.5-phase protocol.

        Phase 1:   Diagnosis — evaluate model health using train/val metrics
        Phase 1.5: Lab Referral — if symptoms are ambiguous, run lab tests (NEW)
        Phase 2:   Treatment — apply evidence-based treatments if SICK
        Phase 3:   Custom fine-tuning — GDS/NUS-based surgery if HEALTHY

        This method is called at surgery epochs by the trainer.
        """
        self.interval_surgery_count = 0
        self._neuron_surgery_count = 0
        self._operation_surgery_count = 0
        self._layer_surgery_count = 0
        self._connection_surgery_count = 0

        # ===== ADAPTIVE RECOVERY: DOSE TITRATION PROTOCOL =====
        # Medical-inspired treatment monitoring with dose titration:
        #  1. ratio ≤ 1.0 → ACCEPT (treatment worked, loss matched pre-treatment)
        #  2. Catastrophic (>3x) → immediately lower dose (not rollback)
        #  3. After titration_window epochs: if ratio > 1.0, lower dose
        #  4. When dose is minimal and ratio > 1.0 → open heart surgery
        #  5. If open heart surgery fails → end training
        # NO rollback to overfitting state (it was bad, don't go back).
        if self._pre_treatment_model is not None and epoch > self._pre_treatment_epoch:
            ratio = val_loss / max(abs(self._pre_treatment_val_loss), 1e-8)
            epochs_since_treatment = epoch - self._pre_treatment_epoch
            acceptable_ratio = getattr(self.config, 'recovery_acceptable_ratio', 1.0)
            catastrophic_ratio = getattr(self.config, 'recovery_catastrophic_ratio', 3.0)
            titration_window = getattr(self.config, 'dose_titration_window', 5)
            max_levels = getattr(self.config, 'dose_titration_levels', 4)

            # Initialize titration on first check
            if not self._titration_active:
                self._titration_active = True
                self._titration_dose_level = 0
                self._titration_epoch = self._pre_treatment_epoch
                self._open_heart_attempted = False
                # Get dose factor from treatment planner
                self._titration_dose_factor = getattr(
                    self.treatment_planner, '_last_dose_factor', 1.0)

            epochs_at_current_dose = epoch - self._titration_epoch
            dose_levels = [1.0 / (2 ** i) for i in range(max_levels)]
            current_dose = dose_levels[min(self._titration_dose_level, len(dose_levels) - 1)]

            should_accept = False
            should_titrate = False
            should_open_heart = False
            should_end_training = False
            reason = ""

            # Check 1: ratio ≤ 1.0 → ACCEPT (treatment worked)
            if ratio <= acceptable_ratio:
                should_accept = True
                reason = (f"dose level {self._titration_dose_level} effective: "
                          f"ratio={ratio:.3f} (<= {acceptable_ratio})")

            # Check 2: Catastrophic → immediately lower dose
            elif ratio > catastrophic_ratio and epochs_at_current_dose <= 2:
                if self._titration_dose_level < max_levels - 1:
                    should_titrate = True
                    reason = (f"catastrophic ({ratio:.1f}x) at dose level "
                              f"{self._titration_dose_level}, lowering immediately")
                else:
                    should_open_heart = True
                    reason = (f"catastrophic ({ratio:.1f}x) at minimum dose, "
                              f"open heart surgery needed")

            # Check 3: After titration window, check if dose should be lowered
            elif epochs_at_current_dose >= titration_window:
                if self._titration_dose_level < max_levels - 1:
                    should_titrate = True
                    reason = (f"ratio={ratio:.3f} after {titration_window} epochs "
                              f"at dose level {self._titration_dose_level}, lowering")
                elif not self._open_heart_attempted:
                    should_open_heart = True
                    reason = (f"ratio={ratio:.3f} at minimum dose after "
                              f"{titration_window} epochs, open heart surgery")
                else:
                    should_end_training = True
                    reason = (f"ratio={ratio:.3f} after open heart surgery failed, "
                              f"ending training")

            # Execute decision
            if should_accept:
                print(f"  [RECOVERY] Treatment {self._rollback_treatment_type} - {reason}. "
                      f"Treatment accepted.")
                post_val = val_acc if val_acc is not None else -val_loss
                for record in reversed(self.patient_history.treatment_log):
                    if record.outcome == "pending":
                        record.outcome = "effective"
                        record.post_metric = post_val
                        break
                self._clear_treatment_state()

            elif should_titrate:
                self._titration_dose_level += 1
                new_dose = dose_levels[self._titration_dose_level]
                print(f"  [TITRATION] {reason}")
                self.treatment_planner.adjust_aggressive_reg_dose(
                    model, optimizer, new_dose, self._titration_dose_factor)
                self._titration_epoch = epoch  # Reset window
                self.diagnosis_engine.notify_treatment_applied(epoch)

            elif should_open_heart:
                print(f"  [OPEN HEART] {reason}")
                self._open_heart_attempted = True
                success = self._perform_open_heart_surgery(
                    model, optimizer, step, loss_fn, val_batch)
                model.verify_output_head()
                if success:
                    # Re-evaluate ratio after surgery
                    print(f"  [OPEN HEART] Surgery successful, continuing training")
                    self._titration_epoch = epoch  # Give it one more window
                else:
                    print(f"  [OPEN HEART] Surgery failed to recover loss ratio")
                    self._treatments_exhausted = True
                    post_val = val_acc if val_acc is not None else -val_loss
                    for record in reversed(self.patient_history.treatment_log):
                        if record.outcome == "pending":
                            record.outcome = "harmful"
                            record.post_metric = post_val
                            break
                    self._clear_treatment_state()

            elif should_end_training:
                print(f"  [TERMINAL] {reason}")
                self._treatments_exhausted = True
                post_val = val_acc if val_acc is not None else -val_loss
                for record in reversed(self.patient_history.treatment_log):
                    if record.outcome == "pending":
                        record.outcome = "harmful"
                        record.post_metric = post_val
                        break
                self._clear_treatment_state()

            else:
                # Still within titration window, keep monitoring. Suppress
                # all diagnosis and treatment until resolved — otherwise a
                # new treatment would overwrite the monitoring state and the
                # treatment never gets a final verdict.
                print(f"  [RECOVERY] Treatment {self._rollback_treatment_type} - "
                      f"dose level {self._titration_dose_level}, "
                      f"{epochs_at_current_dose}/{titration_window} epochs, "
                      f"loss ratio {ratio:.2f}x. Monitoring...")
                return

        # Compute mean LSS from accumulated stats
        signals = self.stats.compute_signals(model) if self.stats.num_accumulated > 0 else {}
        mean_lss = signals.get("mean_LSS", 0.0)

        # ===== PHASE 1: DIAGNOSIS =====
        self.diagnosis_engine.record_snapshot(
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            val_balanced_acc=val_balanced_acc,
            mean_lss=mean_lss,
            num_params=sum(p.numel() for p in model.parameters()),
        )

        diagnosis = self.diagnosis_engine.diagnose(epoch)

        # --- Graph-specific: inject OVERSMOOTHING diagnosis from OSD signal ---
        # Guards against false positives:
        #   1. Warmup guard: OSD is meaningless during early training (random features
        #      naturally have high cosine similarity after aggregation).
        #   2. Depth guard: Oversmoothing requires 3+ graph aggregation layers.
        #      A 2-layer GCN cannot oversmooth — high OSD is just aggregation working.
        #   3. Higher thresholds: Only diagnose when OSD > 0.97 (not 0.85).
        osd_warmup = max(self.config.warmup_epochs + self.config.hard_warmup_epochs, 20)
        if (getattr(model, '_is_graph', False) and signals and "OSD" in signals
                and epoch >= osd_warmup and model.num_layers >= 3):
            osd = signals["OSD"]
            # Check for oversmoothing in layers beyond layer 0
            oversmoothed_layers = {l: v for l, v in osd.items() if l > 0 and v > 0.97}
            if oversmoothed_layers:
                max_osd = max(oversmoothed_layers.values())
                num_severe = sum(1 for v in oversmoothed_layers.values() if v > 0.99)
                if max_osd > 0.995:
                    severity = 3
                elif num_severe >= 2:
                    severity = 2
                else:
                    severity = 1
                from .diagnosis import Disease, DISEASE_SEVERITY
                diagnosis.diseases.append(Disease(
                    disease_type=DiseaseType.OVERSMOOTHING,
                    severity=severity,
                    evidence={
                        "max_osd": max_osd,
                        "num_oversmoothed": len(oversmoothed_layers),
                        "oversmoothed_layers": oversmoothed_layers,
                    },
                ))
                if diagnosis.state == HealthState.HEALTHY:
                    diagnosis.state = HealthState.SICK

        # --- Memorization detection: train loss near-zero + significant gap ---
        # Memorization = model stores individual training examples as lookup entries.
        # Signature: train loss → ~0 while val loss stays high.
        # Separate from overfitting (which is a broader train-val gap pattern).
        #
        # Guards:
        #   1. Training-set-size guard: skip if n_train < memorization_min_train_samples
        #      (tiny datasets like PubMed/CiteSeer naturally memorize — it's not pathological)
        #   2. Consecutive-treatment limit: after N diagnoses without val improvement,
        #      suppress further memorization diagnoses to avoid endless treatment loops.
        mem_warmup = max(self.config.warmup_epochs + self.config.hard_warmup_epochs, 30)
        mem_size_ok = (self.n_train_samples == 0  # Unknown → allow detection
                       or self.n_train_samples >= self.config.memorization_min_train_samples)
        if (epoch >= mem_warmup and train_loss is not None and val_loss is not None
                and mem_size_ok and not self._memorization_suppressed):
            gap = val_loss - train_loss
            if (train_loss < self.config.memorization_train_loss_thresh
                    and gap > self.config.memorization_gap_thresh):
                # Determine severity from gap magnitude
                if gap > 0.5:
                    severity = 3
                elif gap > 0.3:
                    severity = 2
                else:
                    severity = 1
                from .diagnosis import Disease
                # Only inject if not already diagnosed as overfitting
                has_overfit = any(d.disease_type.name.startswith("OVERFITTING")
                                 for d in diagnosis.diseases)
                if not has_overfit:
                    # --- Consecutive-treatment limit ---
                    self._consecutive_memorization += 1
                    if val_loss < self._memorization_best_val:
                        self._memorization_best_val = val_loss
                        self._consecutive_memorization = 1  # Reset — treatment helped
                    if self._consecutive_memorization > self.config.memorization_max_consecutive:
                        self._memorization_suppressed = True
                        if hasattr(self, 'logger') and self.logger:
                            print(f"  [MEMORIZATION] Suppressed after {self.config.memorization_max_consecutive} "
                                  f"consecutive diagnoses without val improvement")
                    else:
                        diagnosis.diseases.append(Disease(
                            disease_type=DiseaseType.MEMORIZATION,
                            severity=severity,
                            evidence={
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "gap": gap,
                                "consecutive": self._consecutive_memorization,
                            },
                        ))
                        if diagnosis.state == HealthState.HEALTHY:
                            diagnosis.state = HealthState.SICK
            else:
                # Conditions no longer met — reset consecutive counter
                if self._consecutive_memorization > 0:
                    self._consecutive_memorization = 0
                    self._memorization_best_val = float('inf')

        # --- Stalled convergence detection: val metric not improving for many epochs ---
        # Tracking is done at every eval epoch via update_stall_tracking().
        # Here we only CHECK if the model is stalled and inject the disease.
        #
        # Treatment ladder escalates: BATCHNORM -> JK -> RESNET_BLOCK -> CAPACITY_GROWTH
        stall_min_epoch = self.config.stalled_convergence_min_epochs
        stall_patience = self.config.stalled_convergence_patience

        # Stall detection: allow on review epochs even when stable, so a
        # stable-but-suboptimal model can be diagnosed and treated.
        allow_stall_check = (
            not self.architecture_stable
            or epoch % self.config.stability_review_epochs == 0
        )

        # Metric-saturation guard: if the target metric is already at its
        # theoretical maximum (1.0 for accuracy/F1/etc.), the model cannot
        # improve further — this is NOT stalling, it's metric saturation.
        _metric_saturated = (
            self._stall_best_val_metric is not None
            and self._stall_higher_is_better
            and self._stall_best_val_metric >= 1.0 - 1e-6
        )

        if (epoch >= stall_min_epoch
                and diagnosis.state == HealthState.HEALTHY
                and allow_stall_check
                and self._stall_best_val_metric is not None
                and not _metric_saturated):
            epochs_since_improvement = epoch - self._stall_best_epoch
            if epochs_since_improvement >= stall_patience:
                # Val trajectory guard — distinguish "stalled" (flat) from
                # "declining" (getting worse).
                diag_signals = getattr(self.diagnosis_engine, 'last_signals', {})
                if self._stall_higher_is_better:
                    val_trend = diag_signals.get("val_acc_trend", 0)
                    val_declining = val_trend < -self.config.stagnation_threshold
                else:
                    val_trend = diag_signals.get("val_loss_trend", 0)
                    val_declining = val_trend > self.config.stagnation_threshold

                if not val_declining:
                    if epochs_since_improvement >= stall_patience * 3:
                        severity = 3
                    elif epochs_since_improvement >= stall_patience * 2:
                        severity = 2
                    else:
                        severity = 1

                    from .diagnosis import Disease
                    has_existing = len(diagnosis.diseases) > 0
                    # Respect disabled_diseases from config
                    _disabled = getattr(self.config, 'disabled_diseases', ())
                    _stall_disabled = 'STALLED_CONVERGENCE' in _disabled
                    if not has_existing and not _stall_disabled:
                        self._stall_diagnosed_count += 1
                        diagnosis.diseases.append(Disease(
                            disease_type=DiseaseType.STALLED_CONVERGENCE,
                            severity=severity,
                            evidence={
                                "epochs_since_improvement": epochs_since_improvement,
                                "target_metric": target_metric_name or "val_loss",
                                "best_val_metric": self._stall_best_val_metric,
                                "best_epoch": self._stall_best_epoch,
                                "diagnosed_count": self._stall_diagnosed_count,
                            },
                        ))
                        if diagnosis.state == HealthState.HEALTHY:
                            diagnosis.state = HealthState.SICK

        # --- Gradient death detection: all layer grad norms collapse to ~0 ---
        # When weight norms explode and activations saturate (e.g., after
        # removing ReLU leaving only Tanh), gradients vanish completely.
        # The model outputs constant predictions and is effectively dead.
        # Detection: check if mean grad_norm across all layers is near zero.
        if self.stats.num_accumulated > 0 and epoch >= self.config.warmup_epochs:
            all_grad_norms = []
            for l in range(model.num_layers):
                if l in self.stats.gradient_norms and self.stats.gradient_norms[l]:
                    vals = self.stats.gradient_norms[l]
                    if vals and isinstance(vals[0], torch.Tensor):
                        all_grad_norms.append(torch.stack(vals).mean().item())
                    else:
                        all_grad_norms.append(float(np.mean(vals)))

            if all_grad_norms:
                max_grad_norm = max(all_grad_norms)
                # Gradient death: max grad norm across all layers is essentially zero
                if max_grad_norm < 1e-7:
                    self._consecutive_grad_death += 1
                else:
                    self._consecutive_grad_death = 0

                # Diagnose after 2 consecutive evaluations with dead gradients
                # to avoid false positives from a single bad batch
                if self._consecutive_grad_death >= 2:
                    from .diagnosis import Disease
                    has_grad_death = any(d.disease_type == DiseaseType.GRADIENT_DEATH
                                        for d in diagnosis.diseases)
                    if not has_grad_death:
                        diagnosis.diseases.append(Disease(
                            disease_type=DiseaseType.GRADIENT_DEATH,
                            severity=4,  # Emergency — model is dead
                            evidence={
                                "max_grad_norm": max_grad_norm,
                                "per_layer_grad_norms": {l: all_grad_norms[i]
                                                        for i, l in enumerate(range(model.num_layers))
                                                        if i < len(all_grad_norms)},
                                "consecutive_dead": self._consecutive_grad_death,
                            },
                        ))
                        if diagnosis.state == HealthState.HEALTHY:
                            diagnosis.state = HealthState.SICK

        self._last_diagnosis = diagnosis

        # Record diagnosis in patient history
        self._last_lab_report = None
        was_referred = False

        # Log the diagnosis
        if self.logger:
            self.logger.log_diagnosis(
                step, epoch, diagnosis,
                self.diagnosis_engine.consecutive_healthy,
            )

        # ===== HEALTH-BASED STABILITY =====
        if diagnosis.state == HealthState.RECOVERING:
            # Still recovering from treatment — skip new diagnosis but the adaptive
            # rollback check above already ran and will catch catastrophic failures.
            print(f"  [DIAGNOSIS] Epoch {epoch}: RECOVERING (treatment recovery period)")
            self.patient_history.record_diagnosis(epoch, diagnosis)
            self.stats.reset()
            return

        if diagnosis.is_healthy:
            # Reset treatment escalation so future diseases start fresh
            self.treatment_planner.reset_escalation_on_healthy()

            print(f"  [DIAGNOSIS] Epoch {epoch}: HEALTHY "
                  f"(consecutive: {self.diagnosis_engine.consecutive_healthy}"
                  f"/{self.config.stability_healthy_epochs})")

            # Evaluate pending treatment outcomes now that model is healthy.
            # Fix 2: Use surgery_warmup_epochs as min delay — gate must fully ramp
            # before we can judge whether a treatment helped.
            _warmup_delay = getattr(self.config, 'surgery_warmup_epochs', 10)
            if val_acc is not None:
                self.patient_history.evaluate_treatment(
                    val_acc, current_epoch=epoch, min_delay=_warmup_delay)
            else:
                self.patient_history.evaluate_treatment(
                    -val_loss, current_epoch=epoch, min_delay=_warmup_delay)

            # Check if architecture should become stable
            if self.diagnosis_engine.is_architecture_stable():
                if not self.architecture_stable:
                    self.architecture_stable = True
                    model.architecture_stable = True
                    self.patient_history.record_stability_event(
                        epoch, "stabilized", "health_based",
                        metric_value=val_loss,
                    )
                    if self.logger:
                        self.logger.log_surgery(step, "architecture_stabilized", {
                            "reason": "health_based",
                            "consecutive_healthy": self.diagnosis_engine.consecutive_healthy,
                            "final_cost": model.compute_architecture_cost(),
                        })
                    print(f"  [STABLE] Architecture stabilized at epoch {epoch} - model is healthy")

            # ===== PHASE 3: CUSTOM FINE-TUNING =====
            if signals and x_batch is not None:
                if not self.architecture_stable:
                    # Normal: run custom surgery every evaluation epoch
                    self._do_custom_surgery(
                        model, optimizer, signals, step,
                        x_batch, y_batch, loss_fn, val_batch,
                    )
                elif epoch % self.config.stability_review_epochs == 0:
                    # Periodic architecture review even when stable
                    self._do_custom_surgery(
                        model, optimizer, signals, step,
                        x_batch, y_batch, loss_fn, val_batch,
                    )

        elif diagnosis.is_sick:
            # Print detected diseases
            disease_names = [d.disease_type.name for d in diagnosis.diseases]
            worst = diagnosis.worst_severity
            print(f"  [DIAGNOSIS] Epoch {epoch}: SICK - {disease_names} (worst severity: {worst})")

            # Evaluate pending treatments even when sick — this closes the
            # feedback loop so PatientHistory knows whether past treatments
            # helped or not.
            # Fix 2: Use surgery_warmup_epochs as min delay for gate ramp.
            _warmup_delay = getattr(self.config, 'surgery_warmup_epochs', 10)
            if val_acc is not None:
                self.patient_history.evaluate_treatment(
                    val_acc, current_epoch=epoch, min_delay=_warmup_delay)
            else:
                self.patient_history.evaluate_treatment(
                    -val_loss, current_epoch=epoch, min_delay=_warmup_delay)

            # Fix 5: Iatrogenic damage detection — if within gate ramp window of
            # last treatment AND metrics worsened, suppress new treatments and let
            # the gate finish ramping. This prevents cascading damage where a
            # treatment crash is diagnosed as UNDERFITTING, triggering MORE
            # destructive treatments (PubMed: JK crash → UNDERFITTING_SEVERE → more ops → collapse).
            warmup_epochs = getattr(self.config, 'surgery_warmup_epochs', 10)
            epochs_since_treatment = epoch - self._last_treatment_epoch
            if (epochs_since_treatment < warmup_epochs
                    and self._last_treatment_pre_metric is not None):
                current_metric = val_acc if val_acc is not None else -val_loss
                if current_metric < self._last_treatment_pre_metric * 0.95:
                    print(f"  [IATROGENIC] Suppressing treatment - metrics worsened after "
                          f"recent treatment (epoch {self._last_treatment_epoch}), "
                          f"gate still ramping ({epochs_since_treatment}/{warmup_epochs})")
                    # Still record diagnosis but skip treatment phases entirely
                    self.patient_history.record_diagnosis(epoch, diagnosis, was_referred=False)
                    self.stats.reset()
                    return

            # Break stability if sick post-stable
            if self.architecture_stable and self.config.stability_can_break:
                self.architecture_stable = False
                model.architecture_stable = False
                self.patient_history.record_stability_event(
                    epoch, "broken", f"sick: {disease_names}",
                    metric_value=val_loss,
                )
                print(f"  [STABILITY BROKEN] Architecture stability revoked for treatment")

            # ===== PHASE 1.25: QUICK INPUT SCREENING =====
            # Before full lab referral, run a fast input structure check.
            # This detects PDE-like problems (low-dim continuous inputs)
            # and steers treatment away from harmful defaults (batchnorm)
            # toward appropriate ops (derivatives, polynomials).
            quick_report = None
            if (val_data is not None
                    and task_type == "regression"
                    and getattr(self.config, 'lab_enabled', True)):
                quick_report = self._quick_screen_input(
                    model, val_data, signals, diagnosis, task_type)

            # ===== PHASE 1.26: QUICK GRAPH SCREENING =====
            # If model has graph auxiliary data, run GraphStructureTest to
            # detect graph structure and recommend graph ops early.
            # Graph aggregation acts as structural regularization, so this
            # applies to both underfitting AND overfitting scenarios.
            if (quick_report is None
                    and val_data is not None
                    and getattr(self.config, 'lab_enabled', True)):
                quick_report = self._quick_screen_graph(
                    model, val_data, signals, diagnosis, task_type)

            # ===== PHASE 1.5: LAB REFERRAL =====
            lab_report = None
            if (self.lab is not None
                    and getattr(self.config, 'lab_enabled', True)
                    and val_data is not None
                    and self.patient_history.should_refer_to_lab(
                        diagnosis,
                        referral_recurrence=getattr(self.config, 'lab_referral_recurrence', 2),
                        referral_on_ambiguous=getattr(self.config, 'lab_referral_on_ambiguous', True),
                    )):
                was_referred = True
                print(f"  [LAB] Referring to lab for differential diagnosis...")

                lab_report = self.lab.run_tests(
                    model=model,
                    diagnosis=diagnosis,
                    val_data=val_data,
                    signals=signals,
                    config=self.config,
                    task_type=task_type,
                    patient_history=self.patient_history,
                )
                self._last_lab_report = lab_report

                if lab_report.results:
                    print(f"  [LAB] {len(lab_report.results)} tests completed. "
                          f"Primary finding: {lab_report.primary_finding}")
                    if lab_report.recommended_treatments:
                        rec_names = [t.name for t in lab_report.recommended_treatments[:3]]
                        print(f"  [LAB] Recommended treatments: {rec_names} "
                              f"(confidence: {lab_report.confidence:.2f})")

                    # Record lab results in patient history
                    self.patient_history.record_lab_result(epoch, lab_report)

            # ===== PHASE 2: TREATMENT =====
            # Determine current primary metric for treatment evaluation
            if val_acc is not None:
                current_val_metric = val_acc
            else:
                current_val_metric = -val_loss  # Negate so higher = better

            # Use full lab report if available, otherwise quick screen report
            effective_report = lab_report if lab_report is not None else quick_report

            # Propagate dataset size to treatment planner for dose scaling
            self.treatment_planner.n_train_samples = self.n_train_samples

            # If effective report has high-confidence finding, use its recommendation
            if (effective_report is not None
                    and effective_report.confidence >= getattr(self.config, 'lab_confidence_threshold', 0.7)
                    and effective_report.recommended_treatments):
                treatments = self.treatment_planner.prescribe(
                    diagnosis, model, lab_report=effective_report,
                    patient_history=self.patient_history)
            else:
                treatments = self.treatment_planner.prescribe(
                    diagnosis, model,
                    patient_history=self.patient_history)

            if treatments:
                # Fix 3: Save model state before Level 2+ treatments for rollback.
                # Level 2-3 treatments (JK, GRAPH_NORM, RESNET_BLOCK) can be just
                # as destructive as Level 4+ (e.g. JK crashed PubMed 0.658→0.426).
                # The continuous rollback monitoring (catastrophic/harmful/acceptable
                # ratios) already handles evaluation correctly for any level.
                max_level = max(t.level for t in treatments)
                is_structural = (max_level >= 2)
                if is_structural:
                    try:
                        # deepcopy preserves full architecture + weights, so rollback
                        # works even when treatments add new ops/layers/parameters.
                        self._pre_treatment_model = copy.deepcopy(model)
                        self._pre_treatment_val_loss = val_loss
                        self._pre_treatment_epoch = epoch
                        self._pre_treatment_is_structural = True
                        self._rollback_treatment_type = treatments[0].treatment_type.name
                        print(f"  [ROLLBACK] Saved pre-treatment model for potential rollback "
                              f"(val_loss={val_loss:.4f})")
                    except Exception as e:
                        print(f"  [ROLLBACK] Could not save pre-treatment model: {e}")
                        self._pre_treatment_model = None
                        self._pre_treatment_is_structural = False

                for treatment in treatments:
                    print(f"  [TREATMENT] Attempting: {treatment.treatment_type.name} "
                          f"(Level {treatment.level}) for {treatment.target_disease.name}")

                # Cache probe data for stride probing during layer addition
                if x_batch is not None and y_batch is not None and loss_fn is not None:
                    self.engine._stride_probe_data = (x_batch, y_batch, loss_fn)

                applied, applied_treatments = self.treatment_planner.apply_treatments(
                    treatments, model, optimizer, step, epoch,
                    current_val_metric=current_val_metric,
                    loss_fn=loss_fn,
                )

                # Clear probe data cache
                self.engine._stride_probe_data = None

                # Log only treatments that actually succeeded
                for treatment in applied_treatments:
                    print(f"  [TREATMENT] Applied: {treatment.treatment_type.name} "
                          f"(Level {treatment.level}) for {treatment.target_disease.name}")
                    if self.logger:
                        self.logger.log_treatment(step, epoch, treatment)

                if applied > 0:
                    self.interval_surgery_count += applied
                    # Reset exhaustion state — new treatment gets a fair chance
                    self._treatments_exhausted = False
                    self._consecutive_no_treatment = 0
                    # Safety: verify output_head matches last layer after treatment
                    model.verify_output_head()
                    # Compute adaptive recovery duration based on treatment type
                    base_rec = getattr(self.config, 'min_recovery_epochs',
                                       getattr(self.config, 'recovery_epochs', 3))
                    if is_structural:
                        multiplier = getattr(
                            self.config, 'structural_recovery_multiplier', 2.5)
                        recovery_duration = int(base_rec * multiplier)
                    else:
                        recovery_duration = base_rec
                    self.diagnosis_engine.notify_treatment_applied(
                        epoch, recovery_duration=recovery_duration)
                    # Fix 5: Track treatment timing for iatrogenic detection
                    self._last_treatment_epoch = epoch
                    self._last_treatment_pre_metric = current_val_metric
                    # Record only successful treatments in patient history
                    for treatment in applied_treatments:
                        self.patient_history.record_treatment(
                            epoch, treatment.target_disease,
                            treatment.treatment_type,
                            pre_metric=current_val_metric,
                            escalation_level=treatment.level,
                        )
                    # Reset stall patience only if a STALLED_CONVERGENCE
                    # treatment was applied. Unrelated treatments (e.g.,
                    # CAPACITY_EXHAUSTION) shouldn't restart the stall clock —
                    # the recovery period already pauses stall checking.
                    if any(t.target_disease == DiseaseType.STALLED_CONVERGENCE
                           for t in applied_treatments):
                        self._stall_best_epoch = epoch
                    struct_tag = " (structural)" if is_structural else ""
                    print(f"  [TREATMENT] Applied {applied} treatments. Recovery period: "
                          f"{recovery_duration} epochs{struct_tag}")
                else:
                    # Fix 6: Treatments were prescribed but none could physically
                    # be applied (e.g., DROPOUT_HEAVY when all layers already have
                    # dropout). Still set a recovery period to prevent an infinite
                    # diagnosis→treatment→fail→diagnosis loop every 2 epochs.
                    # Without this, the model gets re-diagnosed immediately, the
                    # fallback loop re-selects the same inapplicable treatment,
                    # and the cycle repeats (160+ DROPOUT_HEAVY in one experiment).
                    self.diagnosis_engine.notify_treatment_applied(epoch)
                    print(f"  [TREATMENT] Prescribed {len(treatments)} treatments but none "
                          f"took effect (already applied). Recovery period: "
                          f"{self.config.recovery_epochs} epochs")
            else:
                self._consecutive_no_treatment += 1
                exhaustion_patience = getattr(self.config, 'treatment_exhaustion_patience', 3)
                print(f"  [TREATMENT] No applicable treatments found "
                      f"(all tried or conflicting) "
                      f"[{self._consecutive_no_treatment}/{exhaustion_patience}]")
                if self._consecutive_no_treatment >= exhaustion_patience:
                    self._treatments_exhausted = True
                    print(f"  [TREATMENT] Treatments EXHAUSTED after "
                          f"{self._consecutive_no_treatment} consecutive failures. "
                          f"Signaling early stop.")

        # Record diagnosis in patient history (after lab referral decision is known)
        self.patient_history.record_diagnosis(epoch, diagnosis, was_referred=was_referred)

        # Evaluate past treatment outcomes
        if val_acc is not None:
            self.treatment_planner.evaluate_past_treatments(val_acc, epoch)
        else:
            self.treatment_planner.evaluate_past_treatments(-val_loss, epoch)

        # Prune unenriched layers (runs unconditionally, regardless of health state)
        self._prune_useless_layers(model, optimizer, signals, step)

        # Validate connection projections (may be stale after multi-type surgery)
        self._validate_connection_projections(model)

        # Log architecture snapshot
        if self.logger:
            self.logger.log_architecture_snapshot(step, model.describe_architecture())

        # Reset accumulated statistics
        self.stats.reset()

    def _do_custom_surgery(
        self,
        model: ASANNModel,
        optimizer,
        signals: Dict[str, Any],
        step: int,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn,
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Custom fine-tuning using GDS/NUS/CLGC signals (Phase 3).

        Only runs when model is HEALTHY. Uses existing neuron/connection surgery.
        """
        if not signals:
            return

        # Log signals
        if self.logger:
            self.logger.log_surgery_signals(step, {
                "GDS": signals.get("GDS", {}),
                "GDS_threshold": signals.get("GDS_threshold", 0),
                "NUS_threshold": signals.get("NUS_threshold", 0),
                "mean_LSS": signals.get("mean_LSS", 0),
                "loss_plateaued": signals.get("loss_plateaued", False),
                "LCS": signals.get("LCS", {}),
                "CLGC": signals.get("CLGC", {}),
                "num_connections": len(model.connections),
                "num_layers": model.num_layers,
            })

        # Budget enforcement
        cost = model.compute_architecture_cost()
        budget_factor = cost / max(self.config.complexity_target, 1.0)
        add_threshold_multiplier = max(0.5, min(2.0, budget_factor))
        remove_threshold_multiplier = 1.0 / add_threshold_multiplier
        block_additions = self._is_over_complexity_ceiling(cost)

        # Neuron surgery (GDS-based addition, NUS-based pruning)
        self._do_neuron_surgery(
            model, optimizer, signals, step,
            add_threshold_multiplier, remove_threshold_multiplier,
            block_additions=block_additions,
        )

        # Layer surgery (depth modification via LSS)
        # Only allow layer addition from HEALTHY if consecutively healthy enough
        consecutive_healthy = self.diagnosis_engine.consecutive_healthy
        if consecutive_healthy >= self.config.min_healthy_for_layer_add:
            # Cache probe data for stride probing during layer addition
            if x_batch is not None and y_batch is not None and loss_fn is not None:
                self.engine._stride_probe_data = (x_batch, y_batch, loss_fn)
            self._do_layer_surgery(
                model, optimizer, signals, step,
                add_threshold_multiplier,
                block_additions=block_additions,
                from_healthy_diagnosis=True,
            )
            self.engine._stride_probe_data = None  # Clear cache

        # Operation surgery (probing)
        if x_batch is not None and loss_fn is not None:
            current_loss = float(self.stats.loss_history[-1]) if self.stats.loss_history else 0.0
            self._do_operation_surgery(
                model, optimizer, step, x_batch, y_batch, loss_fn, current_loss,
                val_batch=val_batch,
            )

        # Connection surgery (CLGC-based)
        self._do_connection_surgery(
            model, optimizer, signals, step,
            block_additions=block_additions,
        )

        # Track surgery count
        if self.interval_surgery_count > 0:
            self.last_surgery_step = step
            self._total_active_rounds += 1

    # ============================= Fix helper methods =============================

    def _loss_is_still_improving(self) -> bool:
        """Fix 1: Check if loss is still improving (surgery should be blocked).

        Compares the mean of the first half vs second half of the loss gate window.
        If improvement exceeds loss_gate_min_improvement (relative), loss is improving.
        """
        window = self.config.loss_gate_window
        if len(self._loss_gate_history) < window:
            return False  # Not enough data — allow surgery

        losses = list(self._loss_gate_history)
        half = len(losses) // 2
        first_half = np.mean(losses[:half])
        second_half = np.mean(losses[half:])

        if first_half < 1e-8:
            return False  # Avoid division by zero

        relative_improvement = (first_half - second_half) / abs(first_half)
        return relative_improvement > self.config.loss_gate_min_improvement

    def _is_over_complexity_ceiling(self, cost: float) -> bool:
        """Fix 4: Check if architecture cost exceeds the hard complexity ceiling."""
        ceiling = self.config.complexity_target * self.config.complexity_ceiling_mult
        return cost >= ceiling

    def _layer_exceeds_width_budget(self, model: ASANNModel, layer_idx: int) -> bool:
        """Fix 6: Check if a layer exceeds the per-layer width budget.

        A layer's width should not exceed max_layer_width_ratio × median width.
        This prevents any single layer from becoming a "neuron sink".
        """
        if model.num_layers < 2:
            return False

        widths = [model.get_layer_width(l) for l in range(model.num_layers)]
        median_width = float(np.median(widths))
        max_allowed = median_width * self.config.max_layer_width_ratio

        current_width = model.get_layer_width(layer_idx)
        return current_width >= max_allowed

    def _update_convergence(self, model: ASANNModel, step: int):
        """Fix 2: Convergence detection with stability budget.

        Instead of requiring ZERO surgeries for K consecutive intervals,
        check if the architecture cost changed by less than stability_cost_threshold.
        Small oscillations (add+remove churn with <2% net cost change) count as "stable".
        """
        cost = model.compute_architecture_cost()

        if self._prev_cost is not None:
            if self._prev_cost > 0:
                relative_change = abs(cost - self._prev_cost) / self._prev_cost
            else:
                relative_change = 0.0 if cost == 0 else 1.0

            # "Stable" if cost changed less than threshold OR no surgery happened
            is_stable = (
                relative_change < self.config.stability_cost_threshold
                or self.interval_surgery_count == 0
            )
        else:
            # First interval — always stable (no change yet)
            is_stable = True

        self._prev_cost = cost

        if is_stable:
            self.consecutive_stable_intervals += 1
            if self.consecutive_stable_intervals >= self.config.convergence_intervals:
                self.architecture_stable = True
                model.architecture_stable = True
                if self.logger:
                    self.logger.log_surgery(step, "architecture_stabilized", {
                        "consecutive_stable_intervals": self.consecutive_stable_intervals,
                        "final_cost": cost,
                    })
        else:
            self.consecutive_stable_intervals = 0

    def _shift_identity_counts(self, at: int, direction: str = "remove"):
        """Shift layer_identity_counts indices after layer insertion/removal.

        Instead of clearing all counts (which loses removal tracking),
        shift indices to match the new layer numbering.
        """
        old_counts = dict(self.layer_identity_counts)
        self.layer_identity_counts.clear()
        for l, count in old_counts.items():
            if direction == "remove":
                if l < at:
                    self.layer_identity_counts[l] = count
                elif l > at:
                    self.layer_identity_counts[l - 1] = count
                # l == at: dropped (layer removed)
            elif direction == "insert":
                if l < at:
                    self.layer_identity_counts[l] = count
                else:
                    self.layer_identity_counts[l + 1] = count

    def _shift_unenriched_counts(self, at: int, direction: str = "remove"):
        """Shift _unenriched_counts indices after layer insertion/removal.

        Same pattern as _shift_identity_counts.
        """
        old_counts = dict(self._unenriched_counts)
        self._unenriched_counts.clear()
        for l, count in old_counts.items():
            if direction == "remove":
                if l < at:
                    self._unenriched_counts[l] = count
                elif l > at:
                    self._unenriched_counts[l - 1] = count
            elif direction == "insert":
                if l < at:
                    self._unenriched_counts[l] = count
                else:
                    self._unenriched_counts[l + 1] = count

    def _prune_useless_layers(
        self,
        model: ASANNModel,
        optimizer,
        signals: Dict[str, Any],
        step: int,
    ):
        """Prune layers whose ops pipeline has only trivial operations.

        A layer is 'useless' if model.ops[l] contains only activations,
        norms, or regularization (see _TRIVIAL_OPS).  The Conv2d/BN backbone
        is not counted -- without real discovered ops, the layer adds depth
        without adding capacity.

        Uses separate _unenriched_counts tracker with configurable threshold
        (unenriched_removal_intervals, default 5).
        """
        layers_removed = 0
        trivial_ops = _TRIVIAL_OPS
        removal_threshold = self.config.unenriched_removal_intervals

        for l in list(range(model.num_layers)):
            if l >= model.num_layers:
                break
            if model.num_layers <= self.config.min_depth:
                break
            if layers_removed >= self.config.max_layers_remove_per_surgery:
                break

            # Check ops from model.ops[l] (NOT layer.ops which doesn't exist)
            all_trivial = False
            if l < len(model.ops):
                pipeline = model.ops[l]
                if hasattr(pipeline, 'operations'):
                    if len(pipeline.operations) == 0:
                        all_trivial = True
                    else:
                        all_trivial = all(
                            type(
                                op.operation if isinstance(op, GatedOperation) else op
                            ).__name__ in trivial_ops
                            for op in pipeline.operations
                        )

            if all_trivial:
                count = self._unenriched_counts.get(l, 0) + 1
                self._unenriched_counts[l] = count

                if count >= removal_threshold:
                    success = self.engine.remove_layer(model, l, optimizer, step)
                    if success:
                        layers_removed += 1
                        self.interval_surgery_count += 1
                        self._shift_identity_counts(l, direction="remove")
                        self._shift_unenriched_counts(l, direction="remove")
                        self.engine.shift_removal_history(l, direction="remove")
                        if self.logger:
                            self.logger.log_surgery(step, "prune_useless_layer", {
                                "layer": l,
                                "reason": "only_trivial_ops",
                                "unenriched_intervals": count,
                            })
                        break  # Indices shifted, stop
            else:
                # Layer has real ops now -- reset its unenriched count
                if l in self._unenriched_counts:
                    self._unenriched_counts[l] = 0

    # ============================= Surgery sub-methods =============================

    def _do_neuron_surgery(
        self,
        model: ASANNModel,
        optimizer,
        signals: Dict[str, Any],
        step: int,
        add_mult: float,
        remove_mult: float,
        block_additions: bool = False,
    ):
        """Execute neuron addition and removal."""
        gds = signals.get("GDS", {})
        gds_threshold = signals.get("GDS_threshold", float("inf")) * add_mult
        nus = signals.get("NUS", {})
        nus_threshold = signals.get("NUS_threshold", 0.0) * remove_mult

        # Per-type and global surgery caps
        neuron_cap = self.config.max_neuron_surgeries_per_interval
        global_cap = self.config.max_total_surgeries_per_interval

        neurons_added = 0
        neurons_removed = 0
        layers_that_grew = set()  # Track layers that received additions

        # Detect if model has spatial layers (for bulk channel addition)
        is_spatial_model = self.config.spatial_shape is not None

        # === PASS 1: Add neurons (blocked by Fix 4 ceiling and Fix 6 width budget) ===
        if not block_additions:
            for l in range(model.num_layers):
                if l in gds and gds[l] > gds_threshold:
                    # Fix 6: Skip if this layer already exceeds width budget
                    if self._layer_exceeds_width_budget(model, l):
                        continue

                    # Global caps check
                    if neurons_added >= self.config.max_neurons_add_per_surgery:
                        break
                    if self._neuron_surgery_count >= neuron_cap:
                        break
                    if self.interval_surgery_count >= global_cap:
                        break

                    layer = model.layers[l]
                    is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

                    if is_spatial and self.config.per_layer_channel_budget:
                        # === Width-proportional + GDS-scaled growth ===
                        current_width = layer.out_channels
                        max_width = self.config.max_channels

                        # Base growth: fraction of current width
                        base_add = max(
                            self.config.min_channels_add_per_qualifying_layer,
                            int(current_width * self.config.channel_growth_fraction),
                        )

                        # XGBoost-style: scale by GDS/threshold ratio
                        if self.config.channel_growth_gds_scale:
                            gds_ratio = gds[l] / max(gds_threshold, 1e-8)
                            gds_ratio = min(3.0, max(1.0, gds_ratio))
                            base_add = int(base_add * gds_ratio)

                        # Hard cap: don't exceed max_fraction of current width
                        max_add = int(current_width * self.config.channel_growth_max_fraction)
                        num_to_add = min(base_add, max_add)

                        # Don't exceed max_channels
                        num_to_add = min(num_to_add, max_width - current_width)

                        # Don't exceed remaining global budgets
                        num_to_add = min(num_to_add, self.config.max_neurons_add_per_surgery - neurons_added)
                        num_to_add = min(num_to_add, self.config.max_channels_add_per_surgery - neurons_added)

                        num_to_add = max(num_to_add, 0)

                        # Snap to channel alignment (Tensor Core efficiency)
                        if is_spatial and self.config.channel_alignment > 1:
                            from .config import snap_add_for_alignment
                            num_to_add = snap_add_for_alignment(
                                current_width, num_to_add, self.config.channel_alignment)
                            # Re-apply caps after snapping (alignment may increase count)
                            num_to_add = min(num_to_add, max_width - current_width)
                            num_to_add = min(num_to_add, self.config.max_neurons_add_per_surgery - neurons_added)
                            num_to_add = max(num_to_add, 0)
                    else:
                        # === Flat layers: width-proportional + GDS-scaled growth ===
                        current_width = model.get_layer_width(l)
                        base_add = max(
                            1,
                            int(current_width * self.config.channel_growth_fraction),
                        )
                        # Scale by GDS/threshold ratio
                        if self.config.channel_growth_gds_scale:
                            gds_ratio = gds[l] / max(gds_threshold, 1e-8)
                            gds_ratio = min(3.0, max(1.0, gds_ratio))
                            base_add = int(base_add * gds_ratio)
                        # Cap
                        max_add = int(current_width * self.config.channel_growth_max_fraction)
                        num_to_add = min(base_add, max_add)
                        num_to_add = min(num_to_add, self.config.max_neurons_add_per_surgery - neurons_added)

                        # Snap to channel alignment (same as spatial)
                        if self.config.channel_alignment > 1:
                            from .config import snap_add_for_alignment
                            num_to_add = snap_add_for_alignment(
                                current_width, num_to_add, self.config.channel_alignment)
                            num_to_add = min(num_to_add, self.config.max_neurons_add_per_surgery - neurons_added)
                            num_to_add = max(num_to_add, 0)

                    for _ in range(num_to_add):
                        if neurons_added >= self.config.max_neurons_add_per_surgery:
                            break
                        if self._neuron_surgery_count >= neuron_cap:
                            break
                        if self.interval_surgery_count >= global_cap:
                            break
                        # Fix 6: Re-check width budget after each addition
                        if self._layer_exceeds_width_budget(model, l):
                            break
                        success = self.engine.add_neuron(model, l, optimizer, step)
                        if success:
                            neurons_added += 1
                            self._neuron_surgery_count += 1
                            self.interval_surgery_count += 1
                            layers_that_grew.add(l)

        # === PASS 2: Remove neurons ===
        # Sisyphus fix: skip removal on any layer that just received a channel addition.
        # Without this, add+remove on the same layer creates a zero-sum churn loop.
        for l in range(model.num_layers):
            if l in layers_that_grew:
                continue  # Grace period: don't remove from layers that just grew

            if l in nus:
                layer = model.layers[l]
                is_spatial = hasattr(layer, 'mode') and layer.mode == "spatial"

                # Collect neurons below threshold, sort by utility (lowest first)
                removable = [
                    (i, score) for i, score in nus[l].items()
                    if score < nus_threshold and i < model.get_layer_width(l)
                ]
                removable.sort(key=lambda x: x[1])

                if is_spatial and self.config.per_layer_channel_budget:
                    # Width-proportional removal: allow removing a fraction per layer
                    current_width = layer.out_channels
                    min_channels = max(self.config.d_min,
                                       getattr(self.config, 'min_spatial_channels', 8))
                    max_remove_this_layer = min(
                        len(removable),
                        int(current_width * self.config.channel_removal_fraction),
                        int(current_width * self.config.channel_removal_max_fraction),
                        current_width - min_channels,
                        self.config.max_neurons_remove_per_surgery - neurons_removed,
                    )
                    max_remove_this_layer = max(0, max_remove_this_layer)

                    # Snap to channel alignment (Tensor Core efficiency)
                    if self.config.channel_alignment > 1:
                        from .config import snap_remove_for_alignment
                        max_remove_this_layer = snap_remove_for_alignment(
                            current_width, max_remove_this_layer,
                            self.config.channel_alignment, min_channels)

                    # Sort by index DESCENDING to avoid index invalidation
                    removable_batch = removable[:max_remove_this_layer]
                    removable_batch.sort(key=lambda x: x[0], reverse=True)
                else:
                    # Flat layers or legacy: up to remaining global budget, 1 per layer
                    max_remove_this_layer = self.config.max_neurons_remove_per_surgery - neurons_removed
                    removable_batch = removable[:max_remove_this_layer]
                    # Sort by index DESCENDING to avoid index invalidation
                    removable_batch.sort(key=lambda x: x[0], reverse=True)

                removed_in_layer = 0
                for neuron_idx, _ in removable_batch:
                    if removed_in_layer >= max_remove_this_layer:
                        break
                    if neurons_removed >= self.config.max_neurons_remove_per_surgery:
                        break
                    if self._neuron_surgery_count >= neuron_cap:
                        break
                    if self.interval_surgery_count >= global_cap:
                        break
                    if model.get_layer_width(l) <= self.config.d_min:
                        break

                    success = self.engine.remove_neuron(model, l, neuron_idx, optimizer, step)
                    if success:
                        neurons_removed += 1
                        removed_in_layer += 1
                        self._neuron_surgery_count += 1
                        self.interval_surgery_count += 1

    def _clear_treatment_state(self):
        """Clear all treatment monitoring and titration state."""
        self._pre_treatment_model = None
        self._pre_treatment_val_loss = float('inf')
        self._pre_treatment_epoch = -1
        self._rollback_treatment_type = None
        self._pre_treatment_is_structural = False
        self._titration_active = False
        self._titration_dose_level = 0
        self._titration_dose_factor = 1.0
        self._titration_epoch = -1
        self._open_heart_attempted = False

    def _perform_open_heart_surgery(
        self, model, optimizer, step, loss_fn, val_batch,
    ) -> bool:
        """Last resort: find memorizing neurons and remove them.

        Pauses training, runs forward passes on training + validation data,
        identifies neurons that fire strongly on training but not validation
        (memorization signature), and removes them.

        Returns True if the surgery helped (loss ratio improved).
        """
        import torch
        import torch.nn as nn

        if val_batch is None:
            print("  [OPEN HEART] No validation batch available, skipping")
            return False

        x_val, y_val = val_batch
        model.eval()
        device = next(model.parameters()).device

        try:
            # --- 1. Collect per-neuron activations via hooks ---
            train_acts = {}   # {layer_idx: tensor[num_neurons]}
            val_acts = {}

            hooks = []
            layer_outputs_store = {}

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        layer_outputs_store[layer_idx] = output.detach()
                return hook_fn

            # Register hooks on all layers
            for l in range(model.num_layers):
                layer = model.layers[l]
                h = layer.register_forward_hook(make_hook(l))
                hooks.append(h)

            # Forward pass on validation data
            with torch.no_grad():
                x_v = x_val.to(device) if not x_val.is_cuda else x_val
                model(x_v)
                for l, act in layer_outputs_store.items():
                    if act.dim() == 4:  # Spatial: [B, C, H, W] → per-channel mean
                        val_acts[l] = act.abs().mean(dim=(0, 2, 3))
                    elif act.dim() >= 2:  # Flat: [B, D] → per-neuron mean
                        val_acts[l] = act.abs().mean(dim=0)
                layer_outputs_store.clear()

            # Forward pass on training data (use val_batch-sized chunk)
            # We don't have explicit train data here, but we can use
            # the model's gradient_norms and activation_magnitudes
            # already collected during training.
            # Alternative: use the EMA stats for comparison.
            # Simplification: use the stored activation_magnitudes from training.
            for l in range(model.num_layers):
                if l in self.stats.activation_magnitudes and self.stats.activation_magnitudes[l]:
                    stacked = torch.stack(self.stats.activation_magnitudes[l])
                    train_acts[l] = stacked.mean(dim=0)

            # Remove hooks
            for h in hooks:
                h.remove()

            # --- 2. Compute memorization score per neuron ---
            memorizing = []  # [(layer, neuron_idx, score)]
            threshold = getattr(self.config, 'open_heart_memorization_threshold', 3.0)

            for l in range(model.num_layers):
                if l not in train_acts or l not in val_acts:
                    continue
                t_act = train_acts[l]
                v_act = val_acts[l]
                n = min(len(t_act), len(v_act))
                for i in range(n):
                    t_val = float(t_act[i])
                    v_val = float(v_act[i])
                    if v_val < 1e-8 and t_val > 1e-4:
                        score = t_val / 1e-8  # Extremely memorizing
                    elif v_val > 1e-8:
                        score = t_val / v_val
                    else:
                        continue
                    if score > threshold:
                        memorizing.append((l, i, score))

            if not memorizing:
                print("  [OPEN HEART] No memorizing neurons found")
                model.train()
                return False

            # --- 3. Remove memorizing neurons ---
            memorizing.sort(key=lambda x: x[2], reverse=True)
            max_remove_frac = getattr(self.config, 'open_heart_max_remove_frac', 0.15)

            # Group by layer for batch removal
            from collections import defaultdict
            per_layer = defaultdict(list)
            for l, i, score in memorizing:
                per_layer[l].append((i, score))

            total_removed = 0
            for l, neurons in per_layer.items():
                layer_width = model.get_layer_width(l)
                min_width = max(self.config.d_min,
                                getattr(self.config, 'min_spatial_channels', 8))
                max_remove = max(1, int(layer_width * max_remove_frac))
                max_remove = min(max_remove, layer_width - min_width)

                # Snap to channel alignment (Tensor Core efficiency)
                is_spatial = hasattr(model.layers[l], 'mode') and model.layers[l].mode == "spatial"
                if is_spatial and self.config.channel_alignment > 1:
                    from .config import snap_remove_for_alignment
                    max_remove = snap_remove_for_alignment(
                        layer_width, max_remove,
                        self.config.channel_alignment, min_width)

                # Sort by index descending to avoid invalidation
                to_remove = neurons[:max_remove]
                to_remove.sort(key=lambda x: x[0], reverse=True)

                for neuron_idx, score in to_remove:
                    if model.get_layer_width(l) <= min_width:
                        break
                    success = self.engine.remove_neuron(
                        model, l, neuron_idx, optimizer, step)
                    if success:
                        total_removed += 1

            print(f"  [OPEN HEART] Removed {total_removed} memorizing neurons "
                  f"from {len(per_layer)} layers")

            if total_removed == 0:
                model.train()
                return False

            # --- 4. Re-evaluate ---
            model.eval()
            with torch.no_grad():
                x_v = x_val.to(device) if not x_val.is_cuda else x_val
                y_v = y_val.to(device) if not y_val.is_cuda else y_val
                output = model(x_v)
                if loss_fn is not None:
                    new_loss = loss_fn(output, y_v).item()
                    new_ratio = new_loss / max(abs(self._pre_treatment_val_loss), 1e-8)
                    print(f"  [OPEN HEART] Post-surgery val_loss={new_loss:.4f}, "
                          f"ratio={new_ratio:.3f}")
                    model.train()
                    return new_ratio <= 1.0

            model.train()
            return False

        except Exception as e:
            print(f"  [OPEN HEART] Surgery failed with error: {e}")
            # Remove hooks if still attached
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            model.train()
            return False

    def _do_layer_surgery(
        self,
        model: ASANNModel,
        optimizer,
        signals: Dict[str, Any],
        step: int,
        add_mult: float,
        block_additions: bool = False,
        from_healthy_diagnosis: bool = False,
    ):
        """Execute layer addition and removal.

        When from_healthy_diagnosis=True (called from _do_custom_surgery during
        HEALTHY diagnosis), the loss_plateaued check is skipped. The HEALTHY
        diagnosis already confirms model stability — requiring plateau creates
        a Catch-22 where plateau triggers STALLED_CONVERGENCE (SICK), so layer
        addition can never fire.
        """
        lss = signals.get("LSS", {})
        lcs = signals.get("LCS", {})
        mean_lss = signals.get("mean_LSS", 0.0)
        loss_plateaued = signals.get("loss_plateaued", False)

        layers_added = 0
        layers_removed = 0

        # --- Add layer (blocked by Fix 4 ceiling) ---
        # Step-based path: mean LSS > threshold AND loss plateaued (both required)
        # Diagnosis HEALTHY path: mean LSS > threshold only (HEALTHY = stable enough)
        layer_cap = self.config.max_layer_surgeries_per_interval
        global_cap = self.config.max_total_surgeries_per_interval
        sat_threshold = self.config.saturation_threshold * add_mult
        # Use explicit cooldown if set, otherwise fall back to 3× min_post_surgery_steps
        layer_cooldown = (self.config.layer_add_cooldown_steps
                          if self.config.layer_add_cooldown_steps > 0
                          else self.config.min_post_surgery_steps * 3)
        plateau_ok = loss_plateaued or from_healthy_diagnosis
        # Guard: don't add layers if too many existing layers have trivial-only ops
        unenriched = model.count_unenriched_layers()
        enrichment_ok = unenriched <= self.config.max_unenriched_layers
        if (
            enrichment_ok
            and not block_additions
            and mean_lss > sat_threshold
            and plateau_ok
            and layers_added < self.config.max_layers_add_per_surgery
            and self._layer_surgery_count < layer_cap
            and self.interval_surgery_count < global_cap
            and (step - self.last_layer_add_step >= layer_cooldown or layer_cooldown == 0)
        ):
            # Insert where most needed: at the position with highest LSS
            if lss:
                position = max(lss, key=lss.get)
                self.engine.add_layer(model, position, optimizer, step)
                layers_added += 1
                self._layer_surgery_count += 1
                self.interval_surgery_count += 1
                self.last_layer_add_step = step
                # Shift all layer-indexed trackers to match new numbering
                self._shift_identity_counts(position, direction="insert")
                self._shift_unenriched_counts(position, direction="insert")
                self.engine.shift_removal_history(position, direction="insert")

        # --- Remove layer ---
        for l in list(range(model.num_layers)):
            if layers_removed >= self.config.max_layers_remove_per_surgery:
                break
            if l >= model.num_layers:
                break

            if l in lcs and lcs[l] < self.config.identity_threshold:
                # Track consecutive low-contribution intervals
                self.layer_identity_counts[l] = self.layer_identity_counts.get(l, 0) + 1

                if self.layer_identity_counts[l] >= self.config.layer_identity_consecutive:
                    success = self.engine.remove_layer(model, l, optimizer, step)
                    if success:
                        layers_removed += 1
                        self._layer_surgery_count += 1
                        self.interval_surgery_count += 1
                        # Shift all layer-indexed trackers to match new numbering
                        self._shift_identity_counts(l, direction="remove")
                        self._shift_unenriched_counts(l, direction="remove")
                        self.engine.shift_removal_history(l, direction="remove")
                        break
            else:
                # Reset count if the layer is contributing
                if l in self.layer_identity_counts:
                    self.layer_identity_counts[l] = 0

    def _do_operation_surgery(
        self,
        model: ASANNModel,
        optimizer,
        step: int,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn,
        current_loss: float,
        val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Execute operation discovery via probing."""
        # Per-type cap: operation surgery has its own independent budget
        op_cap = self.config.max_operation_surgeries_per_interval
        global_cap = self.config.max_total_surgeries_per_interval
        if self._operation_surgery_count >= op_cap:
            return
        if self.interval_surgery_count >= global_cap:
            return

        # Run probes to find beneficial operations (pass val_batch for RC5b/RC6)
        recommendations = self.engine.probe_operations(
            model, x_batch, y_batch, loss_fn, current_loss,
            current_step=step, val_batch=val_batch,
        )

        # Always call execute_operation_surgery — even with empty recommendations.
        # The regular probing loop handles 0 recommendations gracefully, and the
        # stochastic exploration code at the end of execute_operation_surgery needs
        # to run regardless (it discovers ops that can't show benefit on cold probes).
        self.engine.execute_operation_surgery(
            model, recommendations, optimizer, step,
            skip_exploration=self.architecture_stable,
        )
        # Count how many were actually executed
        if recommendations:
            ops_executed = min(len(recommendations), self.config.max_ops_change_per_surgery)
            self._operation_surgery_count += ops_executed
            self.interval_surgery_count += ops_executed

    def _do_connection_surgery(
        self,
        model: ASANNModel,
        optimizer,
        signals: Dict[str, Any],
        step: int,
        block_additions: bool = False,
    ):
        """Execute connection creation and removal."""
        clgc = signals.get("CLGC", {})
        connections_changed = 0
        conn_cap = self.config.max_connection_surgeries_per_interval
        global_cap = self.config.max_total_surgeries_per_interval

        # --- Create connections (blocked by Fix 4 ceiling) ---
        if block_additions:
            candidates = []
        else:
            # Sort pairs by CLGC value (highest first)
            candidates = [
                ((i, j), corr)
                for (i, j), corr in clgc.items()
                if corr > self.config.connection_threshold
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)

        for (source, target), corr in candidates:
            if connections_changed >= self.config.max_connections_change_per_surgery:
                break
            if self._connection_surgery_count >= conn_cap:
                break
            if self.interval_surgery_count >= global_cap:
                break
            # Ensure valid indices and that target > source
            if target <= source:
                continue
            if target > model.num_layers:
                continue

            # Check if connection already exists
            exists = any(
                c.source == source and c.target == target
                for c in model.connections
            )
            if exists:
                continue

            # Guard: skip connections between layers with different spatial resolutions.
            # Connection semantics: source reads h[source], target adds to h[target-1].
            # h[k] = output of layer k-1 (or stem if k==0).
            # We must check that h[source] and h[target-1] have the same H×W.
            if hasattr(model, '_is_spatial') and model._is_spatial:
                src_spatial = None
                tgt_spatial = None
                # h[source] spatial shape
                if source == 0:
                    src_spatial = getattr(model.input_projection, 'spatial_shape', None)
                elif source - 1 < len(model.layers):
                    src_spatial = getattr(model.layers[source - 1], 'spatial_shape', None)
                # h[target-1] spatial shape (the tensor being added to)
                tgt_h_idx = target - 1
                if tgt_h_idx == 0:
                    tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
                elif tgt_h_idx - 1 < len(model.layers):
                    tgt_spatial = getattr(model.layers[tgt_h_idx - 1], 'spatial_shape', None)
                # If both have spatial shapes, check H×W match
                if (src_spatial is not None and tgt_spatial is not None
                        and (src_spatial[1] != tgt_spatial[1]
                             or src_spatial[2] != tgt_spatial[2])):
                    continue  # Skip: different spatial resolutions

            self.engine.create_connection(model, source, target, optimizer, step)
            connections_changed += 1
            self._connection_surgery_count += 1
            self.interval_surgery_count += 1

        # --- Remove connections ---
        # Check each connection's utility (with protection for young connections)
        protection_steps = self.config.surgery_interval_init * 2  # 2 surgery intervals
        to_remove = []
        for idx, conn in enumerate(model.connections):
            # Skip young connections that haven't had time to develop utility
            created_step = getattr(conn, '_asann_created_step', 0)
            if step - created_step < protection_steps:
                continue
            cu = conn.utility()
            if cu < self.config.connection_remove_threshold:
                conn.low_utility_count += 1
                if conn.low_utility_count >= self.config.connection_remove_consecutive:
                    to_remove.append(idx)
            else:
                conn.low_utility_count = 0

        # Remove in reverse order to preserve indices
        for idx in reversed(to_remove):
            if connections_changed >= self.config.max_connections_change_per_surgery:
                break
            self.engine.remove_connection(model, idx, optimizer, step)
            connections_changed += 1
            self._connection_surgery_count += 1
            self.interval_surgery_count += 1

    def _validate_connection_projections(self, model: ASANNModel):
        """Validate and repair skip connection projections after surgery.

        After multiple surgery types execute in a single interval (neuron add/remove,
        layer add/remove, connection creation), some connection projections may have
        stale dimensions. This method checks each connection and creates/updates
        projections as needed to ensure dimensional consistency.

        For spatial layers, uses Conv2d(1x1) projections + adaptive_avg_pool2d
        when source/target are both spatial.
        """
        device = self.config.device
        for conn in model.connections:
            # Get actual source dimension and spatial info
            if conn.source == 0:
                d_source = model.input_projection.out_features
                source_spatial = getattr(model.input_projection, 'spatial_shape', None)
            else:
                src_layer = model.layers[conn.source - 1]
                d_source = src_layer.out_features
                source_spatial = getattr(src_layer, 'spatial_shape', None)

            # Get actual target dimension: conn.target adds to h[target-1]
            if conn.target - 1 == 0:
                d_target = model.input_projection.out_features
                target_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif conn.target - 1 <= model.num_layers:
                tgt_layer = model.layers[conn.target - 2]
                d_target = tgt_layer.out_features
                target_spatial = getattr(tgt_layer, 'spatial_shape', None)
            else:
                continue  # invalid connection, skip

            # Determine if this is a spatial-to-spatial connection
            both_spatial = (source_spatial is not None and target_spatial is not None)

            if both_spatial:
                # Guard: remove connections across spatial resolution boundaries
                if (source_spatial[1] != target_spatial[1]
                        or source_spatial[2] != target_spatial[2]):
                    # Spatial resolution mismatch — mark for removal
                    conn._asann_invalid_spatial = True
                    continue

                # Spatial connections: use Conv2d(1x1) projection
                C_src = source_spatial[0]
                C_tgt = target_spatial[0]

                if conn.projection is not None:
                    if isinstance(conn.projection, nn.Conv2d):
                        if conn.projection.in_channels != C_src or conn.projection.out_channels != C_tgt:
                            new_proj = nn.Conv2d(C_src, C_tgt, kernel_size=1, bias=False).to(device)
                            # Copy overlapping channels
                            copy_out = min(conn.projection.out_channels, C_tgt)
                            copy_in = min(conn.projection.in_channels, C_src)
                            new_proj.weight.data[:copy_out, :copy_in] = conn.projection.weight.data[:copy_out, :copy_in]
                            conn.projection = new_proj
                    else:
                        # Was linear, now needs to be Conv2d
                        new_proj = nn.Conv2d(C_src, C_tgt, kernel_size=1, bias=False).to(device)
                        nn.init.dirac_(new_proj.weight)
                        conn.projection = new_proj
                else:
                    if C_src != C_tgt:
                        new_proj = nn.Conv2d(C_src, C_tgt, kernel_size=1, bias=False).to(device)
                        nn.init.dirac_(new_proj.weight)
                        conn.projection = new_proj

                # Update spatial shape attributes on connection
                conn.spatial_source_shape = source_spatial
                conn.spatial_target_shape = target_spatial
            else:
                # Flat connections: original behavior
                if conn.projection is not None:
                    # Check if existing projection has correct dimensions
                    if hasattr(conn.projection, 'in_features'):
                        if conn.projection.in_features != d_source or conn.projection.out_features != d_target:
                            old_proj = conn.projection
                            new_proj = nn.Linear(d_source, d_target, bias=False).to(device)
                            copy_rows = min(old_proj.out_features, d_target)
                            copy_cols = min(old_proj.in_features, d_source)
                            new_proj.weight.data[:copy_rows, :copy_cols] = old_proj.weight.data[:copy_rows, :copy_cols]
                            conn.projection = new_proj
                    else:
                        # Was Conv2d, now needs to be Linear (crossed flatten boundary)
                        new_proj = nn.Linear(d_source, d_target, bias=False).to(device)
                        copy_d = min(d_source, d_target)
                        new_proj.weight.data[:copy_d, :copy_d] = torch.eye(copy_d, device=device)
                        conn.projection = new_proj
                else:
                    if d_source != d_target:
                        new_proj = nn.Linear(d_source, d_target, bias=False).to(device)
                        copy_d = min(d_source, d_target)
                        new_proj.weight.data[:copy_d, :copy_d] = torch.eye(copy_d, device=device)
                        conn.projection = new_proj

        # Remove any connections flagged as spatially invalid
        invalid_indices = [
            i for i, conn in enumerate(model.connections)
            if getattr(conn, '_asann_invalid_spatial', False)
        ]
        for idx in reversed(invalid_indices):
            print(f"  [SURGERY] Removing spatially invalid connection "
                  f"{model.connections[idx].source}->{model.connections[idx].target}")
            model.connections.pop(idx)

    def adjust_surgery_interval(self, new_interval: int):
        """Adjust the surgery interval (can be called by meta-learner).

        Takes the MAX of the suggested interval and the current interval.
        This ensures that cooldown (which increases the interval after surgery)
        is never undone by the meta-learner suggesting a lower value.

        Fix 3: Capped by surgery_interval_max.
        """
        self.surgery_interval = min(
            max(50, max(new_interval, self.surgery_interval)),
            self.config.surgery_interval_max,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Save scheduler state for checkpoint resume."""
        return {
            "surgery_interval": self.surgery_interval,
            "consecutive_stable_intervals": self.consecutive_stable_intervals,
            "architecture_stable": self.architecture_stable,
            "interval_surgery_count": self.interval_surgery_count,
            "last_surgery_step": self.last_surgery_step,
            "last_layer_add_step": self.last_layer_add_step,
            "layer_identity_counts": dict(self.layer_identity_counts),
            "_loss_gate_history": list(self._loss_gate_history),
            "_prev_cost": self._prev_cost,
            "_total_active_rounds": self._total_active_rounds,
            "_loss_ema": self._loss_ema,
            "_loss_ema_decay": self._loss_ema_decay,
            "_best_loss_ema": self._best_loss_ema,
            # v2: diagnosis + treatment state
            "diagnosis_engine": self.diagnosis_engine.state_dict(),
            "treatment_planner": self.treatment_planner.state_dict(),
            # v3: patient history
            "patient_history": self.patient_history.state_dict(),
            # Memorization tracking
            "n_train_samples": self.n_train_samples,
            "_consecutive_memorization": self._consecutive_memorization,
            "_memorization_best_val": self._memorization_best_val,
            "_memorization_suppressed": self._memorization_suppressed,
            # Stalled convergence tracking (unified target metric)
            "_stall_best_val_metric": self._stall_best_val_metric,
            "_stall_higher_is_better": self._stall_higher_is_better,
            "_stall_best_epoch": self._stall_best_epoch,
            "_stall_diagnosed_count": self._stall_diagnosed_count,
            # Gradient death tracking
            "_consecutive_grad_death": self._consecutive_grad_death,
            # Iatrogenic damage tracking
            "_last_treatment_epoch": self._last_treatment_epoch,
            "_last_treatment_pre_metric": self._last_treatment_pre_metric,
            # Treatment exhaustion tracking
            "_consecutive_no_treatment": self._consecutive_no_treatment,
            "_treatments_exhausted": self._treatments_exhausted,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore scheduler state from checkpoint."""
        self.surgery_interval = state["surgery_interval"]
        self.consecutive_stable_intervals = state["consecutive_stable_intervals"]
        self.architecture_stable = state["architecture_stable"]
        self.interval_surgery_count = state.get("interval_surgery_count", 0)
        self.last_surgery_step = state["last_surgery_step"]
        self.last_layer_add_step = state.get("last_layer_add_step", 0)
        self.layer_identity_counts = state.get("layer_identity_counts", {})
        self._loss_gate_history = deque(
            state.get("_loss_gate_history", []),
            maxlen=self.config.loss_gate_window,
        )
        self._prev_cost = state.get("_prev_cost")
        self._total_active_rounds = state.get("_total_active_rounds", 0)
        self._loss_ema = state.get("_loss_ema", float("inf"))
        self._loss_ema_decay = state.get("_loss_ema_decay", 0.995)
        self._best_loss_ema = state.get("_best_loss_ema", float("inf"))
        # v2: restore diagnosis + treatment state
        if "diagnosis_engine" in state:
            self.diagnosis_engine.load_state_dict(state["diagnosis_engine"])
        if "treatment_planner" in state:
            self.treatment_planner.load_state_dict(state["treatment_planner"])
        if "patient_history" in state:
            self.patient_history.load_state_dict(state["patient_history"])
        # Memorization tracking
        self.n_train_samples = state.get("n_train_samples", 0)
        self._consecutive_memorization = state.get("_consecutive_memorization", 0)
        self._memorization_best_val = state.get("_memorization_best_val", float('inf'))
        self._memorization_suppressed = state.get("_memorization_suppressed", False)
        # Stalled convergence tracking (unified target metric)
        self._stall_best_val_metric = state.get("_stall_best_val_metric", None)
        self._stall_higher_is_better = state.get("_stall_higher_is_better", True)
        self._stall_best_epoch = state.get("_stall_best_epoch", 0)
        self._stall_diagnosed_count = state.get("_stall_diagnosed_count", 0)
        # Gradient death tracking
        self._consecutive_grad_death = state.get("_consecutive_grad_death", 0)
        # Iatrogenic damage tracking
        self._last_treatment_epoch = state.get("_last_treatment_epoch", -999)
        self._last_treatment_pre_metric = state.get("_last_treatment_pre_metric", None)
        # Treatment exhaustion tracking
        self._consecutive_no_treatment = state.get("_consecutive_no_treatment", 0)
        self._treatments_exhausted = state.get("_treatments_exhausted", False)
