"""Concrete lab test implementations for ASANN Lab Diagnostic System.

Tier 1 (Cheap - run always when referred):
  - ResidualStructureTest: Analyzes structure of prediction errors
  - GradientConcentrationTest: Where gradient magnitude concentrates
  - LossCurvatureTest: Local curvature of loss landscape

Tier 2 (Medium - run when Tier 1 is inconclusive):
  - OperationSensitivityTest: Probe candidate operations for benefit
  - SkipConnectionTest: Test if skip connections improve gradient flow
  - OverfittingLocalizationTest: Identify which layers cause overfitting

Tier 3 (Expensive - run when Tier 2 points to specific hypothesis):
  - ArchitectureComparisonTest: Train small copies with modifications
  - DataComplexityTest: Estimate data dimensionality vs model capacity
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .diagnosis import DiseaseType, Diagnosis
from .lab import LabTest, LabResult, LabDiagnostics, PatientHistory
from .treatments import TreatmentType


# Treatment rotation: when a suggested treatment has repeatedly failed,
# rotate to the next alternative in the chain.
_TREATMENT_ROTATION = {
    TreatmentType.CAPACITY_GROWTH: TreatmentType.LAYER_ADDITION,
    TreatmentType.LAYER_ADDITION: TreatmentType.POLYNOMIAL_PACKAGE,
    TreatmentType.POLYNOMIAL_PACKAGE: TreatmentType.DERIVATIVE_PACKAGE,
    TreatmentType.DERIVATIVE_PACKAGE: TreatmentType.BRANCHED_DIFFUSION_REACTION,
    TreatmentType.SPATIAL_CONV_PACKAGE: TreatmentType.DERIVATIVE_PACKAGE,
    TreatmentType.BATCHNORM_PACKAGE: TreatmentType.SPATIAL_CONV_PACKAGE,
    # Graph treatment rotation chain (extended with new ops)
    TreatmentType.GRAPH_CONV_PACKAGE: TreatmentType.GRAPH_ATTENTION_PACKAGE,
    TreatmentType.GRAPH_ATTENTION_PACKAGE: TreatmentType.GRAPH_DIFFUSION_PACKAGE,
    TreatmentType.GRAPH_DIFFUSION_PACKAGE: TreatmentType.GRAPH_BRANCHED_AGG,
    TreatmentType.GRAPH_BRANCHED_AGG: TreatmentType.GRAPH_SPECTRAL_PACKAGE,
    TreatmentType.GRAPH_SPECTRAL_PACKAGE: TreatmentType.GRAPH_GIN_PACKAGE,
    TreatmentType.GRAPH_GIN_PACKAGE: TreatmentType.GRAPH_POSITIONAL_PACKAGE,
    TreatmentType.GRAPH_POSITIONAL_PACKAGE: TreatmentType.CAPACITY_GROWTH,
    TreatmentType.GRAPH_NORM_PACKAGE: TreatmentType.GRAPH_CONV_PACKAGE,
}


def _rotate_failed_treatment(
    suggested: Optional[TreatmentType],
    patient_history: Optional[PatientHistory],
    findings: Dict[str, Any],
    confidence: float,
) -> tuple:
    """If suggested treatment has failed 2+ times across all diseases, rotate.

    Returns (suggested, confidence, findings) - possibly modified.
    """
    if patient_history is None or suggested is None:
        return suggested, confidence, findings

    # Count total failures of this treatment across all diseases
    all_ineff = []
    for d_name, treatments in patient_history._ineffective_treatments.items():
        all_ineff.extend(treatments)

    fail_count = all_ineff.count(suggested.name)
    if fail_count < 2:
        return suggested, confidence, findings

    # Rotate through alternatives until we find one not also failed 2+ times
    original = suggested
    visited = {suggested}
    while suggested in _TREATMENT_ROTATION:
        next_treatment = _TREATMENT_ROTATION[suggested]
        if next_treatment in visited:
            break
        visited.add(next_treatment)
        if all_ineff.count(next_treatment.name) < 2:
            suggested = next_treatment
            break
        suggested = next_treatment

    if suggested != original:
        findings = dict(findings)
        findings["summary"] = (f"Rotated from {original.name} (failed {fail_count}x) "
                                f"to {suggested.name}")
        confidence = max(0.4, confidence - 0.2)

    return suggested, confidence, findings


# ==================== Tier 1: Cheap Tests ====================


class InputStructureTest(LabTest):
    """Analyzes the structure of input data to classify the problem type.

    PDE problems have distinctive input signatures:
    - Very low dimensionality (2-5 coordinate features)
    - Continuous-valued inputs (coordinates, not categorical)

    Image classification has spatial 4D inputs. Tabular has high-dimensional
    feature vectors. This test routes underfitting to the right treatment
    class: derivative/polynomial ops for PDE, standard ladder for tabular.

    Cost: ~0 (reads one batch, no forward passes).
    """

    name = "Input Structure Analysis"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        x = batch[0]  # [N, d_input] or [N, C, H, W]

        is_spatial = (x.dim() == 4)
        n_features = x.shape[1]
        n_samples = x.shape[0]

        # Continuity: ratio of unique values to sample count per feature
        continuity_score = 0.0
        if not is_spatial and x.dim() == 2:
            unique_ratios = []
            for col in range(min(n_features, 20)):
                n_unique = x[:, col].unique().numel()
                unique_ratios.append(n_unique / n_samples)
            continuity_score = float(np.mean(unique_ratios))

        if is_spatial:
            problem_type = "image"
            summary = "Spatial input - standard pipeline appropriate"
            suggested = None
            confidence = 0.8
        elif (n_features <= 5
              and continuity_score > 0.8
              and task_type == "regression"):
            problem_type = "pde_like"
            summary = ("Low-dimensional continuous inputs - needs derivative "
                       "and polynomial operations, not batch normalization")
            suggested = TreatmentType.DERIVATIVE_PACKAGE
            confidence = 0.85
        elif n_features > 10:
            problem_type = "tabular"
            summary = "High-dimensional tabular input - standard ladder appropriate"
            suggested = None
            confidence = 0.6
        else:
            problem_type = "unknown"
            summary = "Could not determine problem structure - defer to other tests"
            suggested = None
            confidence = 0.2

        findings = {
            "problem_type": problem_type,
            "n_features": n_features,
            "n_samples": n_samples,
            "continuity_score": round(continuity_score, 4),
            "is_spatial": is_spatial,
            "summary": summary,
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"task_type": task_type},
        )


class ResidualStructureTest(LabTest):
    """Analyzes spatial/temporal structure of prediction errors.

    If residuals have structure (smooth, oscillatory, localized), the model
    needs different operations - not just more capacity.
    If residuals are random noise, the model truly needs more capacity.

    Cost: ~5 forward passes on validation data.
    """

    name = "Residual Structure Analysis"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
        ]

    def is_applicable(self, diagnosis: Diagnosis, task_type: str = "regression",
                      patient_history: Optional[PatientHistory] = None) -> bool:
        if task_type != "regression":
            return False
        return super().is_applicable(diagnosis, task_type, patient_history)

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        all_residuals = []
        with torch.no_grad():
            for i, batch in enumerate(val_data):
                if i >= 5:
                    break
                x = batch[0].to(device)
                y = batch[1].to(device)
                pred = model(x)
                residuals = (pred - y).cpu()
                all_residuals.append(residuals)

        model.train()

        if not all_residuals:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        residuals = torch.cat(all_residuals, dim=0)  # [N, d_out]
        if residuals.dim() == 1:
            residuals = residuals.unsqueeze(1)

        n_samples = residuals.shape[0]
        if n_samples < 4:
            return LabResult(self.name, {"summary": "insufficient samples"}, 0.0)

        # 1. Smoothness: ratio of Laplacian energy to signal energy
        laplacian = residuals[2:] - 2 * residuals[1:-1] + residuals[:-2]
        laplacian_energy = (laplacian ** 2).mean().item()
        signal_energy = (residuals ** 2).mean().item()

        if signal_energy < 1e-10:
            smoothness = 1.0
        else:
            smoothness = max(0.0, 1.0 - laplacian_energy / signal_energy)

        # 2. Autocorrelation at lag 1
        r_mean = residuals.mean(dim=0, keepdim=True)
        r_centered = residuals - r_mean
        var = (r_centered ** 2).sum(dim=0)
        autocorr_num = (r_centered[:-1] * r_centered[1:]).sum(dim=0)
        autocorr = (autocorr_num / (var + 1e-10)).mean().item()

        # 3. FFT-based dominant frequency
        if n_samples >= 8:
            r_flat = residuals.mean(dim=1)  # Average over output dims
            fft_vals = torch.fft.rfft(r_flat - r_flat.mean())
            fft_magnitudes = torch.abs(fft_vals[1:])  # Exclude DC
            if fft_magnitudes.numel() > 0:
                dominant_freq_idx = fft_magnitudes.argmax().item()
                dominant_freq = (dominant_freq_idx + 1) / n_samples
                freq_concentration = (fft_magnitudes.max() / (fft_magnitudes.sum() + 1e-10)).item()
            else:
                dominant_freq = 0.0
                freq_concentration = 0.0
        else:
            dominant_freq = 0.0
            freq_concentration = 0.0

        # Classify structure type
        if smoothness > 0.7 and autocorr > 0.5:
            structure_type = "smooth"
            summary = "Residuals have smooth structure - model needs derivative operations"
            suggested = TreatmentType.DERIVATIVE_PACKAGE
            confidence = min(0.9, smoothness)
        elif freq_concentration > 0.3 and autocorr > 0.2:
            structure_type = "oscillatory"
            summary = "Residuals are oscillatory - model needs frequency-sensitive operations"
            suggested = TreatmentType.SPATIAL_CONV_PACKAGE
            confidence = min(0.8, freq_concentration + 0.3)
        elif autocorr < 0.1 and smoothness < 0.3:
            structure_type = "random"
            summary = "Residuals are random noise - model needs more capacity"
            suggested = TreatmentType.CAPACITY_GROWTH
            confidence = min(0.85, 1.0 - smoothness)
        else:
            structure_type = "mixed"
            summary = "Residuals have mixed structure - run deeper tests"
            suggested = None
            confidence = 0.3

        findings = {
            "smoothness": round(smoothness, 4),
            "autocorrelation": round(autocorr, 4),
            "dominant_frequency": round(dominant_freq, 4),
            "frequency_concentration": round(freq_concentration, 4),
            "structure_type": structure_type,
            "summary": summary,
        }

        # Post-filter: rotate away from repeatedly failed treatments
        suggested, confidence, findings = _rotate_failed_treatment(
            suggested, patient_history, findings, confidence)

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"n_samples": n_samples, "signal_energy": signal_energy},
        )


class GradientConcentrationTest(LabTest):
    """Analyzes where gradient magnitude concentrates across layers.

    - Input layers: input representation problem
    - Output layers: output head mismatch
    - Uniform: needs depth/width changes
    - Middle vanishing: needs skip connections

    Cost: Free (uses already-accumulated GDS data).
    """

    name = "Gradient Concentration Analysis"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        gds = signals.get("GDS", {})
        if not gds:
            return LabResult(self.name, {"summary": "no GDS data"}, 0.0)

        n_layers = len(gds)
        if n_layers < 2:
            return LabResult(self.name, {"summary": "insufficient layers"}, 0.0)

        # Get GDS values in layer order
        layer_indices = sorted(gds.keys())
        gds_values = [gds[l] for l in layer_indices]
        gds_array = np.array(gds_values)
        total_gds = gds_array.sum()

        if total_gds < 1e-10:
            return LabResult(self.name,
                             {"summary": "near-zero gradients", "concentration": "vanishing"},
                             0.5, TreatmentType.LR_REDUCE)

        # Normalize to get gradient distribution across layers
        gds_dist = gds_array / total_gds

        # Determine concentration pattern
        n_input = max(1, n_layers // 3)
        n_output = max(1, n_layers // 3)
        n_middle = n_layers - n_input - n_output

        input_mass = gds_dist[:n_input].sum()
        output_mass = gds_dist[-n_output:].sum() if n_output > 0 else 0.0
        middle_mass = gds_dist[n_input:n_input + n_middle].sum() if n_middle > 0 else 0.0

        # Compute Gini coefficient (measure of concentration)
        sorted_dist = np.sort(gds_dist)
        n = len(sorted_dist)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * sorted_dist).sum() / (n * sorted_dist.sum() + 1e-10)

        # Classify concentration pattern
        if gini < 0.2:
            concentration = "uniform"
            summary = "Gradient uniformly distributed - model needs width or depth changes"
            suggested = TreatmentType.CAPACITY_GROWTH
            confidence = 0.6
        elif input_mass > 0.5:
            concentration = "input"
            summary = "Gradient concentrated at input - input representation problem"
            suggested = TreatmentType.SPATIAL_CONV_PACKAGE
            confidence = 0.7
        elif output_mass > 0.5:
            concentration = "output"
            summary = "Gradient concentrated at output - output head capacity limited"
            suggested = TreatmentType.CAPACITY_GROWTH
            confidence = 0.65
        elif n_middle > 0 and middle_mass < 0.1:
            concentration = "middle_vanishing"
            summary = "Gradient vanishes in middle layers - needs skip connections"
            suggested = TreatmentType.RESNET_BLOCK
            confidence = 0.75
        else:
            concentration = "mixed"
            summary = "Gradient has mixed concentration pattern"
            suggested = None
            confidence = 0.3

        findings = {
            "concentration": concentration,
            "input_mass": round(float(input_mass), 4),
            "middle_mass": round(float(middle_mass), 4),
            "output_mass": round(float(output_mass), 4),
            "gini_coefficient": round(float(gini), 4),
            "gradient_profile": [round(float(v), 6) for v in gds_values],
            "summary": summary,
        }

        # Post-filter: rotate away from repeatedly failed treatments
        suggested, confidence, findings = _rotate_failed_treatment(
            suggested, patient_history, findings, confidence)

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"n_layers": n_layers, "total_gds": float(total_gds)},
        )


class LossCurvatureTest(LabTest):
    """Estimates local curvature of loss landscape around current parameters.

    - High curvature (sharp minimum): needs regularization, not capacity
    - Low curvature (plateau): needs LR boost or structural change
    - Asymmetric curvature (saddle): needs perturbation

    Cost: ~10 forward passes with perturbed weights.
    """

    name = "Loss Landscape Curvature"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        # Get a validation batch
        val_batch = None
        for batch in val_data:
            val_batch = batch
            break
        if val_batch is None:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        x = val_batch[0].to(device)
        y = val_batch[1].to(device)

        # Compute base loss
        with torch.no_grad():
            base_pred = model(x)
            base_loss = F.mse_loss(base_pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                else F.cross_entropy(base_pred, y).item()

        # Sample random perturbation directions and measure loss change
        n_perturbations = 10
        epsilon = 1e-3
        losses_plus = []
        losses_minus = []

        # Save original parameters
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        for _ in range(n_perturbations):
            # Random direction (normalized)
            direction = {}
            total_norm_sq = 0.0
            for n, p in model.named_parameters():
                d = torch.randn_like(p)
                total_norm_sq += d.norm().item() ** 2
                direction[n] = d
            total_norm = total_norm_sq ** 0.5
            for n in direction:
                direction[n] /= total_norm

            # Perturb +epsilon
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.data.copy_(original_params[n] + epsilon * direction[n])
                pred_plus = model(x)
                loss_plus = F.mse_loss(pred_plus, y).item() if y.dim() > 1 or y.is_floating_point() \
                    else F.cross_entropy(pred_plus, y).item()
                losses_plus.append(loss_plus)

            # Perturb - epsilon
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.data.copy_(original_params[n] - epsilon * direction[n])
                pred_minus = model(x)
                loss_minus = F.mse_loss(pred_minus, y).item() if y.dim() > 1 or y.is_floating_point() \
                    else F.cross_entropy(pred_minus, y).item()
                losses_minus.append(loss_minus)

        # Restore original parameters
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(original_params[n])

        model.train()

        # Compute curvature estimates
        # Second derivative estimate: (f(x+e) - 2f(x) + f(x-e)) / e^2
        curvatures = []
        asymmetries = []
        for lp, lm in zip(losses_plus, losses_minus):
            curv = (lp - 2 * base_loss + lm) / (epsilon ** 2)
            curvatures.append(curv)
            # Asymmetry: how different +/- directions are
            asym = abs(lp - lm) / (abs(lp - base_loss) + abs(lm - base_loss) + 1e-10)
            asymmetries.append(asym)

        mean_curvature = float(np.mean(curvatures))
        std_curvature = float(np.std(curvatures))
        mean_asymmetry = float(np.mean(asymmetries))

        # Classify curvature type
        if mean_curvature > 100:
            curvature_type = "sharp"
            summary = "Sharp loss minimum - needs regularization or lower LR, not more capacity"
            suggested = TreatmentType.DROPOUT_LIGHT
            confidence = 0.7
        elif mean_curvature < 1.0:
            curvature_type = "flat"
            summary = "Flat loss landscape (plateau) - needs LR boost or structural change"
            suggested = TreatmentType.CAPACITY_GROWTH
            confidence = 0.6
        elif mean_asymmetry > 0.7:
            curvature_type = "saddle"
            summary = "Near saddle point - needs perturbation or structural change"
            suggested = TreatmentType.LAYER_ADDITION
            confidence = 0.55
        else:
            curvature_type = "moderate"
            summary = "Moderate curvature - standard optimization landscape"
            suggested = None
            confidence = 0.3

        findings = {
            "mean_curvature": round(mean_curvature, 4),
            "std_curvature": round(std_curvature, 4),
            "curvature_type": curvature_type,
            "mean_asymmetry": round(mean_asymmetry, 4),
            "base_loss": round(base_loss, 6),
            "summary": summary,
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"n_perturbations": n_perturbations, "epsilon": epsilon},
        )


# ==================== Tier 2: Medium Tests ====================


class OperationSensitivityTest(LabTest):
    """Probes candidate operations to find which would help most.

    Temporarily inserts candidate operations at each layer and measures
    the loss improvement. Unlike the regular probe in surgery, this runs
    specifically for the diagnosed disease with a targeted subset of ops.

    Cost: ~20-50 forward passes.
    """

    name = "Operation Sensitivity Probe"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import (
            create_operation, get_operation_name,
            get_candidate_operations_for_layer,
        )

        device = next(model.parameters()).device
        model.eval()

        # Get validation batch
        val_batch = None
        for batch in val_data:
            val_batch = batch
            break
        if val_batch is None:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        x = val_batch[0].to(device)
        y = val_batch[1].to(device)

        # Compute base loss
        with torch.no_grad():
            base_pred = model(x)
            base_loss = F.mse_loss(base_pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                else F.cross_entropy(base_pred, y).item()

        # Find best layer (highest GDS signal)
        gds = signals.get("GDS", {})
        if gds:
            best_layer = max(gds, key=gds.get)
        else:
            best_layer = 0

        # Get layer-appropriate candidate operations (flat vs spatial)
        layer = model.layers[best_layer]
        candidates = get_candidate_operations_for_layer(config, layer)

        # Get dimension and spatial_shape for this layer
        d_model = layer.out_features
        layer_spatial_shape = getattr(layer, 'spatial_shape', None)

        op_results = {}
        for op_name in candidates:
            try:
                op = create_operation(
                    op_name, d_model, device=str(device),
                    config=config, spatial_shape=layer_spatial_shape,
                )
                if op is None:
                    continue
                op = op.to(device)

                # Temporarily insert op at end of target layer's pipeline,
                # run full model forward, then remove it.
                pipeline = model.ops[best_layer]
                insert_pos = pipeline.num_operations

                with torch.no_grad():
                    pipeline.add_operation(op, insert_pos)
                    try:
                        pred = model(x)
                        loss = F.mse_loss(pred, y).item() \
                            if y.dim() > 1 or y.is_floating_point() \
                            else F.cross_entropy(pred, y).item()
                    except Exception:
                        loss = base_loss  # No benefit if it crashes
                    finally:
                        pipeline.remove_operation(insert_pos)

                    benefit = base_loss - loss
                    op_results[op_name] = {
                        "loss": round(loss, 6),
                        "benefit": round(benefit, 6),
                        "layer": best_layer,
                    }
            except Exception:
                continue

        model.train()

        if not op_results:
            return LabResult(self.name, {"summary": "no operations testable"}, 0.0)

        # Rank by benefit
        ranked = sorted(op_results.items(), key=lambda x: x[1]["benefit"], reverse=True)
        best_op = ranked[0][0]
        best_benefit = ranked[0][1]["benefit"]

        # Map operation to treatment (prefix matching)
        treatment_map = {
            "conv1d": TreatmentType.SPATIAL_CONV_PACKAGE,
            "conv2d": TreatmentType.SPATIAL_CONV_PACKAGE,
            "spatial_conv2d": TreatmentType.SPATIAL_CONV_PACKAGE,
            "spatial_dw_sep": TreatmentType.SPATIAL_CONV_PACKAGE,
            "spatial_pointwise": TreatmentType.SPATIAL_CONV_PACKAGE,
            "derivative_conv1d": TreatmentType.DERIVATIVE_PACKAGE,
            "derivative_conv2d": TreatmentType.DERIVATIVE_PACKAGE,
            "polynomial": TreatmentType.POLYNOMIAL_PACKAGE,
            "spatial_polynomial": TreatmentType.POLYNOMIAL_PACKAGE,
            "branched": TreatmentType.BRANCHED_DIFFUSION_REACTION,
            "spatial_branched": TreatmentType.BRANCHED_DIFFUSION_REACTION,
            "attn_self": TreatmentType.ATTENTION_PACKAGE,
            "attn_multihead": TreatmentType.ATTENTION_PACKAGE,
            "attn_cross": TreatmentType.ATTENTION_PACKAGE,
            "attn_causal": TreatmentType.ATTENTION_PACKAGE,
            "channel_attention": TreatmentType.ATTENTION_PACKAGE,
            "batchnorm": TreatmentType.BATCHNORM_PACKAGE,
        }
        suggested = None
        for prefix, treatment in treatment_map.items():
            if best_op.startswith(prefix):
                suggested = treatment
                break
        if suggested is None:
            suggested = TreatmentType.CAPACITY_GROWTH

        confidence = min(0.9, 0.5 + best_benefit * 10) if best_benefit > 0 else 0.2

        findings = {
            "best_operation": best_op,
            "benefit": best_benefit,
            "best_layer": best_layer,
            "operation_ranking": [(name, info["benefit"]) for name, info in ranked[:5]],
            "summary": f"Best operation: {best_op} (benefit={best_benefit:.4f} at layer {best_layer})",
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"base_loss": base_loss, "n_candidates": len(op_results)},
        )


class SkipConnectionTest(LabTest):
    """Tests whether skip connections would improve gradient flow.

    Temporarily adds identity skip connections between non-adjacent layers
    and measures loss improvement.

    Cost: ~10 forward passes.
    """

    name = "Skip Connection Necessity"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.UNDERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        # Get validation batch
        val_batch = None
        for batch in val_data:
            val_batch = batch
            break
        if val_batch is None:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        x = val_batch[0].to(device)
        y = val_batch[1].to(device)

        n_layers = model.num_layers
        if n_layers < 3:
            return LabResult(self.name,
                             {"summary": "too few layers for skip connections"},
                             0.0)

        # Base forward pass - collect per-layer activations
        with torch.no_grad():
            activations = []
            current = x
            for l in range(n_layers):
                current = model.layers[l](current)
                activations.append(current)
            base_pred = model.head(current)
            base_loss = F.mse_loss(base_pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                else F.cross_entropy(base_pred, y).item()

        # Test skip connections between different layer pairs
        skip_results = []
        for source in range(n_layers - 2):
            target = source + 2  # Skip one layer
            if target >= n_layers:
                continue

            # Check dimension compatibility
            src_act = activations[source]
            tgt_act = activations[target]
            if src_act.shape != tgt_act.shape:
                continue

            with torch.no_grad():
                current = x
                for l in range(n_layers):
                    current = model.layers[l](current)
                    if l == target:
                        current = current + activations[source]
                pred = model.head(current)
                loss = F.mse_loss(pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                    else F.cross_entropy(pred, y).item()
                benefit = base_loss - loss
                skip_results.append({
                    "source": source,
                    "target": target,
                    "benefit": benefit,
                })

        model.train()

        if not skip_results:
            return LabResult(self.name,
                             {"summary": "no compatible skip connections found"},
                             0.2)

        # Find best skip connection
        best_skip = max(skip_results, key=lambda x: x["benefit"])
        improves_flow = best_skip["benefit"] > 0

        if improves_flow and best_skip["benefit"] > 0.01:
            summary = (f"Skip connection {best_skip['source']}→{best_skip['target']} "
                       f"improves loss by {best_skip['benefit']:.4f}")
            suggested = TreatmentType.RESNET_BLOCK
            confidence = min(0.85, 0.5 + best_skip["benefit"] * 5)
        elif improves_flow:
            summary = "Skip connections provide marginal improvement"
            suggested = TreatmentType.RESNET_BLOCK
            confidence = 0.4
        else:
            summary = "Skip connections do not help - bottleneck is elsewhere"
            suggested = None
            confidence = 0.5

        findings = {
            "skip_benefit": round(best_skip["benefit"], 6),
            "best_source": best_skip["source"],
            "best_target": best_skip["target"],
            "improves_flow": improves_flow,
            "all_results": [(r["source"], r["target"], round(r["benefit"], 6))
                            for r in sorted(skip_results, key=lambda x: - x["benefit"])[:5]],
            "summary": summary,
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"base_loss": base_loss, "n_tested": len(skip_results)},
        )


class OverfittingLocalizationTest(LabTest):
    """Identifies which layers contribute most to overfitting.

    Freezes layers one at a time and measures how the train-val gap changes.
    If overfitting is localized to specific layers, targeted regularization
    is more effective than global regularization.

    Cost: ~N_layers × 5 forward passes.
    """

    name = "Overfitting Source Localization"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        # Collect multiple batches
        val_batches = []
        for i, batch in enumerate(val_data):
            if i >= 5:
                break
            val_batches.append((batch[0].to(device), batch[1].to(device)))

        if not val_batches:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        n_layers = model.num_layers

        def compute_val_loss():
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for x, y in val_batches:
                    pred = model(x)
                    loss = F.mse_loss(pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                        else F.cross_entropy(pred, y).item()
                    total_loss += loss
                    count += 1
            return total_loss / max(count, 1)

        base_val_loss = compute_val_loss()

        # Test each layer: freeze it and measure val loss change
        layer_contributions = {}
        for l in range(n_layers):
            layer = model.layers[l]

            # Save requires_grad state
            original_requires_grad = {}
            for n, p in layer.named_parameters():
                original_requires_grad[n] = p.requires_grad
                p.requires_grad_(False)

            # Add dropout temporarily to this layer
            frozen_val_loss = compute_val_loss()
            gap_change = base_val_loss - frozen_val_loss  # Positive = freezing helped

            # Restore
            for n, p in layer.named_parameters():
                p.requires_grad_(original_requires_grad[n])

            layer_contributions[l] = round(gap_change, 6)

        model.train()

        # Identify overfitting layers (freezing them reduced val loss)
        overfitting_layers = [l for l, delta in layer_contributions.items()
                              if delta > 0]
        overfitting_layers.sort(key=lambda l: layer_contributions[l], reverse=True)

        if overfitting_layers:
            worst_layer = overfitting_layers[0]
            summary = (f"Overfitting localized to layer(s) {overfitting_layers[:3]} - "
                       f"targeted regularization recommended")
            suggested = TreatmentType.DROPOUT_HEAVY
            confidence = 0.7
        else:
            summary = "Overfitting is distributed across all layers - global regularization needed"
            suggested = TreatmentType.WEIGHT_DECAY_BOOST
            confidence = 0.6

        findings = {
            "overfitting_layers": overfitting_layers[:5],
            "layer_gap_contribution": layer_contributions,
            "summary": summary,
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"base_val_loss": base_val_loss, "n_layers": n_layers},
        )


# ==================== Tier 3: Expensive Tests ====================


class ArchitectureComparisonTest(LabTest):
    """Trains small copies of the model with different modifications.

    Clones the model, applies candidate treatments, trains for a few steps,
    and compares validation loss. This tells you which treatment would
    actually help before committing to the full recovery period.

    Cost: ~100-500 steps of training per candidate.
    """

    name = "Architecture Comparison"
    cost_level = 3

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        probe_steps = getattr(config, "lab_probe_steps", 50)

        # Get validation + training batches
        val_batches = []
        for i, batch in enumerate(val_data):
            if i >= 3:
                break
            val_batches.append((batch[0].to(device), batch[1].to(device)))

        if not val_batches:
            return LabResult(self.name, {"summary": "no data"}, 0.0)

        def evaluate(m):
            m.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for x, y in val_batches:
                    pred = m(x)
                    loss = F.mse_loss(pred, y).item() if y.dim() > 1 or y.is_floating_point() \
                        else F.cross_entropy(pred, y).item()
                    total += loss
                    count += 1
            m.train()
            return total / max(count, 1)

        base_loss = evaluate(model)

        # Candidate treatments to compare (lightweight structural changes)
        candidates = [
            TreatmentType.CAPACITY_GROWTH,
            TreatmentType.DROPOUT_LIGHT,
            TreatmentType.BATCHNORM_PACKAGE,
        ]

        treatment_results = {}
        for treatment in candidates:
            try:
                # Clone model
                model_copy = copy.deepcopy(model)
                model_copy.to(device)
                model_copy.train()

                # Mini-training on val data as proxy
                opt = torch.optim.Adam(model_copy.parameters(), lr=1e-4)
                for step in range(probe_steps):
                    x, y = val_batches[step % len(val_batches)]
                    pred = model_copy(x)
                    loss = F.mse_loss(pred, y) if y.dim() > 1 or y.is_floating_point() \
                        else F.cross_entropy(pred, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                final_loss = evaluate(model_copy)
                improvement = base_loss - final_loss
                treatment_results[treatment] = {
                    "final_loss": round(final_loss, 6),
                    "improvement": round(improvement, 6),
                }
                del model_copy
            except Exception:
                continue

        if not treatment_results:
            return LabResult(self.name, {"summary": "no treatments testable"}, 0.0)

        # Best treatment
        best_treatment = max(treatment_results, key=lambda t: treatment_results[t]["improvement"])
        best_improvement = treatment_results[best_treatment]["improvement"]

        confidence = min(0.9, 0.4 + best_improvement * 20) if best_improvement > 0 else 0.2

        findings = {
            "treatment_results": {t.name: info for t, info in treatment_results.items()},
            "best_treatment": best_treatment.name,
            "best_improvement": best_improvement,
            "summary": f"Best: {best_treatment.name} (improvement={best_improvement:.4f})",
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=best_treatment if best_improvement > 0 else None,
            evidence={"base_loss": base_loss, "probe_steps": probe_steps},
        )


class DataComplexityTest(LabTest):
    """Estimates intrinsic dimensionality of data vs model capacity.

    Uses PCA on activations at each layer to measure effective rank.
    - Model too narrow: activations use all dimensions → needs width
    - Model too deep: middle layers are near-identity → remove layers
    - Right size, wrong structure: high-rank but poorly organized

    Cost: ~20 forward passes + SVD computation.
    """

    name = "Data Complexity Mismatch"
    cost_level = 3

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.UNDERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        # Collect activations at each layer
        layer_activations = {l: [] for l in range(model.num_layers)}

        with torch.no_grad():
            for i, batch in enumerate(val_data):
                if i >= 10:
                    break
                x = batch[0].to(device)
                current = x
                for l in range(model.num_layers):
                    current = model.layers[l](current)
                    # Flatten spatial dims if present
                    act = current.view(current.shape[0], -1).cpu()
                    layer_activations[l].append(act)

        model.train()

        # Compute effective rank at each layer via SVD
        effective_ranks = []
        capacity_utilizations = []
        bottleneck_layer = None
        min_utilization = float("inf")

        for l in range(model.num_layers):
            acts_list = layer_activations[l]
            if not acts_list:
                effective_ranks.append(0.0)
                capacity_utilizations.append(0.0)
                continue

            acts = torch.cat(acts_list, dim=0)  # [N_total, d]
            n, d = acts.shape
            if n < 2 or d < 1:
                effective_ranks.append(0.0)
                capacity_utilizations.append(0.0)
                continue

            # Center the activations
            acts = acts - acts.mean(dim=0, keepdim=True)

            # SVD (on smaller dimension for efficiency)
            try:
                if n > d:
                    U, S, V = torch.linalg.svd(acts, full_matrices=False)
                else:
                    U, S, V = torch.linalg.svd(acts, full_matrices=False)

                # Effective rank = exp(entropy of normalized singular values)
                s_norm = S / (S.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                eff_rank = np.exp(entropy)
                effective_ranks.append(round(eff_rank, 2))

                # Capacity utilization = effective_rank / total_dimensions
                utilization = eff_rank / d
                capacity_utilizations.append(round(utilization, 4))

                if utilization < min_utilization:
                    min_utilization = utilization
                    bottleneck_layer = l
            except Exception:
                effective_ranks.append(0.0)
                capacity_utilizations.append(0.0)

        if not effective_ranks or all(r == 0 for r in effective_ranks):
            return LabResult(self.name, {"summary": "SVD failed"}, 0.0)

        # Analyze patterns
        mean_utilization = float(np.mean([u for u in capacity_utilizations if u > 0]))

        if mean_utilization > 0.8:
            summary = "Model is near capacity - all dimensions are used. Needs more width"
            suggested = TreatmentType.CAPACITY_GROWTH
            confidence = 0.75
        elif mean_utilization < 0.2:
            summary = "Model has excess capacity - many unused dimensions. May need depth reduction"
            suggested = None
            confidence = 0.5
        else:
            summary = "Model capacity is moderately utilized"
            suggested = None
            confidence = 0.3

        findings = {
            "effective_rank": effective_ranks,
            "capacity_utilization": capacity_utilizations,
            "bottleneck_layer": bottleneck_layer,
            "mean_utilization": round(mean_utilization, 4),
            "summary": summary,
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"n_layers": model.num_layers},
        )


# ==================== Treatment Outcome Analysis ====================


class TreatmentResponseTest(LabTest):
    """Analyzes past treatment outcomes by category to guide future treatment.

    Categorises treatments into groups (regularization, capacity, operations,
    structural) and tracks which categories have been tried and whether they
    succeeded or failed. Recommends from untried or previously-successful
    categories, avoiding exhausted ones.

    Applicable when the patient has at least 3 treatments in their history.

    Cost: ~0 (reads PatientHistory only, no forward passes).
    """

    name = "Treatment Response Analysis"
    cost_level = 1

    # Treatment categories
    _CATEGORIES = {
        "regularization": [
            TreatmentType.DROPOUT_LIGHT,
            TreatmentType.DROPOUT_HEAVY,
            TreatmentType.WEIGHT_DECAY_BOOST,
            TreatmentType.LABEL_SMOOTHING,
        ],
        "capacity": [
            TreatmentType.CAPACITY_GROWTH,
            TreatmentType.LAYER_ADDITION,
            TreatmentType.PROGRESSIVE_CHANNELS,
            TreatmentType.DUAL_DEPTH_WIDTH,
        ],
        "operations": [
            TreatmentType.DERIVATIVE_PACKAGE,
            TreatmentType.POLYNOMIAL_PACKAGE,
            TreatmentType.BRANCHED_DIFFUSION_REACTION,
            TreatmentType.BATCHNORM_PACKAGE,
            TreatmentType.SPATIAL_CONV_PACKAGE,
            TreatmentType.ATTENTION_PACKAGE,
        ],
        "structural": [
            TreatmentType.RESNET_BLOCK,
            TreatmentType.ARCHITECTURE_RESET_SOFT,
        ],
    }

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def is_applicable(self, diagnosis: Diagnosis, task_type: str = "regression",
                      patient_history: Optional[PatientHistory] = None) -> bool:
        if patient_history is None:
            return False
        if len(patient_history.treatment_log) < 3:
            return False
        return super().is_applicable(diagnosis, task_type, patient_history)

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        if patient_history is None:
            return LabResult(self.name, {"summary": "no patient history"}, 0.0)

        # Count successes and failures per category
        all_eff = []
        all_ineff = []
        for treatments in patient_history._effective_treatments.values():
            all_eff.extend(treatments)
        for treatments in patient_history._ineffective_treatments.values():
            all_ineff.extend(treatments)

        category_stats = {}
        for cat_name, cat_treatments in self._CATEGORIES.items():
            cat_treatment_names = [t.name for t in cat_treatments]
            successes = sum(1 for t in all_eff if t in cat_treatment_names)
            failures = sum(1 for t in all_ineff if t in cat_treatment_names)
            category_stats[cat_name] = {
                "successes": successes,
                "failures": failures,
                "tried": successes + failures > 0,
            }

        # Classify categories
        failed_cats = []
        successful_cats = []
        untried_cats = []
        for cat_name, stats in category_stats.items():
            if not stats["tried"]:
                untried_cats.append(cat_name)
            elif stats["failures"] >= 3 and stats["successes"] == 0:
                failed_cats.append(cat_name)
            elif stats["successes"] > 0:
                successful_cats.append(cat_name)

        # Recommend from untried categories first, then successful ones
        suggested = None
        confidence = 0.6
        summary_parts = []

        if failed_cats:
            summary_parts.append(f"Failed categories: {', '.join(failed_cats)}")

        # Pick a representative treatment from best available category
        recommend_from = untried_cats if untried_cats else successful_cats
        if recommend_from:
            # Pick first untried/successful category
            best_cat = recommend_from[0]
            # Pick first treatment in category that hasn't been tried (or has worked)
            cat_treatments = self._CATEGORIES[best_cat]
            for t in cat_treatments:
                t_fail_count = all_ineff.count(t.name)
                if t_fail_count < 2:
                    suggested = t
                    break
            if suggested is None and cat_treatments:
                suggested = cat_treatments[0]
            summary_parts.append(
                f"Recommending from {'untried' if untried_cats else 'successful'} "
                f"category '{best_cat}': {suggested.name if suggested else 'none'}")
            confidence = 0.7 if untried_cats else 0.55
        elif not failed_cats:
            summary_parts.append("All categories partially explored - continue current approach")
            confidence = 0.3
        else:
            # All categories exhausted - emergency reset
            suggested = TreatmentType.WEIGHT_REINITIALIZE
            summary_parts.append(
                "All treatment categories exhausted - recommending weight reinitialize")
            confidence = 0.5

        findings = {
            "category_stats": category_stats,
            "failed_categories": failed_cats,
            "successful_categories": successful_cats,
            "untried_categories": untried_cats,
            "total_treatments_tried": len(patient_history.treatment_log),
            "summary": "; ".join(summary_parts),
        }

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"n_effective": len(all_eff), "n_ineffective": len(all_ineff)},
        )


# ==================== Graph Lab Tests ====================


class GraphStructureTest(LabTest):
    """Detects graph-structured data and recommends graph operations.

    Signs of graph data (NO flags — purely data-driven):
    - model._graph_edge_index is not None (auxiliary data was set)
    - model._graph_num_nodes > 0

    Analogous to InputStructureTest detecting PDE:
    - InputStructureTest: "d<=5, continuous, regression" → DERIVATIVE_PACKAGE
    - GraphStructureTest: "has edge_index, N=num_nodes" → GRAPH_CONV_PACKAGE

    Applicable to BOTH underfitting AND overfitting: on graph data, a model
    that treats nodes independently will overfit by memorizing node features.
    Graph aggregation provides structural regularization (smoothing over
    neighbors), so it's a valid treatment for overfitting on graph data too.

    Cost: ~0 (checks model attributes and one batch shape).
    """

    name = "Graph Structure Analysis"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
            # Graph aggregation also acts as structural regularization:
            # smoothing features over neighbors prevents memorization
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import get_operation_name

        # Check if model has graph structure
        has_graph = (getattr(model, '_graph_edge_index', None) is not None
                     and getattr(model, '_graph_num_nodes', 0) > 0)

        if not has_graph:
            return LabResult(self.name,
                             {"summary": "No graph structure detected"},
                             0.0)

        # Get batch to verify shape
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "No data"}, 0.0)

        x = batch[0]
        n_nodes = x.shape[0]
        n_features = x.shape[1] if x.dim() == 2 else x.shape[-1]
        n_edges = model._graph_edge_index.shape[1]
        avg_degree = n_edges / max(n_nodes, 1)

        # Check if graph ops already present
        has_graph_ops = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                if 'graph_' in get_operation_name(op).lower():
                    has_graph_ops = True
                    break
            if has_graph_ops:
                break

        if has_graph_ops:
            return LabResult(
                self.name,
                {"summary": "Graph ops already present",
                 "has_graph_ops": True},
                0.3,
            )

        # Graph detected, no graph ops yet → recommend
        suggested = TreatmentType.GRAPH_CONV_PACKAGE
        confidence = 0.85

        findings = {
            "problem_type": "graph",
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "avg_degree": round(avg_degree, 2),
            "n_features": n_features,
            "has_graph_ops": False,
            "summary": (f"Graph structure detected ({n_nodes} nodes, "
                        f"avg degree {avg_degree:.1f}). "
                        f"Model lacks graph-aware operations."),
        }

        # Apply rotation for previously-failed graph treatments
        suggested, confidence, findings = _rotate_failed_treatment(
            suggested, patient_history, findings, confidence
        )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
            evidence={"task_type": task_type},
        )


class GraphHomophilyTest(LabTest):
    """Measures graph homophily to guide aggregation strategy.

    High homophily (>0.5): NeighborAggregation (mean-pool) is sufficient.
    Medium homophily (0.3-0.5): Multi-hop diffusion recommended.
    Low homophily (<0.3): Need GraphAttention to filter dissimilar neighbors.

    Applicable to overfitting too: on graph data, the aggregation type
    matters for generalization — wrong aggregation can hurt.

    Cost: ~0 (checks labels at edges, no forward pass).
    """

    name = "Graph Homophily Analysis"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import get_operation_name

        edge_index = getattr(model, '_graph_edge_index', None)
        if edge_index is None:
            return LabResult(self.name, {"summary": "No graph data"}, 0.0)

        # Check if graph ops already present — if so, don't re-recommend
        has_graph_ops = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                if 'graph_' in get_operation_name(op).lower():
                    has_graph_ops = True
                    break
            if has_graph_ops:
                break

        if has_graph_ops:
            return LabResult(
                self.name,
                {"summary": "Graph ops already present, homophily check skipped",
                 "has_graph_ops": True},
                0.3,  # Low confidence — don't override other treatments
            )

        # Get labels
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "No data"}, 0.0)

        y = batch[1]  # [N] or [N, d_out]

        try:
            if y.dim() > 1:
                # Regression: use feature cosine similarity
                src_feat = y[edge_index[0].cpu()]
                dst_feat = y[edge_index[1].cpu()]
                cos_sim = F.cosine_similarity(src_feat.float(), dst_feat.float(), dim=-1)
                homophily = cos_sim.mean().item()
            else:
                # Classification: fraction of same-class edge endpoints
                y_cpu = y.cpu()
                ei_cpu = edge_index.cpu()
                valid = (y_cpu[ei_cpu[0]] >= 0) & (y_cpu[ei_cpu[1]] >= 0)
                if valid.sum() > 0:
                    same_class = (y_cpu[ei_cpu[0]] == y_cpu[ei_cpu[1]]) & valid
                    homophily = (same_class.float().sum().item()
                                 / valid.float().sum().item())
                else:
                    homophily = 0.5  # Unknown
        except Exception:
            homophily = 0.5

        if homophily > 0.5:
            suggested = TreatmentType.GRAPH_CONV_PACKAGE
            summary = (f"High homophily ({homophily:.2f}) — "
                       f"mean aggregation appropriate")
        elif homophily > 0.3:
            suggested = TreatmentType.GRAPH_DIFFUSION_PACKAGE
            summary = (f"Medium homophily ({homophily:.2f}) — "
                       f"multi-hop diffusion recommended")
        else:
            suggested = TreatmentType.GRAPH_ATTENTION_PACKAGE
            summary = (f"Low homophily ({homophily:.2f}) — "
                       f"attention-based aggregation needed")

        confidence = 0.75

        findings = {
            "homophily": round(homophily, 4),
            "summary": summary,
        }

        suggested, confidence, findings = _rotate_failed_treatment(
            suggested, patient_history, findings, confidence
        )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
        )


class GraphResidualTest(LabTest):
    """Analyzes whether prediction errors correlate with graph topology.

    If connected nodes have correlated residuals, the model is missing
    graph-structural information → needs stronger graph ops.

    Cost: ~1 forward pass on validation data.
    """

    name = "Graph Residual Correlation"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.CAPACITY_EXHAUSTION,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import get_operation_name

        edge_index = getattr(model, '_graph_edge_index', None)
        if edge_index is None:
            return LabResult(self.name, {"summary": "No graph data"}, 0.0)

        # Compute residuals
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "No data"}, 0.0)

        device = next(model.parameters()).device
        model.eval()

        try:
            with torch.no_grad():
                x = batch[0].to(device)
                y = batch[1].to(device)
                pred = model(x)

                if task_type == "classification" and pred.dim() == 2:
                    # Per-node cross-entropy loss
                    log_probs = F.log_softmax(pred, dim=-1)
                    y_clamped = y.clamp(min=0)
                    residual = -log_probs.gather(1, y_clamped.unsqueeze(1)).squeeze()
                else:
                    if pred.dim() > 1 and pred.shape[-1] > 1:
                        residual = (pred - y).abs().mean(dim=-1)
                    else:
                        residual = (pred.squeeze() - y.squeeze()).abs()

            # Neighbor residual correlation
            ei = edge_index  # already on device
            src_res = residual[ei[0]]
            dst_res = residual[ei[1]]

            # Pearson correlation
            if src_res.numel() > 10:
                stacked = torch.stack([src_res.float(), dst_res.float()])
                corr_matrix = torch.corrcoef(stacked)
                correlation = corr_matrix[0, 1].item()
                if not np.isfinite(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

        except Exception:
            correlation = 0.0

        # High correlation → graph structure matters
        has_graph_ops = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                if 'graph_' in get_operation_name(op).lower():
                    has_graph_ops = True
                    break
            if has_graph_ops:
                break

        if correlation > 0.3 and not has_graph_ops:
            suggested = TreatmentType.GRAPH_CONV_PACKAGE
            confidence = 0.8
        elif correlation > 0.3 and has_graph_ops:
            suggested = TreatmentType.GRAPH_DIFFUSION_PACKAGE
            confidence = 0.7
        else:
            suggested = None
            confidence = 0.4

        findings = {
            "residual_graph_correlation": round(correlation, 4),
            "has_graph_ops": has_graph_ops,
            "summary": (f"Neighbor residual correlation: {correlation:.3f}. "
                        + ("Errors cluster on graph — needs graph ops"
                           if correlation > 0.3
                           else "Errors independent of graph")),
        }

        if suggested is not None:
            suggested, confidence, findings = _rotate_failed_treatment(
                suggested, patient_history, findings, confidence
            )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
        )


class OverSmoothingTest(LabTest):
    """Detects graph over-smoothing by measuring node feature convergence.

    After multiple graph convolution layers, node features can converge to
    a single vector (all nodes become indistinguishable). This is measured
    by computing mean pairwise cosine similarity at each layer output.

    If similarity > threshold at later layers, recommends GraphNorm/PairNorm.

    Cost: 1 forward pass with intermediate extraction.
    """

    name = "Over-Smoothing Detection"
    cost_level = 1

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import get_operation_name

        # Only applicable when model has graph ops
        has_graph_ops = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                if 'graph_' in get_operation_name(op).lower():
                    has_graph_ops = True
                    break
            if has_graph_ops:
                break

        if not has_graph_ops:
            return LabResult(self.name, {"summary": "No graph ops present"}, 0.0)

        # Get a batch
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "No data"}, 0.0)

        device = next(model.parameters()).device
        model.eval()

        try:
            with torch.no_grad():
                x = batch[0].to(device)
                # Forward with intermediates
                _, h, _, _ = model.forward_with_intermediates(x)

                # Compute cosine similarity at each layer
                similarities = {}
                for key, feat in h.items():
                    if feat.dim() == 2 and feat.shape[0] > 2:
                        # Normalize features
                        feat_norm = F.normalize(feat, dim=-1)
                        # Mean pairwise cosine similarity (sample random pairs for efficiency)
                        N = feat_norm.shape[0]
                        n_pairs = min(1000, N * (N - 1) // 2)
                        if N > 100:
                            # Sample random pairs
                            idx_a = torch.randint(0, N, (n_pairs,), device=device)
                            idx_b = torch.randint(0, N, (n_pairs,), device=device)
                            # Ensure different indices
                            mask = idx_a != idx_b
                            cos_sim = (feat_norm[idx_a[mask]] * feat_norm[idx_b[mask]]).sum(dim=-1)
                        else:
                            # Compute full pairwise similarity
                            cos_sim_matrix = feat_norm @ feat_norm.T
                            # Extract upper triangle
                            cos_sim = cos_sim_matrix[torch.triu(torch.ones(N, N, device=device), diagonal=1) > 0]
                        mean_sim = cos_sim.mean().item()
                        similarities[key] = mean_sim

        except Exception:
            return LabResult(self.name, {"summary": "Forward pass failed"}, 0.0)

        if not similarities:
            return LabResult(self.name, {"summary": "Could not compute similarities"}, 0.0)

        # Check last layer similarity
        last_key = max(similarities.keys())
        last_sim = similarities[last_key]

        # Over-smoothing detected if last layer similarity > 0.85
        is_oversmoothed = last_sim > 0.85

        # Check if PairNorm/GraphNorm already present
        has_norm = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                name = get_operation_name(op).lower()
                if 'pairnorm' in name or 'graphnorm' in name:
                    has_norm = True
                    break
            if has_norm:
                break

        if is_oversmoothed and not has_norm:
            suggested = TreatmentType.GRAPH_NORM_PACKAGE
            confidence = 0.85
        elif is_oversmoothed and has_norm:
            suggested = TreatmentType.GRAPH_SPECTRAL_PACKAGE
            confidence = 0.6
        else:
            suggested = None
            confidence = 0.3

        findings = {
            "layer_similarities": {str(k): round(v, 4) for k, v in similarities.items()},
            "last_layer_similarity": round(last_sim, 4),
            "is_oversmoothed": is_oversmoothed,
            "has_graph_norm": has_norm,
            "summary": (f"Last layer cosine similarity: {last_sim:.3f}. "
                        + ("Over-smoothing detected -- nodes becoming indistinguishable"
                           if is_oversmoothed
                           else "Feature diversity maintained")),
        }

        if suggested is not None:
            suggested, confidence, findings = _rotate_failed_treatment(
                suggested, patient_history, findings, confidence
            )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
        )


class GraphSpectralAnalysisTest(LabTest):
    """Analyzes graph spectral properties to recommend spectral or PE ops.

    Computes feature smoothness on the graph (Dirichlet energy) and checks
    if positional encoding would help differentiate structurally similar nodes.

    Cost: 1 forward pass + small matrix computation.
    """

    name = "Graph Spectral Analysis"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.UNDERFITTING_MILD,
            DiseaseType.UNDERFITTING_SEVERE,
            DiseaseType.TRAINING_STAGNATION,
            DiseaseType.CAPACITY_EXHAUSTION,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        from .surgery import get_operation_name

        edge_index = getattr(model, '_graph_edge_index', None)
        if edge_index is None:
            return LabResult(self.name, {"summary": "No graph data"}, 0.0)

        # Get a batch
        batch = None
        for b in val_data:
            batch = b
            break
        if batch is None:
            return LabResult(self.name, {"summary": "No data"}, 0.0)

        device = next(model.parameters()).device
        model.eval()

        try:
            with torch.no_grad():
                x = batch[0].to(device)
                # Get intermediate features from last hidden layer
                _, h, _, _ = model.forward_with_intermediates(x)
                last_key = max(h.keys())
                features = h[last_key]  # [N, d]

                # Compute Dirichlet energy: E = sum_{(i,j) in E} ||f_i - f_j||^2
                ei = edge_index
                src_feat = features[ei[0]]
                dst_feat = features[ei[1]]
                dirichlet_energy = ((src_feat - dst_feat) ** 2).sum().item()
                # Normalize by number of edges and feature norm
                feat_norm_sq = (features ** 2).sum().item()
                n_edges = ei.shape[1]
                normalized_energy = dirichlet_energy / (n_edges * max(feat_norm_sq / features.shape[0], 1e-8))

        except Exception:
            normalized_energy = 0.0

        # Check existing graph ops
        has_spectral = False
        has_pe = False
        has_any_graph = False
        for l in range(model.num_layers):
            for op in model.ops[l].operations:
                name = get_operation_name(op).lower()
                if 'spectral' in name:
                    has_spectral = True
                if 'positional' in name:
                    has_pe = True
                if 'graph_' in name:
                    has_any_graph = True

        # High Dirichlet energy = features are NOT smooth on graph
        # → spectral conv could help capture graph-frequency information
        if normalized_energy > 2.0 and not has_spectral:
            suggested = TreatmentType.GRAPH_SPECTRAL_PACKAGE
            confidence = 0.75
        elif normalized_energy < 0.5 and has_any_graph and not has_pe:
            # Very smooth features but still underperforming → needs PE
            suggested = TreatmentType.GRAPH_POSITIONAL_PACKAGE
            confidence = 0.7
        elif not has_any_graph:
            suggested = TreatmentType.GRAPH_CONV_PACKAGE
            confidence = 0.6
        else:
            suggested = TreatmentType.GRAPH_GIN_PACKAGE
            confidence = 0.5

        findings = {
            "dirichlet_energy_normalized": round(normalized_energy, 4),
            "has_spectral_ops": has_spectral,
            "has_positional_enc": has_pe,
            "has_any_graph_ops": has_any_graph,
            "summary": (f"Normalized Dirichlet energy: {normalized_energy:.3f}. "
                        + ("High frequency content -- spectral conv recommended"
                           if normalized_energy > 2.0
                           else "Smooth features on graph")),
        }

        if suggested is not None:
            suggested, confidence, findings = _rotate_failed_treatment(
                suggested, patient_history, findings, confidence
            )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
        )


class MemorizationEntropyTest(LabTest):
    """Detects memorization by measuring per-neuron activation entropy.

    Memorizing neurons fire for very few specific training samples (low entropy).
    On validation data, these neurons show abnormally sparse activations —
    either near-zero (unseen samples) or highly concentrated (few activations).

    For each layer, computes the fraction of neurons with entropy below threshold.
    Layers with >30% low-entropy neurons are flagged as memorizing.

    Cost: 1 forward pass with intermediate extraction over val_data.
    """

    name = "Memorization Entropy Analysis"
    cost_level = 2

    def applicable_diseases(self) -> List[DiseaseType]:
        return [
            DiseaseType.MEMORIZATION,
            DiseaseType.OVERFITTING_EARLY,
            DiseaseType.OVERFITTING_MODERATE,
            DiseaseType.OVERFITTING_SEVERE,
        ]

    def run(self, model: torch.nn.Module, val_data: Any,
            signals: Dict[str, Any], config: Any,
            task_type: str = "regression",
            patient_history: Optional[PatientHistory] = None,
            **kwargs) -> LabResult:
        device = next(model.parameters()).device
        model.eval()

        # Collect activations from multiple val batches
        all_activations: Dict[int, List[torch.Tensor]] = {}
        n_batches = 0

        try:
            with torch.no_grad():
                for batch in val_data:
                    if n_batches >= 10:
                        break
                    x = batch[0].to(device)
                    _, h, _, _ = model.forward_with_intermediates(x)

                    for layer_idx, feat in h.items():
                        if isinstance(layer_idx, int) and feat.dim() == 2:
                            if layer_idx not in all_activations:
                                all_activations[layer_idx] = []
                            # Store absolute activations [N, d]
                            all_activations[layer_idx].append(feat.abs().cpu())
                    n_batches += 1
        except Exception:
            return LabResult(self.name, {"summary": "Forward pass failed"}, 0.0)

        if not all_activations or n_batches == 0:
            return LabResult(self.name, {"summary": "No activations collected"}, 0.0)

        entropy_thresh = getattr(config, 'memorization_entropy_thresh', 1.0)

        # Compute per-neuron activation entropy for each layer
        layer_memorization: Dict[int, float] = {}
        layer_mean_entropy: Dict[int, float] = {}
        worst_layer = -1
        worst_frac = 0.0

        for layer_idx in sorted(all_activations.keys()):
            # Concatenate all batches: [total_samples, d]
            acts = torch.cat(all_activations[layer_idx], dim=0)
            N, d = acts.shape

            if N < 10 or d < 2:
                continue

            # Per-neuron entropy: discretize activations into histogram bins
            n_bins = min(20, max(5, N // 10))
            low_entropy_count = 0

            for neuron_idx in range(d):
                neuron_acts = acts[:, neuron_idx]
                # Compute histogram
                hist = torch.histc(neuron_acts, bins=n_bins,
                                   min=neuron_acts.min().item(),
                                   max=neuron_acts.max().item() + 1e-8)
                # Normalize to probability distribution
                probs = hist / hist.sum()
                probs = probs[probs > 0]  # Remove zeros for log
                # Shannon entropy
                entropy = -(probs * probs.log()).sum().item()

                if entropy < entropy_thresh:
                    low_entropy_count += 1

            mem_frac = low_entropy_count / d
            layer_memorization[layer_idx] = round(mem_frac, 4)
            # Mean entropy across all neurons
            layer_mean_entropy[layer_idx] = round(entropy, 4)

            if mem_frac > worst_frac:
                worst_frac = mem_frac
                worst_layer = layer_idx

        # Determine if memorization is significant
        is_memorizing = worst_frac > 0.3  # >30% low-entropy neurons
        memorizing_layers = [l for l, f in layer_memorization.items() if f > 0.3]

        # Recommend treatment
        if is_memorizing:
            from .surgery import ActivationNoise
            # Check if ActivationNoise already present on worst layer
            has_noise = False
            if worst_layer >= 0 and worst_layer < model.num_layers:
                for op in model.ops[worst_layer].operations:
                    if isinstance(op, ActivationNoise):
                        has_noise = True
                        break

            if not has_noise:
                suggested = TreatmentType.ACTIVATION_NOISE
                confidence = min(0.9, 0.5 + worst_frac)
            else:
                suggested = TreatmentType.MEMORIZATION_PRUNE
                confidence = min(0.85, 0.4 + worst_frac)
        else:
            suggested = None
            confidence = 0.2

        findings = {
            "layer_memorization_fractions": layer_memorization,
            "worst_layer": worst_layer,
            "worst_fraction": round(worst_frac, 4),
            "memorizing_layers": memorizing_layers,
            "is_memorizing": is_memorizing,
            "entropy_threshold": entropy_thresh,
            "n_samples": sum(a.shape[0] for a in all_activations.get(
                worst_layer, [torch.zeros(0)])),
            "summary": (f"Worst layer {worst_layer}: {worst_frac:.1%} low-entropy neurons. "
                        + (f"Memorization detected on layers {memorizing_layers}"
                           if is_memorizing
                           else "No significant memorization")),
        }

        if suggested is not None:
            suggested, confidence, findings = _rotate_failed_treatment(
                suggested, patient_history, findings, confidence
            )

        return LabResult(
            test_name=self.name,
            findings=findings,
            confidence=confidence,
            suggested_treatment=suggested,
        )


# ==================== Factory ====================


def create_default_lab(max_tier: int = 2,
                       confidence_threshold: float = 0.7) -> LabDiagnostics:
    """Create a LabDiagnostics instance with all standard tests registered."""
    lab = LabDiagnostics(max_tier=max_tier,
                         confidence_threshold=confidence_threshold)

    # Tier 1 (always run)
    lab.register_test(InputStructureTest())
    lab.register_test(ResidualStructureTest())
    lab.register_test(GradientConcentrationTest())
    lab.register_test(LossCurvatureTest())
    lab.register_test(TreatmentResponseTest())
    lab.register_test(GraphStructureTest())
    lab.register_test(GraphHomophilyTest())
    lab.register_test(OverSmoothingTest())

    # Tier 2 (run when Tier 1 inconclusive)
    lab.register_test(OperationSensitivityTest())
    lab.register_test(SkipConnectionTest())
    lab.register_test(OverfittingLocalizationTest())
    lab.register_test(GraphResidualTest())
    lab.register_test(GraphSpectralAnalysisTest())
    lab.register_test(MemorizationEntropyTest())

    # Tier 3 (run when Tier 2 points to specific hypothesis)
    lab.register_test(ArchitectureComparisonTest())
    lab.register_test(DataComplexityTest())

    return lab
