import csv
import json
import time
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class CSVWriter:
    """Manages a single CSV file with header auto-creation and append-mode writing.

    Each CSV file is opened once, the header is written on creation, and rows
    are flushed immediately so data is never lost even on crash.
    """

    def __init__(self, filepath: Path, columns: List[str], append: bool = False):
        self.filepath = filepath
        self.columns = columns
        if append and filepath.exists():
            # Append mode: open for appending, skip header
            self._file = open(filepath, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=columns, extrasaction="ignore"
            )
        else:
            # Write mode: create/overwrite, write header
            self._file = open(filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file, fieldnames=columns, extrasaction="ignore"
            )
            self._writer.writeheader()
        self._file.flush()

    def write_row(self, row: Dict[str, Any]):
        """Write one row and flush immediately."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()


class SurgeryLogger:
    """Comprehensive logging system for ASANN training, surgery, and architecture evolution.

    Produces the following CSV files (all in log_dir/):

    1. training_step_metrics.csv — One row per training step
       Columns: step, timestamp, total_loss, task_loss, complexity_cost,
                lambda_complexity, learning_rate, grad_norm, num_layers,
                num_connections, total_params, architecture_cost, widths,
                architecture_stable

    2. surgery_events.csv — One row per surgery action
       Columns: step, timestamp, surgery_type, layer, detail_json

    3. architecture_evolution.csv — One row per surgery interval snapshot
       Columns: step, timestamp, num_layers, widths, ops_per_layer,
                num_connections, connection_list, total_params,
                architecture_cost, architecture_stable

    4. surgery_signals.csv — One row per surgery decision point
       Columns: step, timestamp, gds_per_layer, gds_threshold,
                nus_threshold, mean_lss, loss_plateaued, lcs_per_layer,
                clgc_pairs, num_layers, num_connections

    5. meta_learner_state.csv — One row per meta-update
       Columns: step, timestamp, lr_multiplier, learned_lr,
                surgery_interval, gds_k, nus_percentile,
                saturation_threshold, identity_threshold,
                benefit_threshold, connection_threshold,
                connection_remove_threshold, avg_recent_loss

    6. per_layer_details.csv — One row per (step, layer) at snapshot time
       Columns: step, layer_idx, in_features, out_features, operations,
                weight_norm, bias_norm, grad_norm

    7. connection_details.csv — One row per (step, connection) at snapshot time
       Columns: step, conn_idx, source, target, scale, utility,
                has_projection, projection_norm, low_utility_count

    8. evaluation_metrics.csv — One row per (step, split, metric) at snapshot time
       Columns: step, timestamp, split, metric_name, metric_value
       Long-format: regression rows have MAE/RMSE/MAPE/R²/CCC,
       classification rows have accuracy/F1/precision/recall.

    9. optimizer_state.csv — One row per training step with optimizer diagnostics
       Columns: step, timestamp, phase, phase_step, complexity_ema,
                complexity_lr_mult, num_newborn_params, effective_lr_mature,
                effective_lr_newborn, effective_lr_connection, effective_lr_operation,
                effective_lr_input_output, num_param_groups, total_tracked_params
       Crucial for debugging: tracks the new optimizer's internal state
       (WSD phase, complexity modulation, per-group LR, newborn count).

    Also maintains in-memory lists for the JSON log (backward compatibility)
    and the print_summary() method.
    """

    def __init__(self, log_dir: Optional[str] = None, append: bool = False):
        self.surgery_history: List[Dict[str, Any]] = []
        self.architecture_snapshots: List[Dict[str, Any]] = []
        self.training_metrics: List[Dict[str, Any]] = []
        self.log_dir = Path(log_dir) if log_dir else None
        self._append_mode = append

        # CSV writers — created lazily when log_dir is set
        self._csv_writers: Dict[str, CSVWriter] = {}
        self._csv_initialized = False

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._init_csv_writers()

    def _init_csv_writers(self):
        """Initialize all CSV writers with their column schemas."""
        if self._csv_initialized or self.log_dir is None:
            return

        a = self._append_mode  # shorthand

        self._csv_writers["training_steps"] = CSVWriter(
            self.log_dir / "training_step_metrics.csv",
            columns=[
                "step", "timestamp",
                "total_loss", "task_loss", "complexity_cost",
                "lambda_complexity", "learning_rate", "grad_norm",
                "num_layers", "num_connections", "total_params",
                "architecture_cost", "widths", "architecture_stable",
            ],
            append=a,
        )

        self._csv_writers["surgery_events"] = CSVWriter(
            self.log_dir / "surgery_events.csv",
            columns=[
                "step", "timestamp", "surgery_type",
                "layer", "detail_json",
            ],
            append=a,
        )

        self._csv_writers["architecture_evolution"] = CSVWriter(
            self.log_dir / "architecture_evolution.csv",
            columns=[
                "step", "timestamp",
                "num_layers", "widths", "ops_per_layer",
                "num_connections", "connection_list",
                "total_params", "architecture_cost",
                "architecture_stable",
            ],
            append=a,
        )

        self._csv_writers["surgery_signals"] = CSVWriter(
            self.log_dir / "surgery_signals.csv",
            columns=[
                "step", "timestamp",
                "gds_per_layer", "gds_threshold",
                "nus_threshold", "mean_lss",
                "loss_plateaued", "lcs_per_layer",
                "clgc_pairs",
                "num_layers", "num_connections",
            ],
            append=a,
        )

        self._csv_writers["meta_learner"] = CSVWriter(
            self.log_dir / "meta_learner_state.csv",
            columns=[
                "step", "timestamp",
                "complexity_lr_mult", "effective_lr",
                "surgery_interval",
                "gds_k", "nus_percentile",
                "saturation_threshold", "identity_threshold",
                "benefit_threshold", "connection_threshold",
                "connection_remove_threshold",
                "avg_recent_loss",
            ],
            append=a,
        )

        self._csv_writers["per_layer"] = CSVWriter(
            self.log_dir / "per_layer_details.csv",
            columns=[
                "step", "layer_idx", "mode",
                "in_features", "out_features", "channels", "spatial_shape",
                "operations", "weight_norm", "bias_norm", "grad_norm",
            ],
            append=a,
        )

        self._csv_writers["connections"] = CSVWriter(
            self.log_dir / "connection_details.csv",
            columns=[
                "step", "conn_idx",
                "source", "target",
                "scale", "utility",
                "has_projection", "projection_norm",
                "low_utility_count",
            ],
            append=a,
        )

        self._csv_writers["evaluation_metrics"] = CSVWriter(
            self.log_dir / "evaluation_metrics.csv",
            columns=[
                "step", "timestamp", "split",
                "metric_name", "metric_value",
            ],
            append=a,
        )

        self._csv_writers["optimizer_state"] = CSVWriter(
            self.log_dir / "optimizer_state.csv",
            columns=[
                "step", "timestamp",
                "phase", "phase_step",
                "complexity_ema", "complexity_lr_mult",
                "num_newborn_params",
                "effective_lr_mature", "effective_lr_newborn",
                "effective_lr_connection", "effective_lr_operation",
                "effective_lr_input_output",
                "num_param_groups", "total_tracked_params",
            ],
            append=a,
        )

        self._csv_writers["optimizer_groups"] = CSVWriter(
            self.log_dir / "optimizer_groups.csv",
            columns=[
                "step", "group_name",
                "num_params", "total_elements",
                "effective_lr", "lr_scale",
                "update_freq", "grad_clip",
                "m_fast_norm", "m_medium_norm", "m_slow_norm",
                "v_norm", "grad_norm",
            ],
            append=a,
        )

        self._csv_writers["lr_controller"] = CSVWriter(
            self.log_dir / "lr_controller_state.csv",
            columns=[
                "step", "timestamp",
                "loss", "loss_trend",
                "plateau_reductions",
                "intervals_without_improvement",
                "per_group_json",
            ],
            append=a,
        )

        self._csv_writers["diagnosis"] = CSVWriter(
            self.log_dir / "diagnosis_log.csv",
            columns=[
                "step", "epoch", "timestamp",
                "health_state", "diseases",
                "consecutive_healthy", "signals_json",
                "train_loss", "val_loss",
                "train_val_gap", "gap_trend",
            ],
            append=a,
        )

        self._csv_writers["treatments"] = CSVWriter(
            self.log_dir / "treatment_log.csv",
            columns=[
                "step", "epoch", "timestamp",
                "treatment_type", "target_disease",
                "level", "details_json",
            ],
            append=a,
        )

        self._csv_initialized = True

    # ==================== Training Step Logging ====================

    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
        model_info: Dict[str, Any],
    ):
        """Log one training step to training_step_metrics.csv.

        Called every training step by the trainer.
        """
        entry = {
            "step": step,
            "metrics": metrics,
        }
        self.training_metrics.append(entry)

        if "training_steps" in self._csv_writers:
            self._csv_writers["training_steps"].write_row({
                "step": step,
                "timestamp": time.time(),
                "total_loss": metrics.get("total_loss", ""),
                "task_loss": metrics.get("task_loss", ""),
                "complexity_cost": metrics.get("complexity_cost", ""),
                "lambda_complexity": metrics.get("lambda_complexity", ""),
                "learning_rate": metrics.get("learning_rate", ""),
                "grad_norm": metrics.get("grad_norm", ""),
                "num_layers": model_info.get("num_layers", ""),
                "num_connections": model_info.get("num_connections", ""),
                "total_params": model_info.get("total_params", ""),
                "architecture_cost": model_info.get("architecture_cost", ""),
                "widths": json.dumps(model_info.get("widths", [])),
                "architecture_stable": model_info.get("architecture_stable", False),
            })

    # ==================== Surgery Event Logging ====================

    def log_surgery(self, step: int, surgery_type: str, details: Dict[str, Any]):
        """Log a structural surgery event to surgery_events.csv.

        Called by SurgeryEngine whenever a surgery is performed.
        """
        entry = {
            "step": step,
            "type": surgery_type,
            "timestamp": time.time(),
            "details": details,
        }
        self.surgery_history.append(entry)

        if "surgery_events" in self._csv_writers:
            # Extract layer if present in details
            layer = details.get("layer", details.get("layer_idx", details.get("position", "")))
            self._csv_writers["surgery_events"].write_row({
                "step": step,
                "timestamp": time.time(),
                "surgery_type": surgery_type,
                "layer": layer,
                "detail_json": json.dumps(details, default=str),
            })

    # ==================== Architecture Snapshot Logging ====================

    def log_architecture_snapshot(self, step: int, architecture_desc: Dict[str, Any]):
        """Log a complete architecture snapshot to architecture_evolution.csv.

        Called at every surgery interval and at training end.
        """
        entry = {
            "step": step,
            "timestamp": time.time(),
            "architecture": architecture_desc,
        }
        self.architecture_snapshots.append(entry)

        if "architecture_evolution" in self._csv_writers:
            layers = architecture_desc.get("layers", [])
            widths = [l["out_features"] for l in layers]
            ops_per_layer = ["|".join(l["operations"]) for l in layers]
            connections = architecture_desc.get("connections", [])
            conn_list = [f"{c['source']}->{c['target']}(s={c['scale']:.4f})" for c in connections]

            self._csv_writers["architecture_evolution"].write_row({
                "step": step,
                "timestamp": time.time(),
                "num_layers": architecture_desc.get("num_layers", ""),
                "widths": json.dumps(widths),
                "ops_per_layer": json.dumps(ops_per_layer),
                "num_connections": len(connections),
                "connection_list": json.dumps(conn_list),
                "total_params": architecture_desc.get("total_parameters", ""),
                "architecture_cost": architecture_desc.get("architecture_cost", ""),
                "architecture_stable": architecture_desc.get("architecture_stable", False),
            })

    # ==================== Surgery Signals Logging ====================

    def log_surgery_signals(self, step: int, signals: Dict[str, Any]):
        """Log computed surgery signals to surgery_signals.csv.

        Called at every surgery decision point by the SurgeryScheduler.
        """
        entry = {
            "step": step,
            "timestamp": time.time(),
            "signals": signals,
        }
        self.training_metrics.append(entry)

        if "surgery_signals" in self._csv_writers:
            gds = signals.get("GDS", {})
            lcs = signals.get("LCS", {})
            clgc = signals.get("CLGC", {})

            # Format GDS as "layer:value" pairs
            gds_str = json.dumps({str(k): round(v, 6) for k, v in gds.items()})
            lcs_str = json.dumps({str(k): round(v, 6) for k, v in lcs.items()}) if isinstance(lcs, dict) else ""

            # Format CLGC as "(i,j):corr" pairs
            clgc_str = json.dumps(
                {f"{k[0]},{k[1]}": round(v, 4) for k, v in clgc.items()}
            ) if isinstance(clgc, dict) else ""

            self._csv_writers["surgery_signals"].write_row({
                "step": step,
                "timestamp": time.time(),
                "gds_per_layer": gds_str,
                "gds_threshold": signals.get("GDS_threshold", ""),
                "nus_threshold": signals.get("NUS_threshold", ""),
                "mean_lss": signals.get("mean_LSS", signals.get("mean_lss", "")),
                "loss_plateaued": signals.get("loss_plateaued", ""),
                "lcs_per_layer": lcs_str,
                "clgc_pairs": clgc_str,
                "num_layers": signals.get("num_layers", ""),
                "num_connections": signals.get("num_connections", ""),
            })

    # ==================== Meta-Learner State Logging ====================

    def log_meta_state(
        self,
        step: int,
        complexity_lr_mult: float,
        effective_lr: float,
        surgery_interval: int,
        thresholds: Dict[str, float],
        avg_recent_loss: float,
        optimizer_phase: str = "",
        scheduler_lr_factor: float = 1.0,
        num_newborn_params: int = 0,
        complexity_target: float = 0.0,
    ):
        """Log meta-learner state to meta_learner_state.csv.

        Called at every meta-update interval by the trainer.
        Enhanced with optimizer phase, scheduler factor, and complexity target.
        """
        if "meta_learner" in self._csv_writers:
            self._csv_writers["meta_learner"].write_row({
                "step": step,
                "timestamp": time.time(),
                "complexity_lr_mult": complexity_lr_mult,
                "effective_lr": effective_lr,
                "surgery_interval": surgery_interval,
                "gds_k": thresholds.get("gds_k", ""),
                "nus_percentile": thresholds.get("nus_percentile", ""),
                "saturation_threshold": thresholds.get("saturation_threshold", ""),
                "identity_threshold": thresholds.get("identity_threshold", ""),
                "benefit_threshold": thresholds.get("benefit_threshold", ""),
                "connection_threshold": thresholds.get("connection_threshold", ""),
                "connection_remove_threshold": thresholds.get("connection_remove_threshold", ""),
                "avg_recent_loss": avg_recent_loss,
            })

    # ==================== Per-Layer Detail Logging ====================

    def log_per_layer_details(self, step: int, model):
        """Log per-layer weight norms, grad norms, ops, and dimensions.

        Called at snapshot intervals by the trainer.
        """
        if "per_layer" not in self._csv_writers:
            return

        for l in range(model.num_layers):
            layer = model.layers[l]
            weight_norm = layer.weight.data.norm().item()
            bias_norm = layer.bias.data.norm().item() if layer.bias is not None else 0.0
            grad_norm = layer.weight.grad.data.norm().item() if layer.weight.grad is not None else 0.0
            ops_desc = "|".join(model.ops[l].describe())

            # Spatial layer info
            mode = getattr(layer, 'mode', 'flat')
            channels = getattr(layer, 'out_channels', '') if mode == 'spatial' else ''
            spatial_shape = str(getattr(layer, 'spatial_shape', '')) if mode == 'spatial' else ''

            self._csv_writers["per_layer"].write_row({
                "step": step,
                "layer_idx": l,
                "mode": mode,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "channels": channels,
                "spatial_shape": spatial_shape,
                "operations": ops_desc,
                "weight_norm": round(weight_norm, 6),
                "bias_norm": round(bias_norm, 6),
                "grad_norm": round(grad_norm, 6),
            })

    # ==================== Connection Detail Logging ====================

    def log_connection_details(self, step: int, model):
        """Log per-connection scale, utility, projection norms.

        Called at snapshot intervals by the trainer.
        """
        if "connections" not in self._csv_writers:
            return

        for idx, conn in enumerate(model.connections):
            proj_norm = 0.0
            if conn.projection is not None:
                proj_norm = conn.projection.weight.data.norm().item()

            self._csv_writers["connections"].write_row({
                "step": step,
                "conn_idx": idx,
                "source": conn.source,
                "target": conn.target,
                "scale": round(conn.scale.item(), 6),
                "utility": round(conn.utility(), 6),
                "has_projection": conn.projection is not None,
                "projection_norm": round(proj_norm, 6),
                "low_utility_count": conn.low_utility_count,
            })

    # ==================== Evaluation Metrics Logging ====================

    def log_evaluation_metrics(self, step: int, split: str,
                                metrics: Dict[str, float]):
        """Log evaluation metrics to evaluation_metrics.csv (long format).

        One row per metric, so regression and classification share the same
        CSV structure without empty columns.

        Args:
            step: Current training step.
            split: One of 'train', 'val', or 'test'.
            metrics: Dict of metric_name -> metric_value (e.g. {'mae': 0.5, 'r2': 0.8}).
        """
        if "evaluation_metrics" not in self._csv_writers:
            return
        ts = time.time()
        for name, value in sorted(metrics.items()):
            self._csv_writers["evaluation_metrics"].write_row({
                "step": step,
                "timestamp": ts,
                "split": split,
                "metric_name": name,
                "metric_value": value,
            })

    # ==================== Optimizer State Logging ====================

    def log_optimizer_state(self, step: int, optimizer_info: Dict[str, Any]):
        """Log optimizer state to optimizer_state.csv.

        Called every training step by the trainer. Records the SurgeryAwareOptimizer's
        internal state for debugging: WSD phase, complexity LR modulation,
        per-group effective LR, and newborn parameter count.

        Args:
            step: Current training step.
            optimizer_info: Dict with keys matching the CSV columns:
                phase, phase_step, complexity_ema, complexity_lr_mult,
                num_newborn_params, group_stats (dict of group_name -> {lr, ...})
        """
        if "optimizer_state" not in self._csv_writers:
            return

        group_stats = optimizer_info.get("group_stats", {})

        self._csv_writers["optimizer_state"].write_row({
            "step": step,
            "timestamp": time.time(),
            "phase": optimizer_info.get("phase", ""),
            "phase_step": optimizer_info.get("phase_step", ""),
            "complexity_ema": optimizer_info.get("complexity_ema", ""),
            "complexity_lr_mult": optimizer_info.get("complexity_lr_mult", ""),
            "num_newborn_params": optimizer_info.get("num_newborn_params", 0),
            "effective_lr_mature": group_stats.get("MATURE_2D", {}).get("lr", ""),
            "effective_lr_newborn": group_stats.get("NEWBORN", {}).get("lr", ""),
            "effective_lr_connection": group_stats.get("CONNECTION", {}).get("lr", ""),
            "effective_lr_operation": group_stats.get("OPERATION", {}).get("lr", ""),
            "effective_lr_input_output": group_stats.get("INPUT_OUTPUT", {}).get("lr", ""),
            "num_param_groups": optimizer_info.get("num_param_groups", ""),
            "total_tracked_params": optimizer_info.get("total_tracked_params", ""),
        })

    # ==================== Optimizer Group Logging (New) ====================

    def log_optimizer_groups(self, step: int, optimizer):
        """Log per-group optimizer state to optimizer_groups.csv.

        Called at snapshot intervals. One row per parameter group per snapshot.
        """
        if "optimizer_groups" not in self._csv_writers:
            return

        group_stats = optimizer.get_group_stats()
        for name, stats in group_stats.items():
            self._csv_writers["optimizer_groups"].write_row({
                "step": step,
                "group_name": name,
                "num_params": stats.get("n_params", 0),
                "total_elements": stats.get("total_elements", 0),
                "effective_lr": stats.get("effective_lr", 0),
                "lr_scale": stats.get("lr_scale", 1.0),
                "update_freq": stats.get("update_freq", 1),
                "grad_clip": stats.get("grad_clip", 1.0),
                "m_fast_norm": round(stats.get("m_fast_norm", 0), 6),
                "m_medium_norm": round(stats.get("m_medium_norm", 0), 6),
                "m_slow_norm": round(stats.get("m_slow_norm", 0), 6),
                "v_norm": round(stats.get("v_norm", 0), 6),
                "grad_norm": round(stats.get("grad_norm", 0), 6),
            })

    # ==================== LR Controller State Logging (New) ====================

    def log_lr_controller_state(self, step: int, lr_controller, loss: float):
        """Log LR controller state to lr_controller_state.csv.

        Called at snapshot intervals. One row per snapshot with all LR controller state.
        """
        if "lr_controller" not in self._csv_writers:
            return

        csv_data = lr_controller.get_csv_log_data(loss)

        # Serialize per-group data as JSON
        per_group_json = json.dumps({
            name: {
                'base_lr': round(info.get('base_lr', 0), 8),
                'log_scale': round(info.get('log_scale', 0), 6),
                'effective_lr': round(info.get('effective_lr', 0), 8),
                'hypergradient': round(info.get('hypergradient', 0), 6),
            }
            for name, info in csv_data.get('per_group', {}).items()
        })

        self._csv_writers["lr_controller"].write_row({
            "step": step,
            "timestamp": time.time(),
            "loss": csv_data.get("loss", ""),
            "loss_trend": round(csv_data.get("loss_trend", 0), 6),
            "plateau_reductions": csv_data.get("plateau_reductions", 0),
            "intervals_without_improvement": csv_data.get("intervals_without_improvement", 0),
            "per_group_json": per_group_json,
        })

    # ==================== Diagnosis Logging (v2) ====================

    def log_diagnosis(self, step: int, epoch: int, diagnosis, consecutive_healthy: int = 0):
        """Log a diagnostic evaluation to diagnosis_log.csv.

        Args:
            step: Current training step.
            epoch: Current training epoch.
            diagnosis: Diagnosis object from DiagnosisEngine.
            consecutive_healthy: Current consecutive healthy count.
        """
        if "diagnosis" not in self._csv_writers:
            return

        diseases_str = ",".join(d.disease_type.name for d in diagnosis.diseases)
        signals = diagnosis.signals

        self._csv_writers["diagnosis"].write_row({
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            "health_state": diagnosis.state.name,
            "diseases": diseases_str,
            "consecutive_healthy": consecutive_healthy,
            "signals_json": json.dumps(
                {k: round(v, 6) if isinstance(v, float) else v
                 for k, v in signals.items()},
                default=str,
            ),
            "train_loss": signals.get("train_loss", ""),
            "val_loss": signals.get("val_loss", ""),
            "train_val_gap": round(signals.get("train_val_gap", 0), 6),
            "gap_trend": round(signals.get("gap_trend", 0), 8),
        })

    def log_treatment(self, step: int, epoch: int, treatment):
        """Log a treatment application to treatment_log.csv.

        Args:
            step: Current training step.
            epoch: Current training epoch.
            treatment: Treatment object from TreatmentPlanner.
        """
        if "treatments" not in self._csv_writers:
            return

        self._csv_writers["treatments"].write_row({
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            "treatment_type": treatment.treatment_type.name,
            "target_disease": treatment.target_disease.name,
            "level": treatment.level,
            "details_json": json.dumps(treatment.details, default=str),
        })

    # ==================== Utility Methods ====================

    def get_surgery_count(self) -> Dict[str, int]:
        """Get count of each surgery type performed."""
        counts = {}
        for entry in self.surgery_history:
            stype = entry["type"]
            counts[stype] = counts.get(stype, 0) + 1
        return counts

    def get_surgery_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological list of all surgeries."""
        return sorted(self.surgery_history, key=lambda x: x["step"])

    def save(self, filename: Optional[str] = None):
        """Save JSON log (backward compatibility) and close CSV writers."""
        # Save JSON log
        if self.log_dir is not None or filename is not None:
            path = Path(filename) if filename else self.log_dir / "surgery_log.json"
            data = {
                "surgery_history": self.surgery_history,
                "architecture_snapshots": self.architecture_snapshots,
                "surgery_counts": self.get_surgery_count(),
            }

            def make_serializable(obj):
                if isinstance(obj, float):
                    if obj != obj:  # NaN
                        return None
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                return obj

            with open(path, "w") as f:
                json.dump(make_serializable(data), f, indent=2, default=str)

        # Close all CSV writers
        for writer in self._csv_writers.values():
            writer.close()
        self._csv_writers.clear()
        self._csv_initialized = False

    @staticmethod
    def load(filename: str) -> "SurgeryLogger":
        """Load logs from JSON file."""
        logger = SurgeryLogger()
        with open(filename, "r") as f:
            data = json.load(f)
        logger.surgery_history = data.get("surgery_history", [])
        logger.architecture_snapshots = data.get("architecture_snapshots", [])
        return logger

    def print_summary(self):
        """Print a human-readable summary of all surgeries."""
        counts = self.get_surgery_count()
        print("=" * 60)
        print("ASANN Surgery Summary")
        print("=" * 60)
        print(f"Total surgeries: {len(self.surgery_history)}")
        for stype, count in sorted(counts.items()):
            print(f"  {stype}: {count}")
        print("-" * 60)

        if self.architecture_snapshots:
            last = self.architecture_snapshots[-1]
            arch = last["architecture"]
            print(f"Final architecture (step {last['step']}):")
            print(f"  Layers: {arch['num_layers']}")
            for layer in arch["layers"]:
                print(
                    f"    Layer {layer['index']}: "
                    f"{layer['in_features']} -> {layer['out_features']} "
                    f"ops={layer['operations']}"
                )
            if arch["connections"]:
                print(f"  Connections:")
                for conn in arch["connections"]:
                    print(
                        f"    {conn['source']} -> {conn['target']} "
                        f"(scale={conn['scale']:.4f})"
                    )
            print(f"  Total parameters: {arch['total_parameters']}")
            print(f"  Architecture cost: {arch['architecture_cost']:.0f}")

        if self.log_dir:
            print(f"\nCSV logs saved to: {self.log_dir}/")
            for name in [
                "training_step_metrics", "surgery_events",
                "architecture_evolution", "surgery_signals",
                "meta_learner_state", "per_layer_details",
                "connection_details", "evaluation_metrics",
                "optimizer_state", "optimizer_groups",
                "lr_controller_state",
                "diagnosis_log", "treatment_log",
            ]:
                fpath = self.log_dir / f"{name}.csv"
                if fpath.exists():
                    size = fpath.stat().st_size
                    print(f"  {name}.csv ({size:,} bytes)")

        print("=" * 60)
