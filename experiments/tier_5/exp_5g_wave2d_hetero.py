"""
Experiment 5g: Wave 2D Heterogeneous (PINNacle Benchmark)
==========================================================
Dataset: PINNacle wave_darcy.dat (COMSOL time-encoded)
Samples: ~250k (subsampled to 50k) | Input: (x, y, t) → 3 features | Output: u → 1 target
PDE: u_tt = ∇·(c(x,y)²·∇u) — Wave equation with spatially varying coefficients.

Variable coefficient wave propagation (Darcy-type heterogeneous medium).
ASANN should discover: derivative ops + potentially branched ops for the
heterogeneous structure.
"""

import sys
import os
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np
from pathlib import Path

from common import (
    setup_paths, get_device, split_and_standardize,
    create_dataloaders, evaluate_model, print_results,
    save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from tier_5.pde_utils import load_pinnacle_data, compute_pde_experiment_metrics


def run_experiment(results_dir: str):
    """Run Wave 2D Heterogeneous experiment. Returns (metrics, arch, config_dict, train_metrics)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading PINNacle Wave 2D Heterogeneous dataset...")
    X, y = load_pinnacle_data("wave_darcy", max_samples=50000)
    y = y.ravel()  # Single-output: (N,1) → (N,) for split_and_standardize
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} input features")

    # ===== 2. Split and standardize =====
    split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=42)

    # ===== 3. Configure ASANN =====
    d_input = split_data["d_input"]
    d_output = 1
    config = ASANNConfig.from_task(
        task_type="regression",
        modality="pde",
        d_input=d_input,
        d_output=d_output,
        n_samples=split_data["n_train"],
        device=device,
    )

    # ===== 4. Create dataloaders =====
    batch_size = config.recommended_batch_size
    loaders = create_dataloaders(split_data, batch_size=batch_size)

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=torch.nn.MSELoss(),
            log_dir=results_dir, task_type="regression",
            y_scaler=split_data.get("y_scaler"),
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=torch.nn.MSELoss(),
        task_type="regression",
        y_scaler=split_data.get("y_scaler"),
        target_metric="rmse",
    )
    model = trainer.model

    # ===== 6. Train =====
    max_epochs = config.recommended_max_epochs
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=500,
        snapshot_every=1000,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # Re-bind: train_epochs restores best model (new object)

    # ===== 7. Evaluate on test set =====
    metrics = evaluate_model(
        model=model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data["y_scaler"],
        device=device,
    )

    # ===== 8. PDE-specific metrics =====
    pde_metrics = compute_pde_experiment_metrics(model, trainer, trainer.scheduler)
    metrics["pde"] = pde_metrics
    print(f"\n  PDE Discovery: {pde_metrics['derivative_ops_discovered']} derivative ops, "
          f"{pde_metrics['polynomial_ops_discovered']} polynomial ops, "
          f"{pde_metrics['branched_ops_discovered']} branched ops")

    # ===== 9. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_5"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "Wave 2D Heterogeneous", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
