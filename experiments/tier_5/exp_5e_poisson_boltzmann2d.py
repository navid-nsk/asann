"""
Experiment 5e: Poisson-Boltzmann 2D (PINNacle Benchmark)
==========================================================
Dataset: PINNacle poisson_boltzmann2d.dat
Samples: 3,236 | Input: (x, y) → 2 features | Output: u → 1 target
PDE: ∇²u = f(u) — Reaction-dominated nonlinear Poisson equation.

The nonlinear source term f(u) makes this a reaction-diffusion problem.
ASANN should discover: derivative (Laplacian) + polynomial ops for the
nonlinear reaction term.
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
    """Run Poisson-Boltzmann 2D experiment. Returns (metrics, arch, config_dict, train_metrics)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading PINNacle Poisson-Boltzmann 2D dataset...")
    X, y = load_pinnacle_data("poisson_boltzmann2d")
    y = y.ravel()  # Single-output: (N,1) → (N,) for split_and_standardize
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} input features")

    # ===== 2. Split and standardize =====
    split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=42)

    # ===== 3. Create dataloaders =====
    batch_size = 128
    loaders = create_dataloaders(split_data, batch_size=batch_size)

    # ===== 4. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["linear", "fourier"],
        d_init=48,
        initial_num_layers=3,
        surgery_interval_init=300,
        warmup_steps=500,
        complexity_target=100000,
        meta_update_interval=500,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,

        # PDE-specific: enable physics ops for derivative/polynomial discovery
        physics_ops_enabled=True,
        polynomial_max_degree=3,
        branch_count=2,

        device=device,
    )

    # ===== 5. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = 1

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
    max_epochs = 500
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=300,
        snapshot_every=500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=100,
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
        "Poisson-Boltzmann 2D", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
