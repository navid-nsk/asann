"""
Experiment 1d: Bike Sharing
=============================
Dataset: OpenML 'Bike_Sharing_Demand' v2
Samples: 17,379 | Features: 12 | Target: count (total rentals)

SOTA comparison: LightGBM (tuned) RMSE ~ 30-35 counts/hr.
RMSE reported on original scale (counts/hr).
"""

import sys
import os
# Add experiments/ dir (for common.py) and project root (for asann)
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

from common import (
    setup_paths, get_device, preprocess_dataframe,
    split_and_standardize, create_dataloaders, evaluate_model,
    print_results, save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Bike Sharing experiment. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading Bike Sharing dataset from OpenML...")
    bike = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
    df = bike.frame
    print(f"  Raw data: {df.shape[0]} samples, {df.shape[1]} columns")

    # ===== 2. Preprocess =====
    # Target is 'count' (total rentals). Other leakage columns
    # (casual, registered) are not present in this version.
    X, y, feature_names = preprocess_dataframe(
        df=df,
        target_col="count",
        drop_cols=["casual", "registered", "instant", "dteday"],
        max_missing_frac=0.5,
    )

    # ===== 3. Split and standardize =====
    split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=42)

    # ===== 4. Create dataloaders =====
    batch_size = 256
    loaders = create_dataloaders(split_data, batch_size=batch_size)

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        d_init=48,
        initial_num_layers=2,
        complexity_target=50000,    # 12 features, 17K samples -- moderate

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,
        device=device,

        # Tuning for this dataset
        overfitting_gap_early=0.30,
        overfitting_gap_moderate=0.50,
        stalled_convergence_patience=50,
        post_stable_patience_epochs=80,
    )

    # ===== 6. Create model and trainer =====
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
        target_metric="r2",
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = 400
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

    # ===== 8. Evaluate on test set =====
    # RMSE on original scale (counts/hr) is the SOTA metric
    metrics = evaluate_model(
        model=model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data["y_scaler"],
        device=device,
        y_test_scaled=split_data["y_test_scaled"],
    )

    # ===== 9. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_1"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "Bike Sharing", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
