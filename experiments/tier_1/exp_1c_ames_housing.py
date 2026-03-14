"""
Experiment 1c: Ames Housing
=============================
Dataset: OpenML dataset 42165 (Ames Housing)
Samples: 1,460 | Features: 79+ (mixed numeric/categorical) | Target: SalePrice

SOTA comparison: CatBoost (tuned) RMSE ~ 0.11 on log(SalePrice) scale.
We train on log-transformed targets to match.
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
from sklearn.metrics import mean_squared_error, r2_score

from common import (
    setup_paths, get_device, preprocess_dataframe,
    split_and_standardize, create_dataloaders, evaluate_model,
    print_results, save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Ames Housing experiment. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading Ames Housing dataset from OpenML...")
    housing = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    df = housing.frame
    print(f"  Raw data: {df.shape[0]} samples, {df.shape[1]} columns")

    # ===== 2. Preprocess =====
    # Drop ID column and columns with >50% missing
    X, y, feature_names = preprocess_dataframe(
        df=df,
        target_col="SalePrice",
        drop_cols=["Id"],
        max_missing_frac=0.5,
    )

    # ===== 3. Log-transform target =====
    # SOTA reports RMSE on log(SalePrice) scale.
    # Store raw prices for original-scale reporting.
    y_raw = y.copy()
    y = np.log(y)
    print(f"  Target log-transformed: log(SalePrice) range [{y.min():.2f}, {y.max():.2f}]")

    # ===== 4. Split and standardize (on log-scale targets) =====
    split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=42)

    # ===== 5. Create dataloaders =====
    batch_size = 64
    loaders = create_dataloaders(split_data, batch_size=batch_size)

    # ===== 6. Configure ASANN =====
    config = ASANNConfig(
        d_init=48,                  # reduced from 64 -- 1,460 samples is small
        initial_num_layers=2,
        complexity_target=25000,    # reduced from 50000 -- prevent memorization

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,
        device=device,

        # Tuning for this dataset (small, tabular)
        overfitting_gap_early=0.30,
        overfitting_gap_moderate=0.50,
        stalled_convergence_patience=50,
        post_stable_patience_epochs=80,
    )

    # ===== 7. Create model and trainer =====
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

    # ===== 8. Train =====
    max_epochs = 400
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=300,
        snapshot_every=600,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # Re-bind: train_epochs restores best model (new object)

    # ===== 9. Evaluate on test set =====
    # evaluate_model inverse-transforms preds back to log-scale (the "original"
    # scale as seen by split_and_standardize). RMSE on log-scale = SOTA metric.
    metrics = evaluate_model(
        model=model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data["y_scaler"],
        device=device,
        y_test_scaled=split_data["y_test_scaled"],
    )

    # Rename for clarity: the "rmse" from evaluate_model is on log scale
    metrics["rmse_log"] = metrics["rmse"]
    metrics["r2_log"] = metrics["r2"]

    # Also compute original-scale (dollar) metrics for interpretability
    model.eval()
    with torch.no_grad():
        X_t = split_data["X_test"].to(device)
        preds_scaled = model(X_t).cpu().numpy()
    preds_log = split_data["y_scaler"].inverse_transform(preds_scaled)
    y_true_log = split_data["y_test_original"].numpy()

    preds_dollars = np.exp(preds_log)
    y_true_dollars = np.exp(y_true_log)

    metrics["rmse_dollars"] = float(np.sqrt(mean_squared_error(y_true_dollars, preds_dollars)))
    metrics["r2_dollars"] = float(r2_score(y_true_dollars, preds_dollars))

    # ===== 10. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_1"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "Ames Housing", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
