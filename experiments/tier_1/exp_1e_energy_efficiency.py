"""
Experiment 1e: Energy Efficiency
==================================
Dataset: OpenML 'energy-efficiency' (768 samples, 8 features)
Target: y1 (Heating Load)

Note: Dataset has two targets (y1=heating, y2=cooling). We use y1 only
for single-target regression consistency across all experiments.

SOTA comparison: TabPFN v2 RMSE ~ 0.30 kWh (original scale).
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
from io import BytesIO
from urllib.request import urlopen

from common import (
    setup_paths, get_device, preprocess_dataframe,
    split_and_standardize, create_dataloaders, evaluate_model,
    print_results, save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Energy Efficiency experiment. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    # Use original UCI continuous data (OpenML v1 is discretized to 37 integers).
    # The SOTA RMSE ~0.30 is on the continuous version.
    print("  Loading Energy Efficiency dataset (UCI continuous)...")
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        response = urlopen(uci_url, timeout=30)
        df = pd.read_excel(BytesIO(response.read()))
    except Exception:
        # Fallback: OpenML version (discretized)
        print("  [WARN] UCI download failed, falling back to OpenML (discretized)")
        energy = fetch_openml(name="energy-efficiency", version=1, as_frame=True, parser="auto")
        df = energy.frame
        df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']
    print(f"  Raw data: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  Y1 (Heating Load) range: [{df['Y1'].min():.2f}, {df['Y1'].max():.2f}] kWh/m2")
    print(f"  Y1 unique values: {df['Y1'].nunique()}")

    # ===== 2. Preprocess =====
    # Use Y1 (Heating Load) as target, drop Y2 (Cooling Load)
    X, y, feature_names = preprocess_dataframe(
        df=df,
        target_col="Y1",
        drop_cols=["Y2"],
        max_missing_frac=0.5,
    )

    # ===== 3. Split and standardize =====
    split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=42)

    # ===== 4. Configure ASANN =====
    d_input = split_data["d_input"]
    d_output = 1
    config = ASANNConfig.from_task(
        task_type="regression",
        modality="tabular",
        d_input=d_input,
        d_output=d_output,
        n_samples=X.shape[0],
        device=device,
    )

    # ===== 5. Create dataloaders =====
    batch_size = config.recommended_batch_size
    loaders = create_dataloaders(split_data, batch_size=batch_size)

    # ===== 6. Create model and trainer =====
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
    max_epochs = config.recommended_max_epochs
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=250,
        snapshot_every=500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # Re-bind: train_epochs restores best model (new object)

    # ===== 8. Evaluate on test set =====
    # RMSE on original scale (kWh) is the SOTA metric
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
        "Energy Efficiency", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
