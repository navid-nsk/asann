"""
Experiment 5f: Gray-Scott 2D (PINNacle Benchmark)
===================================================
Dataset: PINNacle grayscott.dat
Samples: ~100k (subsampled to 50k) | Input: (x, y, t) → 3 features | Output: (u, v) → 2 targets
PDE: Coupled reaction-diffusion system:
     u_t = D_u·∇²u - u·v² + F·(1-u)
     v_t = D_v·∇²v + u·v² - (F+k)·v

This is the most complex PDE: two coupled species with Turing-type pattern formation.
ASANN should discover: derivative (Laplacian) + polynomial (u·v²) + branched ops.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from common import (
    setup_paths, get_device,
    save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_regression_metrics,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from tier_5.pde_utils import load_pinnacle_data, compute_pde_experiment_metrics


def _split_multioutput(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split and standardize multi-output regression data.

    Unlike split_and_standardize(), this handles y with shape (N, d_out)
    where d_out > 1. Scales each output column independently.
    """
    test_size = test_ratio
    val_size = val_ratio / (1.0 - test_ratio)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Standardize X
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    # Standardize y (multi-output: fit on 2D array directly)
    y_scaler = StandardScaler()
    y_test_original = y_test.copy()
    y_train = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val = y_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test_scaled": torch.tensor(y_test_scaled, dtype=torch.float32),
        "y_test_original": torch.tensor(y_test_original, dtype=torch.float32),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "d_input": X_train.shape[1],
        "d_output": y_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


def run_experiment(results_dir: str):
    """Run Gray-Scott 2D experiment. Returns (metrics, arch, config_dict, train_metrics)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading PINNacle Gray-Scott dataset...")
    X, y = load_pinnacle_data("grayscott", max_samples=50000)
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} input features, {y.shape[1]} outputs")

    # ===== 2. Split and standardize (multi-output aware) =====
    split_data = _split_multioutput(X, y, val_ratio=0.15, test_ratio=0.15, seed=42)

    # ===== 3. Create dataloaders =====
    batch_size = 256
    train_ds = TensorDataset(split_data["X_train"], split_data["y_train"])
    val_ds = TensorDataset(split_data["X_val"], split_data["y_val"])
    test_ds = TensorDataset(split_data["X_test"], split_data["y_test_scaled"])
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }

    # ===== 4. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["linear", "fourier"],
        d_init=64,
        initial_num_layers=4,
        surgery_interval_init=400,
        warmup_steps=1000,
        complexity_target=250000,
        meta_update_interval=1000,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=12,
        recovery_epochs=5,

        # PDE-specific: enable physics ops for derivative/polynomial discovery
        physics_ops_enabled=True,
        polynomial_max_degree=3,
        branch_count=2,

        # More capacity for coupled system
        max_neuron_surgeries_per_interval=16,
        max_operation_surgeries_per_interval=3,

        device=device,
    )

    # ===== 5. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = split_data["d_output"]  # 2 for (u, v)

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
        print_every=500,
        snapshot_every=1000,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # Re-bind: train_epochs restores best model (new object)

    # ===== 7. Evaluate on test set (multi-output aware) =====
    model.eval()
    with torch.no_grad():
        X_test_dev = split_data["X_test"].to(device)
        preds_scaled = model(X_test_dev).cpu().numpy()

    y_scaler = split_data["y_scaler"]
    preds_original = y_scaler.inverse_transform(preds_scaled)
    y_true = split_data["y_test_original"].numpy()

    # Per-output and combined metrics
    metrics = compute_regression_metrics(y_true, preds_original)

    # Also report per-output R² and Rel. L2 error
    for col_idx, col_name in enumerate(["u", "v"]):
        r2_col = r2_score(y_true[:, col_idx], preds_original[:, col_idx])
        metrics[f"r2_{col_name}"] = float(r2_col)
        # Per-output Relative L2 error
        diff_norm = np.linalg.norm(preds_original[:, col_idx] - y_true[:, col_idx])
        true_norm = np.linalg.norm(y_true[:, col_idx])
        rel_l2_col = float(diff_norm / true_norm) if true_norm > 1e-12 else float("nan")
        metrics[f"rel_l2_{col_name}"] = rel_l2_col
    print(f"\n  Test: R2_u={metrics['r2_u']:.4f}, R2_v={metrics['r2_v']:.4f}, R2_overall={metrics['r2']:.4f}")
    print(f"  Rel. L2: overall={metrics['rel_l2']:.6f}, u={metrics['rel_l2_u']:.6f}, v={metrics['rel_l2_v']:.6f}")

    # ===== 8. PDE-specific metrics =====
    pde_metrics = compute_pde_experiment_metrics(model, trainer, trainer.scheduler)
    metrics["pde"] = pde_metrics
    print(f"  PDE Discovery: {pde_metrics['derivative_ops_discovered']} derivative ops, "
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
        "Gray-Scott 2D", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
