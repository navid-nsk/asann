"""
Experiment 4b: Airline Passengers -- Univariate Time-Series Forecasting
=========================================================================
Dataset: Classic Box-Jenkins airline passenger data (via seaborn)
Samples: 144 monthly data points (1949-1960) | Features: 1 | Task: Forecasting

This is one of the smallest possible forecasting benchmarks. The model must
learn seasonal patterns and trend from just ~100 training samples.
Uses sliding windows: past 24 months -> predict next month.

LOG-TRANSFORMED variant: works on log(passengers) so the multiplicative
seasonal pattern becomes additive. SOTA (SARIMA): RMSE ~0.09-0.10 on log
scale, MAPE ~1.6-2.1% on original scale.

Multi-seed support: set AIRLINE_SEED env var (default 42).
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
import json
from pathlib import Path

from common import (
    setup_paths, get_device, create_sliding_windows, split_temporal,
    create_forecasting_dataloaders, evaluate_forecasting_model,
    save_results, config_to_dict, run_forecasting_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.config import ASANNOptimizerConfig


# ---------------------------------------------------------------------------
# Seed from environment (for multi-seed runs)
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("AIRLINE_SEED", "42"))


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(results_dir: str):
    """Run Airline Passengers forecasting (log-transformed).
    Returns (metrics, arch, config_dict, train_metrics)."""
    _set_seed(SEED)
    device = get_device()

    # ===== 1. Load dataset =====
    print(f"  Loading Airline Passengers dataset (seed={SEED})...")
    try:
        import seaborn as sns
        flights = sns.load_dataset('flights')
        passengers_raw = flights['passengers'].values.astype(np.float32)
    except ImportError:
        # Fallback: hardcode the classic dataset
        print("  seaborn not available, using built-in data...")
        passengers_raw = np.array([
            112,118,132,129,121,135,148,148,136,119,104,118,
            115,126,141,135,125,149,170,170,158,133,114,140,
            145,150,178,163,172,178,199,199,184,162,146,166,
            171,180,193,181,183,218,230,242,209,191,172,194,
            196,196,236,235,229,243,264,272,237,211,180,201,
            204,188,235,227,234,264,302,293,259,229,203,229,
            242,233,267,269,270,315,364,347,312,274,237,278,
            284,277,317,313,318,374,413,405,355,306,271,306,
            315,301,356,348,355,422,465,467,404,347,305,336,
            340,318,362,348,363,435,491,505,404,359,310,337,
            360,342,406,396,420,472,548,559,463,407,362,405,
            417,391,419,461,472,535,622,606,508,461,390,432,
        ], dtype=np.float32)

    # Log-transform: converts multiplicative seasonality to additive
    passengers = np.log(passengers_raw)
    print(f"  Loaded: {len(passengers)} monthly data points (log-transformed)")
    print(f"  Log range: [{passengers.min():.3f}, {passengers.max():.3f}]")

    # ===== 2. Create sliding windows =====
    window_size = 24  # 2 years of history
    horizon = 1       # predict next month
    X, y = create_sliding_windows(passengers, window_size=window_size, horizon=horizon)
    y = y.ravel()     # [N, 1] -> [N]
    print(f"  Sliding windows: {X.shape[0]} samples, input_dim={X.shape[1]}, "
          f"output_dim=1")

    # ===== 3. Temporal split + normalize (standardize the log data) =====
    split_data = split_temporal(X, y, train_ratio=0.85, val_ratio=0.05, normalize=True)

    # ===== 4. Create dataloaders =====
    batch_size = 16  # Very small dataset -- small batch
    loaders = create_forecasting_dataloaders(split_data, batch_size=batch_size)

    # ===== 5. Configure ASANN =====
    # NOTE: Very small dataset (~83 train samples) -- conservative settings
    # to prevent training instability / NaN explosion.
    config = ASANNConfig(
        encoder_candidates=["temporal", "transformer"],
        d_init=8,                    # Small hidden dim for tiny dataset
        initial_num_layers=1,
        surgery_interval_init=200,    # Less frequent surgery
        warmup_steps=200,
        complexity_target=2000,       # Low complexity target
        meta_update_interval=500,

        # v2: Epoch-based diagnosis system
        diagnosis_enabled=True,
        warmup_epochs=20,             # Longer warmup -- tiny data is noisy
        surgery_epoch_interval=10,    # Much less frequent surgery
        eval_epoch_interval=5,
        meta_update_epoch_interval=30,
        stability_healthy_epochs=20,
        recovery_epochs=10,

        complexity_target_auto=True,
        complexity_ceiling_mult=2.0,  # Tighter ceiling
        auto_stop_enabled=False,      # Disable extension -- tiny dataset destabilizes
        optimizer=ASANNOptimizerConfig(
            base_lr=5e-4,             # Lower LR for small data stability
            max_grad_norm=0.5,        # Tight gradient clipping to prevent explosion
        ),
        device=device,
    )

    # ===== 6. Create model and trainer =====
    d_input = split_data["d_input"]  # 24
    d_output = 1

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=torch.nn.SmoothL1Loss(),
            log_dir=results_dir, task_type="regression",
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=torch.nn.SmoothL1Loss(),
        task_type="regression",
        y_scaler=split_data["y_scaler"],
        target_metric="mae",
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = 300  # Small dataset but conservative -- avoid over-training
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = {}
    try:
        train_metrics = trainer.train_epochs(
            train_data=loaders["train"],
            max_epochs=max_epochs,
            val_data=loaders["val"],
            test_data=loaders["test"],
            print_every=100,
            snapshot_every=200,
            checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
            checkpoint_every_epochs=100,
        )
    except (ValueError, RuntimeError) as e:
        # NaN explosion can crash during final evaluation inside train_epochs.
        # The best model is still saved in the trainer's internal state.
        print(f"  [WARNING] Training ended with error: {e}")
        print(f"  Restoring best model from trainer state...")
        trainer._restore_best_model()
    model = trainer.model

    # ===== 8. Evaluate on test set =====
    # evaluate_forecasting_model will inverse-standardize to log scale
    # (since our "original" data is log-transformed)
    metrics = evaluate_forecasting_model(
        model=model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data["y_scaler"],
        device=device,
        y_test_scaled=split_data["y_test"],
        y_train_original=split_data.get("y_train_original"),
    )

    # Rename the default metrics to clarify they are on log scale
    metrics["rmse_log"] = metrics["rmse"]
    metrics["mse_log"] = metrics["mse"]
    metrics["mae_log"] = metrics["mae"]

    # Compute MAPE on ORIGINAL (non-log) scale: exp(predictions) vs exp(targets)
    model.eval()
    with torch.no_grad():
        X_test_dev = split_data["X_test"].to(device)
        preds_scaled = model(X_test_dev).cpu().numpy()

    y_scaler = split_data["y_scaler"]
    if y_scaler is not None:
        preds_log = y_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1) if preds_scaled.ndim == 1 else preds_scaled
        ).ravel()
    else:
        preds_log = preds_scaled.ravel()

    y_test_log = split_data["y_test_original"].numpy().ravel()

    # Back to original scale
    preds_orig = np.exp(preds_log)
    y_test_orig = np.exp(y_test_log)

    # MAPE on original scale
    mape_orig = float(np.mean(np.abs((y_test_orig - preds_orig) / y_test_orig)) * 100)
    metrics["mape_original"] = mape_orig
    # RMSE on original scale (for reference)
    metrics["rmse_original"] = float(np.sqrt(np.mean((y_test_orig - preds_orig) ** 2)))

    print(f"\n  === Log-scale metrics ===")
    print(f"    RMSE (log): {metrics['rmse_log']:.4f}")
    print(f"    MAE  (log): {metrics['mae_log']:.4f}")
    print(f"  === Original-scale metrics ===")
    print(f"    MAPE (%):   {metrics['mape_original']:.2f}%")
    print(f"    RMSE:       {metrics['rmse_original']:.2f}")

    # ===== 9. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Save seed-specific results
    result_file = os.path.join(results_dir, f"results_s{SEED}.json")
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved to {result_file}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_4"

    # Results directory includes seed for multi-seed support
    exp_name = f"airline_passengers_s{SEED}"
    name, metrics, arch, elapsed, status = run_forecasting_experiment_wrapper(
        exp_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
