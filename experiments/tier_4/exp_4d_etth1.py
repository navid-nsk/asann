"""
Experiment 4d: ETTh1 -- Multivariate Time-Series Forecasting (96-step horizon)
================================================================================
Dataset: Electricity Transformer Temperature (hourly)
Samples: 17,420 | Features: 7 | Task: Forecast all 7 features 96 steps ahead

Matches PatchTST benchmark protocol EXACTLY:
  - Input (lookback): 336 hours (14 days) of all 7 features
  - Output (horizon):  96 hours ( 4 days) of all 7 features
  - Split: Fixed boundaries (PatchTST standard):
      Train: data[0:8640]       = 12 months
      Val:   data[8304:11520]   = ~4.5 months (overlaps train by seq_len=336)
      Test:  data[11184:14400]  = ~4.5 months (overlaps val by seq_len=336)
  - Normalization: StandardScaler per-feature, fit on train[0:8640] ONLY
  - Metrics: MSE and MAE on STANDARDIZED (z-scored) scale

SPATIAL MODE: Uses ASANN's convolutional backbone instead of flat MLP.
  - spatial_shape = (7, 336, 1): 7 features as channels, 336 time steps as height
  - Conv2d layers operate along the time axis (effectively Conv1d)
  - GAP -> Linear head for regression output
  - Surgery can grow channels, add operations, etc.

SOTA (PatchTST): MSE ~0.370, MAE ~0.390 at 96-step horizon.
  PatchTST uses: patching (16/8), RevIN, channel-independence, 3-layer Transformer.

Multi-seed support: set ETTH1_SEED env var (default 42).
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
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from common import (
    setup_paths, get_device, save_results, config_to_dict,
    run_forecasting_experiment_wrapper, resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.config import ASANNOptimizerConfig


# ---------------------------------------------------------------------------
# Seed from environment (for multi-seed runs)
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("ETTH1_SEED", "42"))


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_etth1(data_dir: str) -> str:
    """Download ETTh1.csv if not present. Returns path to CSV."""
    data_path = Path(data_dir)
    csv_path = data_path / "ETTh1.csv"

    if csv_path.exists():
        return str(csv_path)

    data_path.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    print(f"  Downloading ETTh1 from {url} ...")
    urllib.request.urlretrieve(url, str(csv_path))
    return str(csv_path)


def run_experiment(results_dir: str):
    """Run ETTh1 forecasting using ASANN spatial mode (Conv backbone).
    Returns (metrics, arch, config_dict, train_metrics)."""
    _set_seed(SEED)
    device = get_device()

    # ===== 1. Load dataset =====
    print(f"  Loading ETTh1 dataset (seed={SEED})...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ETTh1")
    csv_path = _download_etth1(data_dir)
    df = pd.read_csv(csv_path)

    # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    raw_data = df[feature_cols].values.astype(np.float32)
    print(f"  Loaded: {raw_data.shape[0]} timesteps, {raw_data.shape[1]} features")

    # ===== 2. PatchTST-standard fixed split boundaries =====
    # Exactly matching PatchTST data_loader.py lines 48-51
    seq_len = 336     # Lookback window (PatchTST standard)
    pred_len = 96     # Forecast horizon
    n_features = len(feature_cols)  # 7

    # Fixed boundaries for ETTh1 (standard in ALL long-horizon forecasting papers)
    train_border = (0, 12 * 30 * 24)                                        # 0:8640
    val_border   = (12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24)    # 8304:11520
    test_border  = (12 * 30 * 24 + 4 * 30 * 24 - seq_len,
                    12 * 30 * 24 + 8 * 30 * 24)                              # 11184:14400

    print(f"  PatchTST-standard split boundaries:")
    print(f"    Train: [{train_border[0]}:{train_border[1]}] = {train_border[1]-train_border[0]} hours")
    print(f"    Val:   [{val_border[0]}:{val_border[1]}] = {val_border[1]-val_border[0]} hours")
    print(f"    Test:  [{test_border[0]}:{test_border[1]}] = {test_border[1]-test_border[0]} hours")

    # ===== 3. Normalize: StandardScaler fit on TRAIN ONLY (per-feature) =====
    scaler = StandardScaler()
    train_raw = raw_data[train_border[0]:train_border[1]]  # [0:8640]
    scaler.fit(train_raw)
    data_scaled = scaler.transform(raw_data)  # Scale ALL data using train stats

    print(f"  StandardScaler: fit on train[{train_border[0]}:{train_border[1]}], "
          f"applied to all {len(data_scaled)} timesteps")

    # ===== 4. Create sliding window datasets =====
    # For spatial mode: input is STILL flat [B, 336*7] — the model reshapes
    # to [B, 7, 336, 1] internally via config.spatial_shape
    def make_windows(data, border1, border2, seq_len, pred_len, n_feat):
        """Create sliding windows matching PatchTST __getitem__.
        Input: [seq_len * n_feat] flat, Target: [pred_len * n_feat] flat.

        Input layout: features interleaved per timestep, then flattened.
        For spatial_shape=(7, 336, 1), the model does:
          x.view(B, 7, 336, 1)
        So we need to arrange data as [feat0_t0..t335, feat1_t0..t335, ..., feat6_t0..t335]
        i.e., channel-first order: all timesteps for feature 0, then feature 1, etc.
        """
        chunk = data[border1:border2]
        n_samples = len(chunk) - seq_len - pred_len + 1
        X = np.empty((n_samples, n_feat * seq_len), dtype=np.float32)
        y = np.empty((n_samples, pred_len * n_feat), dtype=np.float32)
        for i in range(n_samples):
            # Input window: [seq_len, n_feat] -> transpose to [n_feat, seq_len] -> flatten
            # This matches spatial_shape = (7, 336, 1) layout: C=7, H=336, W=1
            window = chunk[i:i + seq_len]  # [336, 7]
            X[i] = window.T.flatten()      # [7, 336] -> [7*336] = channel-first

            # Target: flat [pred_len * n_feat]
            y[i] = chunk[i + seq_len:i + seq_len + pred_len].flatten()
        return X, y

    X_train, y_train = make_windows(data_scaled, *train_border, seq_len, pred_len, n_features)
    X_val, y_val     = make_windows(data_scaled, *val_border, seq_len, pred_len, n_features)
    X_test, y_test   = make_windows(data_scaled, *test_border, seq_len, pred_len, n_features)

    print(f"  Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Input dim: {X_train.shape[1]} (={n_features}*{seq_len}), "
          f"Output dim: {y_train.shape[1]} (={pred_len}*{n_features})")
    print(f"  Spatial mode: input [B, {X_train.shape[1]}] -> reshape [B, {n_features}, {seq_len}, 1]")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # ===== 5. Create dataloaders =====
    batch_size = 128
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds   = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_ds  = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ===== 6. Configure ASANN in SPATIAL mode =====
    d_input = n_features * seq_len    # 2352 (7 * 336) — flat, model reshapes to (7, 336, 1)
    d_output = pred_len * n_features  # 672  (96 * 7)

    config = ASANNConfig(
        # Spatial mode: 7 features as channels, 336 time steps as height, width=1
        # Conv2d operates along the time axis (effectively Conv1d)
        spatial_shape=(n_features, seq_len, 1),  # (7, 336, 1)
        c_stem_init=32,                   # Start with 32 channels (surgery can grow)
        initial_num_layers=4,             # 4 conv layers at full temporal resolution
        spatial_downsample_stages=0,      # No stride-2 downsampling (W=1 breaks tracking)
        max_downsample_stages=0,          # Prevent surgery from adding stride-2
        max_channels=512,                 # Allow significant channel growth
        min_spatial_channels=16,

        # Augmentation: disable image-specific augments (W=1 breaks reflect padding)
        dataset_augmented=True,           # Use cutout-only (no flip/crop/jitter)
        cutout_size=1,                    # cutout_half=0 => no-op cutout for W=1
        mixup_enabled=False,              # Not applicable for regression
        drop_path_rate=0.0,               # Not needed for time series regression
        elastic_enabled=False,            # Image-only augmentation

        # Surgery and diagnosis
        surgery_interval_init=300,
        warmup_steps=600,
        complexity_target=2000000,        # Higher for conv backbone
        meta_update_interval=800,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=12,
        recovery_epochs=4,

        complexity_target_auto=True,
        complexity_ceiling_mult=3.0,
        hard_max_multiplier=1.5,

        # AMP for faster training with conv layers
        amp_enabled=True,

        optimizer=ASANNOptimizerConfig(
            base_lr=1e-3,
            max_grad_norm=1.0,
        ),
        device=device,
    )

    # Use MSE loss (standard for time-series benchmarks)
    loss_fn = torch.nn.MSELoss()

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir, task_type="regression",
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="regression",
        y_scaler=None,  # Already standardized, no inverse needed for training
        target_metric="mae",
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = 200
    print(f"\n  Training for {max_epochs} epochs (spatial mode)...")
    train_metrics = {}
    try:
        train_metrics = trainer.train_epochs(
            train_data=train_loader,
            max_epochs=max_epochs,
            val_data=val_loader,
            test_data=test_loader,
            print_every=300,
            snapshot_every=600,
            checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
            checkpoint_every_epochs=50,
        )
    except (ValueError, RuntimeError) as e:
        print(f"  [WARNING] Training ended with error: {e}")
        print(f"  Restoring best model from trainer state...")
        trainer._restore_best_model()
    model = trainer.model

    # ===== 8. Evaluate on test set =====
    # PatchTST computes MSE/MAE on standardized (z-scored) data - NO inverse transform
    model.eval()
    with torch.no_grad():
        preds_list = []
        bs = 256
        for i in range(0, len(X_test_t), bs):
            batch = X_test_t[i:i + bs].to(device)
            preds_list.append(model(batch).cpu())
        preds_scaled = torch.cat(preds_list, dim=0).numpy()

    y_test_np = y_test  # Already on standardized scale

    # THE benchmark metrics: MSE and MAE on standardized scale
    mse_scaled = float(np.mean((preds_scaled - y_test_np) ** 2))
    mae_scaled = float(np.mean(np.abs(preds_scaled - y_test_np)))

    # Also compute original-scale metrics for reference
    preds_2d = preds_scaled.reshape(-1, n_features)
    y_test_2d = y_test_np.reshape(-1, n_features)
    preds_orig = scaler.inverse_transform(preds_2d)
    y_test_orig = scaler.inverse_transform(y_test_2d)
    mse_orig = float(np.mean((preds_orig - y_test_orig) ** 2))
    mae_orig = float(np.mean(np.abs(preds_orig - y_test_orig)))

    metrics = {
        "mse_scaled": mse_scaled,
        "mae_scaled": mae_scaled,
        "mse": mse_orig,
        "mae": mae_orig,
        "rmse_scaled": float(np.sqrt(mse_scaled)),
        "rmse": float(np.sqrt(mse_orig)),
        "horizon": pred_len,
        "input_len": seq_len,
        "n_features": n_features,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "split": "patchtst_standard",
    }

    print(f"\n  === ETTh1 96-step Forecast Results (Spatial Mode, PatchTST protocol) ===")
    print(f"    Input: [{n_features}, {seq_len}, 1] spatial -> Output: {pred_len}*{n_features}={d_output}")
    print(f"    MSE (scaled):  {mse_scaled:.4f}  (SOTA PatchTST: ~0.370)")
    print(f"    MAE (scaled):  {mae_scaled:.4f}  (SOTA PatchTST: ~0.390)")
    print(f"    MSE (orig):    {mse_orig:.4f}")
    print(f"    MAE (orig):    {mae_orig:.4f}")

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
    exp_name = f"etth1_forecasting_s{SEED}"
    name, metrics, arch, elapsed, status = run_forecasting_experiment_wrapper(
        exp_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
