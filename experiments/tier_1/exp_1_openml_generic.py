"""
Generic OpenML-based Tier-1 experiment runner.

Loads any OpenML regression dataset by its dataset ID, splits with a given
seed, trains ASANN, and saves results — same artifact set as exp_1a/1c/1d/1e.

Usage (programmatic):
    from exp_1_openml_generic import run_experiment_openml
    metrics, arch, cfg = run_experiment_openml(
        results_dir="results/seed_42/abalone",
        dataset_name="abalone",
        openml_did=42726,
        seed=42,
        max_epochs_override=None,
    )

CLI:
    python exp_1_openml_generic.py --name abalone --did 42726 --seed 42 \
        --results-dir results/abalone_test --max-epochs 50
"""
import sys
import os
import argparse
import json
from pathlib import Path

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

from common import (
    setup_paths, get_device, split_and_standardize,
    create_dataloaders, evaluate_model, print_results,
    save_results, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_openml_regression(dataset_name: str, openml_did: int):
    """Load any OpenML regression dataset, handle categorical + NaN.

    Returns (X: float32 numpy, y: float32 numpy, meta_dict).
    """
    import openml
    print(f"  [openml] Fetching dataset id={openml_did} ({dataset_name})...")
    ds = openml.datasets.get_dataset(openml_did, download_data=True,
                                     download_qualities=False,
                                     download_features_meta_data=False)
    target_col = ds.default_target_attribute
    if target_col is None:
        raise RuntimeError(f"Dataset {openml_did} has no default_target_attribute; need manual mapping")

    X_df, y_series, _, _ = ds.get_data(target=target_col)
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    # Handle categorical features via one-hot (rare for tabular regression here)
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        print(f"  [openml] One-hot encoding {len(cat_cols)} categorical cols: {cat_cols}")
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dummy_na=False)

    # Convert all bool to float (openml sometimes returns bool dtype after dummies)
    bool_cols = X_df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        X_df[c] = X_df[c].astype(np.float32)

    # Impute NaN (median per column)
    if X_df.isna().any().any():
        n_nan = int(X_df.isna().sum().sum())
        print(f"  [openml] Imputing {n_nan} NaN values with column medians")
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X_df = X_df.fillna(0.0)  # any remaining (e.g. all-NaN columns)

    # Drop any column that's still non-numeric (shouldn't happen but be safe)
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  [openml] Dropping non-numeric columns: {non_numeric}")
        X_df = X_df.drop(columns=non_numeric)

    X = X_df.values.astype(np.float32)

    # Coerce target to numeric float
    if y_series.dtype.name == "category":
        y_series = y_series.astype(float)
    y = pd.to_numeric(y_series, errors="coerce").values.astype(np.float32)
    if np.isnan(y).any():
        # Drop rows with NaN target
        keep = ~np.isnan(y)
        X = X[keep]
        y = y[keep]
        print(f"  [openml] Dropped {(~keep).sum()} rows with NaN target")

    print(f"  [openml] Loaded {dataset_name}: X={X.shape}, y={y.shape}, "
          f"target_col={target_col!r}")
    return X, y, {"dataset_name": dataset_name, "openml_did": openml_did,
                  "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
                  "target_col": target_col}


def run_experiment_openml(results_dir: str,
                           dataset_name: str,
                           openml_did: int,
                           seed: int = 42,
                           max_epochs_override: int = None):
    """Generic OpenML regression experiment. Returns (metrics, arch, cfg, train_metrics)."""
    _seed_everything(seed)
    device = get_device()

    # 1. Load dataset
    X, y, meta = _load_openml_regression(dataset_name, openml_did)

    # 2. Split + standardize using THIS seed (so different seeds get different splits)
    split_data = split_and_standardize(X, y, val_ratio=0.15, test_ratio=0.15, seed=seed)

    # 3. Configure ASANN
    d_input = split_data["d_input"]
    d_output = 1
    config = ASANNConfig.from_task(
        task_type="regression", modality="tabular",
        d_input=d_input, d_output=d_output,
        n_samples=X.shape[0], device=device,
    )
    if max_epochs_override is not None:
        config.recommended_max_epochs = max_epochs_override
    config.seed = seed

    # 4. DataLoaders — use drop_last=True on train to avoid batch-of-1 BN failures
    batch_size = config.recommended_batch_size
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(split_data["X_train"], split_data["y_train"])
    val_ds = TensorDataset(split_data["X_val"], split_data["y_val"])
    test_ds = TensorDataset(split_data["X_test"], split_data["y_test_scaled"])
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }

    # 5. Trainer
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

    # 6. Train
    max_epochs = config.recommended_max_epochs
    print(f"  Training for {max_epochs} epochs (seed={seed})...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=500,
        snapshot_every=1000,
    )

    # 7. Evaluate (mirror exp_1a pattern: pass numpy arrays directly)
    metrics = evaluate_model(
        model=trainer.model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data.get("y_scaler"),
        device=device,
        y_test_scaled=split_data.get("y_test_scaled"),
    )
    # arch summary — same pattern as exp_1a/1c/1d/1e
    arch = trainer.model.describe_architecture()
    cfg_dict = config_to_dict(config)
    cfg_dict["openml_meta"] = meta
    cfg_dict["seed"] = seed

    # Save final checkpoint
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    print(f"\n  {dataset_name} (seed={seed}) results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
    print(f"    Architecture: {arch.get('num_layers')} layers, "
          f"{arch.get('total_parameters')} params")

    return metrics, arch, cfg_dict, train_metrics


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Dataset short name (e.g. abalone)")
    parser.add_argument("--did", type=int, required=True, help="OpenML dataset ID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    metrics, arch, cfg, train_metrics = run_experiment_openml(
        results_dir=args.results_dir,
        dataset_name=args.name,
        openml_did=args.did,
        seed=args.seed,
        max_epochs_override=args.max_epochs,
    )

    summary = {
        "dataset_name": args.name,
        "openml_did": args.did,
        "seed": args.seed,
        "status": "OK",
        "metrics": metrics,
        "arch": arch,
    }
    summary_path = Path(args.results_dir) / "phase_d_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()
