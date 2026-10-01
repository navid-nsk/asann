"""
Generic PINNacle PDE experiment runner — parameterized by dataset name.

Mirrors the pattern of exp_5b/5d/5e/5f/5g but takes (name, seed,
max_epochs) as inputs so a single script can run any registered PINNacle
PDE under any seed.

Usage (programmatic):
    from exp_5_pinnacle_generic import run_experiment_pinnacle
    run_experiment_pinnacle(
        results_dir="phase_f_results/grayscott/seed_42",
        pde_name="grayscott",
        seed=42,
        max_epochs_override=1500,
    )
"""
import sys
import os
import argparse
import json
from pathlib import Path

_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np

from common import (
    setup_paths, get_device, split_and_standardize,
    create_dataloaders, evaluate_model,
    config_to_dict, resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from tier_5.pde_utils import load_pinnacle_data, _PINNACLE_DATASETS


# Per-PDE default max-samples cap (large datasets get subsampled to keep
# training tractable). None = no cap.
_PDE_MAX_SAMPLES = {
    "burgers1d": None,
    "kuramoto_sivashinsky": None,
    "poisson_boltzmann2d": None,
    "grayscott": None,
    "wave_darcy": 50000,           # COMSOL time-encoded, can be large
    "heat_darcy": None,
    "heat_complex": 50000,
    "heat_longtime": None,
    "heat_multiscale": 50000,
    "poisson_classic": None,
}


def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_multioutput(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split + standardize for multi-output regression (e.g. Gray-Scott u,v).

    Mirrors split_and_standardize() but handles 2D y. Adapted from
    exp_5f_grayscott._split_multioutput.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    test_size = test_ratio
    val_size = val_ratio / (1.0 - test_ratio)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )
    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    X_test = x_scaler.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler()
    y_test_original = y_test.copy().astype(np.float32)
    y_train = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val = y_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val":   torch.tensor(X_val, dtype=torch.float32),
        "y_val":   torch.tensor(y_val, dtype=torch.float32),
        "X_test":  torch.tensor(X_test, dtype=torch.float32),
        "y_test_scaled":   torch.tensor(y_test_scaled, dtype=torch.float32),
        "y_test_original": torch.tensor(y_test_original, dtype=torch.float32),
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "d_input": X_train.shape[1],
        "d_output": y_train.shape[1],
    }


def run_experiment_pinnacle(results_dir: str,
                             pde_name: str,
                             seed: int = 42,
                             max_epochs_override: int = None,
                             max_samples_override: int = None):
    """Run any registered PINNacle PDE experiment.
    Returns (metrics, arch, cfg, train_metrics)."""
    if pde_name not in _PINNACLE_DATASETS:
        raise ValueError(f"Unknown PINNacle PDE: {pde_name}. "
                         f"Available: {list(_PINNACLE_DATASETS.keys())}")

    _seed_everything(seed)
    device = get_device()

    # 1. Load
    print(f"  Loading PINNacle {pde_name} dataset...")
    max_samples = (max_samples_override if max_samples_override is not None
                   else _PDE_MAX_SAMPLES.get(pde_name))
    X, y = load_pinnacle_data(pde_name, max_samples=max_samples, seed=seed)
    # Multi-output PDEs (e.g. grayscott has 2 outputs u,v)
    if y.ndim == 2 and y.shape[1] > 1:
        d_output_native = y.shape[1]
        is_multi_output = True
    else:
        d_output_native = 1
        y = y.ravel()
        is_multi_output = False
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} input features, "
          f"{d_output_native} output(s)")

    # 2. Split — use multi-output split for d_output > 1
    if is_multi_output:
        split_data = _split_multioutput(X, y, val_ratio=0.05, test_ratio=0.10, seed=seed)
    else:
        split_data = split_and_standardize(X, y, val_ratio=0.05, test_ratio=0.10, seed=seed)

    # 3. Configure ASANN (PDE modality)
    d_input = split_data["d_input"]
    d_output = d_output_native
    config = ASANNConfig.from_task(
        task_type="regression", modality="pde",
        d_input=d_input, d_output=d_output,
        n_samples=split_data["n_train"], device=device,
    )
    if max_epochs_override is not None:
        config.recommended_max_epochs = max_epochs_override
    config.seed = seed

    # 4. DataLoaders — drop_last=True for safety
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
        target_metric="rmse",
    )

    # 6. Train
    max_epochs = config.recommended_max_epochs
    print(f"  Training for {max_epochs} epochs (seed={seed}) on {pde_name}...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=1000,
        snapshot_every=1500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=100,
    )

    # 7. Evaluate
    metrics = evaluate_model(
        model=trainer.model,
        X_test=split_data["X_test"],
        y_test_original=split_data["y_test_original"],
        y_scaler=split_data.get("y_scaler"),
        device=device,
        y_test_scaled=split_data.get("y_test_scaled"),
    )
    arch = trainer.model.describe_architecture()
    cfg_dict = config_to_dict(config)
    cfg_dict["pde_name"] = pde_name
    cfg_dict["seed"] = seed

    # Save final checkpoint
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)

    print(f"\n  {pde_name} (seed={seed}) results:")
    for k in ("rel_l2", "rmse", "mae", "r2"):
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            print(f"    {k}: {v:.6f}")
    print(f"    Architecture: {arch.get('num_layers')} layers, "
          f"{arch.get('total_parameters')} params")

    return metrics, arch, cfg_dict, train_metrics


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    metrics, arch, cfg, _ = run_experiment_pinnacle(
        results_dir=args.results_dir, pde_name=args.name, seed=args.seed,
        max_epochs_override=args.max_epochs,
    )
    summary = {
        "pde_name": args.name, "seed": args.seed, "status": "OK",
        "metrics": metrics, "arch": arch,
    }
    summary_path = Path(args.results_dir) / "phase_f_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()
