"""
Generic OpenML-based Tier-2 classification experiment runner.

Loads any OpenML classification dataset by its dataset ID, encodes target as
integer labels, splits with a given seed, trains ASANN classifier, saves
results — same artifact set as exp_2a/2b/2c. Reports ROC AUC + balanced
accuracy + F1 + macro/weighted variants.

CLI:
    python exp_2_openml_generic.py --name phoneme --did 1489 --seed 42 \
        --results-dir results/phoneme_test --max-epochs 50
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add experiments/ dir + project root
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
    setup_paths, get_device,
    split_and_standardize_classification,
    create_dataloaders_classification,
    evaluate_classification_model,
    save_results, config_to_dict,
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


def _load_openml_classification(dataset_name: str, openml_did: int):
    """Load any OpenML classification dataset, handle categorical features
    + encode target as integer labels.

    Returns (X: float32, y: int64, n_classes: int, meta_dict).
    """
    import openml
    from sklearn.preprocessing import LabelEncoder

    print(f"  [openml] Fetching dataset id={openml_did} ({dataset_name})...")
    ds = openml.datasets.get_dataset(openml_did, download_data=True,
                                     download_qualities=False,
                                     download_features_meta_data=False)
    target_col = ds.default_target_attribute
    if target_col is None:
        raise RuntimeError(f"Dataset {openml_did} has no default_target_attribute")

    X_df, y_series, _, _ = ds.get_data(target=target_col)
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    # Encode categorical features via one-hot
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        print(f"  [openml] One-hot encoding {len(cat_cols)} categorical cols")
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dummy_na=False)

    # bool → float
    bool_cols = X_df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        X_df[c] = X_df[c].astype(np.float32)

    # Impute NaN
    if X_df.isna().any().any():
        n_nan = int(X_df.isna().sum().sum())
        print(f"  [openml] Imputing {n_nan} NaN values with column medians")
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X_df = X_df.fillna(0.0)

    # Drop any remaining non-numeric
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  [openml] Dropping non-numeric columns: {non_numeric}")
        X_df = X_df.drop(columns=non_numeric)

    X = X_df.values.astype(np.float32)

    # Encode target
    le = LabelEncoder()
    if y_series.dtype.name == "category":
        y_series = y_series.astype(str)
    y = le.fit_transform(y_series).astype(np.int64)
    n_classes = len(le.classes_)
    classes_str = list(map(str, le.classes_))

    print(f"  [openml] Loaded {dataset_name}: X={X.shape}, y={y.shape}, "
          f"n_classes={n_classes}, classes={classes_str[:5]}{'...' if n_classes>5 else ''}")
    return X, y, n_classes, {
        "dataset_name": dataset_name, "openml_did": openml_did,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "target_col": target_col, "n_classes": n_classes,
        "classes": classes_str,
    }


def run_experiment_openml_classification(results_dir: str,
                                          dataset_name: str,
                                          openml_did: int,
                                          seed: int = 42,
                                          max_epochs_override: int = None):
    """Generic OpenML classification experiment.
    Returns (metrics, arch, cfg, train_metrics)."""
    _seed_everything(seed)
    device = get_device()

    # 1. Load dataset
    X, y, n_classes, meta = _load_openml_classification(dataset_name, openml_did)

    # 2. Stratified split + standardize
    split_data = split_and_standardize_classification(
        X, y, n_classes=n_classes,
        val_ratio=0.15, test_ratio=0.15, seed=seed,
    )

    # 3. Configure ASANN
    d_input = split_data["d_input"]
    d_output = n_classes
    config = ASANNConfig.from_task(
        task_type="classification", modality="tabular",
        d_input=d_input, d_output=d_output,
        n_samples=X.shape[0], device=device,
    )
    if max_epochs_override is not None:
        config.recommended_max_epochs = max_epochs_override
    config.seed = seed

    # 4. DataLoaders — use drop_last=True on train (same fix as Tier 1 for BN-on-1)
    batch_size = config.recommended_batch_size
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(split_data["X_train"], split_data["y_train"])
    val_ds = TensorDataset(split_data["X_val"], split_data["y_val"])
    test_ds = TensorDataset(split_data["X_test"], split_data["y_test"])
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }

    # 5. Trainer
    loss_fn = torch.nn.CrossEntropyLoss()

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir, task_type="classification",
            n_classes=n_classes,
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="classification",
        n_classes=n_classes,
        target_metric="val_loss",
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
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # train_epochs restores best model

    # 7. Evaluate (pass arrays directly — same as exp_2a; function moves to GPU internally)
    metrics = evaluate_classification_model(
        model=model,
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        device=device, n_classes=n_classes,
    )

    # 8. Save final checkpoint + describe arch
    arch = model.describe_architecture()
    cfg_dict = config_to_dict(config)
    cfg_dict["openml_meta"] = meta
    cfg_dict["seed"] = seed
    cfg_dict["n_classes"] = n_classes

    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    print(f"\n  {dataset_name} (seed={seed}) classification results:")
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "auroc", "precision_macro", "recall_macro"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            print(f"    {k}: {metrics[k]:.4f}")
    print(f"    Architecture: {arch.get('num_layers')} layers, "
          f"{arch.get('total_parameters')} params")

    return metrics, arch, cfg_dict, train_metrics


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--did", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    metrics, arch, cfg, train_metrics = run_experiment_openml_classification(
        results_dir=args.results_dir,
        dataset_name=args.name, openml_did=args.did,
        seed=args.seed, max_epochs_override=args.max_epochs,
    )
    summary = {
        "dataset_name": args.name, "openml_did": args.did, "seed": args.seed,
        "status": "OK", "metrics": metrics, "arch": arch,
    }
    summary_path = Path(args.results_dir) / "phase_e_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")


if __name__ == "__main__":
    main()
