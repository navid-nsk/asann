"""
Experiment 2a: Breast Cancer
===============================
Dataset: sklearn.datasets.load_breast_cancer
Samples: 569 | Features: 30 | Classes: 2 (malignant/benign)

Usage:
  python exp_2a_breast_cancer.py          # Single run
  python exp_2a_breast_cancer.py --cv     # 5-fold cross-validation
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
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (
    setup_paths, get_device, split_and_standardize_classification,
    create_dataloaders_classification, evaluate_classification_model,
    save_results, config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Breast Cancer classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    n_classes = len(np.unique(y))  # 2
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    # ===== 2. Split and standardize (features only) =====
    split_data = split_and_standardize_classification(
        X, y, n_classes=n_classes, val_ratio=0.15, test_ratio=0.15, seed=2
    )

    # ===== 3. Create dataloaders =====
    batch_size = 32
    loaders = create_dataloaders_classification(split_data, batch_size=batch_size)

    # ===== 4. Configure ASANN =====
    config = ASANNConfig(
        d_init=32,
        initial_num_layers=1,
        surgery_interval_init=150,
        warmup_steps=300,
        complexity_target=20000,
        meta_update_interval=500,

        # v2: Epoch-based diagnosis system
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,
        device=device,
    )

    # ===== 5. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = n_classes  # 2 classes

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
    model = trainer.model

    # ===== 6. Train =====
    max_epochs = 300
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=200,
        snapshot_every=500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )
    model = trainer.model  # Re-bind: train_epochs restores best model (new object)

    # ===== 7. Evaluate on test set =====
    metrics = evaluate_classification_model(
        model=model,
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        device=device,
        n_classes=n_classes,
    )

    # ===== 8. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


def run_cv(n_folds: int = 5):
    """Run stratified k-fold cross-validation with ASANN."""
    import time
    import json
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset

    project_root, results_base = setup_paths()
    device = get_device()

    # Load full dataset
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    n_classes = len(np.unique(y))
    d_input = X.shape[1]

    print(f"\n{'='*60}")
    print(f"  Breast Cancer -- {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)} | Features: {d_input} | Classes: {n_classes}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    total_start = time.time()

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'#'*60}")
        print(f"  Fold {fold_idx + 1}/{n_folds}")
        print(f"{'#'*60}")

        # Split train_val into train + val (85/15)
        X_trainval, y_trainval = X[train_val_idx], y[train_val_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]

        # Further split trainval -> train + val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.15,
            random_state=42, stratify=y_trainval,
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test_fold = scaler.transform(X_test_fold)

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test_fold, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_fold, dtype=torch.long)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test_fold)}")

        # Create dataloaders
        batch_size = 32
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=batch_size,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=batch_size,
        )

        # Config
        config = ASANNConfig(
            d_init=32,
            initial_num_layers=1,
            surgery_interval_init=150,
            warmup_steps=300,
            complexity_target=20000,
            meta_update_interval=500,
            diagnosis_enabled=True,
            warmup_epochs=5,
            surgery_epoch_interval=3,
            eval_epoch_interval=2,
            meta_update_epoch_interval=10,
            stability_healthy_epochs=10,
            recovery_epochs=4,
            device=device,
        )

        # Create model + trainer
        fold_dir = str(results_base / "tier_2" / "breast_cancer_cv" / f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        model = ASANNModel(d_input=d_input, d_output=n_classes, config=config)
        model.to(device)
        trainer = ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=fold_dir, task_type="classification",
            n_classes=n_classes,
            target_metric="val_loss",
        )

        # Train
        fold_start = time.time()
        trainer.train_epochs(
            train_data=train_loader,
            max_epochs=300,
            val_data=val_loader,
            test_data=test_loader,
            print_every=200,
            snapshot_every=500,
            checkpoint_path=os.path.join(fold_dir, "training_checkpoint.pt"),
            checkpoint_every_epochs=50,
        )
        model = trainer.model
        fold_time = time.time() - fold_start

        # Evaluate
        metrics = evaluate_classification_model(
            model=model, X_test=X_test_t, y_test=y_test_t,
            device=device, n_classes=n_classes,
        )
        arch = model.describe_architecture()

        widths = [l["out_features"] for l in arch["layers"]]
        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "num_layers": arch["num_layers"],
            "total_parameters": arch["total_parameters"],
            "widths": widths,
            "time": fold_time,
        })

        print(f"\n  Fold {fold_idx + 1} -- Accuracy: {metrics['accuracy']:.4f} "
              f"| F1: {metrics['f1_weighted']:.4f} "
              f"| Arch: {widths} ({arch['total_parameters']} params) "
              f"| Time: {fold_time:.1f}s")

    # Summary
    total_time = time.time() - total_start
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_weighted"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold Cross-Validation Results")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}  "
              f"Arch={r['widths']}  Params={r['total_parameters']}")
    print(f"  {'---'}")
    print(f"  Mean Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Min Accuracy:   {np.min(accs):.4f}")
    print(f"  Max Accuracy:   {np.max(accs):.4f}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"{'='*60}")

    # Save CV results
    cv_dir = str(results_base / "tier_2" / "breast_cancer_cv")
    cv_results = {
        "experiment": "Breast Cancer 5-Fold CV",
        "n_folds": n_folds,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1_weighted": float(np.mean(f1s)),
        "std_f1_weighted": float(np.std(f1s)),
        "fold_results": fold_results,
        "total_time": total_time,
    }
    with open(os.path.join(cv_dir, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n  CV results saved to {cv_dir}/cv_results.json")


if __name__ == "__main__":
    if "--cv" in sys.argv:
        run_cv(n_folds=5)
    else:
        project_root, results_base = setup_paths()
        tier_results = results_base / "tier_2"
        name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
            "Breast Cancer", run_experiment, tier_results
        )
        if status != "OK":
            print(f"\nExperiment failed: {status}")
            sys.exit(1)
