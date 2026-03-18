"""
Experiment 2c: Covertype
==========================
Dataset: sklearn.datasets.fetch_covtype
Samples: 581,012 | Features: 54 | Classes: 7 (forest cover types)

Note: Labels are 1-7 in the original dataset. We subtract 1 to get 0-indexed
labels (0-6) as required by CrossEntropyLoss.

Note: Due to very large size, we use a stratified 50K subsample for tractability.
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
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split as sklearn_split

from common import (
    setup_paths, get_device, split_and_standardize_classification,
    create_dataloaders_classification, evaluate_classification_model,
    save_results, config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Covertype classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading Covertype dataset...")
    data = fetch_covtype(as_frame=False)
    X_full = data.data.astype(np.float32)
    y_full = data.target.astype(np.int64)

    # Labels are 1-7, convert to 0-6 for CrossEntropyLoss
    y_full = y_full - 1
    n_classes = len(np.unique(y_full))  # 7
    print(f"  Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features, {n_classes} classes")

    # ===== 1b. Stratified subsample for tractability =====
    max_samples = 50000
    if len(X_full) > max_samples:
        print(f"  Subsampling to {max_samples} samples (stratified)...")
        X, _, y, _ = sklearn_split(
            X_full, y_full, train_size=max_samples,
            random_state=42, stratify=y_full
        )
        print(f"  Subsampled: {X.shape[0]} samples")
    else:
        X, y = X_full, y_full

    # ===== 2. Split and standardize (features only) =====
    split_data = split_and_standardize_classification(
        X, y, n_classes=n_classes, val_ratio=0.05, test_ratio=0.10, seed=2
    )

    # ===== 3. Configure ASANN =====
    d_input = split_data["d_input"]
    d_output = n_classes  # 7 classes
    config = ASANNConfig.from_task(
        task_type="classification",
        modality="tabular",
        d_input=d_input,
        d_output=d_output,
        n_samples=X.shape[0],
        device=device,
    )

    # ===== 4. Create dataloaders =====
    batch_size = config.recommended_batch_size
    loaders = create_dataloaders_classification(split_data, batch_size=batch_size)

    # ===== 5. Create model and trainer =====
    # Class-weighted loss for heavily imbalanced dataset
    # Classes 0,1 are ~85% of data; class 3 is only 0.5%
    y_train = split_data["y_train"].numpy()
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights]}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)

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
    max_epochs = config.recommended_max_epochs
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=500,
        snapshot_every=1500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=5,
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


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_2"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        "Covertype", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
