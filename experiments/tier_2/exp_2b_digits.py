"""
Experiment 2b: Digits
======================
Dataset: sklearn.datasets.load_digits
Samples: 1,797 | Features: 64 (8x8 pixel values) | Classes: 10 (digits 0-9)
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
from sklearn.datasets import load_digits

from common import (
    setup_paths, get_device, split_and_standardize_classification,
    create_dataloaders_classification, evaluate_classification_model,
    save_results, config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run Digits classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading Digits dataset...")
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)  # labels 0-9, already 0-indexed
    n_classes = len(np.unique(y))  # 10
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    # ===== 2. Split and standardize (features only) =====
    split_data = split_and_standardize_classification(
        X, y, n_classes=n_classes, val_ratio=0.05, test_ratio=0.10, seed=2
    )

    # ===== 3. Create dataloaders =====
    batch_size = 64
    loaders = create_dataloaders_classification(split_data, batch_size=batch_size)

    # ===== 4. Configure ASANN =====
    config = ASANNConfig(
        d_init=64,
        initial_num_layers=1,
        surgery_interval_init=200,
        warmup_steps=500,
        complexity_target=40000,
        meta_update_interval=600,

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
    d_output = n_classes  # 10 classes

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=torch.nn.CrossEntropyLoss(),
            log_dir=results_dir, task_type="classification",
            n_classes=n_classes,
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=torch.nn.CrossEntropyLoss(),
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
        print_every=300,
        snapshot_every=600,
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


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_2"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        "Digits", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
