"""
Experiment 3d: SVHN (Street View House Numbers)
==================================================
Dataset: torchvision.datasets.SVHN
Samples: ~99,289 (73K train + 26K test) | Features: 3072 (32x32x3 flattened) | Classes: 10

Data augmentation:
  - AutoAugment (SVHN policy) at the PIL image level
  - RandomCrop(32, padding=4), NO horizontal flip (digits are not flip-invariant)
  - Per-channel normalization (mean/std from train set)
  - GPU-side Cutout (4x4) applied in the trainer
  - Mixup (alpha=0.1)
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
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from common import (
    setup_paths, get_device,
    create_image_augmented_loaders, evaluate_classification_model,
    save_results, config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run SVHN classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading SVHN dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.SVHN(data_dir, split='train', download=True)
    test_ds = torchvision.datasets.SVHN(data_dir, split='test', download=True)

    # SVHN: .data is [N, 3, 32, 32] uint8 (CHW format!), .labels is numpy array
    # Transpose to HWC for PIL-based augmentation
    images_raw = np.concatenate([
        train_ds.data.transpose(0, 2, 3, 1),  # CHW → HWC
        test_ds.data.transpose(0, 2, 3, 1),
    ], axis=0)  # [~99K, 32, 32, 3] uint8 HWC
    labels = np.concatenate([
        train_ds.labels.astype(np.int64),
        test_ds.labels.astype(np.int64),
    ], axis=0)
    n_classes = len(np.unique(labels))  # 10

    # TEMPORARY: Use 50% of data for faster iteration while debugging.
    _FAST_DEBUG = False
    if _FAST_DEBUG:
        rng = np.random.RandomState(42)
        n_half = len(images_raw) // 2
        idx = rng.permutation(len(images_raw))[:n_half]
        images_raw, labels = images_raw[idx], labels[idx]
        print(f"  [FAST_DEBUG] Using {n_half}/{n_half*2} samples (50%)")

    print(f"  Loaded: {images_raw.shape[0]} samples, {images_raw.shape[1:]} images, {n_classes} classes")

    # ===== 2. Split into train/val/test (stratified) =====
    val_ratio, test_ratio, seed = 0.05, 0.10, 42
    imgs_trainval, imgs_test, y_trainval, y_test = train_test_split(
        images_raw, labels, test_size=test_ratio, random_state=seed, stratify=labels)
    val_size = val_ratio / (1.0 - test_ratio)
    imgs_train, imgs_val, y_train, y_val = train_test_split(
        imgs_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval)
    print(f"  Split: train={len(imgs_train)}, val={len(imgs_val)}, test={len(imgs_test)}")

    # ===== 3. Compute per-channel normalization stats =====
    train_float = imgs_train.astype(np.float32) / 255.0
    channel_mean = train_float.mean(axis=(0, 1, 2))
    channel_std = np.maximum(train_float.std(axis=(0, 1, 2)), 1e-8)
    del train_float
    print(f"  Per-channel normalization: mean={channel_mean}, std={channel_std}")

    # ===== 4. Configure ASANN =====
    d_input = 3 * 32 * 32
    config = ASANNConfig.from_task(
        task_type="classification",
        modality="image",
        d_input=d_input,
        d_output=n_classes,
        n_samples=len(imgs_train),
        spatial_shape=(3, 32, 32),
        device=device,
    )
    # ===== 5. Create augmented dataloaders =====
    # SVHN: no horizontal flip (digits are not flip-invariant), use SVHN AutoAugment policy
    batch_size = config.recommended_batch_size
    loaders = create_image_augmented_loaders(
        X_train_raw=imgs_train, y_train=y_train,
        X_val_raw=imgs_val, y_val=y_val,
        X_test_raw=imgs_test, y_test=y_test,
        channel_mean=channel_mean, channel_std=channel_std,
        img_size=32, crop_padding=4,
        batch_size=batch_size, num_workers=4,
        auto_augment_policy=T.AutoAugmentPolicy.SVHN,
        horizontal_flip=False,  # digits should not be flipped
    )

    # ===== 5b. Pre-normalized flat tensors for evaluation =====
    def _normalize_and_flatten(imgs_hwc, mean, std):
        f = imgs_hwc.astype(np.float32) / 255.0
        chw = f.transpose(0, 3, 1, 2)
        chw = (chw - mean[None, :, None, None]) / std[None, :, None, None]
        return chw.reshape(len(imgs_hwc), -1)

    X_test_tensor = torch.tensor(_normalize_and_flatten(imgs_test, channel_mean, channel_std), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ===== 6. Create model and trainer =====
    d_output = n_classes

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
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = config.recommended_max_epochs
    checkpoint_file = os.path.join(results_dir, "training_checkpoint.pt")
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=500,
        snapshot_every=2000,
        checkpoint_path=checkpoint_file,
        checkpoint_every_epochs=20,
    )

    # ===== 8. Evaluate on test set =====
    eval_model = trainer.model
    eval_model.to(device)
    metrics = evaluate_classification_model(
        model=eval_model, X_test=X_test_tensor, y_test=y_test_tensor,
        device=device, n_classes=n_classes,
    )

    # ===== 9. Save final checkpoint =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_3"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        "SVHN", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
