"""
Experiment 3g: STL-10
=======================
Dataset: torchvision.datasets.STL10
Samples: ~13,000 (5K train + 8K test) | Features: 27648 (96x96x3 flattened) | Classes: 10

High resolution (96x96) with very few labeled training samples.

Data augmentation:
  - AutoAugment (CIFAR-10 policy) at the PIL image level
  - RandomCrop(96, padding=12) + RandomHorizontalFlip
  - Per-channel normalization (mean/std from train set)
  - GPU-side Cutout (48x48) applied in the trainer
  - Mixup (alpha=0.2)
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
from asann.config import ASANNOptimizerConfig


def run_experiment(results_dir: str):
    """Run STL-10 classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading STL-10 dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.STL10(data_dir, split='train', download=True)
    test_ds = torchvision.datasets.STL10(data_dir, split='test', download=True)

    # STL-10: .data is [N, 3, 96, 96] uint8 (CHW format!), .labels is numpy array
    # Transpose to HWC for PIL-based augmentation
    images_raw = np.concatenate([
        train_ds.data.transpose(0, 2, 3, 1),  # CHW → HWC
        test_ds.data.transpose(0, 2, 3, 1),
    ], axis=0)  # [~13K, 96, 96, 3] uint8 HWC
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
    val_ratio, test_ratio, seed = 0.03, 0.08, 42
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

    # ===== 4. Create augmented dataloaders =====
    # STL-10: 96x96, use CIFAR10 policy (no dedicated STL policy), pad=12 (proportional to 4/32)
    batch_size = 64
    loaders = create_image_augmented_loaders(
        X_train_raw=imgs_train, y_train=y_train,
        X_val_raw=imgs_val, y_val=y_val,
        X_test_raw=imgs_test, y_test=y_test,
        channel_mean=channel_mean, channel_std=channel_std,
        img_size=96, crop_padding=12,
        batch_size=batch_size, num_workers=4,
        auto_augment_policy=T.AutoAugmentPolicy.CIFAR10,
        horizontal_flip=True,
    )

    # ===== 4b. Pre-normalized flat tensors for evaluation =====
    def _normalize_and_flatten(imgs_hwc, mean, std):
        f = imgs_hwc.astype(np.float32) / 255.0
        chw = f.transpose(0, 3, 1, 2)
        chw = (chw - mean[None, :, None, None]) / std[None, :, None, None]
        return chw.reshape(len(imgs_hwc), -1)

    X_test_tensor = torch.tensor(_normalize_and_flatten(imgs_test, channel_mean, channel_std), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    d_input = 3 * 96 * 96

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["conv", "patch_embed"],
        spatial_shape=(3, 96, 96),
        c_stem_init=32,
        spatial_downsample_stages=3,   # 96->48->24->12
        max_channels=256,
        initial_num_layers=4,          # Start small, self-architect as needed
        surgery_interval_init=500,
        warmup_steps=1500,
        complexity_target_auto=True,
        layer_add_cooldown_steps=5000,
        meta_update_interval=1500,
        amp_enabled=True,
        torch_compile_enabled=False,
        use_cuda_ops=True,
        dataset_augmented=True,
        diagnosis_enabled=True,
        warmup_epochs=3,
        surgery_epoch_interval=2,
        eval_epoch_interval=1,
        meta_update_epoch_interval=5,
        stability_healthy_epochs=6,
        recovery_epochs=3,
        # Feature toggles
        op_gating_enabled=False,
        spinal_enabled=False,
        # LR tuning
        optimizer=ASANNOptimizerConfig(
            base_lr=1e-3,
            lr_controller_scale_max=3.0,
            lr_controller_dead_zone=(-0.03, 0.03),
        ),
        device=device,
    )

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
    max_epochs = 300
    checkpoint_file = os.path.join(results_dir, "training_checkpoint.pt")
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=400,
        snapshot_every=1200,
        checkpoint_path=checkpoint_file,
        checkpoint_every_epochs=40,
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
        "STL-10", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
