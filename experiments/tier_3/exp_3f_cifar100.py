"""
Experiment 3f: CIFAR-100
==========================
Dataset: torchvision.datasets.CIFAR100
Samples: 60,000 (50K train + 10K test) | Features: 3072 (32x32x3 flattened) | Classes: 100

Data augmentation (strong):
  - AutoAugment (CIFAR-10 policy) at the PIL image level
  - ColorJitter (brightness/contrast/saturation=0.3, hue=0.1)
  - RandomCrop(32, padding=4) + RandomHorizontalFlip
  - Per-channel normalization (mean/std from train set)
  - RandomErasing (p=0.25) after normalization
  - GPU-side Cutout (16x16) applied in the trainer
  - Mixup (alpha=0.2)
  - Label smoothing (0.1)
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
    """Run CIFAR-100 classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset (keep raw uint8 HWC format for AutoAugment) =====
    print("  Loading CIFAR-100 dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)

    # CIFAR-100: .data is [N, 32, 32, 3] uint8 numpy array (HWC), .targets is list
    images_raw = np.concatenate([train_ds.data, test_ds.data], axis=0)  # [60K, 32, 32, 3] uint8
    labels = np.concatenate([
        np.array(train_ds.targets, dtype=np.int64),
        np.array(test_ds.targets, dtype=np.int64),
    ], axis=0)
    n_classes = len(np.unique(labels))  # 100

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

    # ===== 3. Compute per-channel normalization stats from train set =====
    train_float = imgs_train.astype(np.float32) / 255.0
    channel_mean = train_float.mean(axis=(0, 1, 2))  # [3]
    channel_std = np.maximum(train_float.std(axis=(0, 1, 2)), 1e-8)
    del train_float
    print(f"  Per-channel normalization: mean={channel_mean}, std={channel_std}")

    # ===== 4. Create augmented dataloaders =====
    batch_size = 256
    loaders = create_image_augmented_loaders(
        X_train_raw=imgs_train, y_train=y_train,
        X_val_raw=imgs_val, y_val=y_val,
        X_test_raw=imgs_test, y_test=y_test,
        channel_mean=channel_mean, channel_std=channel_std,
        img_size=32, crop_padding=4,
        batch_size=batch_size, num_workers=4,
        auto_augment_policy=T.AutoAugmentPolicy.CIFAR10,
        horizontal_flip=True,
        extra_transforms=[
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ],
    )
    # Add RandomErasing after Normalize in the train transform pipeline
    train_tf = loaders["train"].dataset.transform
    loaders["train"].dataset.transform = T.Compose(
        train_tf.transforms + [T.RandomErasing(p=0.25, scale=(0.02, 0.2))]
    )

    # ===== 4b. Pre-normalized flat tensors for evaluation =====
    def _normalize_and_flatten(imgs_hwc, mean, std):
        f = imgs_hwc.astype(np.float32) / 255.0
        chw = f.transpose(0, 3, 1, 2)
        chw = (chw - mean[None, :, None, None]) / std[None, :, None, None]
        return chw.reshape(len(imgs_hwc), -1)

    X_test_tensor = torch.tensor(_normalize_and_flatten(imgs_test, channel_mean, channel_std), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    d_input = 3 * 32 * 32  # 3072

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["conv", "patch_embed"],
        spatial_shape=(3, 32, 32),
        c_stem_init=96,
        spatial_downsample_stages=3,   # 32->16->8->4
        max_channels=512,
        initial_num_layers=4,          # Start small, self-architect as needed
        surgery_interval_init=600,
        warmup_steps=2000,
        complexity_target_base_per_class=1_000_000,
        complexity_ceiling_mult=5.0,
        complexity_target_auto=True,
        layer_add_cooldown_steps=5000,
        meta_update_interval=2000,
        amp_enabled=True,
        torch_compile_enabled=False,
        use_cuda_ops=True,
        dataset_augmented=True,
        diagnosis_enabled=True,
        train_eval_max_batches=30,
        warmup_epochs=5,
        surgery_epoch_interval=5,
        eval_epoch_interval=1,
        meta_update_epoch_interval=5,
        stability_healthy_epochs=15,
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
            task_loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
            log_dir=results_dir, task_type="classification",
            n_classes=n_classes,
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
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
        print_every=500,
        snapshot_every=2500,
        checkpoint_path=checkpoint_file,
        checkpoint_every_epochs=25,
    )

    # ===== 8. Save final checkpoint BEFORE evaluation =====
    # trainer.model is the restored best model; local `model` may be stale (final arch)
    eval_model = trainer.model
    eval_model.to(device)
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # ===== 9. Evaluate on test set (using best model) =====
    metrics = evaluate_classification_model(
        model=eval_model, X_test=X_test_tensor, y_test=y_test_tensor,
        device=device, n_classes=n_classes,
    )

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_3"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        "CIFAR-100", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
