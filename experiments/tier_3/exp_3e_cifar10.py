"""
Experiment 3e: CIFAR-10
=========================
Dataset: torchvision.datasets.CIFAR10
Samples: 60,000 (50K train + 10K test) | Features: 3072 (32x32x3 flattened) | Classes: 10

THE standard NAS benchmark dataset.

Data augmentation:
  - AutoAugment (CIFAR-10 policy) at the PIL image level
  - RandomCrop(32, padding=4) + RandomHorizontalFlip
  - Per-channel normalization (mean/std from train set)
  - GPU-side Cutout (16x16) applied in the trainer
  - Mixup (alpha=0.2)
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
    """Run CIFAR-10 classification. Returns (metrics, arch, config_dict)."""
    device = get_device()

    # ===== 1. Load dataset (keep raw uint8 HWC format for AutoAugment) =====
    print("  Loading CIFAR-10 dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

    # CIFAR-10: .data is [N, 32, 32, 3] uint8 numpy array, .targets is list
    # Keep raw uint8 HWC for AutoAugment (PIL-level transforms need uint8 images)
    images_raw = np.concatenate([train_ds.data, test_ds.data], axis=0)  # [60K, 32, 32, 3] uint8
    labels = np.concatenate([
        np.array(train_ds.targets, dtype=np.int64),
        np.array(test_ds.targets, dtype=np.int64),
    ], axis=0)
    n_classes = len(np.unique(labels))  # 10

    # FAST_DEBUG disabled — use full dataset
    _FAST_DEBUG = False
    if _FAST_DEBUG:
        rng = np.random.RandomState(42)
        n_half = len(images_raw) // 2
        idx = rng.permutation(len(images_raw))[:n_half]
        images_raw, labels = images_raw[idx], labels[idx]
        print(f"  [FAST_DEBUG] Using {n_half}/{n_half*2} samples (50%)")

    print(f"  Loaded: {images_raw.shape[0]} samples, {images_raw.shape[1:]} images, {n_classes} classes")

    # ===== 2. Split into train/val/test (stratified, preserves class balance) =====
    val_ratio = 0.05
    test_ratio = 0.10
    seed = 42

    imgs_trainval, imgs_test, y_trainval, y_test = train_test_split(
        images_raw, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    val_size = val_ratio / (1.0 - test_ratio)
    imgs_train, imgs_val, y_train, y_val = train_test_split(
        imgs_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    print(f"  Split: train={len(imgs_train)}, val={len(imgs_val)}, test={len(imgs_test)}")

    # ===== 3. Compute per-channel normalization stats from train set =====
    # Stats are in [0,1] range (for torchvision.transforms.Normalize after ToTensor)
    train_float = imgs_train.astype(np.float32) / 255.0  # [N, 32, 32, 3] in [0,1]
    channel_mean = train_float.mean(axis=(0, 1, 2))  # [3] — per-channel mean
    channel_std = train_float.std(axis=(0, 1, 2))     # [3] — per-channel std
    channel_std = np.maximum(channel_std, 1e-8)
    del train_float  # Free memory

    print(f"  Per-channel normalization: mean={channel_mean}, std={channel_std}")

    # ===== 4. Create augmented dataloaders (AutoAugment + standard transforms) =====
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
    )

    # ===== 4b. Create pre-normalized flat tensors for evaluation =====
    # evaluate_classification_model() needs flat pre-normalized tensors
    def _normalize_and_flatten(imgs_hwc, mean, std):
        """[N,H,W,C] uint8 → [N, C*H*W] float32 normalized."""
        imgs_float = imgs_hwc.astype(np.float32) / 255.0           # [N, H, W, C] in [0,1]
        imgs_chw = imgs_float.transpose(0, 3, 1, 2)                # [N, C, H, W]
        # Per-channel normalize
        imgs_chw = (imgs_chw - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs_chw.reshape(len(imgs_hwc), -1)                  # [N, D_flat]

    X_test_flat = _normalize_and_flatten(imgs_test, channel_mean, channel_std)
    X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # d_input for model creation
    d_input = 3 * 32 * 32  # 3072

    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Train class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["conv", "patch_embed"],
        spatial_shape=(3, 32, 32),
        c_stem_init=128,
        spatial_downsample_stages="auto",  # Self-discover stride-2 stages via probing
        max_downsample_stages= 3, 
        max_channels=512,
        initial_num_layers=1,          # Start small, self-architect as needed
        surgery_interval_init=500,
        warmup_steps=1500,
        complexity_target_base_per_class=500_000,
        complexity_ceiling_mult=3.0,
        complexity_target_auto=True,
        layer_add_cooldown_steps=5000,
        meta_update_interval=1500,
        amp_enabled=True,
        torch_compile_enabled=False,
        use_cuda_ops=True,
        train_eval_max_batches=30,
        dataset_augmented=True,
        diagnosis_enabled=True,
        # ---- Epoch timing (relaxed for Fashion-MNIST) ----
        warmup_epochs=5,
        surgery_epoch_interval=4,      # Less frequent surgery (was 2)
        eval_epoch_interval=1,
        meta_update_epoch_interval=8,  # Less frequent meta updates (was 5)
        stability_healthy_epochs=10,   # More consecutive healthy needed (was 6)
        recovery_epochs=5,             # Longer recovery (was 3)
        min_recovery_epochs=5,         # Longer min recovery (was 3)
        max_recovery_epochs=20,        # Allow longer recovery for structural (was 15)
        structural_recovery_multiplier=3.0,  # Longer structural recovery (was 2.5)

        # ---- Diagnosis thresholds (relaxed to reduce false positives) ----
        overfitting_gap_early=0.30,    # Was 0.15 — too trigger-happy
        overfitting_gap_moderate=0.50, # Was 0.30
        overfitting_gap_severe=0.70,   # Was 0.50 — only truly catastrophic
        stagnation_threshold=0.003,    # Was 0.005 — tighter stagnation def
        saturation_threshold=0.90,     # Was 0.80 — higher bar for capacity exhaustion
        diagnosis_window=8,            # Was 5 — smoother trend computation
        hard_warmup_epochs=30,         # Was 20 — let model learn longer before diagnosis
        soft_warmup_epochs=15,         # Was 10
        stalled_convergence_patience=40,  # Was 25 — more patient

        # ---- Performance gate (tighter = suppress more false positives near best) ----
        perf_gate_tight=1.2,           # Was 1.5 — suppress UNDERFIT/CAPACITY when near best
        perf_gate_loose=1.5,           # Was 2.0 — suppress STAGNATION when moderately near

        # ---- Treatment intensity (gentler to prevent weight decay death spiral) ----
        dropout_light_p=0.05,          # Was 0.1 — very light dropout
        dropout_heavy_p=0.15,          # Was 0.3 — moderate instead of heavy
        wd_boost_factor=1.5,           # Was 2.0 — gentler WD boost
        wd_boost_max_stacks=2,         # Was 3 — max WD = 0.01 * 1.5^2 = 0.0225 (not 0.12!)
        lr_reduce_factor=0.7,          # Was 0.5 — gentler LR reduction
        lr_reduce_max_stacks=2,        # Was 3 — min LR = base * 0.7^2 = 0.49x (not 0.125x)
        aggressive_reg_dropout_p=0.15, # Was 0.3 — less destructive
        aggressive_reg_wd_factor=2.0,  # Was 3.0 — less destructive
        aggressive_reg_lr_factor=0.7,  # Was 0.5 — less destructive

        # ---- Dose-adaptive treatment (kid vs adult dosing) ----
        dose_adaptive_enabled=True,
        dose_reference_capacity=1024.0,  # 4 layers × 256 channels = "adult"
        dose_min_factor=0.15,
        dose_titration_levels=4,         # 4 discrete dose levels before open heart
        dose_titration_window=5,         # 5 epochs per level
        recovery_acceptable_ratio=1.0,   # Must match pre-treatment (was 1.3)
        open_heart_max_remove_frac=0.15,
        open_heart_memorization_threshold=3.0,

        # Augmentation tuning
        cutout_size=8,             # 8×8 cutout (moderate for 28×28 images)
        mixup_alpha=0.1,           # Mixup with blended labels
        elastic_enabled=False,     # Elastic is for handwriting, not clothing

        # Feature toggles
        op_gating_enabled=False,
        spinal_enabled=False,

        # ---- Gradient-proportional channel growth ----
        channel_growth_fraction=0.125,       # 12.5% of width per qualifying surgery
        channel_growth_max_fraction=0.25,    # Don't add more than 25% in one shot
        channel_growth_gds_scale=True,       # Scale by gradient demand
        per_layer_channel_budget=True,       # Each layer independent
        channel_removal_fraction=0.10,       # Remove up to 10% dead channels per layer
        channel_removal_max_fraction=0.20,   # Hard cap
        max_neurons_add_per_surgery=64,
        max_neurons_remove_per_surgery=32,
        max_channels_add_per_surgery=64,
        max_channels_remove_per_surgery=32,
        max_neuron_surgeries_per_interval=128,

        # LR tuning
        optimizer=ASANNOptimizerConfig(
            base_lr=1e-3,
            lr_controller_scale_max=3.0,
            lr_controller_dead_zone=(-0.03, 0.03),
        ),
        device=device,
    )

    # ===== 6. Create model and trainer (with resume support) =====
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
        snapshot_every=2000,
        checkpoint_path=checkpoint_file,
        checkpoint_every_epochs=25,
    )

    # ===== 8. Evaluate on test set =====
    eval_model = trainer.model
    eval_model.to(device)
    metrics = evaluate_classification_model(
        model=eval_model,
        X_test=X_test_tensor,
        y_test=y_test_tensor,
        device=device,
        n_classes=n_classes,
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
        "CIFAR-10", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
