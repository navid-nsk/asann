"""
Experiment 7E: Munich Leukemia Lab -- Blood Cell Classification (5-Fold CV)
============================================================================
Dataset: Munich Leukemia Lab Peripheral Blood (local, data/Munich_Leukemia_Lab/)
Samples: 41,621 microscopy images (288x288 RGB TIF)
Features: 2,048 (pretrained ResNet-50 features, frozen ImageNet weights)
Classes: 18 blood cell types (extreme imbalance: 33 -> 8,606 per class)
Task: 18-class classification of blood cells from pretrained visual features.

Competitor: Matek et al. (Nature Machine Intelligence 2019) used ResNeXt on
raw 400x400 images with 5-fold stratified CV (80/20), reporting per-class
Precision and Sensitivity (recall) with mean +/- std across folds.

We match their evaluation: 5-fold stratified CV, per-class metrics.
Fold index is controlled via LEUKEMIA_FOLD env var (0-4).

Cell types (by group):
  Lymphoid: lymphocyte, plasma_cell, lymphocyte_large_granular,
            lymphocyte_reactive, hairy_cell, lymphocyte_neoplastic, smudge_cell
  Myeloid mature: neutrophil_band, neutrophil_segmented, eosinophil,
                  basophil, monocyte
  Myeloid immature: myeloblast, metamyelocyte, promyelocyte, myelocyte,
                    promyelocyte_atypical
  Other: normoblast

Key challenge: extreme class imbalance (260x ratio) handled by FocalLoss
with capped per-class weights.
"""

import sys
import os
import json
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np
from pathlib import Path

from common import (
    setup_paths, get_device,
    create_dataloaders_classification, evaluate_classification_model,
    config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer, FocalLoss,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.asann_optimizer import ASANNOptimizerConfig
from tier_7.bio_utils import load_munich_leukemia_data, LEUKEMIA_CLASSES


def run_experiment(results_dir: str):
    """Run Munich Leukemia Lab blood cell classification experiment (single fold)."""
    device = get_device()

    # Fold index from env (0-4 for 5-fold CV, or -1 for legacy single split)
    fold_idx = int(os.environ.get("LEUKEMIA_FOLD", "-1"))
    print(f"  Fold: {fold_idx}" if fold_idx >= 0 else "  Mode: single split (legacy)")

    # ===== 1. Load dataset (ResNet-50 feature extraction, cached) =====
    print("  Loading Munich Leukemia Lab data (ResNet-50 features)...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "Munich_Leukemia_Lab")
    X, y, n_classes = load_munich_leukemia_data(data_dir)
    print(f"  Loaded: {X.shape[0]} images, {X.shape[1]} features, "
          f"{n_classes} classes")

    # ===== 2. Split =====
    from sklearn.preprocessing import StandardScaler

    if fold_idx >= 0:
        # 5-fold stratified CV: 80% train+val, 20% test per fold
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        train_val_idx, test_idx = splits[fold_idx]

        # Within train+val: hold out 10% of total as validation
        # (12.5% of the 80% train+val portion)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, random_state=42, stratify=y[train_val_idx]
        )

        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
              f"test={len(test_idx)}")
    else:
        # Legacy single split (70/15/15)
        from sklearn.model_selection import train_test_split
        n_total = len(X)
        all_idx = np.arange(n_total)
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=0.30, random_state=42, stratify=y
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=42, stratify=y[temp_idx]
        )
        print(f"  Single split: train={len(train_idx)}, val={len(val_idx)}, "
              f"test={len(test_idx)}")

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    split_data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y[train_idx], dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y[val_idx], dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y[test_idx], dtype=torch.long),
        "n_classes": n_classes,
        "d_input": X_train.shape[1],
    }

    # ===== 3. Create mini-batch dataloaders =====
    batch_size = 256
    loaders = create_dataloaders_classification(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, "
          f"steps/epoch={steps_per_epoch}")

    # ===== 4. Configure ASANN =====
    config = ASANNConfig(
        d_init=64,
        initial_num_layers=3,
        complexity_target_auto=True,
        complexity_target_multiplier=5.0,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,

        # Relaxed overfitting thresholds -- class imbalance causes larger gaps
        overfitting_gap_early=0.30,
        overfitting_gap_moderate=0.60,

        max_treatment_escalations=4,
        mixup_enabled=True,
        drop_path_enabled=True,
        ema_enabled=True,

        device=device,
        optimizer=ASANNOptimizerConfig(
            base_lr=1e-3,
            weight_decay=0.01,
        ),
    )

    # ===== 5. Focal loss with capped class weights (extreme imbalance) =====
    y_train = split_data["y_train"].numpy()
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts.astype(np.float32))
    # Cap at 20x ratio to prevent tiny classes from dominating
    max_weight_ratio = 20.0
    class_weights = np.clip(class_weights, None,
                            class_weights.min() * max_weight_ratio)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights (capped {max_weight_ratio}x):")
    for i, cls_name in enumerate(LEUKEMIA_CLASSES):
        print(f"    {cls_name:30s}  n={class_counts[i]:>5d}  w={class_weights[i].item():.3f}")
    print(f"  Weight ratio: "
          f"{class_weights.max().item() / class_weights.min().item():.1f}x")

    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

    # ===== 6. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = n_classes

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
        target_metric="f1_weighted",
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = 200
    print(f"\n  Training for {max_epochs} epochs "
          f"(~{max_epochs * steps_per_epoch} steps)...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=1000,
        snapshot_every=2000,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )

    # ===== 8. Evaluate on test set using BEST model =====
    best_model_path = os.path.join(results_dir, "best_model_full.pt")
    if os.path.exists(best_model_path):
        print(f"  Loading best model from {best_model_path}")
        eval_model = torch.load(best_model_path, map_location=device, weights_only=False)
        eval_model.to(device)
        best_val = trainer._best_val_metric
        if best_val is not None:
            print(f"  Best model loaded (val f1_weighted: {best_val:.4f})")
    else:
        print("  WARNING: best_model_full.pt not found, using final model")
        eval_model = trainer.model
        eval_model.to(device)

    metrics = evaluate_classification_model(
        eval_model, split_data["X_test"], split_data["y_test"],
        device, n_classes,
    )

    # ===== 9. Per-class Precision and Sensitivity (Recall) =====
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        precision_score, recall_score,
    )

    eval_model.eval()
    with torch.no_grad():
        X_test_t = split_data["X_test"].to(device)
        test_logits = eval_model(X_test_t).cpu()
    probs = torch.softmax(test_logits, dim=1).numpy()
    y_test_np = split_data["y_test"].numpy()
    y_pred_np = test_logits.argmax(dim=1).numpy()

    # Per-class precision and sensitivity (recall)
    per_class_precision = precision_score(
        y_test_np, y_pred_np, labels=list(range(n_classes)),
        average=None, zero_division=0
    )
    per_class_sensitivity = recall_score(
        y_test_np, y_pred_np, labels=list(range(n_classes)),
        average=None, zero_division=0
    )

    print("\n  Per-class Precision and Sensitivity:")
    print(f"  {'Class':30s}  {'Precision':>10s}  {'Sensitivity':>12s}  {'N_test':>7s}")
    print(f"  {'-'*65}")
    test_class_counts = np.bincount(y_test_np, minlength=n_classes)
    for i, cls_name in enumerate(LEUKEMIA_CLASSES):
        print(f"  {cls_name:30s}  {per_class_precision[i]:>10.4f}  "
              f"{per_class_sensitivity[i]:>12.4f}  {test_class_counts[i]:>7d}")

    # Store per-class metrics in results
    metrics["per_class_precision"] = {
        LEUKEMIA_CLASSES[i]: float(per_class_precision[i])
        for i in range(n_classes)
    }
    metrics["per_class_sensitivity"] = {
        LEUKEMIA_CLASSES[i]: float(per_class_sensitivity[i])
        for i in range(n_classes)
    }
    metrics["per_class_n_test"] = {
        LEUKEMIA_CLASSES[i]: int(test_class_counts[i])
        for i in range(n_classes)
    }

    # Per-class classification report (full)
    print("\n  Full Classification Report:")
    report = classification_report(
        y_test_np, y_pred_np,
        target_names=LEUKEMIA_CLASSES, zero_division=0,
    )
    print(report)

    # Macro AUROC (One-vs-Rest)
    try:
        auroc = float(roc_auc_score(
            y_test_np, probs, multi_class="ovr", average="macro"
        ))
        metrics["auroc_macro"] = auroc
        print(f"  Macro AUROC (OVR): {auroc:.4f}")
    except ValueError as e:
        print(f"  AUROC computation failed: {e}")
        metrics["auroc_macro"] = None

    metrics["fold"] = fold_idx
    metrics["n_train"] = len(train_idx)
    metrics["n_val"] = len(val_idx)
    metrics["n_test"] = len(test_idx)

    # ===== 10. Save results and checkpoint =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Save fold results JSON
    fold_label = f"fold{fold_idx}" if fold_idx >= 0 else "single"
    results_json_path = os.path.join(results_dir, f"results_{fold_label}.json")
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = v
    with open(results_json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {results_json_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"

    fold_idx = int(os.environ.get("LEUKEMIA_FOLD", "-1"))
    if fold_idx >= 0:
        exp_dir_name = f"leukemia_5fold_f{fold_idx}"
    else:
        exp_dir_name = "Leukemia Blood Cell"

    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        exp_dir_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
