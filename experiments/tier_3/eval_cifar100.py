"""
Standalone CIFAR-100 Evaluation Script
=======================================
Loads a saved model and evaluates on the test set.
Use this when training completed but evaluation crashed (e.g., OOM).

Usage:
    python experiments/tier_3/eval_cifar100.py

Looks for saved models in this order:
  1. best_model_full.pt  — full pickled model (saved during training at each new best)
  2. checkpoint.pt       — full trainer checkpoint (dict with "model" key)
  3. training_checkpoint.pt — periodic in-training checkpoint

Saves results to: experiments/results/tier_3/cifar-100/experiment_results.json
"""

import sys
import os
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import json
import time
import torch
import numpy as np
import asann  # noqa: F401  — activates compat shim so torch.load() can unpickle old 'csann' models
import torchvision
from sklearn.model_selection import train_test_split

from common import (
    setup_paths, get_device, evaluate_classification_model,
    save_results, config_to_dict,
)


def main():
    device = get_device()
    project_root, results_base = setup_paths()
    results_dir = results_base / "tier_3" / "cifar-100"

    # ===== 1. Find saved model =====
    # Priority: best_model_full.pt (raw pickled model) > checkpoint.pt > training_checkpoint.pt
    ckpt_candidates = [
        ("best_model_full", results_dir / "best_model_full.pt"),
        ("checkpoint", results_dir / "checkpoint.pt"),
        ("training_checkpoint", results_dir / "training_checkpoint.pt"),
    ]
    ckpt_type = None
    ckpt_path = None
    for ctype, candidate in ckpt_candidates:
        if candidate.exists():
            size_mb = candidate.stat().st_size / 1e6
            print(f"  Found {ctype}: {candidate} ({size_mb:.1f} MB)")
            ckpt_type = ctype
            ckpt_path = candidate
            break

    if ckpt_path is None:
        print(f"  ERROR: No saved model found in {results_dir}")
        print(f"  Looked for: {[str(c) for _, c in ckpt_candidates]}")
        sys.exit(1)

    # ===== 2. Load model =====
    print(f"  Loading model from: {ckpt_path}")
    loaded = torch.load(str(ckpt_path), weights_only=False, map_location=device)

    model = None
    config = None
    metadata = {}  # best_val_metric, epoch, step, etc.

    if ckpt_type == "best_model_full":
        # best_model_full.pt is a raw pickled ASANNModel object (not a dict)
        model = loaded
        print(f"  Loaded raw pickled model from best_model_full.pt")

        # Try to get metadata from the companion best_model.pt (state dict + metadata)
        best_model_meta_path = results_dir / "best_model.pt"
        if best_model_meta_path.exists():
            try:
                meta = torch.load(str(best_model_meta_path), weights_only=False, map_location="cpu")
                config = meta.get("config")
                metadata = {
                    "best_val_metric": meta.get("best_val_metric", "?"),
                    "best_val_step": meta.get("global_step", "?"),
                    "metric_name": meta.get("metric_name", "accuracy"),
                }
                print(f"  Loaded metadata from best_model.pt: "
                      f"val {metadata.get('metric_name')}={metadata.get('best_val_metric')}")
            except Exception as e:
                print(f"  Could not load metadata from best_model.pt: {e}")

    else:
        # checkpoint.pt or training_checkpoint.pt — dict with "model" key
        checkpoint = loaded

        # Try pickled model object
        if "model" in checkpoint:
            try:
                model = checkpoint["model"]
                _ = model.describe_architecture()
                print(f"  Loaded pickled model from checkpoint")
            except Exception as e:
                print(f"  Pickled model failed: {e}")
                model = None

        # Try best model copy
        if model is None and "_best_model_copy" in checkpoint and checkpoint["_best_model_copy"] is not None:
            try:
                model = checkpoint["_best_model_copy"]
                _ = model.describe_architecture()
                print(f"  Loaded best model copy from checkpoint")
            except Exception as e:
                print(f"  Best model copy failed: {e}")
                model = None

        config = checkpoint.get("config")
        metadata = {
            "current_epoch": checkpoint.get("current_epoch", "?"),
            "global_step": checkpoint.get("global_step", "?"),
            "best_val_metric": checkpoint.get("_best_val_metric", "?"),
            "best_val_epoch": checkpoint.get("_best_val_epoch", "?"),
            "best_val_step": checkpoint.get("_best_val_step", "?"),
            "target_metric": checkpoint.get("target_metric", "?"),
            "stop_reason": checkpoint.get("_stop_reason", "?"),
        }

    if model is None:
        print(f"  ERROR: Could not extract model from {ckpt_path}")
        sys.exit(1)

    model.to(device)
    model.eval()

    print(f"  Metadata: {metadata}")

    arch = model.describe_architecture()
    n_params = arch["total_parameters"]
    n_layers = arch["num_layers"]
    widths = [l["out_features"] for l in arch["layers"]]
    print(f"  Architecture: {n_layers} layers, {n_params:,} params, widths={widths}")

    # ===== 3. Load CIFAR-100 test data (same preprocessing as training) =====
    print("\n  Loading CIFAR-100 dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)

    images_raw = np.concatenate([train_ds.data, test_ds.data], axis=0)
    labels = np.concatenate([
        np.array(train_ds.targets, dtype=np.int64),
        np.array(test_ds.targets, dtype=np.int64),
    ], axis=0)
    n_classes = len(np.unique(labels))

    # Reproduce the exact same split as training
    val_ratio, test_ratio, seed = 0.15, 0.15, 42
    imgs_trainval, imgs_test, y_trainval, y_test = train_test_split(
        images_raw, labels, test_size=test_ratio, random_state=seed, stratify=labels)
    val_size = val_ratio / (1.0 - test_ratio)
    imgs_train, imgs_val, y_train, y_val = train_test_split(
        imgs_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval)

    print(f"  Split: train={len(imgs_train)}, val={len(imgs_val)}, test={len(imgs_test)}")

    # Compute per-channel normalization stats from train set (must match training)
    train_float = imgs_train.astype(np.float32) / 255.0
    channel_mean = train_float.mean(axis=(0, 1, 2))
    channel_std = np.maximum(train_float.std(axis=(0, 1, 2)), 1e-8)
    del train_float
    print(f"  Per-channel normalization: mean={channel_mean}, std={channel_std}")

    # Normalize and flatten test data
    def _normalize_and_flatten(imgs_hwc, mean, std):
        f = imgs_hwc.astype(np.float32) / 255.0
        chw = f.transpose(0, 3, 1, 2)
        chw = (chw - mean[None, :, None, None]) / std[None, :, None, None]
        return chw.reshape(len(imgs_hwc), -1)

    X_test_tensor = torch.tensor(
        _normalize_and_flatten(imgs_test, channel_mean, channel_std),
        dtype=torch.float32,
    )
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    print(f"  Test set: {X_test_tensor.shape[0]} samples, {X_test_tensor.shape[1]} features")

    # Also prepare val set for reference
    X_val_tensor = torch.tensor(
        _normalize_and_flatten(imgs_val, channel_mean, channel_std),
        dtype=torch.float32,
    )
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # ===== 4. Evaluate =====
    print("\n  Evaluating on test set (batched, batch_size=256)...")
    start_time = time.time()
    metrics = evaluate_classification_model(
        model=model, X_test=X_test_tensor, y_test=y_test_tensor,
        device=device, n_classes=n_classes, batch_size=256,
    )
    eval_time = time.time() - start_time

    print(f"\n  {'='*60}")
    print(f"  CIFAR-100 — Evaluation Results")
    print(f"  {'='*60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  ---")
    print(f"  Layers:      {n_layers}")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Widths:      {widths}")
    print(f"  Arch Cost:   {arch['architecture_cost']:.0f}")
    print(f"  Eval Time:   {eval_time:.1f}s")
    print(f"  {'='*60}")

    # Also evaluate on val set for reference
    print("\n  Evaluating on validation set...")
    val_metrics = evaluate_classification_model(
        model=model, X_test=X_val_tensor, y_test=y_val_tensor,
        device=device, n_classes=n_classes, batch_size=256,
    )
    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

    # ===== 5. Save results (same format as original experiment) =====
    config_dict = config_to_dict(config) if config is not None else {}

    # Build training_info from metadata
    training_info = {
        "actual_epochs": metadata.get("current_epoch"),
        "actual_steps": metadata.get("global_step"),
        "best_val_metric": metadata.get("best_val_metric"),
        "best_val_epoch": metadata.get("best_val_epoch"),
        "best_val_step": metadata.get("best_val_step"),
        "target_metric": metadata.get("target_metric", metadata.get("metric_name")),
        "stop_reason": metadata.get("stop_reason", "completed"),
    }

    save_results(
        results_dir=str(results_dir),
        name="CIFAR-100",
        metrics=metrics,
        arch=arch,
        config_dict=config_dict,
        elapsed_time=eval_time,
        training_info=training_info,
    )

    # Also save val metrics separately
    val_results_path = results_dir / "val_metrics.json"
    with open(val_results_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"  Val metrics saved to {val_results_path}")

    print(f"\n  Done! Results saved to {results_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
