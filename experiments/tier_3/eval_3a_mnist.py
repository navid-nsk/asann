"""
Standalone test evaluation for Experiment 3a: MNIST
====================================================
Loads the best saved model and evaluates on the test set.
Saves results to experiment_results.json in the results directory.

Usage:
    python experiments/tier_3/eval_3a_mnist.py
"""

import sys
import os
import json
import time

_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from common import get_device, evaluate_classification_model


def main():
    device = get_device()
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "tier_3", "mnist"
    )

    # ===== 1. Load dataset with same splits as training =====
    print("  Loading MNIST dataset...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    images_raw = np.concatenate([
        train_ds.data.numpy()[:, :, :, None],
        test_ds.data.numpy()[:, :, :, None],
    ], axis=0)
    labels = np.concatenate([
        train_ds.targets.numpy().astype(np.int64),
        test_ds.targets.numpy().astype(np.int64),
    ], axis=0)
    n_classes = len(np.unique(labels))

    # ===== 2. Reproduce exact train/val/test split =====
    val_ratio, test_ratio, seed = 0.05, 0.10, 42
    imgs_trainval, imgs_test, y_trainval, y_test = train_test_split(
        images_raw, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    val_size = val_ratio / (1.0 - test_ratio)
    imgs_train, imgs_val, y_train, y_val = train_test_split(
        imgs_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )
    print(f"  Split: train={len(imgs_train)}, val={len(imgs_val)}, test={len(imgs_test)}")

    # ===== 3. Compute normalization stats from train set =====
    train_float = imgs_train.astype(np.float32) / 255.0
    channel_mean = train_float.mean(axis=(0, 1, 2))
    channel_std = np.maximum(train_float.std(axis=(0, 1, 2)), 1e-8)
    del train_float

    # ===== 4. Normalize test data =====
    def normalize_and_flatten(imgs_hwc, mean, std):
        f = imgs_hwc.astype(np.float32) / 255.0
        chw = f.transpose(0, 3, 1, 2)
        chw = (chw - mean[None, :, None, None]) / std[None, :, None, None]
        return chw.reshape(len(imgs_hwc), -1)

    X_test = torch.tensor(normalize_and_flatten(imgs_test, channel_mean, channel_std), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    X_val = torch.tensor(normalize_and_flatten(imgs_val, channel_mean, channel_std), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # ===== 5. Load best model =====
    best_model_path = os.path.join(results_dir, "best_model_full.pt")
    if not os.path.exists(best_model_path):
        print(f"  ERROR: Best model not found at {best_model_path}")
        sys.exit(1)

    print(f"  Loading best model from {best_model_path}")
    model = torch.load(best_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # ===== 6. Evaluate on val and test =====
    print("\n  Evaluating on validation set...")
    val_metrics = evaluate_classification_model(
        model=model, X_test=X_val, y_test=y_val_tensor,
        device=device, n_classes=n_classes,
    )

    print("\n  Evaluating on test set...")
    test_metrics = evaluate_classification_model(
        model=model, X_test=X_test, y_test=y_test_tensor,
        device=device, n_classes=n_classes,
    )

    # ===== 7. Get architecture info =====
    arch = model.describe_architecture()

    # ===== 8. Build and save results JSON =====
    results = {
        "experiment": "MNIST",
        "model_path": best_model_path,
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "test_confusion_matrix": test_metrics.get("confusion_matrix"),
        "architecture": {
            "num_layers": arch["num_layers"],
            "total_parameters": arch["total_parameters"],
            "architecture_cost": arch["architecture_cost"],
            "widths": [l["out_features"] for l in arch["layers"]],
            "num_connections": len(arch.get("connections", [])),
        },
    }

    output_path = os.path.join(results_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # ===== 9. Print summary =====
    print(f"\n{'='*60}")
    print(f"  MNIST Test Evaluation Results")
    print(f"{'='*60}")
    print(f"  Val  Accuracy:  {val_metrics['accuracy']:.4f}  ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}  ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Test F1 Macro:  {test_metrics['f1_macro']:.4f}")
    print(f"  Test Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  Parameters:     {arch['total_parameters']:,}")
    print(f"  Layers:         {arch['num_layers']}")
    print(f"  Connections:    {len(arch.get('connections', []))}")
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
