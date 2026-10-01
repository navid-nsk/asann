"""
Experiment 4c: ECG5000 Time-Series Classification
====================================================
Dataset: UCR Time Series Archive -- ECG5000
Samples: 5,000 | Timesteps: 140 | Classes: 5
Task: Classify heartbeat types from single-lead ECG recordings.

Each sample is a 140-step univariate time series -> directly usable as
a [B, 140] flat input for ASANN. This is essentially a Tier 2 classification
task where the features happen to be temporal. ASANN can discover temporal
patterns (Conv1d, dilated Conv1d, EMA, attention) via surgery.

SOTA: InceptionTime / ResNet: Accuracy 99.1% - 99.6%

Multi-seed support: set ECG5000_SEED env var (default 42).
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
import json
from pathlib import Path

from common import (
    setup_paths, get_device, split_and_standardize_classification,
    create_dataloaders_classification, evaluate_classification_model,
    save_results, config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


# ---------------------------------------------------------------------------
# Seed from environment (for multi-seed runs)
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("ECG5000_SEED", "42"))


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_ecg5000(data_dir: str):
    """Download ECG5000 from UCR archive if not present."""
    data_path = Path(data_dir)

    # Check for any supported format: .tsv, .txt, or .ts
    for ext in [".tsv", ".txt"]:
        train_file = data_path / f"ECG5000_TRAIN{ext}"
        test_file = data_path / f"ECG5000_TEST{ext}"
        if train_file.exists() and test_file.exists():
            return train_file, test_file

    data_path.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import zipfile

    zip_path = data_path / "ECG5000.zip"
    url = "http://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
    print(f"  Downloading ECG5000 from {url} ...")
    urllib.request.urlretrieve(url, str(zip_path))

    print("  Extracting...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(data_path))

    # The archive may extract into a subfolder; try multiple formats
    for candidate in [data_path, data_path / "ECG5000"]:
        for ext in [".tsv", ".txt"]:
            t = candidate / f"ECG5000_TRAIN{ext}"
            te = candidate / f"ECG5000_TEST{ext}"
            if t.exists() and te.exists():
                return t, te

    raise FileNotFoundError(
        f"Could not find ECG5000 data files in {data_path}. "
        f"Found: {[f.name for f in data_path.iterdir()]}"
    )


def _load_ecg5000(data_dir: str):
    """Load ECG5000 dataset and return X, y arrays.

    Supports both .tsv (tab-separated) and .txt (whitespace-separated) formats.
    Both formats: first column is label, rest are 140 timestep features.
    """
    train_file, test_file = _download_ecg5000(data_dir)

    # np.loadtxt handles both tab and whitespace-separated formats
    train_data = np.loadtxt(str(train_file))
    test_data = np.loadtxt(str(test_file))

    # Combine train + test, then re-split with our standard ratios
    all_data = np.vstack([train_data, test_data])

    # First column is the label (1-indexed), rest are 140 timesteps
    y = all_data[:, 0].astype(np.int64)
    X = all_data[:, 1:].astype(np.float32)

    # Convert labels to 0-indexed
    unique_labels = np.sort(np.unique(y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y], dtype=np.int64)

    return X, y, len(unique_labels)


def run_experiment(results_dir: str):
    """Run ECG5000 classification. Returns (metrics, arch, config_dict, train_metrics)."""
    _set_seed(SEED)
    device = get_device()

    # ===== 1. Load dataset =====
    print(f"  Loading ECG5000 dataset (seed={SEED})...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ECG5000")
    X, y, n_classes = _load_ecg5000(data_dir)
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} timesteps, {n_classes} classes")

    # ===== 2. Split and standardize (like Tier 2) =====
    split_data = split_and_standardize_classification(
        X, y, n_classes=n_classes, val_ratio=0.05, test_ratio=0.10, seed=SEED
    )

    # ===== 3. Class-weighted loss for imbalanced classes =====
    y_train = split_data["y_train"].numpy()
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    print(f"  Class distribution: {dict(enumerate(class_counts.tolist()))}")
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # ===== 4. Create dataloaders =====
    batch_size = 64
    loaders = create_dataloaders_classification(split_data, batch_size=batch_size)

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["autoencoder", "temporal", "transformer"],
        d_init=16,
        initial_num_layers=1,
        surgery_interval_init=200,
        warmup_steps=400,
        complexity_target=50000,
        meta_update_interval=500,

        # v2: Epoch-based diagnosis system
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=1,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=12,
        recovery_epochs=4,
        
        max_ops_per_layer=5,
        complexity_target_auto=True,
        complexity_ceiling_mult=3.0,
        hard_max_multiplier=1.0,      # Cap diagnosis extension (default 3.0 too long)
        device=device,
    )

    # ===== 6. Create model and trainer =====
    d_input = split_data["d_input"]  # 140
    d_output = n_classes              # 5

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
        target_metric="auc_ovr",
    )
    model = trainer.model

    # ===== 7. Train =====
    max_epochs = 200
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = {}
    try:
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
    except (ValueError, RuntimeError) as e:
        print(f"  [WARNING] Training ended with error: {e}")
        print(f"  Restoring best model from trainer state...")
        trainer._restore_best_model()
    model = trainer.model

    # ===== 8. Evaluate on test set =====
    metrics = evaluate_classification_model(
        model=model,
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        device=device,
        n_classes=n_classes,
    )

    # ===== 9. Save final checkpoint =====
    arch = model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Save seed-specific results
    result_file = os.path.join(results_dir, f"results_s{SEED}.json")
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_4"

    # Results directory includes seed for multi-seed support
    exp_name = f"ecg5000_classification_s{SEED}"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        exp_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
