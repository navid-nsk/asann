"""
ASANN Experiments -- Shared Utilities

Provides preprocessing, splitting, evaluation, and results saving
for regression (Tier 1), classification (Tier 2/3), and
time-series / sequence (Tier 4) experiments.
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)
from typing import Dict, Any, Optional, List, Callable
from torch.utils.data import DataLoader, TensorDataset


# ===== Metric Direction Registry =====
# Maps metric names to whether higher values are better (True) or lower (False).
# Used by the trainer to determine when a new "best" model has been found.
METRIC_DIRECTIONS = {
    # Classification (higher = better)
    "accuracy": True, "f1_macro": True, "f1_weighted": True,
    "precision_macro": True, "recall_macro": True, "balanced_accuracy": True,
    "auroc": True, "auc_ovr": True,
    # Regression (higher = better)
    "r2": True, "ccc": True, "pearson_r": True, "spearman_rho": True,
    # Regression / Forecasting (lower = better)
    "mae": False, "rmse": False, "mape": False, "mse": False, "rel_l2": False,
    # Loss-based tracking (lower = better) -- useful for small datasets where
    # discrete metrics (accuracy, F1) saturate at 1.0 and stop providing signal.
    "val_loss": False,
}


# ===== Loss Functions =====

class FocalLoss(torch.nn.Module):
    """Focal Loss for imbalanced classification (Lin et al., 2017).

    Downweights easy (well-classified) examples so the model focuses on hard
    ones. Much better than plain class-weighted CE for extreme imbalance:

      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With gamma=2.0:
      - Confident correct (p_t=0.9): weight = (0.1)^2 = 0.01  → nearly ignored
      - Uncertain      (p_t=0.5): weight = (0.5)^2 = 0.25  → moderate
      - Wrong          (p_t=0.1): weight = (0.9)^2 = 0.81  → full attention

    Args:
        alpha: Per-class weights [n_classes] tensor, or None for uniform.
               Unlike raw CE weights, focal loss already downweights easy
               examples, so alpha should be modest (cap at ~10x).
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard
               examples. gamma=0 recovers standard CE.
        reduction: 'mean' (default) or 'none'.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor)
                                 else torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] raw model outputs (before softmax)
            targets: [N] integer class labels
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # probability assigned to correct class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss


def get_metric_direction(metric_name: str) -> bool:
    """Return True if higher is better for the given metric.

    Raises ValueError for unknown metric names.
    """
    if metric_name not in METRIC_DIRECTIONS:
        raise ValueError(
            f"Unknown target_metric '{metric_name}'. "
            f"Valid metrics: {sorted(METRIC_DIRECTIONS.keys())}"
        )
    return METRIC_DIRECTIONS[metric_name]


def setup_paths():
    """Add project root to sys.path and return results base directory."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    return project_root, results_dir


def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_batch_size(n_train: int) -> int:
    """Heuristic batch size based on training set size."""
    if n_train < 500:
        return 32
    elif n_train < 2000:
        return 64
    elif n_train < 10000:
        return 128
    else:
        return 256


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics on original-scale values.

    Returns: MAE, RMSE, MAPE, R², CCC (Concordance Correlation Coefficient).
    """
    y_t = y_true.ravel()
    y_p = y_pred.ravel()

    # Filter out NaN targets (used in graph full-batch regression where
    # non-target splits are masked with NaN).
    valid = ~np.isnan(y_t)
    if valid.sum() == 0:
        return {
            "mae": float("nan"), "mse": float("nan"), "rmse": float("nan"),
            "r2": float("nan"), "mape": float("nan"), "ccc": float("nan"),
            "pearson_r": float("nan"), "spearman_rho": float("nan"),
        }
    y_t = y_t[valid]
    y_p = y_p[valid]

    mae = mean_absolute_error(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_t, y_p)

    # MAPE — protect against zero / near-zero targets
    mask = np.abs(y_t) > 1e-8
    if mask.any():
        mape = mean_absolute_percentage_error(y_t[mask], y_p[mask])
    else:
        mape = float("nan")

    # CCC (Concordance Correlation Coefficient)
    mean_t, mean_p = y_t.mean(), y_p.mean()
    var_t, var_p = y_t.var(), y_p.var()
    cov_tp = np.mean((y_t - mean_t) * (y_p - mean_p))
    ccc = (2.0 * cov_tp) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-12)

    # Pearson and Spearman correlation
    from scipy.stats import pearsonr, spearmanr
    if len(y_t) > 2:
        pearson_r = float(pearsonr(y_t, y_p)[0])
        spearman_rho = float(spearmanr(y_t, y_p)[0])
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")

    # Relative L2 error: ||pred - true||_2 / ||true||_2
    true_norm = np.linalg.norm(y_t)
    rel_l2 = float(np.linalg.norm(y_p - y_t) / true_norm) if true_norm > 1e-12 else float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "rel_l2": rel_l2,
        "ccc": float(ccc),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    n_classes: int,
                                    ignore_index: int = -100,
                                    y_probs: Optional[np.ndarray] = None,
                                    ) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Filters out samples where y_true == ignore_index (used for masked nodes
    in graph node classification where only a subset of nodes have labels).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (argmax of logits).
        n_classes: Number of classes.
        ignore_index: Label value to ignore (e.g., masked nodes in GNN).
        y_probs: Optional (N, n_classes) probability array for AUROC computation.
                 When provided, computes 'auroc' (binary) or 'auc_ovr' (multi-class).

    Returns: accuracy, F1 (macro & weighted), precision (macro), recall (macro),
             and optionally auroc/auc_ovr if y_probs is provided.
    """
    # Filter out masked/ignored samples
    valid_mask = y_true != ignore_index
    if valid_mask.sum() == 0:
        return {
            "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0,
        }
    y_true_f = y_true[valid_mask]
    y_pred_f = y_pred[valid_mask]

    acc = accuracy_score(y_true_f, y_pred_f)
    from sklearn.metrics import balanced_accuracy_score
    bal_acc = balanced_accuracy_score(y_true_f, y_pred_f)
    f1_m = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    f1_w = f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)
    prec = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    rec = recall_score(y_true_f, y_pred_f, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
    }

    # AUROC / AUC-OVR when probability predictions are available
    if y_probs is not None:
        from sklearn.metrics import roc_auc_score
        # Filter probs with the same mask
        y_probs_f = y_probs[valid_mask]
        try:
            if n_classes == 2:
                metrics["auroc"] = float(roc_auc_score(y_true_f, y_probs_f[:, 1]))
            else:
                metrics["auc_ovr"] = float(roc_auc_score(
                    y_true_f, y_probs_f, multi_class="ovr", average="macro",
                ))
        except ValueError:
            # Can happen if a class is entirely missing from y_true_f
            pass

    return metrics


def preprocess_dataframe(df, target_col: str, drop_cols: Optional[List[str]] = None,
                         max_missing_frac: float = 0.5):
    """Preprocess a pandas DataFrame for regression.

    1. Drop specified columns
    2. Drop columns with more than max_missing_frac fraction of NaN values
    3. Fill remaining missing values: median for numeric, mode for categorical
    4. One-hot encode categorical columns
    5. Separate features and target

    Returns (X as numpy array, y as numpy array, feature_names list)
    """
    import pandas as pd

    df = df.copy()

    # Drop specified columns
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Separate target before preprocessing
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")
    y = df[target_col].values.astype(np.float32)
    df = df.drop(columns=[target_col])

    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > max_missing_frac].index.tolist()
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns with >{max_missing_frac*100:.0f}% missing: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        df = df.drop(columns=cols_to_drop)

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Fill missing values
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "MISSING")

    # One-hot encode categoricals
    if categorical_cols:
        print(f"  One-hot encoding {len(categorical_cols)} categorical columns...")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)

    # Convert to float32
    X = df.values.astype(np.float32)
    feature_names = df.columns.tolist()

    # Remove any remaining NaN/inf
    nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    if nan_mask.any():
        print(f"  Removing {nan_mask.sum()} rows with NaN/inf values")
        X = X[~nan_mask]
        y = y[~nan_mask]

    print(f"  Preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_names


def split_and_standardize(X: np.ndarray, y: np.ndarray,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          seed: int = 42) -> Dict[str, Any]:
    """Split data into train/val/test and standardize.

    - 70/15/15 split (default)
    - StandardScaler fit on train only (both X and y)
    - Returns dict with tensors, scalers, and dimensions
    """
    # First split: train+val vs test
    test_size = test_ratio
    val_size = val_ratio / (1.0 - test_ratio)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Standardize X (fit on train only)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    # Standardize y (fit on train only)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Convert to tensors
    device = get_device()
    data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test_scaled": torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1),
        "y_test_original": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "d_input": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    return data


def create_dataloaders(data: Dict[str, Any], batch_size: int) -> Dict[str, DataLoader]:
    """Create DataLoaders from the data dict."""
    train_ds = TensorDataset(data["X_train"], data["y_train"])
    val_ds = TensorDataset(data["X_val"], data["y_val"])
    test_ds = TensorDataset(data["X_test"], data["y_test_scaled"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


class MolIndexDataset(torch.utils.data.Dataset):
    """TensorDataset that also yields a molecule index for each sample.

    Used by molecular mini-batch dataloaders so the MolecularGraphEncoder
    knows which molecule graphs to process in each mini-batch.

    Each sample returns (x, y, global_mol_idx) where global_mol_idx
    indexes into the concatenated [train, val, test] molecular graphs list.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, start_idx: int):
        self.X = X
        self.y = y
        self.start_idx = start_idx  # offset into global graphs list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], torch.tensor(self.start_idx + idx, dtype=torch.long)


def create_mol_dataloaders(data: Dict[str, Any], batch_size: int,
                           drop_last: bool = False) -> Dict[str, DataLoader]:
    """Create DataLoaders for molecular regression with molecule indices.

    Each batch yields (x, y, mol_idx) where mol_idx maps into the
    concatenated [train, val, test] molecular graphs list.

    Args:
        drop_last: If True, drop last incomplete training batch. Useful for
                   small datasets where the last batch may have only 1 sample,
                   causing BatchNorm to fail.
    """
    n_train = len(data["X_train"])
    n_val = len(data["X_val"])

    train_ds = MolIndexDataset(data["X_train"], data["y_train"], start_idx=0)
    val_ds = MolIndexDataset(data["X_val"], data["y_val"], start_idx=n_train)
    test_ds = MolIndexDataset(data["X_test"], data["y_test_scaled"], start_idx=n_train + n_val)

    # Auto-enable drop_last if last batch would have only 1 sample
    # (causes BatchNorm failure during training)
    if not drop_last and n_train % batch_size == 1:
        drop_last = True

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            drop_last=drop_last),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


def create_mol_dataloaders_classification(data: Dict[str, Any],
                                           batch_size: int) -> Dict[str, DataLoader]:
    """Create DataLoaders for molecular classification with molecule indices.

    Each batch yields (x, y, mol_idx) where mol_idx maps into the
    concatenated [train, val, test] molecular graphs list.
    """
    n_train = len(data["X_train"])
    n_val = len(data["X_val"])

    train_ds = MolIndexDataset(data["X_train"], data["y_train"], start_idx=0)
    val_ds = MolIndexDataset(data["X_val"], data["y_val"], start_idx=n_train)
    test_ds = MolIndexDataset(data["X_test"], data["y_test"], start_idx=n_train + n_val)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


def evaluate_model(model, X_test: torch.Tensor, y_test_original: torch.Tensor,
                   y_scaler, device: str,
                   y_test_scaled: torch.Tensor = None) -> Dict[str, float]:
    """Evaluate model on test set, reporting metrics on original scale.

    1. Forward pass to get predictions (in standardized space)
    2. Inverse-transform predictions back to original scale
    3. Compute MSE, RMSE, MAE, R² on original scale
    4. If y_test_scaled provided, also compute normalized RMSE
    """
    model.eval()
    with torch.no_grad():
        X = X_test.to(device)
        preds_scaled = model(X).cpu().numpy()

    # Inverse-transform predictions to original scale
    preds_original = y_scaler.inverse_transform(preds_scaled)
    y_true = y_test_original.numpy()

    # Compute metrics on original scale
    mse = mean_squared_error(y_true, preds_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, preds_original)
    r2 = r2_score(y_true, preds_original)

    # Relative L2 error: ||pred - true||_2 / ||true||_2
    # Standard PDE benchmark metric (PINNacle, DeepXDE, etc.)
    diff_norm = np.linalg.norm(preds_original.ravel() - y_true.ravel())
    true_norm = np.linalg.norm(y_true.ravel())
    rel_l2 = float(diff_norm / true_norm) if true_norm > 1e-12 else float("nan")

    result = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "rel_l2": rel_l2,
    }

    # Normalized RMSE: computed on standardized (zero-mean, unit-var) targets
    if y_test_scaled is not None:
        y_scaled_np = y_test_scaled.numpy()
        rmse_norm = np.sqrt(mean_squared_error(y_scaled_np, preds_scaled))
        result["rmse_normalized"] = float(rmse_norm)

    return result


def print_results(name: str, metrics: Dict[str, float], arch: Dict[str, Any],
                  elapsed_time: float):
    """Print formatted experiment results."""
    print(f"\n{'='*60}")
    print(f"  {name} - Results")
    print(f"{'='*60}")
    # Support both regression (mse/rmse/mae/r2) and classification (test_accuracy)
    if 'test_accuracy' in metrics:
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        if 'num_classes' in metrics:
            print(f"  Classes:       {metrics['num_classes']}")
        if 'graph_ops_discovered' in metrics:
            print(f"  Graph Ops:     {metrics['graph_ops_discovered']}")
    elif 'mse' in metrics:
        print(f"  MSE:  {metrics['mse']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        if 'rmse_normalized' in metrics:
            print(f"  RMSE (normalized): {metrics['rmse_normalized']:.6f}")
        if 'rmse_log' in metrics:
            print(f"  RMSE (log scale):  {metrics['rmse_log']:.6f}")
        if 'rmse_dollars' in metrics:
            print(f"  RMSE (dollars):    {metrics['rmse_dollars']:.2f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        if 'rel_l2' in metrics:
            print(f"  Rel. L2 Error: {metrics['rel_l2']:.6f}")
        print(f"  R2:   {metrics['r2']:.6f}")
        if 'r2_dollars' in metrics:
            print(f"  R2 (dollars):      {metrics['r2_dollars']:.6f}")
    else:
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
    print(f"  ---")
    print(f"  Layers:      {arch.get('num_layers', '?')}")
    print(f"  Parameters:  {arch.get('total_parameters', '?')}")
    print(f"  Connections: {len(arch.get('connections', []))}")
    widths = [l['out_features'] for l in arch.get('layers', [])]
    print(f"  Widths:      {widths}")
    print(f"  Arch Cost:   {arch.get('architecture_cost', 0):.0f}")
    print(f"  Time:        {elapsed_time:.1f}s")
    print(f"{'='*60}\n")


def save_results(results_dir: str, name: str, metrics: Dict[str, float],
                 arch: Dict[str, Any], config_dict: Dict[str, Any],
                 elapsed_time: float, training_info: Optional[Dict] = None):
    """Save experiment results to JSON."""
    results = {
        "experiment": name,
        "metrics": metrics,
        "architecture": {
            "num_layers": arch["num_layers"],
            "total_parameters": arch["total_parameters"],
            "architecture_cost": arch["architecture_cost"],
            "widths": [l["out_features"] for l in arch["layers"]],
            "num_connections": len(arch.get("connections", [])),
        },
        "config": config_dict,
        "elapsed_time_seconds": elapsed_time,
    }

    # Add auto-stop / training info if available
    if training_info is not None:
        results["training_info"] = {
            "stop_reason": training_info.get("stop_reason"),
            "actual_steps": training_info.get("actual_steps"),
            "actual_epochs": training_info.get("actual_epochs"),
            "min_steps": training_info.get("min_steps"),
            "min_epochs": training_info.get("min_epochs"),
            "effective_max_steps": training_info.get("effective_max_steps"),
            "effective_max_epochs": training_info.get("effective_max_epochs"),
            "steps_per_epoch": training_info.get("steps_per_epoch"),
            "arch_stable_step": training_info.get("arch_stable_step"),
            "arch_stable_epoch": training_info.get("arch_stable_epoch"),
            "best_val_step": training_info.get("best_val_step"),
            "best_val_epoch": training_info.get("best_val_epoch"),
            "best_val_metric": training_info.get("best_val_metric"),
            "target_metric": training_info.get("target_metric"),
        }
        # Include best model test metrics if available
        if training_info.get("best_test_metrics"):
            results["best_model_test_metrics"] = training_info["best_test_metrics"]
        # Include last (final) model test metrics if available
        if training_info.get("last_test_metrics"):
            results["last_model_test_metrics"] = training_info["last_test_metrics"]

    path = os.path.join(results_dir, "experiment_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")


def config_to_dict(config) -> Dict[str, Any]:
    """Convert ASANNConfig to a JSON-serializable dict (handles nested dataclasses)."""
    from dataclasses import fields, asdict
    import dataclasses
    result = {}
    for k, v in config.__dict__.items():
        if k.startswith("_"):
            continue
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            result[k] = asdict(v)
        else:
            result[k] = v
    return result


def _backup_existing_csvs(results_dir: str):
    """Back up existing CSV log files before a fresh run overwrites them.

    Creates a timestamped subdirectory (e.g. previous_run_20260216_143022/)
    and moves all CSV files into it. This prevents data loss when a crashed
    run is followed by a fresh restart.
    """
    import glob
    from datetime import datetime

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        return  # Nothing to back up

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(results_dir, f"previous_run_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    import shutil
    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        shutil.move(csv_path, os.path.join(backup_dir, fname))

    # Also back up experiment_results.json if it exists
    results_json = os.path.join(results_dir, "experiment_results.json")
    if os.path.exists(results_json):
        shutil.move(results_json, os.path.join(backup_dir, "experiment_results.json"))

    print(f"  [BACKUP] Moved {len(csv_files)} CSV files to {backup_dir}")


def resume_or_create_trainer(
    results_dir: str,
    create_fn: Callable,
    task_loss_fn: Callable,
    task_type: str = "regression",
    y_scaler=None,
    n_classes: Optional[int] = None,
    checkpoint_name: str = "training_checkpoint.pt",
    input_transform=None,
    extra_params=None,
    target_metric: Optional[str] = None,
):
    """Check for existing checkpoint and resume, or create fresh trainer.

    Searches for checkpoints in priority order:
      1. training_checkpoint.pt — periodic in-training checkpoint (preferred)
      2. checkpoint.pt — final checkpoint from a previous run (fallback)

    If neither exists or loading fails, creates a fresh trainer and backs up
    any existing CSV log files to prevent data loss.

    Args:
        results_dir: Path to results directory (contains checkpoints).
        create_fn: Zero-arg callable that creates a fresh ASANNTrainer
                   (or a tuple ending with one).
        task_loss_fn: Loss function (needed for resume since it's not serializable).
        task_type: "regression" or "classification".
        y_scaler: Scaler for regression targets.
        n_classes: Number of classes for classification.
        checkpoint_name: Filename for the in-training checkpoint.
        input_transform: Optional callable to transform x_batch before model.
        extra_params: Optional list of extra parameters for the optimizer.
        target_metric: Metric name for best-model tracking (e.g., "rmse", "auroc").
                       If None, defaults to "accuracy" (classification) or "r2" (regression).

    Returns:
        (trainer, is_resumed) tuple.
    """
    from asann.trainer import ASANNTrainer

    # Check for in-training checkpoint (from a crashed/interrupted run)
    training_ckpt = os.path.join(results_dir, checkpoint_name)

    if os.path.exists(training_ckpt):
        ckpt_size = os.path.getsize(training_ckpt)
        print(f"  [RESUME] Found in-training checkpoint at {training_ckpt} "
              f"({ckpt_size / 1e6:.1f} MB)")
        try:
            trainer = ASANNTrainer.load_checkpoint(
                path=training_ckpt,
                task_loss_fn=task_loss_fn,
                log_dir=results_dir,
                task_type=task_type,
                y_scaler=y_scaler,
                n_classes=n_classes,
                input_transform=input_transform,
                extra_params=extra_params,
                target_metric=target_metric,
            )
            return trainer, True
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"  [RESUME] FAILED to load checkpoint: {e}")
            print(f"  [RESUME] This can happen when code was modified between runs.")
            print(f"  [RESUME] Falling back to fresh start.")
            print(f"{'!'*60}\n")
            import traceback
            traceback.print_exc()
            # Fall through to fresh start
    else:
        print(f"  [RESUME] No checkpoint found at {training_ckpt}")

    # No checkpoint found or all failed — back up existing CSVs and start fresh
    _backup_existing_csvs(results_dir)
    print(f"  [FRESH] Starting fresh training")

    result = create_fn()
    if isinstance(result, tuple):
        # create_fn returned (model, trainer) or similar
        trainer = result[-1] if hasattr(result[-1], 'train') else result[0]
    else:
        trainer = result

    # Override target_metric on freshly created trainer if specified.
    # The create_fn closure typically doesn't know about target_metric,
    # so we apply it here after creation.
    if target_metric is not None and hasattr(trainer, 'target_metric'):
        if trainer.target_metric != target_metric:
            trainer.target_metric = target_metric
            trainer._target_metric_higher_is_better = get_metric_direction(target_metric)
            print(f"  [FRESH] Target metric set to: {target_metric} "
                  f"(higher_is_better={trainer._target_metric_higher_is_better})")

    return trainer, False


def run_experiment_wrapper(name: str, run_fn: Callable, results_base_dir: Path):
    """Wrapper that handles timing, error catching, and results directory setup.

    run_fn signature: run_fn(results_dir: str) -> (metrics, arch, config_dict[, training_info])
    Returns (name, metrics, arch, elapsed_time, status)
    """
    print(f"\n{'#'*60}")
    print(f"  Starting: {name}")
    print(f"{'#'*60}\n")

    results_dir = results_base_dir / name.lower().replace(" ", "_").replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        result = run_fn(str(results_dir))
        # Support both 3-tuple and 4-tuple returns (backward compatible)
        if len(result) == 4:
            metrics, arch, config_dict, training_info = result
        else:
            metrics, arch, config_dict = result
            training_info = None
        elapsed = time.time() - start_time
        print_results(name, metrics, arch, elapsed)
        save_results(str(results_dir), name, metrics, arch, config_dict, elapsed, training_info)
        return name, metrics, arch, elapsed, "OK"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return name, None, None, elapsed, f"FAILED: {e}"


# ==================== Classification Support ====================


def split_and_standardize_classification(X: np.ndarray, y: np.ndarray,
                                          n_classes: int,
                                          val_ratio: float = 0.15,
                                          test_ratio: float = 0.15,
                                          seed: int = 42,
                                          spatial_shape=None) -> Dict[str, Any]:
    """Split data into train/val/test and normalize features.

    For image data (spatial_shape provided):
        Uses per-channel mean/std normalization (standard practice for CIFAR etc).
        Data stays in a natural range so augmentation (color jitter, cutout) works correctly.
    For tabular data (spatial_shape=None):
        Uses sklearn StandardScaler (z-scores each feature independently).

    Key differences from regression split_and_standardize():
    - Uses stratified splitting to preserve class balance
    - y is kept as int64 / torch.long (class labels)
    - y is NOT unsqueezed (CrossEntropyLoss expects [N] shape)
    """
    test_size = test_ratio
    val_size = val_ratio / (1.0 - test_ratio)

    # Stratified split to preserve class distributions
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    x_scaler = None
    if spatial_shape is not None:
        # Image data: per-channel normalization (NOT per-pixel StandardScaler).
        # StandardScaler z-scores each of 3072 pixels independently, destroying
        # spatial statistics and breaking augmentation (color jitter, cutout fill).
        # Per-channel normalization keeps values in a natural ~[-2, +2] range.
        C, H, W = spatial_shape
        D = C * H * W

        # Reshape to (N, C, H*W) to compute per-channel stats from train set
        train_imgs = X_train.reshape(-1, C, H * W)
        channel_mean = train_imgs.mean(axis=(0, 2))  # [C]
        channel_std = train_imgs.std(axis=(0, 2))     # [C]
        channel_std = np.maximum(channel_std, 1e-8)   # avoid div-by-zero

        print(f"  Per-channel normalization: mean={channel_mean}, std={channel_std}")

        # Normalize: broadcast (C,) over (N, C, H*W)
        def _normalize_channels(X_flat, mean, std, C, H, W):
            imgs = X_flat.reshape(-1, C, H * W)
            imgs = (imgs - mean[None, :, None]) / std[None, :, None]
            return imgs.reshape(-1, C * H * W)

        X_train = _normalize_channels(X_train, channel_mean, channel_std, C, H, W)
        X_val = _normalize_channels(X_val, channel_mean, channel_std, C, H, W)
        X_test = _normalize_channels(X_test, channel_mean, channel_std, C, H, W)

        # Store channel stats for reference (not a sklearn scaler)
        x_scaler = {"type": "per_channel", "mean": channel_mean, "std": channel_std,
                     "spatial_shape": spatial_shape}
    else:
        # Tabular data: standard z-score per feature
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)

    # Convert to tensors -- y as torch.long, NOT unsqueezed
    data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
        "x_scaler": x_scaler,
        "d_input": X_train.shape[1],
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Train class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return data


def create_dataloaders_classification(data: Dict[str, Any],
                                       batch_size: int) -> Dict[str, DataLoader]:
    """Create DataLoaders for classification (y is [N] long, not [N,1] float)."""
    train_ds = TensorDataset(data["X_train"], data["y_train"])
    val_ds = TensorDataset(data["X_val"], data["y_val"])
    test_ds = TensorDataset(data["X_test"], data["y_test"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


def evaluate_classification_model(model, X_test: torch.Tensor, y_test: torch.Tensor,
                                   device: str, n_classes: int,
                                   batch_size: int = 256) -> Dict[str, Any]:
    """Evaluate classification model on test set.

    Computes: accuracy, macro-F1, weighted-F1, macro precision/recall,
    AUC (OvR for multi-class), confusion matrix.
    Uses batched inference to avoid OOM on large models.
    """
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size].to(device)
            logits = model(batch).cpu()
            all_preds.append(logits.argmax(dim=1))
            all_probs.append(torch.softmax(logits, dim=1))
    preds = torch.cat(all_preds, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()

    y_true = y_test.numpy()

    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    f1_m = f1_score(y_true, preds, average='macro', zero_division=0)
    f1_w = f1_score(y_true, preds, average='weighted', zero_division=0)
    prec = precision_score(y_true, preds, average='macro', zero_division=0)
    rec = recall_score(y_true, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, preds)

    result = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "confusion_matrix": cm.tolist(),
        "n_classes": n_classes,
    }

    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        if n_classes == 2:
            result["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            result["auc_ovr"] = float(roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro",
            ))
    except ValueError:
        pass  # Can fail if a class has zero test samples

    return result


def print_classification_results(name: str, metrics: Dict[str, Any],
                                  arch: Dict[str, Any], elapsed_time: float):
    """Print formatted classification experiment results."""
    print(f"\n{'='*60}")
    print(f"  {name} -- Results")
    print(f"{'='*60}")
    # Standard single-label classification metrics (may be absent for multi-label)
    if 'accuracy' in metrics:
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    if 'balanced_accuracy' in metrics:
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if 'f1_macro' in metrics:
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    if 'f1_weighted' in metrics:
        print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    if 'precision_macro' in metrics:
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    if 'recall_macro' in metrics:
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    # Multi-label / multi-task metrics
    if 'auroc' in metrics:
        print(f"  AUROC:             {metrics['auroc']:.4f}")
    if 'mean_task_auroc' in metrics:
        print(f"  Mean Task AUROC:   {metrics['mean_task_auroc']:.4f}")
    if 'auc_ovr' in metrics:
        print(f"  AUC (OvR macro):   {metrics['auc_ovr']:.4f}")
    if 'n_tasks' in metrics:
        print(f"  Tasks:             {metrics['n_tasks']}")
    print(f"  ---")
    print(f"  Layers:      {arch['num_layers']}")
    print(f"  Parameters:  {arch['total_parameters']}")
    print(f"  Connections: {len(arch.get('connections', []))}")
    widths = [l['out_features'] for l in arch['layers']]
    print(f"  Widths:      {widths}")
    print(f"  Arch Cost:   {arch['architecture_cost']:.0f}")
    print(f"  Time:        {elapsed_time:.1f}s")
    print(f"{'='*60}\n")


def run_classification_experiment_wrapper(name: str, run_fn: Callable,
                                           results_base_dir: Path):
    """Wrapper for classification experiments.

    run_fn signature: run_fn(results_dir: str) -> (metrics, arch, config_dict[, training_info])
    Returns (name, metrics, arch, elapsed_time, status)
    """
    print(f"\n{'#'*60}")
    print(f"  Starting: {name}")
    print(f"{'#'*60}\n")

    results_dir = results_base_dir / name.lower().replace(" ", "_").replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        result = run_fn(str(results_dir))
        # Support both 3-tuple and 4-tuple returns (backward compatible)
        if len(result) == 4:
            metrics, arch, config_dict, training_info = result
        else:
            metrics, arch, config_dict = result
            training_info = None
        elapsed = time.time() - start_time
        print_classification_results(name, metrics, arch, elapsed)
        save_results(str(results_dir), name, metrics, arch, config_dict, elapsed, training_info)
        return name, metrics, arch, elapsed, "OK"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return name, None, None, elapsed, f"FAILED: {e}"


# ==================== Epoch Computation ====================


def compute_max_epochs(desired_steps: int, steps_per_epoch: int, minimum: int = 10) -> int:
    """Convert a desired step count to an epoch count.

    Args:
        desired_steps: Rough total training steps desired.
        steps_per_epoch: Batches per epoch (len(train_loader)).
        minimum: Minimum epochs (default 10).

    Returns:
        Number of epochs (at least `minimum`).
    """
    epochs = max(minimum, desired_steps // max(steps_per_epoch, 1))
    return epochs


# ==================== GPU Batch Augmentation for Image Experiments ====================

def gpu_augment_batch(x_batch: torch.Tensor, spatial_shape, pad: int = 4) -> torch.Tensor:
    """Vectorized augmentation on GPU for a full batch of flat image tensors.

    Works correctly with per-channel normalized data (mean≈0, std≈1) — does NOT
    assume [0,1] range. All operations are scale-invariant or use per-sample stats.

    Applies (all fully vectorized on GPU):
      1. Random horizontal flip (50%)
      2. Random crop with reflect padding (±pad pixels)
      3. Color jitter: contrast and saturation (scale-invariant operations)
      4. Cutout: 16×16 random mask filled with per-channel mean (not 0)

    Args:
        x_batch: [B, D_flat] tensor (already on GPU). May be normalized.
        spatial_shape: (C, H, W) tuple.
        pad: Padding pixels for random crop (default 4).

    Returns:
        [B, D_flat] augmented tensor on same device.
    """
    C, H, W = spatial_shape
    B = x_batch.shape[0]

    # Reshape to spatial: [B, C, H, W]
    imgs = x_batch.view(B, C, H, W)

    # --- 1. Random horizontal flip (per-sample) ---
    flip_mask = torch.rand(B, 1, 1, 1, device=x_batch.device) < 0.5
    imgs = torch.where(flip_mask, imgs.flip(-1), imgs)

    # --- 2. Random crop with reflect padding ---
    padded = F.pad(imgs, [pad] * 4, mode='reflect')
    crop_i = torch.randint(0, 2 * pad + 1, (B,), device=x_batch.device)
    crop_j = torch.randint(0, 2 * pad + 1, (B,), device=x_batch.device)
    row_idx = torch.arange(H, device=x_batch.device).unsqueeze(0) + crop_i.unsqueeze(1)
    col_idx = torch.arange(W, device=x_batch.device).unsqueeze(0) + crop_j.unsqueeze(1)
    batch_idx = torch.arange(B, device=x_batch.device)
    imgs = padded[
        batch_idx[:, None, None, None],
        torch.arange(C, device=x_batch.device)[None, :, None, None],
        row_idx[:, None, :, None],
        col_idx[:, None, None, :],
    ]  # [B, C, H, W]

    # --- 3. Color jitter (contrast + saturation) ---
    # These operations are scale-invariant: they work on normalized data because
    # they scale relative to per-sample means, not absolute pixel values.
    # Brightness jitter is skipped because adding a constant to normalized data
    # shifts the distribution away from what BN expects. Contrast/saturation
    # preserve the mean, which is safe.
    if C >= 3:
        # Contrast: scale deviation from per-channel spatial mean (±20%)
        contrast_range = 0.2
        c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x_batch.device) - 0.5) * 2 * contrast_range
        channel_mean = imgs.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        imgs = channel_mean + c_factor * (imgs - channel_mean)

        # Saturation: blend toward grayscale (±20%)
        saturation_range = 0.2
        s_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x_batch.device) - 0.5) * 2 * saturation_range
        gray = imgs.mean(dim=1, keepdim=True).expand_as(imgs)  # [B, C, H, W]
        imgs = gray + s_factor * (imgs - gray)

    # --- 4. Cutout (random erasing) ---
    # Standard 16×16 for CIFAR (32×32). Fill with per-sample channel mean
    # (≈0 for normalized data), which is information-neutral.
    cutout_size = max(1, H // 2)  # 16×16 for 32×32 images
    cutout_half = cutout_size // 2
    cy = torch.randint(0, H, (B,), device=x_batch.device)
    cx = torch.randint(0, W, (B,), device=x_batch.device)
    yy = torch.arange(H, device=x_batch.device).unsqueeze(0)  # [1, H]
    xx = torch.arange(W, device=x_batch.device).unsqueeze(0)  # [1, W]
    y_mask = (yy - cy.unsqueeze(1)).abs() < cutout_half  # [B, H]
    x_mask = (xx - cx.unsqueeze(1)).abs() < cutout_half  # [B, W]
    cutout_mask = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1)).unsqueeze(1)  # [B, 1, H, W]
    # Fill with per-sample per-channel mean (neutral value for normalized data)
    fill_value = imgs.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    imgs = torch.where(cutout_mask, fill_value.expand_as(imgs), imgs)

    return imgs.reshape(B, -1)


# ==================== Cutout-Only GPU Augmentation ====================

def gpu_cutout_only(x_batch: torch.Tensor, spatial_shape, cutout_size=None) -> torch.Tensor:
    """Apply only Cutout augmentation on GPU.

    Used when the dataset already handles augmentation (AutoAugment, flip, crop)
    at the PIL image level. Cutout is the only thing that's efficient to do on GPU.

    Args:
        x_batch: [B, D_flat] tensor on GPU, already normalized.
        spatial_shape: (C, H, W) tuple.
        cutout_size: Size of cutout patch. None = H // 2 (default).

    Returns:
        [B, D_flat] tensor with cutout applied.
    """
    C, H, W = spatial_shape
    B = x_batch.shape[0]
    imgs = x_batch.view(B, C, H, W)

    cutout_size = cutout_size if cutout_size is not None else max(1, H // 2)
    cutout_half = cutout_size // 2
    cy = torch.randint(0, H, (B,), device=x_batch.device)
    cx = torch.randint(0, W, (B,), device=x_batch.device)
    yy = torch.arange(H, device=x_batch.device).unsqueeze(0)
    xx = torch.arange(W, device=x_batch.device).unsqueeze(0)
    y_mask = (yy - cy.unsqueeze(1)).abs() < cutout_half
    x_mask = (xx - cx.unsqueeze(1)).abs() < cutout_half
    cutout_mask = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1)).unsqueeze(1)
    fill_value = imgs.mean(dim=(2, 3), keepdim=True)
    imgs = torch.where(cutout_mask, fill_value.expand_as(imgs), imgs)

    return imgs.reshape(B, -1)


# ==================== PIL-Based Augmented Datasets ====================

class AutoAugImageDataset(torch.utils.data.Dataset):
    """Dataset that applies torchvision transforms (including AutoAugment) at the
    PIL image level, then normalizes and flattens to [D_flat] for ASANN.

    This is the proper way to do AutoAugment: the learned augmentation policies
    (rotate, shear, equalize, posterize, solarize, etc.) operate on uint8 PIL images,
    not on already-normalized float tensors.

    Pipeline (train, RGB with AutoAugment):
        PIL image → AutoAugment(policy) → RandomCrop(H, pad) →
        RandomHorizontalFlip → ToTensor → Normalize(mean, std) → Flatten to [D]

    Pipeline (train, grayscale without AutoAugment):
        PIL image → RandomAffine(10°, translate=5%, scale=0.9-1.1) →
        RandomCrop(H, pad) → ToTensor → Normalize(mean, std) → Flatten to [D]

    Pipeline (val/test):
        PIL image → ToTensor → Normalize(mean, std) → Flatten to [D]

    Args:
        images: numpy array [N, H, W, C] uint8 (HWC format, C=1 or 3)
        labels: numpy array [N] int64
        channel_mean: per-channel mean [C] (computed from train set, [0,1] range)
        channel_std: per-channel std [C] (computed from train set, [0,1] range)
        img_size: int — image height/width (e.g., 32 for CIFAR, 28 for MNIST, 96 for STL)
        crop_padding: int — padding for RandomCrop (default 4)
        train: whether to apply augmentation (True for train, False for val/test)
        auto_augment_policy: torchvision.transforms.AutoAugmentPolicy or None.
            Use CIFAR10 for CIFAR-10/100/STL-10, SVHN for SVHN, None to skip AutoAugment.
        horizontal_flip: whether to apply random horizontal flip (default True).
            Set to False for digit datasets (SVHN, MNIST) where flipping changes meaning.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 channel_mean, channel_std,
                 img_size: int = 32, crop_padding: int = 4,
                 train: bool = True, auto_augment_policy=None,
                 horizontal_flip: bool = True,
                 affine_degrees: int = 10,
                 extra_transforms=None):
        import torchvision.transforms as T

        self.images = images  # [N, H, W, C] uint8
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.is_grayscale = (images.shape[-1] == 1)

        # Build transform pipeline
        if train:
            transforms_list = []
            if auto_augment_policy is not None:
                # AutoAugment requires RGB PIL images
                transforms_list.append(T.AutoAugment(auto_augment_policy))
            else:
                # For grayscale / datasets without AutoAugment policy:
                # use geometric augmentation (rotation, slight translate/scale)
                transforms_list.append(T.RandomAffine(
                    degrees=affine_degrees, translate=(0.05, 0.05), scale=(0.9, 1.1)))
            # Extra transforms (e.g., ElasticTransform, GaussianBlur)
            if extra_transforms:
                transforms_list.extend(extra_transforms)
            transforms_list.append(T.RandomCrop(img_size, padding=crop_padding))
            if horizontal_flip:
                transforms_list.append(T.RandomHorizontalFlip())
            transforms_list.extend([
                T.ToTensor(),  # [0,255] uint8 → [0,1] float32, HWC → CHW
                T.Normalize(mean=channel_mean.tolist(), std=channel_std.tolist()),
            ])
            self.transform = T.Compose(transforms_list)
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=channel_mean.tolist(), std=channel_std.tolist()),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img_np = self.images[idx]  # [H, W, C] uint8
        # PIL.Image.fromarray: for grayscale [H,W,1] → squeeze to [H,W]
        if self.is_grayscale:
            img = Image.fromarray(img_np.squeeze(-1), mode='L')
        else:
            img = Image.fromarray(img_np)
        x = self.transform(img)  # [C, H, W] float32, normalized
        x = x.reshape(-1)        # Flatten to [D] for ASANN
        return x, self.labels[idx]


def create_image_augmented_loaders(
    X_train_raw: np.ndarray, y_train: np.ndarray,
    X_val_raw: np.ndarray, y_val: np.ndarray,
    X_test_raw: np.ndarray, y_test: np.ndarray,
    channel_mean: np.ndarray, channel_std: np.ndarray,
    img_size: int = 32, crop_padding: int = 4,
    batch_size: int = 256, num_workers: int = 4,
    auto_augment_policy=None,
    horizontal_flip: bool = True,
    affine_degrees: int = 10,
    extra_transforms=None,
) -> Dict[str, DataLoader]:
    """Create DataLoaders with proper augmentation for any image dataset.

    Images must be in [N, H, W, C] uint8 format.
    Normalization stats should be in [0,1] range (computed from raw /255 data).

    Args:
        X_train_raw, y_train, etc.: raw uint8 HWC images and labels per split.
        channel_mean, channel_std: per-channel stats in [0,1] range.
        img_size: image height/width (assumes square images).
        crop_padding: padding for RandomCrop.
        batch_size: batch size for DataLoaders.
        num_workers: number of DataLoader workers.
        auto_augment_policy: AutoAugmentPolicy (CIFAR10, SVHN, IMAGENET) or None.
        horizontal_flip: whether to apply random horizontal flip.
        affine_degrees: rotation range in degrees for RandomAffine (default 10).
        extra_transforms: list of additional torchvision transforms to apply
            after affine but before crop (e.g., ElasticTransform).

    GPU-side cutout is applied separately in the trainer (fast on GPU).
    """
    train_ds = AutoAugImageDataset(
        X_train_raw, y_train, channel_mean, channel_std,
        img_size=img_size, crop_padding=crop_padding,
        train=True, auto_augment_policy=auto_augment_policy,
        horizontal_flip=horizontal_flip,
        affine_degrees=affine_degrees,
        extra_transforms=extra_transforms)
    val_ds = AutoAugImageDataset(
        X_val_raw, y_val, channel_mean, channel_std,
        img_size=img_size, crop_padding=crop_padding,
        train=False)
    test_ds = AutoAugImageDataset(
        X_test_raw, y_test, channel_mean, channel_std,
        img_size=img_size, crop_padding=crop_padding,
        train=False)

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True,
        }

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **worker_kwargs),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **worker_kwargs),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **worker_kwargs),
    }


# Backward-compatible alias
def create_cifar10_augmented_loaders(
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
    channel_mean, channel_std, batch_size=256, num_workers=4, auto_augment=True,
):
    """Backward-compatible wrapper for CIFAR-10. Calls create_image_augmented_loaders."""
    import torchvision.transforms as T
    policy = T.AutoAugmentPolicy.CIFAR10 if auto_augment else None
    return create_image_augmented_loaders(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
        channel_mean, channel_std, img_size=32, crop_padding=4,
        batch_size=batch_size, num_workers=num_workers,
        auto_augment_policy=policy, horizontal_flip=True,
    )


# ==================== Augmented Dataset (Legacy, kept for reference) ====================

class AugmentedTensorDataset(torch.utils.data.Dataset):
    """Per-sample CPU augmentation for flat image tensors.

    DEPRECATED: Use gpu_augment_batch() or AutoAugImageDataset instead.
    Kept for backward compatibility with non-GPU workflows.
    """

    def __init__(self, X, y, spatial_shape, augment=True):
        self.X, self.y = X, y
        self.C, self.H, self.W = spatial_shape
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            img = x.view(self.C, self.H, self.W)
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
            # Random crop with reflect padding
            pad = 4
            padded = F.pad(img, [pad] * 4, mode='reflect')
            i = torch.randint(0, 2 * pad + 1, (1,)).item()
            j = torch.randint(0, 2 * pad + 1, (1,)).item()
            img = padded[:, i:i + self.H, j:j + self.W]
            x = img.reshape(-1)
        return x, y


def create_dataloaders_classification_augmented(data: Dict[str, Any],
                                                 batch_size: int,
                                                 spatial_shape=None,
                                                 num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create DataLoaders for image classification.

    When spatial_shape is provided, uses plain TensorDataset (no per-sample
    augmentation). GPU augmentation is applied in the trainer via
    gpu_augment_batch() instead.

    Args:
        data: Dict with X_train, y_train, X_val, y_val, X_test, y_test.
        batch_size: Batch size for all loaders.
        spatial_shape: (C, H, W) tuple — passed through for reference but
            augmentation is now handled on GPU in the trainer.
        num_workers: Number of DataLoader workers (default 4).

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    # Always use plain TensorDataset — augmentation moved to GPU in trainer
    train_ds = TensorDataset(data["X_train"], data["y_train"])
    val_ds = TensorDataset(data["X_val"], data["y_val"])
    test_ds = TensorDataset(data["X_test"], data["y_test"])

    # Use multi-worker loading with pinned memory for GPU transfer
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = {
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True,
        }

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **worker_kwargs),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **worker_kwargs),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **worker_kwargs),
    }


# ==================== Tier 4: Time-Series / Sequence Support ====================


def split_temporal(data: np.ndarray, targets: np.ndarray,
                   train_ratio: float = 0.7, val_ratio: float = 0.15,
                   normalize: bool = True) -> Dict[str, Any]:
    """Split time-series data chronologically (NO shuffling — preserves order).

    Unlike split_and_standardize(), this respects temporal ordering.
    First train_ratio points for train, next val_ratio for val, rest for test.

    Args:
        data: [N, d_input] feature array (already in sliding-window form).
        targets: [N, d_output] target array.
        train_ratio: Fraction for training (default 0.7).
        val_ratio: Fraction for validation (default 0.15).
        normalize: Whether to z-score normalize features and targets.

    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test, d_input,
        plus x_scaler and y_scaler if normalize=True.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = data[:train_end].astype(np.float32)
    X_val = data[train_end:val_end].astype(np.float32)
    X_test = data[val_end:].astype(np.float32)
    y_train = targets[:train_end].astype(np.float32)
    y_val = targets[train_end:val_end].astype(np.float32)
    y_test = targets[val_end:].astype(np.float32)

    x_scaler = None
    y_scaler = None

    if normalize:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
        y_train = y_scaler.fit_transform(y_train_2d).astype(np.float32)
        y_val = y_scaler.transform(y_val_2d).astype(np.float32)
        y_test = y_scaler.transform(y_test_2d).astype(np.float32)
        if targets.ndim == 1:
            y_train = y_train.ravel()
            y_val = y_val.ravel()
            y_test = y_test.ravel()

    result = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "y_train_original": torch.tensor(
            targets[:train_end].astype(np.float32), dtype=torch.float32),
        "y_test_original": torch.tensor(
            targets[val_end:].astype(np.float32), dtype=torch.float32),
        "d_input": X_train.shape[1],
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    print(f"  Temporal split: train={result['n_train']}, "
          f"val={result['n_val']}, test={result['n_test']}")
    return result


def create_sliding_windows(series: np.ndarray, window_size: int,
                           horizon: int = 1,
                           target_cols: Optional[List[int]] = None,
                           ) -> tuple:
    """Create sliding window samples from a time series.

    Args:
        series: [T, F] array (T timesteps, F features) or [T] for univariate.
        window_size: Number of past time steps as input.
        horizon: Number of future time steps to predict.
        target_cols: Which columns to predict (None = all). Ignored for univariate.

    Returns:
        X: [N, window_size * F] flattened input windows.
        y: [N, horizon * n_targets] flattened targets.
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    T, F = series.shape
    if target_cols is None:
        target_cols = list(range(F))
    n_targets = len(target_cols)

    N = T - window_size - horizon + 1
    if N <= 0:
        raise ValueError(f"Series too short ({T}) for window={window_size}, horizon={horizon}")

    X = np.empty((N, window_size * F), dtype=np.float32)
    y = np.empty((N, horizon * n_targets), dtype=np.float32)
    for i in range(N):
        X[i] = series[i:i + window_size].flatten()
        y[i] = series[i + window_size:i + window_size + horizon, target_cols].flatten()

    return X, y


def create_forecasting_dataloaders(split_data: Dict[str, Any],
                                   batch_size: int) -> Dict[str, DataLoader]:
    """Create DataLoaders for forecasting tasks.

    Handles y dimension properly for MSELoss compatibility.
    """
    def _to_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        return torch.tensor(arr, dtype=torch.float32)

    def _ensure_2d(t):
        return t.unsqueeze(1) if t.ndim == 1 else t

    X_train = _to_tensor(split_data["X_train"])
    y_train = _ensure_2d(_to_tensor(split_data["y_train"]))
    X_val = _to_tensor(split_data["X_val"])
    y_val = _ensure_2d(_to_tensor(split_data["y_val"]))
    X_test = _to_tensor(split_data["X_test"])
    y_test = _ensure_2d(_to_tensor(split_data["y_test"]))

    return {
        "train": DataLoader(TensorDataset(X_train, y_train),
                            batch_size=batch_size, shuffle=True),
        "val": DataLoader(TensorDataset(X_val, y_val),
                          batch_size=batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(X_test, y_test),
                           batch_size=batch_size, shuffle=False),
    }


def evaluate_forecasting_model(model, X_test: torch.Tensor,
                               y_test_original: torch.Tensor,
                               y_scaler, device: str,
                               y_test_scaled: Optional[torch.Tensor] = None,
                               y_train_original: Optional[torch.Tensor] = None,
                               ) -> Dict[str, float]:
    """Evaluate forecasting model on test set with both original and standardized metrics.

    Args:
        model: Trained model.
        X_test: Test inputs (standardized).
        y_test_original: Test targets on ORIGINAL scale (not standardized).
        y_scaler: sklearn StandardScaler fit on training targets (or None).
        device: 'cuda' or 'cpu'.
        y_test_scaled: Test targets on STANDARDIZED scale (optional).
            If provided, also computes mse_scaled/mae_scaled for benchmark comparison.
        y_train_original: Training targets on ORIGINAL scale (optional).
            If provided, computes MASE (Mean Absolute Scaled Error).

    Returns:
        Dict with mse, mae, rmse, mape on original scale,
        plus mse_scaled, mae_scaled on standardized scale (if y_test_scaled given),
        plus mase (if y_train_original given).
    """
    model.eval()
    with torch.no_grad():
        X = X_test.to(device)
        preds_scaled = model(X).cpu().numpy()

    # Inverse-transform predictions to original scale
    if y_scaler is not None:
        preds_original = y_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1) if preds_scaled.ndim == 1 else preds_scaled
        )
    else:
        preds_original = preds_scaled

    y_true = y_test_original.numpy()
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if preds_original.ndim == 1:
        preds_original = preds_original.reshape(-1, 1)

    mse = mean_squared_error(y_true, preds_original)
    mae = mean_absolute_error(y_true, preds_original)
    rmse = np.sqrt(mse)
    # MAPE: protected against near-zero values
    mask = np.abs(y_true) > 1e-8
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - preds_original[mask]) / y_true[mask])) * 100)
    else:
        mape = float("inf")

    result = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }

    # Standardized-scale metrics (for benchmark comparison with DLinear, PatchTST, etc.)
    if y_test_scaled is not None:
        y_scaled = y_test_scaled.numpy()
        p_scaled = preds_scaled
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        if p_scaled.ndim == 1:
            p_scaled = p_scaled.reshape(-1, 1)
        result["mse_scaled"] = float(mean_squared_error(y_scaled, p_scaled))
        result["mae_scaled"] = float(mean_absolute_error(y_scaled, p_scaled))

    # MASE: Mean Absolute Scaled Error
    # MASE = MAE / naive_MAE, where naive_MAE is the in-sample mean absolute
    # difference between consecutive values (naive "repeat last" forecast).
    if y_train_original is not None:
        y_tr = y_train_original.numpy().ravel()
        if len(y_tr) > 1:
            naive_mae = float(np.mean(np.abs(np.diff(y_tr))))
            if naive_mae > 1e-10:
                result["mase"] = float(mae / naive_mae)
            else:
                result["mase"] = float("inf")

    return result


def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss: PPL = exp(avg_loss)."""
    return float(np.exp(min(avg_cross_entropy_loss, 100.0)))  # clamp to prevent overflow


def print_forecasting_results(name: str, metrics: Dict[str, float],
                              arch: Dict[str, Any], elapsed_time: float):
    """Print formatted forecasting experiment results."""
    print(f"\n{'='*60}")
    print(f"  {name} - Results")
    print(f"{'='*60}")
    if 'mse' in metrics:
        print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    if 'mape' in metrics:
        print(f"  MAPE: {metrics['mape']:.2f}%")
    if 'mase' in metrics:
        print(f"  MASE: {metrics['mase']:.4f}")
    if 'mse_scaled' in metrics:
        print(f"  MSE (standardized): {metrics['mse_scaled']:.6f}")
        print(f"  MAE (standardized): {metrics['mae_scaled']:.6f}")
    if 'perplexity' in metrics:
        print(f"  PPL:  {metrics['perplexity']:.2f}")
    print(f"  ---")
    print(f"  Layers:      {arch['num_layers']}")
    print(f"  Parameters:  {arch['total_parameters']}")
    print(f"  Connections: {len(arch.get('connections', []))}")
    widths = [l['out_features'] for l in arch['layers']]
    print(f"  Widths:      {widths}")
    print(f"  Arch Cost:   {arch['architecture_cost']:.0f}")
    print(f"  Time:        {elapsed_time:.1f}s")
    print(f"{'='*60}\n")


def run_forecasting_experiment_wrapper(name: str, run_fn: Callable,
                                       results_base_dir: Path):
    """Wrapper for forecasting experiments with timing, error catching, results saving.

    run_fn signature: run_fn(results_dir: str) -> (metrics, arch, config_dict[, training_info])
    Returns (name, metrics, arch, elapsed_time, status)
    """
    print(f"\n{'#'*60}")
    print(f"  Starting: {name}")
    print(f"{'#'*60}\n")

    results_dir = results_base_dir / name.lower().replace(" ", "_").replace("/", "_")
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        result = run_fn(str(results_dir))
        if len(result) == 4:
            metrics, arch, config_dict, training_info = result
        else:
            metrics, arch, config_dict = result
            training_info = None
        elapsed = time.time() - start_time
        print_forecasting_results(name, metrics, arch, elapsed)
        save_results(str(results_dir), name, metrics, arch, config_dict, elapsed, training_info)
        return name, metrics, arch, elapsed, "OK"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR in {name}: {e}")
        import traceback
        traceback.print_exc()
        return name, None, None, elapsed, f"FAILED: {e}"
