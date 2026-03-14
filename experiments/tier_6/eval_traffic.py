"""
Standalone evaluation script for traffic experiments (METR-LA / PEMS-BAY).

Loads the best_model_full.pt (full pickle) or reconstructs from best_model.pt,
evaluates on the full test set, and reports raw-scale metrics (MAE, RMSE, MAPE).

Usage:
    python experiments/tier_6/eval_traffic.py --dataset metr-la
    python experiments/tier_6/eval_traffic.py --dataset pems-bay
"""

import sys
import os
import argparse

_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np
from pathlib import Path

from common import setup_paths
from tier_6.graph_utils import (
    load_traffic_preprocessed, traffic_to_asann_format, create_traffic_dataloaders,
)


def evaluate(dataset: str):
    """Evaluate best model on traffic dataset and report raw-scale metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    window = 12
    horizon = 12

    # ===== 1. Locate results =====
    project_root, results_base = setup_paths()
    results_dir = str(results_base / "tier_6" / dataset)

    full_model_path = os.path.join(results_dir, "best_model_full.pt")
    state_dict_path = os.path.join(results_dir, "best_model.pt")

    if not os.path.exists(full_model_path) and not os.path.exists(state_dict_path):
        print(f"ERROR: No model found in {results_dir}")
        print(f"  Expected: {full_model_path} or {state_dict_path}")
        return

    # ===== 2. Load dataset =====
    print(f"\n  Loading {dataset.upper()} dataset...")
    X_all, Y_all, adj = load_traffic_preprocessed(dataset, window=window, horizon=horizon)
    split_data, graph_info = traffic_to_asann_format(X_all, Y_all, adj)

    _, _, test_loader = create_traffic_dataloaders(
        split_data,
        subsample_train=0,
        subsample_val=0,
        subsample_test=0,  # Full test set
    )

    y_std = split_data['y_std'].to(device)
    y_mean = split_data['y_mean'].to(device)

    # ===== 3. Load model =====
    if os.path.exists(full_model_path):
        print(f"  Loading full model pickle from {full_model_path}")
        model = torch.load(full_model_path, map_location=device, weights_only=False)
    else:
        print(f"  WARNING: No full model pickle found ({full_model_path})")
        print(f"  Cannot load from state_dict alone (architecture unknown).")
        print(f"  Re-run the experiment with the fixed code to generate best_model_full.pt")
        return

    model.to(device)
    model.set_graph_data(
        edge_index=graph_info['edge_index'].to(device),
        num_nodes=graph_info['num_nodes'],
        degree=graph_info['degree'].to(device),
    )
    model.eval()

    # ===== 4. Evaluate =====
    print(f"  Evaluating on test set ({len(test_loader)} batches)...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            all_preds.append(pred)
            all_labels.append(batch_y)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Normalized metrics
    test_mse = torch.nn.functional.mse_loss(preds, labels).item()
    test_mae_norm = torch.nn.functional.l1_loss(preds, labels).item()
    ss_res = ((labels - preds) ** 2).sum().item()
    ss_tot = ((labels - labels.mean()) ** 2).sum().item()
    test_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Raw-scale metrics (de-standardize)
    preds_orig = preds * y_std + y_mean
    labels_orig = labels * y_std + y_mean

    test_mae_raw = torch.nn.functional.l1_loss(preds_orig, labels_orig).item()
    test_rmse_raw = torch.sqrt(torch.nn.functional.mse_loss(preds_orig, labels_orig)).item()

    # MAPE (protect against zero)
    mask = labels_orig.abs() > 1e-8
    if mask.any():
        mape = (((preds_orig - labels_orig).abs() / labels_orig.abs())[mask]).mean().item()
    else:
        mape = float("nan")

    # ===== 5. Report =====
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} — Best Model Test Results")
    print(f"{'='*60}")
    print(f"  MAE  (raw scale):  {test_mae_raw:.4f}")
    print(f"  RMSE (raw scale):  {test_rmse_raw:.4f}")
    print(f"  MAPE:              {mape:.4f}")
    print(f"  R²:                {test_r2:.4f}")
    print(f"  MAE  (normalized): {test_mae_norm:.4f}")
    print(f"  RMSE (normalized): {test_mse**0.5:.4f}")
    print(f"{'='*60}")

    # Architecture info
    arch = model.describe_architecture()
    print(f"  Architecture: {arch['num_layers']} layers, {arch['total_parameters']} params")
    print(f"  Widths: {[l['out_features'] for l in arch['layers']]}")
    for i, layer in enumerate(arch["layers"]):
        print(f"  Layer {i}: {layer.get('ops', 'N/A')}")

    return {
        "mae_raw": test_mae_raw,
        "rmse_raw": test_rmse_raw,
        "mape": mape,
        "r2": test_r2,
        "mae_norm": test_mae_norm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traffic best model")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["metr-la", "pems-bay"],
                        help="Dataset to evaluate")
    args = parser.parse_args()
    evaluate(args.dataset)
