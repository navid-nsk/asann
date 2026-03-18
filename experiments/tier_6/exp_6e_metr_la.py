"""
Experiment 6e: METR-LA Traffic Forecasting
============================================
Dataset: METR-LA (Los Angeles traffic speed sensors)
Sensors: 207 | Time-series regression
Task: Predict next 12 time steps (1 hour) from past 12 steps per sensor

Graph structure: DCRNN benchmark adjacency (Gaussian kernel of road distances).
Each forward pass receives a full graph snapshot: [207, window] -> [207, horizon].
Graph ops use sparse matmul ([N,N] @ [N,d]) so each snapshot = 1 forward pass.

ASANN should discover: graph ops to leverage spatial sensor correlations.
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
import torch.nn as nn
from pathlib import Path

from common import (
    setup_paths, get_device, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from tier_6.graph_utils import (
    load_traffic_preprocessed, traffic_to_asann_format, create_traffic_dataloaders,
    TrafficScaler,
)


class MLPHead(nn.Module):
    """2-layer MLP output head compatible with ASANN surgery.

    Proxies .in_features, .out_features, .weight, .bias to the hidden
    (first) layer via __getattr__/__setattr__ so add_neuron/remove_neuron
    surgery can resize the input dimension correctly. Uses PyTorch-compatible
    attribute routing (not properties, which conflict with nn.Parameter assignment).
    """

    _PROXY = frozenset({'weight', 'bias', 'in_features', 'out_features'})

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.hidden = nn.Linear(d_in, d_in)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_in, d_out)

    def __getattr__(self, name: str):
        if name in MLPHead._PROXY:
            return getattr(self._modules['hidden'], name)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        if name in MLPHead._PROXY and '_modules' in self.__dict__ and 'hidden' in self._modules:
            setattr(self._modules['hidden'], name, value)
        else:
            super().__setattr__(name, value)

    def forward(self, x):
        return self.proj(self.act(self.hidden(x)))


def autoregressive_predict(model, x, step_size, total_horizon, d_speed=12):
    """Chain predictions: predict step_size steps at a time, shift input, repeat.

    Args:
        model: Trained ASANN model (d_output = step_size)
        x: Input tensor [B*N, d_input] where d_input = d_speed + d_time_features
        step_size: Steps per prediction (e.g. 3)
        total_horizon: Total steps to predict (e.g. 12)
        d_speed: Number of speed features in input (before time features)

    Returns:
        [B*N, total_horizon] predictions in standardized space
    """
    preds = []
    current_x = x.clone()
    n_iter = total_horizon // step_size

    for i in range(n_iter):
        pred_step = model(current_x)  # [B*N, step_size]
        preds.append(pred_step)

        if i < n_iter - 1:
            # Shift speed window: drop first step_size, append predictions
            speed_part = current_x[:, :d_speed]
            time_part = current_x[:, d_speed:]  # time features (keep constant)
            new_speed = torch.cat([speed_part[:, step_size:], pred_step], dim=1)
            current_x = torch.cat([new_speed, time_part], dim=1)

    return torch.cat(preds, dim=1)  # [B*N, total_horizon]


def run_experiment(results_dir: str):
    """Run METR-LA traffic forecasting experiment."""
    device = get_device()

    window = 12
    horizon = 12
    ar_step_size = 3  # Auto-regressive: predict 3 steps, chain 4 times

    # ===== 1. Load dataset =====
    print("  Loading METR-LA dataset...")
    X_all, Y_all, adj = load_traffic_preprocessed("metr-la", window=window, horizon=horizon)
    split_data, graph_info = traffic_to_asann_format(
        X_all, Y_all, adj, add_time_features=True, steps_per_day=288,
        ar_step_size=ar_step_size,
    )

    # ===== 2. Create graph-snapshot dataloaders =====
    # Batched graph processing: [B*N, d] input, graph ops reshape to [N, B*d] for
    # sparse matmul internally. batch_size=32 gives ~32x GPU utilization improvement.
    train_loader, val_loader, test_loader = create_traffic_dataloaders(
        split_data,
        batch_size=32,
        subsample_train=2000,
        subsample_val=500,
        subsample_test=1000,
    )

    # ===== 3. Configure ASANN =====
    d_input = split_data['d_input']    # 16 (12 window + 4 time features)
    d_output = split_data['d_output']  # 3 (ar_step_size) — predicts 3 steps at a time

    config = ASANNConfig.from_task(
        task_type="regression",
        modality="temporal_graph",
        d_input=d_input,
        d_output=d_output,
        n_samples=len(split_data['X_train']),
        device=device,
    )

    # ===== 4. Create model and attach graph data =====
    loss_fn = torch.nn.L1Loss()  # MAE loss — standard for traffic forecasting (DCRNN, D2STGNN)
    y_scaler = TrafficScaler(split_data['y_mean'], split_data['y_std'])

    def _setup_graph(m):
        """Attach graph data and MLP head to model."""
        m.set_graph_data(
            edge_index=graph_info['edge_index'].to(device),
            num_nodes=graph_info['num_nodes'],
            degree=graph_info['degree'].to(device),
        )
        m.seed_graph_ops()
        # MLP output head (replaces default nn.Linear) for better expressiveness
        if not isinstance(m.output_head, MLPHead):
            m.output_head = MLPHead(config.d_init, d_output)
        m.to(device)

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        _setup_graph(model)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir,
            task_type="regression",
            y_scaler=y_scaler,
            target_metric="mae",
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="regression",
        y_scaler=y_scaler,
        target_metric="mae",
    )
    model = trainer.model
    _setup_graph(model)

    # ===== 5. Train =====
    max_epochs = config.recommended_max_epochs
    print(f"\n  Training for {max_epochs} epochs{'  (resumed)' if is_resumed else ''}...")
    train_metrics = trainer.train_epochs(
        train_data=train_loader,
        max_epochs=max_epochs,
        val_data=val_loader,
        test_data=test_loader,
        print_every=200,
        snapshot_every=500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=20,
    )

    # ===== 6. Evaluate on test set (using best model) =====
    # After train_epochs(), trainer.model IS the best model (restored by _restore_best_model).
    # The local `model` variable still points to the FINAL model (stale reference).
    eval_model = trainer.model
    eval_model.to(device)
    eval_model.set_graph_data(
        edge_index=graph_info['edge_index'].to(device),
        num_nodes=graph_info['num_nodes'],
        degree=graph_info['degree'].to(device),
    )
    print("  Using best model (trainer.model after restore) for test evaluation")

    # Build full-horizon test loader for auto-regressive evaluation
    # We need the full 12-step targets, not the truncated 3-step targets
    total_horizon = split_data['total_horizon']

    if ar_step_size > 0 and 'y_test_full' in split_data:
        from tier_6.graph_utils import GraphSnapshotDataset, _snapshot_collate
        from torch.utils.data import DataLoader
        full_test_ds = GraphSnapshotDataset(
            split_data['X_test'][:1000], split_data['y_test_full'][:1000]
        )
        full_test_loader = DataLoader(
            full_test_ds, batch_size=32, shuffle=False,
            collate_fn=_snapshot_collate,
        )
    else:
        full_test_loader = test_loader

    eval_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in full_test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if ar_step_size > 0 and ar_step_size < total_horizon:
                # Auto-regressive: chain predictions
                pred = autoregressive_predict(
                    eval_model, batch_x, ar_step_size, total_horizon, d_speed=window
                )
            else:
                pred = eval_model(batch_x)

            all_preds.append(pred)
            all_labels.append(batch_y)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # MSE on standardized predictions
    test_mse = torch.nn.functional.mse_loss(preds, labels).item()

    # R-squared
    ss_res = ((labels - preds) ** 2).sum().item()
    ss_tot = ((labels - labels.mean()) ** 2).sum().item()
    test_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE in original scale (de-standardize)
    # Use full-horizon stats for AR eval (preds are [B*N, 12]), truncated stats otherwise
    if ar_step_size > 0 and 'y_std_full' in split_data:
        y_std = split_data['y_std_full'].to(device)
        y_mean = split_data['y_mean_full'].to(device)
    else:
        y_std = split_data['y_std'].to(device)
        y_mean = split_data['y_mean'].to(device)
    preds_orig = preds * y_std + y_mean
    labels_orig = labels * y_std + y_mean
    test_rmse_orig = torch.sqrt(torch.nn.functional.mse_loss(preds_orig, labels_orig)).item()
    test_mae = torch.nn.functional.l1_loss(preds, labels).item()
    test_mae_orig = torch.nn.functional.l1_loss(preds_orig, labels_orig).item()
    test_rmse = float(test_mse ** 0.5)

    print(f"\n  Test MSE (standardized): {test_mse:.6f}")
    print(f"  Test R-squared: {test_r2:.4f}")
    print(f"  Test MAE (original scale): {test_mae_orig:.4f}")
    print(f"  Test RMSE (original scale): {test_rmse_orig:.4f}")

    # ===== 7. Collect metrics =====
    arch = eval_model.describe_architecture()

    # Count graph ops discovered
    from asann.surgery import get_operation_name
    graph_ops_found = set()
    for l in range(eval_model.num_layers):
        for op in eval_model.ops[l].operations:
            name = get_operation_name(op)
            if 'graph_' in name:
                graph_ops_found.add(name)
    print(f"  Graph ops discovered: {graph_ops_found or 'none'}")

    metrics = {
        "mse": test_mse,
        "rmse": test_rmse,
        "mae": test_mae,
        "r2": test_r2,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "test_rmse_original_scale": test_rmse_orig,
        "test_mae_original_scale": test_mae_orig,
        "n_sensors": split_data['N_nodes'],
        "window": window,
        "horizon": horizon,
        "graph_ops_discovered": list(graph_ops_found),
        "n_graph_ops": len(graph_ops_found),
    }

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_6"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "METR-LA", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
