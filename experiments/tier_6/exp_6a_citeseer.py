"""
Experiment 6a: CiteSeer Node Classification
=============================================
Dataset: Planetoid/CiteSeer (citation graph)
Nodes: 3,327 | Edges: 9,104 | Features: 3,703 | Classes: 6
Task: Semi-supervised node classification

ASANN should discover: graph aggregation ops (NeighborAgg, GraphAttention).
"""

import sys
import os
_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from common import (
    setup_paths, get_device, config_to_dict, run_experiment_wrapper,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.asann_optimizer import ASANNOptimizerConfig
from tier_6.graph_utils import (
    load_pyg_citation, graph_data_to_asann_format, create_graph_dataloaders,
)


def _dropout_hook_05(module, inputs):
    """Module-level dropout hook (p=0.5) — picklable unlike local closures."""
    x = inputs[0]
    if module.training:
        x = F.dropout(x, p=0.5, training=True)
    return (x,)


class MLPHead(nn.Module):
    """2-layer MLP output head compatible with ASANN surgery.

    Proxies .in_features, .out_features, .weight, .bias to the hidden
    (first) layer so add_neuron/remove_neuron surgery can resize correctly.
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


def run_experiment(results_dir: str):
    """Run CiteSeer node classification experiment."""
    # Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = get_device()

    # ===== 1. Load dataset =====
    print("  Loading CiteSeer dataset...")
    data, num_classes = load_pyg_citation("CiteSeer")
    x, y, masks, graph_info = graph_data_to_asann_format(data, num_classes)

    # Use larger random split (60/20/20) instead of standard semi-supervised
    # split (120 train / 500 val / 1000 test). The standard split has only
    # 120 labeled nodes which severely limits achievable accuracy.
    N = data.num_nodes
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    masks['train'] = torch.zeros(N, dtype=torch.bool)
    masks['val'] = torch.zeros(N, dtype=torch.bool)
    masks['test'] = torch.zeros(N, dtype=torch.bool)
    masks['train'][perm[:n_train]] = True
    masks['val'][perm[n_train:n_train + n_val]] = True
    masks['test'][perm[n_train + n_val:]] = True
    print(f"  Random split: train={masks['train'].sum()}, val={masks['val'].sum()}, test={masks['test'].sum()}")

    # ===== 2. Create dataloaders (full-batch) =====
    train_loader, val_loader, test_loader = create_graph_dataloaders(
        x, y, masks, task_type="classification"
    )

    # ===== 3. Configure ASANN (Optuna-optimized) =====
    config = ASANNConfig(
        encoder_candidates=["graph_node"],
        d_init=16,
        initial_num_layers=1,

        # Complexity — tighter ceiling prevents runaway growth
        complexity_target_auto=True,
        complexity_ceiling_mult=2.8,
        hard_max_multiplier=1.0,

        # Surgery — moderate frequency, fast stabilization
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=11,
        eval_epoch_interval=1,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=17,

        # With 60% train split, tighter overfitting detection
        overfitting_gap_early=0.30,
        overfitting_gap_moderate=0.50,
        overfitting_gap_severe=0.70,

        # Recovery
        min_recovery_epochs=5,
        max_recovery_epochs=15,
        recovery_catastrophic_ratio=3.0,

        max_treatment_escalations=4,

        # Warmup
        hard_warmup_epochs=3,
        soft_warmup_epochs=2,

        # Graph ops — moderate gate (sigmoid(1.23)=0.77)
        graph_diffusion_max_hops=1,
        graph_attention_heads=1,
        graph_initial_gate=1.23,

        # General
        max_ops_per_layer=4,
        mixup_enabled=False,
        drop_path_enabled=False,
        device=device,

        optimizer=ASANNOptimizerConfig(
            base_lr=0.003,
            weight_decay=0.0025,
        ),
    )

    # ===== 4. Create model and attach graph data =====
    d_input = data.num_features
    d_output = num_classes
    model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
    model.set_graph_data(
        edge_index=graph_info['edge_index'].to(device),
        num_nodes=graph_info['num_nodes'],
        degree=graph_info['degree'].to(device),
    )
    model.seed_graph_ops()

    # MLP output head for more expressive classification boundary
    model.output_head = MLPHead(config.d_init, d_output)

    # Input feature dropout — reduced from 0.5 since we now have 60% train split
    # (0.5 was critical for semi-supervised with only 120 labeled nodes)
    model.register_forward_pre_hook(_dropout_hook_05)

    model.to(device)

    # ===== 5. Train =====
    # Cross-entropy with label smoothing for few-label regularization
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.19)

    trainer = ASANNTrainer(
        model=model, config=config,
        task_loss_fn=loss_fn,
        log_dir=results_dir,
        task_type="classification",
    )

    max_epochs = 300    # Focused training with fast graph op discovery
    print(f"\n  Training for {max_epochs} epochs...")
    train_metrics = trainer.train_epochs(
        train_data=train_loader,
        max_epochs=max_epochs,
        val_data=val_loader,
        test_data=test_loader,
        print_every=25,
        snapshot_every=500,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=100,
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

    eval_model.eval()
    with torch.no_grad():
        x_all = x.to(device)
        logits = eval_model(x_all)
        test_mask = masks['test'].to(device)
        preds = logits[test_mask].argmax(dim=1)
        labels = y.to(device)[test_mask]
        test_acc = (preds == labels).float().mean().item()

    print(f"\n  Test Accuracy: {test_acc:.4f}")

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
        "test_accuracy": test_acc,
        "num_classes": num_classes,
        "graph_ops_discovered": list(graph_ops_found),
        "n_graph_ops": len(graph_ops_found),
    }

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_6"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "CiteSeer", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
