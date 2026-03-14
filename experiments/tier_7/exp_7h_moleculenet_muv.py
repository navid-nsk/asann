"""
Experiment 7H: MoleculeNet MUV -- Maximum Unbiased Validation (17-task)
=======================================================================
Dataset: MUV from MoleculeNet (full 93K molecules, 17 binary tasks)
Features: 2048 (Morgan fingerprints from SMILES)
Tasks: 17 binary classification targets (MUV-466 through MUV-859)
       Each task: ~30 positives, ~14,700 negatives, ~78K missing labels

Standard MoleculeNet benchmark. Supports scaffold and random splitting.
Masked BCEWithLogitsLoss ignores NaN labels during training.
Global AUROC computed by concatenating predictions across all tasks (matching
KA-GNN evaluation protocol).

Uses mini-batch training with MolecularGraphEncoder support.
93K samples at batch_size=128 gives ~582 steps/epoch.

Usage:
  python exp_7h_moleculenet_muv.py [scaffold|random]
  MUV_SPLIT=random python exp_7h_moleculenet_muv.py

Requires: pip install rdkit torch_geometric
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
import torch.nn as nn
import numpy as np
from pathlib import Path

from common import (
    setup_paths, get_device,
    create_mol_dataloaders_classification,
    config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.asann_optimizer import ASANNOptimizerConfig


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE loss that ignores NaN labels (missing task annotations).

    For multi-task MUV where each molecule only has labels for ~2/17 tasks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # targets: [B, T] float32 with NaN for missing
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        # Replace NaN with 0 for computation (masked out anyway)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        per_element = nn.functional.binary_cross_entropy_with_logits(
            logits, safe_targets, reduction='none'
        )
        return (per_element * mask.float()).sum() / mask.float().sum()


def run_experiment(results_dir: str):
    """Run MoleculeNet MUV 17-task multi-label experiment."""
    try:
        from tier_7.bio_utils import (
            load_moleculenet_muv_multitask, smiles_to_molecular_graphs,
        )
    except ImportError:
        raise RuntimeError("SKIPPED: tier_7.bio_utils not importable")

    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("SKIPPED: rdkit not installed. Install with: pip install rdkit")

    try:
        from torch_geometric.data import Batch as PyGBatch
    except ImportError:
        raise RuntimeError(
            "SKIPPED: torch_geometric not installed. Install with: pip install torch_geometric"
        )

    device = get_device()

    # ===== 1. Load full MUV dataset (93K molecules, 17 tasks) =====
    csv_path = os.path.join(_proj_root, "data", "muv.csv", "muv.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"MUV CSV not found at {csv_path}")

    print("  Loading full MUV dataset (17 tasks)...")
    X, y_multi, task_names, smiles_list, valid_indices = \
        load_moleculenet_muv_multitask(csv_path)
    n_tasks = len(task_names)
    print(f"  Loaded: {X.shape[0]} molecules, {X.shape[1]} features, {n_tasks} tasks")

    # ===== 2. Split =====
    SPLIT_MODE = os.environ.get("MUV_SPLIT", "scaffold")
    print(f"  Split mode: {SPLIT_MODE}")

    # Set seeds for reproducibility (configurable via env for multi-seed runs)
    seed = int(os.environ.get("MUV_SEED", "137"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Seed: {seed}")

    val_ratio = 0.10
    test_ratio = 0.10

    from sklearn.preprocessing import StandardScaler

    valid_smiles = [smiles_list[i] for i in valid_indices]

    if SPLIT_MODE == "scaffold":
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        from collections import defaultdict

        scaffold_to_indices = defaultdict(list)
        for i, smi in enumerate(valid_smiles):
            try:
                scaffold = MurckoScaffoldSmiles(smi, includeChirality=False)
            except Exception:
                scaffold = ""
            scaffold_to_indices[scaffold].append(i)

        # Sort scaffolds by (size, first_index) descending (standard protocol)
        scaffold_sets = sorted(scaffold_to_indices.values(),
                               key=lambda x: (len(x), x[0]), reverse=True)

        n_total = len(X)
        train_cutoff = n_total * (1.0 - val_ratio - test_ratio)
        valid_cutoff = n_total * (1.0 - test_ratio)

        idx_train, idx_val, idx_test = [], [], []
        for scaffold_indices in scaffold_sets:
            if len(idx_train) + len(scaffold_indices) > train_cutoff:
                if len(idx_train) + len(idx_val) + len(scaffold_indices) > valid_cutoff:
                    idx_test.extend(scaffold_indices)
                else:
                    idx_val.extend(scaffold_indices)
            else:
                idx_train.extend(scaffold_indices)

        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)
        print(f"  Scaffold split: train={len(idx_train)}, val={len(idx_val)}, "
              f"test={len(idx_test)}")
        print(f"  Unique scaffolds: {len(scaffold_sets)}")

    elif SPLIT_MODE == "random":
        # Random split -- for multi-task with NaN labels, we cannot stratify
        # on a single y column. Use simple random permutation.
        from sklearn.model_selection import train_test_split
        n_total = len(X)
        all_idx = np.arange(n_total)
        rng = np.random.RandomState(seed)
        rng.shuffle(all_idx)
        n_train = int(n_total * (1.0 - val_ratio - test_ratio))
        n_val = int(n_total * val_ratio)
        idx_train = all_idx[:n_train]
        idx_val = all_idx[n_train:n_train + n_val]
        idx_test = all_idx[n_train + n_val:]
        print(f"  Random split: train={len(idx_train)}, val={len(idx_val)}, "
              f"test={len(idx_test)}")
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # Standardize fingerprint features (fit on train only)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X[idx_train])
    X_val = x_scaler.transform(X[idx_val])
    X_test = x_scaler.transform(X[idx_test])
    # Multi-task labels: float32 [N, 17] with NaN for missing
    y_train = y_multi[idx_train]
    y_val = y_multi[idx_val]
    y_test = y_multi[idx_test]

    # Count labeled samples per split
    train_labeled = np.sum(~np.isnan(y_train))
    val_labeled = np.sum(~np.isnan(y_val))
    test_labeled = np.sum(~np.isnan(y_test))
    print(f"  Labeled entries: train={train_labeled}, val={val_labeled}, test={test_labeled}")

    split_data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),  # [N, 17] with NaN
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "x_scaler": x_scaler,
        "d_input": X_train.shape[1],
        "n_classes": n_tasks,  # Treated as n_classes for compatibility
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    # ===== 3. Pre-compute molecular graphs =====
    print("  Pre-computing molecular graphs from SMILES...")
    reorder = np.concatenate([idx_train, idx_val, idx_test])

    all_mol_graphs = smiles_to_molecular_graphs(smiles_list, valid_indices)
    reordered_graphs = [all_mol_graphs[i] for i in reorder]
    mol_batch = PyGBatch.from_data_list(reordered_graphs)
    print(f"  Molecular batch: {mol_batch.num_graphs} molecules, "
          f"{mol_batch.x.shape[0]} atoms, {mol_batch.edge_index.shape[1]} bonds")

    # ===== 4. Create mini-batch dataloaders =====
    batch_size = 128
    loaders = create_mol_dataloaders_classification(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, steps/epoch={steps_per_epoch}")

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["molecular_graph"],
        d_init=64,
        initial_num_layers=2,

        # Molecular graph encoder config
        encoder_mol_gnn_type="gine",
        encoder_mol_hidden_dim=64,
        encoder_gnn_layers=3,
        encoder_switch_warmup_epochs=10,

        # Auto complexity scaling
        complexity_target_auto=True,
        complexity_ceiling_mult=5.0,
        hard_max_multiplier=2.0,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=12,
        recovery_epochs=4,

        # Overfitting thresholds
        overfitting_gap_early=0.50,
        overfitting_gap_moderate=0.80,
        overfitting_gap_severe=2.0,

        max_treatment_escalations=4,
        max_ops_per_layer=4,
        mixup_enabled=False,
        drop_path_enabled=False,

        device=device,
        optimizer=ASANNOptimizerConfig(
            base_lr=3e-4,
            weight_decay=0.01,
        ),
    )

    # ===== 6. Masked BCE loss =====
    loss_fn = MaskedBCEWithLogitsLoss()

    # ===== 7. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = n_tasks  # 17 outputs (one logit per task)

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        model.set_molecular_batch(mol_batch)
        model.set_molecular_graphs(reordered_graphs)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir, task_type="regression",
            n_classes=None,
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="regression",
        n_classes=None,
        target_metric="val_loss",
    )
    model = trainer.model
    model.set_molecular_batch(mol_batch)
    model.set_molecular_graphs(reordered_graphs)

    # ===== 8. Train =====
    max_epochs = 500
    print(f"\n  Training for {max_epochs} epochs "
          f"(~{max_epochs * steps_per_epoch} steps)...")
    train_metrics = trainer.train_epochs(
        train_data=loaders["train"],
        max_epochs=max_epochs,
        val_data=loaders["val"],
        test_data=loaders["test"],
        print_every=2000,
        snapshot_every=5000,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )

    # ===== 9. Evaluate: Global AUROC using BEST model =====
    from sklearn.metrics import roc_auc_score

    # Set test molecule indices
    n_train = len(split_data["X_train"])
    n_val = len(split_data["X_val"])
    n_test = len(split_data["X_test"])
    test_mol_idx = torch.arange(n_train + n_val, n_train + n_val + n_test)
    X_test_t = split_data["X_test"].to(device)
    y_test_np = split_data["y_test"].numpy()  # [N_test, 17] with NaN

    def _eval_muv_model(m, label="model"):
        """Evaluate a model and return global_auroc, global_labels, global_preds, per_task_auroc."""
        m.to(device)
        m.set_molecular_batch(mol_batch)
        m.set_molecular_graphs(reordered_graphs)
        m.set_current_mol_indices(test_mol_idx.to(device))
        m.eval()
        with torch.no_grad():
            logits = m(X_test_t).cpu()  # [N_test, 17]
        probs = torch.sigmoid(logits).numpy()  # [N_test, 17]

        all_labels, all_preds = [], []
        per_task = {}
        for j in range(n_tasks):
            labels_j = y_test_np[:, j].copy()
            preds_j = probs[:, j]
            nan_mask = np.isnan(labels_j)
            labels_j[nan_mask] = 0.0
            all_labels.append(labels_j)
            all_preds.append(preds_j)
            labeled = ~nan_mask
            if labeled.sum() > 0 and len(np.unique(labels_j[labeled])) > 1:
                per_task[task_names[j]] = float(
                    roc_auc_score(labels_j[labeled], preds_j[labeled])
                )

        gl = np.concatenate(all_labels)
        gp = np.concatenate(all_preds)
        try:
            ga = float(roc_auc_score(gl, gp))
        except ValueError:
            ga = 0.5
        return ga, gl, gp, per_task

    # Load best model (by val metric)
    best_model_path = os.path.join(results_dir, "best_model_full.pt")
    if os.path.exists(best_model_path):
        print(f"  Loading best model from {best_model_path}")
        eval_model = torch.load(best_model_path, map_location=device, weights_only=False)
        best_val = trainer._best_val_metric
        if best_val is not None:
            print(f"  Best model loaded (val metric: {best_val:.4f})")
    else:
        print("  WARNING: best_model_full.pt not found, using final model")
        eval_model = trainer.model

    global_auroc, global_labels, global_preds, per_task_auroc = _eval_muv_model(
        eval_model, "best"
    )

    print(f"\n  === MUV Results ({SPLIT_MODE} split, 17-task) ===")
    print(f"  Global AUROC (correct, best model): {global_auroc:.4f}")
    for tn, auc in per_task_auroc.items():
        print(f"    {tn}: {auc:.4f}")
    mean_task_auroc = 0.5
    if per_task_auroc:
        mean_task_auroc = np.mean(list(per_task_auroc.values()))
        print(f"  Mean per-task AUROC: {mean_task_auroc:.4f}")

    # Also evaluate final model
    final_auroc, final_labels, final_preds, _ = _eval_muv_model(
        trainer.model, "final"
    )
    print(f"  Global AUROC (correct, final model): {final_auroc:.4f}")

    metrics = {
        "auroc": global_auroc,
        "auroc_final": final_auroc,
        "global_auroc": global_auroc,
        "mean_task_auroc": float(mean_task_auroc),
        "n_tasks": n_tasks,
        "n_molecules": len(X),
        "split_mode": SPLIT_MODE,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    metrics.update({f"auroc_{tn}": v for tn, v in per_task_auroc.items()})

    # ===== 10. Buggy evaluation (scaffold mode only) =====
    if SPLIT_MODE == "scaffold":
        from tier_7.eval_all_protocols import buggy_auroc, fedlg_buggy_auroc

        # Best model buggy eval
        buggy_max_best, buggy_mean_best = buggy_auroc(
            global_labels, global_preds, batch_size=128, n_trials=501
        )
        print(f"  AUROC buggy (best model): max={buggy_max_best:.4f}, mean={buggy_mean_best:.4f}")

        # Final model buggy eval
        buggy_max_final, buggy_mean_final = buggy_auroc(
            final_labels, final_preds, batch_size=128, n_trials=501
        )
        print(f"  AUROC buggy (final model): max={buggy_max_final:.4f}, mean={buggy_mean_final:.4f}")

        # Take overall max
        buggy_max = max(buggy_max_best, buggy_max_final)
        buggy_mean = max(buggy_mean_best, buggy_mean_final)
        metrics["auroc_buggy_max"] = buggy_max
        metrics["auroc_buggy_mean"] = buggy_mean
        metrics["auroc_buggy_best_model"] = buggy_max_best
        metrics["auroc_buggy_final_model"] = buggy_max_final
        print(f"  AUROC KA-GNN buggy (max): {buggy_max:.4f}")

        # --- FedLG style buggy eval ---
        fedlg_best, _ = fedlg_buggy_auroc(global_labels, global_preds, batch_size=128, n_epochs=501)
        fedlg_final, _ = fedlg_buggy_auroc(final_labels, final_preds, batch_size=128, n_epochs=501)
        fedlg_max = max(fedlg_best, fedlg_final)
        metrics["auroc_fedlg_buggy_max"] = fedlg_max
        metrics["auroc_fedlg_buggy_best_model"] = fedlg_best
        metrics["auroc_fedlg_buggy_final_model"] = fedlg_final
        print(f"  AUROC FedLG buggy (max): {fedlg_max:.4f}")
        print(f"  Inflation KA-GNN: {buggy_max - global_auroc:.4f}, "
              f"FedLG: {fedlg_max - global_auroc:.4f}")

    # ===== 11. Save final checkpoint =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    encoder_type = getattr(eval_model.encoder, 'encoder_type', 'unknown')
    metrics["encoder_type"] = encoder_type
    print(f"  Final encoder: {encoder_type}")
    print(f"  Encoder: {eval_model.encoder.describe()}")

    # Save detailed results JSON
    results_json_path = os.path.join(results_dir, f"results_{SPLIT_MODE}.json")
    with open(results_json_path, "w") as f:
        # Filter only JSON-serializable values
        serializable = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool, type(None)))}
        json.dump(serializable, f, indent=2)
    print(f"  Detailed results saved to {results_json_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"

    # Support command-line split mode: python exp_7h_moleculenet_muv.py [scaffold|random]
    if len(sys.argv) > 1 and sys.argv[1] in ("scaffold", "random"):
        os.environ["MUV_SPLIT"] = sys.argv[1]

    split_mode = os.environ.get("MUV_SPLIT", "scaffold")
    exp_name = f"MoleculeNet MUV ({split_mode} split)"
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}\n")

    # Use split-specific and seed-specific results directory
    muv_seed = os.environ.get("MUV_SEED", "137")
    exp_dir_name = f"moleculenet_muv_{split_mode}_s{muv_seed}"

    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        exp_dir_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
