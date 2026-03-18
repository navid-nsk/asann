"""
Experiment 7J: MoleculeNet SIDER -- Side Effect Resource (27-task)
==================================================================
Dataset: SIDER from our own sider.csv
Samples: ~1,427 molecules
Features: 2048 (Morgan fingerprints from SMILES)
Tasks: 27 binary classification targets (side effect categories)
       Each task has complete labels (no NaN).

Standard MoleculeNet benchmark. Supports scaffold and random splitting.
BCEWithLogitsLoss (no masking needed -- all labels present).
Global AUROC computed by concatenating predictions across all tasks.

In scaffold mode, reports both KA-GNN-style and FedLG-style buggy evaluations.

Usage:
  python exp_7j_moleculenet_sider.py [scaffold|random]
  SIDER_SPLIT=random python exp_7j_moleculenet_sider.py

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


def run_experiment(results_dir: str):
    """Run MoleculeNet SIDER 27-task multi-label experiment."""
    try:
        from tier_7.bio_utils import load_sider_csv, smiles_to_molecular_graphs
    except ImportError:
        raise RuntimeError("SKIPPED: tier_7.bio_utils not importable")

    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("SKIPPED: rdkit not installed.")

    try:
        from torch_geometric.data import Batch as PyGBatch
    except ImportError:
        raise RuntimeError("SKIPPED: torch_geometric not installed.")

    device = get_device()

    # Set seeds for reproducibility
    seed = int(os.environ.get("SIDER_SEED", "137"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Seed: {seed}")

    # ===== 1. Load dataset =====
    csv_path = os.path.join(_proj_root, "data", "sider", "sider.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"SIDER CSV not found at {csv_path}")

    print("  Loading SIDER dataset from sider.csv...")
    X, y_multi, task_names, smiles_list, valid_indices = load_sider_csv(csv_path)
    n_tasks = len(task_names)
    print(f"  Loaded: {X.shape[0]} molecules, {X.shape[1]} features, {n_tasks} tasks")

    # ===== 2. Split =====
    SPLIT_MODE = os.environ.get("SIDER_SPLIT", "scaffold")
    print(f"  Split mode: {SPLIT_MODE}")

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
        n_total = len(X)
        all_idx = np.arange(n_total)
        rng = np.random.RandomState(seed)
        rng.shuffle(all_idx)
        n_train = int(n_total * (1.0 - val_ratio - test_ratio))
        n_val_split = int(n_total * val_ratio)
        idx_train = all_idx[:n_train]
        idx_val = all_idx[n_train:n_train + n_val_split]
        idx_test = all_idx[n_train + n_val_split:]
        print(f"  Random split: train={len(idx_train)}, val={len(idx_val)}, "
              f"test={len(idx_test)}")
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # Standardize features
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X[idx_train])
    X_val = x_scaler.transform(X[idx_val])
    X_test = x_scaler.transform(X[idx_test])
    y_train = y_multi[idx_train]
    y_val = y_multi[idx_val]
    y_test = y_multi[idx_test]

    # Count labeled samples
    train_labeled = int(np.sum(~np.isnan(y_train)))
    val_labeled = int(np.sum(~np.isnan(y_val)))
    test_labeled = int(np.sum(~np.isnan(y_test)))
    print(f"  Labeled entries: train={train_labeled}, val={val_labeled}, test={test_labeled}")

    split_data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "x_scaler": x_scaler,
        "d_input": X_train.shape[1],
        "n_classes": n_tasks,
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

    # ===== 4. Configure ASANN =====
    config = ASANNConfig.from_task(
        task_type="classification",
        modality="molecular_classification",
        d_input=split_data["d_input"],
        d_output=n_tasks,
        n_samples=len(split_data["X_train"]),
        device=device,
    )
    batch_size = config.recommended_batch_size

    # ===== 5. Create dataloaders =====
    loaders = create_mol_dataloaders_classification(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, steps/epoch={steps_per_epoch}")

    # ===== 6. BCE loss (no masking -- SIDER has no NaN) =====
    loss_fn = nn.BCEWithLogitsLoss()

    # ===== 7. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = n_tasks

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
    max_epochs = config.recommended_max_epochs
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

    n_train = len(split_data["X_train"])
    n_val = len(split_data["X_val"])
    n_test = len(split_data["X_test"])
    test_mol_idx = torch.arange(n_train + n_val, n_train + n_val + n_test)
    X_test_t = split_data["X_test"].to(device)
    y_test_np = split_data["y_test"].numpy()

    def _eval_model(m, label="model"):
        """Evaluate model, return global_auroc, global_labels, global_preds, per_task_auroc."""
        m.to(device)
        m.set_molecular_batch(mol_batch)
        m.set_molecular_graphs(reordered_graphs)
        m.set_current_mol_indices(test_mol_idx.to(device))
        m.eval()
        with torch.no_grad():
            logits = m(X_test_t).cpu()
        probs = torch.sigmoid(logits).numpy()

        all_labels, all_preds = [], []
        per_task = {}
        for j in range(n_tasks):
            labels_j = y_test_np[:, j]
            preds_j = probs[:, j]
            all_labels.append(labels_j)
            all_preds.append(preds_j)
            if len(np.unique(labels_j)) > 1:
                per_task[task_names[j]] = float(roc_auc_score(labels_j, preds_j))

        gl = np.concatenate(all_labels)
        gp = np.concatenate(all_preds)
        try:
            ga = float(roc_auc_score(gl, gp))
        except ValueError:
            ga = 0.5
        return ga, gl, gp, per_task

    # Load best model
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

    global_auroc, global_labels, global_preds, per_task_auroc = _eval_model(eval_model, "best")

    print(f"\n  === SIDER Results ({SPLIT_MODE} split, {n_tasks}-task) ===")
    print(f"  Global AUROC (correct, best model): {global_auroc:.4f}")
    for tn, auc in per_task_auroc.items():
        print(f"    {tn}: {auc:.4f}")
    mean_task_auroc = 0.5
    if per_task_auroc:
        mean_task_auroc = np.mean(list(per_task_auroc.values()))
        print(f"  Mean per-task AUROC: {mean_task_auroc:.4f}")

    # Also evaluate final model
    final_auroc, final_labels, final_preds, _ = _eval_model(trainer.model, "final")
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

    # ===== 10. Buggy evaluations (scaffold mode only) =====
    if SPLIT_MODE == "scaffold":
        from tier_7.eval_all_protocols import buggy_auroc, fedlg_buggy_auroc

        # --- KA-GNN style ---
        buggy_max_best, buggy_mean_best = buggy_auroc(
            global_labels, global_preds, batch_size=128, n_trials=501)
        buggy_max_final, buggy_mean_final = buggy_auroc(
            final_labels, final_preds, batch_size=128, n_trials=501)
        buggy_max = max(buggy_max_best, buggy_max_final)
        metrics["auroc_buggy_max"] = buggy_max
        metrics["auroc_buggy_mean"] = max(buggy_mean_best, buggy_mean_final)
        metrics["auroc_buggy_best_model"] = buggy_max_best
        metrics["auroc_buggy_final_model"] = buggy_max_final
        print(f"  AUROC KA-GNN buggy (max): {buggy_max:.4f}")

        # --- FedLG style ---
        fedlg_best, _ = fedlg_buggy_auroc(global_labels, global_preds, batch_size=128, n_epochs=501)
        fedlg_final, _ = fedlg_buggy_auroc(final_labels, final_preds, batch_size=128, n_epochs=501)
        fedlg_max = max(fedlg_best, fedlg_final)
        metrics["auroc_fedlg_buggy_max"] = fedlg_max
        metrics["auroc_fedlg_buggy_best_model"] = fedlg_best
        metrics["auroc_fedlg_buggy_final_model"] = fedlg_final
        print(f"  AUROC FedLG buggy (max): {fedlg_max:.4f}")
        print(f"  Inflation KA-GNN: {buggy_max - global_auroc:.4f}, "
              f"FedLG: {fedlg_max - global_auroc:.4f}")

    # ===== 11. Save =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)

    encoder_type = getattr(eval_model.encoder, 'encoder_type', 'unknown')
    metrics["encoder_type"] = encoder_type
    print(f"  Final encoder: {encoder_type}")
    print(f"  Encoder: {eval_model.encoder.describe()}")

    results_json_path = os.path.join(results_dir, f"results_{SPLIT_MODE}.json")
    with open(results_json_path, "w") as f:
        serializable = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool, type(None)))}
        json.dump(serializable, f, indent=2)
    print(f"  Detailed results saved to {results_json_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"

    if len(sys.argv) > 1 and sys.argv[1] in ("scaffold", "random"):
        os.environ["SIDER_SPLIT"] = sys.argv[1]

    split_mode = os.environ.get("SIDER_SPLIT", "scaffold")
    exp_name = f"MoleculeNet SIDER ({split_mode} split)"
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}\n")

    sider_seed = os.environ.get("SIDER_SEED", "137")
    exp_dir_name = f"moleculenet_sider_{split_mode}_s{sider_seed}"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        exp_dir_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
