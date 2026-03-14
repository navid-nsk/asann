"""
Experiment 7B: MoleculeNet BBBP -- Blood-Brain Barrier Penetration
=================================================================
Dataset: BBBP from our own bbb_martins.tab (PyTDC source)
Samples: ~2,039 molecules
Features: 2048 (Morgan fingerprints from SMILES)
Classes: 2 (penetrating=1, non-penetrating=0)
Task: Binary classification of molecular property

Supports scaffold and random splitting.
In scaffold mode, also reports buggy KA-GNN-style evaluation for comparison.

Uses 2D molecular graphs (32-dim atoms, 12-dim bonds).
ASANN self-discovers optimal architecture via surgery system.

Usage:
  python exp_7b_moleculenet.py [scaffold|random]
  BBBP_SPLIT=random python exp_7b_moleculenet.py

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
import numpy as np
from pathlib import Path

from common import (
    setup_paths, get_device,
    create_mol_dataloaders_classification, evaluate_classification_model,
    config_to_dict, run_classification_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.asann_optimizer import ASANNOptimizerConfig


def run_experiment(results_dir: str):
    """Run MoleculeNet BBBP classification experiment."""
    # Check dependencies
    try:
        from tier_7.bio_utils import load_bbbp_tab, smiles_to_molecular_graphs
    except ImportError:
        raise RuntimeError("SKIPPED: tier_7.bio_utils not importable")

    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError(
            "SKIPPED: rdkit not installed. Install with: pip install rdkit"
        )

    try:
        from torch_geometric.data import Batch as PyGBatch
    except ImportError:
        raise RuntimeError(
            "SKIPPED: torch_geometric not installed. Install with: pip install torch_geometric"
        )

    device = get_device()

    # Set seeds for reproducibility (configurable via env for multi-seed runs)
    seed = int(os.environ.get("BBBP_SEED", "137"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Seed: {seed}")

    # ===== 1. Load dataset from our own bbb_martins.tab =====
    tab_path = os.path.join(_proj_root, "data", "bbb_martins.tab")
    if not os.path.exists(tab_path):
        raise RuntimeError(f"BBBP tab not found at {tab_path}")

    print("  Loading BBBP dataset from bbb_martins.tab...")
    X, y, n_classes, smiles_list, valid_indices = load_bbbp_tab(
        tab_path, return_smiles=True
    )
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")

    # ===== 2. Split =====
    SPLIT_MODE = os.environ.get("BBBP_SPLIT", "scaffold")
    print(f"  Split mode: {SPLIT_MODE}")

    val_ratio = 0.10
    test_ratio = 0.10

    from sklearn.preprocessing import StandardScaler

    # Get valid SMILES (matching the filtered fingerprints)
    valid_smiles = [smiles_list[i] for i in valid_indices]

    if SPLIT_MODE == "scaffold":
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        from collections import defaultdict

        # Murcko scaffold splitting (standard protocol)
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
        # Standard stratified random split
        from sklearn.model_selection import train_test_split
        n_total = len(X)
        all_idx = np.arange(n_total)
        idx_train, idx_temp = train_test_split(
            all_idx, test_size=val_ratio + test_ratio,
            random_state=seed, stratify=y
        )
        # Split remaining into val and test
        y_temp = y[idx_temp]
        relative_test = test_ratio / (val_ratio + test_ratio)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=relative_test,
            random_state=seed, stratify=y_temp
        )
        print(f"  Random split: train={len(idx_train)}, val={len(idx_val)}, "
              f"test={len(idx_test)}")

    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # Standardize features (fit on train only)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X[idx_train])
    X_val = x_scaler.transform(X[idx_val])
    X_test = x_scaler.transform(X[idx_test])
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(split_y, return_counts=True)
        print(f"  {split_name} class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    split_data = {
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

    # ===== 3. Pre-compute 2D molecular graphs =====
    print("  Pre-computing 2D molecular graphs from SMILES...")
    reorder = np.concatenate([idx_train, idx_val, idx_test])

    all_mol_graphs = smiles_to_molecular_graphs(smiles_list, valid_indices)
    reordered_graphs = [all_mol_graphs[i] for i in reorder]
    mol_batch = PyGBatch.from_data_list(reordered_graphs)
    print(f"  Molecular batch: {mol_batch.num_graphs} molecules, "
          f"{mol_batch.x.shape[0]} atoms, {mol_batch.edge_index.shape[1]} bonds")

    # ===== 4. Create mini-batch dataloaders with molecule indices =====
    batch_size = 128
    loaders = create_mol_dataloaders_classification(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, "
          f"steps/epoch={steps_per_epoch}")

    # ===== 5. Configure ASANN =====
    config = ASANNConfig(
        encoder_candidates=["molecular_graph"],
        d_init=64,
        initial_num_layers=2,

        # Molecular graph encoder config -- GINE with 2D features
        # NOTE: 5L x 128d was worse (0.62 vs 0.67). Keep 3L x 64d.
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
        warmup_epochs=20,
        surgery_epoch_interval=5,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=15,
        recovery_epochs=6,

        # Very relaxed overfitting thresholds -- scaffold split inherently
        # causes large train/val gap; don't diagnose this as overfitting
        overfitting_gap_early=1.0,
        overfitting_gap_moderate=2.0,
        overfitting_gap_severe=5.0,

        # More patience -- let training run longer
        max_treatment_escalations=10,
        treatment_exhaustion_patience=10,
        stalled_convergence_patience=80,
        stalled_convergence_min_epochs=200,
        auto_stop_enabled=False,                 # Don't stop early

        max_ops_per_layer=4,
        mixup_enabled=False,
        drop_path_enabled=False,

        device=device,
        optimizer=ASANNOptimizerConfig(
            base_lr=3e-4,
            weight_decay=0.01,
        ),
    )

    # ===== 6. Compute class weights for imbalanced data =====
    y_train_np = split_data["y_train"].numpy()
    class_counts = np.bincount(y_train_np, minlength=n_classes)
    class_weights = len(y_train_np) / (n_classes * class_counts.astype(np.float32))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights: {class_weights.cpu().tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ===== 7. Create model and trainer =====
    d_input = split_data["d_input"]
    d_output = n_classes

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        # Store molecular data for MolecularGraphEncoder
        model.set_molecular_batch(mol_batch)
        model.set_molecular_graphs(reordered_graphs)
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
        target_metric="auroc",
    )
    model = trainer.model
    # Always re-set molecular data (not persisted in checkpoints)
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
        print_every=500,
        snapshot_every=1000,
        checkpoint_path=os.path.join(results_dir, "training_checkpoint.pt"),
        checkpoint_every_epochs=50,
    )

    # ===== 9. Evaluate on test set using BEST model (by val AUROC) =====
    from sklearn.metrics import roc_auc_score

    best_model_path = os.path.join(results_dir, "best_model_full.pt")
    if os.path.exists(best_model_path):
        print(f"  Loading best model from {best_model_path}")
        eval_model = torch.load(best_model_path, map_location=device, weights_only=False)
        eval_model.to(device)
        best_val = trainer._best_val_metric
        if best_val is not None:
            print(f"  Best model loaded (val auroc: {best_val:.4f})")
    else:
        print("  WARNING: best_model_full.pt not found, using final model")
        eval_model = trainer.model
        eval_model.to(device)
    eval_model.set_molecular_batch(mol_batch)
    eval_model.set_molecular_graphs(reordered_graphs)

    # Set test molecule indices for full-test-set evaluation
    n_train = len(split_data["X_train"])
    n_val = len(split_data["X_val"])
    n_test = len(split_data["X_test"])
    test_mol_idx = torch.arange(n_train + n_val, n_train + n_val + n_test)
    eval_model.set_current_mol_indices(test_mol_idx.to(device))

    # Use single batch to match mol_indices (set for full test set)
    metrics = evaluate_classification_model(
        eval_model, split_data["X_test"], split_data["y_test"],
        device, n_classes, batch_size=n_test,
    )

    # AUROC -- correct evaluation (all test samples)
    eval_model.eval()
    with torch.no_grad():
        X_test_t = split_data["X_test"].to(device)
        test_logits = eval_model(X_test_t).cpu()
    probs = torch.softmax(test_logits, dim=1)[:, 1].numpy()
    y_test_np = split_data["y_test"].numpy()
    metrics["auroc"] = float(roc_auc_score(y_test_np, probs))
    metrics["split_mode"] = SPLIT_MODE
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["n_test"] = n_test
    print(f"  AUROC (correct, all {len(y_test_np)} samples): {metrics['auroc']:.4f}")

    # ===== 10. Buggy evaluation (scaffold mode only) =====
    # KA-GNN evaluates a random test subset at EACH training epoch and takes max
    # across all (epoch, subset) pairs. We simulate this by evaluating both
    # the best model AND the final model, taking the overall max.
    if SPLIT_MODE == "scaffold":
        from tier_7.eval_all_protocols import buggy_auroc, fedlg_buggy_auroc

        # Best model buggy eval
        buggy_max_best, buggy_mean_best = buggy_auroc(
            y_test_np, probs, batch_size=128, n_trials=501
        )
        print(f"  AUROC buggy (best model): max={buggy_max_best:.4f}, mean={buggy_mean_best:.4f}")

        # Final model buggy eval (different generalization pattern)
        final_model = trainer.model
        final_model.to(device)
        final_model.set_molecular_batch(mol_batch)
        final_model.set_molecular_graphs(reordered_graphs)
        final_model.set_current_mol_indices(test_mol_idx.to(device))
        final_model.eval()
        with torch.no_grad():
            final_logits = final_model(X_test_t).cpu()
        probs_final = torch.softmax(final_logits, dim=1)[:, 1].numpy()
        final_correct = float(roc_auc_score(y_test_np, probs_final))
        buggy_max_final, buggy_mean_final = buggy_auroc(
            y_test_np, probs_final, batch_size=128, n_trials=501
        )
        print(f"  AUROC buggy (final model): max={buggy_max_final:.4f}, "
              f"mean={buggy_mean_final:.4f} (correct={final_correct:.4f})")

        # Take overall max (simulates KA-GNN's max across all epochs)
        buggy_max = max(buggy_max_best, buggy_max_final)
        buggy_mean = max(buggy_mean_best, buggy_mean_final)
        metrics["auroc_buggy_max"] = buggy_max
        metrics["auroc_buggy_mean"] = buggy_mean
        metrics["auroc_buggy_best_model"] = buggy_max_best
        metrics["auroc_buggy_final_model"] = buggy_max_final
        metrics["auroc_final_correct"] = final_correct
        print(f"  AUROC KA-GNN buggy (max): {buggy_max:.4f}")

        # --- FedLG style buggy eval (per-batch AUROC averaging) ---
        fedlg_best, fedlg_mean_best = fedlg_buggy_auroc(
            y_test_np, probs, batch_size=128, n_epochs=501
        )
        fedlg_final, fedlg_mean_final = fedlg_buggy_auroc(
            y_test_np, probs_final, batch_size=128, n_epochs=501
        )
        fedlg_max = max(fedlg_best, fedlg_final)
        metrics["auroc_fedlg_buggy_max"] = fedlg_max
        metrics["auroc_fedlg_buggy_best_model"] = fedlg_best
        metrics["auroc_fedlg_buggy_final_model"] = fedlg_final
        print(f"  AUROC FedLG buggy (max): {fedlg_max:.4f}")
        print(f"  Inflation KA-GNN: {buggy_max - metrics['auroc']:.4f}, "
              f"FedLG: {fedlg_max - metrics['auroc']:.4f}")

    # ===== 11. Save final checkpoint =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Report encoder discovery
    encoder_type = getattr(eval_model.encoder, 'encoder_type', 'unknown')
    metrics["encoder_type"] = encoder_type
    print(f"  Final encoder: {encoder_type}")
    print(f"  Encoder: {eval_model.encoder.describe()}")

    # Save detailed results JSON
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

    # Support command-line split mode: python exp_7b_moleculenet.py [scaffold|random]
    if len(sys.argv) > 1 and sys.argv[1] in ("scaffold", "random"):
        os.environ["BBBP_SPLIT"] = sys.argv[1]

    split_mode = os.environ.get("BBBP_SPLIT", "scaffold")
    exp_name = f"MoleculeNet BBBP ({split_mode} split)"
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}\n")

    # Use split-specific and seed-specific results directory
    bbbp_seed = os.environ.get("BBBP_SEED", "137")
    exp_dir_name = f"moleculenet_bbbp_{split_mode}_s{bbbp_seed}"
    name, metrics, arch, elapsed, status = run_classification_experiment_wrapper(
        exp_dir_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
