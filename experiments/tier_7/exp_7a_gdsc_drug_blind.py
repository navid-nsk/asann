"""
Experiment 7A-Drug-Blind: GDSC2 Drug Sensitivity with Unseen Drug Split
=========================================================================
Dataset: GDSC2 (local CSV) + DepMap gene expression + PyTDC SMILES
Task: Predict drug sensitivity for UNSEEN drugs (drug-blind evaluation)

Split strategy: Drug-Blind (Unseen Drug)
  - 70% of drugs in train, 15% in val, 15% in test
  - ALL samples for a given drug go to the same split
  - Test drugs are completely absent from training
  - Tests true generalization to novel chemical structures

Same dual encoder as exp_7a_gdsc_dual.py:
  - Cell line branch: MLP on PCA-reduced gene expression
  - Drug branch: GINEConv molecular graph encoder on drug SMILES
  - Fusion: concatenation + linear projection

Requires: pip install PyTDC torch_geometric rdkit scikit-learn
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
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from common import (
    setup_paths, get_device, MolIndexDataset,
    evaluate_model, config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer
from asann.config import ASANNOptimizerConfig


def run_experiment(results_dir: str):
    """Run GDSC2 drug-blind dual-encoder drug sensitivity regression."""
    # Check dependencies
    try:
        from tier_7.bio_utils import (
            load_gdsc2_dual_data_drug_split, smiles_to_molecular_graphs,
            _patch_tdc_imports,
        )
    except ImportError:
        raise RuntimeError("SKIPPED: tier_7.bio_utils not importable")

    try:
        _patch_tdc_imports()
        from tdc.multi_pred.drugres import DrugRes
    except ImportError:
        raise RuntimeError(
            "SKIPPED: PyTDC not installed. Install with: pip install PyTDC"
        )

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
            "SKIPPED: torch_geometric not installed. "
            "Install with: pip install torch_geometric"
        )

    device = get_device()

    # ===== 1. Load dual data with drug-blind split =====
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "gdsc2")
    local_test = os.environ.get("GDSC_LOCAL_TEST", "0") == "1"
    pca_dim = 512

    print("  Loading GDSC2 data with DRUG-BLIND split...")
    raw = load_gdsc2_dual_data_drug_split(
        data_dir=data_dir, pca_dim=pca_dim, seed=42,
    )

    d_cell = raw['d_cell']
    print(f"  Drug-blind: {raw['n_train_drugs']} train drugs, "
          f"{raw['n_val_drugs']} val drugs, {raw['n_test_drugs']} test drugs")

    # ===== 2. Standardize features & targets (fit on train only) =====
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(raw['X_train']).astype(np.float32)
    X_val = x_scaler.transform(raw['X_val']).astype(np.float32)
    X_test = x_scaler.transform(raw['X_test']).astype(np.float32)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(
        raw['y_train'].reshape(-1, 1)).ravel().astype(np.float32)
    y_val = y_scaler.transform(
        raw['y_val'].reshape(-1, 1)).ravel().astype(np.float32)
    y_test_scaled = y_scaler.transform(
        raw['y_test'].reshape(-1, 1)).ravel().astype(np.float32)
    y_test_original = raw['y_test'].copy()

    # Build split_data dict compatible with common.py utilities
    split_data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test_scaled": torch.tensor(y_test_scaled,
                                       dtype=torch.float32).unsqueeze(1),
        "y_test_original": torch.tensor(y_test_original,
                                         dtype=torch.float32).unsqueeze(1),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "d_input": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    print(f"  Standardized: train={len(X_train)}, val={len(X_val)}, "
          f"test={len(X_test)}")

    # ===== 3. Pre-compute molecular graphs =====
    # Concatenate smiles in [train, val, test] order for graph indexing
    all_smiles = (raw['smiles_train'] + raw['smiles_val']
                  + raw['smiles_test'])
    all_valid = np.arange(len(all_smiles))

    print("  Pre-computing molecular graphs from SMILES...")
    all_mol_graphs = smiles_to_molecular_graphs(all_smiles, all_valid)
    mol_batch = PyGBatch.from_data_list(all_mol_graphs)

    n_unique_train = len(set(raw['smiles_train']))
    n_unique_test = len(set(raw['smiles_test']))
    print(f"  Molecular batch: {mol_batch.num_graphs} graphs, "
          f"{mol_batch.x.shape[0]} atoms, "
          f"{mol_batch.edge_index.shape[1]} bonds")
    print(f"  Unique drug molecules: train={n_unique_train}, "
          f"test={n_unique_test} (ZERO overlap)")

    # ===== 4. Create mini-batch dataloaders with molecule indices =====
    batch_size = 256
    n_train = len(X_train)
    n_val = len(X_val)

    train_ds = MolIndexDataset(split_data["X_train"],
                               split_data["y_train"], start_idx=0)
    val_ds = MolIndexDataset(split_data["X_val"],
                             split_data["y_val"], start_idx=n_train)
    test_ds = MolIndexDataset(split_data["X_test"],
                              split_data["y_test_scaled"],
                              start_idx=n_train + n_val)

    drop_last = n_train % batch_size == 1
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            drop_last=drop_last),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, "
          f"steps/epoch={steps_per_epoch}")

    # ===== 5. Configure ASANN with dual encoder =====
    max_epochs = 20 if local_test else 100
    config = ASANNConfig(
        encoder_candidates=["dual_drug_cell"],
        d_init=128,
        initial_num_layers=3,

        # Dual encoder config
        dual_encoder_cell_hidden=256,
        dual_encoder_cell_out=64,
        dual_encoder_drug_out=64,

        # Molecular graph encoder (drug branch)
        encoder_mol_gnn_type="gine",
        encoder_mol_hidden_dim=64,
        encoder_gnn_layers=3,
        encoder_switch_warmup_epochs=10,

        # Auto complexity scaling
        complexity_target_auto=True,
        complexity_target_multiplier=8.0,
        complexity_ceiling_mult=5.0,
        hard_max_multiplier=2.0,

        # Epoch-based diagnosis
        diagnosis_enabled=True,
        warmup_epochs=5,
        surgery_epoch_interval=3,
        eval_epoch_interval=2,
        meta_update_epoch_interval=10,
        stability_healthy_epochs=10,
        recovery_epochs=4,

        max_treatment_escalations=4,
        max_ops_per_layer=4,

        device=device,
        optimizer=ASANNOptimizerConfig(
            base_lr=5e-4,
            weight_decay=0.05,
        ),
    )

    # ===== 6. Create model and trainer =====
    d_input = d_cell
    d_output = 1
    loss_fn = torch.nn.MSELoss()

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        model.set_molecular_batch(mol_batch)
        model.set_molecular_graphs(all_mol_graphs)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir, task_type="regression",
            y_scaler=y_scaler,
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="regression",
        y_scaler=y_scaler,
        target_metric="pearson_r",
    )
    model = trainer.model
    model.set_molecular_batch(mol_batch)
    model.set_molecular_graphs(all_mol_graphs)

    # ===== 7. Train =====
    print(f"\n  Training for {max_epochs} epochs "
          f"(~{max_epochs * steps_per_epoch} steps)...")
    print(f"  NOTE: Drug-blind split -- test drugs are UNSEEN during training")
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

    # ===== 8. Evaluate on test set (unseen drugs) =====
    eval_model = trainer.model
    eval_model.to(device)
    eval_model.set_molecular_batch(mol_batch)
    eval_model.set_molecular_graphs(all_mol_graphs)

    test_mol_idx = torch.arange(n_train + n_val,
                                n_train + n_val + len(X_test))
    eval_model.set_current_mol_indices(test_mol_idx.to(device))

    metrics = evaluate_model(
        eval_model, split_data["X_test"], split_data["y_test_original"],
        y_scaler, device,
    )

    # ===== 9. Pearson/Spearman on unseen drugs =====
    from scipy.stats import pearsonr, spearmanr
    eval_model.eval()
    with torch.no_grad():
        X_t = split_data["X_test"].to(device)
        preds_scaled = eval_model(X_t).cpu().numpy()
    preds_original = y_scaler.inverse_transform(preds_scaled).flatten()
    y_true = y_test_original.flatten()

    metrics["pearson_r"] = float(pearsonr(y_true, preds_original)[0])
    metrics["spearman_rho"] = float(spearmanr(y_true, preds_original)[0])
    metrics["split_type"] = "drug_blind"
    metrics["n_train_drugs"] = raw['n_train_drugs']
    metrics["n_val_drugs"] = raw['n_val_drugs']
    metrics["n_test_drugs"] = raw['n_test_drugs']
    metrics["test_drug_names"] = raw['test_drug_names']

    print(f"\n  === DRUG-BLIND TEST RESULTS (unseen drugs) ===")
    print(f"  Pearson r:    {metrics['pearson_r']:.4f}")
    print(f"  Spearman rho: {metrics['spearman_rho']:.4f}")
    print(f"  Test drugs ({raw['n_test_drugs']}): "
          f"{', '.join(raw['test_drug_names'][:10])}...")

    # ===== 10. Save =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "GDSC2 Drug-Blind", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
