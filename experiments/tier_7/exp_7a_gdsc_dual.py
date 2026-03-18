"""
Experiment 7A-Dual: GDSC2 Drug Sensitivity with Dual Encoder
=============================================================
Dataset: GDSC2 (local CSV) + DepMap gene expression + PyTDC SMILES
Samples: ~93K (drug-cell line pairs with gene expression + SMILES)
Features: 512 (PCA-reduced cell line gene expression, 19K -> 512)
Target: LN_IC50 (continuous, regression)
Task: Predict drug sensitivity from cell line genomics + drug structure

Dual encoder:
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

from common import (
    setup_paths, get_device, split_and_standardize,
    create_mol_dataloaders, evaluate_model,
    config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer, compute_max_epochs,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run GDSC2 dual-encoder drug sensitivity regression."""
    # Check dependencies
    try:
        from tier_7.bio_utils import (
            load_gdsc2_dual_data, smiles_to_molecular_graphs,
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

    # ===== 1. Load dual data =====
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "gdsc2")
    local_test = os.environ.get("GDSC_LOCAL_TEST", "0") == "1"
    max_samples = 5000 if local_test else None
    pca_dim = 512

    print("  Loading GDSC2 dual data (cell expression + drug SMILES)...")
    X_cell, y, smiles_per_sample, valid_indices, d_cell = \
        load_gdsc2_dual_data(
            data_dir=data_dir,
            max_samples=max_samples,
            pca_dim=pca_dim,
        )
    print(f"  Loaded: {X_cell.shape[0]} samples, {d_cell} cell features")

    # ===== 2. Split and standardize =====
    # X_cell is the feature matrix; standardize it
    split_data = split_and_standardize(
        X_cell, y, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # ===== 3. Pre-compute molecular graphs =====
    print("  Pre-computing molecular graphs from SMILES...")
    from sklearn.model_selection import train_test_split as _tts

    N = len(X_cell)
    all_idx = np.arange(N)
    idx_trainval, idx_test = _tts(all_idx, test_size=0.15, random_state=42)
    val_frac = 0.15 / 0.85
    idx_train, idx_val = _tts(idx_trainval, test_size=val_frac,
                              random_state=42)
    reorder = np.concatenate([idx_train, idx_val, idx_test])

    all_mol_graphs = smiles_to_molecular_graphs(smiles_per_sample,
                                                valid_indices)
    reordered_graphs = [all_mol_graphs[i] for i in reorder]
    mol_batch = PyGBatch.from_data_list(reordered_graphs)
    n_unique_drugs = len(set(smiles_per_sample))
    print(f"  Molecular batch: {mol_batch.num_graphs} graphs "
          f"({n_unique_drugs} unique drugs), "
          f"{mol_batch.x.shape[0]} atoms, "
          f"{mol_batch.edge_index.shape[1]} bonds")

    # ===== 4. Configure ASANN with dual encoder =====
    config = ASANNConfig.from_task(
        task_type="regression",
        modality="pharmacogenomic",
        d_input=d_cell,
        d_output=1,
        n_samples=len(split_data["X_train"]),
        device=device,
    )
    max_epochs = 20 if local_test else config.recommended_max_epochs
    batch_size = config.recommended_batch_size

    # ===== 5. Create mini-batch dataloaders with molecule indices =====
    loaders = create_mol_dataloaders(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, "
          f"steps/epoch={steps_per_epoch}")

    # ===== 6. Create model and trainer =====
    d_input = d_cell  # cell line features (PCA-reduced)
    d_output = 1  # regression
    loss_fn = torch.nn.MSELoss()

    def create_fresh_trainer():
        model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
        model.to(device)
        model.set_molecular_batch(mol_batch)
        model.set_molecular_graphs(reordered_graphs)
        return ASANNTrainer(
            model=model, config=config,
            task_loss_fn=loss_fn,
            log_dir=results_dir, task_type="regression",
            y_scaler=split_data.get("y_scaler"),
        )

    trainer, is_resumed = resume_or_create_trainer(
        results_dir=results_dir,
        create_fn=create_fresh_trainer,
        task_loss_fn=loss_fn,
        task_type="regression",
        y_scaler=split_data.get("y_scaler"),
        target_metric="pearson_r",
    )
    model = trainer.model
    # Always re-set molecular data (not persisted in checkpoints)
    model.set_molecular_batch(mol_batch)
    model.set_molecular_graphs(reordered_graphs)

    # ===== 7. Train =====
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

    # ===== 8. Evaluate on test set =====
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

    metrics = evaluate_model(
        eval_model, split_data["X_test"], split_data["y_test_original"],
        split_data["y_scaler"], device,
    )

    # ===== 9. Domain-specific: Pearson/Spearman correlation =====
    from scipy.stats import pearsonr, spearmanr
    eval_model.eval()
    with torch.no_grad():
        X_test = split_data["X_test"].to(device)
        preds_scaled = eval_model(X_test).cpu().numpy()
    preds_original = split_data["y_scaler"].inverse_transform(
        preds_scaled).flatten()
    y_true = split_data["y_test_original"].numpy().flatten()

    metrics["pearson_r"] = float(pearsonr(y_true, preds_original)[0])
    metrics["spearman_rho"] = float(spearmanr(y_true, preds_original)[0])
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman rho: {metrics['spearman_rho']:.4f}")

    # ===== 10. Save final checkpoint =====
    arch = eval_model.describe_architecture()
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"
    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        "GDSC2 Dual Encoder", run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
