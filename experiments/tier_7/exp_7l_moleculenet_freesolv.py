"""
Experiment 7L: MoleculeNet FreeSolv -- Hydration Free Energy Prediction
========================================================================
Dataset: FreeSolv (Free Solvation Database)
Samples: ~642 molecules
Features: 524 (512-bit Morgan fingerprints + 12 RDKit descriptors)
Target: Experimental hydration free energy in kcal/mol (continuous, regression)
Task: Predict hydration free energy from molecular structure.

FedLG (Nature Machine Intelligence 2024) reported RMSE for this dataset.
We evaluate under both correct protocol and FedLG's buggy protocol
(per-batch RMSE averaging + min across shuffled epochs).

Uses scaffold splitting for fair evaluation.
Uses mini-batch training with MolecularGraphEncoder support.

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
    create_mol_dataloaders, evaluate_model,
    config_to_dict, run_experiment_wrapper,
    resume_or_create_trainer,
)
from asann import ASANNConfig, ASANNModel, ASANNTrainer


def run_experiment(results_dir: str):
    """Run MoleculeNet FreeSolv regression experiment."""
    try:
        from tier_7.bio_utils import load_freesolv_csv, smiles_to_molecular_graphs
    except ImportError:
        raise RuntimeError("SKIPPED: tier_7.bio_utils not importable")

    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError("SKIPPED: rdkit not installed. Install with: pip install rdkit")

    try:
        from torch_geometric.data import Batch as PyGBatch
    except ImportError:
        raise RuntimeError("SKIPPED: torch_geometric not installed.")

    device = get_device()

    # Set seeds for reproducibility (configurable via env for multi-seed runs)
    seed = int(os.environ.get("FREESOLV_SEED", "42"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Seed: {seed}")

    SPLIT_MODE = os.environ.get("FREESOLV_SPLIT", "scaffold")
    print(f"  Split mode: {SPLIT_MODE}")

    # ===== 1. Load dataset from CSV =====
    csv_path = os.path.join(_proj_root, "data", "FreeSolv", "freesolve.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"FreeSolv CSV not found at {csv_path}")

    print("  Loading FreeSolv dataset...")
    X, y, smiles_list, valid_indices = load_freesolv_csv(csv_path, return_smiles=True)
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # ===== 2. Split =====
    val_ratio = 0.10
    test_ratio = 0.20  # Match FedLG's 80/20 split (they use 80% train, 20% test)

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
        from sklearn.model_selection import train_test_split
        n_total = len(X)
        all_idx = np.arange(n_total)
        idx_train, idx_temp = train_test_split(
            all_idx, test_size=val_ratio + test_ratio, random_state=seed
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=relative_test, random_state=seed
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

    # Standardize targets (fit on train only)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y[idx_train].reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y[idx_val].reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y[idx_test].reshape(-1, 1)).flatten()

    split_data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val_scaled, dtype=torch.float32).unsqueeze(1),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test_scaled": torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1),
        "y_test_original": torch.tensor(y[idx_test], dtype=torch.float32).unsqueeze(1),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "d_input": X_train.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    print(f"  Target stats (train): mean={y[idx_train].mean():.2f}, "
          f"std={y[idx_train].std():.2f}")

    # ===== 3. Pre-compute molecular graphs =====
    print("  Pre-computing 2D molecular graphs from SMILES...")
    reorder = np.concatenate([idx_train, idx_val, idx_test])

    all_mol_graphs = smiles_to_molecular_graphs(smiles_list, valid_indices)
    reordered_graphs = [all_mol_graphs[i] for i in reorder]
    mol_batch = PyGBatch.from_data_list(reordered_graphs)
    print(f"  Molecular batch: {mol_batch.num_graphs} molecules, "
          f"{mol_batch.x.shape[0]} atoms, {mol_batch.edge_index.shape[1]} bonds")

    # ===== 4. Configure ASANN =====
    config = ASANNConfig.from_task(
        task_type="regression",
        modality="molecular",
        d_input=split_data["d_input"],
        d_output=1,
        n_samples=len(split_data["X_train"]),
        device=device,
    )
    batch_size = config.recommended_batch_size

    # ===== 5. Create mini-batch dataloaders with molecule indices =====
    loaders = create_mol_dataloaders(split_data, batch_size=batch_size)
    steps_per_epoch = len(loaders["train"])
    print(f"  DataLoaders: batch_size={batch_size}, steps/epoch={steps_per_epoch}")

    # ===== 6. Create model and trainer =====
    d_input = split_data["d_input"]
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
        target_metric="rmse",
    )
    model = trainer.model
    model.set_molecular_batch(mol_batch)
    model.set_molecular_graphs(reordered_graphs)

    # ===== 7. Train =====
    max_epochs = config.recommended_max_epochs
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

    # ===== 8. Evaluate on test set using BEST model =====
    best_model_path = os.path.join(results_dir, "best_model_full.pt")
    if os.path.exists(best_model_path):
        print(f"  Loading best model from {best_model_path}")
        eval_model = torch.load(best_model_path, map_location=device, weights_only=False)
        eval_model.to(device)
        best_val = trainer._best_val_metric
        if best_val is not None:
            print(f"  Best model loaded (val rmse: {best_val:.4f})")
    else:
        print("  WARNING: best_model_full.pt not found, using final model")
        eval_model = trainer.model
        eval_model.to(device)
    eval_model.set_molecular_batch(mol_batch)
    eval_model.set_molecular_graphs(reordered_graphs)

    n_train = len(split_data["X_train"])
    n_val = len(split_data["X_val"])
    n_test = len(split_data["X_test"])
    test_mol_idx = torch.arange(n_train + n_val, n_train + n_val + n_test)
    eval_model.set_current_mol_indices(test_mol_idx.to(device))

    metrics = evaluate_model(
        eval_model, split_data["X_test"], split_data["y_test_original"],
        split_data["y_scaler"], device,
    )

    # ===== 9. Pearson/Spearman correlation =====
    from scipy.stats import pearsonr, spearmanr
    eval_model.eval()
    with torch.no_grad():
        X_test_t = split_data["X_test"].to(device)
        preds_scaled = eval_model(X_test_t).cpu().numpy()
    y_pred = split_data["y_scaler"].inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    y_true = split_data["y_test_original"].numpy().flatten()

    metrics["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    metrics["spearman_rho"] = float(spearmanr(y_true, y_pred)[0])
    metrics["split_mode"] = SPLIT_MODE
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["n_test"] = n_test
    metrics["n_molecules"] = n_train + n_val + n_test
    print(f"  RMSE (correct): {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman rho: {metrics['spearman_rho']:.4f}")

    # ===== 10. FedLG buggy evaluation (scaffold mode) =====
    if SPLIT_MODE == "scaffold":
        from tier_7.eval_all_protocols import fedlg_buggy_rmse

        # --- FedLG buggy eval on best model ---
        fedlg_min_best, fedlg_mean_best = fedlg_buggy_rmse(
            y_true, y_pred, batch_size=32, n_epochs=501
        )
        print(f"  RMSE FedLG buggy (best model): min={fedlg_min_best:.4f}")

        # --- FedLG buggy eval on final model ---
        final_model = trainer.model
        final_model.to(device)
        final_model.set_molecular_batch(mol_batch)
        final_model.set_molecular_graphs(reordered_graphs)
        final_model.set_current_mol_indices(test_mol_idx.to(device))
        final_model.eval()
        with torch.no_grad():
            final_preds_scaled = final_model(X_test_t).cpu().numpy()
        y_pred_final = split_data["y_scaler"].inverse_transform(
            final_preds_scaled.reshape(-1, 1)
        ).flatten()

        from sklearn.metrics import mean_squared_error
        final_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_final)))
        metrics["rmse_final"] = final_rmse

        fedlg_min_final, fedlg_mean_final = fedlg_buggy_rmse(
            y_true, y_pred_final, batch_size=32, n_epochs=501
        )
        print(f"  RMSE FedLG buggy (final model): min={fedlg_min_final:.4f} "
              f"(correct={final_rmse:.4f})")

        # For RMSE, lower is better -- take min across both models
        fedlg_min = min(fedlg_min_best, fedlg_min_final)
        metrics["rmse_fedlg_buggy_min"] = fedlg_min
        metrics["rmse_fedlg_buggy_mean"] = min(fedlg_mean_best, fedlg_mean_final)
        metrics["rmse_fedlg_buggy_best_model"] = fedlg_min_best
        metrics["rmse_fedlg_buggy_final_model"] = fedlg_min_final
        print(f"  RMSE FedLG buggy (min): {fedlg_min:.4f}")
        print(f"  Deflation: {metrics['rmse'] - fedlg_min:.4f}")

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
    print(f"  Results saved to {results_json_path}")

    return metrics, arch, config_to_dict(config), train_metrics


if __name__ == "__main__":
    project_root, results_base = setup_paths()
    tier_results = results_base / "tier_7"

    if len(sys.argv) > 1 and sys.argv[1] in ("scaffold", "random"):
        os.environ["FREESOLV_SPLIT"] = sys.argv[1]

    split_mode = os.environ.get("FREESOLV_SPLIT", "scaffold")
    freesolv_seed = os.environ.get("FREESOLV_SEED", "42")
    exp_dir_name = f"moleculenet_freesolv_{split_mode}_s{freesolv_seed}"

    name, metrics, arch, elapsed, status = run_experiment_wrapper(
        exp_dir_name, run_experiment, tier_results
    )
    if status != "OK":
        print(f"\nExperiment failed: {status}")
        sys.exit(1)
