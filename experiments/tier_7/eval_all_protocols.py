"""
Standalone multi-protocol evaluation for MoleculeNet benchmarks.

Given a trained ASANN model checkpoint + the same dataset, evaluates under:
  1. "correct"  - all test samples, no shuffle (honest evaluation)
  2. "buggy"    - drop_last=True, shuffle=True, max over N_TRIALS (KA-GNN's protocol)

Usage:
  python eval_all_protocols.py <checkpoint_dir> <dataset> <split_mode>

Example:
  python eval_all_protocols.py results/tier_7/moleculenet_bbbp bbbp random
  python eval_all_protocols.py results/tier_7/moleculenet_bbbp_scaffold bbbp scaffold
"""

import sys
import os
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

_exp_dir = os.path.join(os.path.dirname(__file__), "..")
_proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _exp_dir not in sys.path:
    sys.path.insert(0, _exp_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def buggy_auroc(y_true, y_probs, batch_size=128, n_trials=501, seed=42):
    """Simulate KA-GNN's buggy evaluation protocol.

    Their test DataLoader uses drop_last=True + shuffle=True,
    evaluating only batch_size samples per epoch from a randomly
    shuffled test set. They take the best AUC across all epochs.

    We simulate this by sampling n_trials random subsets of
    batch_size samples and returning the maximum AUC.
    """
    n = len(y_true)
    if n <= batch_size:
        # Can't subsample, just return normal AUC
        return float(roc_auc_score(y_true, y_probs)), 0.0

    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_trials):
        idx = rng.choice(n, size=batch_size, replace=False)
        try:
            auc = roc_auc_score(y_true[idx], y_probs[idx])
            aucs.append(auc)
        except ValueError:
            # Only one class in subset
            continue

    if not aucs:
        return 0.0, 0.0

    return float(np.max(aucs)), float(np.mean(aucs))


def fedlg_buggy_auroc(y_true, y_probs, batch_size=128, n_epochs=501, seed=42):
    """Simulate FedLG's buggy evaluation protocol.

    FedLG's bugs (from nnutils.py):
    1. shuffle=True on test DataLoader (different order each epoch)
    2. AUROC computed PER-BATCH then averaged (not on full test set)
    3. Max average-of-per-batch AUROC reported across all epochs

    This is different from KA-GNN's bug:
    - KA-GNN: subsample 128 from test, compute AUROC on subsample, take max
    - FedLG: split test into batches, compute AUROC per batch, average, take max

    Per-batch AUROC averaging is mathematically biased -- AUROC is a global
    ranking metric that should be computed on the full test set.
    """
    n = len(y_true)
    rng = np.random.RandomState(seed)
    epoch_aurocs = []

    for _ in range(n_epochs):
        # Shuffle test set (FedLG does shuffle=True on test loader)
        perm = rng.permutation(n)
        y_shuf = y_true[perm]
        p_shuf = y_probs[perm]

        # Compute per-batch AUROC and average (FedLG's protocol)
        batch_aucs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            y_batch = y_shuf[start:end]
            p_batch = p_shuf[start:end]
            # Skip batches with only one class (can't compute AUROC)
            if len(np.unique(y_batch[~np.isnan(y_batch)])) < 2:
                continue
            try:
                auc = roc_auc_score(y_batch, p_batch)
                batch_aucs.append(auc)
            except ValueError:
                continue

        if batch_aucs:
            epoch_aurocs.append(float(np.mean(batch_aucs)))

    if not epoch_aurocs:
        return 0.0, 0.0

    return float(np.max(epoch_aurocs)), float(np.mean(epoch_aurocs))


def fedlg_buggy_rmse(y_true, y_pred, batch_size=128, n_epochs=501, seed=42):
    """Simulate FedLG's buggy evaluation protocol for regression (RMSE).

    Same bugs as fedlg_buggy_auroc but for regression metrics:
    1. shuffle=True on test DataLoader (different order each epoch)
    2. RMSE computed PER-BATCH then averaged (not on full test set)
    3. Min average-of-per-batch RMSE reported across all epochs

    Per-batch RMSE averaging is biased: by Jensen's inequality,
    E[sqrt(MSE_batch)] <= sqrt(E[MSE_batch]), so averaging per-batch
    RMSEs systematically underestimates the true RMSE. Taking the
    minimum across shuffled epochs compounds this.

    Returns: (min_rmse, mean_rmse) across all simulated epochs.
    """
    from sklearn.metrics import mean_squared_error

    n = len(y_true)
    rng = np.random.RandomState(seed)
    epoch_rmses = []

    for _ in range(n_epochs):
        # Shuffle test set (FedLG does shuffle=True on test loader)
        perm = rng.permutation(n)
        y_shuf = y_true[perm]
        p_shuf = y_pred[perm]

        # Compute per-batch RMSE and average (FedLG's protocol)
        batch_rmses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            y_batch = y_shuf[start:end]
            p_batch = p_shuf[start:end]

            mse = float(mean_squared_error(y_batch, p_batch))
            batch_rmses.append(float(np.sqrt(mse)))

        if batch_rmses:
            epoch_rmses.append(float(np.mean(batch_rmses)))

    if not epoch_rmses:
        return float('inf'), float('inf')

    # For RMSE, lower is better -- so "best" = minimum
    return float(np.min(epoch_rmses)), float(np.mean(epoch_rmses))


def scaffold_split_indices(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """Murcko scaffold split matching KA-GNN exactly."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    N = len(smiles_list)
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffoldSmiles(smi, includeChirality=True)
        except Exception:
            scaffold = ""
        scaffold_to_indices[scaffold].append(i)

    scaffold_sets = sorted(scaffold_to_indices.values(),
                           key=lambda x: (len(x), x[0]), reverse=True)

    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    idx_train, idx_val, idx_test = [], [], []
    for scaffold_indices in scaffold_sets:
        if len(idx_train) + len(scaffold_indices) > train_cutoff:
            if len(idx_train) + len(idx_val) + len(scaffold_indices) > valid_cutoff:
                idx_test.extend(scaffold_indices)
            else:
                idx_val.extend(scaffold_indices)
        else:
            idx_train.extend(scaffold_indices)

    return (np.array(idx_train), np.array(idx_val), np.array(idx_test),
            len(scaffold_sets))


def random_split_indices(y, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Stratified random split."""
    all_idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        all_idx, test_size=val_ratio + test_ratio,
        random_state=seed, stratify=y
    )
    y_temp = y[idx_temp]
    relative_test = test_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=relative_test,
        random_state=seed, stratify=y_temp
    )
    return idx_train, idx_val, idx_test


def evaluate_model_all_protocols(model, X_test, y_test, mol_graphs,
                                  mol_indices, device, n_classes=2):
    """Evaluate a model under correct and buggy protocols."""
    from torch_geometric.data import Batch as PyGBatch

    model.eval()
    model.to(device)

    # Set molecular data
    if mol_graphs is not None:
        model.set_molecular_graphs(mol_graphs)
        mol_batch = PyGBatch.from_data_list(mol_graphs)
        model.set_molecular_batch(mol_batch)
    if mol_indices is not None:
        model.set_current_mol_indices(mol_indices.to(device))

    # Forward pass
    with torch.no_grad():
        X = X_test.to(device)
        logits = model(X).cpu()

    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_np = y_test.numpy()

    # 1. Correct evaluation (all samples)
    correct_auc = float(roc_auc_score(y_np, probs))

    # 2. Buggy evaluation (subsample + max)
    buggy_max_auc, buggy_mean_auc = buggy_auroc(y_np, probs)

    return {
        "correct_auroc": correct_auc,
        "buggy_max_auroc": buggy_max_auc,
        "buggy_mean_auroc": buggy_mean_auc,
        "n_test": len(y_np),
        "n_subsample": min(128, len(y_np)),
    }


if __name__ == "__main__":
    # Quick demo: evaluate buggy protocol on random predictions
    np.random.seed(42)
    y = np.array([0]*100 + [1]*100)
    probs = np.random.rand(200)

    correct = roc_auc_score(y, probs)
    buggy_max, buggy_mean = buggy_auroc(y, probs, batch_size=128, n_trials=501)

    print(f"Random predictions (200 samples):")
    print(f"  Correct AUC:    {correct:.4f}")
    print(f"  Buggy max AUC:  {buggy_max:.4f} (max of 501 draws of 128 samples)")
    print(f"  Buggy mean AUC: {buggy_mean:.4f}")
    print(f"  Inflation:      {buggy_max - correct:.4f}")
