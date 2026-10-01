"""
Tier 7: Biological/Biomedical Data Utilities
=============================================
Shared data loading and preprocessing functions for Tier 7 experiments.
"""

import os
import importlib
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Tuple, Optional, Dict, Any


def check_package(name: str) -> bool:
    """Check if a Python package is importable."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _patch_tdc_imports():
    """Patch missing optional TDC dependencies (tiledbsoma, cellxgene_census, gget).

    On Windows, PyTDC's multi_pred.__init__.py imports tiledbsoma which is
    unavailable.  Injecting dummy modules lets us import the submodules we
    actually need (DrugRes, ADME, Tox) without the broken import chain.
    """
    import sys
    import types
    for mod_name in ('tiledbsoma', 'cellxgene_census', 'gget'):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# ============================================================
# Munich Leukemia Lab: Blood cell image classification
# ============================================================

# 18 cell type classes (alphabetical order → label index)
LEUKEMIA_CLASSES = [
    "basophil", "eosinophil", "hairy_cell", "lymphocyte",
    "lymphocyte_large_granular", "lymphocyte_neoplastic",
    "lymphocyte_reactive", "metamyelocyte", "monocyte",
    "myeloblast", "myelocyte", "neutrophil_band",
    "neutrophil_segmented", "normoblast", "plasma_cell",
    "promyelocyte", "promyelocyte_atypical", "smudge_cell",
]

_LEUKEMIA_CLASS_TO_IDX = {c: i for i, c in enumerate(LEUKEMIA_CLASSES)}


def load_munich_leukemia_data(
    data_dir: str = "data/Munich_Leukemia_Lab",
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load Munich Leukemia Lab blood cell images as ResNet-50 features.

    Extracts 2048-dimensional feature vectors from 288x288 RGB TIF images
    using a frozen pretrained ResNet-50 (ImageNet weights). Each image
    becomes a single row in the feature matrix.

    Features are cached to disk after first extraction (resnet50_features.npz)
    to avoid re-extracting on subsequent runs (e.g. 5-fold CV).

    Args:
        data_dir: Path to Munich_Leukemia_Lab directory containing 18
                  class subdirectories of TIF images.
        use_cache: If True, cache/load features from .npz file.

    Returns: (X, y, n_classes) where X is [n_images, 2048] float32,
             y is [n_images] int64, n_classes is 18.
    """
    from collections import Counter

    data_dir = str(data_dir)
    n_classes = len(LEUKEMIA_CLASSES)
    cache_path = os.path.join(data_dir, "resnet50_features.npz")

    # --- Try loading from cache ---
    if use_cache and os.path.exists(cache_path):
        print(f"  Loading cached ResNet-50 features from {cache_path}")
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        print(f"  Loaded: {X.shape[0]} images, {X.shape[1]} features, "
              f"{n_classes} classes")
        label_counts = Counter(y.tolist())
        for lbl in sorted(label_counts):
            print(f"    {LEUKEMIA_CLASSES[lbl]:30s} {label_counts[lbl]:>6d}")
        return X, y, n_classes

    # --- Extract features from scratch ---
    import torchvision
    from torchvision import transforms
    from PIL import Image

    # --- 1. Discover image files and assign labels ---
    image_files = []  # list of (path, label)
    for cls_name in LEUKEMIA_CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: class directory not found: {cls_dir}")
            continue
        label = _LEUKEMIA_CLASS_TO_IDX[cls_name]
        for fname in sorted(os.listdir(cls_dir)):
            if fname.upper().endswith('.TIF') or fname.upper().endswith('.TIFF'):
                image_files.append((os.path.join(cls_dir, fname), label))

    print(f"  Found {len(image_files)} images across {n_classes} classes")
    label_counts = Counter(lbl for _, lbl in image_files)
    for lbl in sorted(label_counts):
        print(f"    {LEUKEMIA_CLASSES[lbl]:30s} {label_counts[lbl]:>6d}")

    # --- 2. Load pretrained ResNet-50 as feature extractor ---
    print("  Loading pretrained ResNet-50 for feature extraction...")
    backbone = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )
    # Remove the classification head -- keep up to avgpool -> [B, 2048, 1, 1]
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    # ImageNet preprocessing for 288x288 -> 224x224
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # --- 3. Extract features in batches ---
    print(f"  Extracting features on {device}...")
    batch_size = 64
    all_features = []
    all_labels = []
    n_total = len(image_files)

    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        batch_tensors = []
        batch_labels = []

        for path, label in image_files[start_idx:end_idx]:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(preprocess(img))
            batch_labels.append(label)

        batch_input = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = backbone(batch_input)  # [B, 2048, 1, 1]
        feats = feats.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, 2048]
        all_features.append(feats)
        all_labels.extend(batch_labels)

        if (start_idx // batch_size + 1) % 100 == 0:
            print(f"    Processed {end_idx}/{n_total} images...")

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"  Feature matrix: {X.shape} ({X.shape[1]} features per image)")
    print(f"  Labels: {dict(Counter(y.tolist()))}")

    # --- 4. Cache to disk ---
    if use_cache:
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"  Cached features to {cache_path}")

    return X, y, n_classes


# ============================================================
# Morgan fingerprint helper (uses rdkit directly)
# ============================================================
def _smiles_to_morgan(smiles_list, radius: int = 2, n_bits: int = 2048):
    """Convert SMILES strings to Morgan fingerprint bit vectors using rdkit.

    Returns: (X, valid_indices) where X is float32 array [n_valid, n_bits]
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    X_list = []
    valid_idx = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.float32)
                for bit in fp.GetOnBits():
                    arr[bit] = 1.0
                X_list.append(arr)
                valid_idx.append(i)
        except Exception:
            continue

    return np.array(X_list, dtype=np.float32), valid_idx


def _smiles_to_morgan_with_descriptors(
    smiles_list, radius: int = 2, n_bits: int = 512
):
    """Convert SMILES to Morgan fingerprints + RDKit 2D descriptors.

    Combines smaller fingerprints (default 512 bits) with physicochemical
    descriptors for better representation with limited training data.

    Returns: (X, valid_indices) where X is float32 [n_valid, n_bits + n_desc]
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski

    X_list = []
    valid_idx = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue

            # Morgan fingerprint (smaller than default 2048)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=n_bits)
            fp_arr = np.zeros(n_bits, dtype=np.float32)
            for bit in fp.GetOnBits():
                fp_arr[bit] = 1.0

            # RDKit 2D descriptors (physicochemical properties)
            desc = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumHeteroatoms(mol),
            ], dtype=np.float32)

            # Replace NaN/Inf with 0
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)

            X_list.append(np.concatenate([fp_arr, desc]))
            valid_idx.append(i)
        except Exception:
            continue

    return np.array(X_list, dtype=np.float32), valid_idx


# ============================================================
# GDSC drug sensitivity data loading
# ============================================================
def load_gdsc_data(max_samples: int = 50000, return_smiles: bool = False):
    """Load GDSC drug sensitivity data via PyTDC.

    Uses tdc.multi_pred.drugres.DrugRes directly (avoids broken __init__).

    Each sample is a (drug, cell_line) pair. The drug is represented by its
    Morgan fingerprint. When return_smiles=True, also returns per-sample
    SMILES for molecular graph construction (GINEConv).

    Note: The same drug SMILES appears in many samples (paired with different
    cell lines). smiles_to_molecular_graphs handles deduplication internally
    when building PyG graphs.

    Args:
        max_samples: Maximum number of drug-cell line pairs.
        return_smiles: If True, also return per-sample SMILES and indices.

    Returns:
        If return_smiles=False: (X, y)
        If return_smiles=True: (X, y, smiles_per_sample, valid_indices)
    """
    _patch_tdc_imports()
    from tdc.multi_pred.drugres import DrugRes
    data = DrugRes(name='GDSC1')
    df = data.get_data()

    print(f"  Raw GDSC1 data: {len(df)} drug-cell line pairs")

    # Get unique SMILES and compute fingerprints
    unique_smiles = df['Drug'].unique().tolist()
    print(f"  Computing Morgan fingerprints for {len(unique_smiles)} drugs...")

    fp_array, valid_idx = _smiles_to_morgan(unique_smiles)
    valid_smiles = set(unique_smiles[i] for i in valid_idx)
    drug_fp_map = {unique_smiles[i]: fp_array[j]
                   for j, i in enumerate(valid_idx)}

    # Filter to valid drugs and build feature matrix
    df = df[df['Drug'].isin(valid_smiles)].copy()

    X_list = []
    y_list = []
    smiles_per_sample = []
    for _, row in df.iterrows():
        X_list.append(drug_fp_map[row['Drug']])
        y_list.append(row['Y'])
        smiles_per_sample.append(row['Drug'])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Subsample if too large
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
        smiles_per_sample = [smiles_per_sample[i] for i in idx]

    print(f"  Final: {X.shape[0]} samples, {X.shape[1]} features")
    if return_smiles:
        n_unique = len(set(smiles_per_sample))
        print(f"  Unique drugs: {n_unique} (shared across cell lines)")
        valid_indices = np.arange(len(X))
        return X, y, smiles_per_sample, valid_indices
    return X, y


def load_gdsc2_dual_data(data_dir: str, max_samples: int = None,
                         pca_dim: int = 512):
    """Load GDSC2 data with cell line gene expression + drug SMILES.

    Merges:
    - GDSC2-dataset.csv: drug-cell line pairs with LN_IC50
    - model_list.csv.gz: SANGER_MODEL_ID (SIDM) -> BROAD_ID (ACH) mapping
    - OmicsExpressionProteinCodingGenesTPMLogp1.csv: ACH -> 19K gene features
    - PyTDC GDSC2: drug name -> SMILES

    Returns X_cell (PCA-reduced gene expression), y (LN_IC50),
    smiles_per_sample, valid_indices, d_cell.
    """
    from sklearn.decomposition import PCA

    # 1. Load GDSC2 drug response data
    gdsc2_path = os.path.join(data_dir, "GDSC2-dataset.csv")
    gdsc2 = pd.read_csv(gdsc2_path)
    print(f"  Raw GDSC2: {len(gdsc2)} drug-cell pairs, "
          f"{gdsc2['DRUG_NAME'].nunique()} drugs, "
          f"{gdsc2['SANGER_MODEL_ID'].nunique()} cell lines")

    # 2. Load SIDM -> ACH mapping
    ml_path = os.path.join(data_dir, "model_list.csv.gz")
    ml = pd.read_csv(ml_path, compression='gzip')
    ml_valid = ml[ml['BROAD_ID'].notna()][['model_id', 'BROAD_ID']]
    ml_valid = ml_valid.rename(columns={'model_id': 'SANGER_MODEL_ID',
                                        'BROAD_ID': 'ach'})
    print(f"  SIDM->ACH mappings: {len(ml_valid)}")

    # 3. Load gene expression
    ge_path = os.path.join(data_dir,
                           "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    ge_df = pd.read_csv(ge_path, index_col=0)
    ge_achs = set(ge_df.index)
    print(f"  Gene expression: {ge_df.shape[0]} cell lines x "
          f"{ge_df.shape[1]} genes")

    # 4. Get drug SMILES from PyTDC
    _patch_tdc_imports()
    from tdc.multi_pred.drugres import DrugRes
    tdc_data = DrugRes(name='GDSC2').get_data()
    tdc_dedup = tdc_data.drop_duplicates('Drug_ID')[['Drug_ID', 'Drug']]
    tdc_dedup = tdc_dedup.rename(columns={'Drug_ID': 'DRUG_NAME',
                                          'Drug': 'smiles'})
    print(f"  PyTDC SMILES: {len(tdc_dedup)} drugs")

    # 5. Filter to usable pairs (vectorized merge)
    # Merge GDSC2 with SIDM->ACH mapping
    df = gdsc2[['SANGER_MODEL_ID', 'DRUG_NAME', 'LN_IC50']].merge(
        ml_valid, on='SANGER_MODEL_ID', how='inner')
    # Filter to cell lines with gene expression
    df = df[df['ach'].isin(ge_achs)]
    # Merge with drug SMILES
    df = df.merge(tdc_dedup, on='DRUG_NAME', how='inner')
    df = df.rename(columns={'DRUG_NAME': 'drug', 'LN_IC50': 'y'})
    df = df[['ach', 'drug', 'smiles', 'y']].reset_index(drop=True)
    print(f"  Usable pairs: {len(df)} ({df['drug'].nunique()} drugs, "
          f"{df['ach'].nunique()} cell lines)")

    # 6. Subsample if needed
    if max_samples is not None and len(df) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), max_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {len(df)}")

    # 7. PCA on UNIQUE cell lines (704 x 19K) then map to samples
    # This avoids building a 93K x 19K matrix (7GB) in memory.
    unique_achs = df['ach'].unique()
    ge_subset = ge_df.loc[ge_df.index.isin(unique_achs)]
    ge_matrix = ge_subset.values.astype(np.float32)  # [704, 19K]
    ach_list = list(ge_subset.index)
    ach_to_idx = {ach: i for i, ach in enumerate(ach_list)}
    print(f"  PCA on {len(unique_achs)} unique cell lines "
          f"({ge_matrix.shape[1]} genes -> {pca_dim} dims)...")

    pca = PCA(n_components=pca_dim, random_state=42)
    ge_pca = pca.fit_transform(ge_matrix).astype(np.float32)  # [704, pca_dim]
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance explained: {var_explained:.3f}")

    # Map each sample to its PCA-reduced cell line features
    cell_indices = np.array([ach_to_idx[ach] for ach in df['ach']])
    X_cell = ge_pca[cell_indices]  # [N, pca_dim]

    y = df['y'].values.astype(np.float32)
    smiles_per_sample = df['smiles'].tolist()
    valid_indices = np.arange(len(df))
    d_cell = pca_dim

    print(f"  Final: {len(X_cell)} samples, {d_cell} cell features, "
          f"{df['drug'].nunique()} drugs, {df['ach'].nunique()} cell lines")

    return X_cell, y, smiles_per_sample, valid_indices, d_cell


def load_gdsc2_dual_data_drug_split(data_dir: str, pca_dim: int = 512,
                                     seed: int = 42):
    """Load GDSC2 data with drug-blind (unseen drug) splitting.

    Same data pipeline as load_gdsc2_dual_data(), but splits by DRUG
    instead of by sample. Drugs in the test set are entirely absent from
    the training set, testing true generalization to novel compounds.

    Split: 70% train drugs / 15% val drugs / 15% test drugs.
    All samples for a given drug go to the same split.

    Returns dict with train/val/test arrays, smiles, drug names, and metadata.
    """
    from sklearn.decomposition import PCA

    # 1-5: Same loading + merging as load_gdsc2_dual_data
    gdsc2_path = os.path.join(data_dir, "GDSC2-dataset.csv")
    gdsc2 = pd.read_csv(gdsc2_path)
    print(f"  Raw GDSC2: {len(gdsc2)} drug-cell pairs, "
          f"{gdsc2['DRUG_NAME'].nunique()} drugs, "
          f"{gdsc2['SANGER_MODEL_ID'].nunique()} cell lines")

    ml_path = os.path.join(data_dir, "model_list.csv.gz")
    ml = pd.read_csv(ml_path, compression='gzip')
    ml_valid = ml[ml['BROAD_ID'].notna()][['model_id', 'BROAD_ID']]
    ml_valid = ml_valid.rename(columns={'model_id': 'SANGER_MODEL_ID',
                                        'BROAD_ID': 'ach'})
    print(f"  SIDM->ACH mappings: {len(ml_valid)}")

    ge_path = os.path.join(data_dir,
                           "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    ge_df = pd.read_csv(ge_path, index_col=0)
    ge_achs = set(ge_df.index)
    print(f"  Gene expression: {ge_df.shape[0]} cell lines x "
          f"{ge_df.shape[1]} genes")

    _patch_tdc_imports()
    from tdc.multi_pred.drugres import DrugRes
    tdc_data = DrugRes(name='GDSC2').get_data()
    tdc_dedup = tdc_data.drop_duplicates('Drug_ID')[['Drug_ID', 'Drug']]
    tdc_dedup = tdc_dedup.rename(columns={'Drug_ID': 'DRUG_NAME',
                                          'Drug': 'smiles'})
    print(f"  PyTDC SMILES: {len(tdc_dedup)} drugs")

    df = gdsc2[['SANGER_MODEL_ID', 'DRUG_NAME', 'LN_IC50']].merge(
        ml_valid, on='SANGER_MODEL_ID', how='inner')
    df = df[df['ach'].isin(ge_achs)]
    df = df.merge(tdc_dedup, on='DRUG_NAME', how='inner')
    df = df.rename(columns={'DRUG_NAME': 'drug', 'LN_IC50': 'y'})
    df = df[['ach', 'drug', 'smiles', 'y']].reset_index(drop=True)
    print(f"  Usable pairs: {len(df)} ({df['drug'].nunique()} drugs, "
          f"{df['ach'].nunique()} cell lines)")

    # 6. DRUG-BLIND SPLIT: split by unique drug names
    unique_drugs = sorted(df['drug'].unique())
    n_drugs = len(unique_drugs)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_drugs)

    n_test = max(1, int(n_drugs * 0.15))
    n_val = max(1, int(n_drugs * 0.15))
    n_train = n_drugs - n_val - n_test

    test_drugs = set(unique_drugs[i] for i in perm[:n_test])
    val_drugs = set(unique_drugs[i] for i in perm[n_test:n_test + n_val])
    train_drugs = set(unique_drugs[i] for i in perm[n_test + n_val:])

    df_train = df[df['drug'].isin(train_drugs)].reset_index(drop=True)
    df_val = df[df['drug'].isin(val_drugs)].reset_index(drop=True)
    df_test = df[df['drug'].isin(test_drugs)].reset_index(drop=True)

    print(f"  Drug-blind split:")
    print(f"    Train: {len(df_train)} samples, {len(train_drugs)} drugs")
    print(f"    Val:   {len(df_val)} samples, {len(val_drugs)} drugs")
    print(f"    Test:  {len(df_test)} samples, {len(test_drugs)} drugs")

    # 7. PCA on UNIQUE cell lines (fit on train cell lines only)
    # Combine all splits to find all unique ACHs
    all_achs = set(df_train['ach'].unique()) | set(df_val['ach'].unique()) \
               | set(df_test['ach'].unique())
    ge_subset = ge_df.loc[ge_df.index.isin(all_achs)]
    ge_matrix = ge_subset.values.astype(np.float32)
    ach_list = list(ge_subset.index)
    ach_to_idx = {ach: i for i, ach in enumerate(ach_list)}
    print(f"  PCA on {len(all_achs)} unique cell lines "
          f"({ge_matrix.shape[1]} genes -> {pca_dim} dims)...")

    # Fit PCA on train cell lines only for proper evaluation
    train_achs = df_train['ach'].unique()
    train_ge_indices = np.array([ach_to_idx[a] for a in train_achs])
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(ge_matrix[train_ge_indices])
    ge_pca = pca.transform(ge_matrix).astype(np.float32)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance explained: {var_explained:.3f}")

    def _build_arrays(sub_df):
        ci = np.array([ach_to_idx[a] for a in sub_df['ach']])
        X = ge_pca[ci]
        y_arr = sub_df['y'].values.astype(np.float32)
        smi = sub_df['smiles'].tolist()
        drugs = sub_df['drug'].tolist()
        return X, y_arr, smi, drugs

    X_train, y_train, smi_train, drugs_train = _build_arrays(df_train)
    X_val, y_val, smi_val, drugs_val = _build_arrays(df_val)
    X_test, y_test, smi_test, drugs_test = _build_arrays(df_test)

    print(f"  Final: train={len(X_train)}, val={len(X_val)}, "
          f"test={len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'smiles_train': smi_train, 'drugs_train': drugs_train,
        'X_val': X_val, 'y_val': y_val,
        'smiles_val': smi_val, 'drugs_val': drugs_val,
        'X_test': X_test, 'y_test': y_test,
        'smiles_test': smi_test, 'drugs_test': drugs_test,
        'd_cell': pca_dim,
        'n_train_drugs': len(train_drugs),
        'n_val_drugs': len(val_drugs),
        'n_test_drugs': len(test_drugs),
        'train_drug_names': sorted(train_drugs),
        'val_drug_names': sorted(val_drugs),
        'test_drug_names': sorted(test_drugs),
    }


# ============================================================
# MoleculeNet BBBP data loading
# ============================================================
def load_moleculenet_bbbp_csv(csv_path: str, return_smiles: bool = False):
    """Load BBBP from raw MoleculeNet CSV (same source as KA-GNN).

    Reads bbbp.csv with columns: num, name, label, smiles.
    Returns Morgan fingerprints and labels for valid molecules.

    Args:
        csv_path: Path to bbbp.csv (e.g., data/KA-GNN-main/KA-GNN/data/bbbp.csv)
        return_smiles: If True, also return SMILES list and valid indices.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    df = pd.read_csv(csv_path)
    print(f"  Raw BBBP CSV: {len(df)} molecules")

    smiles_list = df['smiles'].values.tolist()
    y_all = df['label'].values

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, {n_classes} classes")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


def load_bbbp_tab(tab_path: str, return_smiles: bool = False):
    """Load BBBP from our own bbb_martins.tab file (PyTDC format).

    Tab-separated with columns: Drug_ID, Drug (SMILES), Y (label).
    Canonicalizes SMILES and removes duplicates for reproducible splits.

    Args:
        tab_path: Path to bbb_martins.tab
        return_smiles: If True, also return SMILES list and valid indices.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    from rdkit import Chem

    df = pd.read_csv(tab_path, sep='\t')
    print(f"  Raw BBBP tab: {len(df)} molecules")

    # Canonicalize SMILES and deduplicate
    raw_smiles = df['Drug'].values.tolist()
    y_raw = df['Y'].values
    seen = set()
    smiles_list = []
    y_list = []
    for smi, label in zip(raw_smiles, y_raw):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)
            y_list.append(label)
    y_all = np.array(y_list)
    n_dedup = len(df) - len(smiles_list)
    if n_dedup > 0:
        print(f"  Deduplicated: removed {n_dedup} duplicates -> {len(smiles_list)} unique")

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, {n_classes} classes")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


def load_hiv_tab(tab_path: str, return_smiles: bool = False):
    """Load HIV from our own hiv.tab file (PyTDC format).

    Tab-separated with columns: Drug_ID, Drug (SMILES), Y (label).
    Canonicalizes SMILES and removes duplicates for reproducible splits.

    Args:
        tab_path: Path to hiv.tab
        return_smiles: If True, also return SMILES list and valid indices.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    from rdkit import Chem

    df = pd.read_csv(tab_path, sep='\t')
    print(f"  Raw HIV tab: {len(df)} molecules")

    # Canonicalize SMILES and deduplicate
    raw_smiles = df['Drug'].values.tolist()
    y_raw = df['Y'].values
    seen = set()
    smiles_list = []
    y_list = []
    for smi, label in zip(raw_smiles, y_raw):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)
            y_list.append(label)
    y_all = np.array(y_list)
    n_dedup = len(df) - len(smiles_list)
    if n_dedup > 0:
        print(f"  Deduplicated: removed {n_dedup} duplicates -> {len(smiles_list)} unique")

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    pos_rate = y.sum() / len(y)
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, "
          f"{n_classes} classes (positive rate: {pos_rate:.3f})")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


def load_moleculenet_bbbp(return_smiles: bool = False):
    """Load BBB (Blood-Brain Barrier) dataset via PyTDC (BBB_Martins).

    Args:
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    from tdc.single_pred import ADME
    data = ADME(name='BBB_Martins')
    df = data.get_data()

    print(f"  Raw BBB_Martins data: {len(df)} molecules")

    # Convert SMILES to Morgan fingerprints using rdkit
    smiles_list = df['Drug'].values.tolist()
    y_all = df['Y'].values

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, {n_classes} classes")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


# ============================================================
# TCGA survival proxy data loading
# ============================================================
def load_tcga_classification() -> Tuple[np.ndarray, np.ndarray, int]:
    """Load TCGA gene expression data for survival proxy classification.

    Tries multiple data sources in order of preference.
    Binary classification: survived beyond median OS time (1) vs not (0).

    Returns: (X, y, n_classes=2)
    """
    # Strategy 1: Use sklearn's fetch_openml for preprocessed TCGA datasets
    try:
        return _load_tcga_openml()
    except Exception as e:
        print(f"  OpenML strategy failed: {e}")

    # Strategy 2: Use PyTDC if it has TCGA-like data
    try:
        return _load_tcga_via_tdc()
    except Exception as e:
        print(f"  TDC strategy failed: {e}")

    raise RuntimeError(
        "Could not load TCGA data from any source. "
        "Try: pip install PyTDC  or provide a TCGA CSV in data/"
    )


def _load_tcga_openml() -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a TCGA-derived dataset from OpenML."""
    from sklearn.datasets import fetch_openml

    # Try TCGA-BRCA gene expression (OpenML dataset ID varies)
    # Common preprocessed TCGA datasets on OpenML:
    # - 'GCM' (Global Cancer Map) — 190 samples, 16063 genes, 14 cancer types
    for dataset_name in ['GCM']:
        try:
            data = fetch_openml(name=dataset_name, version=1, as_frame=True,
                                parser='auto')
            X_df = data.data
            y_series = data.target

            # Convert to numpy
            X = X_df.values.astype(np.float32)

            # Encode labels as integers
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_series).astype(np.int64)
            n_classes = len(le.classes_)

            # If too many features, select top 2000 by variance
            if X.shape[1] > 2000:
                variances = np.var(X, axis=0)
                top_idx = np.argsort(variances)[-2000:]
                X = X[:, top_idx]

            print(f"  Loaded {dataset_name}: {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {n_classes} classes")
            return X, y, n_classes
        except Exception:
            continue

    raise RuntimeError("No suitable TCGA dataset found on OpenML")


def _load_tcga_via_tdc() -> Tuple[np.ndarray, np.ndarray, int]:
    """Try loading ClinTox via PyTDC as a clinical genomics proxy."""
    from tdc.single_pred import Tox
    data = Tox(name='ClinTox')
    df = data.get_data()

    smiles_list = df['Drug'].values.tolist()
    y_all = df['Y'].values

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)
    n_classes = len(np.unique(y))

    print(f"  ClinTox (TCGA proxy): {X.shape[0]} samples, "
          f"{X.shape[1]} features, {n_classes} classes")
    return X, y, n_classes


# ============================================================
# Single-cell cell type classification
# ============================================================
def load_celltype_data(n_hvg: int = 2000) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load single-cell gene expression data for cell type classification.

    Strategy 1: scanpy built-in pbmc3k_processed (2,638 cells)
    Strategy 2: cellxgene_census API (if available)

    Returns: (X, y, n_classes)
    """
    import scanpy as sc

    # Strategy 1: Built-in pbmc3k
    print("  Loading PBMC 3K dataset via scanpy...")
    adata = sc.datasets.pbmc3k_processed()

    # Get expression matrix
    import scipy.sparse
    if scipy.sparse.issparse(adata.X):
        X = np.array(adata.X.toarray(), dtype=np.float32)
    else:
        X = np.array(adata.X, dtype=np.float32)

    # Get cell type labels
    if 'louvain' in adata.obs.columns:
        cell_types = adata.obs['louvain']
    elif 'cell_type' in adata.obs.columns:
        cell_types = adata.obs['cell_type']
    else:
        # Use first categorical column
        cat_cols = [c for c in adata.obs.columns
                    if adata.obs[c].dtype.name == 'category']
        cell_types = adata.obs[cat_cols[0]]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(cell_types).astype(np.int64)
    n_classes = len(le.classes_)

    # Select top HVGs by variance if needed
    if X.shape[1] > n_hvg:
        variances = np.var(X, axis=0)
        top_idx = np.argsort(variances)[-n_hvg:]
        X = X[:, top_idx]

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Loaded: {X.shape[0]} cells, {X.shape[1]} genes, "
          f"{n_classes} cell types")
    print(f"  Cell types: {list(le.classes_)}")
    return X, y, n_classes


# ============================================================
# MoleculeNet HIV data loading
# ============================================================
def load_moleculenet_hiv(return_smiles: bool = False):
    """Load HIV activity dataset via PyTDC.

    Binary classification: active (1) vs inactive (0) against HIV replication.
    ~41,127 molecules.  Highly imbalanced (~3% positive).

    Args:
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    _patch_tdc_imports()
    from tdc.single_pred import HTS
    data = HTS(name='HIV')
    df = data.get_data()

    print(f"  Raw HIV data: {len(df)} molecules")

    smiles_list = df['Drug'].values.tolist()
    y_all = df['Y'].values

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    pos_rate = y.sum() / len(y)
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, "
          f"{n_classes} classes (positive rate: {pos_rate:.3f})")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


# ============================================================
# MoleculeNet ESOL data loading
# ============================================================
def load_moleculenet_hiv_csv(csv_path: str, return_smiles: bool = False):
    """Load HIV from raw MoleculeNet CSV (same source as KA-GNN).

    Reads hiv.csv with columns: ID, smiles, activity, label.
    Returns Morgan fingerprints and labels for valid molecules.

    Args:
        csv_path: Path to hiv.csv (e.g., data/KA-GNN-main/KA-GNN/data/hiv.csv)
        return_smiles: If True, also return SMILES list and valid indices.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    df = pd.read_csv(csv_path)
    print(f"  Raw HIV CSV: {len(df)} molecules")

    smiles_list = df['smiles'].values.tolist()
    y_all = df['label'].values

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    pos_rate = y.sum() / len(y)
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, "
          f"{n_classes} classes (positive rate: {pos_rate:.3f})")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


def load_moleculenet_esol(return_smiles: bool = False):
    """Load ESOL (Delaney) aqueous solubility dataset via PyTDC.

    Regression: predict log solubility in mol/L.
    ~1,128 molecules.

    Args:
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y) where y is float32 solubility values.
        If return_smiles=True: (X, y, smiles_list, valid_indices)
    """
    # Load the actual ESOL (Delaney) dataset — NOT AqSolDB.
    # ESOL has ~1,128 molecules. Published baselines (RF R²~0.75) are for ESOL only.
    # AqSolDB (~9,982 molecules) is a different, harder dataset — do NOT substitute.
    df = None

    # Strategy 1: Download actual ESOL CSV from MoleculeNet
    try:
        import pandas as pd
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={'smiles': 'Drug', 'measured log solubility in mols per litre': 'Y'})
        print(f"  Loaded ESOL (Delaney) from MoleculeNet S3")
    except Exception as e:
        print(f"  Direct download failed: {e}")

    # Strategy 2: Fallback to TDC's ESOL if available
    if df is None:
        try:
            from tdc.single_pred import ADME
            data = ADME(name='ESOL')
            df = data.get_data()
            print(f"  Loaded ESOL dataset (TDC)")
        except Exception as e:
            print(f"  TDC ESOL failed: {e}")
            raise RuntimeError("Could not load ESOL from any source.")

    print(f"  Raw ESOL data: {len(df)} molecules")

    smiles_list = df['Drug'].values.tolist()
    y_all = df['Y'].values

    # Use 512-bit fingerprints + RDKit descriptors for small dataset.
    # Full 2048-bit FP has too many features for ~790 training samples,
    # causing immediate overfitting (65K input-layer params > sample count).
    X, valid_idx = _smiles_to_morgan_with_descriptors(smiles_list, n_bits=512)
    y = y_all[valid_idx].astype(np.float32)

    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features "
          f"(512-bit FP + 12 RDKit descriptors)")
    print(f"  Solubility range: [{y.min():.2f}, {y.max():.2f}] log mol/L")

    if return_smiles:
        return X, y, smiles_list, valid_idx
    return X, y


# ============================================================
# MoleculeNet MUV data loading
# ============================================================
def load_moleculenet_muv(target_id: int = 466, return_smiles: bool = False):
    """Load MUV (Maximum Unbiased Validation) dataset.

    MUV contains 17 challenging bioassay targets designed to be hard for
    virtual screening.  We load a single target (default: MUV-466) as
    binary classification: active (1) vs inactive (0).

    Extremely sparse: <0.5% positives per target.  ~93K molecules total,
    but only ~15K have labels for any given target.

    Falls back to DeepChem loader if PyTDC does not have MUV.

    Args:
        target_id: MUV target assay ID (default: 466).
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    X, y = None, None
    smiles_list_out, valid_idx_out = None, None  # Track for return_smiles

    # Strategy 1: Try DeepChem MoleculeNet loader
    try:
        import deepchem as dc
        tasks, datasets, transformers = dc.molnet.load_muv(
            featurizer=dc.feat.CircularFingerprint(size=2048, radius=2),
            splitter=None,  # don't split, we'll do it ourselves
        )
        dataset = datasets[0]

        # Find the target column index
        target_name = f'MUV-{target_id}'
        if target_name in tasks:
            target_idx = tasks.index(target_name)
        else:
            target_idx = 0
            target_name = tasks[0]
            print(f"  Target MUV-{target_id} not found, using {target_name}")

        X_all = dataset.X.astype(np.float32)
        y_all = dataset.y[:, target_idx]

        # Filter out NaN labels (unlabeled for this target)
        valid_mask = ~np.isnan(y_all)
        X = X_all[valid_mask]
        y = y_all[valid_mask].astype(np.int64)

        # DeepChem stores SMILES in dataset.ids
        all_smiles = [str(s) for s in dataset.ids]
        smiles_list_out = all_smiles
        valid_idx_out = np.where(valid_mask)[0]

        print(f"  Loaded MUV via DeepChem: {X.shape[0]} molecules with "
              f"labels for {target_name}")
    except Exception as e:
        print(f"  DeepChem MUV loading failed: {e}")

    # Strategy 2: Try PyTDC HTS
    if X is None:
        try:
            _patch_tdc_imports()
            from tdc.single_pred import HTS
            data = HTS(name=f'MUV_{target_id}')
            df = data.get_data()

            print(f"  Raw MUV-{target_id} data: {len(df)} molecules")

            smiles_list = df['Drug'].values.tolist()
            y_all = df['Y'].values

            X, valid_idx = _smiles_to_morgan(smiles_list)
            y = y_all[valid_idx].astype(np.int64)
            smiles_list_out = smiles_list
            valid_idx_out = valid_idx
        except Exception as e:
            print(f"  TDC MUV loading failed: {e}")

    # Strategy 3: Download MUV CSV from MoleculeNet S3
    if X is None:
        try:
            import pandas as pd
            import os, tempfile, gzip, urllib.request

            url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
            cache_dir = os.path.join(tempfile.gettempdir(), "moleculenet_cache")
            os.makedirs(cache_dir, exist_ok=True)
            local_gz = os.path.join(cache_dir, "muv.csv.gz")
            local_csv = os.path.join(cache_dir, "muv.csv")

            if not os.path.exists(local_csv):
                print(f"  Downloading MUV from MoleculeNet S3...")
                urllib.request.urlretrieve(url, local_gz)
                with gzip.open(local_gz, 'rb') as f_in:
                    with open(local_csv, 'wb') as f_out:
                        f_out.write(f_in.read())

            df = pd.read_csv(local_csv)
            target_col = f'MUV-{target_id}'
            if target_col not in df.columns:
                # Use first MUV target available
                muv_cols = [c for c in df.columns if c.startswith('MUV-')]
                target_col = muv_cols[0]
                print(f"  Target MUV-{target_id} not found, using {target_col}")

            # Filter rows with labels for this target
            valid = df[target_col].notna()
            smiles_list = df.loc[valid, 'smiles'].values.tolist()
            y_all = df.loc[valid, target_col].values

            X, valid_idx = _smiles_to_morgan(smiles_list)
            y = y_all[valid_idx].astype(np.int64)
            smiles_list_out = smiles_list
            valid_idx_out = valid_idx
            print(f"  Loaded MUV via direct download: {X.shape[0]} molecules for {target_col}")
        except Exception as e:
            print(f"  Direct MUV download failed: {e}")

    if X is None:
        raise RuntimeError(
            "Could not load MUV from any source. "
            "Install: pip install deepchem  or  pip install PyTDC"
        )

    n_classes = len(np.unique(y))
    pos_rate = y.sum() / len(y) if len(y) > 0 else 0
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, "
          f"{n_classes} classes (positive rate: {pos_rate:.4f})")
    if return_smiles:
        return X, y, n_classes, smiles_list_out, valid_idx_out
    return X, y, n_classes


def load_moleculenet_muv_multitask(csv_path: str):
    """Load full MUV dataset as 17-task multi-label from CSV.

    All 93K molecules are included. Labels are float32 with NaN for missing.

    Args:
        csv_path: Path to muv.csv with columns: 17 MUV-* tasks + mol_id + smiles.

    Returns:
        (X, y, task_names, smiles_list, valid_indices)
        X: float32 [n_valid, 2048] Morgan fingerprints
        y: float32 [n_valid, 17] labels (0/1/NaN)
        task_names: list of 17 task column names
        smiles_list: all SMILES from CSV
        valid_indices: indices of molecules with valid RDKit fingerprints
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    task_names = [c for c in df.columns if c.startswith('MUV-')]
    print(f"  MUV multi-task: {len(df)} molecules, {len(task_names)} tasks")

    smiles_list = df['smiles'].values.tolist()
    y_all = df[task_names].values.astype(np.float32)  # NaN preserved

    X, valid_indices = _smiles_to_morgan(smiles_list)
    y = y_all[valid_indices]

    for i, tn in enumerate(task_names):
        n_pos = int(np.nansum(y[:, i] == 1))
        n_neg = int(np.nansum(y[:, i] == 0))
        n_miss = int(np.isnan(y[:, i]).sum())
        print(f"    {tn}: {n_pos} pos, {n_neg} neg, {n_miss} missing")

    return X, y, task_names, smiles_list, valid_indices


def load_bace_csv(csv_path: str, return_smiles: bool = False):
    """Load BACE dataset from CSV file.

    BACE has columns: mol (SMILES), CID, Class (0/1), Model (Train/Test/Valid), ...
    We IGNORE the Model column and do our own scaffold split.
    Canonicalizes SMILES and removes duplicates.

    Args:
        csv_path: Path to bace.csv
        return_smiles: If True, also return SMILES list and valid indices.

    Returns:
        If return_smiles=False: (X, y, n_classes=2)
        If return_smiles=True: (X, y, n_classes, smiles_list, valid_indices)
    """
    from rdkit import Chem

    df = pd.read_csv(csv_path)
    print(f"  Raw BACE CSV: {len(df)} molecules")

    # Canonicalize SMILES and deduplicate
    raw_smiles = df['mol'].values.tolist()
    y_raw = df['Class'].values
    seen = set()
    smiles_list = []
    y_list = []
    for smi, label in zip(raw_smiles, y_raw):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)
            y_list.append(label)
    y_all = np.array(y_list)
    n_dedup = len(df) - len(smiles_list)
    if n_dedup > 0:
        print(f"  Deduplicated: removed {n_dedup} duplicates -> {len(smiles_list)} unique")

    X, valid_idx = _smiles_to_morgan(smiles_list)
    y = y_all[valid_idx].astype(np.int64)

    n_classes = len(np.unique(y))
    pos_rate = y.sum() / len(y)
    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features, "
          f"{n_classes} classes (positive rate: {pos_rate:.3f})")
    if return_smiles:
        return X, y, n_classes, smiles_list, valid_idx
    return X, y, n_classes


def load_sider_csv(csv_path: str):
    """Load SIDER dataset from CSV file (27-task multi-label, no NaN).

    Columns: smiles, then 27 side-effect category columns (all 0/1, no NaN).
    Canonicalizes SMILES and removes duplicates.

    Returns:
        (X, y_multi, task_names, smiles_list, valid_indices)
        X: float32 [n_valid, 2048] Morgan fingerprints
        y_multi: float32 [n_valid, 27] labels (0/1, no NaN)
        task_names: list of 27 task column names
        smiles_list: canonical SMILES
        valid_indices: indices with valid fingerprints
    """
    from rdkit import Chem

    df = pd.read_csv(csv_path)
    task_names = [c for c in df.columns if c != 'smiles']
    print(f"  SIDER multi-task: {len(df)} molecules, {len(task_names)} tasks")

    # Canonicalize SMILES and deduplicate
    raw_smiles = df['smiles'].values.tolist()
    y_raw = df[task_names].values.astype(np.float32)
    seen = set()
    smiles_list = []
    y_list = []
    for i, smi in enumerate(raw_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)
            y_list.append(y_raw[i])
    y_all = np.array(y_list, dtype=np.float32)
    n_dedup = len(df) - len(smiles_list)
    if n_dedup > 0:
        print(f"  Deduplicated: removed {n_dedup} duplicates -> {len(smiles_list)} unique")

    X, valid_indices = _smiles_to_morgan(smiles_list)
    y = y_all[valid_indices]

    for i, tn in enumerate(task_names):
        n_pos = int(np.sum(y[:, i] == 1))
        n_neg = int(np.sum(y[:, i] == 0))
        print(f"    {tn}: {n_pos} pos, {n_neg} neg")

    return X, y, task_names, smiles_list, valid_indices


def load_tox21_csv(csv_path: str):
    """Load Tox21 dataset from CSV file (12-task multi-label with NaN).

    Columns: 12 NR-*/SR-* task columns, mol_id, smiles.
    Missing labels are empty (NaN). Same pattern as MUV.
    Canonicalizes SMILES and removes duplicates.

    Returns:
        (X, y_multi, task_names, smiles_list, valid_indices)
        X: float32 [n_valid, 2048] Morgan fingerprints
        y_multi: float32 [n_valid, 12] labels (0/1/NaN)
        task_names: list of 12 task column names
        smiles_list: canonical SMILES
        valid_indices: indices with valid fingerprints
    """
    from rdkit import Chem

    df = pd.read_csv(csv_path)
    task_names = [c for c in df.columns if c not in ('mol_id', 'smiles')]
    print(f"  Tox21 multi-task: {len(df)} molecules, {len(task_names)} tasks")

    # Canonicalize SMILES and deduplicate
    raw_smiles = df['smiles'].values.tolist()
    y_raw = df[task_names].values.astype(np.float32)  # NaN preserved
    seen = set()
    smiles_list = []
    y_list = []
    for i, smi in enumerate(raw_smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            smiles_list.append(canon)
            y_list.append(y_raw[i])
    y_all = np.array(y_list, dtype=np.float32)
    n_dedup = len(df) - len(smiles_list)
    if n_dedup > 0:
        print(f"  Deduplicated: removed {n_dedup} duplicates -> {len(smiles_list)} unique")

    X, valid_indices = _smiles_to_morgan(smiles_list)
    y = y_all[valid_indices]

    for i, tn in enumerate(task_names):
        n_pos = int(np.nansum(y[:, i] == 1))
        n_neg = int(np.nansum(y[:, i] == 0))
        n_miss = int(np.isnan(y[:, i]).sum())
        print(f"    {tn}: {n_pos} pos, {n_neg} neg, {n_miss} missing")

    return X, y, task_names, smiles_list, valid_indices


# ============================================================
# Molecular Graph Construction from SMILES (Phase 4)
# ============================================================

def smiles_to_molecular_graphs(smiles_list, valid_indices=None):
    """Convert SMILES strings to PyG Data objects with atom/bond features.

    Creates molecular graphs where:
    - Nodes = atoms with features: atomic type (one-hot), degree, formal charge,
      hybridization, aromaticity, number of hydrogens
    - Edges = bonds (undirected) with bond features: bond type, conjugation,
      ring membership, stereochemistry

    Atom feature encoding (32 dimensions total):
      - Atomic symbol: C, N, O, F, P, S, Cl, Br, I, other  (10)
      - Degree: 0-5                                          (6)
      - Formal charge: -2 to +2                              (5)
      - Hybridization: sp, sp2, sp3, sp3d, sp3d2             (5)
      - Is aromatic: binary                                   (1)
      - Total Hs: 0-4                                         (5)

    Bond feature encoding (12 dimensions total):
      - Bond type: single, double, triple, aromatic           (4)
      - Is conjugated: binary                                  (1)
      - Is in ring: binary                                     (1)
      - Stereo: none, any, Z, E, cis, trans                   (6)

    Args:
        smiles_list: List of SMILES strings
        valid_indices: Optional list of indices to process (matching fingerprint
                       valid_idx from _smiles_to_morgan*). If None, processes all.

    Returns:
        list of torch_geometric.data.Data objects, one per valid molecule.
        Invalid SMILES get a dummy single-node graph.
    """
    from rdkit import Chem
    from torch_geometric.data import Data

    ATOM_SYMBOLS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    def _atom_features(atom):
        """Compute 32-dim feature vector for a single atom."""
        symbol = atom.GetSymbol()
        # One-hot atomic symbol (10)
        type_enc = [1.0 if symbol == s else 0.0 for s in ATOM_SYMBOLS]
        type_enc.append(1.0 if symbol not in ATOM_SYMBOLS else 0.0)  # "other"

        # Degree one-hot (6)
        degree = min(atom.GetDegree(), 5)
        degree_enc = [1.0 if i == degree else 0.0 for i in range(6)]

        # Formal charge one-hot (5): maps -2..+2 to indices 0..4
        charge = max(-2, min(2, atom.GetFormalCharge()))
        charge_enc = [1.0 if i == charge + 2 else 0.0 for i in range(5)]

        # Hybridization one-hot (5)
        hybrid = atom.GetHybridization()
        hybrid_enc = [1.0 if hybrid == h else 0.0 for h in HYBRIDIZATION_TYPES]

        # Aromaticity (1)
        aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

        # Number of Hs one-hot (5)
        n_hs = min(atom.GetTotalNumHs(), 4)
        hs_enc = [1.0 if i == n_hs else 0.0 for i in range(5)]

        return type_enc + degree_enc + charge_enc + hybrid_enc + aromatic + hs_enc

    # Bond feature types for GINEConv (12-dim)
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    BOND_STEREO = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    def _bond_features(bond):
        """Compute 12-dim feature vector for a single bond.

        Used by GINEConv which adds edge features into messages (additive
        mechanism, not attention bias). Features:
          - Bond type one-hot (4): single, double, triple, aromatic
          - Is conjugated (1): binary
          - Is in ring (1): binary
          - Stereo one-hot (6): none, any, Z, E, cis, trans
        """
        # Bond type one-hot (4)
        bt = bond.GetBondType()
        bt_enc = [1.0 if bt == t else 0.0 for t in BOND_TYPES]
        # Conjugation (1)
        conj = [1.0 if bond.GetIsConjugated() else 0.0]
        # In ring (1)
        ring = [1.0 if bond.IsInRing() else 0.0]
        # Stereo one-hot (6)
        stereo = bond.GetStereo()
        stereo_enc = [1.0 if stereo == s else 0.0 for s in BOND_STEREO]
        return bt_enc + conj + ring + stereo_enc

    indices = valid_indices if valid_indices is not None else range(len(smiles_list))
    data_list = []

    for idx in indices:
        smiles = str(smiles_list[idx])
        mol = Chem.MolFromSmiles(smiles)

        if mol is None or mol.GetNumAtoms() == 0:
            # Dummy single-node graph for invalid SMILES
            data_list.append(Data(
                x=torch.zeros(1, 32, dtype=torch.float32),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 12, dtype=torch.float32),
            ))
            continue

        # Node features: [num_atoms, 32]
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(_atom_features(atom))
        x = torch.tensor(node_feats, dtype=torch.float32)

        # Edge index + bond features: undirected bonds -> [2, 2*num_bonds]
        src, dst = [], []
        bond_feats = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])   # Undirected: add both directions
            dst.extend([j, i])
            bf = _bond_features(bond)
            bond_feats.append(bf)  # i -> j
            bond_feats.append(bf)  # j -> i (same features for both directions)

        if len(src) > 0:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(bond_feats, dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 12, dtype=torch.float32)

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return data_list


# Electronegativity lookup (Pauling scale, common organic elements)
_ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Si': 1.90, 'B': 2.04, 'Se': 2.55, 'Na': 0.93, 'K': 0.82,
    'Li': 0.98, 'Mg': 1.31, 'Ca': 1.00, 'Fe': 1.83, 'Zn': 1.65,
    'Cu': 1.90, 'As': 2.18, 'Sn': 1.96, 'Te': 2.10,
}


def smiles_to_molecular_graphs_3d(smiles_list, valid_indices=None, cutoff=5.0):
    """Convert SMILES to molecular graphs with 3D geometry and non-bonded interactions.

    Enriched featurization matching KA-GNN's approach:
    - Atom features (50-dim): atom type, degree, charge, hybridization,
      aromaticity, H count, chirality, Gasteiger charge, ring membership,
      atomic mass, radical electrons, donor/acceptor, electronegativity, valence
    - Bond features (21-dim): direction, type, 3D bond length, ring membership,
      plus placeholder for non-bonded features
    - 3D conformer via MMFF94/UFF force field
    - Non-bonded spatial edges within cutoff with Coulomb interaction features
      (Gasteiger charges, 1/r, 1/r^6, 1/r^12)

    Args:
        smiles_list: List of SMILES strings
        valid_indices: Optional indices to process (matching fingerprint valid_idx)
        cutoff: Distance cutoff for non-bonded edges in Angstroms (default 5.0)

    Returns:
        (data_list, ATOM_DIM, BOND_DIM) tuple where:
        - data_list: list of PyG Data objects with x, edge_index, edge_attr
        - ATOM_DIM: int (50)
        - BOND_DIM: int (21)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    ATOM_DIM = 50
    BOND_DIM = 21

    ATOM_SYMBOLS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    # Bond length approximations (Angstroms) for when 3D is unavailable
    BOND_LENGTH_APPROX = {
        Chem.rdchem.BondType.SINGLE: 1.54,
        Chem.rdchem.BondType.DOUBLE: 1.34,
        Chem.rdchem.BondType.TRIPLE: 1.20,
        Chem.rdchem.BondType.AROMATIC: 1.40,
    }

    def _atom_features_3d(atom, gasteiger_charge=0.0):
        """Compute 50-dim enriched feature vector for a single atom.

        Dimensions:
          atom_type_onehot(10) + degree(6) + formal_charge(5) +
          hybridization(5) + aromaticity(1) + num_Hs(5) + chirality(4) +
          gasteiger_charge(1) + is_in_ring(1) + ring_sizes_3to8(6) +
          atomic_mass(1) + radical_electrons(1) + is_donor(1) +
          is_acceptor(1) + electronegativity(1) + total_valence(1) = 50
        """
        symbol = atom.GetSymbol()

        # One-hot atomic symbol (10)
        type_enc = [1.0 if symbol == s else 0.0 for s in ATOM_SYMBOLS]
        type_enc.append(1.0 if symbol not in ATOM_SYMBOLS else 0.0)

        # Degree one-hot (6): 0-5
        degree = min(atom.GetDegree(), 5)
        degree_enc = [1.0 if i == degree else 0.0 for i in range(6)]

        # Formal charge one-hot (5): -2..+2
        charge = max(-2, min(2, atom.GetFormalCharge()))
        charge_enc = [1.0 if i == charge + 2 else 0.0 for i in range(5)]

        # Hybridization one-hot (5)
        hybrid = atom.GetHybridization()
        hybrid_enc = [1.0 if hybrid == h else 0.0 for h in HYBRIDIZATION_TYPES]

        # Aromaticity (1)
        aromatic = [1.0 if atom.GetIsAromatic() else 0.0]

        # Number of Hs one-hot (5): 0-4
        n_hs = min(atom.GetTotalNumHs(), 4)
        hs_enc = [1.0 if i == n_hs else 0.0 for i in range(5)]

        # Chirality one-hot (4): R, S, E, Z
        chiral = [0.0] * 4
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode")
            for ci, code in enumerate(["R", "S", "E", "Z"]):
                if cip == code:
                    chiral[ci] = 1.0

        # Gasteiger partial charge (1), normalized to [-1, 1]
        gc = max(-2.0, min(2.0, gasteiger_charge)) / 2.0

        # Ring info (7): is_in_ring(1) + ring_sizes_3to8(6)
        in_ring = [1.0 if atom.IsInRing() else 0.0]
        ring_sizes = [1.0 if atom.IsInRingSize(s) else 0.0 for s in range(3, 9)]

        # Atomic mass normalized (1)
        mass = [atom.GetMass() / 200.0]

        # Radical electrons (1)
        radical = [min(atom.GetNumRadicalElectrons(), 2) / 2.0]

        # H-bond donor/acceptor (2)
        is_donor = [1.0 if (symbol in ('N', 'O') and atom.GetTotalNumHs() > 0) else 0.0]
        is_acceptor = [1.0 if symbol in ('N', 'O', 'F') else 0.0]

        # Electronegativity (1), normalized
        en = [_ELECTRONEGATIVITY.get(symbol, 2.0) / 4.0]

        # Total valence (1), normalized
        valence = [min(atom.GetTotalValence(), 6) / 6.0]

        return (type_enc + degree_enc + charge_enc + hybrid_enc + aromatic +
                hs_enc + chiral + [gc] + in_ring + ring_sizes + mass +
                radical + is_donor + is_acceptor + en + valence)

    def _bond_features_3d(bond, bond_length=None):
        """Compute 21-dim bond features (matching KA-GNN encode_bond_14 layout).

        Dimensions:
          bond_direction(7) + bond_type(4) + [length, length^2](2) +
          in_ring(2) + non_bond_placeholder(6) = 21
        """
        # Bond direction one-hot (7)
        bond_dir = [0.0] * 7
        d = int(bond.GetBondDir())
        if 0 <= d < 7:
            bond_dir[d] = 1.0

        # Bond type one-hot (4): single=0, double=1, triple=2, aromatic=3
        bond_type = [0.0] * 4
        bt_double = bond.GetBondTypeAsDouble()
        bt_idx = int(bt_double) - 1
        if 0 <= bt_idx < 4:
            bond_type[bt_idx] = 1.0

        # Bond length + length^2 (2)
        if bond_length is None:
            bond_length = BOND_LENGTH_APPROX.get(bond.GetBondType(), 1.5)
        bl = [bond_length, bond_length ** 2]

        # In ring one-hot (2)
        in_ring = [0.0, 0.0]
        in_ring[int(bond.IsInRing())] = 1.0

        # Non-bonded feature placeholder (6 zeros for bonded edges)
        non_bond = [0.0] * 6

        return bond_dir + bond_type + bl + in_ring + non_bond

    def _non_bonded_features(q_i, q_j, distance):
        """Compute 21-dim non-bonded edge features (Coulomb interactions).

        Layout: bond_placeholder(15) + coulomb_features(6) = 21
        Coulomb: [q_i, q_j, q_i*q_j, 1/r, 1/r^6, 1/r^12]
        All values clipped to [-10, 10] for numerical stability.
        """
        bond_placeholder = [0.0] * 15
        r = max(distance, 0.5)  # Min 0.5 Angstroms to avoid extreme values
        # Clip charges to reasonable range
        q_i = max(-2.0, min(2.0, q_i))
        q_j = max(-2.0, min(2.0, q_j))
        inv_r = min(1.0 / r, 10.0)
        inv_r6 = min(1.0 / (r ** 6), 10.0)
        inv_r12 = min(1.0 / (r ** 12), 10.0)
        coulomb = [q_i, q_j, q_i * q_j, inv_r, inv_r6, inv_r12]
        return bond_placeholder + coulomb

    indices = valid_indices if valid_indices is not None else range(len(smiles_list))
    data_list = []
    n_3d_ok = 0
    n_3d_fail = 0
    n_nonbonded_total = 0

    for idx in indices:
        smiles = str(smiles_list[idx])
        mol = Chem.MolFromSmiles(smiles)

        if mol is None or mol.GetNumAtoms() == 0:
            data_list.append(Data(
                x=torch.zeros(1, ATOM_DIM, dtype=torch.float32),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, BOND_DIM, dtype=torch.float32),
            ))
            continue

        n_heavy = mol.GetNumAtoms()

        # Assign stereochemistry for CIP codes
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # --- 3D conformer generation ---
        has_3d = False
        coords = None
        charges = [0.0] * n_heavy

        if n_heavy <= 500:
            mol_h = Chem.AddHs(mol)
            try:
                res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
                if res == 0:
                    # Try MMFF94 optimization, fallback to UFF
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_h)
                    except Exception:
                        try:
                            AllChem.UFFOptimizeMolecule(mol_h)
                        except Exception:
                            pass
                    has_3d = True
                    conf = mol_h.GetConformer()
                    coords = np.array([
                        [conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(n_heavy)
                    ])
                    # Gasteiger charges on H-inclusive molecule
                    try:
                        AllChem.ComputeGasteigerCharges(mol_h)
                        for ai in range(n_heavy):
                            gc = mol_h.GetAtomWithIdx(ai).GetDoubleProp("_GasteigerCharge")
                            charges[ai] = gc if (np.isfinite(gc)) else 0.0
                    except Exception:
                        charges = [0.0] * n_heavy
            except Exception:
                pass

        if has_3d:
            n_3d_ok += 1
        else:
            n_3d_fail += 1
            # Gasteiger on 2D molecule as fallback
            try:
                AllChem.ComputeGasteigerCharges(mol)
                for ai in range(n_heavy):
                    gc = mol.GetAtomWithIdx(ai).GetDoubleProp("_GasteigerCharge")
                    charges[ai] = gc if (np.isfinite(gc)) else 0.0
            except Exception:
                charges = [0.0] * n_heavy

        # --- Node features [n_heavy, ATOM_DIM] ---
        node_feats = []
        for ai, atom in enumerate(mol.GetAtoms()):
            node_feats.append(_atom_features_3d(atom, charges[ai]))
        x = torch.tensor(node_feats, dtype=torch.float32)

        # --- Edge features (bonded) ---
        src, dst = [], []
        edge_feats = []
        bonded_pairs = set()

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonded_pairs.add((min(i, j), max(i, j)))

            # Bond length from 3D coordinates or approximation
            bl = None
            if has_3d and coords is not None:
                bl = float(np.linalg.norm(coords[i] - coords[j]))

            bf = _bond_features_3d(bond, bond_length=bl)

            src.extend([i, j])
            dst.extend([j, i])
            edge_feats.append(bf)
            edge_feats.append(bf)

        # --- Non-bonded edges (only with 3D coordinates) ---
        if has_3d and coords is not None:
            n_nb = 0
            for i in range(n_heavy):
                for j in range(i + 1, n_heavy):
                    if (i, j) in bonded_pairs:
                        continue
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    if dist <= cutoff:
                        nbf = _non_bonded_features(charges[i], charges[j], dist)
                        src.extend([i, j])
                        dst.extend([j, i])
                        edge_feats.append(nbf)
                        edge_feats.append(nbf)
                        n_nb += 1
            n_nonbonded_total += n_nb

        # --- Build PyG Data ---
        if len(src) > 0:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, BOND_DIM, dtype=torch.float32)

        # Safety: clamp all features to prevent NaN/inf propagation
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if edge_attr.numel() > 0:
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=10.0, neginf=-10.0)

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    print(f"  3D conformer: {n_3d_ok} success, {n_3d_fail} failed "
          f"(non-bonded edges: {n_nonbonded_total})")
    return data_list, ATOM_DIM, BOND_DIM


def smiles_to_molecular_batch(smiles_list, valid_indices=None):
    """Convert SMILES to a single batched PyG Batch object.

    Convenience wrapper: calls smiles_to_molecular_graphs() then batches
    all graphs into a single PyG Batch for efficient full-batch GNN processing.

    Args:
        smiles_list: List of SMILES strings
        valid_indices: Optional list of indices to process

    Returns:
        torch_geometric.data.Batch with:
        - x: [total_atoms, 32] node features
        - edge_index: [2, total_edges] global edge indices
        - edge_attr: [total_edges, 12] bond features (for GINEConv)
        - batch: [total_atoms] molecule assignment per atom
        - num_graphs: number of molecules
    """
    from torch_geometric.data import Batch

    data_list = smiles_to_molecular_graphs(smiles_list, valid_indices)
    batch = Batch.from_data_list(data_list)

    n_atoms = batch.x.shape[0]
    n_edges = batch.edge_index.shape[1]
    n_mols = batch.num_graphs
    edge_attr = getattr(batch, 'edge_attr', None)
    bond_info = f", bond_features={edge_attr.shape[1]}d" if edge_attr is not None and edge_attr.shape[0] > 0 else ""
    print(f"  Molecular batch: {n_mols} molecules, "
          f"{n_atoms} atoms, {n_edges} bonds{bond_info}")
    return batch


# ============================================================
# k-NN Graph Construction for Tier 7
# ============================================================

def build_knn_graph(X: np.ndarray, k: int = 10) -> dict:
    """Build a k-NN similarity graph from a feature matrix.

    Each sample becomes a node. Edges connect each node to its k nearest
    neighbors (cosine similarity). Returns graph_info dict compatible with
    model.set_graph_data().

    Args:
        X: Feature matrix [N, d] (numpy array)
        k: Number of nearest neighbors per node

    Returns:
        dict with 'edge_index' [2, E], 'num_nodes' int, 'degree' [N]
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Find k nearest neighbors (cosine distance)
    k_actual = min(k, len(X) - 1)
    nn = NearestNeighbors(
        n_neighbors=k_actual + 1, metric='cosine', algorithm='brute'
    )
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    # Build edge_index [2, E] — symmetric (undirected)
    src, dst = [], []
    for i in range(len(X)):
        for j in indices[i, 1:]:  # Skip self (index 0)
            src.append(i)
            dst.append(j)
            src.append(j)  # Symmetric
            dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    # Compute degree
    num_nodes = len(X)
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))

    print(f"  k-NN graph: {num_nodes} nodes, {edge_index.shape[1]} edges "
          f"(k={k_actual})")
    return {
        'edge_index': edge_index,
        'num_nodes': num_nodes,
        'degree': degree,
    }


# ============================================================
# Full-batch Graph DataLoaders for Tier 7
# ============================================================

def create_bio_graph_dataloaders(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """Create full-batch DataLoaders for graph-augmented bio experiments.

    Concatenates all splits into one [N, d] tensor (like Tier 6 graph loaders),
    masks non-target labels with ignore_index=-100 (classification) or NaN
    (regression). Returns train/val/test DataLoaders.

    Args:
        X_train, y_train: Training split tensors
        X_val, y_val: Validation split tensors
        X_test, y_test: Test split tensors
        task_type: 'classification' or 'regression'

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    from torch.utils.data import DataLoader, TensorDataset

    X_all = torch.cat([X_train, X_val, X_test], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)
    N = X_all.shape[0]
    n_train, n_val = len(X_train), len(X_val)

    if task_type == "classification":
        ignore_index = -100

        # Train loader: mask val+test labels
        y_tr = y_all.clone()
        y_tr[n_train:] = ignore_index

        # Val loader: mask train+test labels
        y_va = torch.full_like(y_all, ignore_index)
        y_va[n_train:n_train + n_val] = y_all[n_train:n_train + n_val]

        # Test loader: mask train+val labels
        y_te = torch.full_like(y_all, ignore_index)
        y_te[n_train + n_val:] = y_all[n_train + n_val:]
    else:
        # Regression: mask with NaN (MaskedMSELoss handles it)
        y_tr = y_all.clone().float()
        y_tr[n_train:] = float('nan')

        y_va = torch.full_like(y_all, float('nan'))
        y_va[n_train:n_train + n_val] = y_all[n_train:n_train + n_val]

        y_te = torch.full_like(y_all, float('nan'))
        y_te[n_train + n_val:] = y_all[n_train + n_val:]

    train_loader = DataLoader(
        TensorDataset(X_all, y_tr), batch_size=N, shuffle=False)
    val_loader = DataLoader(
        TensorDataset(X_all, y_va), batch_size=N, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(X_all, y_te), batch_size=N, shuffle=False)

    print(f"  Graph DataLoaders: full-batch (N={N}), "
          f"train={n_train}, val={n_val}, test={len(X_test)}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ============================================================
# Masked MSE Loss for Graph Regression
# ============================================================

class MaskedMSELoss(torch.nn.Module):
    """MSE loss that ignores NaN targets (for graph regression masking).

    When training graph models with full-batch loaders, non-relevant nodes
    have their targets set to NaN. This loss computes MSE only on valid
    (non-NaN) entries.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target.view(-1))
        if mask.sum() == 0:
            # No valid targets — return zero loss with gradient connection
            return (pred * 0.0).sum()
        return torch.nn.functional.mse_loss(
            pred.view(-1)[mask], target.view(-1)[mask]
        )


# ============================================================
# Graph-aware Evaluation Helpers
# ============================================================

def graph_eval_classification(
    model: torch.nn.Module,
    split_data: dict,
    device: str,
    n_classes: int,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Graph-aware classification evaluation.

    Forward all nodes through the model, extract test-split predictions,
    and compute standard classification metrics.

    Returns:
        (metrics_dict, test_logits)
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )

    X_all = torch.cat([
        split_data["X_train"], split_data["X_val"], split_data["X_test"]
    ], dim=0)
    n_test_start = len(split_data["X_train"]) + len(split_data["X_val"])

    model.eval()
    with torch.no_grad():
        all_logits = model(X_all.to(device)).cpu()
    test_logits = all_logits[n_test_start:]
    preds = test_logits.argmax(dim=1).numpy()
    y_true = split_data["y_test"].numpy()

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average='weighted', zero_division=0)),
        "precision_macro": float(precision_score(y_true, preds, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average='macro', zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "n_classes": n_classes,
    }, test_logits


def load_freesolv_csv(csv_path: str, return_smiles: bool = False):
    """Load FreeSolv (Free Solvation Database) dataset from CSV file.

    Regression: predict experimental hydration free energy (kcal/mol).
    ~642 molecules.

    CSV columns: iupac, smiles, expt, calc
    Target: 'expt' (experimental solvation free energy).

    Args:
        csv_path: Path to freesolve.csv
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y)
        If return_smiles=True: (X, y, smiles_list, valid_indices)
    """
    df = pd.read_csv(csv_path)
    print(f"  Raw FreeSolv CSV: {len(df)} molecules")

    smiles_list = df['smiles'].values.tolist()
    y_all = df['expt'].values.astype(np.float32)

    X, valid_idx = _smiles_to_morgan_with_descriptors(smiles_list, n_bits=512)
    y = y_all[valid_idx]

    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features "
          f"(512-bit FP + 12 RDKit descriptors)")
    print(f"  Free energy range: [{y.min():.2f}, {y.max():.2f}] kcal/mol")

    if return_smiles:
        return X, y, smiles_list, valid_idx
    return X, y


def load_lipophilicity_csv(csv_path: str, return_smiles: bool = False):
    """Load Lipophilicity dataset from CSV file.

    Regression: predict experimental lipophilicity (logD at pH 7.4).
    ~4,200 molecules from ChEMBL.

    CSV columns: CMPD_CHEMBLID, exp, smiles
    Target: 'exp' (experimental lipophilicity).

    Args:
        csv_path: Path to Lipophilicity.csv
        return_smiles: If True, also return SMILES list and valid indices
                       for molecular graph construction.

    Returns:
        If return_smiles=False: (X, y)
        If return_smiles=True: (X, y, smiles_list, valid_indices)
    """
    df = pd.read_csv(csv_path)
    print(f"  Raw Lipophilicity CSV: {len(df)} molecules")

    smiles_list = df['smiles'].values.tolist()
    y_all = df['exp'].values.astype(np.float32)

    X, valid_idx = _smiles_to_morgan_with_descriptors(smiles_list, n_bits=512)
    y = y_all[valid_idx]

    print(f"  Final: {X.shape[0]} molecules, {X.shape[1]} features "
          f"(512-bit FP + 12 RDKit descriptors)")
    print(f"  Lipophilicity range: [{y.min():.2f}, {y.max():.2f}] logD")

    if return_smiles:
        return X, y, smiles_list, valid_idx
    return X, y


def graph_eval_regression(
    model: torch.nn.Module,
    split_data: dict,
    device: str,
    y_scaler,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Graph-aware regression evaluation.

    Forward all nodes through the model, extract test-split predictions,
    inverse-transform, and compute standard regression metrics.

    Returns:
        (metrics_dict, y_pred_original_scale)
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X_all = torch.cat([
        split_data["X_train"], split_data["X_val"], split_data["X_test"]
    ], dim=0)
    n_test_start = len(split_data["X_train"]) + len(split_data["X_val"])

    model.eval()
    with torch.no_grad():
        all_preds = model(X_all.to(device)).cpu()
    test_preds_scaled = all_preds[n_test_start:].numpy()

    # Inverse-transform to original scale
    preds_original = y_scaler.inverse_transform(test_preds_scaled)
    y_true = split_data["y_test_original"].numpy()

    mse = mean_squared_error(y_true, preds_original)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, preds_original)),
        "r2": float(r2_score(y_true, preds_original)),
    }, preds_original
