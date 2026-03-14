"""
Tier 6: Graph Dataset Utilities
================================
Loaders and converters for graph-structured datasets.
Converts PyG Data objects and HDF5 traffic data into ASANN-compatible format.

Key design: ASANN sees [N, d_features] flat vectors -- same as tabular.
Graph structure is stored as auxiliary data on the model via model.set_graph_data().

Traffic data preserves graph-snapshot structure: each forward pass receives
[N_nodes, window] matching the adjacency [N_nodes, N_nodes].

Data Pipeline:
- Raw data (HDF5/pkl with pandas internals) is preprocessed once into pure numpy
  arrays stored as .npz files in data/preprocessed/<dataset_name>/
- During training, only .npz files are loaded -- no pandas, no pickle, no HDF5
- All Dataset classes use direct numpy/tensor indexing (thread-safe, no df.iloc)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, List, Tuple, Optional

# PyTorch 2.8+ defaults weights_only=True which breaks OGB/PyG data loading.
# Monkey-patch to use weights_only=False by default.
if hasattr(torch, '__version__') and tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 6):
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load


def load_pyg_citation(name: str, root: str = "data/"):
    """Load a Planetoid citation graph (CiteSeer, PubMed, Cora).

    Returns torch_geometric.data.Data with x, y, edge_index, train/val/test masks.
    """
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    print(f"  Loaded {name}: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_features} features, {dataset.num_classes} classes")
    return data, dataset.num_classes


def load_ogbn_arxiv(root: str = "data/"):
    """Load ogbn-arxiv node classification dataset via OGB.

    Requires: pip install ogb
    Returns: (data, num_classes) where data has x, y, edge_index, and split indices.
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    # Convert split indices to masks
    num_nodes = data.x.shape[0]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[split_idx['train']] = True
    data.val_mask[split_idx['valid']] = True
    data.test_mask[split_idx['test']] = True
    data.y = data.y.squeeze()  # (N, 1) -> (N,)
    num_classes = dataset.num_classes
    print(f"  Loaded ogbn-arxiv: {num_nodes} nodes, {data.edge_index.shape[1]} edges, "
          f"{data.num_features} features, {num_classes} classes")
    return data, num_classes


def load_reddit(root: str = "data/"):
    """Load Reddit node classification dataset via PyG.

    Returns: (data, num_classes)
    """
    from torch_geometric.datasets import Reddit
    dataset = Reddit(root=os.path.join(root, 'reddit'))
    data = dataset[0]
    num_classes = dataset.num_classes
    print(f"  Loaded Reddit: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_features} features, {num_classes} classes")
    return data, num_classes


def _get_preprocessed_dir(name: str, data_dir: str = "data/") -> str:
    """Get path to preprocessed data directory for a traffic dataset."""
    return os.path.join(data_dir, "preprocessed", name.replace("-", "_"))


def preprocess_traffic(name: str, data_dir: str = "data/",
                       window: int = 12, horizon: int = 12,
                       force: bool = False) -> str:
    """Preprocess raw traffic data (HDF5 + pkl) into cached numpy arrays.

    Reads the raw pandas-format HDF5 and pickle adjacency, extracts pure numpy
    arrays, builds sliding windows, and saves everything as .npz files in
    data/preprocessed/<dataset_name>/.

    This should be called once before training. If cached files already exist,
    preprocessing is skipped (unless force=True).

    Args:
        name: 'metr-la' or 'pems-bay'
        data_dir: Directory containing raw HDF5 and pkl files
        window: Input window size (default 12 = 1 hour at 5-min intervals)
        horizon: Prediction horizon (default 12 = 1 hour ahead)
        force: If True, reprocess even if cache exists

    Returns:
        Path to the preprocessed directory
    """
    out_dir = _get_preprocessed_dir(name, data_dir)
    cache_file = os.path.join(out_dir, f"windows_w{window}_h{horizon}.npz")
    adj_file = os.path.join(out_dir, "adj.npy")

    if not force and os.path.exists(cache_file) and os.path.exists(adj_file):
        print(f"  Preprocessed data found: {out_dir}")
        return out_dir

    print(f"  Preprocessing {name} data...")
    os.makedirs(out_dir, exist_ok=True)

    # === Step 1: Extract raw time series from HDF5 (pandas-format) ===
    import h5py

    h5_path = os.path.join(data_dir, f"{name}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Raw HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # HDF5 files are pandas-exported: df/block0_values or speed/block0_values
        if "df" in f:
            raw = np.array(f["df"]["block0_values"], dtype=np.float32)
        elif "speed" in f:
            raw = np.array(f["speed"]["block0_values"], dtype=np.float32)
        else:
            # Fallback: try first group's block0_values
            first_key = list(f.keys())[0]
            if "block0_values" in f[first_key]:
                raw = np.array(f[first_key]["block0_values"], dtype=np.float32)
            else:
                raw = np.array(f[first_key], dtype=np.float32)

    T, N = raw.shape
    print(f"    Raw time series: {T} timesteps, {N} sensors")

    # === Step 2: Build sliding windows using vectorized indexing ===
    n_samples = T - window - horizon + 1
    # Vectorized window construction (no Python loop)
    t_indices = np.arange(n_samples)
    # X: for each sample t, take raw[t:t+window, :].T -> [N, window]
    x_idx = t_indices[:, None] + np.arange(window)[None, :]  # [n_samples, window]
    y_idx = t_indices[:, None] + window + np.arange(horizon)[None, :]  # [n_samples, horizon]
    # raw[x_idx] -> [n_samples, window, N] -> transpose to [n_samples, N, window]
    X_all = raw[x_idx].transpose(0, 2, 1).copy()  # [n_samples, N, window]
    Y_all = raw[y_idx].transpose(0, 2, 1).copy()  # [n_samples, N, horizon]

    print(f"    Sliding windows: {n_samples} samples, X={X_all.shape}, Y={Y_all.shape}")

    # Save windowed data as .npz
    np.savez_compressed(cache_file, X=X_all, Y=Y_all)
    print(f"    Saved windowed data: {cache_file}")

    # === Step 3: Extract adjacency from pickle ===
    pkl_map = {
        'metr-la': 'adj_mx_metr.pkl',
        'pems-bay': 'adj_mx_bay.pkl',
    }
    pkl_path = os.path.join(data_dir, pkl_map[name])
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Adjacency pkl not found: {pkl_path}\n"
            f"Download the DCRNN benchmark adjacency matrices."
        )

    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f, encoding='latin1')

    if isinstance(pkl_data, list) and len(pkl_data) >= 3:
        adj = np.array(pkl_data[2], dtype=np.float32)
    elif isinstance(pkl_data, np.ndarray):
        adj = pkl_data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected pickle format in {pkl_path}")

    # Remove self-loops (graph ops add them via A_hat = A + I)
    np.fill_diagonal(adj, 0.0)

    assert adj.shape[0] == N, (
        f"Adjacency shape {adj.shape} doesn't match sensor count {N}"
    )

    # Save adjacency as plain .npy (no pickle needed at runtime)
    np.save(adj_file, adj)
    n_edges = int((adj > 0).sum())
    print(f"    Saved adjacency: {adj_file} ({N}x{N}, {n_edges} edges)")

    # Delete raw data references to free memory
    del raw, X_all, Y_all, adj, pkl_data

    print(f"  Preprocessing complete: {out_dir}")
    return out_dir


def load_traffic_preprocessed(name: str, data_dir: str = "data/",
                               window: int = 12, horizon: int = 12,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed traffic data from cached .npz/.npy files.

    If preprocessed files don't exist, runs preprocessing first.
    Returns pure numpy arrays -- no pandas, no pickle, no HDF5 at runtime.

    This is the primary entry point for traffic data loading.

    Args:
        name: 'metr-la' or 'pems-bay'
        data_dir: Directory containing data/ and data/preprocessed/
        window: Input window size
        horizon: Prediction horizon

    Returns:
        X_all: [n_samples, N_nodes, window] float32 numpy array
        Y_all: [n_samples, N_nodes, horizon] float32 numpy array
        adj: [N_nodes, N_nodes] float32 weighted adjacency matrix
    """
    # Ensure preprocessing is done
    out_dir = preprocess_traffic(name, data_dir, window, horizon)

    # Load from cache -- pure numpy, thread-safe, no pandas/pickle
    cache_file = os.path.join(out_dir, f"windows_w{window}_h{horizon}.npz")
    adj_file = os.path.join(out_dir, "adj.npy")

    print(f"  Loading preprocessed {name} data...")

    # np.load with mmap_mode='r' for memory-efficient loading of large arrays
    with np.load(cache_file) as data:
        X_all = data['X'].astype(np.float32)  # [n_samples, N, window]
        Y_all = data['Y'].astype(np.float32)  # [n_samples, N, horizon]

    adj = np.load(adj_file).astype(np.float32)  # [N, N]

    n_edges = int((adj > 0).sum())
    print(f"  Loaded: {X_all.shape[0]} samples, {X_all.shape[1]} sensors, "
          f"window={window}, horizon={horizon}, {n_edges} edges")

    return X_all, Y_all, adj


def load_traffic_adj(name: str, data_dir: str = "data/") -> np.ndarray:
    """Load pre-computed benchmark adjacency matrix from pickle file.

    The DCRNN benchmark provides weighted adjacency matrices:
    - adj_mx_metr.pkl: 207x207 (METR-LA, Los Angeles traffic)
    - adj_mx_bay.pkl: 325x325 (PEMS-BAY, Bay Area traffic)

    These are Gaussian kernel matrices: exp(-d^2 / sigma^2) where d is road distance.
    Values are continuous [0, 1], include self-loops on diagonal, may be asymmetric
    (directed road distances).

    Args:
        name: 'metr-la' or 'pems-bay'
        data_dir: Directory containing the .pkl files

    Returns:
        adj: [N_nodes, N_nodes] float32 weighted adjacency matrix
    """
    pkl_map = {
        'metr-la': 'adj_mx_metr.pkl',
        'pems-bay': 'adj_mx_bay.pkl',
    }

    if name not in pkl_map:
        raise ValueError(f"Unknown traffic dataset: {name}. Expected one of {list(pkl_map.keys())}")

    filepath = os.path.join(data_dir, pkl_map[name])
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Adjacency file not found: {filepath}\n"
            f"Please download the DCRNN benchmark adjacency matrices."
        )

    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Format: [sensor_ids, id_to_idx_map, adj_matrix]
    if isinstance(data, list) and len(data) >= 3:
        adj = np.array(data[2], dtype=np.float32)
    elif isinstance(data, np.ndarray):
        adj = data.astype(np.float32)
    else:
        raise ValueError(f"Unexpected pickle format in {filepath}")

    # Remove self-loops from adjacency (graph ops handle self-loops via A_hat = A + I)
    np.fill_diagonal(adj, 0.0)

    n_edges = int((adj > 0).sum())
    print(f"  Loaded benchmark adjacency ({name}): {adj.shape[0]}x{adj.shape[1]}, "
          f"{n_edges} edges (weighted, directed)")

    return adj


def load_traffic_hdf5(name: str, data_dir: str = "data/",
                      window: int = 12, horizon: int = 12,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load traffic data using preprocessed cache (preferred) or raw HDF5 fallback.

    This function now routes through the preprocessing pipeline:
    1. Check for preprocessed .npz/.npy cache in data/preprocessed/<name>/
    2. If not found, preprocess raw HDF5 + pkl → cache, then load from cache
    3. All runtime data access uses pure numpy arrays (no pandas, no pickle)

    Args:
        name: 'metr-la' or 'pems-bay'
        data_dir: Directory containing raw data files and preprocessed/ subfolder
        window: Input window size (12 = 1 hour at 5-min intervals)
        horizon: Prediction horizon (12 = 1 hour ahead)

    Returns:
        X_all: [n_samples, N_nodes, window] float32 numpy array
        Y_all: [n_samples, N_nodes, horizon] float32 numpy array
        adj: [N_nodes, N_nodes] float32 weighted adjacency
    """
    return load_traffic_preprocessed(name, data_dir, window, horizon)


def graph_data_to_asann_format(data, num_classes: int = 0,
                                task_type: str = "classification"
                                ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """Convert PyG Data to ASANN-compatible (x, y, masks, graph_info).

    For node classification:
    - x = data.x [N, d_features]
    - y = data.y [N] (unlabeled nodes get ignore_index=-100 in loss)
    - masks = {train: [N] bool, val: [N] bool, test: [N] bool}
    - graph_info = {edge_index: [2,E], num_nodes: int}

    Returns:
        (x, y, masks, graph_info)
    """
    x = data.x.float()
    y = data.y.long() if task_type == "classification" else data.y.float()

    masks = {
        'train': data.train_mask,
        'val': data.val_mask,
        'test': data.test_mask,
    }

    graph_info = {
        'edge_index': data.edge_index,
        'num_nodes': data.num_nodes,
    }

    # Compute degree
    degree = torch.zeros(data.num_nodes, dtype=torch.float)
    degree.scatter_add_(0, data.edge_index[1],
                        torch.ones(data.edge_index.shape[1]))
    graph_info['degree'] = degree

    print(f"  ASANN format: x={x.shape}, y={y.shape}, "
          f"train={masks['train'].sum()}, val={masks['val'].sum()}, "
          f"test={masks['test'].sum()}")

    return x, y, masks, graph_info


def create_graph_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    task_type: str = "classification",
    ignore_index: int = -100,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create full-batch DataLoaders for graph node tasks.

    For graph training, each "batch" is ALL nodes (full-batch).
    Labels for non-train nodes are masked via ignore_index=-100.

    The trainer sees a single TensorDataset([N, d], [N]) per split,
    identical to tabular data. The difference is that N = all_nodes
    and the loss uses ignore_index to mask unlabeled nodes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    N = x.shape[0]

    # Train: mask out non-train labels
    y_train = y.clone()
    if task_type == "classification":
        y_train[~masks['train']] = ignore_index
    train_ds = TensorDataset(x, y_train)
    train_loader = DataLoader(train_ds, batch_size=N, shuffle=False)

    # Val: mask out non-val labels
    y_val = torch.full_like(y, ignore_index) if task_type == "classification" else y.clone()
    if task_type == "classification":
        y_val[masks['val']] = y[masks['val']]
    val_ds = TensorDataset(x, y_val)
    val_loader = DataLoader(val_ds, batch_size=N, shuffle=False)

    # Test: mask out non-test labels
    y_test = torch.full_like(y, ignore_index) if task_type == "classification" else y.clone()
    if task_type == "classification":
        y_test[masks['test']] = y[masks['test']]
    test_ds = TensorDataset(x, y_test)
    test_loader = DataLoader(test_ds, batch_size=N, shuffle=False)

    print(f"  DataLoaders: full-batch (batch_size={N}), "
          f"train_labels={masks['train'].sum()}, "
          f"val_labels={masks['val'].sum()}, "
          f"test_labels={masks['test'].sum()}")

    return train_loader, val_loader, test_loader


def extract_train_subgraph(
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Extract the subgraph containing only train nodes for inductive training.

    Filters edges to keep only those where BOTH endpoints are train nodes,
    then renumbers node indices to [0, N_train-1].

    Args:
        edge_index: [2, E] full graph edge index
        train_mask: [N] boolean mask of train nodes

    Returns:
        train_edge_index: [2, E_train] renumbered edge index
        train_degree: [N_train] in-degree of each train node in the subgraph
        n_train: number of train nodes
        node_mapping: [N] tensor mapping original node IDs to train IDs
                      (-1 for non-train nodes)
    """
    train_nodes = train_mask.nonzero(as_tuple=True)[0]
    n_train = train_nodes.shape[0]

    # Build mapping: original node ID -> train node ID
    node_mapping = torch.full((train_mask.shape[0],), -1, dtype=torch.long)
    node_mapping[train_nodes] = torch.arange(n_train)

    # Filter edges: keep only edges where both src and dst are train nodes
    src, dst = edge_index[0], edge_index[1]
    src_is_train = train_mask[src]
    dst_is_train = train_mask[dst]
    edge_mask = src_is_train & dst_is_train

    # Renumber edge indices
    train_edge_index = torch.stack([
        node_mapping[edge_index[0, edge_mask]],
        node_mapping[edge_index[1, edge_mask]],
    ])

    # Compute degree in the subgraph
    train_degree = torch.zeros(n_train, dtype=torch.float)
    train_degree.scatter_add_(
        0, train_edge_index[1],
        torch.ones(train_edge_index.shape[1])
    )

    print(f"  Train subgraph: {n_train} nodes, {train_edge_index.shape[1]} edges "
          f"(from {edge_index.shape[1]} total)")

    return train_edge_index, train_degree, n_train, node_mapping


def create_inductive_graph_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    task_type: str = "classification",
    ignore_index: int = -100,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create dataloaders for inductive graph training.

    Train loader contains ONLY train node features and labels (no masking).
    Val/test loaders contain ALL nodes with masked labels (for full-graph eval).

    This avoids the train/val loss gap caused by transductive training where
    the model is optimized on train nodes but val nodes get no gradient signal.

    Args:
        x: [N, d] all node features
        y: [N] all node labels
        masks: dict with 'train', 'val', 'test' boolean masks
        task_type: "classification" or "regression"
        ignore_index: label value for masked nodes in val/test loaders

    Returns:
        (train_loader, val_loader, test_loader, subgraph_info)
        subgraph_info contains edge_index/degree/num_nodes for train subgraph
        and full graph, used for swapping during eval.
    """
    N = x.shape[0]
    train_mask = masks['train']
    N_train = int(train_mask.sum().item())

    # Train: ONLY train node features and real labels (no masking, no ignore_index)
    x_train = x[train_mask]       # [N_train, d]
    y_train = y[train_mask]       # [N_train]
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=N_train, shuffle=False)

    # Val: ALL nodes, val labels only (same as transductive for full-graph eval)
    y_val = torch.full_like(y, ignore_index) if task_type == "classification" else y.clone()
    if task_type == "classification":
        y_val[masks['val']] = y[masks['val']]
    val_ds = TensorDataset(x, y_val)
    val_loader = DataLoader(val_ds, batch_size=N, shuffle=False)

    # Test: ALL nodes, test labels only
    y_test = torch.full_like(y, ignore_index) if task_type == "classification" else y.clone()
    if task_type == "classification":
        y_test[masks['test']] = y[masks['test']]
    test_ds = TensorDataset(x, y_test)
    test_loader = DataLoader(test_ds, batch_size=N, shuffle=False)

    print(f"  Inductive DataLoaders: train={N_train} nodes (train-only), "
          f"val/test={N} nodes (full-graph)")
    print(f"    train_labels={N_train}, "
          f"val_labels={masks['val'].sum()}, "
          f"test_labels={masks['test'].sum()}")

    return train_loader, val_loader, test_loader


class GraphSnapshotDataset(Dataset):
    """Dataset where each sample is a full graph snapshot [N_nodes, features].

    For traffic forecasting, each time step is one complete graph snapshot.
    The model forward pass receives [N_nodes, window] matching adj[N_nodes, N_nodes].
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: [n_samples, N_nodes, window]
            Y: [n_samples, N_nodes, horizon]
        """
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == Y.shape[1]  # Same N_nodes
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Returns [N_nodes, window], [N_nodes, horizon]
        return self.X[idx], self.Y[idx]


def compute_time_features(n_samples: int, N_nodes: int,
                          steps_per_day: int = 288,
                          start_step: int = 0) -> np.ndarray:
    """Generate sin/cos time-of-day and day-of-week features for each sample.

    Returns [n_samples, N_nodes, 4] with:
      - sin(2π * time_of_day), cos(2π * time_of_day)
      - sin(2π * day_of_week/7), cos(2π * day_of_week/7)

    METR-LA: 5-min intervals → 288 steps/day, starts March 1 2012 (Thursday=3)
    PEMS-BAY: same 5-min interval structure

    Args:
        n_samples: Number of time steps/snapshots
        N_nodes: Number of sensors/nodes
        steps_per_day: Number of measurement intervals per day (288 for 5-min)
        start_step: Starting time step index (for correct day-of-week alignment)

    Returns:
        [n_samples, N_nodes, 4] float32 array (same features for all nodes
        at each time step — broadcast along node dimension)
    """
    step_ids = np.arange(start_step, start_step + n_samples)
    tod = (step_ids % steps_per_day) / steps_per_day  # [0, 1)
    dow = ((step_ids // steps_per_day + 3) % 7) / 7.0  # Thursday=3 start, [0, 1)
    # [n_samples, 4]
    features = np.stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
    ], axis=1).astype(np.float32)
    # Broadcast to [n_samples, N_nodes, 4] — same for all nodes at each time step
    return np.tile(features[:, np.newaxis, :], (1, N_nodes, 1))


def traffic_to_asann_format(
    X_all: np.ndarray, Y_all: np.ndarray, adj: np.ndarray,
    train_ratio: float = 0.7, val_ratio: float = 0.1,
    add_time_features: bool = False, steps_per_day: int = 288,
    ar_step_size: int = 0,
    anchor_horizons: Optional[List[int]] = None,
) -> Tuple[Dict, Dict]:
    """Convert traffic data to ASANN format preserving graph-snapshot structure.

    Each sample is a full graph snapshot: [N_nodes, window] -> [N_nodes, horizon].
    This preserves spatial structure so graph ops (adj[N,N]) can operate correctly.

    Temporal split (no shuffle -- time series, no data leakage).
    Standardization computed on training set only.

    Args:
        X_all: [n_samples, N_nodes, window]
        Y_all: [n_samples, N_nodes, horizon]
        adj: [N_nodes, N_nodes] adjacency matrix
        add_time_features: If True, appends 4 sin/cos time features to input
        steps_per_day: Measurements per day (288 for 5-min intervals)
        ar_step_size: If > 0, enable auto-regressive mode. Training targets are
            truncated to first ar_step_size steps (d_output=ar_step_size).
            Full-horizon targets saved separately for auto-regressive eval.
        anchor_horizons: If set, select specific horizon indices as targets
            (e.g. [2, 5, 11] for H3, H6, H12). d_output = len(anchor_horizons).
            Full-horizon targets saved for interpolated evaluation.
            Takes precedence over ar_step_size.

    Returns:
        (split_data, graph_info) where split_data has 3D train/val/test tensors.
    """
    n_samples, N, window = X_all.shape
    horizon = Y_all.shape[2]

    # Optionally append time-of-day / day-of-week features
    if add_time_features:
        time_feats = compute_time_features(n_samples, N, steps_per_day)
        X_all = np.concatenate([X_all, time_feats], axis=-1)  # [n_samples, N, window+4]
        window = X_all.shape[2]  # updated window size
        print(f"  Added time features: window {window - 4} -> {window} "
              f"(+4 sin/cos tod/dow)")

    # Temporal split (no shuffle -- time series!)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    # Keep 3D: [n_time, N, features] -- each sample is a full graph snapshot
    X_train = torch.tensor(X_all[:n_train], dtype=torch.float32)
    Y_train = torch.tensor(Y_all[:n_train], dtype=torch.float32)
    X_val = torch.tensor(X_all[n_train:n_train + n_val], dtype=torch.float32)
    Y_val = torch.tensor(Y_all[n_train:n_train + n_val], dtype=torch.float32)
    X_test = torch.tensor(X_all[n_train + n_val:], dtype=torch.float32)
    Y_test = torch.tensor(Y_all[n_train + n_val:], dtype=torch.float32)

    # Standardize features: compute stats over [n_train, N, window] -> per-feature
    # Reshape to [n_train*N, window] for stats, then apply to 3D tensors
    x_flat = X_train.reshape(-1, window)
    x_mean = x_flat.mean(dim=0, keepdim=True)  # [1, window]
    x_std = x_flat.std(dim=0, keepdim=True).clamp(min=1e-6)  # [1, window]
    X_train = (X_train - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)
    X_val = (X_val - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)
    X_test = (X_test - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)

    # Standardize targets
    y_flat = Y_train.reshape(-1, horizon)
    y_mean = y_flat.mean(dim=0, keepdim=True)  # [1, horizon]
    y_std = y_flat.std(dim=0, keepdim=True).clamp(min=1e-6)  # [1, horizon]
    Y_train = (Y_train - y_mean.unsqueeze(0)) / y_std.unsqueeze(0)
    Y_val = (Y_val - y_mean.unsqueeze(0)) / y_std.unsqueeze(0)
    Y_test = (Y_test - y_mean.unsqueeze(0)) / y_std.unsqueeze(0)

    # Build edge_index from adjacency (including weighted edges)
    edge_src, edge_dst = np.where(adj > 0)
    edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
    degree = torch.tensor(adj.sum(axis=1), dtype=torch.float32)

    # Anchor-horizon mode: select specific horizon indices (e.g. H3, H6, H12)
    if anchor_horizons is not None:
        anchor_idx = anchor_horizons
        d_output_effective = len(anchor_idx)
        anchor_names = [f"H{i+1}" for i in anchor_idx]
        print(f"  Anchor-horizon mode: predicting {anchor_names} "
              f"(indices {anchor_idx}), d_output={d_output_effective}")

        split_data = {
            'X_train': X_train, 'y_train': Y_train[:, :, anchor_idx],
            'X_val': X_val, 'y_val': Y_val[:, :, anchor_idx],
            'X_test': X_test, 'y_test': Y_test[:, :, anchor_idx],
            'y_mean': y_mean.squeeze(0)[anchor_idx],
            'y_std': y_std.squeeze(0)[anchor_idx],
            'y_mean_full': y_mean.squeeze(0),
            'y_std_full': y_std.squeeze(0),
            'd_input': window, 'd_output': d_output_effective,
            'N_nodes': N,
            'total_horizon': horizon,
            'ar_step_size': 0,
            'anchor_horizons': anchor_idx,
            # Full targets for interpolated eval
            'y_test_full': Y_test,
            'y_val_full': Y_val,
        }
    else:
        # Auto-regressive mode: truncate targets for training, save full for eval
        d_output_effective = horizon
        if ar_step_size > 0 and ar_step_size < horizon:
            assert horizon % ar_step_size == 0, \
                f"horizon ({horizon}) must be divisible by ar_step_size ({ar_step_size})"
            d_output_effective = ar_step_size
            print(f"  Auto-regressive mode: step_size={ar_step_size}, "
                  f"n_steps={horizon // ar_step_size}, total_horizon={horizon}")

        split_data = {
            'X_train': X_train, 'y_train': Y_train[:, :, :d_output_effective],
            'X_val': X_val, 'y_val': Y_val[:, :, :d_output_effective],
            'X_test': X_test, 'y_test': Y_test[:, :, :d_output_effective],
            'y_mean': y_mean.squeeze(0)[:d_output_effective],
            'y_std': y_std.squeeze(0)[:d_output_effective],
            'y_mean_full': y_mean.squeeze(0),
            'y_std_full': y_std.squeeze(0),
            'd_input': window, 'd_output': d_output_effective,
            'N_nodes': N,
            'total_horizon': horizon,
            'ar_step_size': ar_step_size,
        }
        # Save full-horizon targets for auto-regressive evaluation
        if ar_step_size > 0 and ar_step_size < horizon:
            split_data['y_test_full'] = Y_test
            split_data['y_val_full'] = Y_val

    graph_info = {
        'edge_index': edge_index,
        'num_nodes': N,
        'degree': degree,
    }

    print(f"  Traffic ASANN format: train={n_train} snapshots, val={n_val}, "
          f"test={n_test}, nodes={N}, features={window}, targets={horizon}")

    return split_data, graph_info


class TrafficScaler:
    """Simple scaler for inverse-transforming traffic predictions to original scale.

    Works with the trainer's y_scaler interface (inverse_transform on numpy arrays).
    """

    def __init__(self, y_mean: torch.Tensor, y_std: torch.Tensor):
        self.y_mean = y_mean.numpy()  # [horizon]
        self.y_std = y_std.numpy()    # [horizon]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.y_std + self.y_mean


def _snapshot_collate(batch):
    """Custom collate for graph snapshot DataLoader.

    With batch_size=1, squeezes the batch dim so the model sees [N_nodes, features].
    With batch_size>1, flattens [B, N, d] -> [B*N, d] so graph ops can detect
    batching via x.shape[0] > num_nodes and reshape internally.
    """
    xs, ys = zip(*batch)
    # Stack adds batch dim: [B, N, d]
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    if x.shape[0] == 1:
        # batch_size=1: squeeze to [N, d] — backward compatible
        return x.squeeze(0), y.squeeze(0)
    # batch_size>1: flatten to [B*N, d] for graph ops
    B, N, d = x.shape
    return x.reshape(B * N, d), y.reshape(B * N, -1)


def create_traffic_dataloaders(
    split_data: Dict,
    batch_size: int = 1,
    subsample_train: int = 0,
    subsample_val: int = 0,
    subsample_test: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for traffic graph-snapshot data.

    Each sample is a full graph snapshot [N_nodes, features].
    With batch_size>1, collate flattens [B,N,d] -> [B*N,d] and graph ops
    internally reshape to [N,B*d] for batched sparse matmul (same adjacency
    for all snapshots in a batch).

    Args:
        split_data: Dict from traffic_to_asann_format with 3D tensors
        batch_size: Number of graph snapshots per batch (default=1 for backward compat)
        subsample_train: If > 0, randomly subsample training snapshots per epoch
        subsample_val: If > 0, randomly subsample val snapshots for faster eval
        subsample_test: If > 0, randomly subsample test snapshots for faster eval

    Returns:
        (train_loader, val_loader, test_loader)
    """
    X_train = split_data['X_train']
    Y_train = split_data['y_train']
    X_val = split_data['X_val']
    Y_val = split_data['y_val']
    X_test = split_data['X_test']
    Y_test = split_data['y_test']

    # Subsample splits for speed using strided (evenly-spaced) sampling.
    # Traffic data is temporal -- random subsampling can bias toward certain
    # times-of-day or days-of-week. Strided sampling preserves the temporal
    # distribution so train/val/test all see the same mix of morning, evening,
    # weekday, weekend patterns.
    if subsample_train > 0 and subsample_train < X_train.shape[0]:
        stride = max(1, X_train.shape[0] // subsample_train)
        indices = torch.arange(0, X_train.shape[0], stride)[:subsample_train]
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        print(f"  Subsampled training set: {len(indices)} snapshots (stride={stride})")

    if subsample_val > 0 and subsample_val < X_val.shape[0]:
        stride = max(1, X_val.shape[0] // subsample_val)
        indices = torch.arange(0, X_val.shape[0], stride)[:subsample_val]
        X_val = X_val[indices]
        Y_val = Y_val[indices]
        print(f"  Subsampled val set: {len(indices)} snapshots (stride={stride})")

    if subsample_test > 0 and subsample_test < X_test.shape[0]:
        stride = max(1, X_test.shape[0] // subsample_test)
        indices = torch.arange(0, X_test.shape[0], stride)[:subsample_test]
        X_test = X_test[indices]
        Y_test = Y_test[indices]
        print(f"  Subsampled test set: {len(indices)} snapshots (stride={stride})")

    train_ds = GraphSnapshotDataset(X_train, Y_train)
    val_ds = GraphSnapshotDataset(X_val, Y_val)
    test_ds = GraphSnapshotDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_snapshot_collate, drop_last=(batch_size > 1))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_snapshot_collate, drop_last=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_snapshot_collate, drop_last=False)

    print(f"  Traffic DataLoaders: batch_size={batch_size}, "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_loader, val_loader, test_loader
