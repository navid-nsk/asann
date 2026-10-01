"""
Tier 5: PDE Data Generation Utilities
======================================

Generates synthetic datasets from known PDE solutions for regression
experiments.

All generators produce (X, y) pairs where:
  X = input features (spatial coordinates, time, initial conditions)
  y = target values (PDE solution or time-derivative)

The data is structured as 1D regression: X is a flat feature vector,
y is the quantity to predict.
"""

import numpy as np
from typing import Tuple, Optional


def generate_heat_data(
    nx: int = 200,
    nt: int = 200,
    L: float = 2 * np.pi,
    T_max: float = 1.0,
    alpha: float = 0.1,
    n_modes: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from the 1D heat equation: u_t = alpha * u_xx.

    Uses Fourier series analytical solution:
        u(x, t) = sum_k a_k * exp(-alpha * (k*pi/L)^2 * t) * sin(k*pi*x/L)

    Input features X[i] = [x, t, u(x,t)]  (3 features)
    Target y[i] = u_t(x,t) = alpha * u_xx(x,t)

    The model learns to map (x, t, u) -> u_t.  If ASANN discovers the PDE
    u_t = alpha * u_xx, the discovered coefficients should show u_xx dominant.
    """
    rng = np.random.RandomState(seed)

    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T_max, nt)
    xx, tt = np.meshgrid(x, t, indexing='ij')  # [nx, nt]

    # Random Fourier coefficients for initial condition
    a_k = rng.randn(n_modes) * 0.5

    u = np.zeros((nx, nt))
    u_t = np.zeros((nx, nt))

    for k_idx in range(n_modes):
        k = k_idx + 1
        lam = (k * np.pi / L) ** 2
        decay = np.exp(-alpha * lam * tt)
        mode = np.sin(k * np.pi * xx / L)
        u += a_k[k_idx] * decay * mode
        u_t += -alpha * lam * a_k[k_idx] * decay * mode

    # Flatten to (N, 3) input and (N, 1) target
    X = np.stack([xx.ravel(), tt.ravel(), u.ravel()], axis=1).astype(np.float32)
    y = u_t.ravel().astype(np.float32)

    return X, y


def generate_wave_data(
    nx: int = 200,
    nt: int = 200,
    L: float = 2 * np.pi,
    T_max: float = 2.0,
    c: float = 1.0,
    n_modes: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from the 1D wave equation: u_tt = c^2 * u_xx.

    D'Alembert-type Fourier solution with zero initial velocity:
        u(x, t) = sum_k a_k * cos(c * k * pi * t / L) * sin(k * pi * x / L)

    Input X[i] = [x, t, u(x,t)]  (3 features)
    Target y[i] = u_tt(x,t) = c^2 * u_xx(x,t)
    """
    rng = np.random.RandomState(seed)

    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T_max, nt)
    xx, tt = np.meshgrid(x, t, indexing='ij')

    a_k = rng.randn(n_modes) * 0.5

    u = np.zeros((nx, nt))
    u_tt = np.zeros((nx, nt))

    for k_idx in range(n_modes):
        k = k_idx + 1
        omega = c * k * np.pi / L
        lam_xx = -(k * np.pi / L) ** 2
        temporal = np.cos(omega * tt)
        spatial = np.sin(k * np.pi * xx / L)
        u += a_k[k_idx] * temporal * spatial
        # u_tt = -omega^2 * a_k * cos(omega*t) * sin(kpx/L)
        u_tt += -(omega ** 2) * a_k[k_idx] * temporal * spatial

    X = np.stack([xx.ravel(), tt.ravel(), u.ravel()], axis=1).astype(np.float32)
    y = u_tt.ravel().astype(np.float32)

    return X, y


def generate_burgers_data(
    nx: int = 256,
    nt: int = 200,
    L: float = 2 * np.pi,
    T_max: float = 0.5,
    nu: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from the 1D viscous Burgers equation: u_t + u*u_x = nu*u_xx.

    Uses Cole-Hopf transform for exact solution:
        phi(x,t) = sum_k a_k * exp(-nu * (k*pi/L)^2 * t) * sin(k*pi*x/L) + C
        u(x,t) = -2*nu * phi_x / phi

    Target y[i] = u_t(x,t) computed via finite differences from the exact solution.

    Input X[i] = [x, t, u(x,t)]  (3 features)
    Target y[i] = u_t(x,t)

    Discovery should find: u*u_x and u_xx as dominant terms.
    """
    rng = np.random.RandomState(seed)

    x = np.linspace(0, L, nx, endpoint=False)
    dt = T_max / nt
    t = np.linspace(dt, T_max, nt)  # start at dt to avoid phi=0 issues
    xx, tt = np.meshgrid(x, t, indexing='ij')

    # Build Cole-Hopf potential phi
    n_modes = 5
    a_k = rng.randn(n_modes) * 0.3
    C = 10.0  # large constant ensures phi > 0

    phi = np.full((nx, nt), C)
    phi_x = np.zeros((nx, nt))

    for k_idx in range(n_modes):
        k = k_idx + 1
        lam = (k * np.pi / L) ** 2
        decay = np.exp(-nu * lam * tt)
        s = np.sin(k * np.pi * xx / L)
        c_mode = np.cos(k * np.pi * xx / L)
        phi += a_k[k_idx] * decay * s
        phi_x += a_k[k_idx] * decay * c_mode * (k * np.pi / L)

    # Cole-Hopf: u = -2*nu * phi_x / phi
    u = -2.0 * nu * phi_x / phi

    # Compute u_t via 2nd-order finite differences in time
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt

    X = np.stack([xx.ravel(), tt.ravel(), u.ravel()], axis=1).astype(np.float32)
    y = u_t.ravel().astype(np.float32)

    return X, y


def generate_poisson_data(
    nx: int = 100,
    ny: int = 100,
    L: float = np.pi,
    n_modes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from the 2D Poisson equation: u_xx + u_yy = f(x,y).

    Manufactured solution using Fourier modes:
        u(x,y) = sum_{j,k} a_{jk} * sin(j*pi*x/L) * sin(k*pi*y/L)
        f(x,y) = -((j*pi/L)^2 + (k*pi/L)^2) * u(x,y)   [Laplacian of u]

    Input X[i] = [x_coord, y_coord, u(x,y)]   (3 features)
    Target y[i] = f(x,y) = Laplacian(u)

    Note: Poisson is time-independent (elliptic). Despite being 2D in space,
    the data is presented as a flat regression problem: N = nx*ny samples,
    d_input = 3.  ASANN should discover u_xx + u_yy structure.
    """
    rng = np.random.RandomState(seed)

    x = np.linspace(0, L, nx, endpoint=False)
    y_coord = np.linspace(0, L, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y_coord, indexing='ij')

    # Random mode coefficients
    a_jk = rng.randn(n_modes, n_modes) * 0.3

    u = np.zeros((nx, ny))
    f = np.zeros((nx, ny))

    for j in range(1, n_modes + 1):
        for k in range(1, n_modes + 1):
            spatial = np.sin(j * np.pi * xx / L) * np.sin(k * np.pi * yy / L)
            lam = (j * np.pi / L) ** 2 + (k * np.pi / L) ** 2
            u += a_jk[j - 1, k - 1] * spatial
            f += -lam * a_jk[j - 1, k - 1] * spatial

    X = np.stack([xx.ravel(), yy.ravel(), u.ravel()], axis=1).astype(np.float32)
    y_out = f.ravel().astype(np.float32)

    return X, y_out


# ==================== PINNacle Data Loaders ====================


def _default_ref_dir() -> str:
    """Return the default PINNacle ref/ directory path."""
    import os
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "PINNacle-main", "ref",
    )


def _parse_comsol_time_data(
    filepath: str,
    spatial_dim: int,
    output_dim: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse COMSOL time-encoded .dat files into (X, y) regression datasets.

    COMSOL files have a header like:
        % X  Y  @comp1 t=0.0 @comp1 t=0.1 @comp1 t=0.2 ...
    The data portion has columns: [x, (y), u@t0, u@t1, u@t2, ...]

    This function expands the time-encoded data into a flat regression dataset:
        X[i] = [x, (y), t]   — spatial coordinates + time
        y[i] = [u]            — solution value(s) at that point and time

    Args:
        filepath: Path to the .dat file.
        spatial_dim: Number of spatial columns (1 for 1D PDEs, 2 for 2D).
        output_dim: Number of solution components (1 for scalar, 2 for coupled).

    Returns:
        (X, y) where X has shape (N*T, spatial_dim+1) and y has shape (N*T, output_dim).
    """
    import re

    # Parse header for time values
    times = []
    with open(filepath, 'rb') as f:
        for line_bytes in f:
            line = line_bytes.decode('latin-1').strip()
            if '@' in line and line.startswith('%'):
                parts = line.split('@')
                for p in parts[1:]:
                    m = re.search(r't=([0-9.eE+-]+)', p)
                    if m:
                        times.append(float(m.group(1)))
                break

    if not times:
        raise ValueError(f"No time values found in COMSOL header: {filepath}")

    # Load numerical data
    data = np.genfromtxt(filepath, comments='%', encoding='latin-1').astype(np.float32)
    n_nodes, n_cols = data.shape

    n_time = len(times) // output_dim  # times repeated per component
    if n_time == 0:
        n_time = len(times)

    spatial = data[:, :spatial_dim]  # [N, spatial_dim]
    u_all = data[:, spatial_dim:]     # [N, n_time * output_dim]

    t_arr = np.array(times[:n_time], dtype=np.float32)

    # Build flat dataset: each node at each timestep = one sample
    X_list = []
    y_list = []
    for ti in range(n_time):
        x_spatial = spatial.copy()
        t_col = np.full((n_nodes, 1), t_arr[ti], dtype=np.float32)
        u_at_t = u_all[:, ti * output_dim:(ti + 1) * output_dim]
        X_sample = np.concatenate([x_spatial, t_col], axis=1)
        X_list.append(X_sample)
        y_list.append(u_at_t)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# Dataset configurations for each PINNacle PDE experiment
_PINNACLE_DATASETS = {
    "poisson_classic": {
        "file": "poisson_classic.dat",
        "format": "simple",
        "input_cols": [0, 1],    # x, y
        "output_cols": [2],       # u
    },
    "burgers1d": {
        "file": "burgers1d.dat",
        "format": "comsol",
        "spatial_dim": 1,
        "output_dim": 1,
    },
    "heat_multiscale": {
        "file": "heat_multiscale.dat",
        "format": "comsol",
        "spatial_dim": 2,
        "output_dim": 1,
    },
    "kuramoto_sivashinsky": {
        "file": "Kuramoto_Sivashinsky.dat",
        "format": "simple",
        "input_cols": [0, 1],    # x, t
        "output_cols": [2],       # u
    },
    "poisson_boltzmann2d": {
        "file": "poisson_boltzmann2d.dat",
        "format": "simple",
        "input_cols": [0, 1],    # x, y
        "output_cols": [2],       # u
    },
    "grayscott": {
        "file": "grayscott.dat",
        "format": "simple",
        "input_cols": [0, 1, 2],  # x, y, t
        "output_cols": [3, 4],    # u, v
    },
    "wave_darcy": {
        "file": "wave_darcy.dat",
        "format": "comsol",
        "spatial_dim": 2,
        "output_dim": 1,
    },
}


def load_pinnacle_data(
    name: str,
    ref_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a PINNacle benchmark dataset by name.

    Handles both simple text (.dat) and COMSOL time-encoded (.dat) formats.

    Args:
        name: Dataset name (e.g., 'poisson_classic', 'burgers1d', 'grayscott').
        ref_dir: Directory containing .dat files. Defaults to data/PINNacle-main/ref/.
        max_samples: If set, randomly subsample to this many samples (for large datasets).
        seed: Random seed for subsampling.

    Returns:
        (X, y) where X is input features and y is target values, both as float32 arrays.
    """
    import os

    if name not in _PINNACLE_DATASETS:
        available = list(_PINNACLE_DATASETS.keys())
        raise ValueError(f"Unknown PINNacle dataset: {name}. Available: {available}")

    cfg = _PINNACLE_DATASETS[name]
    if ref_dir is None:
        ref_dir = _default_ref_dir()

    filepath = os.path.join(ref_dir, cfg["file"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PINNacle data file not found: {filepath}")

    if cfg["format"] == "simple":
        data = np.genfromtxt(filepath, comments='%', encoding='latin-1').astype(np.float32)
        X = data[:, cfg["input_cols"]]
        y = data[:, cfg["output_cols"]]
    elif cfg["format"] == "comsol":
        X, y = _parse_comsol_time_data(
            filepath,
            spatial_dim=cfg["spatial_dim"],
            output_dim=cfg["output_dim"],
        )
    else:
        raise ValueError(f"Unknown format: {cfg['format']}")

    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Subsample if requested (for very large datasets)
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def compute_pde_experiment_metrics(
    model,
    trainer=None,
    scheduler=None,
) -> dict:
    """Extract PDE-specific metrics from trained model.

    Scans the model architecture for derivative/polynomial/branched ops
    and collects lab/treatment history from the scheduler.
    """
    from asann.surgery import get_operation_name

    metrics = {}

    # Scan architecture for discovered operations
    deriv_ops = []
    poly_ops = []
    branched_ops = []
    all_ops = []

    for l_idx in range(model.num_layers):
        pipeline = model.ops[l_idx]
        for op in pipeline.operations:
            op_name = get_operation_name(op)
            all_ops.append((l_idx, op_name))
            if 'derivative' in op_name:
                deriv_ops.append((l_idx, op_name))
            if 'polynomial' in op_name:
                poly_ops.append((l_idx, op_name))
            if 'branched' in op_name:
                branched_ops.append((l_idx, op_name))

    metrics["derivative_ops_discovered"] = len(deriv_ops)
    metrics["polynomial_ops_discovered"] = len(poly_ops)
    metrics["branched_ops_discovered"] = len(branched_ops)
    metrics["discovered_ops_detail"] = deriv_ops + poly_ops + branched_ops
    metrics["total_ops"] = len(all_ops)
    metrics["architecture"] = all_ops

    # Collect lab/treatment stats from scheduler
    if scheduler is not None and hasattr(scheduler, 'patient_history'):
        ph = scheduler.patient_history
        metrics["lab_referrals"] = len(ph.lab_log)
        metrics["total_treatments"] = len(ph.treatment_log)
        metrics["total_diagnoses"] = len(ph.diagnosis_log)

        # Count treatments by type
        treatment_counts = {}
        for rec in ph.treatment_log:
            t_type = rec.treatment
            treatment_counts[t_type] = treatment_counts.get(t_type, 0) + 1
        metrics["treatment_counts"] = treatment_counts

    return metrics
