# ASANN: Adaptive Self-Architecting Neural Network

**Neural network architecture as an outcome of training under a rule-based controller**

ASANN is a training loop in which a rule-based controller, driven by signals computed during back-propagation, grows and prunes the depth, width, connectivity and per-layer operations of a neural network during a single training run. Training starts from a minimal seed network. The configuration of a run is derived by fixed rules from task descriptors, and the framework was evaluated on 85 datasets in seven data modalities (tabular, image, physics-informed, graph, spatio-temporal, molecular and haematological).

This repository contains the complete source code, experiment scripts, and figure-generation code accompanying the article:

> **Neural network architecture as an outcome of training under a rule-based controller**

---

## Table of Contents

- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data and Experiment Results](#data-and-experiment-results)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Core Architecture](#core-architecture)
- [Extending ASANN: Adding New Operations](#extending-asann-adding-new-operations)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- **Rule-derived configuration**: The settings of a run are derived by fixed rules from task descriptors (modality, input and output dimensions, number of training samples); there is one rule-based configuration per modality and no dataset-specific tuning.
- **Architecture modified during training**: The architecture changes at sparse modification checkpoints of a single training run. Five signals are computed at every checkpoint (Gradient Demand Score, Neuron Utility Score, Layer Saturation Score, Cross-Layer Gradient Correlation and a loss-stationarity flag); the first four gate the structural changes and the flag is logged.
- **Operation selection**: The vocabulary holds 88 registered operations (activations, normalisations, convolutions, attention mechanisms, graph operators, physics-specific layers, KAN and others), offered through modality-specific candidate pools. At each checkpoint the candidates of a layer's pool are evaluated and kept only if they reduce the loss.
- **Monitoring and interventions**: A monitoring module classifies training as healthy, failing or recovering with fourteen threshold rules, one per failure mode (overfitting, underfitting, stagnation, capacity exhaustion, class imbalance and others), and applies interventions from an ordered ladder of 43 interventions in five levels. The code uses medical names for these parts (diagnosis, surgery, treatment); the article calls them monitoring, modification and intervention.
- **Reproducibility across seeds**: Five-seed campaigns under identical splits and configuration are reported in the article; for example, all 30 seeded runs on six MoleculeNet benchmarks ended with two-layer networks.
- **Extensible vocabulary**: Any primitive implemented as a standard layer can be added to the operation vocabulary and evaluated under the same protocol.
- **Custom CUDA kernels**: Twenty-eight fused CUDA operations accelerate training with automatic fallback to pure PyTorch when CUDA is unavailable.

---

## Repository Structure

```
asann/
├── asann/                        # Core ASANN package
│   ├── __init__.py               # Public API exports
│   ├── model.py                  # ASANNModel, OperationPipeline, GatedOperation
│   ├── surgery.py                # SurgeryEngine, operation vocabulary, KANLinearOp
│   ├── trainer.py                # ASANNTrainer (training orchestration)
│   ├── diagnosis.py              # DiagnosisEngine (health monitoring)
│   ├── treatments.py             # TreatmentPlanner (adaptive interventions)
│   ├── scheduler.py              # SurgeryScheduler (structural plasticity annealing)
│   ├── meta_learner.py           # MetaLearner (co-adapts surgery thresholds)
│   ├── config.py                 # ASANNConfig (configuration; all parameters have defaults)
│   ├── encoders.py               # Input encoders (tabular, convolutional, graph, Fourier)
│   ├── asann_optimizer.py        # ASANNOptimizer (multi-scale momentum, hypergradient LR)
│   ├── lr_controller.py          # Adaptive learning-rate controller
│   ├── warmup_scheduler.py       # Warmup scheduler with cosine annealing
│   ├── loss.py                   # ASANNLoss (task-aware loss wrapper)
│   ├── lab.py                    # PatientHistory, LabDiagnostics
│   ├── lab_tests.py              # Default diagnostic lab configuration
│   └── logger.py                 # SurgeryLogger (records every structural event)
│
├── asann_cuda/                   # Optional CUDA-accelerated operations
│   ├── setup.py                  # Build script for CUDA extensions
│   ├── __init__.py               # CUDA toolkit auto-detection and exports
│   ├── kernels/                  # CUDA kernel source files (.cu, .cuh)
│   ├── bindings/                 # C++/Python bindings (asann_cuda_ops.cpp)
│   ├── ops/                      # Python wrappers for 28 CUDA operations
│   └── tests/                    # CUDA operation unit tests
│
├── experiments/                  # All experiment scripts (one file per dataset)
│   ├── common.py                 # Shared utilities (data splitting, device setup)
│   ├── tier_1/                   # Tabular regression
│   ├── tier_2/                   # Tabular classification
│   ├── tier_3/                   # Image classification
│   ├── tier_5/                   # Physics-informed PDEs
│   ├── tier_6/                   # Graph and spatio-temporal
│   ├── tier_7/                   # Molecular, pharmacogenomic, haematological
│   └── results/                  # Saved results (download separately, see below)
│
├── compat/                      # Backward-compatibility shims for loading older checkpoints
│   └── __init__.py              # Meta-path finder for old model unpickling
│
├── utils/                       # Utility scripts
│   └── create_benchmark_table.py  # Benchmark table generation
│
└── data/                        # Datasets (download separately, see below)
```

---

## Requirements

### Software

- **Python** >= 3.10
- **PyTorch** >= 2.0 (with CUDA support for GPU training). The experiments of the article used Python 3.12, PyTorch 2.8.0 and CUDA 12.8 on NVIDIA RTX 5090 GPUs.
- **CUDA Toolkit** >= 12.1 (only if building custom CUDA kernels)
- **Ninja** build system (only if building custom CUDA kernels)

### Hardware

- **GPU**: Any NVIDIA GPU with compute capability >= 6.0 (Pascal or newer). Tested on RTX 3090, RTX 4090, RTX 5090, A100, and H100.
- **RAM for CUDA build**: Building the custom CUDA extensions with Ninja requires approximately **32 GB of system RAM** due to parallel kernel compilation. If your system has less RAM, you can either (a) set `MAX_JOBS=1` to compile kernels sequentially (slower but uses less memory), or (b) skip the CUDA build entirely — ASANN will fall back to pure PyTorch operations automatically.
- **RAM for training**: 16 GB system RAM is sufficient for all experiments. GPU VRAM requirements depend on the dataset (8 GB is sufficient for most experiments; 24 GB recommended for image and large molecular tasks).

### Python packages

```
torch>=2.0
numpy
scipy
scikit-learn
pandas
openpyxl
rdkit              # molecular experiments only
torch-geometric    # graph experiments only
torch-scatter      # graph experiments only
torch-sparse       # graph experiments only
deepchem           # MoleculeNet data loading
ogb                # Open Graph Benchmark datasets
h5py               # HDF5 data files
torchvision        # image experiments
matplotlib         # figure generation
```

---

## Installation

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/navid-nsk/asann.git
cd asann
pip install torch numpy scipy scikit-learn pandas openpyxl matplotlib
```

### 2. (Optional) Build custom CUDA kernels

The CUDA kernels provide fused operations that accelerate training. ASANN works without them — it falls back to equivalent pure-PyTorch implementations automatically.

**Prerequisites:**
- NVIDIA CUDA Toolkit >= 12.1 installed and on your PATH
- Ninja build system (`pip install ninja`)
- ~32 GB system RAM (Ninja compiles kernels in parallel)

```bash
cd asann_cuda
python setup.py install
cd ..
```

If you encounter out-of-memory errors during compilation:

```bash
MAX_JOBS=1 python setup.py install
```

To verify the CUDA build:

```bash
cd asann_cuda
python -m pytest tests/ -v
```

**Supported GPU architectures** (automatically selected during build):

| Compute Capability | GPU Family |
|---|---|
| 6.0 | Pascal (P100, Titan X) |
| 7.0 | Volta (V100) |
| 7.5 | Turing (RTX 2000 series) |
| 8.0 | Ampere (A100) |
| 8.6 | Ampere (RTX 3000 series) |
| 8.9 | Ada Lovelace (RTX 4000 series) |
| 9.0 | Hopper (H100) |

---

## Data and Experiment Results

The datasets (`./data/`) and pre-computed experiment results (`./experiments/results/`) are hosted separately due to their size. Download them from:

**https://doi.org/10.6084/m9.figshare.31738417**

The download contains zip files. Extract them into the repository root so that the directory structure matches:

```
asann/
├── data/                         # Extract data.zip here
│   ├── California_Housing/
│   ├── MoleculeNet/
│   ├── METR-LA/
│   ├── PEMS-BAY/
│   ├── PINNacle-main/
│   ├── Munich_Leukemia_Lab/
│   └── ...
└── experiments/
    └── results/                  # Extract results.zip here
        ├── tier_1/
        ├── tier_2/
        ├── tier_3/
        └── ...
```

---

## Quick Start

### Minimal example: Tabular regression

```python
import torch
from asann import ASANNConfig, ASANNModel, ASANNTrainer

# Load your data (X: features, y: targets)
# Split into train/val/test tensors and create DataLoaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=256, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val),
    batch_size=256
)

# Create model — starts as a 2-layer ReLU network
config = ASANNConfig()
model = ASANNModel(d_input=X_train.shape[1], d_output=1, config=config)

# Train — the architecture is modified during training
trainer = ASANNTrainer(
    model, config,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=torch.nn.MSELoss(),
    task_type="regression"
)
trainer.train_epochs(max_epochs=300)

# Inspect the architecture at the end of training
print(model.describe_architecture())
```

### Running a pre-built experiment

```bash
# Run a single experiment (e.g., California Housing)
python experiments/tier_1/exp_1a_california.py

# Run all experiments in a tier
python experiments/tier_1/run_all.py
```

---

## Running Experiments

Each tier contains standalone experiment scripts that can be run directly. The configuration of every experiment is derived by the same rules from the task descriptors (modality, input and output dimensions, number of training samples).

| Tier | Domain | Experiments | Script prefix |
|---|---|---|---|
| 1 | Tabular regression | OpenML regression datasets | `exp_1*.py` |
| 2 | Tabular classification | OpenML classification datasets | `exp_2*.py` |
| 3 | Image classification | MNIST, Fashion-MNIST, KMNIST, SVHN, CIFAR-10, CIFAR-100, STL-10 | `exp_3*.py` |
| 5 | Physics-informed PDEs | PINNacle problems (Burgers, Kuramoto–Sivashinsky, Poisson, heat, Gray–Scott, wave) | `exp_5*.py` |
| 6 | Graph / spatio-temporal | CiteSeer, PubMed, METR-LA, PEMS-BAY | `exp_6*.py` |
| 7 | Molecular / biomedical | MoleculeNet, GDSC2, leukaemia cell classification | `exp_7*.py` |

To run all experiments in a tier:

```bash
python experiments/tier_1/run_all.py    # All tabular regression
python experiments/tier_3/run_all.py    # All image classification
python experiments/tier_5/run_all.py    # All PDE experiments
python experiments/tier_6/run_all.py    # All graph experiments
```

Results (metrics, surgery traces, model checkpoints) are saved automatically to `experiments/results/<tier>/<dataset>/`.

---

## Core Architecture

ASANN has three subsystems that run at the modification (surgery) checkpoints during training:

### 1. Diagnosis

Five signals are computed at every checkpoint from the current gradients, activations and training loss:

- **Gradient Demand Score (GDS)**: Mean per-neuron gradient magnitude per layer. High GDS indicates a layer needs more capacity.
- **Neuron Utility Score (NUS)**: Product of incoming weight norm, outgoing weight norm, and mean activation magnitude. Low NUS neurons are candidates for pruning.
- **Layer Saturation Score (LSS)**: Ratio of the norm of a layer's output change to the norm of its input. A high mean LSS together with flat losses signals capacity exhaustion; a layer whose contribution stays near zero is a candidate for removal.
- **Loss-stationarity flag**: Set when the training loss has stopped improving over a recent window; it is logged at every checkpoint.
- **Cross-Layer Gradient Correlation (CLGC)**: Correlation between gradient vectors at different layers. High correlation between non-adjacent layers suggests a skip connection would help.

The monitoring module also classifies the training state from recent training and validation losses: fourteen conjunctive threshold rules map them to fourteen failure modes (overfitting, underfitting, stagnation, capacity exhaustion, memorisation, class imbalance and others), each with a severity level.

### 2. Surgery

When the model is healthy and the surgery interval has elapsed, the surgery engine proposes structural modifications:

- **Width changes**: Neurons are added (by splitting the highest-norm neuron) or pruned (by removing low-utility neurons) based on GDS and NUS thresholds.
- **Depth changes**: New layers are inserted near the identity when the mean LSS exceeds its threshold and the model has been healthy for several consecutive assessments.
- **Operation probing**: Candidate operations from the layer's modality pool are temporarily inserted, and their benefit is measured as the immediate loss reduction. Operations are accepted only if their benefit exceeds a threshold that tightens over training.
- **Skip connections**: Created between layers with high cross-layer gradient correlation, initialised with zero-scale projections.

Every proposed modification is verified before acceptance: the system compares the loss before and after the change and reverts modifications that do not improve performance.

### 3. Treatment

When the monitoring module detects a failure mode, the treatment planner selects an intervention from the ordered ladder of that failure mode (43 interventions in five levels):

- **Level 1** (parametric): Adjust dropout, weight decay, or learning rate.
- **Level 2** (normalisation and operation packages): Insert BatchNorm, label smoothing, or domain-specific operations.
- **Level 3** (structural): Add or remove layers, widen or narrow layers, or add a ResNet block.
- **Level 4** (aggressive): Combined regularisation, soft architecture reset, or simultaneous depth and width expansion.
- **Level 5**: Weight re-initialisation.

A meta-controller (class `MetaLearner`) adjusts the surgery interval and tightens the acceptance thresholds during training.

### Operation Vocabulary (examples)

| Category | Operations |
|---|---|
| Activations | ReLU, GELU, Swish, Mish, Sigmoid, Tanh |
| Normalisations | BatchNorm, LayerNorm, GroupNorm |
| Regularisations | Dropout, DropPath |
| Convolutions | Conv1d, depthwise-separable Conv2d, pointwise 1x1, dilated Conv1d |
| Attention | Squeeze-and-excitation, multi-head self-attention, cross-attention |
| Temporal | Temporal differencing, exponential moving average |
| Recurrence | GRU |
| Graph | GCN, GAT, GIN, SGC, GraphSAGE, spectral Chebyshev, graph diffusion |
| Physics | Derivative convolution (configurable order), polynomial expansion (configurable degree), branched diffusion-reaction |
| KAN | KANLinearOp (radial basis function variant) |

Supplementary Data 2 of the article lists all 88 registered operations. Operations are filtered by modality at each surgery checkpoint: spatial layers receive only spatially compatible operations, graph layers receive only message-passing operations, and physics operations are offered only when the physics flag is enabled.

---

## Extending ASANN: Adding New Operations

Adding a new operation to the vocabulary requires three steps. The controller then evaluates it under the same protocol as every other candidate.

### Step 1: Define the operation class

Create a new `nn.Module` in `asann/surgery.py` (or in a separate file and import it). The operation must accept a tensor of shape `(batch, features)` and return a tensor of the same shape:

```python
class MyNewOp(nn.Module):
    """A custom operation for ASANN's vocabulary."""

    def __init__(self, d_model: int, my_param: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.my_param = my_param
        # Define any learnable parameters
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your operation logic here
        return x + self.my_param * self.linear(x)
```

**Requirements:**
- Input and output dimensions must match (`d_model` in, `d_model` out).
- The operation should be differentiable.
- Include `d_model` as a constructor argument so the surgery engine can resize it when neurons are added or pruned.

### Step 2: Register in the operation vocabulary

In `asann/surgery.py`, locate the `_get_candidate_operations` method (around line 3100). Add your operation to the appropriate dictionary:

```python
# Inside _get_candidate_operations(), in the general_ops dictionary:
general_ops = {
    # ... existing operations ...
    "my_new_op": lambda: MyNewOp(d, my_param=0.5),
}
```

The key `"my_new_op"` is the name that will appear in surgery logs and architecture descriptions. The lambda defers construction until the operation is actually probed.

### Step 3: Handle resizing (for width changes)

When ASANN adds or removes neurons from a layer, it must resize all operations in that layer's pipeline. In `asann/surgery.py`, locate the `_resize_operation` function (around line 6300) and add a case for your operation:

```python
elif isinstance(op, MyNewOp):
    new_op = MyNewOp(new_d, my_param=op.my_param).to(device)
    # Optionally copy existing weights (partial transfer)
    with torch.no_grad():
        min_d = min(op.d_model, new_d)
        new_op.linear.weight[:min_d, :min_d] = op.linear.weight[:min_d, :min_d]
        new_op.linear.bias[:min_d] = op.linear.bias[:min_d]
    return new_op
```

### What happens next

Once registered, ASANN will:

1. **Probe** your operation at surgery checkpoints by temporarily inserting it and measuring the loss change.
2. **Accept** it only if the benefit exceeds the current threshold (which tightens over training).
3. **Protect** it for three surgery intervals after insertion (parametric operations get an immunosuppression grace period).
4. **Remove** it later if it stops contributing (removal requires 3x the benefit threshold of insertion).
5. **Report** adoption rates across domains, layers, and training phases in the surgery logs.

This means you do not need to decide which tasks, layers, or training phases benefit from your operation; it goes through the same evaluation as every other candidate.

### Example: What happened with KAN

The Kolmogorov–Arnold layer (KANLinearOp, radial basis function variant) was added to the vocabulary as one candidate with no privileged status. In the 31 single-seed tabular regression runs of the article it was adopted in 20 runs and retained at the end of training in 17; its adoption in the other modalities is reported in the article.

---

## Citation

If you use this code, please cite the article:

> Mashhadi Moghaddam, S. N., Sawada, M., Knudby, A. & Cao, H. Neural network architecture as an outcome of training under a rule-based controller.

## License

This code is released under the MIT License (see `LICENSE`).
