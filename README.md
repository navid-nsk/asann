# ASANN: Autonomously Self-Architecting Neural Network

**Architecture as an emergent property of neural network training**

ASANN is a framework that inverts the foundational assumption of neural architecture design. Instead of defining a fixed network structure before training begins, ASANN starts from a minimal two-layer seed and grows its own depth, width, connectivity, operations, and hyperparameters during training through iterative cycles of diagnosis, surgery, and treatment. The system requires zero per-task configuration and has been evaluated across more than seventy-five experiments spanning seven computational domains without modifying any hyperparameter, loss function, or training protocol between domains.

This repository contains the complete source code, experiment scripts, and figure-generation code accompanying the article:

> **Architecture as an emergent property of neural network training**

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

---

## Key Features

- **Zero-configuration generalisation**: A single set of hyperparameters works across tabular regression, image classification, graph learning, molecular property prediction, physics-informed PDEs, pharmacogenomics, and haematological cell classification.
- **Grow-during-train paradigm**: Architecture emerges from training dynamics rather than being searched before training begins. Five task-agnostic signals (gradient diversity, neuron utility, loss stationarity, layer contribution, cross-layer gradient correlation) drive all structural decisions.
- **Operation discovery**: A vocabulary of twenty-nine candidate operations (activations, normalisations, convolutions, attention mechanisms, graph operators, physics-specific layers, and KAN) is available for autonomous selection. The system discovers which operations help and which do not, per layer, per task.
- **Medical-metaphor diagnostic system**: A diagnostic engine inspired by clinical medicine detects ten pathology types (overfitting, underfitting, stagnation, capacity exhaustion, class imbalance, etc.) and prescribes graduated interventions across four treatment levels.
- **Reproducible architecture convergence**: Thirty independent runs on the same molecular benchmarks converge to architecturally identical networks with width coefficient of variation below four per cent.
- **Open experimental platform**: Any new architectural primitive can be added to the operation vocabulary and subjected to the same impartial, data-driven evaluation across arbitrary domains.
- **Custom CUDA kernels**: Twenty-eight fused CUDA operations accelerate training with automatic fallback to pure PyTorch when CUDA is unavailable.

---

## Repository Structure

```
asann_article/
├── asann/                        # Core ASANN package
│   ├── __init__.py               # Public API exports
│   ├── model.py                  # ASANNModel, OperationPipeline, GatedOperation
│   ├── surgery.py                # SurgeryEngine, 29-operation vocabulary, KANLinearOp
│   ├── trainer.py                # ASANNTrainer (training orchestration)
│   ├── diagnosis.py              # DiagnosisEngine (health monitoring)
│   ├── treatments.py             # TreatmentPlanner (adaptive interventions)
│   ├── scheduler.py              # SurgeryScheduler (structural plasticity annealing)
│   ├── meta_learner.py           # MetaLearner (co-adapts surgery thresholds)
│   ├── config.py                 # ASANNConfig (~400 hyperparameters, all with defaults)
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
│   ├── tier_1/                   # Tabular regression (27 datasets)
│   ├── tier_2/                   # Tabular classification (5 datasets)
│   ├── tier_3/                   # Image classification (7 datasets)
│   ├── tier_5/                   # Physics-informed PDEs (5 equations)
│   ├── tier_6/                   # Graph and spatio-temporal (4 datasets)
│   ├── tier_7/                   # Molecular, pharmacogenomic, haematological (12+ experiments)
│   └── results/                  # Saved results (download separately, see below)
│
├── compat/                      # Backward-compatibility shims (csann -> asann)
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
- **PyTorch** >= 2.0 (with CUDA support for GPU training)
- **CUDA Toolkit** >= 12.1 (only if building custom CUDA kernels)
- **Ninja** build system (only if building custom CUDA kernels)

### Hardware

- **GPU**: Any NVIDIA GPU with compute capability >= 6.0 (Pascal or newer). Tested on RTX 3090, RTX 4090, A100, and H100.
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
asann_article/
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

# Train — architecture grows autonomously
trainer = ASANNTrainer(
    model, config,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=torch.nn.MSELoss(),
    task_type="regression"
)
trainer.train_epochs(max_epochs=300)

# The model has now self-architected: inspect what it built
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

Each tier contains standalone experiment scripts that can be run directly. Every experiment uses the same ASANN configuration with minimal dataset-specific adjustments (input/output dimensions, dataset loading).

| Tier | Domain | Experiments | Script prefix |
|---|---|---|---|
| 1 | Tabular regression | 27 datasets | `exp_1*.py` |
| 2 | Tabular classification | 5 datasets | `exp_2*.py` |
| 3 | Image classification | 7 datasets (MNIST to STL-10) | `exp_3*.py` |
| 5 | Physics-informed PDEs | 5 equations (Burgers, KS, Poisson, Gray-Scott, Wave) | `exp_5*.py` |
| 6 | Graph / spatio-temporal | 4 datasets (CiteSeer, PubMed, METR-LA, PEMS-BAY) | `exp_6*.py` |
| 7 | Molecular / biomedical | 12+ experiments (MoleculeNet, GDSC2, Leukemia) | `exp_7*.py` |

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

ASANN operates through three interlocking subsystems that run at every surgery checkpoint during training:

### 1. Diagnosis

The diagnostic engine monitors five task-agnostic signals computed from local gradient and activation statistics:

- **Gradient Demand Score (GDS)**: Mean per-neuron gradient magnitude per layer. High GDS indicates a layer needs more capacity.
- **Neuron Utility Score (NUS)**: Product of incoming weight norm, outgoing weight norm, and mean activation magnitude. Low NUS neurons are candidates for pruning.
- **Loss Stationarity**: Slope of the recent loss trajectory. Flat loss triggers structural interventions.
- **Layer Contribution**: Measures how much each layer changes its input. Near-zero contribution flags redundant layers.
- **Cross-Layer Gradient Correlation (CLGC)**: Correlation between gradient vectors at different layers. High correlation between non-adjacent layers suggests a skip connection would help.

These signals feed into fourteen conjunctive disease rules that detect ten pathology types (overfitting, underfitting, stagnation, capacity exhaustion, memorisation, class imbalance, etc.) at four severity levels.

### 2. Surgery

When the model is healthy and the surgery interval has elapsed, the surgery engine proposes structural modifications:

- **Width changes**: Neurons are added (by splitting high-utility neurons) or pruned (by removing low-utility neurons) based on GDS and NUS thresholds.
- **Depth changes**: New layers are inserted near the identity when loss stationarity is detected and the model has been healthy for multiple consecutive checkpoints.
- **Operation probing**: Candidate operations from the twenty-nine-element vocabulary are temporarily inserted, and their benefit is measured as the immediate loss reduction. Operations are accepted only if their benefit exceeds a threshold that tightens over training (immunosuppression ramp).
- **Skip connections**: Created between layers with high cross-layer gradient correlation, initialised with zero-scale projections.

Every proposed modification is verified before acceptance: the system compares the loss before and after the change and reverts modifications that do not improve performance.

### 3. Treatment

When the diagnostic engine detects a disease, the treatment planner selects an intervention from a disease-specific escalation ladder:

- **Level 1** (parametric): Adjust dropout, weight decay, or learning rate.
- **Level 2** (normalisation): Insert BatchNorm, label smoothing, or domain-specific operations.
- **Level 3** (structural): Add layers, widen layers, or create skip connections.
- **Level 4** (aggressive): Simultaneous multi-intervention or soft architecture reset.

A meta-learner co-adapts the surgery interval, benefit threshold, and learning rate after every cycle, producing a natural annealing of structural plasticity that parallels critical-period dynamics in biological neural development.

### Operation Vocabulary (29 candidates)

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

Operations are filtered by modality at each surgery checkpoint: spatial layers receive only spatially compatible operations, graph layers receive only message-passing operations, and physics operations are offered only when the physics flag is enabled.

---

## Extending ASANN: Adding New Operations

ASANN is designed as an open experimental platform. Adding a new operation to the vocabulary requires three steps. The system will then autonomously decide whether, where, and when to use it across any domain.

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

This means you do not need to decide which tasks, layers, or training phases benefit from your operation. ASANN determines this autonomously through the same data-driven evaluation it applies to every other candidate.

### Example: What happened with KAN

The Kolmogorov-Arnold Network (KANLinearOp) was added to the vocabulary as one of twenty-nine candidates with no privileged status. ASANN autonomously discovered it in 55% of tabular regression experiments, adopted it in the PEMS-BAY traffic task with a benefit score of 5.44, retained it in 60% of FreeSolv molecular regression seeds, and rejected it in every single classification-only molecular and citation-graph experiment. This selective adoption pattern — discovered without any human guidance — demonstrates the kind of fine-grained, domain-specific evaluation that the platform provides automatically.

---
