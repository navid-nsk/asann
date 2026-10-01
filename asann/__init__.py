# Activate backward-compatibility aliases (csann -> asann, csann_cuda -> asann_cuda)
# so that torch.load() can unpickle models saved under the old package name.
from compat import activate as _activate_compat
_activate_compat()

from .config import ASANNConfig, ASANNOptimizerConfig, SurgeryOptimizerConfig
from .model import ASANNModel
from .encoders import (BaseEncoder, LinearEncoder, ConvEncoder, FourierEncoder,
                       PatchEmbedEncoder, TemporalEncoder, GraphNodeEncoder,
                       MolecularGraphEncoder,
                       GatedEncoderBridge, ProjectedEncoder,
                       create_encoder, build_encoder_kwargs, ENCODER_REGISTRY)
from .surgery import SurgeryEngine
from .scheduler import SurgeryScheduler
from .asann_optimizer import ASANNOptimizer, create_asann_parameter_groups
from .lr_controller import ASANNLRController
from .warmup_scheduler import ASANNWarmupScheduler
from .meta_learner import MetaLearner
from .loss import ASANNLoss
from .trainer import ASANNTrainer
from .logger import SurgeryLogger
from .diagnosis import DiagnosisEngine, Diagnosis, HealthState, DiseaseType
from .treatments import TreatmentPlanner, TreatmentType
from .lab import PatientHistory, LabDiagnostics, LabReport
from .lab_tests import create_default_lab

__all__ = [
    "ASANNConfig",
    "ASANNOptimizerConfig",
    "SurgeryOptimizerConfig",
    "ASANNModel",
    "SurgeryEngine",
    "SurgeryScheduler",
    "ASANNOptimizer",
    "create_asann_parameter_groups",
    "ASANNLRController",
    "ASANNWarmupScheduler",
    "MetaLearner",
    "ASANNLoss",
    "ASANNTrainer",
    "SurgeryLogger",
    "DiagnosisEngine",
    "Diagnosis",
    "HealthState",
    "DiseaseType",
    "TreatmentPlanner",
    "TreatmentType",
    "PatientHistory",
    "LabDiagnostics",
    "LabReport",
    "create_default_lab",
]

# ============================================================================
# Backward-compat class aliases: experiment scripts use the old CSANN* names
# ============================================================================
CSANNConfig = ASANNConfig
CSANNOptimizerConfig = ASANNOptimizerConfig
CSANNModel = ASANNModel
CSANNTrainer = ASANNTrainer
CSANNLoss = ASANNLoss
CSANNLRController = ASANNLRController
CSANNWarmupScheduler = ASANNWarmupScheduler
CSANNOptimizer = ASANNOptimizer

