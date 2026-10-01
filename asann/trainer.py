import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, Optional, Any, List, Tuple
from collections import deque
from contextlib import nullcontext
from .config import ASANNConfig, ASANNOptimizerConfig
from .model import ASANNModel
from .surgery import SurgeryEngine
from .scheduler import SurgeryScheduler
from .meta_learner import MetaLearner
from .loss import ASANNLoss
from .logger import SurgeryLogger
from .asann_optimizer import ASANNOptimizer, create_asann_parameter_groups
from .lr_controller import ASANNLRController
from .warmup_scheduler import ASANNWarmupScheduler


class ModelEMA:
    """Exponential Moving Average of model parameters (Polyak averaging).

    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Surgery-aware: call `rebuild(model)` after any structural change
    (layer/channel add/remove) to re-initialize from current parameters.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # After each training step:
        ema.update(model)
        # For validation:
        with ema.apply(model):
            validate(model)
        # After surgery:
        ema.rebuild(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._device = device
        self._rebuild(model)

    def _rebuild(self, model: nn.Module):
        """Initialize/rebuild shadow from current model parameters."""
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def rebuild(self, model: nn.Module):
        """Public interface: rebuild after surgery."""
        self._rebuild(model)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters with EMA of current model parameters."""
        d = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].shape == param.data.shape:
                    self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)
                else:
                    # Shape changed (surgery resized without rebuild) — reinitialize
                    self.shadow[name] = param.data.clone()
            elif param.requires_grad:
                # New parameter (from surgery): add to shadow
                self.shadow[name] = param.data.clone()
        # Remove stale entries
        live_names = {n for n, p in model.named_parameters() if p.requires_grad}
        for stale in list(self.shadow.keys()):
            if stale not in live_names:
                del self.shadow[stale]

    class _ApplyContext:
        """Context manager that swaps model params with EMA shadow and restores on exit."""
        def __init__(self, ema, model):
            self.ema = ema
            self.model = model

        def __enter__(self):
            self.ema._backup = {}
            for name, param in self.model.named_parameters():
                if name in self.ema.shadow:
                    if self.ema.shadow[name].shape == param.data.shape:
                        self.ema._backup[name] = param.data.clone()
                        param.data.copy_(self.ema.shadow[name])
                    # else: shape mismatch (surgery without rebuild) — skip this param
            return self.model

        def __exit__(self, *args):
            for name, param in self.model.named_parameters():
                if name in self.ema._backup:
                    param.data.copy_(self.ema._backup[name])
            self.ema._backup = {}

    def apply(self, model: nn.Module):
        """Context manager: temporarily swap model params with EMA shadow."""
        return self._ApplyContext(self, model)


class ASANNTrainer:
    """ASANN v4 — True Self-Architecting Training (Section 10).

    Full training algorithm:
    - Normal training step (every step): forward, backward, weight update with
      ASANNOptimizer (multi-scale momentum, per-group LR, NorMuon,
      multi-frequency updates, newborn graduation)
    - LR adaptation (every N steps via ASANNLRController): hypergradient-based
      per-group learning rate adjustment
    - Warmup scheduling (during warmup/post-surgery): linear warmup + cosine annealing
    - Surgery step (every S steps, after warm-up): compute signals, execute surgeries
    - Meta update (every C steps): update adaptive surgery thresholds
    - Convergence check: detect architecture stabilization

    The network starts minimal and earns its complexity through gradient signals.
    Between surgeries, it IS a standard neural network with zero overhead.
    """

    def __init__(
        self,
        model: ASANNModel,
        config: ASANNConfig,
        task_loss_fn: Callable,
        log_dir: Optional[str] = None,
        task_type: str = "regression",
        y_scaler: Optional[Any] = None,
        n_classes: Optional[int] = None,
        append_logs: bool = False,
        input_transform: Optional[Callable] = None,
        extra_params: Optional[List[nn.Parameter]] = None,
        target_metric: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = config.device

        # Optional input preprocessor: transforms x_batch before it reaches the model.
        # Used for PTB (token IDs → embedding → flatten) or any task where raw data
        # is not directly float features. When None, x_batch passes through unchanged.
        self._input_transform = input_transform

        # Eval hooks for inductive graph training: swap graph data before/after eval
        self._eval_pre_hook = None   # callable, called before each eval epoch
        self._eval_post_hook = None  # callable, called after each eval epoch

        # Extra parameters not part of ASANNModel but need joint training
        # (e.g., nn.Embedding weights for PTB). Added to optimizer after creation.
        self._extra_params = extra_params or []

        # Task-type-aware evaluation metrics
        self.task_type = task_type          # "regression" or "classification"
        self.y_scaler = y_scaler            # for regression: inverse-transform predictions
        self.n_classes = n_classes           # for classification

        # Target metric for best-model tracking
        # Default: accuracy for classification, r2 for regression (backward compatible)
        if target_metric is not None:
            self.target_metric = target_metric
        else:
            self.target_metric = "accuracy" if task_type == "classification" else "r2"
        # Resolve direction (higher_is_better) from the metric registry
        import sys, os
        _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
        if _exp_dir not in sys.path:
            sys.path.insert(0, _exp_dir)
        from common import get_metric_direction
        self._target_metric_higher_is_better = get_metric_direction(self.target_metric)

        # Initialize components
        self.logger = SurgeryLogger(log_dir=log_dir, append=append_logs)
        self.surgery_engine = SurgeryEngine(config, logger=self.logger)
        self.scheduler = SurgeryScheduler(
            config, self.surgery_engine, logger=self.logger,
            n_classes=n_classes or 0,
        )
        self.meta_learner = MetaLearner(config)
        self.loss_fn = ASANNLoss(task_loss_fn, config)

        # Auto complexity target: scale-free approach for spatial models
        if config.complexity_target_auto:
            initial_cost = model.compute_architecture_cost()

            if config.spatial_shape is not None and hasattr(model, 'output_head'):
                # Scale-free target based on dataset complexity
                d_output = model.output_head.out_features  # num classes
                C, H, W = config.spatial_shape
                spatial_scale = (H * W) ** 0.5 / 10.0  # normalized (CIFAR 32×32 -> 3.2)
                auto_target = d_output * config.complexity_target_base_per_class * spatial_scale
                # Ensure target is at least 10× initial cost (room to grow)
                auto_target = max(auto_target, initial_cost * 10.0)
            else:
                # Tabular: use simple multiplier (backward compatible)
                auto_target = initial_cost * config.complexity_target_multiplier

            config.complexity_target = auto_target
            self.loss_fn.complexity_target = auto_target
            print(f"  Auto complexity target: {auto_target:.0f} "
                  f"(initial cost: {initial_cost:.0f})")

        # --- New Optimizer Stack ---
        opt_config = config.optimizer
        if not isinstance(opt_config, ASANNOptimizerConfig):
            # Backward compatibility: convert old SurgeryOptimizerConfig
            opt_config = ASANNOptimizerConfig(
                base_lr=getattr(opt_config, 'lr_mature_weights', 1e-3),
                betas=(0.9, 0.95, 0.99, 0.999),
                eps=getattr(opt_config, 'eps', 1e-8),
                weight_decay=getattr(opt_config, 'weight_decay', 0.01),
                max_grad_norm=getattr(opt_config, 'max_grad_norm', 1.0),
                neuron_norm_enabled=getattr(opt_config, 'neuron_norm_enabled', True),
                neuron_norm_eps=getattr(opt_config, 'neuron_norm_eps', 1e-6),
                variance_transfer_enabled=getattr(opt_config, 'variance_transfer_enabled', True),
                newborn_warmup_steps=getattr(opt_config, 'newborn_warmup_steps', 100),
                newborn_graduation_steps=getattr(opt_config, 'newborn_graduation_steps', 200),
                warmup_steps=getattr(opt_config, 'warmup_steps', 500),
            )

        # Create optimizer with per-module parameter groups
        print("  Creating ASANN parameter groups:")
        param_groups = create_asann_parameter_groups(model, opt_config)

        self.optimizer = ASANNOptimizer(
            param_groups,
            lr=opt_config.base_lr,
            betas=opt_config.betas,
            eps=opt_config.eps,
            weight_decay=opt_config.weight_decay,
            max_grad_norm=opt_config.max_grad_norm,
            neuron_norm_enabled=opt_config.neuron_norm_enabled,
            neuron_norm_eps=opt_config.neuron_norm_eps,
            variance_transfer_enabled=opt_config.variance_transfer_enabled,
            newborn_warmup_steps=opt_config.newborn_warmup_steps,
            newborn_graduation_steps=opt_config.newborn_graduation_steps,
            newborn_lr_scale=opt_config.newborn_lr_scale,
            use_cuda_ops=config.use_cuda_ops,
        )

        # Add extra parameters (e.g., embedding weights) to optimizer
        if self._extra_params:
            extra_group = {
                'params': list(self._extra_params),
                'lr': opt_config.base_lr,
                'lr_scale': 1.0,
                'update_freq': 1,
                'weight_decay': 0.0,
                'grad_clip': opt_config.max_grad_norm,
                'betas': opt_config.betas,
                'eps': opt_config.eps,
                'name': 'extra',
            }
            self.optimizer.param_groups.append(extra_group)
            for p in self._extra_params:
                self.optimizer._all_param_ids.add(id(p))
            n_extra = sum(p.numel() for p in self._extra_params)
            print(f"  extra               : {len(self._extra_params):4d} params "
                  f"({n_extra:>8d} elements), lr={opt_config.base_lr:.6f}, freq= 1")

        # Warmup scheduler (must be created AFTER optimizer)
        self.warmup_scheduler = ASANNWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=opt_config.warmup_steps,
            total_steps=100000,  # Will be updated in train()
            min_lr_ratio=opt_config.cosine_min_lr_ratio,
            restart_period=opt_config.cosine_restart_period,
            restart_mult=opt_config.cosine_restart_mult,
        )

        # Learnable LR controller (activates after warmup)
        # Pass target_lrs from warmup scheduler so the LR controller knows the
        # correct base LRs (scheduler has already zeroed optimizer group['lr']).
        self.lr_controller = None
        if opt_config.lr_controller_enabled:
            self.lr_controller = ASANNLRController(
                optimizer=self.optimizer,
                update_interval=opt_config.lr_controller_update_interval,
                warmup_steps=opt_config.warmup_steps,
                meta_lr=opt_config.lr_controller_meta_lr,
                momentum=opt_config.lr_controller_momentum,
                dead_zone=opt_config.lr_controller_dead_zone,
                scale_min=opt_config.lr_controller_scale_min,
                scale_max=opt_config.lr_controller_scale_max,
                plateau_patience=opt_config.plateau_patience,
                plateau_factor=opt_config.plateau_factor,
                plateau_cooldown=opt_config.plateau_cooldown,
                plateau_min_scale=opt_config.plateau_min_scale,
                plateau_max_reductions=opt_config.plateau_max_reductions,
                target_lrs=self.warmup_scheduler.target_lrs,
            )

        # Store opt_config for later use
        self._opt_config = opt_config

        # --- AMP (Automatic Mixed Precision) ---
        self._amp_enabled = (
            config.amp_enabled
            and config.device == "cuda"
            and torch.cuda.is_available()
        )
        self._grad_scaler = torch.amp.GradScaler('cuda', enabled=self._amp_enabled)
        if self._amp_enabled:
            print("  AMP enabled (FP16 mixed precision)")

        # --- torch.compile (for validation/eval forward passes) ---
        self._compiled_model = None
        if (config.torch_compile_enabled
                and config.device == "cuda"
                and hasattr(torch, 'compile')):
            try:
                self._compiled_model = torch.compile(
                    model, mode=config.torch_compile_mode)
                print(f"  torch.compile enabled (mode={config.torch_compile_mode})")
            except Exception as e:
                print(f"  torch.compile failed ({e}), using eager mode")
                self._compiled_model = None

        # --- GPU Batch Augmentation ---
        # Lazy import: only set if spatial_shape is provided (image models)
        # When config.dataset_augmented=True, the dataset already applies AutoAugment,
        # flip, crop at the PIL level — so GPU augment only does cutout.
        # When config.dataset_augmented=False, GPU augment does the full pipeline.
        self._gpu_augment_fn = None
        if config.spatial_shape is not None:
            try:
                if getattr(config, 'dataset_augmented', False):
                    # Dataset handles AutoAugment + flip + crop — GPU does cutout only
                    _cfg_cutout_size = getattr(config, 'cutout_size', None)
                    try:
                        from common import gpu_cutout_only as _gpu_cutout_only_base
                        # Wrap to pass config cutout_size
                        def gpu_cutout_only(x_batch, spatial_shape, _cs=_cfg_cutout_size):
                            return _gpu_cutout_only_base(x_batch, spatial_shape, cutout_size=_cs)
                    except ImportError:
                        # Inline cutout-only fallback
                        _cfg_cutout = getattr(config, 'cutout_size', None)
                        def gpu_cutout_only(x_batch, spatial_shape, _cs=_cfg_cutout):
                            C, H, W = spatial_shape
                            B = x_batch.shape[0]
                            imgs = x_batch.view(B, C, H, W)
                            cutout_size = _cs if _cs is not None else max(1, H // 2)
                            cutout_half = cutout_size // 2
                            cy = torch.randint(0, H, (B,), device=x_batch.device)
                            cx = torch.randint(0, W, (B,), device=x_batch.device)
                            yy = torch.arange(H, device=x_batch.device).unsqueeze(0)
                            xx = torch.arange(W, device=x_batch.device).unsqueeze(0)
                            y_mask = (yy - cy.unsqueeze(1)).abs() < cutout_half
                            x_mask = (xx - cx.unsqueeze(1)).abs() < cutout_half
                            cutout_mask = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1)).unsqueeze(1)
                            fill_value = imgs.mean(dim=(2, 3), keepdim=True)
                            imgs = torch.where(cutout_mask, fill_value.expand_as(imgs), imgs)
                            return imgs.reshape(B, -1)
                    self._gpu_augment_fn = gpu_cutout_only
                    print("  GPU cutout-only augmentation enabled (dataset handles AutoAugment+flip+crop)")
                else:
                    # No dataset-level augmentation — GPU does everything
                    try:
                        from common import gpu_augment_batch
                    except ImportError:
                        # Fallback: define inline (same logic, no external dependency)
                        import torch.nn.functional as _F
                        _cfg_cutout2 = getattr(config, 'cutout_size', None)
                        def gpu_augment_batch(x_batch, spatial_shape, pad=4, _cs=_cfg_cutout2):
                            C, H, W = spatial_shape
                            B = x_batch.shape[0]
                            imgs = x_batch.view(B, C, H, W)
                            flip_mask = torch.rand(B, 1, 1, 1, device=x_batch.device) < 0.5
                            imgs = torch.where(flip_mask, imgs.flip(-1), imgs)
                            padded = _F.pad(imgs, [pad] * 4, mode='reflect')
                            crop_i = torch.randint(0, 2 * pad + 1, (B,), device=x_batch.device)
                            crop_j = torch.randint(0, 2 * pad + 1, (B,), device=x_batch.device)
                            row_idx = torch.arange(H, device=x_batch.device).unsqueeze(0) + crop_i.unsqueeze(1)
                            col_idx = torch.arange(W, device=x_batch.device).unsqueeze(0) + crop_j.unsqueeze(1)
                            batch_idx = torch.arange(B, device=x_batch.device)
                            imgs = padded[
                                batch_idx[:, None, None, None],
                                torch.arange(C, device=x_batch.device)[None, :, None, None],
                                row_idx[:, None, :, None],
                                col_idx[:, None, None, :],
                            ]
                            if C >= 3:
                                c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x_batch.device) - 0.5) * 2 * 0.2
                                channel_mean = imgs.mean(dim=(2, 3), keepdim=True)
                                imgs = channel_mean + c_factor * (imgs - channel_mean)
                                s_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=x_batch.device) - 0.5) * 2 * 0.2
                                gray = imgs.mean(dim=1, keepdim=True).expand_as(imgs)
                                imgs = gray + s_factor * (imgs - gray)
                            cutout_size = _cs if _cs is not None else max(1, H // 2)
                            cutout_half = cutout_size // 2
                            cy = torch.randint(0, H, (B,), device=x_batch.device)
                            cx = torch.randint(0, W, (B,), device=x_batch.device)
                            yy = torch.arange(H, device=x_batch.device).unsqueeze(0)
                            xx = torch.arange(W, device=x_batch.device).unsqueeze(0)
                            y_mask = (yy - cy.unsqueeze(1)).abs() < cutout_half
                            x_mask = (xx - cx.unsqueeze(1)).abs() < cutout_half
                            cutout_mask = (y_mask.unsqueeze(2) & x_mask.unsqueeze(1)).unsqueeze(1)
                            fill_value = imgs.mean(dim=(2, 3), keepdim=True)
                            imgs = torch.where(cutout_mask, fill_value.expand_as(imgs), imgs)
                            return imgs.reshape(B, -1)
                    self._gpu_augment_fn = gpu_augment_batch
                    print("  GPU full batch augmentation enabled (flip+crop+jitter+cutout)")
            except Exception as e:
                print(f"  GPU augmentation setup failed ({e}), using plain data")

        # --- GPU Elastic Deformation ---
        self._gpu_elastic_fn = None
        if (config.spatial_shape is not None
                and getattr(config, 'elastic_enabled', False)):
            _el_alpha = config.elastic_alpha
            _el_sigma = config.elastic_sigma
            try:
                from asann_cuda.ops.elastic_deform import gpu_elastic_deform as _elastic_base
                def _elastic_fn(x_batch, spatial_shape,
                                _a=_el_alpha, _s=_el_sigma):
                    return _elastic_base(x_batch, spatial_shape,
                                         alpha=_a, sigma=_s)
                self._gpu_elastic_fn = _elastic_fn
                print(f"  GPU elastic deformation enabled "
                      f"(alpha={_el_alpha}, sigma={_el_sigma})")
            except Exception as e:
                print(f"  GPU elastic deformation setup failed ({e})")

        # --- Mixup augmentation (classification only) ---
        self._mixup_alpha = 0.0
        if (config.mixup_enabled
                and config.spatial_shape is not None
                and task_type == "classification"
                and n_classes and n_classes > 1):
            self._mixup_alpha = config.mixup_alpha
            print(f"  Mixup enabled (alpha={self._mixup_alpha})")

        # --- CutMix augmentation (classification only, paired with Mixup at the batch level) ---
        self._cutmix_alpha = 0.0
        self._cutmix_prob = 0.0
        if (getattr(config, "cutmix_enabled", False)
                and config.spatial_shape is not None
                and task_type == "classification"
                and n_classes and n_classes > 1):
            self._cutmix_alpha = float(getattr(config, "cutmix_alpha", 1.0))
            self._cutmix_prob = float(getattr(config, "cutmix_prob", 0.5))
            print(f"  CutMix enabled (alpha={self._cutmix_alpha}, p={self._cutmix_prob})")

        # --- EMA of model weights (Polyak averaging) ---
        self._ema = None
        if config.ema_enabled:
            self._ema = ModelEMA(model, decay=config.ema_decay, device=config.device)
            print(f"  EMA enabled (decay={config.ema_decay})")

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.steps_per_epoch = 0  # Set at train time from len(train_loader)
        self.recent_losses = deque(maxlen=config.meta_update_interval)
        self.best_loss = float("inf")
        self._stable_model_saved = False

        # Best model tracking (based on validation metric)
        self._best_val_metric: Optional[float] = None  # best primary metric value
        self._best_val_loss_at_best: Optional[float] = None  # val_loss when best metric was achieved (tiebreaker)
        self._best_val_step: Optional[int] = None       # step at which best was achieved
        self._best_val_epoch: Optional[int] = None      # epoch at which best was achieved
        self._best_model_state: Optional[dict] = None   # state_dict of best model
        self._best_model_copy: Optional[ASANNModel] = None  # deep-copied full model at best
        self._best_model_arch: Optional[dict] = None    # architecture description at best
        self._best_model_connections: Optional[list] = None  # connections at best

        # Architecture-aware auto-stopping state
        self._arch_stable_step: Optional[int] = None
        self._arch_stable_epoch: Optional[int] = None
        self._best_val_loss_post_stable: float = float("inf")
        self._val_patience_counter: int = 0
        self._stop_reason: Optional[str] = None

        # Set task type on diagnosis engine
        self.scheduler.diagnosis_engine.set_task_type(task_type)

    def train_epochs(
        self,
        train_data: DataLoader,
        max_epochs: int,
        val_data: Optional[DataLoader] = None,
        test_data: Optional[DataLoader] = None,
        print_every: int = 100,
        snapshot_every: int = 500,
        checkpoint_path: Optional[str] = None,
        checkpoint_every_epochs: int = 10,
    ) -> Dict[str, Any]:
        """Execute the ASANN training algorithm with epoch-based surgery/diagnosis.

        This is the v2 training loop that uses:
        - Epoch-based timing for surgery, validation, and meta-updates
        - DiagnosisEngine for health monitoring
        - TreatmentPlanner for evidence-based architecture interventions
        - Health-based architecture stability

        The training loop structure:
        - For each epoch:
          - Train all batches (with per-step gradient accumulation for surgery signals)
          - At epoch boundaries divisible by eval_epoch_interval:
              -> Run full validation + compute eval metrics
              -> If also a surgery epoch: run diagnosis-based surgery
          - At epoch boundaries divisible by meta_update_epoch_interval:
              -> Run meta-learner threshold updates
          - Auto-stop if architecture stable + val plateaued
        """
        self.model.train()
        self.steps_per_epoch = len(train_data)

        # Inform scheduler of training set size for memorization guards.
        # For graph tasks (semi-supervised), the dataset contains ALL nodes but
        # only a subset have training labels (rest are masked with -100).
        # We must count actual labeled samples, not total dataset size.
        if hasattr(train_data, 'dataset'):
            try:
                first_batch = next(iter(train_data))
                if isinstance(first_batch, (tuple, list)) and len(first_batch) >= 2:
                    labels = first_batch[1]
                    n_labeled = int((labels != -100).sum().item())
                    if n_labeled > 0 and n_labeled < len(labels):
                        # Semi-supervised: some labels masked -> use labeled count
                        self.scheduler.n_train_samples = n_labeled
                    else:
                        self.scheduler.n_train_samples = len(train_data.dataset)
                else:
                    self.scheduler.n_train_samples = len(train_data.dataset)
            except Exception:
                self.scheduler.n_train_samples = len(train_data.dataset)
        else:
            # DataLoader without .dataset — estimate from batch count × batch_size
            self.scheduler.n_train_samples = len(train_data) * getattr(
                train_data, 'batch_size', 1)

        # Compute effective max epochs for auto-stop
        min_epochs = max_epochs
        if self.config.auto_stop_enabled:
            effective_max_epochs = int(max_epochs * self.config.hard_max_multiplier)
        else:
            effective_max_epochs = max_epochs

        # Update scheduler's total steps estimate
        estimated_total_steps = effective_max_epochs * self.steps_per_epoch
        self.warmup_scheduler.total_steps = estimated_total_steps

        # Determine warmup end step (epoch-based)
        warmup_end_step = self.config.warmup_epochs * self.steps_per_epoch

        # Scale warmup parameters for actual steps_per_epoch.
        # The warmup scheduler and LR controller use step-based warmup counts
        # (default 500 steps), which is fine for multi-batch-per-epoch training
        # but catastrophic for full-batch graph training (1 step/epoch) where
        # 500 steps = 500 epochs. Scale to use epoch-based counts instead.
        self.warmup_scheduler.warmup_steps = warmup_end_step
        # Post-surgery re-warmup: ~3-5 epochs worth of steps, capped at 50
        self.warmup_scheduler._post_surgery_warmup_len = max(
            5, min(50, 3 * self.steps_per_epoch)
        )
        if self.lr_controller is not None:
            self.lr_controller.warmup_steps = warmup_end_step

        # Determine start epoch (for resume support)
        start_epoch = self.current_epoch + 1
        if start_epoch > 1:
            print(f"  [RESUME] Continuing training from epoch {start_epoch}")

        print(f"  Epoch-based training: {max_epochs} epochs "
              f"({self.steps_per_epoch} steps/epoch, "
              f"total ~{max_epochs * self.steps_per_epoch} steps)")
        print(f"  Surgery every {self.config.surgery_epoch_interval} epochs, "
              f"eval every {self.config.eval_epoch_interval} epochs, "
              f"meta-update every {self.config.meta_update_epoch_interval} epochs")
        print(f"  Warmup: {self.config.warmup_epochs} epochs "
              f"({warmup_end_step} steps)")

        # Training metrics (in-memory, returned at end)
        metrics = {
            "train_losses": [],
            "task_losses": [],
            "complexity_costs": [],
            "learning_rates": [],
            "val_losses": [],
            "architecture_snapshots": [],
        }

        for epoch in range(start_epoch, effective_max_epochs + 1):
            self.current_epoch = epoch
            epoch_loss_sum = 0.0
            epoch_task_loss_sum = 0.0
            epoch_steps = 0

            # Check if treatment planner requested balanced sampling
            if (hasattr(self.scheduler, 'treatment_planner')
                    and self.scheduler.treatment_planner._pending_balanced_sampler):
                self.scheduler.treatment_planner._pending_balanced_sampler = False
                train_data = self._apply_balanced_sampler(train_data)

            # ===== TRAIN ONE EPOCH =====
            for batch_idx, batch in enumerate(train_data):
                self.global_step += 1
                step = self.global_step

                self._maybe_set_mol_indices(batch)
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # GPU elastic deformation (before cutout)
                if self._gpu_elastic_fn is not None:
                    x_batch = self._gpu_elastic_fn(x_batch, self.config.spatial_shape)
                # GPU batch augmentation (spatial models only)
                if self._gpu_augment_fn is not None:
                    x_batch = self._gpu_augment_fn(x_batch, self.config.spatial_shape)

                # Mixup / CutMix augmentation (classification only).
                # If both are enabled, pick CutMix per-batch with prob self._cutmix_prob,
                # otherwise Mixup. They are NEVER stacked on the same image.
                mixup_lam = None
                if self.model.training:
                    use_cutmix = (
                        self._cutmix_alpha > 0
                        and self._cutmix_prob > 0
                        and torch.rand(1).item() < self._cutmix_prob
                    )
                    if use_cutmix:
                        x_batch, y_batch, mixup_lam = self._apply_cutmix(x_batch, y_batch)
                    elif self._mixup_alpha > 0:
                        x_batch, y_batch, mixup_lam = self._apply_mixup(x_batch, y_batch)

                # ===== NORMAL TRAINING STEP =====
                step_metrics = self._training_step(x_batch, y_batch, step, mixup_lam=mixup_lam)

                metrics["train_losses"].append(step_metrics["total_loss"])
                metrics["task_losses"].append(step_metrics["task_loss"])
                metrics["complexity_costs"].append(step_metrics["complexity_cost"])
                metrics["learning_rates"].append(step_metrics["learning_rate"])

                epoch_loss_sum += step_metrics["total_loss"]
                epoch_task_loss_sum += step_metrics["task_loss"]
                epoch_steps += 1

                # ===== CSV: per-step training metrics =====
                if self.config.csv_logging_enabled:
                    widths = [self.model.get_layer_width(l) for l in range(self.model.num_layers)]
                    total_params = sum(p.numel() for p in self.model.all_parameters())
                    self.logger.log_training_step(
                        step=step,
                        metrics=step_metrics,
                        model_info={
                            "num_layers": self.model.num_layers,
                            "num_connections": len(self.model.connections),
                            "total_params": total_params,
                            "architecture_cost": self.model.compute_architecture_cost(),
                            "widths": widths,
                            "architecture_stable": self.model.architecture_stable,
                        },
                    )

                    # ===== CSV: optimizer state =====
                    group_stats = self.optimizer.get_group_stats()
                    self.logger.log_optimizer_state(step, {
                        "phase": self.warmup_scheduler.get_phase(),
                        "phase_step": self.warmup_scheduler.current_step,
                        "complexity_ema": 0.0,
                        "complexity_lr_mult": 1.0,
                        "num_newborn_params": self.optimizer.num_newborn_params,
                        "group_stats": group_stats,
                        "num_param_groups": len(self.optimizer.param_groups),
                        "total_tracked_params": len(self.optimizer._all_param_ids),
                    })

                # ===== PER-STEP LOGGING =====
                if step % print_every == 0:
                    self._print_status_epoch(step, epoch, effective_max_epochs, step_metrics)

                if step % snapshot_every == 0:
                    self.logger.log_architecture_snapshot(
                        step, self.model.describe_architecture()
                    )
                    self.logger.log_per_layer_details(step, self.model)
                    self.logger.log_connection_details(step, self.model)
                    self.logger.log_optimizer_groups(step, self.optimizer)
                    if self.lr_controller is not None:
                        self.logger.log_lr_controller_state(
                            step, self.lr_controller, step_metrics["task_loss"]
                        )

                # Track best loss
                if step_metrics["task_loss"] < self.best_loss:
                    self.best_loss = step_metrics["task_loss"]

            # ===== END OF EPOCH =====
            avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
            avg_epoch_task_loss = epoch_task_loss_sum / max(epoch_steps, 1)

            # Advance immunosuppression gates: gradually increase effect of
            # recently inserted operations (BatchNorm, JK, ResNet, etc.)
            architecture_changed = self.model.advance_surgery_gates()

            # Note: with deepcopy-based rollback, architecture changes (gate/bridge
            # absorption) don't invalidate the snapshot — the full model copy is
            # self-consistent and will be restored via __dict__ swap if needed.

            # ===== EVALUATION EPOCH =====
            is_eval_epoch = (epoch % self.config.eval_epoch_interval == 0)
            is_surgery_epoch = (
                epoch % self.config.surgery_epoch_interval == 0
                and epoch > self.config.warmup_epochs
                and self.config.diagnosis_enabled
            )
            is_meta_epoch = (epoch % self.config.meta_update_epoch_interval == 0)

            train_metrics_result = None
            val_metrics_result = None
            val_loss = None
            train_loss_for_diag = avg_epoch_task_loss
            train_acc_for_diag = None
            val_loss_for_diag = None
            val_acc_for_diag = None
            val_balanced_acc_for_diag = None
            val_target_metric_value = None

            if is_eval_epoch:
                # Single EMA swap for entire eval block
                ema_ctx = self._ema.apply(self.model) if self._ema is not None else nullcontext()
                with ema_ctx:
                    self.model.eval()

                    # Evaluate on train set (subsampled when configured)
                    # NOTE: done BEFORE the eval_pre_hook so that inductive
                    # training evaluates train metrics on the train subgraph.
                    train_metrics_result = self._evaluate_metrics_inner(
                        train_data, "train", self.global_step,
                        max_batches=self.config.train_eval_max_batches,
                    )
                    if self.task_type == "classification":
                        train_acc_for_diag = train_metrics_result.get("accuracy", None)

                    # Inductive graph training: swap to full graph for val/test
                    if self._eval_pre_hook is not None:
                        self._eval_pre_hook()

                    # Evaluate on val set: loss + metrics in ONE pass
                    if val_data is not None:
                        val_loss, val_metrics_result = self._validate_with_metrics_inner(
                            val_data, "val", self.global_step
                        )
                        metrics["val_losses"].append((self.global_step, val_loss))
                        val_loss_for_diag = val_loss
                        if self.task_type == "classification":
                            val_acc_for_diag = val_metrics_result.get("accuracy", None)
                            val_balanced_acc_for_diag = val_metrics_result.get("balanced_accuracy", None)

                        # Inject val_loss into metrics dict so it can be used as target_metric
                        val_metrics_result["val_loss"] = val_loss

                        # Extract target metric value for diagnosis (auroc, accuracy, r2, etc.)
                        val_target_metric_value = val_metrics_result.get(self.target_metric, None)

                        # Update stall tracking at EVERY eval epoch (not just surgery epochs)
                        # so the tracker sees every metric sample and doesn't miss peaks.
                        if val_target_metric_value is not None:
                            self.scheduler.update_stall_tracking(
                                epoch=epoch,
                                target_metric_value=val_target_metric_value,
                                target_metric_higher_is_better=self._target_metric_higher_is_better,
                            )

                        # Best model tracking
                        self._update_best_model_epoch(epoch, self.global_step, val_metrics_result, val_loss=val_loss)

                    # Inductive graph training: swap back to train subgraph
                    if self._eval_post_hook is not None:
                        self._eval_post_hook()

                    self.model.train()

                # Log architecture snapshot at eval (outside EMA - uses structure not weights)
                self.logger.log_architecture_snapshot(
                    self.global_step, self.model.describe_architecture()
                )
                self.logger.log_per_layer_details(self.global_step, self.model)
                self.logger.log_connection_details(self.global_step, self.model)

            # ===== SURGERY EPOCH (Diagnosis-Based) =====
            if is_surgery_epoch and val_loss_for_diag is not None:
                # Get a train batch for potential operation probing
                probe_x, probe_y = None, None
                try:
                    probe_batch = next(iter(train_data))
                    self._maybe_set_mol_indices(probe_batch)
                    probe_x = probe_batch[0].to(self.device)
                    probe_y = probe_batch[1].to(self.device)
                    if self._input_transform is not None:
                        with torch.no_grad():
                            probe_x = self._input_transform(probe_x)
                except StopIteration:
                    pass

                # Get a val batch for validation-gated surgery
                val_batch = None
                if val_data is not None:
                    try:
                        vb = next(iter(val_data))
                        self._maybe_set_mol_indices(vb)
                        vb_x = vb[0].to(self.device)
                        vb_y = vb[1].to(self.device)
                        if self._input_transform is not None:
                            with torch.no_grad():
                                vb_x = self._input_transform(vb_x)
                        val_batch = (vb_x, vb_y)
                    except StopIteration:
                        pass

                self.scheduler.execute_diagnosis_surgery(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.global_step,
                    epoch=epoch,
                    train_loss=train_loss_for_diag,
                    val_loss=val_loss_for_diag,
                    train_acc=train_acc_for_diag,
                    val_acc=val_acc_for_diag,
                    val_balanced_acc=val_balanced_acc_for_diag,
                    x_batch=probe_x,
                    y_batch=probe_y,
                    loss_fn=self.loss_fn.task_loss_fn,
                    val_batch=val_batch,
                    val_data=val_data,
                    task_type=self.task_type,
                    target_metric_name=self.target_metric,
                    target_metric_value=val_target_metric_value,
                    target_metric_higher_is_better=self._target_metric_higher_is_better,
                )

                # Fix 9: Apply pending LR reduction from treatment system.
                # The treatment planner sets _pending_lr_factor when LR_REDUCE
                # is prescribed. We apply it here through the proper channels
                # (LR controller base_lrs + warmup scheduler target_lrs) instead
                # of the brute-force g['lr'] *= factor which gets overwritten.
                pending_lr = self.scheduler.treatment_planner._pending_lr_factor
                if pending_lr is not None:
                    self.scheduler.treatment_planner._pending_lr_factor = None
                    if self.lr_controller is not None:
                        self.lr_controller.reduce_base_lrs(pending_lr)
                    # Also reduce warmup scheduler targets for consistency
                    self.warmup_scheduler.target_lrs = [
                        lr * pending_lr for lr in self.warmup_scheduler.target_lrs
                    ]

                # Apply pending LR warm restart from treatment system.
                # Resets the cosine annealing cycle so LR jumps back to
                # base_lr — gentle alternative to BATCHNORM_PACKAGE for
                # flat/tabular models with stalled convergence.
                if self.scheduler.treatment_planner._pending_lr_restart:
                    self.scheduler.treatment_planner._pending_lr_restart = False
                    self.warmup_scheduler.trigger_warm_restart()
                    print("  [LR] Warm restart applied -- cosine cycle reset")

                # Post-surgery: re-warmup and resync LR controller
                if self.scheduler.interval_surgery_count > 0:
                    self.warmup_scheduler.enter_post_surgery_warmup()
                    self.warmup_scheduler.resync_target_lrs(self.optimizer)
                    if self.lr_controller is not None:
                        self.lr_controller.on_surgery()
                        self.lr_controller.resync_groups(self.optimizer)
                    # Rebuild EMA from current params (architecture changed)
                    if self._ema is not None:
                        self._ema.rebuild(self.model)

                    # Save checkpoint immediately after surgery to avoid losing
                    # progress if the modified architecture causes a crash
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path)

            # ===== META UPDATE EPOCH =====
            if is_meta_epoch:
                self.meta_learner.meta_update(list(self.recent_losses))

                thresholds = self.meta_learner.get_current_thresholds()
                self.config.gds_k = thresholds["gds_k"]
                self.config.benefit_threshold = thresholds["benefit_threshold"]
                self.config.connection_threshold = thresholds["connection_threshold"]

                self.loss_fn.update_lambda(self.model)

                # CSV: meta-learner state
                avg_recent = (
                    sum(self.recent_losses) / len(self.recent_losses)
                    if self.recent_losses else 0.0
                )
                self.logger.log_meta_state(
                    step=self.global_step,
                    complexity_lr_mult=1.0,
                    effective_lr=self.optimizer.current_lr,
                    surgery_interval=self.meta_learner.current_surgery_interval,
                    thresholds=thresholds,
                    avg_recent_loss=avg_recent,
                    optimizer_phase=self.warmup_scheduler.get_phase(),
                    scheduler_lr_factor=self.warmup_scheduler.get_lr_factor(),
                    num_newborn_params=self.optimizer.num_newborn_params,
                    complexity_target=self.config.complexity_target,
                )

            # ===== CONVERGENCE / STABILITY CHECK =====
            if self.model.architecture_stable:
                if self._arch_stable_epoch is None:
                    self._arch_stable_epoch = epoch
                    self._arch_stable_step = self.global_step
                if not self._stable_model_saved:
                    if self.logger.log_dir is not None:
                        stable_path = str(self.logger.log_dir / "stable_model.pt")
                        self.save_model(stable_path)
                        print(f"  [STABLE] Architecture stabilized at epoch {epoch} "
                              f"— model saved to {stable_path}")
                    self._stable_model_saved = True
            else:
                # Architecture lost stability (treatment broke it)
                self._arch_stable_epoch = None
                self._arch_stable_step = None

            # ===== AUTO-STOP CHECK (epoch-based) =====
            if (self.config.auto_stop_enabled
                    and epoch >= min_epochs
                    and is_eval_epoch
                    and val_loss is not None):
                should_stop, reason = self._check_epoch_convergence(epoch, val_loss)
                if should_stop:
                    self._stop_reason = reason
                    print(f"\n  [AUTO-STOP] Training stopped at epoch {epoch}: {reason}")
                    break

            # ===== TREATMENT EXHAUSTION CHECK =====
            if self.scheduler._treatments_exhausted and self.config.stop_on_treatment_exhaustion:
                self._stop_reason = "treatments exhausted (no applicable treatments)"
                print(f"\n  [EARLY-STOP] Training stopped at epoch {epoch}: "
                      f"all treatments exhausted — continuing would only degrade the model")
                break

            # ===== PERIODIC CHECKPOINT =====
            if checkpoint_path and epoch % checkpoint_every_epochs == 0:
                self.save_checkpoint(checkpoint_path)

            # End-of-epoch print
            stable_str = " [STABLE]" if self.model.architecture_stable else ""
            val_str = f" | Val loss: {val_loss:.6f}" if val_loss is not None else ""
            print(f"Epoch {epoch}/{max_epochs} | "
                  f"Avg loss: {avg_epoch_task_loss:.6f}{val_str} | "
                  f"Steps: {self.global_step} | "
                  f"Params: {sum(p.numel() for p in self.model.all_parameters())}"
                  f"{stable_str}")

        # ===== TRAINING COMPLETE =====
        # Determine stop reason
        if self._stop_reason is None:
            if self.config.auto_stop_enabled and self.current_epoch >= effective_max_epochs:
                self._stop_reason = f"hard cap reached ({effective_max_epochs} epochs)"
            else:
                self._stop_reason = f"max_epochs reached ({max_epochs} epochs)"

        # Log summary
        print(f"\n  [TRAINING COMPLETE] Actual epochs: {self.current_epoch} "
              f"(min: {min_epochs}, hard cap: {effective_max_epochs})")
        print(f"  Stop reason: {self._stop_reason}")
        if self._arch_stable_epoch is not None:
            print(f"  Architecture stabilized at epoch: {self._arch_stable_epoch}")
        else:
            print(f"  Architecture did NOT stabilize")

        # Final snapshot
        self.logger.log_architecture_snapshot(
            self.global_step, self.model.describe_architecture()
        )
        self.logger.log_per_layer_details(self.global_step, self.model)
        self.logger.log_connection_details(self.global_step, self.model)

        # Final evaluation with LAST (current) model
        # For inductive graph training: swap to full graph for val/test eval
        self._evaluate_metrics(train_data, "train", self.global_step)
        if self._eval_pre_hook is not None:
            self._eval_pre_hook()
        if val_data is not None:
            self._evaluate_metrics(val_data, "val", self.global_step)
        last_test_metrics = None
        if test_data is not None:
            last_test_metrics = self._evaluate_metrics(
                test_data, "test", self.global_step
            )

        # Restore best model and evaluate separately
        has_best = (self._best_model_copy is not None or self._best_model_state is not None)
        if has_best and val_data is not None:
            self._restore_best_model()
            # Rebuild EMA for the restored model (architecture may differ from final)
            if self._ema is not None:
                self._ema.rebuild(self.model)
            if test_data is not None:
                best_test_metrics = self._evaluate_metrics(
                    test_data, "test_best", self.global_step
                )
                metrics["best_test_metrics"] = best_test_metrics

        # Save best model to disk
        if has_best and self.logger.log_dir is not None:
            best_path = str(self.logger.log_dir / "best_model.pt")
            self._save_best_model(best_path)

        # Store last-model test metrics (separate from best-model metrics)
        if last_test_metrics is not None:
            metrics["last_test_metrics"] = last_test_metrics

        # Save patient history for cross-run memory
        if self.logger.log_dir is not None:
            history_path = str(self.logger.log_dir / "patient_history.json")
            self.scheduler.patient_history.save(history_path)

        self.logger.save()
        self.logger.print_summary()

        # Add metadata to metrics
        metrics["stop_reason"] = self._stop_reason
        metrics["actual_steps"] = self.global_step
        metrics["actual_epochs"] = self.current_epoch
        metrics["min_epochs"] = min_epochs
        metrics["effective_max_epochs"] = effective_max_epochs
        metrics["steps_per_epoch"] = self.steps_per_epoch
        metrics["arch_stable_epoch"] = self._arch_stable_epoch
        metrics["arch_stable_step"] = self._arch_stable_step
        metrics["best_val_step"] = self._best_val_step
        metrics["best_val_epoch"] = self._best_val_epoch
        metrics["best_val_metric"] = self._best_val_metric
        metrics["target_metric"] = self.target_metric

        # Clean up periodic checkpoint
        if checkpoint_path:
            import os
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"  Removed periodic checkpoint: {checkpoint_path}")

        return metrics

    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor):
        """Apply Mixup augmentation: x_mix = λ*x + (1-λ)*x[perm], y kept as tuple.

        Returns (x_mixed, y_tuple, lam) where y_tuple = (y_a, y_b, lam).
        The loss function should compute: lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b).
        """
        lam = torch.distributions.Beta(self._mixup_alpha, self._mixup_alpha).sample().item()
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 so primary sample dominates

        batch_size = x.size(0)
        perm = torch.randperm(batch_size, device=x.device)

        x_mixed = lam * x + (1.0 - lam) * x[perm]
        y_tuple = (y, y[perm], lam)  # (y_a, y_b, lam)

        return x_mixed, y_tuple, lam

    def _apply_cutmix(self, x: torch.Tensor, y: torch.Tensor):
        """Apply CutMix (Yun et al., 2019): paste a random rectangle from x[perm] into x.

        Reuses the Mixup loss path: returns (x_mixed, (y_a, y_b, lam), lam) where
        lam is the area-adjusted ratio of original-image pixels remaining.
        Input x is a flat tensor [B, C*H*W]; reshapes to (B,C,H,W) using config.spatial_shape.
        """
        spatial = self.config.spatial_shape
        if spatial is None:
            return self._apply_mixup(x, y)
        C, H, W = spatial

        lam = torch.distributions.Beta(self._cutmix_alpha, self._cutmix_alpha).sample().item()
        lam = max(lam, 1.0 - lam)  # primary sample dominates (matches _apply_mixup convention)

        B = x.size(0)
        perm = torch.randperm(B, device=x.device)

        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        if cut_h == 0 or cut_w == 0:
            # Bbox collapsed — fall back to Mixup so we still augment
            return self._apply_mixup(x, y)

        cy = torch.randint(0, H, (1,)).item()
        cx = torch.randint(0, W, (1,)).item()
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)

        imgs = x.view(B, C, H, W).clone()
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        x_mixed = imgs.view(B, -1)

        # Adjust lam to actual pasted area (paper formula)
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(H * W)
        y_tuple = (y, y[perm], lam)
        return x_mixed, y_tuple, lam

    def _apply_balanced_sampler(self, train_data):
        """Replace the DataLoader's sampler with a moderately-balanced WeightedRandomSampler.

        Uses sqrt(inverse-frequency) instead of full inverse-frequency to avoid
        extreme oversampling of tiny classes (e.g., 18 samples seen 200+ times
        per epoch causes memorization). The sqrt smoothing provides a compromise:
        minority classes are boosted but not to full parity.
        """
        from torch.utils.data import DataLoader, WeightedRandomSampler
        import numpy as np

        dataset = train_data.dataset
        # Extract labels from dataset
        if hasattr(dataset, 'tensors') and len(dataset.tensors) >= 2:
            labels = dataset.tensors[1].cpu().numpy()
        elif hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            print(f"  [BALANCED_SAMPLER] Cannot extract labels from dataset, skipping")
            return train_data

        # Moderate oversampling: sqrt(inverse-frequency) instead of full inverse.
        # Full inverse: class with 18 samples gets weight 122x vs class with 2189.
        # Sqrt: class with 18 samples gets weight 11x vs class with 2189.
        # This avoids memorization of tiny classes while still boosting them.
        class_counts = np.bincount(labels.astype(int))
        class_weights = np.sqrt(1.0 / np.maximum(class_counts, 1).astype(np.float64))
        sample_weights = class_weights[labels.astype(int)]
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

        # Create new DataLoader with balanced sampler
        new_loader = DataLoader(
            dataset=dataset,
            batch_size=train_data.batch_size,
            sampler=sampler,
            num_workers=train_data.num_workers,
            pin_memory=train_data.pin_memory,
            drop_last=getattr(train_data, 'drop_last', False),
        )
        self.steps_per_epoch = len(new_loader)
        print(f"  [BALANCED_SAMPLER] Activated: {len(class_counts)} classes, "
              f"counts={class_counts.tolist()}, "
              f"oversampling to {len(dataset)} samples/epoch")
        return new_loader

    def _maybe_set_mol_indices(self, batch):
        """If batch has molecule indices (3rd element), set on model.

        Molecular mini-batch dataloaders yield (x, y, mol_idx) where mol_idx
        maps each sample to its position in the pre-computed molecular graphs
        list. The MolecularGraphEncoder uses these indices to build per-batch
        sub-graphs instead of processing all molecules at once.

        For non-molecular batches (2 elements), this is a no-op.
        """
        if len(batch) >= 3:
            mol_idx = batch[2].to(self.device)
            self.model.set_current_mol_indices(mol_idx)

    def _training_step(
        self,
        x_batch: torch.Tensor,
        y_batch,
        step: int,
        mixup_lam=None,
    ) -> Dict[str, float]:
        """Execute one normal training step.

        1. Forward pass
        2. Compute loss (task + complexity penalty)
        3. Backward pass
        4. LR controller step (BEFORE gradient clipping — needs raw gradients)
        5. Accumulate surgery statistics
        6. Optimizer step (handles: multi-scale momentum, per-group LR,
           multi-frequency updates, gradient clipping, NorMuon, newborn graduation)
        7. Scheduler step (only during warmup + cosine phases)

        Args:
            x_batch: Input tensor.
            y_batch: Labels tensor OR (y_a, y_b, lam) tuple for Mixup.
            step: Current global step.
            mixup_lam: If not None, Mixup was applied. y_batch is (y_a, y_b, lam).
        """
        # === TEMP DEBUG TIMING (remove after profiling) ===
        _DBG = getattr(self, '_dbg_timing', False)
        if _DBG:
            import time as _time
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            _timings = {}
            def _tick(name):
                torch.cuda.synchronize()
                nonlocal _t0
                _t = _time.perf_counter()
                _timings[name] = (_t - _t0) * 1000
                _t0 = _t

        self.model.train()
        if _DBG: _tick('01_model_train')

        # AMP autocast context for forward + loss computation
        amp_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)

        with amp_ctx:
            # Apply input transform if provided (e.g., token IDs -> embedding -> flatten)
            if self._input_transform is not None:
                x_batch = self._input_transform(x_batch)

            # Training noise augmentation (Gaussian, only during training)
            if self.config.train_noise_std > 0 and self.model.training:
                x_batch = x_batch + self.config.train_noise_std * torch.randn_like(x_batch)

            # === DETAILED FORWARD BREAKDOWN ===
            model = self.model
            h: dict = {}

            # Encoder
            if model._is_spatial:
                B = x_batch.shape[0]
                C_in, H_in, W_in = model.config.spatial_shape
                x_spatial = x_batch.view(B, C_in, H_in, W_in)
                h[0] = model.encoder(x_spatial)
            else:
                # Graph pre-aggregation: 2-hop neighborhood smoothing with
                # learnable gate, matching model.forward(). Without this,
                # training sees raw features while validation sees pre-aggregated
                # features, causing a massive train-val distribution mismatch.
                if model._is_graph:
                    from .surgery import _batch_graph_mm
                    N = model._graph_num_nodes
                    alpha = torch.sigmoid(model._graph_pre_agg_gate)
                    x_agg = _batch_graph_mm(model._graph_adj_sparse, x_batch, N)
                    x_agg = _batch_graph_mm(model._graph_adj_sparse, x_agg, N)
                    x_batch = alpha * x_agg + (1.0 - alpha) * x_batch
                h[0] = model.encoder(x_batch)
            if _DBG: _tick('02a_encoder')

            layer_inputs: dict = {}
            layer_outputs: dict = {}

            # Per-layer forward
            for l in range(model.num_layers):
                h_in = h[l]
                for conn in model.connections:
                    if conn.target == l + 1:
                        _src_out = conn.forward(h[conn.source])
                        # Defensive auto-resize for spatial mismatch
                        if (_src_out.dim() == 4 and h_in.dim() == 4
                                and _src_out.shape[-2:] != h_in.shape[-2:]):
                            import torch.nn.functional as _F_safe
                            _src_out = _F_safe.adaptive_avg_pool2d(
                                _src_out, h_in.shape[-2:])
                        h_in = h_in + _src_out

                layer_inputs[l] = h_in
                z = model.layers[l](h_in)
                layer_outputs[l] = z
                h[l + 1] = model.ops[l](z)

                # Flatten transition
                if model._is_spatial and l == model._flatten_position:
                    if getattr(model, '_use_gap', False):
                        h[l + 1] = torch.nn.functional.adaptive_avg_pool2d(
                            h[l + 1], 1).flatten(start_dim=1)
                    else:
                        h[l + 1] = h[l + 1].flatten(start_dim=1)

                if _DBG: _tick(f'02b_fwd_L{l}')

            if not _DBG:
                pass  # no-op
            else:
                pass  # per-layer ticks already recorded above

            # Output head (with JK handling)
            if getattr(model, '_jk_enabled', False):
                model._rebuild_jk()
                projs = []
                for _jl in range(model.num_layers):
                    h_jl = h[_jl + 1]
                    if model._is_spatial and h_jl.dim() == 4:
                        h_jl = torch.nn.functional.adaptive_avg_pool2d(h_jl, 1).flatten(1)
                    projs.append(model._jk_projections[_jl](h_jl))
                jk_logits = torch.cat([a for a in model._jk_attn_logits])
                jk_attn = torch.softmax(jk_logits, dim=0)
                jk_stacked = torch.stack(projs, dim=0)
                jk_out = (jk_attn.view(-1, 1, 1) * jk_stacked).sum(dim=0)
                output = model.output_head(jk_out)
            else:
                output = model.output_head(h[model.num_layers])
            if _DBG: _tick('02c_output_head')

            # Register backward hooks for surgery signals (replaces retain_grad).
            hook_results = {}
            _hooks = []

            # For backward timing: hooks that record wall-clock when grad arrives
            if _DBG:
                _bwd_times = {}
                def _make_bwd_timer(name):
                    def _timer_hook(grad):
                        torch.cuda.synchronize()
                        _bwd_times[name] = _time.perf_counter()
                        return None  # don't modify grad
                    return _timer_hook
                # Timer on output head output (first thing in backward)
                if output.requires_grad:
                    _hooks.append(output.register_hook(_make_bwd_timer('bwd_output_head')))

            for l in range(model.num_layers):
                # GDS hook on h[l+1]
                if l + 1 in h and h[l + 1].requires_grad:
                    _layer_idx = l

                    def _gds_hook(grad, _l=_layer_idx):
                        if grad.dim() == 4:
                            per_neuron = grad.abs().mean(dim=(0, 2, 3))
                            H, W = grad.shape[2], grad.shape[3]
                            spatial_factor = (H * W) ** 0.5
                            hook_results[('gds_mean', _l)] = per_neuron.mean() * spatial_factor
                        elif grad.dim() >= 2:
                            per_neuron = grad.abs().mean(dim=0)
                            hook_results[('gds_mean', _l)] = per_neuron.mean()
                        else:
                            hook_results[('gds_mean', _l)] = grad.abs().mean()

                    _hooks.append(h[l + 1].register_hook(_gds_hook))

                # CLGC hook on layer_outputs[l]
                if l in layer_outputs and layer_outputs[l].requires_grad:
                    _layer_idx = l

                    def _clgc_hook(grad, _l=_layer_idx):
                        if grad.dim() == 4:
                            hook_results[('clgc', _l)] = grad.mean(dim=(0, 2, 3))
                        elif grad.dim() >= 2:
                            hook_results[('clgc', _l)] = grad.mean(dim=0)
                        else:
                            hook_results[('clgc', _l)] = grad

                    _hooks.append(layer_outputs[l].register_hook(_clgc_hook))

                # Backward per-layer timer: fires when grad arrives at h[l] (input of layer l)
                if _DBG and l in h and h[l].requires_grad:
                    _hooks.append(h[l].register_hook(_make_bwd_timer(f'bwd_L{l}_input')))

            if _DBG: _tick('03_hooks_register')

            # Target noise regularization
            _noise_scale = getattr(self.config, 'target_noise_scale', 0.0)
            if _noise_scale > 0 and not isinstance(y_batch, tuple):
                if y_batch.is_floating_point():
                    y_batch = y_batch + _noise_scale * torch.randn_like(y_batch)

            # Compute loss
            if mixup_lam is not None and isinstance(y_batch, tuple):
                y_a, y_b, lam = y_batch
                total_a, task_a, complexity_cost, lambda_val = self.loss_fn.compute(
                    output, y_a, model)
                total_b, task_b, _, _ = self.loss_fn.compute(
                    output, y_b, model)
                total_loss = lam * total_a + (1.0 - lam) * total_b
                task_loss_val = lam * task_a + (1.0 - lam) * task_b
            else:
                total_loss, task_loss_val, complexity_cost, lambda_val = self.loss_fn.compute(
                    output, y_batch, model
                )

            # Encoder auxiliary loss (e.g., autoencoder reconstruction)
            encoder = getattr(model, 'encoder', None)
            if encoder is not None:
                _get_aux = getattr(encoder, 'get_auxiliary_loss', None)
                if _get_aux is not None:
                    aux_loss = _get_aux()
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

        if _DBG: _tick('04_loss')

        # Backward pass
        self.optimizer.zero_grad()
        if _DBG: _tick('05_zero_grad')

        if _DBG:
            _bwd_times['bwd_start'] = _time.perf_counter()

        self._grad_scaler.scale(total_loss).backward()

        if _DBG:
            torch.cuda.synchronize()
            _bwd_times['bwd_end'] = _time.perf_counter()
            # Compute per-layer backward durations from hook timestamps
            _bwd_total = (_bwd_times['bwd_end'] - _bwd_times['bwd_start']) * 1000
            _timings['06_backward_total'] = _bwd_total
            # Sorted layer times: backward goes from last layer to first
            _sorted_keys = sorted([k for k in _bwd_times if k.startswith('bwd_L')],
                                  key=lambda k: int(k.split('L')[1].split('_')[0]),
                                  reverse=True)
            if 'bwd_output_head' in _bwd_times:
                _prev = _bwd_times['bwd_output_head']
                _timings['06a_bwd_head'] = (_prev - _bwd_times['bwd_start']) * 1000
                for k in _sorted_keys:
                    l_idx = k.split('L')[1].split('_')[0]
                    _cur = _bwd_times[k]
                    _timings[f'06b_bwd_L{l_idx}'] = (_cur - _prev) * 1000
                    _prev = _cur
                _timings['06c_bwd_encoder'] = (_bwd_times['bwd_end'] - _prev) * 1000
            _t0 = _time.perf_counter()  # reset for next tick
        else:
            pass
        if not _DBG:
            pass  # backward already timed above

        # Remove hooks immediately after backward
        for hook_handle in _hooks:
            hook_handle.remove()
        del _hooks
        if _DBG: _tick('07_hooks_remove')

        # Unscale gradients
        self._grad_scaler.unscale_(self.optimizer)
        if _DBG: _tick('08_unscale')

        # Compute gradient norm on GPU
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            per_param_norms = torch._foreach_norm(grads, 2)
            grad_norm_t = torch.stack(per_param_norms).norm()
        else:
            grad_norm_t = torch.tensor(0.0, device=total_loss.device)
        if _DBG: _tick('09_grad_norm')

        # Accumulate surgery statistics
        self.scheduler.accumulate_step(
            model, total_loss, h, layer_inputs, layer_outputs,
            hook_results=hook_results,
        )
        if _DBG: _tick('10_accumulate')

        # Optimizer step
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()
        if _DBG: _tick('11_optimizer')

        # Scheduler step
        if self._should_step_scheduler():
            self.warmup_scheduler.step()
        if _DBG: _tick('12_scheduler')

        # EMA update
        if self._ema is not None:
            self._ema.update(model)
        if _DBG: _tick('13_ema')

        # === SINGLE GPU->CPU SYNC POINT ===
        _scalars_gpu = torch.stack([
            task_loss_val.detach().float(),
            total_loss.detach().float(),
            grad_norm_t.detach(),
        ])
        _scalars_cpu = _scalars_gpu.cpu()
        task_loss_f = _scalars_cpu[0].item()
        total_loss_f = _scalars_cpu[1].item()
        grad_norm_f = _scalars_cpu[2].item()
        if _DBG: _tick('14_cpu_sync')

        # CPU-side updates
        if self.lr_controller is not None:
            self.lr_controller.step(task_loss_f)

        self.scheduler.update_loss_tracking(total_loss_f)
        self.meta_learner.record_step(task_loss_f, grad_norm_f)
        self.recent_losses.append(task_loss_f)

        if _DBG:
            _tick('15_cpu_updates')
            if not hasattr(self, '_dbg_step_count'):
                self._dbg_step_count = 0
                self._dbg_accum = {}
            self._dbg_step_count += 1
            for k, v in _timings.items():
                self._dbg_accum[k] = self._dbg_accum.get(k, 0.0) + v
            if self._dbg_step_count % 20 == 0:
                n = 20
                print(f"\n  [DBG TIMING] avg over last {n} steps (step {step}):")
                for k in sorted(self._dbg_accum.keys()):
                    avg_ms = self._dbg_accum[k] / n
                    print(f"    {k:30s} {avg_ms:8.2f} ms")
                print(f"    {'TOTAL':30s} {sum(self._dbg_accum.values())/n:8.2f} ms")
                self._dbg_accum = {}

        return {
            "total_loss": total_loss_f,
            "task_loss": task_loss_f,
            "complexity_cost": complexity_cost,
            "lambda_complexity": lambda_val,
            "learning_rate": self.optimizer.current_lr,
            "grad_norm": grad_norm_f,
        }

    def _should_step_scheduler(self) -> bool:
        """Determine whether to step the warmup/cosine scheduler.

        During warmup: always step (critical for stable start).
        After warmup: if LR controller is active, it takes over LR management
        via hypergradient-based adaptation from the base_lrs it captured.
        If no LR controller, cosine scheduler continues stepping.
        """
        # Always step during warmup phases
        if self.warmup_scheduler.is_in_warmup():
            return True
        # After warmup: LR controller handles LR if active
        if self.lr_controller is not None:
            return False
        # No LR controller: let cosine scheduler manage LR
        return True

    def _validate(self, val_data: DataLoader) -> float:
        """Run validation and return mean loss. Uses EMA weights if available."""
        ema_ctx = self._ema.apply(self.model) if self._ema is not None else nullcontext()
        with ema_ctx:
            self.model.eval()
            fwd_model = self._compiled_model if self._compiled_model is not None else self.model
            total_loss = 0.0
            num_batches = 0
            amp_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)

            with torch.no_grad():
                for batch in val_data:
                    self._maybe_set_mol_indices(batch)
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    if self._input_transform is not None:
                        x = self._input_transform(x)
                    with amp_ctx:
                        output = fwd_model(x)
                        loss = self.loss_fn.task_loss_fn(output, y)
                    total_loss += loss.item()
                    num_batches += 1

            self.model.train()
        return total_loss / max(num_batches, 1)

    def _evaluate_metrics(
        self,
        data_loader: DataLoader,
        split: str,
        step: int,
    ) -> Dict[str, float]:
        """Compute task-specific metrics on a DataLoader and log them. Uses EMA weights if available.

        For classification, only stores argmax labels per batch (not full logits)
        to avoid OOM on large-vocabulary tasks like PTB (887K × 10K logits = 35 GB).
        """
        ema_ctx = self._ema.apply(self.model) if self._ema is not None else nullcontext()
        with ema_ctx:
            self.model.eval()
            fwd_model = self._compiled_model if self._compiled_model is not None else self.model
            amp_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)

            if self.task_type == "classification":
                # Classification: store argmax labels (+ optionally probs for AUROC)
                _needs_probs = self.target_metric in ("auroc", "auc_ovr")
                all_pred_labels: List[torch.Tensor] = []
                all_targets: List[torch.Tensor] = []
                all_probs: List[torch.Tensor] = [] if _needs_probs else None

                with torch.no_grad():
                    for batch in data_loader:
                        self._maybe_set_mol_indices(batch)
                        x = batch[0].to(self.device)
                        y = batch[1]
                        if self._input_transform is not None:
                            x = self._input_transform(x)
                        with amp_ctx:
                            output = fwd_model(x)
                        all_pred_labels.append(output.argmax(dim=1).cpu())
                        all_targets.append(y.cpu() if y.is_cuda else y)
                        if _needs_probs:
                            all_probs.append(torch.softmax(output, dim=1).cpu())

                self.model.train()

                import sys, os
                _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
                if _exp_dir not in sys.path:
                    sys.path.insert(0, _exp_dir)
                from common import compute_classification_metrics

                pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
                true_labels = torch.cat(all_targets, dim=0).numpy()
                y_probs_np = (
                    torch.cat(all_probs, dim=0).numpy()
                    if _needs_probs else None
                )
                eval_metrics = compute_classification_metrics(
                    true_labels, pred_labels, self.n_classes or 2,
                    y_probs=y_probs_np,
                )
            else:
                # Regression: store full outputs (always small, typically 1-dim)
                all_preds: List[torch.Tensor] = []
                all_targets: List[torch.Tensor] = []

                with torch.no_grad():
                    for batch in data_loader:
                        self._maybe_set_mol_indices(batch)
                        x = batch[0].to(self.device)
                        y = batch[1]
                        if self._input_transform is not None:
                            x = self._input_transform(x)
                        with amp_ctx:
                            output = fwd_model(x).cpu()
                        all_preds.append(output)
                        all_targets.append(y.cpu() if y.is_cuda else y)

                self.model.train()

                import sys, os
                _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
                if _exp_dir not in sys.path:
                    sys.path.insert(0, _exp_dir)
                from common import compute_regression_metrics

                preds = torch.cat(all_preds, dim=0)
                targets = torch.cat(all_targets, dim=0)
                p = preds.numpy()
                t = targets.numpy()
                if self.y_scaler is not None:
                    p = self.y_scaler.inverse_transform(p)
                    t = self.y_scaler.inverse_transform(t)
                eval_metrics = compute_regression_metrics(t, p)

        self.logger.log_evaluation_metrics(step, split, eval_metrics)
        return eval_metrics

    # ------------------------------------------------------------------
    # Optimised inner helpers (no EMA context – caller wraps EMA once)
    # ------------------------------------------------------------------

    def _validate_with_metrics_inner(
        self,
        data_loader: DataLoader,
        split: str,
        step: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Merged validate + evaluate_metrics in a single forward pass.

        Caller is responsible for EMA context and model.eval().
        Returns ``(mean_loss, metrics_dict)``.
        """
        fwd_model = self._compiled_model if self._compiled_model is not None else self.model
        amp_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)

        total_loss = 0.0
        num_batches = 0

        if self.task_type == "classification":
            _needs_probs = self.target_metric in ("auroc", "auc_ovr")
            all_pred_labels: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []
            all_probs: List[torch.Tensor] = [] if _needs_probs else None

            with torch.no_grad():
                for batch in data_loader:
                    self._maybe_set_mol_indices(batch)
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    if self._input_transform is not None:
                        x = self._input_transform(x)
                    with amp_ctx:
                        output = fwd_model(x)
                        loss = self.loss_fn.task_loss_fn(output, y)
                    total_loss += loss.item()
                    num_batches += 1
                    all_pred_labels.append(output.argmax(dim=1).cpu())
                    all_targets.append(y.cpu())
                    if _needs_probs:
                        all_probs.append(torch.softmax(output, dim=1).cpu())

            import sys, os
            _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
            if _exp_dir not in sys.path:
                sys.path.insert(0, _exp_dir)
            from common import compute_classification_metrics

            pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
            true_labels = torch.cat(all_targets, dim=0).numpy()
            y_probs_np = (
                torch.cat(all_probs, dim=0).numpy()
                if _needs_probs else None
            )
            eval_metrics = compute_classification_metrics(
                true_labels, pred_labels, self.n_classes or 2,
                y_probs=y_probs_np,
            )
        else:
            # Regression
            all_preds: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []

            with torch.no_grad():
                for batch in data_loader:
                    self._maybe_set_mol_indices(batch)
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    if self._input_transform is not None:
                        x = self._input_transform(x)
                    with amp_ctx:
                        output = fwd_model(x)
                        loss = self.loss_fn.task_loss_fn(output, y)
                    total_loss += loss.item()
                    num_batches += 1
                    all_preds.append(output.cpu())
                    all_targets.append(y.cpu())

            import sys, os
            _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
            if _exp_dir not in sys.path:
                sys.path.insert(0, _exp_dir)
            from common import compute_regression_metrics

            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            p = preds.numpy()
            t = targets.numpy()
            if self.y_scaler is not None:
                p = self.y_scaler.inverse_transform(p)
                t = self.y_scaler.inverse_transform(t)
            eval_metrics = compute_regression_metrics(t, p)

        mean_loss = total_loss / max(num_batches, 1)
        self.logger.log_evaluation_metrics(step, split, eval_metrics)
        return mean_loss, eval_metrics

    def _evaluate_metrics_inner(
        self,
        data_loader: DataLoader,
        split: str,
        step: int,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate metrics without EMA context (caller wraps EMA once).

        When *max_batches* is set, only the first N batches are evaluated
        (useful for subsampling the train set during epoch-level eval).
        """
        fwd_model = self._compiled_model if self._compiled_model is not None else self.model
        amp_ctx = torch.amp.autocast('cuda', enabled=self._amp_enabled)

        if self.task_type == "classification":
            _needs_probs = self.target_metric in ("auroc", "auc_ovr")
            all_pred_labels: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []
            all_probs: List[torch.Tensor] = [] if _needs_probs else None

            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    self._maybe_set_mol_indices(batch)
                    x = batch[0].to(self.device)
                    y = batch[1]
                    if self._input_transform is not None:
                        x = self._input_transform(x)
                    with amp_ctx:
                        output = fwd_model(x)
                    all_pred_labels.append(output.argmax(dim=1).cpu())
                    all_targets.append(y.cpu() if y.is_cuda else y)
                    if _needs_probs:
                        all_probs.append(torch.softmax(output, dim=1).cpu())

            import sys, os
            _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
            if _exp_dir not in sys.path:
                sys.path.insert(0, _exp_dir)
            from common import compute_classification_metrics

            pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
            true_labels = torch.cat(all_targets, dim=0).numpy()
            y_probs_np = (
                torch.cat(all_probs, dim=0).numpy()
                if _needs_probs else None
            )
            eval_metrics = compute_classification_metrics(
                true_labels, pred_labels, self.n_classes or 2,
                y_probs=y_probs_np,
            )
        else:
            # Regression
            all_preds: List[torch.Tensor] = []
            all_targets: List[torch.Tensor] = []

            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    self._maybe_set_mol_indices(batch)
                    x = batch[0].to(self.device)
                    y = batch[1]
                    if self._input_transform is not None:
                        x = self._input_transform(x)
                    with amp_ctx:
                        output = fwd_model(x).cpu()
                    all_preds.append(output)
                    all_targets.append(y.cpu() if y.is_cuda else y)

            import sys, os
            _exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
            if _exp_dir not in sys.path:
                sys.path.insert(0, _exp_dir)
            from common import compute_regression_metrics

            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            p = preds.numpy()
            t = targets.numpy()
            if self.y_scaler is not None:
                p = self.y_scaler.inverse_transform(p)
                t = self.y_scaler.inverse_transform(t)
            eval_metrics = compute_regression_metrics(t, p)

        self.logger.log_evaluation_metrics(step, split, eval_metrics)
        return eval_metrics

    def _print_status(self, step: int, max_steps: int, metrics: Dict[str, float]):
        """Print training status."""
        arch = self.model.describe_architecture()
        stable_str = " [STABLE]" if self.model.architecture_stable else ""
        phase_str = f" ({self.warmup_scheduler.get_phase()})" if self.warmup_scheduler.get_phase() != "cosine" else ""

        # Show channel counts for spatial layers, widths for flat layers
        layer_descs = []
        for l_info in arch["layers"]:
            mode = l_info.get("mode", "flat")
            if mode == "spatial":
                channels = l_info.get("channels", l_info["out_features"])
                stride = l_info.get("stride", 1)
                stride_str = f"/s{stride}" if stride > 1 else ""
                layer_descs.append(f"C{channels}{stride_str}")
            else:
                layer_descs.append(str(l_info["out_features"]))

        # Add flatten position indicator for spatial models
        flatten_pos = arch.get("flatten_position", None)
        if flatten_pos is not None and flatten_pos < len(layer_descs):
            layer_descs.insert(flatten_pos + 1, "|F|")

        print(
            f"Step {step}/{max_steps} | "
            f"Loss: {metrics['task_loss']:.6f} | "
            f"LR: {metrics['learning_rate']:.6f}{phase_str} | "
            f"Layers: {arch['num_layers']} | "
            f"Arch: [{', '.join(layer_descs)}] | "
            f"Conns: {len(arch['connections'])} | "
            f"Params: {arch['total_parameters']} | "
            f"Cost: {arch['architecture_cost']:.0f}"
            f"{stable_str}"
        )

    def save_checkpoint(self, path: str):
        """Save full training state for resumption."""
        checkpoint = {
            # Full model object (captures architecture + weights for resume)
            "model": self.model,
            # Also save state_dict for potential manual recovery
            "model_state_dict": self.model.state_dict(),
            # Optimizer and scheduler state
            "optimizer_state_dict": self.optimizer.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "meta_learner_state_dict": self.meta_learner.state_dict(),
            "surgery_scheduler_state_dict": self.scheduler.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            # LR controller (optional component)
            "lr_controller_state_dict": (
                self.lr_controller.state_dict()
                if self.lr_controller is not None else None
            ),
            # Trainer-level state
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "best_loss": self.best_loss,
            "recent_losses": list(self.recent_losses),
            "_stable_model_saved": self._stable_model_saved,
            "_arch_stable_step": self._arch_stable_step,
            "_arch_stable_epoch": self._arch_stable_epoch,
            "_best_val_loss_post_stable": self._best_val_loss_post_stable,
            "_val_patience_counter": self._val_patience_counter,
            "_stop_reason": self._stop_reason,
            # Best model state
            "_best_val_metric": self._best_val_metric,
            "_best_val_loss_at_best": self._best_val_loss_at_best,
            "_best_val_step": self._best_val_step,
            "_best_val_epoch": self._best_val_epoch,
            "_best_model_state": self._best_model_state,  # already on CPU
            "_best_model_copy": self._best_model_copy,  # deep-copied model (CPU)
            "_best_model_arch": self._best_model_arch,
            "_best_model_connections": self._best_model_connections,
            # AMP state
            "grad_scaler_state_dict": (
                self._grad_scaler.state_dict() if self._amp_enabled else None
            ),
            # Metadata
            "config": self.config,
            "d_input": self.model.d_input,
            "d_output": self.model.d_output,
            "architecture": self.model.describe_architecture(),
            "target_metric": self.target_metric,
            "checkpoint_version": 3,
        }
        torch.save(checkpoint, path)
        import os
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [CHECKPOINT] Saved epoch {self.current_epoch} "
              f"(step {self.global_step}) -> {path} ({size_mb:.1f} MB)")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        task_loss_fn: Callable,
        log_dir: Optional[str] = None,
        task_type: str = "regression",
        y_scaler: Optional[Any] = None,
        n_classes: Optional[int] = None,
        input_transform: Optional[Callable] = None,
        extra_params: Optional[List[nn.Parameter]] = None,
        target_metric: Optional[str] = None,
    ) -> "ASANNTrainer":
        """Load a full training state from checkpoint for resumption.

        Returns a fully reconstructed ASANNTrainer ready to continue training.
        The caller must provide the same task_loss_fn and task-related params
        since those are not serializable.

        Robust against code changes: if the pickled model object fails to load,
        falls back to reconstructing from config + model_state_dict.
        """
        import os
        from asann.model import ASANNModel

        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]

        # Restore target_metric: prefer explicit arg > checkpoint > default
        if target_metric is None:
            target_metric = checkpoint.get("target_metric")  # may still be None for old checkpoints

        # Try to restore model — prefer pickled object, fall back to state_dict
        model = None
        try:
            model = checkpoint["model"]
            # Quick sanity check: make sure model actually works
            _ = model.describe_architecture()
        except Exception as e:
            print(f"  [RESUME] Pickled model failed ({e}), reconstructing from state_dict...")
            model = None

        if model is None:
            # The pickled model failed — likely due to code changes between runs.
            # We cannot reconstruct surgery-modified architectures from state_dict
            # alone (it would create a fresh 2-layer model, not the evolved 6-layer one).
            # The user needs to delete the stale checkpoint and start fresh.
            epoch = checkpoint.get("current_epoch", "?")
            step = checkpoint.get("global_step", "?")
            raise RuntimeError(
                f"Checkpoint model (epoch {epoch}, step {step}) is incompatible "
                f"with current code. This happens when code is modified between "
                f"runs. Please delete the checkpoint file and restart: {path}"
            )

        # Create trainer (re-creates optimizer, schedulers, etc. from model)
        trainer = cls(
            model=model,
            config=config,
            task_loss_fn=task_loss_fn,
            log_dir=log_dir,
            task_type=task_type,
            y_scaler=y_scaler,
            n_classes=n_classes,
            append_logs=True,
            input_transform=input_transform,
            extra_params=extra_params,
            target_metric=target_metric,
        )

        # Restore optimizer state (momentum buffers, step counters, etc.)
        # Note: param groups may differ if surgery changed the model structure
        # during training. In that case, we fall back to the fresh optimizer
        # (loses momentum buffers but retains correct param groups).
        optimizer_restored = False
        try:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_restored = True
        except (ValueError, KeyError) as e:
            print(f"  [RESUME] Warning: Could not restore optimizer state ({e})")
            print(f"  [RESUME] Using fresh optimizer - momentum buffers reset")

        # Restore scheduler states
        try:
            trainer.warmup_scheduler.load_state_dict(
                checkpoint["warmup_scheduler_state_dict"]
            )
        except (ValueError, KeyError) as e:
            print(f"  [RESUME] Warning: Could not restore warmup scheduler ({e})")

        trainer.meta_learner.load_state_dict(
            checkpoint["meta_learner_state_dict"]
        )

        # Surgery scheduler (new in checkpoint v2)
        if "surgery_scheduler_state_dict" in checkpoint:
            trainer.scheduler.load_state_dict(
                checkpoint["surgery_scheduler_state_dict"]
            )

        # Loss function state (new in checkpoint v2)
        if "loss_fn_state_dict" in checkpoint:
            trainer.loss_fn.load_state_dict(
                checkpoint["loss_fn_state_dict"]
            )

        # LR controller (optional component)
        # Only restore if optimizer was also restored — the lr_controller's
        # group_names must match the optimizer's parameter groups.
        if (trainer.lr_controller is not None
                and optimizer_restored
                and checkpoint.get("lr_controller_state_dict")):
            try:
                trainer.lr_controller.load_state_dict(
                    checkpoint["lr_controller_state_dict"]
                )
            except (ValueError, KeyError) as e:
                print(f"  [RESUME] Warning: Could not restore LR controller ({e})")
        elif trainer.lr_controller is not None and not optimizer_restored:
            print(f"  [RESUME] Skipping LR controller restore (optimizer groups changed)")

        # Restore trainer-level state
        trainer.global_step = checkpoint["global_step"]
        trainer.current_epoch = checkpoint.get("current_epoch", 0)
        trainer.steps_per_epoch = checkpoint.get("steps_per_epoch", 0)
        trainer.best_loss = checkpoint["best_loss"]
        trainer.recent_losses = deque(
            checkpoint.get("recent_losses", []),
            maxlen=config.meta_update_interval,
        )
        trainer._stable_model_saved = checkpoint.get("_stable_model_saved", False)
        trainer._arch_stable_step = checkpoint.get("_arch_stable_step")
        trainer._arch_stable_epoch = checkpoint.get("_arch_stable_epoch")
        trainer._best_val_loss_post_stable = checkpoint.get(
            "_best_val_loss_post_stable", float("inf")
        )
        trainer._val_patience_counter = checkpoint.get("_val_patience_counter", 0)
        trainer._stop_reason = checkpoint.get("_stop_reason")

        # Restore best model state (checkpoint v3+)
        trainer._best_val_metric = checkpoint.get("_best_val_metric")
        trainer._best_val_loss_at_best = checkpoint.get("_best_val_loss_at_best")
        trainer._best_val_step = checkpoint.get("_best_val_step")
        trainer._best_val_epoch = checkpoint.get("_best_val_epoch")
        trainer._best_model_state = checkpoint.get("_best_model_state")
        trainer._best_model_copy = checkpoint.get("_best_model_copy")
        trainer._best_model_arch = checkpoint.get("_best_model_arch")
        trainer._best_model_connections = checkpoint.get("_best_model_connections")
        if trainer._best_val_metric is not None:
            print(f"  [RESUME] Best val {trainer.target_metric}: {trainer._best_val_metric:.4f} "
                  f"at step {trainer._best_val_step}")

        # Restore AMP GradScaler state
        if trainer._amp_enabled and checkpoint.get("grad_scaler_state_dict"):
            try:
                trainer._grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            except Exception as e:
                print(f"  [RESUME] Warning: Could not restore GradScaler ({e})")

        # Sync architecture_stable flag on model from scheduler
        if "surgery_scheduler_state_dict" in checkpoint:
            model.architecture_stable = checkpoint["surgery_scheduler_state_dict"].get(
                "architecture_stable", False
            )

        # --- Repair stale skip connection projections ---
        # Checkpoints saved before the _repair_connection_projections fix may
        # contain connections whose projections have wrong channel dimensions
        # (e.g., Conv2d(C_old_src, C_old_tgt) doesn't match the actual layer
        # channels after surgery shifted indices). Fix them here at load time.
        _repaired = cls._repair_loaded_connections(model, config)
        if _repaired > 0:
            print(f"  [RESUME] Repaired {_repaired} stale skip connection projection(s)")

        arch = checkpoint.get("architecture", {})
        print(f"  [RESUME] Loaded checkpoint from {path}")
        print(f"  [RESUME] Resuming from step {trainer.global_step}")
        print(f"  [RESUME] Architecture: {arch.get('num_layers', '?')} layers, "
              f"{arch.get('total_parameters', '?')} params")

        return trainer

    @staticmethod
    def _repair_loaded_connections(model, config) -> int:
        """Repair stale skip connection projections on a loaded model.

        Returns the number of connections repaired.

        This handles checkpoints saved before the _repair_connection_projections
        fix: projections may have wrong channel/feature dimensions because
        surgery shifted indices but never updated the projections.
        """
        repaired = 0
        device = config.device

        for conn in model.connections:
            # --- Get actual source dimensions ---
            if conn.source == 0:
                src_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif conn.source - 1 < len(model.layers):
                src_spatial = getattr(model.layers[conn.source - 1], 'spatial_shape', None)
            else:
                continue

            # --- Get actual target dimensions ---
            tgt_h_idx = conn.target - 1
            if tgt_h_idx == 0:
                tgt_spatial = getattr(model.input_projection, 'spatial_shape', None)
            elif tgt_h_idx - 1 < len(model.layers):
                tgt_spatial = getattr(model.layers[tgt_h_idx - 1], 'spatial_shape', None)
            else:
                continue

            is_spatial = (src_spatial is not None and tgt_spatial is not None)

            if is_spatial:
                actual_C_src = src_spatial[0]
                actual_C_tgt = tgt_spatial[0]

                if conn.projection is not None and hasattr(conn.projection, 'in_channels'):
                    needs_repair = (conn.projection.in_channels != actual_C_src
                                    or conn.projection.out_channels != actual_C_tgt)
                    if needs_repair:
                        old_proj = conn.projection
                        new_proj = nn.Conv2d(
                            actual_C_src, actual_C_tgt,
                            kernel_size=1, bias=False,
                        ).to(device)
                        # Copy overlapping weights
                        c_in = min(old_proj.in_channels, actual_C_src)
                        c_out = min(old_proj.out_channels, actual_C_tgt)
                        new_proj.weight.data[:c_out, :c_in] = old_proj.weight.data[:c_out, :c_in]
                        conn.projection = new_proj
                        conn.spatial_source_shape = src_spatial
                        conn.spatial_target_shape = tgt_spatial
                        repaired += 1
                elif conn.projection is None and actual_C_src != actual_C_tgt:
                    # Need a projection that didn't exist
                    new_proj = nn.Conv2d(
                        actual_C_src, actual_C_tgt,
                        kernel_size=1, bias=False,
                    ).to(device)
                    nn.init.zeros_(new_proj.weight)
                    conn.projection = new_proj
                    conn.spatial_source_shape = src_spatial
                    conn.spatial_target_shape = tgt_spatial
                    repaired += 1
            else:
                # Flat connections
                if conn.source == 0:
                    d_source = model.input_projection.out_features
                else:
                    d_source = model.layers[conn.source - 1].out_features
                if tgt_h_idx == 0:
                    d_target = model.input_projection.out_features
                else:
                    d_target = model.layers[tgt_h_idx - 1].out_features

                if conn.projection is not None and hasattr(conn.projection, 'in_features'):
                    needs_repair = (conn.projection.in_features != d_source
                                    or conn.projection.out_features != d_target)
                    if needs_repair:
                        old_proj = conn.projection
                        new_proj = nn.Linear(d_source, d_target, bias=False).to(device)
                        r = min(old_proj.out_features, d_target)
                        c = min(old_proj.in_features, d_source)
                        new_proj.weight.data[:r, :c] = old_proj.weight.data[:r, :c]
                        conn.projection = new_proj
                        repaired += 1
                elif conn.projection is None and d_source != d_target:
                    new_proj = nn.Linear(d_source, d_target, bias=False).to(device)
                    nn.init.zeros_(new_proj.weight)
                    conn.projection = new_proj
                    repaired += 1

        return repaired

    def save_model(self, path: str):
        """Save stable model for inference (no optimizer/scheduler state)."""
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "architecture": self.model.describe_architecture(),
            "architecture_stable": self.model.architecture_stable,
            "connections": [
                {
                    "source": c.source,
                    "target": c.target,
                    "scale": c.scale.item(),
                    "has_projection": c.projection is not None,
                }
                for c in self.model.connections
            ],
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
        torch.save(model_data, path)

    def _get_primary_metric(self, eval_metrics: Dict[str, float]) -> tuple:
        """Return (metric_value, higher_is_better) for the configured target metric.

        The target metric and its direction are set in __init__ via `target_metric`.
        """
        if self.target_metric in eval_metrics:
            return eval_metrics[self.target_metric], self._target_metric_higher_is_better
        # Metric not in eval_metrics (e.g., auroc requires probs but probs
        # weren't collected yet). Return worst-possible default.
        default = -float("inf") if self._target_metric_higher_is_better else float("inf")
        return default, self._target_metric_higher_is_better

    def _update_best_model(self, step: int, val_metrics: Dict[str, float]):
        """Check if current model is the best so far and snapshot it if so.

        Only tracks best model after architecture stabilization, since restoring
        a state_dict from a different architecture would fail. Before stabilization,
        surgery may add/remove neurons and change parameter shapes.
        """
        if not self.model.architecture_stable:
            return

        metric_val, higher_is_better = self._get_primary_metric(val_metrics)
        metric_name = self.target_metric

        is_better = False
        if self._best_val_metric is None:
            is_better = True
        elif higher_is_better:
            is_better = metric_val > self._best_val_metric
        else:
            is_better = metric_val < self._best_val_metric

        if is_better:
            self._best_val_metric = metric_val
            self._best_val_step = step
            # Deep-copy entire model (architecture + weights) for restore
            self._best_model_copy = copy.deepcopy(self.model).cpu()
            # Also keep state dict + arch description for checkpoint saving
            self._best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            self._best_model_arch = self.model.describe_architecture()
            self._best_model_connections = [
                {
                    "source": c.source,
                    "target": c.target,
                    "scale": c.scale.item(),
                    "has_projection": c.projection is not None,
                }
                for c in self.model.connections
            ]
            print(f"  [BEST] New best val {metric_name}: {metric_val:.4f} at step {step}")
            # Save best model to disk immediately (don't wait until end of training)
            if self.logger.log_dir is not None:
                best_path = str(self.logger.log_dir / "best_model.pt")
                self._save_best_model(best_path)
        else:
            print(f"  [BEST] Val {metric_name}: {metric_val:.4f} "
                  f"(best: {self._best_val_metric:.4f} @ step {self._best_val_step})")

    def _restore_best_model(self):
        """Restore model from the best validation checkpoint.

        Strategy (preserves object identity when possible):
        1. Try state_dict from best model copy into current model (in-place, same object)
        2. If arch changed, replace self.model with the deep copy (new object)
        3. Last resort: use final model as-is

        Note: callers that cache `model = trainer.model` before training should
        re-read `trainer.model` after train_epochs() in case step 2 was used.
        """
        metric_name = self.target_metric

        # Step 1: Try in-place restoration via state_dict (preserves object identity).
        # This works when architecture hasn't changed since the best was saved.
        if self._best_model_state is not None:
            try:
                device = next(self.model.parameters()).device
                state_dict = {
                    k: v.to(device) for k, v in self._best_model_state.items()
                }
                self.model.load_state_dict(state_dict)
                epoch_str = (f"epoch {self._best_val_epoch}"
                             if hasattr(self, '_best_val_epoch') and self._best_val_epoch is not None
                             else f"step {self._best_val_step}")
                print(f"  [BEST] Restored best model in-place from {epoch_str} "
                      f"(val {metric_name}: {self._best_val_metric:.4f})")
                return
            except (RuntimeError, KeyError):
                # Architecture changed (surgery added/removed layers) -- need deep copy
                pass

        # Step 2: Architecture changed -- use deep-copied model (new object).
        # WARNING: this rebinds self.model; external references become stale.
        if hasattr(self, '_best_model_copy') and self._best_model_copy is not None:
            try:
                device = next(self.model.parameters()).device
                self.model = self._best_model_copy.to(device)
                epoch_str = (f"epoch {self._best_val_epoch}"
                             if hasattr(self, '_best_val_epoch') and self._best_val_epoch is not None
                             else f"step {self._best_val_step}")
                print(f"  [BEST] Restored best model (new object) from {epoch_str} "
                      f"(val {metric_name}: {self._best_val_metric:.4f})")
                return
            except Exception as e:
                print(f"  [BEST] Warning: Deep-copy restore failed ({e})")

        print(f"  [BEST] Using final model instead")

    def _save_best_model(self, path: str):
        """Save the best model to disk (state_dict + full pickle)."""
        if self._best_model_state is None:
            return
        model_data = {
            "model_state_dict": self._best_model_state,
            "config": self.config,
            "architecture": self._best_model_arch,
            "connections": self._best_model_connections,
            "global_step": self._best_val_step,
            "best_val_metric": self._best_val_metric,
            "metric_name": self.target_metric,
            "architecture_stable": self.model.architecture_stable,
        }
        torch.save(model_data, path)
        metric_name = model_data["metric_name"]
        print(f"  [BEST] Best model saved to {path} "
              f"(step {self._best_val_step}, val {metric_name}: {self._best_val_metric:.4f})")

        # Also save full model pickle for standalone evaluation
        if self._best_model_copy is not None:
            full_path = path.replace(".pt", "_full.pt")
            # Save on CPU for portability, then restore original device.
            # After _restore_best_model(), self.model IS self._best_model_copy
            # (same object), so .cpu() would also move self.model to CPU.
            device = next(self._best_model_copy.parameters()).device
            torch.save(self._best_model_copy.cpu(), full_path)
            self._best_model_copy.to(device)  # restore to original device
            print(f"  [BEST] Full model pickle saved to {full_path}")

    def _check_weight_convergence(
        self, step: int, val_loss: float,
    ) -> tuple:
        """Check weight convergence using validation loss after architecture stabilizes.

        Called at eval intervals when auto_stop_enabled and step >= min_steps.
        Returns (should_stop: bool, reason: str).
        """
        if not self.model.architecture_stable:
            # Architecture not stable — reset tracking
            self._arch_stable_step = None
            self._val_patience_counter = 0
            self._best_val_loss_post_stable = float("inf")
            return False, ""

        # Track when architecture first became stable
        if self._arch_stable_step is None:
            self._arch_stable_step = step

        # Ensure minimum post-stable training before checking
        if step - self._arch_stable_step < self.config.post_stable_min_steps:
            return False, ""

        # Check if val loss improved
        if self._best_val_loss_post_stable == float("inf"):
            # First check — initialize baseline
            self._best_val_loss_post_stable = val_loss
            self._val_patience_counter = 0
            return False, ""

        improvement = (
            (self._best_val_loss_post_stable - val_loss)
            / max(abs(self._best_val_loss_post_stable), 1e-8)
        )
        if improvement > self.config.post_stable_improvement_threshold:
            self._best_val_loss_post_stable = val_loss
            self._val_patience_counter = 0
        else:
            self._val_patience_counter += 1

        if self._val_patience_counter >= self.config.post_stable_patience_intervals:
            return True, (
                f"architecture stable since step {self._arch_stable_step}, "
                f"val loss converged ({self._val_patience_counter} eval intervals "
                f"without {self.config.post_stable_improvement_threshold*100:.1f}% improvement)"
            )

        return False, ""

    def _check_train_convergence(self, step: int) -> tuple:
        """Fallback convergence check using train loss when no val data available.

        Returns (should_stop: bool, reason: str).
        """
        if not self.model.architecture_stable:
            return False, ""

        if self._arch_stable_step is None:
            self._arch_stable_step = step

        if step - self._arch_stable_step < self.config.post_stable_min_steps:
            return False, ""

        # Use recent_losses deque for plateau detection
        if len(self.recent_losses) < 100:
            return False, ""

        recent = list(self.recent_losses)
        half = len(recent) // 2
        first_half_mean = sum(recent[:half]) / half
        second_half_mean = sum(recent[half:]) / half
        relative_improvement = (
            (first_half_mean - second_half_mean)
            / max(abs(first_half_mean), 1e-8)
        )

        if relative_improvement < self.config.post_stable_improvement_threshold:
            self._val_patience_counter += 1
        else:
            self._val_patience_counter = 0

        if self._val_patience_counter >= self.config.post_stable_patience_intervals:
            return True, (
                f"architecture stable since step {self._arch_stable_step}, "
                f"train loss converged (plateau detected)"
            )

        return False, ""

    # ============================= Epoch-Based Helpers =============================

    def _print_status_epoch(
        self, step: int, epoch: int, max_epochs: int, metrics: Dict[str, float],
    ):
        """Print training status for epoch-based training."""
        arch = self.model.describe_architecture()
        stable_str = " [STABLE]" if self.model.architecture_stable else ""
        phase_str = (f" ({self.warmup_scheduler.get_phase()})"
                     if self.warmup_scheduler.get_phase() != "cosine" else "")

        # Show channel counts for spatial layers, widths for flat layers
        layer_descs = []
        for l_info in arch["layers"]:
            mode = l_info.get("mode", "flat")
            if mode == "spatial":
                channels = l_info.get("channels", l_info["out_features"])
                stride = l_info.get("stride", 1)
                stride_str = f"/s{stride}" if stride > 1 else ""
                layer_descs.append(f"C{channels}{stride_str}")
            else:
                layer_descs.append(str(l_info["out_features"]))

        # Add flatten position indicator
        flatten_pos = arch.get("flatten_position", None)
        if flatten_pos is not None and flatten_pos < len(layer_descs):
            layer_descs.insert(flatten_pos + 1, "|F|")

        print(
            f"E{epoch}/{max_epochs} S{step} | "
            f"Loss: {metrics['task_loss']:.6f} | "
            f"LR: {metrics['learning_rate']:.6f}{phase_str} | "
            f"Layers: {arch['num_layers']} | "
            f"Arch: [{', '.join(layer_descs)}] | "
            f"Conns: {len(arch['connections'])} | "
            f"Params: {arch['total_parameters']} | "
            f"Cost: {arch['architecture_cost']:.0f}"
            f"{stable_str}"
        )

    def _update_best_model_epoch(
        self, epoch: int, step: int, val_metrics: Dict[str, float],
        val_loss: Optional[float] = None,
    ):
        """Check if current model is the best so far (epoch-based version).

        In epoch mode, we track best model even before architecture stabilization,
        since the diagnosis system properly manages stability. We only require
        being past the warmup phase.

        When the primary metric ties (e.g., F1=1.0 vs F1=1.0), uses val_loss
        as a tiebreaker — lower val_loss means better generalization even when
        the discrete metric (accuracy, F1) has saturated.
        """
        if epoch <= self.config.warmup_epochs:
            return

        metric_val, higher_is_better = self._get_primary_metric(val_metrics)
        metric_name = self.target_metric

        is_better = False
        if self._best_val_metric is None:
            is_better = True
        elif higher_is_better:
            if metric_val > self._best_val_metric:
                is_better = True
            elif (metric_val == self._best_val_metric
                  and val_loss is not None
                  and hasattr(self, '_best_val_loss_at_best')
                  and self._best_val_loss_at_best is not None
                  and val_loss < self._best_val_loss_at_best):
                is_better = True
        else:
            if metric_val < self._best_val_metric:
                is_better = True
            elif (metric_val == self._best_val_metric
                  and val_loss is not None
                  and hasattr(self, '_best_val_loss_at_best')
                  and self._best_val_loss_at_best is not None
                  and val_loss < self._best_val_loss_at_best):
                is_better = True

        if is_better:
            self._best_val_metric = metric_val
            self._best_val_loss_at_best = val_loss
            self._best_val_step = step
            self._best_val_epoch = epoch
            # Deep-copy entire model (architecture + weights) so we can restore
            # even after surgery changes the architecture.
            # Move to CPU to save GPU memory.
            self._best_model_copy = copy.deepcopy(self.model).cpu()
            # Also keep state dict + arch description for checkpoint saving
            self._best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            self._best_model_arch = self.model.describe_architecture()
            self._best_model_connections = [
                {
                    "source": c.source,
                    "target": c.target,
                    "scale": c.scale.item(),
                    "has_projection": c.projection is not None,
                }
                for c in self.model.connections
            ]
            print(f"  [BEST] New best val {metric_name}: {metric_val:.4f} "
                  f"at epoch {epoch} (step {step})")
            # Save best model to disk immediately (don't wait until end of training)
            if self.logger.log_dir is not None:
                best_path = str(self.logger.log_dir / "best_model.pt")
                self._save_best_model(best_path)
        else:
            print(f"  [BEST] Val {metric_name}: {metric_val:.4f} "
                  f"(best: {self._best_val_metric:.4f} @ epoch {self._best_val_epoch})")

    def _check_epoch_convergence(
        self, epoch: int, val_loss: float,
    ) -> tuple:
        """Check weight convergence using validation loss (epoch-based version).

        For epoch-based training, auto-stop requires:
        1. Architecture is stable (healthy for N consecutive epochs)
        2. Val loss hasn't improved for post_stable_patience_epochs consecutive evals

        Returns (should_stop: bool, reason: str).
        """
        if not self.model.architecture_stable:
            self._val_patience_counter = 0
            self._best_val_loss_post_stable = float("inf")
            return False, ""

        # Ensure minimum post-stable epochs before checking
        if self._arch_stable_epoch is None:
            return False, ""

        min_post_stable = max(3, self.config.recovery_epochs * 2)
        if epoch - self._arch_stable_epoch < min_post_stable:
            return False, ""

        # Check if val loss improved
        if self._best_val_loss_post_stable == float("inf"):
            self._best_val_loss_post_stable = val_loss
            self._val_patience_counter = 0
            return False, ""

        improvement = (
            (self._best_val_loss_post_stable - val_loss)
            / max(abs(self._best_val_loss_post_stable), 1e-8)
        )
        if improvement > self.config.post_stable_improvement_threshold:
            self._best_val_loss_post_stable = val_loss
            self._val_patience_counter = 0
        else:
            self._val_patience_counter += 1

        if self._val_patience_counter >= self.config.post_stable_patience_epochs:
            return True, (
                f"architecture stable since epoch {self._arch_stable_epoch}, "
                f"val loss converged ({self._val_patience_counter} eval epochs "
                f"without {self.config.post_stable_improvement_threshold*100:.1f}% improvement)"
            )

        return False, ""

    @staticmethod
    def create_dataloader(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Helper to create a DataLoader from tensors."""
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Backward-compat alias
CSANNTrainer = ASANNTrainer
