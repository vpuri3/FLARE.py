#
import gc
import math
import time
from typing import TYPE_CHECKING
import torch
from torch import nn, optim
from torch import distributed as dist
from torch.utils.data import DistributedSampler, BatchSampler, RandomSampler, SequentialSampler

from tqdm import tqdm

# builtin
import os
import collections
from typing import Union, List, Optional, Callable, Any, Tuple

# local
from mlutils.utils import (
    num_parameters, select_device, is_torchrun,
    RepeatBatchSampler,
    StepBudgetBatchSampler,
)
from mlutils.ema import *
from mlutils.metrics import format_metric, normalize_metric, unset_metric

if TYPE_CHECKING:
    from mlutils.run_timer import RunTimer

__all__ = [
    'Trainer',
]


def _safe_one_cycle_pct_start(pct_start: float, total_steps: int) -> float:
    min_pct_start = (1.0 + 1e-6) / max(1, total_steps)
    return max(pct_start, min_pct_start)


class _DDPStatsModelProxy(nn.Module):
    """Expose an eager module through `.module` for stats functions that unwrap DDP."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


#======================================================================#
class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        _data: Any,  # Must be iterable (Dataset, PyG Dataset, etc.)
        data_: Optional[Any] = None,  # Must be iterable (Dataset, PyG Dataset, etc.)

        gnn_loader: bool = False,
        graph_loader_backend: Optional[str] = None,  # 'pyg' or 'dgl' when gnn_loader=True
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: bool = False,
        amp_dtype: Optional[str] = None,  # fp16, bf16, or None (use torch.autocast default)

        # compilation
        compile_model: bool = True,
        compile_stats_model: bool = True,
        static_graph: bool = False,
        
        # EMA
        ema: bool = False,
        ema_decay: float = 0.9999,

        # DDP optimizations
        ddp_find_unused_params: bool = False,
        ddp_gradient_as_bucket_view: bool = False,

        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,

        _batch_size: Optional[int] = None,  # global train batch (graphs/step); see batch_size_is_per_rank
        batch_size_: Optional[int] = None,  # global eval batch on train split
        _batch_size_: Optional[int] = None,  # global eval batch on train split (alias)
        repeat_train_batch: int = 1,
        schedule_step_multiplier: int = 1,
        batch_size_is_per_rank: bool = False,  # if False (default), _batch_size is global and divided by WORLD_SIZE
        use_distributed_sampler: bool = True,

        # collate function (on host)
        _collate_fn: Optional[Callable] = None,
        collate_fn_: Optional[Callable] = None,
        
        # preprocess function (on device)
        _preprocess_fn: Optional[Callable] = None,
        preprocess_fn_: Optional[Callable] = None,

        # optimizer
        make_optimizer: Optional[Callable] = None, # (model, lr, weight_decay, beta1, beta2, eps) -> optimizer
        lr: Optional[Union[float, List[float]]] = None,
        weight_decay: Optional[Union[float, List[float]]] = None,
        opt_beta1: Optional[Union[float, List[float]]] = None,
        opt_beta2: Optional[Union[float, List[float]]] = None,
        opt_eps: Optional[Union[float, List[float]]] = None,
        #
        clip_grad_norm: Optional[float] = None,
        grad_accumulation_steps: Optional[int] = None,

        Schedule: Optional[type] = None,
        drop_last_batch: bool = True,

        # OneCycleLR schedule
        one_cycle_pct_start: float = 0.3,        # % of cycle spent increasing LR. Default: 0.3
        one_cycle_div_factor: float = 25,        # initial_lr = max_lr/div_factor. Default: 25
        one_cycle_final_div_factor: float = 1e4, # min_lr = initial_lr/final_div_factor Default: 1e4
        one_cycle_three_phase: bool = False,     # first two phases will be symmetrical about pct_start third phase: initial_lr -> initial_lr/final_div_factor
        one_cycle_cycle_momentum: bool = True,
        one_cycle_base_momentum: float = 0.85,
        one_cycle_max_momentum: float = 0.95,
        one_cycle_anneal_strategy: str = 'cos',
        warmup_epochs: int = 0,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        plateau_factor: float = 0.7,
        plateau_patience: int = 10,

        lossfun: Optional[Callable] = None,
        batch_lossfun: Optional[Callable] = None, # (trainer, model, batch) -> loss
        epochs: Optional[int] = 0,
        steps: Optional[int] = 0,

        statsfun: Optional[Callable] = None, # (trainer, loader) -> (loss, stats)
        verbose: bool = True,
        print_iterator: bool = True,
        stats_every: Optional[int] = None, # stats every k epochs/ steps based on train_based_on_epochs
        stats_on_start: bool = True,
        fullbatch_stats_on_start: bool = False,
        _fullbatch_stats: bool = True,
        fullbatch_stats_: bool = True,
        log_rank_every_steps: int = 0,
        overlap_train_dataloader: bool = True,
        continuous_train_batches: Optional[bool] = None,
        run_timer: Optional["RunTimer"] = None,
    ):

        ###
        # DEVICE
        ###

        self.DISTRIBUTED = is_torchrun()
        self.GLOBAL_RANK = int(os.environ['RANK']) if self.DISTRIBUTED else 0
        self.LOCAL_RANK = int(os.environ['LOCAL_RANK']) if self.DISTRIBUTED else 0
        self.WORLD_SIZE = int(os.environ['WORLD_SIZE']) if self.DISTRIBUTED else 1

        if self.DISTRIBUTED:
            assert dist.is_initialized()
            self.DDP = dist.get_world_size() > 1
            self.device = torch.device(self.LOCAL_RANK)
        else:
            self.DDP = False
            self.device = select_device(device, verbose=True)

        self.is_cuda = self.device not in ['cpu', torch.device('cpu')]
        self.device_type = self.device.type if isinstance(self.device, torch.device) else self.device

        ###
        # PRINTING
        ###

        self.verbose = verbose
        self.print_iterator = print_iterator and self.verbose and (self.GLOBAL_RANK == 0)
        self.stats_on_start = bool(stats_on_start)
        self.fullbatch_stats_on_start = bool(fullbatch_stats_on_start)
        self.run_timer = run_timer
        self._first_train_step_marked = False
        self._statistics_timing_marked = False

        ###
        # PRECISION & ATTENTION BACKEND
        ###

        self.mixed_precision = mixed_precision
        self.amp_dtype = None
        if amp_dtype is not None:
            amp_dtype_value = str(amp_dtype).lower()
            if amp_dtype_value in ['fp16', 'float16', 'half']:
                self.amp_dtype = torch.float16
            elif amp_dtype_value in ['bf16', 'bfloat16']:
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Invalid amp_dtype: {amp_dtype}. Choose from: fp16, bf16, or None.")

        autocast_kwargs = dict(device_type=self.device_type, enabled=self.mixed_precision)
        if self.mixed_precision and self.amp_dtype is not None:
            autocast_kwargs['dtype'] = self.amp_dtype
        self.auto_cast = torch.autocast(**autocast_kwargs)

        self.effective_amp_dtype = self.amp_dtype
        if self.mixed_precision and self.effective_amp_dtype is None:
            try:
                self.effective_amp_dtype = torch.get_autocast_dtype(self.device_type)
            except Exception:
                self.effective_amp_dtype = None

        # GradScaler is only needed for fp16 CUDA AMP.
        use_grad_scaler = self.mixed_precision and (self.device_type == 'cuda') and (self.effective_amp_dtype == torch.float16)
        self.grad_scaler = torch.amp.GradScaler(device=self.device_type, enabled=use_grad_scaler)
        
        if self.mixed_precision:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"Mixed precision training enabled with autocast dtype={self.effective_amp_dtype}.")

        ###
        # DATA
        ###

        if _data is None:
            raise ValueError('_data passed to Trainer cannot be None.')

        self._data = _data
        self.data_ = data_

        self._batch_size = self.WORLD_SIZE if _batch_size is None else _batch_size  # global training batch size
        self._batch_size_ = self._batch_size * 2 if _batch_size_ is None else _batch_size_ # validation batch size on training data
        self.batch_size_ = self._batch_size * 2 if batch_size_ is None else batch_size_ # validation batch size on test data
        self.drop_last_batch = drop_last_batch
        self.repeat_train_batch = max(1, repeat_train_batch)
        self.schedule_step_multiplier = max(1, int(schedule_step_multiplier))
        self.batch_size_is_per_rank = bool(batch_size_is_per_rank)
        self.use_distributed_sampler = bool(use_distributed_sampler)

        if not self.batch_size_is_per_rank:
            assert self._batch_size % self.WORLD_SIZE == 0, f"Batch size {self._batch_size} must be divisible by world size {self.WORLD_SIZE}."

        self.num_workers = max(int(num_workers), 0)
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

        self._collate_fn = _collate_fn
        self.collate_fn_ = collate_fn_ if collate_fn_ is not None else _collate_fn
        
        self._preprocess_fn = _preprocess_fn
        self.preprocess_fn_ = preprocess_fn_ if preprocess_fn_ is not None else _preprocess_fn

        self.gnn_loader = bool(gnn_loader)
        if self.gnn_loader:
            backend = "pyg" if graph_loader_backend is None else str(graph_loader_backend).lower()
            if backend not in {"pyg", "dgl"}:
                raise ValueError(
                    f"Invalid graph_loader_backend='{graph_loader_backend}'. "
                    "Choose one of: pyg, dgl."
                )
            self.graph_loader_backend = backend
        else:
            self.graph_loader_backend = None

        ###
        # MODEL
        ###

        self.model = model.to(self.device)
        self._eager_model = self.model
        self._uses_compiled_model = False
        self.compile_stats_model = bool(compile_stats_model)

        if compile_model:

            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"Compiling model with {num_parameters(self.model)} parameters to device {self.device}")

            try:
                self.model = torch.compile(self.model)
                self._uses_compiled_model = True
                if self.verbose and (self.GLOBAL_RANK == 0):
                    print(f"Compilation successful.")
                self._timing_mark("trainer_compile")
            except Exception as e:
                if self.verbose and (self.GLOBAL_RANK == 0):
                    print(f"Compilation failed ({type(e).__name__}: {e}). Running without compile.")
        else:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print("Compilation disabled (compile_model=False).")
            self._timing_mark("trainer_compile_skipped")

        if self.DDP:
            ddp_kwargs = {
                'device_ids': [self.LOCAL_RANK],
                'static_graph': static_graph,
                'find_unused_parameters': ddp_find_unused_params,
                'gradient_as_bucket_view': ddp_gradient_as_bucket_view,
            }
            self.model = nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)

        ###
        # EMA (Exponential Moving Average)
        ###

        self.use_ema = ema
        self.ema_decay = ema_decay
        if self.use_ema:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"EMA tracking enabled with decay={self.ema_decay}")

            self.ema = EMA(self.model, decay=self.ema_decay)

        ###
        # OPTIMIZER
        ###

        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 0.0
        if make_optimizer is not None:
            if self.GLOBAL_RANK == 0:
                print(f"Using custom optimizer: {make_optimizer.__name__} with lr={lr}, weight_decay={weight_decay}, beta1={opt_beta1}, beta2={opt_beta2}, eps={opt_eps}")
            self.opt = make_optimizer(model=self.model, lr=lr, weight_decay=weight_decay, beta1=opt_beta1, beta2=opt_beta2, eps=opt_eps)
        else:
            opt_beta1 = 0.9 if opt_beta1 is None else opt_beta1
            opt_beta2 = 0.999 if opt_beta2 is None else opt_beta2
            opt_eps = 1e-8 if opt_eps is None else opt_eps
            self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(opt_beta1, opt_beta2), eps=opt_eps)

        self.clip_grad_norm = clip_grad_norm if clip_grad_norm is not None else torch.inf
        self.grad_accumulation_steps = max(1, grad_accumulation_steps) if grad_accumulation_steps is not None else 1

        ###
        # LOSS CALCULATION
        ###

        self.lossfun = nn.MSELoss() if lossfun is None else lossfun
        self.batch_lossfun = batch_lossfun

        ###
        # ITERATION
        ###

        if (epochs == 0) and (steps == 0):
            if self.GLOBAL_RANK == 0:
                print("No epochs or steps provided. Setting steps to 100.")
            steps = 100
        if (epochs != 0) and (steps != 0):
            raise ValueError(f"Both epochs ({epochs}) and steps ({steps}) provided. Please provide only one.")
        if steps != 0:
            # train based on steps
            self.steps = steps * self.repeat_train_batch
            self.epochs = 0
            self.train_based_on_epochs = False
        if epochs != 0:
            # train based on epochs
            self.epochs = epochs
            self.train_based_on_epochs = True
            if len(_data) == 0:
                raise ValueError("Training dataset is empty.")
            steps_per_epoch = len(_data) / self._batch_size
            if self.drop_last_batch:
                self.steps_per_epoch = math.floor(steps_per_epoch) * self.repeat_train_batch
            else:
                self.steps_per_epoch = math.ceil(steps_per_epoch) * self.repeat_train_batch
            self.steps = self.steps_per_epoch * self.epochs

        self.schedule_steps_per_epoch = self.steps_per_epoch * self.schedule_step_multiplier \
            if self.train_based_on_epochs else None
        self.schedule_total_steps = self.steps * self.schedule_step_multiplier

        self.step = 0
        self.epoch = 0
        self.reduce_lr_on_plateau = False

        ###
        # Learning rate scheduler
        # TODO: move scheduler to external function call like optimizer
        # e.g., self.schedule = make_scheduler(self.opt, **kwargs)
        ###

        if Schedule == "OneCycleLR":
            one_cycle_total_steps = self.schedule_total_steps if not self.train_based_on_epochs else (
                self.epochs * self.schedule_steps_per_epoch
            )
            one_cycle_pct_start = _safe_one_cycle_pct_start(one_cycle_pct_start, one_cycle_total_steps)
            one_cycle_args = dict(
                max_lr=lr,
                pct_start=one_cycle_pct_start,
                div_factor=one_cycle_div_factor,
                final_div_factor=one_cycle_final_div_factor,
                three_phase=one_cycle_three_phase,
                cycle_momentum=one_cycle_cycle_momentum,
                base_momentum=one_cycle_base_momentum,
                max_momentum=one_cycle_max_momentum,
                anneal_strategy=one_cycle_anneal_strategy,
            )
            if self.train_based_on_epochs:
                one_cycle_args['epochs'] = self.epochs
                one_cycle_args['steps_per_epoch'] = self.schedule_steps_per_epoch
            else:
                one_cycle_args['total_steps'] = self.schedule_total_steps

            self.schedule = optim.lr_scheduler.OneCycleLR(self.opt, **one_cycle_args)
            self.update_schedule_every_epoch = False
        elif Schedule == "CosineAnnealingLR":
            total_steps = max(1, int(self.schedule_total_steps))
            warmup_total_steps = int(warmup_steps or 0)
            if warmup_total_steps <= 0 and warmup_epochs and self.train_based_on_epochs:
                warmup_total_steps = int(max(0, warmup_epochs) * max(1, self.schedule_steps_per_epoch))
            warmup_total_steps = max(0, min(warmup_total_steps, total_steps - 1))

            if warmup_total_steps > 0:
                # Warm up from a small LR to base LR, then cosine decay to min_lr.
                warmup = optim.lr_scheduler.LinearLR(
                    self.opt,
                    start_factor=1.0 / float(warmup_total_steps),
                    end_factor=1.0,
                    total_iters=warmup_total_steps,
                )
                cosine = optim.lr_scheduler.CosineAnnealingLR(
                    self.opt,
                    T_max=max(1, total_steps - warmup_total_steps),
                    eta_min=float(min_lr or 0.0),
                )
                self.schedule = optim.lr_scheduler.SequentialLR(
                    self.opt,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_total_steps],
                )
            else:
                self.schedule = optim.lr_scheduler.CosineAnnealingLR(
                    self.opt,
                    T_max=total_steps,
                    eta_min=float(min_lr or 0.0),
                )
            self.update_schedule_every_epoch = False
        elif Schedule == "CosineAnnealingWarmRestarts":
            self.schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=self.epochs, T_mult=1, eta_min=0.)
            self.update_schedule_every_epoch = True
        elif Schedule == "ReduceLROnPlateau":
            self.schedule = optim.lr_scheduler.ReduceLROnPlateau(
                self.opt,
                factor=plateau_factor,
                patience=plateau_patience,
            )
            self.update_schedule_every_epoch = True
            self.reduce_lr_on_plateau = True
        elif Schedule is None:
            self.schedule = optim.lr_scheduler.ConstantLR(self.opt, factor=1.0, total_iters=1e10)
            self.update_schedule_every_epoch = True
        else:
            raise NotImplementedError()

        ###
        # STATISTICS
        ###

        self.is_training = False

        self.statsfun = statsfun

        # model accuracy statistics
        self.train_loss_per_batch = []
        self.num_steps_fullbatch  = []
        self.train_loss_fullbatch = []
        self.test_loss_fullbatch  = []
        self.train_stats_fullbatch = []
        self.test_stats_fullbatch  = []

        if self.use_ema:
            self.train_loss_fullbatch_ema = []
            self.test_loss_fullbatch_ema = []
            self.train_stats_fullbatch_ema = []
            self.test_stats_fullbatch_ema = []

        # training time/memory statistics
        self.train_stats_time = []
        self.test_stats_time = []
        self.time_per_epoch = []
        self.time_per_step = []
        self.time_dataload_per_step = []
        self.time_model_eval_per_step = []
        self.memory_utilization = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.max_memory_allocated = []
        self.max_memory_reserved = []

        self.grad_norm_per_step = []
        self.learning_rates_per_step = [[] for _ in range(len(self.opt.param_groups))]

        if self.train_based_on_epochs:
            self.stats_every = stats_every if stats_every else max(1, epochs // 10)
        else:
            self.stats_every = stats_every if stats_every else max(1, self.steps // 10)

        self._fullbatch_stats = _fullbatch_stats
        self.fullbatch_stats_ = fullbatch_stats_
        self.log_rank_every_steps = max(0, int(log_rank_every_steps or 0))
        self.overlap_train_dataloader = bool(overlap_train_dataloader)
        self.continuous_train_batches = continuous_train_batches
        self._use_continuous_train_batches = False
        self._batches_per_train_epoch = 1

        ###
        # Callbacks
        ###

        self.callbacks = collections.defaultdict(list)

        return

    #------------------------#
    # CALLBACKS
    #------------------------#

    # https://github.com/karpathy/minGPT/
    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def trigger_callbacks(self, event: str):
        for callback in self.callbacks[event]:
            callback(self)

    def _timing_mark(self, name: str) -> None:
        if self.run_timer is not None:
            self.run_timer.mark(name)

    def _eval_fullbatch(self, loader, *, enabled: bool, split: str) -> Tuple[Any, dict]:
        if not enabled or loader is None:
            return unset_metric(), {}
        loss, stats = self.call_statsfun(loader, split=split)
        return normalize_metric(loss), stats

    def _run_train_start(self) -> None:
        """Optional startup stats plus callback hooks (preserves step-0 batch_end behavior)."""
        self.trigger_callbacks("epoch_start")
        self.trigger_callbacks("batch_start")
        if self.stats_on_start:
            self.statistics()
        self.trigger_callbacks("batch_end")
        self.trigger_callbacks("epoch_end")

    #------------------------#
    # SAVE / LOAD
    #------------------------#

    def save(self, save_path: str): # call only if device==0
        if self.GLOBAL_RANK != 0:
            return

        snapshot = dict()

        # model
        if self.DDP:
            snapshot['model_state'] = self.model.module.state_dict()
        else:
            snapshot['model_state'] = self.model.state_dict()

        # iteration
        snapshot['step'] = self.step
        snapshot['epoch'] = self.epoch
        snapshot['opt_state'] = self.opt.state_dict()
        snapshot['schedule_state'] = None if (self.schedule is None) else self.schedule.state_dict()

        if self.use_ema:
            assert self.ema is not None, "EMA is not initialized"
            snapshot['ema_shadow'] = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}

        # model accuracy statistics
        snapshot['train_loss_per_batch'] = self.train_loss_per_batch
        snapshot['num_steps_fullbatch'] = self.num_steps_fullbatch
        snapshot['train_loss_fullbatch'] = self.train_loss_fullbatch
        snapshot['test_loss_fullbatch'] = self.test_loss_fullbatch
        snapshot['train_stats_fullbatch'] = self.train_stats_fullbatch
        snapshot['test_stats_fullbatch'] = self.test_stats_fullbatch

        if self.use_ema:
            snapshot['train_loss_fullbatch_ema'] = self.train_loss_fullbatch_ema
            snapshot['test_loss_fullbatch_ema'] = self.test_loss_fullbatch_ema
            snapshot['train_stats_fullbatch_ema'] = self.train_stats_fullbatch_ema
            snapshot['test_stats_fullbatch_ema'] = self.test_stats_fullbatch_ema

        # training time/memory statistics
        snapshot['train_stats_time'] = self.train_stats_time
        snapshot['test_stats_time'] = self.test_stats_time
        snapshot['time_per_epoch'] = self.time_per_epoch
        snapshot['time_per_step'] = self.time_per_step
        snapshot['time_dataload_per_step'] = self.time_dataload_per_step
        snapshot['time_model_eval_per_step'] = self.time_model_eval_per_step
        snapshot['memory_utilization'] = self.memory_utilization
        snapshot['memory_allocated'] = self.memory_allocated
        snapshot['memory_reserved'] = self.memory_reserved
        snapshot['max_memory_allocated'] = self.max_memory_allocated
        snapshot['max_memory_reserved'] = self.max_memory_reserved

        snapshot['grad_norm_per_step'] = self.grad_norm_per_step
        snapshot['learning_rates_per_step'] = self.learning_rates_per_step

        torch.save(snapshot, save_path)

        return

    def load_weights(self, load_path: str):
        '''
        load only model weights from file.
        used in __main__.py to load weights from file.
        '''

        if self.GLOBAL_RANK == 0:
            print(f"Loading weights from {load_path}")

        snapshot = torch.load(load_path, weights_only=False, map_location=self.device)

        # model
        if self.DDP:
            self.model.module.load_state_dict(snapshot['model_state'])
        else:
            self.model.load_state_dict(snapshot['model_state'])

        # ema
        if self.use_ema:
            assert self.ema is not None, "EMA is not initialized"
            assert snapshot.get('ema_shadow') is not None, "EMA shadow not found in snapshot"
            self.ema.shadow = {k: v.to(self.device) for k, v in snapshot['ema_shadow'].items()}

        del snapshot

        return

    def load(self, load_path: str):
        '''
        load full checkpoint (including stats) from file.
        used in callbacks to load latest checkpoint.
        '''

        if self.GLOBAL_RANK == 0:
            print(f"Loading checkpoint {load_path}")

        snapshot = torch.load(load_path, weights_only=False, map_location=self.device)

        # model
        if self.DDP:
            self.model.module.load_state_dict(snapshot['model_state'])
        else:
            self.model.load_state_dict(snapshot['model_state'])

        # ema
        if self.use_ema:
            assert self.ema is not None, "EMA is not initialized"
            assert snapshot.get('ema_shadow') is not None, "EMA shadow not found in snapshot"
            self.ema.shadow = {k: v.to(self.device) for k, v in snapshot['ema_shadow'].items()}

        # iteration
        self.step = snapshot['step']
        self.epoch = snapshot['epoch']
        self.opt.load_state_dict(snapshot['opt_state'])
        self.schedule.load_state_dict(snapshot['schedule_state'])

        # model accuracy statistics
        self.train_loss_per_batch = snapshot['train_loss_per_batch']
        self.num_steps_fullbatch = snapshot['num_steps_fullbatch']
        self.train_loss_fullbatch = snapshot['train_loss_fullbatch']
        self.test_loss_fullbatch = snapshot['test_loss_fullbatch']
        self.train_stats_fullbatch = snapshot['train_stats_fullbatch']
        self.test_stats_fullbatch = snapshot['test_stats_fullbatch']

        if self.use_ema:
            self.train_loss_fullbatch_ema = snapshot['train_loss_fullbatch_ema']
            self.test_loss_fullbatch_ema = snapshot['test_loss_fullbatch_ema']
            self.train_stats_fullbatch_ema = snapshot['train_stats_fullbatch_ema']
            self.test_stats_fullbatch_ema = snapshot['test_stats_fullbatch_ema']

        # training time/memory statistics
        self.train_stats_time = snapshot['train_stats_time']
        self.test_stats_time = snapshot['test_stats_time']
        self.time_per_epoch = snapshot['time_per_epoch']
        self.time_per_step = snapshot['time_per_step']
        self.time_dataload_per_step = snapshot['time_dataload_per_step']
        self.time_model_eval_per_step = snapshot['time_model_eval_per_step']
        self.memory_utilization = snapshot['memory_utilization']
        self.memory_allocated = snapshot.get('memory_allocated', [])
        self.memory_reserved = snapshot.get('memory_reserved', [])
        self.max_memory_allocated = snapshot.get('max_memory_allocated', self.memory_utilization)
        self.max_memory_reserved = snapshot.get('max_memory_reserved', [])

        self.grad_norm_per_step = snapshot['grad_norm_per_step']
        self.learning_rates_per_step = snapshot['learning_rates_per_step']

        del snapshot

        return

    #------------------------#
    # DATALOADER
    #------------------------#

    def make_dataloader(self):

        ###
        # Fix dataloader
        ###
        if self.gnn_loader:
            if self.graph_loader_backend == "pyg":
                import torch_geometric as pyg
                DL = pyg.loader.DataLoader
            elif self.graph_loader_backend == "dgl":
                from dgl.dataloading import GraphDataLoader
                DL = GraphDataLoader
            else:
                raise ValueError(f"Unsupported graph loader backend: {self.graph_loader_backend}")
        else:
            DL = torch.utils.data.DataLoader

        ###
        # Sampler
        ###
        if self.DDP and self.use_distributed_sampler:
            _sampler, _sampler_ = DistributedSampler(self._data), DistributedSampler(self._data, shuffle=False)
        else:
            _sampler, _sampler_ = None, None

        if self.data_ is not None:
            sampler_ = DistributedSampler(self.data_, shuffle=False) if (self.DDP and self.use_distributed_sampler) else None
        else:
            sampler_ = None

        ###
        # Batch size
        ###

        # Calculate per-rank batch sizes.
        # By default Trainer expects global batch sizes and divides by WORLD_SIZE.
        # Some pipelines (e.g., custom DALI loaders) provide per-rank sizes directly.
        if self.batch_size_is_per_rank:
            _batch_size  = self._batch_size
            _batch_size_ = self._batch_size_
            batch_size_  = self.batch_size_
        else:
            _batch_size  = self._batch_size // self.WORLD_SIZE
            _batch_size_ = self._batch_size_ // self.WORLD_SIZE
            batch_size_  = self.batch_size_ // self.WORLD_SIZE

        # Ensure minimum batch sizes for stability
        _batch_size = max(1, _batch_size)
        _batch_size_ = max(1, _batch_size_)
        batch_size_  = max(1, batch_size_)

        ###
        # Make dataloaders
        ###

        common_args = dict(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.is_cuda,
            persistent_workers=(self.num_workers > 0),
        )
        if self.num_workers > 0 and self.is_cuda:
            import multiprocessing as mp

            common_args["multiprocessing_context"] = mp.get_context("spawn")

        batch_sampler = None
        if not self.DDP and hasattr(self._data, "make_batch_sampler"):
            batch_sampler = self._data.make_batch_sampler(
                batch_size=_batch_size,
                drop_last=self.drop_last_batch,
                seed=self.epoch,
            )
        if batch_sampler is None:
            if _sampler is not None:
                train_sampler = _sampler
            elif self.DDP and not self.use_distributed_sampler:
                train_sampler = SequentialSampler(self._data)
            else:
                train_sampler = RandomSampler(self._data)
            batch_sampler = BatchSampler(train_sampler, _batch_size, drop_last=self.drop_last_batch)
        if self.repeat_train_batch > 1:
            batch_sampler = RepeatBatchSampler(batch_sampler, self.repeat_train_batch)

        self._batches_per_train_epoch = len(batch_sampler)
        self._use_continuous_train_batches = self._resolve_continuous_train_batches(self._batches_per_train_epoch)
        if self._use_continuous_train_batches:
            if self.train_based_on_epochs:
                raise ValueError("continuous_train_batches requires step-based training (epochs=0, steps>0).")
            epoch_sampler = self._get_epoch_sampler(batch_sampler)
            batch_sampler = StepBudgetBatchSampler(
                batch_sampler,
                total_batches=self.steps,
                sampler_for_epoch=epoch_sampler,
                get_epoch=lambda: self.epoch,
            )

        self._loader = DL(
            self._data,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            **common_args,
        )

        self._loader_ = self._make_eval_dataloader(
            DL,
            self._data,
            batch_size=_batch_size_,
            collate_fn=self.collate_fn_,
            distributed_sampler=_sampler_,
            common_args=common_args,
        )
        self.loader_ = (
            self._make_eval_dataloader(
                DL,
                self.data_,
                batch_size=batch_size_,
                collate_fn=self.collate_fn_,
                distributed_sampler=sampler_,
                common_args=common_args,
            )
            if self.data_ is not None
            else None
        )

        self._timing_mark("trainer_dataloader")

        return

    def _make_eval_dataloader(
        self,
        loader_cls,
        dataset,
        *,
        batch_size: int,
        collate_fn,
        distributed_sampler,
        common_args: dict,
    ):
        if dataset is None:
            return None
        batch_sampler = None
        if not self.DDP:
            resolve = getattr(dataset, "make_batch_sampler", None)
            if resolve is not None:
                batch_sampler = resolve(batch_size=batch_size, drop_last=False, seed=0)
        if batch_sampler is not None:
            return loader_cls(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **common_args,
            )
        return loader_cls(
            dataset,
            shuffle=False,
            sampler=distributed_sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **common_args,
        )

    def _get_epoch_sampler(self, batch_sampler):
        if isinstance(batch_sampler, RepeatBatchSampler):
            batch_sampler = batch_sampler.batch_sampler
        return getattr(batch_sampler, "sampler", None)

    def _resolve_continuous_train_batches(self, batches_per_epoch: int) -> bool:
        if self.continuous_train_batches is True:
            return True
        if self.continuous_train_batches is False:
            return False
        return (not self.train_based_on_epochs) and int(batches_per_epoch) == 1

    def _maybe_advance_continuous_epoch(self) -> bool:
        if not self._use_continuous_train_batches:
            return True
        self._batches_in_train_epoch += 1
        if self._batches_in_train_epoch < self._batches_per_train_epoch:
            return True
        self._batches_in_train_epoch = 0
        return self._advance_train_epoch()

    def _set_sampler_epoch(self, epoch: int) -> None:
        loader = getattr(self, "_loader", None)
        if loader is None:
            return

        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            return

        batch_sampler = getattr(loader, "batch_sampler", None)
        if isinstance(batch_sampler, RepeatBatchSampler):
            batch_sampler = batch_sampler.batch_sampler
        inner_sampler = getattr(batch_sampler, "sampler", None) if batch_sampler is not None else None
        if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
            inner_sampler.set_epoch(epoch)

    def _advance_train_epoch(self) -> bool:
        """Advance epoch bookkeeping after the train loader is exhausted."""
        self.time_per_epoch.append(time.time() - self._train_epoch_start_time)
        self._train_epoch_start_time = time.time()

        if self.train_based_on_epochs:
            if (self.epoch % self.stats_every) == 0:
                self.statistics()

        self.trigger_callbacks("epoch_end")

        if self.update_schedule_every_epoch:
            if self.reduce_lr_on_plateau:
                metric = self.test_loss_fullbatch[-1] if self.test_loss_fullbatch else self.train_loss_per_batch[-1]
                self.schedule.step(metric)
            else:
                self.schedule.step()

        self.epoch += 1

        if self.train_based_on_epochs and self.epoch > self.epochs:
            return False

        self.trigger_callbacks("epoch_start")
        self._set_sampler_epoch(self.epoch)
        return True

    def _fetch_train_batch(self, loader_iter):
        data_fetch_start = time.time()
        try:
            batch = next(loader_iter)
            data_fetch_end = time.time()
            return batch, loader_iter, data_fetch_end - data_fetch_start, False
        except StopIteration:
            if not self._advance_train_epoch():
                data_fetch_end = time.time()
                return None, loader_iter, data_fetch_end - data_fetch_start, True

            loader_iter = iter(self._loader)
            data_fetch_start = time.time()
            batch = next(loader_iter)
            data_fetch_end = time.time()
            return batch, loader_iter, data_fetch_end - data_fetch_start, False

    #------------------------#
    # TRAINING
    #------------------------#

    def train(self):

        from mlutils.train_batch_stream import OverlappedTrainBatchStream

        self._timing_mark("trainer_train_enter")

        self.is_training = True
        self.make_dataloader()

        self._run_train_start()

        # increment epoch and start training
        self.epoch += 1
        self._set_sampler_epoch(self.epoch)

        # make batch iterator
        self.make_batch_iterator()

        self._train_epoch_start_time = time.time()
        self._batches_in_train_epoch = 0
        batch_stream = OverlappedTrainBatchStream(self) if self.overlap_train_dataloader else None
        loader_iter = None if (batch_stream is not None or self._use_continuous_train_batches) else iter(self._loader)

        if batch_stream is not None:
            batch, dataload_wait = batch_stream.initial_batch()
        elif self._use_continuous_train_batches:
            loader_iter = iter(self._loader)
            data_fetch_start = time.time()
            batch = next(loader_iter)
            dataload_wait = time.time() - data_fetch_start
        else:
            batch, loader_iter, dataload_wait, should_stop = self._fetch_train_batch(loader_iter)
            if should_stop:
                self.is_training = False
                return

        self._timing_mark("trainer_first_batch_ready")

        try:
            while self.step < self.steps:
                self.step += 1
                self.time_dataload_per_step.append(dataload_wait)

                self.trigger_callbacks("batch_start")

                if batch_stream is not None and self.step < self.steps:
                    batch_stream.prefetch()

                loss = self.train_step(batch)
                self.update_batch_iterator(loss.item())

                if self.log_rank_every_steps > 0 and (self.step % self.log_rank_every_steps) == 0:
                    print(
                        f"[Rank {self.GLOBAL_RANK}] step={self.step} epoch={self.epoch} "
                        f"loss={loss.item():.6f}"
                    )

                if not self.train_based_on_epochs:
                    if (self.step % self.stats_every) == 0:
                        self.statistics()

                self.trigger_callbacks("batch_end")

                if self.step >= self.steps:
                    break

                if not self._maybe_advance_continuous_epoch():
                    break

                if batch_stream is not None:
                    batch, dataload_wait = batch_stream.get_prefetched()
                    if batch is None:
                        break
                elif self._use_continuous_train_batches:
                    data_fetch_start = time.time()
                    batch = next(loader_iter)
                    dataload_wait = time.time() - data_fetch_start
                else:
                    batch, loader_iter, dataload_wait, should_stop = self._fetch_train_batch(loader_iter)
                    if should_stop:
                        break
        finally:
            if batch_stream is not None:
                batch_stream.close()

        self.statistics()
        self.trigger_callbacks("epoch_end")

        self.is_training = False

        return

    def train_step(self, batch):

        if not self._first_train_step_marked:
            self._timing_mark("trainer_first_train_step")
            self._first_train_step_marked = True

        # reset peak memory stats (less frequently to reduce overhead)
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        # start time
        batch_start_time = time.time()

        self.model.train()

        batch = self.prepare_batch(batch, split='train')

        # forward/model eval timing (loss only)
        model_eval_start = time.time()

        # calculate loss
        with self.auto_cast:
            loss = self.batch_loss(batch, split='train', prepared=True)

        # measure model eval time
        model_eval_end = time.time()
        self.time_model_eval_per_step.append(model_eval_end - model_eval_start)

        # append loss to list
        self.train_loss_per_batch.append(loss.item())

        # backward pass with gradient scaling
        self.grad_scaler.scale(loss).backward() # replaces loss.backward()

        # trigger post grad callback
        self.trigger_callbacks("batch_post_grad")

        should_step = (self.step % self.grad_accumulation_steps) == 0
        grad_norm = float('nan')
        if should_step:
            # GradScaler only allows `unscale_` once per optimizer update.
            self.grad_scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm).item()

        # append grad norm and learning rate to list
        self.grad_norm_per_step.append(grad_norm)
        for (i, lr) in enumerate(self.schedule.get_last_lr()):
            self.learning_rates_per_step[i].append(lr)

        # # print warning if grad norm is too large
        # if grad_norm > 1e3:
        #     print(f"\n[WARNING] Exploding grad norm: {grad_norm:.2f}")
        #     # maybe trigger early stop or dump checkpoint
        #     # raise ValueError("Exploding grad norm")

        if should_step:
            # step optimizer with gradient scaling
            self.grad_scaler.step(self.opt) # replace self.opt.step()

            # update gradient scaler value
            self.grad_scaler.update()

            # zero out gradients
            self.opt.zero_grad()

            if self.use_ema:
                self.ema.update(self.model)

        # update schedule after every batch
        if not self.update_schedule_every_epoch:
            self.schedule.step()

        # update time per step
        self.time_per_step.append(time.time() - batch_start_time)

        # update memory utilization per step (less frequently to reduce overhead)
        if self.is_cuda:
            self.record_cuda_memory()

        return loss

    def make_batch_iterator(self):
        if self.print_iterator:
            if self.train_based_on_epochs:
                bar_format = '{desc}{n_fmt}/{total_fmt} {bar}[{rate_fmt}]'
            else:
                bar_format = '{desc} {bar}[{rate_fmt}]'
            self.batch_iterator = tqdm(
                total=self.steps, bar_format=bar_format, ncols=90, initial=self.step,
            )
        else:
            self.batch_iterator = None

        return

    def update_batch_iterator(self, loss: float):
        if self.print_iterator:
            if self.train_based_on_epochs:
                iter_msg = f"[Epoch {self.epoch} / {self.epochs}] "
            else:
                iter_msg = f"[Step {self.step} / {self.steps}] "
            grad_norm = self.grad_norm_per_step[-1] if self.grad_norm_per_step else float('nan')
            self.batch_iterator.set_description(
                iter_msg +
                f"LR {self.schedule.get_last_lr()[0]:.2e} " +
                f"LOSS {loss:.4e} " +
                f"GNORM: {grad_norm:.2e}"
            )
            self.batch_iterator.update(1)

    def move_to_device(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return [self.move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            kw = dict(non_blocking=True) if self.is_cuda else dict()
            return batch.to(self.device, **kw)
        elif batch is None:
            return None
        elif self.gnn_loader and (self.graph_loader_backend == "dgl") and hasattr(batch, "ndata"):
            # Prefer moving DGL graphs to trainer device; if unavailable (e.g., CPU-only DGL),
            # fall back to host graph handling.
            if hasattr(batch, "to"):
                try:
                    return batch.to(self.device)
                except Exception:
                    return batch
            return batch
        elif hasattr(batch, 'to'):
            # Support Graph/Data objects (e.g. PyG) that provide their own .to()
            return batch.to(self.device)
        else:
            return batch

    def _graph_targets(self, batch):
        if hasattr(batch, "y"):
            return batch.y
        if hasattr(batch, "ndata") and ("y" in batch.ndata):
            return batch.ndata["y"]
        raise ValueError("Graph batch does not contain target field 'y'.")

    def _graph_num_graphs(self, batch) -> int:
        if isinstance(batch, dict) and "num_graphs" in batch:
            return int(batch["num_graphs"])
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if hasattr(batch, "batch_size"):
            return int(batch.batch_size)
        if hasattr(batch, "batch_num_nodes"):
            counts = batch.batch_num_nodes()
            return int(len(counts))
        return 1

    def prepare_batch(self, batch, split: str):

        # move to device
        batch = self.move_to_device(batch)

        # apply preprocessor
        batch = self.apply_preprocessor(batch, split=split)

        return batch

    def batch_loss(self, batch, split: str, prepared: bool = False):

        if not prepared:
            batch = self.prepare_batch(batch, split=split)

        # calculate loss
        if self.batch_lossfun is not None:
            loss = self.batch_lossfun(self, self.model, batch)
        elif self.gnn_loader:
            yh = self.model(batch)
            y = self._graph_targets(batch)
            if torch.is_tensor(y) and (y.device != yh.device):
                y = y.to(yh.device)
            loss = self.lossfun(yh, y)
        else:
            # assume batch is a tuple of (x, y)
            x, y = batch
            yh = self.model(x)
            loss = self.lossfun(yh, y)

        return loss

    def record_cuda_memory(self) -> None:
        if not self.is_cuda:
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3

        self.memory_allocated.append(allocated)
        self.memory_reserved.append(reserved)
        self.max_memory_allocated.append(max_allocated)
        self.max_memory_reserved.append(max_reserved)
        # Legacy mirror for older checkpoints / callbacks (same as max_memory_allocated per step).
        self.memory_utilization.append(max_allocated)

    def _stats_model_context(self):
        trainer = self

        class _Context:
            def __enter__(self):
                self.model = trainer.model
                use_eager_stats = (
                    not trainer.compile_stats_model
                    and trainer._uses_compiled_model
                    and getattr(trainer, "_eager_model", None) is not None
                    and trainer.model is not trainer._eager_model
                )
                if use_eager_stats:
                    trainer.model = _DDPStatsModelProxy(trainer._eager_model) if trainer.DDP else trainer._eager_model
                return trainer.model

            def __exit__(self, exc_type, exc, tb):
                trainer.model = self.model
                return False

        return _Context()

    def cleanup_after_statistics(self):
        if not self.is_cuda:
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def apply_preprocessor(self, batch, split: str):
        if self._preprocess_fn is not None and split == 'train':
            return self._preprocess_fn(batch)
        if self.preprocess_fn_ is not None and split == 'val':
            return self.preprocess_fn_(batch)
        return batch

    #------------------------#
    # STATISTICS
    #------------------------#

    def get_batch_size(self, batch, loader):
        try:
            if self.gnn_loader:
                bs = self._graph_num_graphs(batch)
            elif isinstance(batch, tuple) or isinstance(batch, list):
                bs = len(batch[0])
            elif isinstance(batch, dict):
                bs = len(batch[list(batch.keys())[0]])
            else:
                bs = batch.size(0)
        except:
            bs = loader.batch_size
        return min(bs, loader.batch_size)

    @torch.no_grad()
    def call_statsfun(self, loader, split: str):
        with self._stats_model_context():
            self.model.eval()
            if self.statsfun is not None:
                return self.statsfun(self, loader, split=split)
            else:
                return self.fallback_statsfun(loader, split=split)

    def fallback_statsfun(self, loader, split: str):

        print_iterator = self.verbose and (self.GLOBAL_RANK == 0) and self.print_iterator

        if print_iterator:
            # Optimize tqdm for evaluation - disable smoothing for faster updates
            batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80,
                                smoothing=0.0, miniters=1)
        else:
            batch_iterator = loader

        N, L = 0, 0.0
        for batch in batch_iterator:
            n = self.get_batch_size(batch, loader)
            with self.auto_cast:
                l = self.batch_loss(batch, split=split).item()
            N += n
            L += l * n

        # Only synchronize once at the end, not for every batch
        if self.DDP:
            L_tensor = torch.tensor(L, device=self.device)
            N_tensor = torch.tensor(N, device=self.device)
            dist.all_reduce(L_tensor, dist.ReduceOp.SUM)
            dist.all_reduce(N_tensor, dist.ReduceOp.SUM)
            L, N = L_tensor.item(), N_tensor.item()

        if N == 0:
            loss = unset_metric()
        else:
            loss = normalize_metric(L / N)

        return loss, dict()
    
    def statistics(self):

        mark_statistics_timing = self.run_timer is not None and not self._statistics_timing_marked
        if mark_statistics_timing:
            self._timing_mark("trainer_statistics_start")

        # train stats
        train_stats_time_start = time.time()
        _loss, _stats = self._eval_fullbatch(self._loader_, enabled=self._fullbatch_stats, split='train')
        self.train_stats_time.append(time.time() - train_stats_time_start)

        # test stats
        test_stats_time_start = time.time()
        loss_, stats_ = self._eval_fullbatch(self.loader_, enabled=self.fullbatch_stats_, split='val')
        self.test_stats_time.append(time.time() - test_stats_time_start)

        _loss_ema = unset_metric()
        _stats_ema: dict = {}
        loss_ema_ = unset_metric()
        stats_ema_: dict = {}

        if self.use_ema:
            assert self.ema is not None, "EMA is not initialized"
            state_dict_bkp = copy_model_state(self.model)
            self.ema.load_ema_weights(self.model)
            _loss_ema, _stats_ema = self._eval_fullbatch(self._loader_, enabled=self._fullbatch_stats, split='train')
            loss_ema_, stats_ema_ = self._eval_fullbatch(self.loader_, enabled=self.fullbatch_stats_, split='val')
            load_model_state(self.model, state_dict_bkp)

        self.cleanup_after_statistics()

        if mark_statistics_timing:
            self._timing_mark("trainer_statistics_done")
            self._statistics_timing_marked = True

        # printing
        if self.verbose and (self.GLOBAL_RANK == 0):
            msg = f"\n"
            if self.train_based_on_epochs:
                msg += f"[Epoch {self.epoch} / {self.epochs}] "
            else:
                msg += f"[Step {self.step} / {self.steps}] "

            msg += f"TRAIN LOSS: {format_metric(_loss)} | TEST LOSS: {format_metric(loss_)}"

            if self.use_ema:
                msg += f" | TRAIN LOSS (EMA): {format_metric(_loss_ema)} | TEST LOSS (EMA): {format_metric(loss_ema_)}"

            msg += f"\nTRAIN STATS TIME: {self.train_stats_time[-1]:.4e}s | TEST STATS TIME: {self.test_stats_time[-1]:.4e}s"
            print(msg)

        self.train_loss_fullbatch.append(_loss)
        self.test_loss_fullbatch.append(loss_)
        self.num_steps_fullbatch.append(len(self.train_loss_per_batch))
        self.train_stats_fullbatch.append(_stats)
        self.test_stats_fullbatch.append(stats_)

        if self.use_ema:
            self.train_loss_fullbatch_ema.append(_loss_ema)
            self.test_loss_fullbatch_ema.append(loss_ema_)
            self.train_stats_fullbatch_ema.append(_stats_ema)
            self.test_stats_fullbatch_ema.append(stats_ema_)

        return
#======================================================================#
#
