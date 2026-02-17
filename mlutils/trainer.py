#
import math
import time
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
)
from mlutils.ema import *

__all__ = [
    'Trainer',
]

#======================================================================#
class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        _data: Any,  # Must be iterable (Dataset, PyG Dataset, etc.)
        data_: Optional[Any] = None,  # Must be iterable (Dataset, PyG Dataset, etc.)

        gnn_loader: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: bool = False,

        # compilation
        compile_model: bool = True,
        static_graph: bool = False,
        
        # EMA
        ema: bool = False,
        ema_decay: float = 0.9999,

        # DDP optimizations
        ddp_find_unused_params: bool = False,
        ddp_gradient_as_bucket_view: bool = False,

        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,

        _batch_size: Optional[int] = None,  # bwd over _data
        batch_size_: Optional[int] = None,  # fwd over data_
        _batch_size_: Optional[int] = None, # fwd over _data
        repeat_train_batch: int = 1,

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

        lossfun: Optional[Callable] = None,
        batch_lossfun: Optional[Callable] = None, # (trainer, model, batch) -> loss
        epochs: Optional[int] = 0,
        steps: Optional[int] = 0,

        statsfun: Optional[Callable] = None, # (trainer, loader) -> (loss, stats)
        verbose: bool = True,
        print_iterator: bool = True,
        stats_every: Optional[int] = None, # stats every k epochs/ steps based on train_based_on_epochs
        _fullbatch_stats: bool = True,
        fullbatch_stats_: bool = True,
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

        ###
        # PRECISION & ATTENTION BACKEND
        ###

        self.mixed_precision = mixed_precision
        self.auto_cast = torch.autocast(device_type=self.device_type, enabled=self.mixed_precision)
        self.grad_scaler = torch.amp.GradScaler(device=self.device_type, enabled=self.mixed_precision)
        
        if self.mixed_precision:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"Mixed precision training enabled.")

        ###
        # DATA
        ###

        if _data is None:
            raise ValueError('_data passed to Trainer cannot be None.')

        self._data = _data
        self.data_ = data_

        self._batch_size = self.WORLD_SIZE if _batch_size is None else _batch_size    # training batch size
        self._batch_size_ = self._batch_size * 2 if _batch_size_ is None else _batch_size_ # validation batch size on training data
        self.batch_size_ = self._batch_size * 2 if batch_size_ is None else batch_size_ # validation batch size on test data
        self.drop_last_batch = drop_last_batch
        self.repeat_train_batch = max(1, repeat_train_batch)

        assert self._batch_size % self.WORLD_SIZE == 0, f"Batch size {self._batch_size} must be divisible by world size {self.WORLD_SIZE}."

        self.num_workers = min(num_workers, self._batch_size, os.cpu_count() // self.WORLD_SIZE)
        self.num_workers = max(self.num_workers, 0)
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

        self._collate_fn = _collate_fn
        self.collate_fn_ = collate_fn_ if collate_fn_ is not None else _collate_fn
        
        self._preprocess_fn = _preprocess_fn
        self.preprocess_fn_ = preprocess_fn_ if preprocess_fn_ is not None else _preprocess_fn

        self.gnn_loader = gnn_loader

        ###
        # MODEL
        ###

        self.model = model.to(self.device)

        if compile_model:

            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"Compiling model with {num_parameters(self.model)} parameters to device {self.device}")

            try:
                self.model = torch.compile(self.model)
                if self.verbose and (self.GLOBAL_RANK == 0):
                    print(f"Compilation successful.")
            except Exception as e:
                if self.verbose and (self.GLOBAL_RANK == 0):
                    print(f"Compilation failed ({type(e).__name__}: {e}). Running without compile.")
        else:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print("Compilation disabled (compile_model=False).")

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

        self.step = 0
        self.epoch = 0

        ###
        # Learning rate scheduler
        # TODO: move scheduler to external function call like optimizer
        # e.g., self.schedule = make_scheduler(self.opt, **kwargs)
        ###

        if Schedule == "OneCycleLR":
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
                one_cycle_args['steps_per_epoch'] = self.steps_per_epoch
            else:
                one_cycle_args['total_steps'] = self.steps

            self.schedule = optim.lr_scheduler.OneCycleLR(self.opt, **one_cycle_args)
            self.update_schedule_every_epoch = False
        elif Schedule == "CosineAnnealingWarmRestarts":
            self.schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=self.epochs, T_mult=1, eta_min=0.)
            self.update_schedule_every_epoch = True
        elif Schedule == "CosineAnnealingLR":
            self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=0.)
            self.update_schedule_every_epoch = True
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

        self.grad_norm_per_step = []
        self.learning_rates_per_step = [[] for _ in range(len(self.opt.param_groups))]

        if self.train_based_on_epochs:
            self.stats_every = stats_every if stats_every else max(1, epochs // 10)
        else:
            self.stats_every = stats_every if stats_every else max(1, self.steps // 10)

        self._fullbatch_stats = _fullbatch_stats
        self.fullbatch_stats_ = fullbatch_stats_

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
            import torch_geometric as pyg
            DL = pyg.loader.DataLoader
        else:
            DL = torch.utils.data.DataLoader

        ###
        # Sampler
        ###
        if self.DDP:
            _sampler, _sampler_ = DistributedSampler(self._data), DistributedSampler(self._data, shuffle=False)
        else:
            _sampler, _sampler_ = None, None

        if self.data_ is not None:
            sampler_ = DistributedSampler(self.data_, shuffle=False) if self.DDP else None
        else:
            sampler_ = None

        ###
        # Batch size
        ###

        # Calculate per-rank batch sizes
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

        train_sampler = _sampler if _sampler is not None else RandomSampler(self._data)
        batch_sampler = BatchSampler(train_sampler, _batch_size, drop_last=self.drop_last_batch)
        if self.repeat_train_batch > 1:
            batch_sampler = RepeatBatchSampler(batch_sampler, self.repeat_train_batch)

        self._loader = DL(
            self._data,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            **common_args,
        )

        _args_ = dict(shuffle=False, sampler=_sampler_, batch_size=_batch_size_, collate_fn=self.collate_fn_, **common_args)
        args_  = dict(shuffle=False, sampler=sampler_ , batch_size=batch_size_ , collate_fn=self.collate_fn_, **common_args)

        self._loader_ = DL(self._data, **_args_)
        self.loader_  = DL(self.data_, **args_) if self.data_ is not None else None

        return

    #------------------------#
    # TRAINING
    #------------------------#

    def train(self):

        self.is_training = True
        self.make_dataloader()

        self.trigger_callbacks("epoch_start")
        self.trigger_callbacks("batch_start")
        self.statistics()
        self.trigger_callbacks("batch_end")
        self.trigger_callbacks("epoch_end")

        # increment epoch and start training
        self.epoch += 1
        loader_iter = iter(self._loader)
        if self.DDP and hasattr(self._loader, 'sampler') and hasattr(self._loader.sampler, 'set_epoch'):
            self._loader.sampler.set_epoch(self.epoch)

        # make batch iterator
        self.make_batch_iterator()

        epoch_start_time = time.time()

        while self.step < self.steps:
            self.step += 1

            # load next batch
            # measure data loading time (time to fetch the next batch)
            try:
                data_fetch_start = time.time()
                batch = next(loader_iter)
                data_fetch_end = time.time()
            except StopIteration:
                # increment epoch and loop back through the dataset

                # update time_per_epoch
                self.time_per_epoch.append(time.time() - epoch_start_time)
                epoch_start_time = time.time()

                # calculate statistics if training based on epochs
                if self.train_based_on_epochs:
                    if (self.epoch % self.stats_every) == 0:
                        self.statistics()

                # trigger epoch end callback
                self.trigger_callbacks("epoch_end")

                # update schedule
                if self.update_schedule_every_epoch:
                    self.schedule.step()

                # increment epoch
                self.epoch += 1

                # commence next epoch
                if self.train_based_on_epochs:
                    if self.epoch > self.epochs:
                        break

                # trigger epoch start callback
                self.trigger_callbacks("epoch_start")

                loader_iter = iter(self._loader)
                if self.DDP and hasattr(self._loader, 'sampler') and hasattr(self._loader.sampler, 'set_epoch'):
                    self._loader.sampler.set_epoch(self.epoch)

                data_fetch_start = time.time()
                batch = next(loader_iter)
                data_fetch_end = time.time()

            self.time_dataload_per_step.append(data_fetch_end - data_fetch_start)

            # trigger batch start callback
            self.trigger_callbacks("batch_start")

            # training step
            loss = self.train_step(batch)

            # print batch iterator
            self.update_batch_iterator(loss.item())

            # calculate statistics if training based on steps
            if not self.train_based_on_epochs:
                if (self.step % self.stats_every) == 0:
                    self.statistics()

            # trigger batch end callback
            self.trigger_callbacks("batch_end")

        # calculate final statistics
        self.statistics()
        self.trigger_callbacks("epoch_end")

        self.is_training = False

        return

    def train_step(self, batch):

        # reset peak memory stats (less frequently to reduce overhead)
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        # start time
        batch_start_time = time.time()

        self.model.train()

        # forward/model eval timing (loss only)
        model_eval_start = time.time()

        # calculate loss
        with self.auto_cast:
            loss = self.batch_loss(batch, split='train')

        # measure model eval time
        model_eval_end = time.time()
        self.time_model_eval_per_step.append(model_eval_end - model_eval_start)

        # append loss to list
        self.train_loss_per_batch.append(loss.item())

        # backward pass with gradient scaling
        self.grad_scaler.scale(loss).backward() # replaces loss.backward()

        # trigger post grad callback
        self.trigger_callbacks("batch_post_grad")

        # unscale gradients
        self.grad_scaler.unscale_(self.opt)

        # clip gradients
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

        if (self.step % self.grad_accumulation_steps) == 0:
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
            self.memory_utilization.append(torch.cuda.max_memory_allocated() / 1024**3)

        return loss

    def make_batch_iterator(self):
        if self.print_iterator:
            if self.train_based_on_epochs:
                bar_format = '{desc}{n_fmt}/{total_fmt} {bar}[{rate_fmt}]'
            else:
                bar_format = '{desc} {bar}[{rate_fmt}]'
            self.batch_iterator = tqdm(
                total=self.steps, bar_format=bar_format, ncols=80, initial=self.step,
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
            self.batch_iterator.set_description(
                iter_msg +
                f"LR {self.schedule.get_last_lr()[0]:.2e} " +
                f"LOSS {loss:.8e}"
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
        else:
            return batch

    def batch_loss(self, batch, split: str):

        # move to device
        batch = self.move_to_device(batch)

        # apply preprocessor
        batch = self.apply_preprocessor(batch, split=split)

        # calculate loss
        if self.batch_lossfun is not None:
            loss = self.batch_lossfun(self, self.model, batch)
        elif self.gnn_loader:
            batch = batch.to(self.device)
            yh = self.model(batch)
            loss = self.lossfun(yh, batch.y)
        else:
            # assume batch is a tuple of (x, y)
            x, y = batch
            yh = self.model(x)
            loss = self.lossfun(yh, y)

        return loss

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
                bs = batch.num_graphs
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
            loss = float('nan')
        else:
            loss = L / N

        return loss, dict()
    
    def statistics(self):

        # train stats
        train_stats_time_start = time.time()
        _loss, _stats = self.call_statsfun(self._loader_, split='train') if self._fullbatch_stats else (float('nan'), dict())
        self.train_stats_time.append(time.time() - train_stats_time_start)

        # test stats
        test_stats_time_start = time.time()
        loss_, stats_ = self.call_statsfun(self.loader_, split='val') if (self.fullbatch_stats_ and self.loader_ is not None) else (float('nan'), dict())
        self.test_stats_time.append(time.time() - test_stats_time_start)

        if self.use_ema:
            assert self.ema is not None, "EMA is not initialized"
            # save model state
            state_dict_bkp = copy_model_state(self.model)
            # load ema weights
            self.ema.load_ema_weights(self.model)
            # calculate stats
            _loss_ema, _stats_ema = self.call_statsfun(self._loader_, split='train') if self._fullbatch_stats else (float('nan'), dict())
            loss_ema_, stats_ema_ = self.call_statsfun(self.loader_, split='val') if (self.fullbatch_stats_ and self.loader_ is not None) else (float('nan'), dict())
            # restore model state
            load_model_state(self.model, state_dict_bkp)

        # printing
        if self.verbose and (self.GLOBAL_RANK == 0):
            if self.train_based_on_epochs:
                msg = f"[Epoch {self.epoch} / {self.epochs}] "
            else:
                msg = f"[Step {self.step} / {self.steps}] "

            msg += f"TRAIN LOSS: {_loss:.6e} | TEST LOSS: {loss_:.6e}"

            if self.use_ema:
                msg += f" | TRAIN LOSS (EMA): {_loss_ema:.6e} | TEST LOSS (EMA): {loss_ema_:.6e}"

            msg += f"\nTRAIN STATS TIME: {self.train_stats_time[-1]:.4e}s | TEST STATS TIME: {self.test_stats_time[-1]:.4e}s"
            print(msg)

        if self.is_training:
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