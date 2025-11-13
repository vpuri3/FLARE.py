#
import time
import torch
from torch import nn, optim
from torch import distributed as dist

from tqdm import tqdm

# builtin
import os
import collections

# local
from mlutils.schedule import DecayScheduler
from mlutils.utils import (
    num_parameters, select_device, is_torchrun, check_package_version_lteq,
)

__all__ = [
    'Trainer',
]

#======================================================================#

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_=None,

        gnn_loader=False,
        device=None,
        mixed_precision=False,
        attn_backend=None, # ['math', 'flash', 'efficient', 'cudnn']

        collate_fn=None,
        num_workers=0,
        prefetch_factor=2,
        _batch_size=None,  # bwd over _data
        batch_size_=None,  # fwd over data_
        _batch_size_=None, # fwd over _data

        lr=None,
        weight_decay=None,
        clip_grad_norm=None,

        make_optimizer=None, # (model, lr, weight_decay, betas, eps) -> optimizer
        opt_betas=None,
        opt_eps=None,

        Schedule=None,
        drop_last_batch=True,

        # OneCycleLR schedule
        one_cycle_pct_start=0.3,        # % of cycle spent increasing LR. Default: 0.3
        one_cycle_div_factor=25,        # initial_lr = max_lr/div_factor. Default: 25
        one_cycle_final_div_factor=1e4, # min_lr = initial_lr/final_div_factor Default: 1e4
        one_cycle_three_phase=False,    # first two phases will be symmetrical about pct_start third phase: initial_lr -> initial_lr/final_div_factor

        # noise_schedule='linear',
        # noise_init=0.1,
        # noise_min=0.0,

        lossfun=None,
        batch_lossfun=None, # (trainer, model, batch) -> loss
        epochs=None,

        statsfun=None, # (trainer, loader) -> (loss, stats)
        verbose=True,
        print_config=False,
        print_batch=True,
        print_epoch=True,
        stats_every=1, # stats every k epochs
    ):

        ###
        # PRINTING
        ###

        self.verbose = verbose
        self.print_config = print_config
        self.print_batch = print_batch
        self.print_epoch = print_epoch
        self.stats_every = stats_every if stats_every != 0 else 1

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
        # PRECISION & ATTENTION BACKEND
        ###

        self.mixed_precision = mixed_precision
        self.auto_cast = torch.autocast(device_type=self.device_type, enabled=self.mixed_precision)
        self.grad_scaler = torch.amp.GradScaler(device=self.device_type, enabled=self.mixed_precision)

        if attn_backend is None:
            self.attn_backend = None
        elif attn_backend == 'math':
            self.attn_backend = torch.nn.attention.SDPBackend.MATH
        elif attn_backend == 'flash':
            self.attn_backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION
        elif attn_backend == 'efficient':
            self.attn_backend = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
        elif attn_backend == 'cudnn':
            self.attn_backend = torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        else:
            raise ValueError(f"Invalid attention backend: {attn_backend}. Must be one of: 'math', 'flash', 'efficient', 'cudnn'.")

        if self.GLOBAL_RANK == 0 and self.attn_backend is not None:
            print(f"Using SDPA backend: {self.attn_backend}")

        ###
        # DATA
        ###

        if _data is None:
            raise ValueError('_data passed to Trainer cannot be None.')

        self._data = _data
        self.data_ = data_

        self._batch_size = 1 if _batch_size is None else _batch_size
        self._batch_size_ = len(_data) if _batch_size_ is None else _batch_size_
        self.batch_size_ = batch_size_
        self.drop_last_batch = drop_last_batch

        assert self._batch_size % self.WORLD_SIZE == 0, f"Batch size {self._batch_size} must be divisible by world size {self.WORLD_SIZE}."

        if (data_ is not None) and (batch_size_ is None):
            self.batch_size_ = len(data_)

        self.num_workers = num_workers if num_workers != 0 else os.cpu_count() // self.WORLD_SIZE
        self.num_workers = min(self.num_workers, max(self._batch_size, self.batch_size_, self._batch_size_))

        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.gnn_loader = gnn_loader

        ###
        # MODEL
        ###

        self.model = model.to(self.device)

        if self.verbose and (self.GLOBAL_RANK == 0):
            print(f"Compiling model with {num_parameters(self.model)} parameters to device {self.device}")

        try:
            self.model = torch.compile(self.model)
        except:
            if self.verbose and (self.GLOBAL_RANK == 0):
                print(f"Compilation failed. Running along anyways.")
            pass

        if self.DDP:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[device])

        ###
        # OPTIMIZER
        ###

        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 0.0
        if make_optimizer is not None:
            if self.GLOBAL_RANK == 0:
                print(f"Using custom optimizer: {make_optimizer.__name__} with lr={lr}, weight_decay={weight_decay}, betas={opt_betas}, eps={opt_eps}")
            self.opt = make_optimizer(model=self.model, lr=lr, weight_decay=weight_decay, betas=opt_betas, eps=opt_eps)
        else:
            opt_betas = (0.9, 0.999) if opt_betas is None else opt_betas
            opt_eps = 1e-8 if opt_eps is None else opt_eps
            self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=opt_betas, eps=opt_eps)

        self.clip_grad_norm = clip_grad_norm if clip_grad_norm is not None else torch.inf

        ###
        # LOSS CALCULATION
        ###

        self.lossfun = nn.MSELoss() if lossfun is None else lossfun
        self.batch_lossfun = batch_lossfun

        ###
        # iteration
        ###

        self.epoch = 0
        self.epochs = 100 if epochs is None else epochs
        self.steps_per_epoch = len(_data) // self._batch_size + 1
        self.total_steps = self.steps_per_epoch * self.epochs

        if Schedule == "OneCycleLR":
            self.schedule = optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=one_cycle_pct_start,
                div_factor=one_cycle_div_factor,
                final_div_factor=one_cycle_final_div_factor,
                three_phase=one_cycle_three_phase,
            )
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
        
        # self.noise_schedule = DecayScheduler(
        #     total_steps=self.total_steps,
        #     decay_type=noise_schedule,
        #     init_val=noise_init,
        #     min_val=noise_min,
        # )
        
        ###
        # STATISTICS
        ###
        
        self.is_training = False

        self.statsfun = statsfun
        self.stat_vals = {
            "train_loss" : None,
            "train_stats" : None,
            "test_loss" : None,
            "test_stats" : None,
        }
        self.train_loss_per_batch = []
        self.num_steps_fullbatch  = []
        self.train_loss_fullbatch = []
        self.test_loss_fullbatch  = []
        
        self.time_per_epoch = []
        self.time_per_step = []
        self.memory_utilization = []
        self.grad_norm_per_step = []
        self.learning_rate_per_step = []

        ###
        # Callbacks
        ###

        self.callbacks = collections.defaultdict(list)

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
        snapshot['epoch'] = self.epoch
        if self.DDP:
            snapshot['model_state'] = self.model.module.state_dict()
        else:
            snapshot['model_state'] = self.model.state_dict()
        snapshot['opt_state'] = self.opt.state_dict()
        snapshot['schedule_state'] = None if (self.schedule is None) else self.schedule.state_dict()
        snapshot['train_loss_per_batch'] = self.train_loss_per_batch

        torch.save(snapshot, save_path)

        return

    def load(self, load_path: str):
        if self.GLOBAL_RANK == 0:
            print(f"Loading checkpoint {load_path}")

        if check_package_version_lteq('torch', '2.4'):
            snapshot = torch.load(load_path, map_location=self.device)
        else:
            snapshot = torch.load(load_path, weights_only=False, map_location=self.device)

        self.epoch = snapshot['epoch']
        if self.DDP:
            self.model.module.load_state_dict(snapshot['model_state'])
        else:
            self.model.load_state_dict(snapshot['model_state'])
        self.opt.load_state_dict(snapshot['opt_state'])
        self.schedule.load_state_dict(snapshot['schedule_state'])
        self.train_loss_per_batch = snapshot['train_loss_per_batch']
        
        # # noise schedul
        # self.noise_schedule.set_current_step(self.epoch * self.steps_per_epoch)

    #------------------------#
    # DATALOADER
    #------------------------#

    def make_dataloader(self):
        if self.gnn_loader:
            import torch_geometric as pyg
            DL = pyg.loader.DataLoader
        else:
            DL = torch.utils.data.DataLoader

        if self.DDP:
            DS = torch.utils.data.distributed.DistributedSampler
            _shuffle, _shuffle_ = False, False
            _sampler, _sampler_ = DS(self._data), DS(self._data, shuffle=False)

            if self.data_ is not None:
                shuffle_ = False
                sampler_ = DS(self.data_, shuffle=False)
            else:
                shuffle_ = False
                sampler_ = None
        else:
            _shuffle, _shuffle_, shuffle_ = True, False, False
            _sampler, _sampler_, sampler_ = None, None , None
            
        dl_args = dict(
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

        _args  = dict(shuffle=_shuffle , sampler=_sampler , **dl_args, drop_last=self.drop_last_batch)
        _args_ = dict(shuffle=_shuffle_, sampler=_sampler_, **dl_args)
        args_  = dict(shuffle=shuffle_ , sampler=sampler_ , **dl_args)

        _batch_size  = self._batch_size // self.WORLD_SIZE if self.DISTRIBUTED else self._batch_size
        _batch_size_ = self._batch_size_
        batch_size_  = self.batch_size_

        self._loader  = DL(self._data, batch_size=_batch_size , **_args , persistent_workers=True)
        self._loader_ = DL(self._data, batch_size=_batch_size_, **_args_)
        
        if self.data_ is not None:
            self.loader_ = DL(self.data_, batch_size=batch_size_, **args_)
        else:
            self.loader_ = None

        ###
        # Printing
        ###

        if self.verbose and self.print_config and self.GLOBAL_RANK == 0:
            print(f"Number of training samples: {len(self._data)}")
            if self.data_ is not None:
                print(f"Number of test samples: {len(self.data_)}")
            else:
                print(f"No test data provided")

            if self.gnn_loader:
                for batch in self._loader:
                    print(batch)
                    break
                if self.data_ is not None:
                    for batch in self.loader_:
                        print(batch)
                        break
            else:
                for (x, y) in self._loader:
                    print(f"Shape of x: {x.shape} {x.dtype}")
                    print(f"Shape of u: {y.shape} {y.dtype}")
                    break
        return

    #------------------------#
    # TRAINING
    #------------------------#

    def train(self):
        
        self.is_training = True
        self.make_dataloader()

        self.trigger_callbacks("epoch_start")
        self.statistics()
        self.trigger_callbacks("epoch_end")

        while self.epoch < self.epochs:
            self.epoch += 1

            self.trigger_callbacks("epoch_start")
            self.train_epoch()
            if (self.epoch % self.stats_every) == 0:
                self.statistics()
            self.trigger_callbacks("epoch_end")
            if self.update_schedule_every_epoch:
                self.schedule.step()

        self.is_training = False

        return

    def train_epoch(self):
        if self.DDP:
            self._loader.sampler.set_epoch(self.epoch)

        print_batch = self.verbose and (self.GLOBAL_RANK == 0) and self.print_batch # and (len(self._loader) > 1)

        if print_batch:
            batch_iterator = tqdm(
                self._loader,
                bar_format='{desc}{n_fmt}/{total_fmt} {bar}[{rate_fmt}]',
                ncols=80,
            )
        else:
            batch_iterator = self._loader
            
        # start time
        epoch_start_time = time.time()

        for batch in batch_iterator:

            # reset peak memory stats
            if self.is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # start time
            batch_start_time = time.time()

            # zero out gradients
            self.opt.zero_grad()

            # trigger batch start callback
            self.trigger_callbacks("batch_start")

            # self.noise_schedule.step()

            self.model.train()

            with self.auto_cast:
                if self.attn_backend is not None:
                    with torch.nn.attention.sdpa_kernel(self.attn_backend):
                        loss = self.batch_loss(batch)
                else:
                    loss = self.batch_loss(batch)

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
            self.learning_rate_per_step.append(self.schedule.get_last_lr()[0])
            
            # print warning if grad norm is too large
            if grad_norm > 1e3:
                print(f"[WARNING] Exploding grad norm: {grad_norm:.2f}")
                # maybe trigger early stop or dump checkpoint
                # raise ValueError("Exploding grad norm")

            # step optimizer with gradient scaling
            self.grad_scaler.step(self.opt) # replace self.opt.step()

            # update gradient scaler value
            self.grad_scaler.update()

            # update schedule after every batch
            if not self.update_schedule_every_epoch:
                self.schedule.step()

            # update time per step
            self.time_per_step.append(time.time() - batch_start_time)

            # update memory utilization per step
            if self.is_cuda:
                torch.cuda.synchronize()
                self.memory_utilization.append(torch.cuda.max_memory_allocated() / 1024**3)

            # trigger batch end callback
            self.trigger_callbacks("batch_end")

            # print batch info
            if print_batch:
                batch_iterator.set_description(
                    f"[Epoch {self.epoch} / {self.epochs}] " +
                    f"LR {self.schedule.get_last_lr()[0]:.2e} " +
                    f"LOSS {loss.item():.8e}"
                )

        self.time_per_epoch.append(time.time() - epoch_start_time)

        return
    
    def move_to_device(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            batch = {k: self.move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        else:
            batch = batch.to(self.device)
        return batch

    def batch_loss(self, batch):
        batch = self.move_to_device(batch)

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
    def evaluate(self, loader):
        self.model.eval()
        
        if self.statsfun is not None:
            return self.statsfun(self, loader)

        print_batch = self.verbose and (self.GLOBAL_RANK == 0) and self.print_batch # and (len(self._loader) > 1)
        if print_batch:
            batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80)
        else:
            batch_iterator = loader

        N, L = 0, 0.0
        for batch in batch_iterator:
            n = self.get_batch_size(batch, loader)
            with self.auto_cast:
                l = self.batch_loss(batch).item()
            N += n
            L += l * n

        if self.DDP:
            L = torch.tensor(L, device=self.device)
            N = torch.tensor(N, device=self.device)
            dist.all_reduce(L, dist.ReduceOp.SUM)
            dist.all_reduce(N, dist.ReduceOp.SUM)
            L, N = L.item(), N.item()

        if N == 0:
            loss = float('nan')
        else:
            loss = L / N

        return loss, None

    def statistics(self):

        _loss, _stats = self.evaluate(self._loader_)

        if self.loader_ is not None:
            loss_, stats_ = self.evaluate(self.loader_)
        else:
            loss_, stats_ = _loss, _stats

        # printing
        if self.print_epoch and self.verbose and (self.GLOBAL_RANK == 0):
            msg = f"[Epoch {self.epoch} / {self.epochs}] "
            if self.loader_ is not None:
                msg += f"TRAIN LOSS: {_loss:.6e} | TEST LOSS: {loss_:.6e}"
            else:
                msg += f"LOSS: {_loss:.6e}"
            print(msg)

        self.stat_vals = {
            "train_loss" : _loss,
            "train_stats" : _stats,
            "test_loss" : loss_,
            "test_stats" : stats_,
        }

        if self.is_training:
            self.train_loss_fullbatch.append(_loss)
            self.test_loss_fullbatch.append(loss_)
            self.num_steps_fullbatch.append(len(self.train_loss_per_batch))
        
        return
#======================================================================#
#
