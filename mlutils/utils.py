#
import torch
from torch import nn
import torch.distributed as dist

import numpy as np

from datetime import timedelta
import os
import random
import pathlib

__all__ = [
    # cache directory
    'set_cache_path',
    'dotdot',

    # Sampler
    'RepeatBatchSampler',

    # experiment management
    "get_next_exp_name",

    # device and seed
    "set_seed",
    'set_num_threads',
    "select_device",

    'is_torchrun',
    'dist_backend',
    'dist_setup',
    'dist_finalize',
    'get_module',

    # model utility
    "num_parameters",

    # statistics
    "r2",

    # versioning hell
    'to_numpy',
    'check_package_version_lteq'
]

#======================================================================#
def dotdot(dir: str):
    # assert os.path.exists(dir), f"Directory {dir} does not exist."
    return os.path.abspath(os.path.join(dir, '..'))

def set_cache_path(BASE_DIR: str):
    CACHE_BASE = os.path.join(BASE_DIR, "cache")
    env_vars = {
        "PIP_CACHE_DIR": os.path.join(CACHE_BASE, "pip"),
        "UV_CACHE_DIR": os.path.join(CACHE_BASE, "uv"),
        "XDG_CACHE_HOME": CACHE_BASE,
        "TORCH_HOME": os.path.join(CACHE_BASE, "torch"),
        "WANDB_CACHE_DIR": os.path.join(CACHE_BASE, "wandb"),
        "TRITON_CACHE_DIR": os.path.join(CACHE_BASE, "triton"),
        "DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "MPLCONFIGDIR": os.path.join(CACHE_BASE, "matplotlib"),
        "HF_HOME": os.path.join(CACHE_BASE, "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
        "HF_DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "HF_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
    }

    for var, path in env_vars.items():
        os.environ[var] = str(path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return

#======================================================================#
class RepeatBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, batch_sampler: torch.utils.data.BatchSampler, repeat: int):
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1. Got {repeat}.")
        self.batch_sampler = batch_sampler
        self.repeat = repeat

    def __iter__(self):
        for batch in self.batch_sampler:
            for _ in range(self.repeat):
                yield batch

    def __len__(self):
        return len(self.batch_sampler) * self.repeat

#=======================================================================#
def get_next_exp_name(CASEDIR: str, exp_name: str):
    """
    Get the next experiment name in the given directory.
    If the experiment name already exists, it is modified by adding a '_<number>' suffix.
    If the next number is already taken, the next experiment name is the base name with the next number.
    This is repeated until a unique experiment name is found.
    
    Works with arbitrarily deep paths like "a/b/c/d/e/experiment" by preserving the parent
    directory structure and only modifying the final directory name.

    Args:
        CASEDIR: The directory to store the experiments.
        exp_name: The base name of the experiment (can be a nested path like "path/to/dir").

    Returns:
        The next experiment name (preserving the full path structure).
    """

    # Extract the final directory name and parent path
    # os.path.basename/dirname work correctly with paths of any depth
    exp_base = os.path.basename(exp_name)  # e.g., "dir" from "a/b/c/dir"
    exp_parent = os.path.dirname(exp_name)  # e.g., "a/b/c" from "a/b/c/dir"
    parent_dir = os.path.join(CASEDIR, exp_parent) if exp_parent else CASEDIR

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    existing_dirs = [d for d in os.listdir(parent_dir) if d.startswith(exp_base)]

    if exp_base in existing_dirs:
        numbers = [0 if d == exp_base else int(d.split('_')[-1]) 
                   for d in existing_dirs if d == exp_base or 
                   (d.startswith(exp_base + '_') and d.split('_')[-1].isdigit())]
        next_num = max(numbers, default=-1) + 1
        exp_base_new = f"{exp_base}_{next_num:02d}"
        exp_name_new = os.path.join(exp_parent, exp_base_new) if exp_parent else exp_base_new

        if exp_base_new in existing_dirs:
            return get_next_exp_name(CASEDIR, exp_name_new)
        
        return exp_name_new

    return exp_name

#=======================================================================#
def to_numpy(t: torch.Tensor):
    '''
    Torch 1.10 compatible equivalent of `t.numpy(force=True)`.
    '''
    return t.detach().cpu().resolve_conj().resolve_neg().numpy()

def check_package_version_lteq(pkg: str, version: str):
    import importlib.metadata
    from packaging import version as packaging_version
    
    try:
        current_version = importlib.metadata.version(pkg)
        return packaging_version.parse(current_version) <= packaging_version.parse(version)
    except importlib.metadata.PackageNotFoundError:
        # If package not found, assume it doesn't meet the version requirement
        return False

#=======================================================================#
def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    return

#=======================================================================#
def set_num_threads(threads=None):
    if threads is not None:
        threads = os.cpu_count()

    torch.set_num_threads(threads)

    os.environ["OMP_NUM_THREADS"]        = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(threads)
    os.environ["MKL_NUM_THREADS"]        = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(threads)

    return

#=======================================================================#
def select_device(device=None, verbose=False):
    if device is not None:
        return device

    if is_torchrun():
        if not dist.is_initialized():
            dist_setup()
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        return torch.device(LOCAL_RANK)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if verbose:
        print(f'using device {device}.')

    return device

def dist_backend():
    if dist.is_nccl_available():
        return 'nccl'
    elif dist.is_gloo_available():
        return 'gloo'
    elif dist.is_mpi_available():
        return 'mpi'
    else:
        raise RuntimeError("No suitable backend found!")

def is_torchrun():
    required_env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    return all(var in os.environ for var in required_env_vars)

def dist_setup():
    backend = dist_backend()
    if backend != 'nccl':
        print(f'using {backend} backend for torch.distributed.')

    if is_torchrun():
        timeout_minutes_env = os.environ.get("MLUTILS_DDP_TIMEOUT_MINUTES", "10")
        try:
            timeout_minutes = int(timeout_minutes_env)
        except ValueError:
            timeout_minutes = 10

        GLOBAL_RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend=backend,
            rank=GLOBAL_RANK,
            world_size=WORLD_SIZE,
            device_id=torch.device(LOCAL_RANK),
            timeout=timedelta(minutes=timeout_minutes),
        )
    else:
        pass

    return

def dist_finalize():
    if dist.is_initialized():
        dist.destroy_process_group()
    return

def get_module(model: nn.Module) -> nn.Module:
    DDP_TYPES = (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)
    return model.module if isinstance(model, DDP_TYPES) else model

#=======================================================================#
def num_parameters(model : nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def r2(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    y_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-6)
    return r2.item()

#=======================================================================#
#
