from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch.distributed as dist


@dataclass
class ContextParallelState:
    rank: int
    world_size: int
    cp_group: Optional[dist.ProcessGroup]
    cp_size: int
    cp_rank: int
    seq_start: int
    seq_end: int


def _compute_shard_bounds(length: int, shard_idx: int, num_shards: int) -> Tuple[int, int]:
    if num_shards <= 1:
        return 0, length
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"Invalid shard index {shard_idx} for num_shards={num_shards}.")

    base = length // num_shards
    rem = length % num_shards
    start = shard_idx * base + min(shard_idx, rem)
    stop = start + base + (1 if shard_idx < rem else 0)
    return start, stop


def build_context_parallel_state(cp_size: int, sequence_length: Optional[int] = None) -> ContextParallelState:
    if cp_size < 1:
        raise ValueError(f"context_parallel_size must be >= 1, got {cp_size}.")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before building ContextParallelState.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if cp_size > world_size:
        raise ValueError(f"context_parallel_size={cp_size} exceeds world_size={world_size}.")
    if world_size % cp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by context_parallel_size={cp_size}.")

    group_base = (rank // cp_size) * cp_size
    ranks = list(range(group_base, group_base + cp_size))
    # Prefer WORLD when the CP group is the full world. A redundant NCCL
    # new_group(ranks=all) is a known hang/flake source under torchrun.
    if cp_size <= 1:
        cp_group = None
    elif cp_size == world_size:
        cp_group = dist.group.WORLD
    else:
        cp_group = dist.new_group(ranks=ranks)
    cp_rank = rank - group_base

    if sequence_length is None:
        seq_start, seq_end = 0, 0
    else:
        seq_start, seq_end = _compute_shard_bounds(sequence_length, cp_rank, cp_size)

    return ContextParallelState(
        rank=rank,
        world_size=world_size,
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_start=seq_start,
        seq_end=seq_end,
    )

