"""Shared torch.distributed rank/barrier helpers for dataset cache builds."""

from __future__ import annotations

import torch.distributed as dist


def distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
