from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from .context_parallel import ContextParallelState, _compute_shard_bounds


def _resolve_seq_dim(x: torch.Tensor, seq_dim: int) -> int:
    dim = seq_dim if seq_dim >= 0 else x.ndim + seq_dim
    if dim < 0 or dim >= x.ndim:
        raise ValueError(f"Invalid seq_dim={seq_dim} for tensor with shape={tuple(x.shape)}.")
    return dim


def shard_sequence_tensor(x: torch.Tensor, cp_state: ContextParallelState, seq_dim: int = 1) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("shard_sequence_tensor expects a torch.Tensor input.")
    dim = _resolve_seq_dim(x, seq_dim)
    start, end = _compute_shard_bounds(x.size(dim), cp_state.cp_rank, cp_state.cp_size)
    cp_state.seq_start = start
    cp_state.seq_end = end

    slicer = [slice(None)] * x.ndim
    slicer[dim] = slice(start, end)
    return x[tuple(slicer)].contiguous()


def _should_shard_tensor(x: torch.Tensor, seq_dim: int, cp_size: int) -> bool:
    dim = _resolve_seq_dim(x, seq_dim)
    return x.ndim > dim and x.size(dim) >= cp_size and x.size(dim) > 1


def shard_batch(
    batch: Any,
    cp_state: ContextParallelState,
    keys: Optional[Sequence[str]] = None,
    seq_dim: Union[int, Dict[str, int]] = 1,
) -> Any:
    if cp_state.cp_size <= 1:
        return batch

    def get_dim(key: Optional[str]) -> int:
        if isinstance(seq_dim, dict):
            if key is not None and key in seq_dim:
                return seq_dim[key]
            return 1
        return seq_dim

    if isinstance(batch, dict):
        shard_keys = set(batch.keys()) if keys is None else set(keys)
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and key in shard_keys:
                dim = get_dim(key)
                out[key] = shard_sequence_tensor(value, cp_state, seq_dim=dim) if _should_shard_tensor(value, dim, cp_state.cp_size) else value
            else:
                out[key] = value
        return out

    if isinstance(batch, list):
        out = []
        for value in batch:
            if isinstance(value, torch.Tensor):
                dim = get_dim(None)
                out.append(shard_sequence_tensor(value, cp_state, seq_dim=dim) if _should_shard_tensor(value, dim, cp_state.cp_size) else value)
            else:
                out.append(value)
        return out

    if isinstance(batch, tuple):
        out = []
        for value in batch:
            if isinstance(value, torch.Tensor):
                dim = get_dim(None)
                out.append(shard_sequence_tensor(value, cp_state, seq_dim=dim) if _should_shard_tensor(value, dim, cp_state.cp_size) else value)
            else:
                out.append(value)
        return tuple(out)

    if isinstance(batch, torch.Tensor):
        dim = get_dim(None)
        return shard_sequence_tensor(batch, cp_state, seq_dim=dim) if _should_shard_tensor(batch, dim, cp_state.cp_size) else batch

    return batch


def gather_sequence_tensor(x: torch.Tensor, cp_state: ContextParallelState, seq_dim: int = 1) -> torch.Tensor:
    if cp_state.cp_size <= 1:
        return x
    if cp_state.cp_group is None:
        raise RuntimeError("Context parallel group is not initialized.")

    dim = _resolve_seq_dim(x, seq_dim)
    group = cp_state.cp_group
    device = x.device

    local_len = torch.tensor([x.size(dim)], device=device, dtype=torch.long)
    len_list = [torch.zeros_like(local_len) for _ in range(cp_state.cp_size)]
    dist.all_gather(len_list, local_len, group=group)
    lengths = [int(t.item()) for t in len_list]
    max_len = max(lengths)

    if x.size(dim) < max_len:
        pad_shape = list(x.shape)
        pad_shape[dim] = max_len - x.size(dim)
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=device)
        x_pad = torch.cat([x, padding], dim=dim)
    else:
        x_pad = x

    gather_list = [torch.empty_like(x_pad) for _ in range(cp_state.cp_size)]
    dist.all_gather(gather_list, x_pad, group=group)

    shards = []
    for shard, shard_len in zip(gather_list, lengths):
        slicer = [slice(None)] * shard.ndim
        slicer[dim] = slice(0, shard_len)
        shards.append(shard[tuple(slicer)])

    return torch.cat(shards, dim=dim)


def reduce_scalar_pair(
    scalar_sum: torch.Tensor,
    scalar_count: torch.Tensor,
    cp_state: ContextParallelState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cp_state.cp_size <= 1:
        return scalar_sum, scalar_count
    if cp_state.cp_group is None:
        raise RuntimeError("Context parallel group is not initialized.")

    if scalar_sum.requires_grad:
        total_sum = dist_nn.all_reduce(scalar_sum, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
    else:
        total_sum = scalar_sum.clone()
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM, group=cp_state.cp_group)

    total_count = scalar_count.clone()
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
    return total_sum, total_count


def cp_reduced_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cp_state: ContextParallelState,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match. Got {pred.shape} vs {target.shape}.")

    # Reduce in fp32. Under AMP fp16, casting numel/counts to pred.dtype overflows
    # once numel > 65504 (e.g. NASA-CRM CP shards), yielding Inf count and loss=0.
    sq_err = (pred.float() - target.float()).pow(2)
    if mask is not None:
        w = mask.to(dtype=torch.float32, device=pred.device)
        while w.ndim < sq_err.ndim:
            w = w.unsqueeze(-1)
        sq_err = sq_err * w
        local_count = w.expand_as(sq_err).sum()
    else:
        local_count = torch.tensor(float(sq_err.numel()), dtype=torch.float32, device=pred.device)

    local_sum = sq_err.sum()
    total_sum, total_count = reduce_scalar_pair(local_sum, local_count, cp_state)
    return total_sum / total_count.clamp_min(eps)


def cp_reduced_rel_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cp_state: ContextParallelState,
    eps: float = 1e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match. Got {pred.shape} vs {target.shape}.")

    reduce_dims = tuple(range(1, pred.ndim))
    num_local = torch.sum((pred - target).pow(2), dim=reduce_dims)
    den_local = torch.sum(target.pow(2), dim=reduce_dims).clamp_min(eps)

    if cp_state.cp_size > 1:
        if cp_state.cp_group is None:
            raise RuntimeError("Context parallel group is not initialized.")
        num_local = dist_nn.all_reduce(num_local, op=dist.ReduceOp.SUM, group=cp_state.cp_group)
        dist.all_reduce(den_local, op=dist.ReduceOp.SUM, group=cp_state.cp_group)

    rel = torch.sqrt(num_local.clamp_min(eps)) / torch.sqrt(den_local.clamp_min(eps))
    return rel.mean()
