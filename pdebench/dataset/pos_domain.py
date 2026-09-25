"""Train-domain geometry metadata for GLT PEs (shift/scale + normalized AABB)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PosDomain:
    shift: Tensor  # [1, D]
    scale: Tensor  # [1, D]
    normalized_pos_expanse: Tensor  # [D, 2] columns (lo, hi)


def extract_batch_query_pos(batch) -> Tensor:
    """Return encoded query positions ``[N, D]`` (no pad tokens)."""
    if isinstance(batch, dict):
        if "flat_pos" in batch and batch["flat_pos"] is not None:
            pos = batch["flat_pos"]
            if pos.ndim != 2:
                raise ValueError(f"flat_pos must be [N,D]; got {tuple(pos.shape)}")
            return pos
        if "pos" in batch and batch["pos"] is not None:
            pos = batch["pos"]
            mask = batch.get("mask")
            if pos.ndim == 3 and mask is not None:
                return pos[mask.bool()]
            if pos.ndim == 2:
                return pos
            raise ValueError(f"Unsupported dict pos shape {tuple(pos.shape)}")
    pos = getattr(batch, "pos", None)
    if pos is not None:
        if pos.ndim != 2:
            raise ValueError(f"batch.pos must be [N,D]; got {tuple(pos.shape)}")
        return pos
    raise ValueError(f"Cannot extract query pos from batch type {type(batch)!r}")


def compute_pos_domain_from_dataloader(
    loader,
    *,
    shift: Tensor,
    scale: Tensor,
) -> PosDomain:
    shift = shift.detach().float().cpu().reshape(1, -1)
    scale = scale.detach().float().cpu().reshape(1, -1)
    if shift.shape[-1] != scale.shape[-1]:
        raise ValueError(f"shift/scale dim mismatch: {tuple(shift.shape)} vs {tuple(scale.shape)}")
    dim = int(shift.shape[-1])
    lo = None
    hi = None
    for batch in loader:
        pos = extract_batch_query_pos(batch).detach().float().cpu()
        if pos.numel() == 0:
            continue
        if pos.shape[-1] != dim:
            raise ValueError(f"query pos dim {pos.shape[-1]} != shift dim {dim}")
        batch_lo = pos.amin(dim=0)
        batch_hi = pos.amax(dim=0)
        lo = batch_lo if lo is None else torch.minimum(lo, batch_lo)
        hi = batch_hi if hi is None else torch.maximum(hi, batch_hi)
    if lo is None or hi is None:
        raise ValueError("Cannot compute PosDomain: train dataloader yielded no query positions.")
    expanse = torch.stack([lo, hi], dim=-1)  # [D, 2]
    return PosDomain(shift=shift.clone(), scale=scale.clone(), normalized_pos_expanse=expanse)


def resolve_pos_normalizer_shift_scale(metadata: dict) -> tuple[Tensor, Tensor]:
    normalizer = metadata.get("pos_normalizer", metadata.get("x_normalizer"))
    if normalizer is None:
        raise ValueError(
            "PosDomain requested but metadata lacks pos_normalizer/x_normalizer with mean/std."
        )
    if not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        raise ValueError(
            f"PosDomain normalizer must expose mean/std; got {type(normalizer)!r}."
        )
    return normalizer.mean, normalizer.std


def maybe_attach_pos_domain(
    metadata: dict,
    train_dataset,
    *,
    batch_size: int,
    feature_request,
    num_workers: int = 0,
) -> dict:
    """If ``feature_request.pos_domain``, scan train DataLoader and set metadata['pos_domain']."""
    if not bool(getattr(feature_request, "pos_domain", False)):
        return metadata
    from torch.utils.data import DataLoader

    shift, scale = resolve_pos_normalizer_shift_scale(metadata)
    collate_fn = metadata.get("train_collate_fn")
    loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
    )
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = DataLoader(**loader_kwargs)
    metadata = dict(metadata)
    metadata["pos_domain"] = compute_pos_domain_from_dataloader(loader, shift=shift, scale=scale)
    return metadata
