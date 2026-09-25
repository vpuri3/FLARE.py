from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn


class FlareEncoderCPNaiive(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    @torch.compiler.disable
    def forward(
        self,
        q_latent: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cp_group: Optional[dist.ProcessGroup],
        scale: float,
    ) -> torch.Tensor:
        if q_latent.ndim != 4 or k_local.ndim != 4 or v_local.ndim != 4:
            raise ValueError("q_latent, k_local, and v_local must have shape [B, H, N/M, D].")
        if k_local.shape != v_local.shape:
            raise ValueError(f"k_local and v_local shapes must match. Got {k_local.shape} vs {v_local.shape}.")
        if q_latent.shape[0] != k_local.shape[0] or q_latent.shape[1] != k_local.shape[1]:
            raise ValueError("q_latent and k_local must share batch/head dimensions.")
        if q_latent.shape[-1] != k_local.shape[-1]:
            raise ValueError("q_latent and k_local must share head dimension.")

        # Online-softmax accumulators u = exp(s) @ v have magnitude ~O(N). Under AMP
        # fp16, .float() alone is not enough — autocast still runs the GEMMs in fp16
        # and the CP all_reduce(SUM) overflows. Force true fp32 for this block.
        device_type = q_latent.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            q_fp32 = q_latent.float()
            k_fp32 = k_local.float()
            v_fp32 = v_local.float()
            scale_f = float(scale)

            # Local summary terms over local token shard.
            scores_local = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale_f
            m_local = scores_local.max(dim=-1, keepdim=True).values

            if cp_group is not None:
                m_global = m_local.detach().clone()
                dist.all_reduce(m_global, op=dist.ReduceOp.MAX, group=cp_group)
            else:
                m_global = m_local

            exp_scores_local = torch.exp(scores_local - m_global)
            d_local = exp_scores_local.sum(dim=-1, keepdim=True)
            u_local = torch.matmul(exp_scores_local, v_fp32)

            if cp_group is not None:
                d_global = dist_nn.all_reduce(d_local, op=dist.ReduceOp.SUM, group=cp_group)
                u_global = dist_nn.all_reduce(u_local, op=dist.ReduceOp.SUM, group=cp_group)
            else:
                d_global = d_local
                u_global = u_local

            z_global = u_global / d_global.clamp_min(self.eps)
        return z_global.to(dtype=v_local.dtype)


def flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, unused, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_flash_attention.default(
            q, k, v, 0.0, False, False, scale=float(scale),
        )
    )
    metadata = {
        "cum_seq_q": cum_seq_q,
        "cum_seq_k": cum_seq_k,
        "max_q": max_q,
        "max_k": max_k,
        "rng_state": rng_state,
        "unused": unused,
        "debug_attn_mask": debug_attn_mask,
    }
    return out, lse, metadata


def flash_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    metadata: dict[str, Any],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # PyTorch 2.8: forward rng_state -> philox_seed; unused -> philox_offset
    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        dout,
        q,
        k,
        v,
        out,
        lse,
        metadata["cum_seq_q"],
        metadata["cum_seq_k"],
        metadata["max_q"],
        metadata["max_k"],
        0.0,
        False,
        metadata["rng_state"],
        metadata["unused"],
        scale=float(scale),
    )
    return dq, dk, dv


def _require_flash_dtypes(*tensors: torch.Tensor) -> torch.dtype:
    """Flash CP requires uniform fp16/bf16 inputs (no silent cast / autocast fallback)."""
    if not tensors:
        raise RuntimeError("FlareEncoderCPFlash requires at least one input tensor.")
    dtype = tensors[0].dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "FlareEncoderCPFlash requires fp16/bf16 inputs; "
            f"got {[t.dtype for t in tensors]}."
        )
    if any(t.dtype != dtype for t in tensors):
        raise RuntimeError(
            "FlareEncoderCPFlash requires all inputs to share one fp16/bf16 dtype; "
            f"got {[t.dtype for t in tensors]}."
        )
    return dtype


def promote_for_flash(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cast tensors to a shared fp16/bf16 dtype for FlareEncoderCPFlash.

    Handles the common AMP case where LayerNorm/RMSNorm leave some tensors in
    fp32 while others are already low-precision. Rejects mixed fp16/bf16.
    Falls back to the active CUDA autocast dtype when every input is fp32.
    """
    if not tensors:
        raise RuntimeError("FlareEncoderCPFlash requires at least one input tensor.")

    dtypes = {t.dtype for t in tensors}
    low = dtypes & {torch.float16, torch.bfloat16}
    if len(low) > 1:
        raise RuntimeError(
            "FlareEncoderCPFlash requires all inputs to share one fp16/bf16 dtype; "
            f"got {[t.dtype for t in tensors]}."
        )
    if len(low) == 1:
        flash_dtype = next(iter(low))
    elif torch.is_autocast_enabled("cuda"):
        flash_dtype = torch.get_autocast_dtype("cuda")
        if flash_dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlareEncoderCPFlash requires fp16/bf16 activations "
                f"(enable AMP or cast q/k/v); got autocast dtype={flash_dtype}."
            )
    else:
        raise RuntimeError(
            "FlareEncoderCPFlash requires fp16/bf16 activations "
            "(enable AMP or cast q/k/v); "
            f"got dtypes={[t.dtype for t in tensors]}."
        )
    return tuple(t.to(dtype=flash_dtype) for t in tensors)


def pack_flash_cp_mass_numerator(mass: torch.Tensor, numerator: torch.Tensor) -> torch.Tensor:
    """Pack fp32 mass [..., M] and numerator [..., M, D] into [..., M, D+1] for one SUM."""
    if numerator.shape[:-1] != mass.shape:
        raise ValueError(
            f"mass shape {tuple(mass.shape)} must match numerator leading dims {tuple(numerator.shape[:-1])}."
        )
    return torch.cat([numerator, mass.unsqueeze(-1)], dim=-1)


def unpack_flash_cp_mass_numerator(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack [..., M, D+1] into mass [..., M] and numerator [..., M, D]."""
    return packed[..., -1], packed[..., :-1]


def merge_flash_cp_lse_stats(
    out_shards: tuple[torch.Tensor, ...],
    lse_shards: tuple[torch.Tensor, ...],
    lse_max: torch.Tensor,
    *,
    packed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference multi-shard LSE merge (no NCCL). Used to prove packed SUM ≡ separate SUMs."""
    if len(out_shards) != len(lse_shards):
        raise ValueError("out_shards and lse_shards must have the same length.")
    masses = [torch.exp(lse.float() - lse_max) for lse in lse_shards]
    numerators = [mass.unsqueeze(-1) * out.float() for mass, out in zip(masses, out_shards)]
    if packed:
        packed_sum = pack_flash_cp_mass_numerator(masses[0], numerators[0])
        for mass, numerator in zip(masses[1:], numerators[1:]):
            packed_sum = packed_sum + pack_flash_cp_mass_numerator(mass, numerator)
        mass_global, numerator_global = unpack_flash_cp_mass_numerator(packed_sum)
    else:
        mass_global = masses[0]
        numerator_global = numerators[0]
        for mass, numerator in zip(masses[1:], numerators[1:]):
            mass_global = mass_global + mass
            numerator_global = numerator_global + numerator
    z_global = numerator_global / mass_global.unsqueeze(-1)
    lse_global = lse_max + torch.log(mass_global)
    return z_global, lse_global


class FlareFlashCPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_local, v_local, cp_group, scale):
        flash_dtype = _require_flash_dtypes(q, k_local, v_local)
        out_local, lse_local, metadata = flash_forward(q, k_local, v_local, float(scale))
        # LSE merge stays in fp32 for numerical stability across CP ranks.
        lse_local = lse_local.float()
        if cp_group is not None:
            lse_max = lse_local.detach().clone()
            dist.all_reduce(lse_max, op=dist.ReduceOp.MAX, group=cp_group)
            mass_local = torch.exp(lse_local - lse_max)
            numerator_local = mass_local.unsqueeze(-1) * out_local.float()
            # One SUM for mass + numerator (was two separate all_reduces).
            packed = pack_flash_cp_mass_numerator(mass_local, numerator_local).clone()
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=cp_group)
            mass_global, numerator_global = unpack_flash_cp_mass_numerator(packed)
            lse_global = lse_max + torch.log(mass_global)
            z_global = (numerator_global / mass_global.unsqueeze(-1)).to(dtype=flash_dtype)
        else:
            lse_global = lse_local
            z_global = out_local

        ctx.cp_group = cp_group
        ctx.scale = float(scale)
        ctx.flash_metadata = metadata
        ctx.save_for_backward(q, k_local, v_local, z_global, lse_global)
        return z_global

    @staticmethod
    def backward(ctx, dz_local):
        q, k_local, v_local, z_global, lse_global = ctx.saved_tensors
        cp_group = ctx.cp_group
        dz_global = dz_local.contiguous()
        if cp_group is not None:
            dz_global = dz_global.clone()
            dist.all_reduce(dz_global, op=dist.ReduceOp.SUM, group=cp_group)
        if dz_global.dtype != q.dtype:
            raise RuntimeError(
                "FlareEncoderCPFlash backward expects dz in the Flash input dtype; "
                f"got dz={dz_global.dtype}, expected={q.dtype}."
            )
        dq_partial, dk_local, dv_local = flash_backward(
            dout=dz_global,
            q=q,
            k=k_local,
            v=v_local,
            out=z_global,
            lse=lse_global,
            metadata=ctx.flash_metadata,
            scale=ctx.scale,
        )
        return dq_partial, dk_local, dv_local, None, None


class FlareEncoderCPFlash(nn.Module):
    @torch.compiler.disable
    def forward(
        self,
        q_latent: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cp_group: Optional[dist.ProcessGroup],
        scale: float,
    ) -> torch.Tensor:
        if q_latent.ndim != 4 or k_local.ndim != 4 or v_local.ndim != 4:
            raise ValueError("q_latent, k_local, and v_local must have shape [B, H, N/M, D].")
        if k_local.shape != v_local.shape:
            raise ValueError(f"k_local and v_local shapes must match. Got {k_local.shape} vs {v_local.shape}.")
        if q_latent.shape[0] != k_local.shape[0] or q_latent.shape[1] != k_local.shape[1]:
            raise ValueError("q_latent and k_local must share batch/head dimensions.")
        if q_latent.shape[-1] != k_local.shape[-1]:
            raise ValueError("q_latent and k_local must share head dimension.")
        q_latent, k_local, v_local = promote_for_flash(q_latent, k_local, v_local)
        return FlareFlashCPFunction.apply(q_latent, k_local, v_local, cp_group, scale)

