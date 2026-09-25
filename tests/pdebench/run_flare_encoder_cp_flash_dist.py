#!/usr/bin/env python3
"""Standalone torchrun harness for FlareEncoderCPFlash (no pytest).

Avoids pytest fixture/collection deadlocks under multi-GPU. Run:

    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    torchrun --standalone --nproc_per_node=2 -m tests.pdebench.run_flare_encoder_cp_flash_dist
    torchrun --standalone --nproc_per_node=4 -m tests.pdebench.run_flare_encoder_cp_flash_dist
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from pdebench.distributed.context_parallel import build_context_parallel_state
from pdebench.distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive
from pdebench.distributed.utils import shard_sequence_tensor
from pdebench.models.flare import FlareConfig, FLAREModel

B, H, M, N, D = 2, 4, 8, 128, 16
SEQ_LEN = 128
CHANNEL_DIM = 32
NUM_HEADS = 4
NUM_LATENTS = 8


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def shard_kv(k, v, rank, world_size, seq_dim=2):
    n = k.size(seq_dim)
    chunk = n // world_size
    start = rank * chunk
    end = n if rank == world_size - 1 else (rank + 1) * chunk
    return (
        k.narrow(seq_dim, start, end - start).contiguous(),
        v.narrow(seq_dim, start, end - start).contiguous(),
    )


def check_forward(rank, world_size, device, cp_state):
    _log(rank, "[1/4] forward vs dense naiive")
    scale = D ** -0.5
    torch.manual_seed(0)
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k_local, v_local = shard_kv(k_full, v_full, rank, world_size)

    z_ref = FlareEncoderCPNaiive()(q, k_full, v_full, cp_group=None, scale=scale)
    z_flash = FlareEncoderCPFlash()(q, k_local, v_local, cp_group=cp_state.cp_group, scale=scale)

    assert z_flash.dtype == q.dtype
    assert torch.isfinite(z_flash).all()
    assert torch.allclose(z_flash.float(), z_ref.float(), atol=2e-2, rtol=2e-2)
    _log(rank, "[1/4] ok")


def check_grads(rank, world_size, device, cp_state):
    _log(rank, "[2/4] grads vs dense naiive")
    scale = D ** -0.5
    torch.manual_seed(1)
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k_local, v_local = shard_kv(k_full, v_full, rank, world_size)

    q_ref = q.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    q_f = q.clone().requires_grad_(True)
    k_f = k_local.clone().requires_grad_(True)
    v_f = v_local.clone().requires_grad_(True)

    z_ref = FlareEncoderCPNaiive()(q_ref, k_ref, v_ref, cp_group=None, scale=scale)
    z_f = FlareEncoderCPFlash()(q_f, k_f, v_f, cp_group=cp_state.cp_group, scale=scale)

    # Upstream dz must partition across ranks: Flash backward all_reduce(SUM)s dz.
    # Applying the same dz on every rank would scale grads by world_size.
    torch.manual_seed(2)
    g_full = torch.randn_like(z_ref)
    g_flash = g_full if rank == 0 else torch.zeros_like(g_full)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(z_ref, (q_ref, k_ref, v_ref), g_full)
    dq_f, dk_f, dv_f = torch.autograd.grad(z_f, (q_f, k_f, v_f), g_flash)

    dq_sum = dq_f.clone()
    if cp_state.cp_group is not None:
        dist.all_reduce(dq_sum, op=dist.ReduceOp.SUM, group=cp_state.cp_group)

    dk_ref_local, dv_ref_local = shard_kv(dk_ref, dv_ref, rank, world_size)
    assert torch.isfinite(dq_f).all() and torch.isfinite(dk_f).all() and torch.isfinite(dv_f).all()
    assert torch.allclose(dq_sum.float(), dq_ref.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(dk_f.float(), dk_ref_local.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(dv_f.float(), dv_ref_local.float(), atol=3e-2, rtol=3e-2)
    _log(rank, "[2/4] ok")


def check_train_step(rank, world_size, device, cp_state, amp_dtype):
    _log(rank, f"[train] amp={amp_dtype}")
    torch.manual_seed(3)
    model = FLAREModel(
        FlareConfig(
            num_blocks=1,
            channel_dim=CHANNEL_DIM,
            num_heads=NUM_HEADS,
            num_latents=NUM_LATENTS,
            encoder_cp_backend="flash",
        ),
        metadata={"c_in": 3, "c_out": 2},
    ).to(device)
    assert isinstance(model.blocks[0].att.encoder_cp, FlareEncoderCPFlash)
    model.set_context_parallel(cp_state)

    x_full = torch.randn(2, SEQ_LEN, 3, device=device)
    x_local = shard_sequence_tensor(x_full, cp_state, seq_dim=1)
    target_local = torch.randn(2, x_local.size(1), 2, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        pred = model(x_local)
        loss = torch.nn.functional.mse_loss(pred, target_local)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    opt.step()
    _log(rank, f"[train] amp={amp_dtype} ok")


def main() -> int:
    if "LOCAL_RANK" not in os.environ:
        print("Launch with torchrun (LOCAL_RANK unset).", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _log(rank, f"world_size={world_size} device={device}")

    try:
        cp_state = build_context_parallel_state(cp_size=world_size, sequence_length=SEQ_LEN)
        check_forward(rank, world_size, device, cp_state)
        check_grads(rank, world_size, device, cp_state)
        check_train_step(rank, world_size, device, cp_state, torch.float16)
        check_train_step(rank, world_size, device, cp_state, torch.bfloat16)
        _log(rank, "ALL PASSED")
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
