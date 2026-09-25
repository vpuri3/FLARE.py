#!/usr/bin/env python3
"""One-step AhmedML full-surface capacity probe for FLARE and FLAREPP."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from pdebench.callbacks import ahmedml_surface_metric_sums
from pdebench.config import Config
from pdebench.dataset.ahmedml import (
    Y_MEAN,
    Y_STD,
    AhmedMLSurfaceRunDataset,
)
from pdebench.distributed import build_context_parallel_state, cp_reduced_mse_loss, shard_batch
from pdebench.models.model_factory import make_model


def _eight_blocks(value: str) -> int:
    blocks = int(value)
    if blocks != 8:
        raise argparse.ArgumentTypeError("the matched capacity matrix requires --num-blocks 8")
    return blocks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("flare", "flarepp"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-blocks", type=_eight_blocks, default=8)
    return parser.parse_args(argv)


def make_run_dataset(args: argparse.Namespace) -> AhmedMLSurfaceRunDataset:
    return AhmedMLSurfaceRunDataset(args.data_root, [args.run_id])


def make_result(*, model: str, world_size: int, run_id: str, point_count: int) -> dict[str, Any]:
    return {
        "model": model,
        "world_size": world_size,
        "cp_size": world_size,
        "run_id": run_id,
        "point_count": point_count,
        "inference_status": "ERROR",
        "training_status": "ERROR",
        "metrics": {},
        "rank_memory": [],
    }


def dumps_result(result: dict[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, allow_nan=False)


def audit_event(event: str, **fields: Any) -> None:
    rank = int(os.environ.get("RANK", "0"))
    details = " ".join(f"{name}={str(value).lower() if isinstance(value, bool) else value}" for name, value in fields.items())
    line = f"AHMEDML_PROBE event={event} rank={rank}{' ' + details if details else ''}\n"
    os.write(2, line.encode())


def classify_exception(error: BaseException) -> str:
    if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower():
        return "OOM"
    return "ERROR"


def run_phase(action: Callable[[], Any]) -> tuple[str, Any | None, str | None]:
    try:
        return "PASS", action(), None
    except BaseException as error:  # Probe results must classify failures instead of losing the JSON record.
        return classify_exception(error), None, str(error)


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        # Bound failed collectives so torchrun gets a nonzero process outcome rather than an indefinite hang.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        timeout_seconds = int(os.environ.get("AHMEDML_PROBE_TIMEOUT_SECONDS", "180"))
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=timeout_seconds))
    return rank, local_rank, world_size


def _metadata(point_count: int) -> dict[str, Any]:
    return {
        "c_in": 6,
        "c_out": 4,
        "space_dim": 6,
        "pos_dim": 3,
        "time_cond": False,
        "max_length": point_count,
    }


def _global_phase(local: tuple[str, Any | None, str | None], world_size: int) -> tuple[str, Any | None, str | None]:
    if world_size == 1:
        return local
    local_summary = (local[0], None, local[2])
    gathered: list[tuple[str, Any | None, str | None] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_summary)
    failures = [outcome for outcome in gathered if outcome is not None and outcome[0] != "PASS"]
    if not failures:
        return local
    status = "OOM" if any(outcome[0] == "OOM" for outcome in failures) else "ERROR"
    messages = "; ".join(outcome[2] or status for outcome in failures)
    return status, None, messages


def gather_rank_memory(local: dict[str, int], world_size: int) -> list[dict[str, int]]:
    if world_size == 1:
        return [local]
    gathered: list[dict[str, int] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [item for item in gathered if item is not None]


def _memory_record(rank: int, local_rank: int) -> dict[str, int]:
    try:
        allocated = torch.cuda.max_memory_allocated(local_rank)
        reserved = torch.cuda.max_memory_reserved(local_rank)
    except (RuntimeError, torch.OutOfMemoryError):
        allocated = 0
        reserved = 0
    return {"rank": rank, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}


def _decode(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(Y_MEAN, device=tensor.device, dtype=tensor.dtype)
    std = torch.as_tensor(Y_STD, device=tensor.device, dtype=tensor.dtype)
    return tensor * std + mean


def _metrics(pred: torch.Tensor, target: torch.Tensor, cp_state) -> dict[str, float]:
    sums = ahmedml_surface_metric_sums(_decode(pred), _decode(target))
    values: dict[str, float] = {}
    for name, (numerator, denominator) in sums.items():
        pair = torch.stack((numerator, denominator))
        if cp_state is not None and cp_state.cp_size > 1:
            dist.all_reduce(pair, group=cp_state.cp_group)
        values[name] = float(torch.sqrt(pair[0] / pair[1]).item())
    return values


def build_probe_model(
    args: argparse.Namespace,
    point_count: int,
    rank: int,
    *,
    factory: Callable = make_model,
) -> tuple[Config, torch.nn.Module]:
    cfg = Config(
        dataset={"dataset": "ahmedml_surface", "data_root": str(args.data_root)},
        training={"batch_size": 1, "compile_model": False, "mixed_precision": True},
        model={"model": args.model, "num_blocks": args.num_blocks},
    )
    with contextlib.redirect_stdout(sys.stderr):
        return factory(cfg, _metadata(point_count), rank)


def run_workload(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    cp_state,
    autocast: Callable,
    optimizer_factory: Callable,
) -> tuple[tuple[str, Any | None, str | None], tuple[str, Any | None, str | None]]:
    inference = run_phase(lambda: _run_inference(model, x, target, cp_state, autocast))
    training = run_phase(lambda: _run_training(model, x, target, cp_state, autocast, optimizer_factory))
    return inference, training


def _run_inference(model, x, target, cp_state, autocast: Callable) -> dict[str, float]:
    model.eval()
    audit_event("inference_forward_start", model_call=1, full_mesh=True, local_points=x.shape[1])
    with torch.no_grad(), autocast():
        pred = model(x)
    audit_event("inference_forward_complete", model_call=1, full_mesh=True, local_points=x.shape[1])
    metrics = _metrics(pred, target, cp_state)
    audit_event("physical_metrics_complete", **metrics)
    return metrics


def _run_training(model, x, target, cp_state, autocast: Callable, optimizer_factory: Callable) -> None:
    model.train()
    optimizer = optimizer_factory(model.parameters())
    optimizer.zero_grad(set_to_none=True)
    audit_event("training_forward_start", model_call=2, full_mesh=True, local_points=x.shape[1])
    with autocast():
        pred = model(x)
        loss = (
            cp_reduced_mse_loss(pred, target, cp_state)
            if cp_state is not None
            else torch.nn.functional.mse_loss(pred, target)
        )
    audit_event("training_forward_complete", model_call=2, full_mesh=True, local_points=x.shape[1])
    loss.backward()
    audit_event("backward_complete")
    optimizer.step()
    audit_event("optimizer_step_complete", optimizer="AdamW")


def probe(args: argparse.Namespace) -> dict[str, Any]:
    rank, local_rank, world_size = _init_distributed()
    affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    audit_event(
        "distributed_initialized",
        pid=os.getpid(),
        local_rank=local_rank,
        world_size=world_size,
        cp_size=world_size,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        cpu_affinity=affinity,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    result = make_result(model=args.model, world_size=world_size, run_id=args.run_id, point_count=0)
    # Reset before dataset/device/model allocations so setup OOMs still have useful peak evidence.
    torch.cuda.reset_peak_memory_stats(local_rank)

    def setup():
        x, target = make_run_dataset(args)[0]
        point_count = int(x.shape[0])
        audit_event("run_reconstructed", run_id=args.run_id, point_count=point_count)
        x = x.unsqueeze(0).cuda(local_rank)
        target = target.unsqueeze(0).cuda(local_rank)
        cp_state = build_context_parallel_state(world_size, x.shape[1]) if world_size > 1 else None
        if cp_state is not None:
            x, target = shard_batch((x, target), cp_state, seq_dim=1)
        cfg, model = build_probe_model(args, point_count, rank)
        model = model.cuda(local_rank)
        if cp_state is not None:
            model.set_context_parallel(cp_state)
        audit_event("model_initialized", model=args.model, num_blocks=args.num_blocks, point_count=point_count)
        return x, target, point_count, cp_state, cfg, model

    # This is both setup outcome exchange and the readiness gate: no rank enters model collectives unless all are ready.
    setup_status, setup_value, setup_error = _global_phase(run_phase(setup), world_size)
    if setup_status != "PASS":
        result["inference_status"] = setup_status
        result["training_status"] = setup_status
        result["error"] = setup_error
        result["rank_memory"] = gather_rank_memory(_memory_record(rank, local_rank), world_size)
        audit_event("peak_memory_report_complete", rank_memory=result["rank_memory"])
        return result

    x, target, point_count, cp_state, cfg, model = setup_value
    result["point_count"] = point_count

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16)

    def optimizer_factory(parameters):
        return torch.optim.AdamW(parameters, lr=cfg.optimizer.learning_rate)

    inference_outcome = _global_phase(
        run_phase(lambda: _run_inference(model, x, target, cp_state, autocast)), world_size
    )
    result["inference_status"] = inference_outcome[0]
    if inference_outcome[1] is not None:
        result["metrics"] = inference_outcome[1]
    elif inference_outcome[2]:
        result["inference_error"] = inference_outcome[2]

    # Outcome exchange above is also the training readiness gate, preventing phase-mismatched collectives.
    training_outcome = _global_phase(
        run_phase(lambda: _run_training(model, x, target, cp_state, autocast, optimizer_factory)), world_size
    )
    result["training_status"] = training_outcome[0]
    if training_outcome[2]:
        result["training_error"] = training_outcome[2]

    result["rank_memory"] = gather_rank_memory(_memory_record(rank, local_rank), world_size)
    audit_event("peak_memory_report_complete", rank_memory=result["rank_memory"])
    audit_event(
        "probe_complete",
        inference_status=result["inference_status"],
        training_status=result["training_status"],
        point_count=result["point_count"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    try:
        try:
            result = probe(args)
        except BaseException as error:
            result = make_result(
                model=args.model,
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                run_id=args.run_id,
                point_count=0,
            )
            status = classify_exception(error)
            result["inference_status"] = status
            result["training_status"] = status
            result["error"] = str(error)
        if rank == 0:
            print(dumps_result(result), flush=True)
        return 0 if result["inference_status"] == result["training_status"] == "PASS" else 2
    finally:
        if dist.is_initialized():
            audit_event("process_group_cleanup_start")
            dist.destroy_process_group()
            audit_event("process_group_cleanup_complete")


if __name__ == "__main__":
    raise SystemExit(main())
