"""Multiprocess shard builder for el-pl v3."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from pdebench.dataset.plaid_elpl_v3.constants import CACHE_SCHEMA_VERSION, TRAJECTORIES_PER_SHARD
from pdebench.dataset.plaid_elpl_v3.laplacian import pack_laplacian_for_trajectory
from pdebench.dataset.plaid_elpl_v3.manifest import partition_sim_ids
from pdebench.dataset.plaid_elpl_v3.parse import parse_trajectory_bundle, save_shard_payload
from pdebench.dataset.plaid_elpl_v3.schema import ShardPayload
from pdebench.dataset.thread_limits import set_compute_thread_limits

_BUILDER_STATE: dict[str, Any] = {}


def _precompute_workers(num_tasks: int) -> int:
    default = 12
    cap = int(os.environ.get("PLAID_ELPL_PRECOMPUTE_WORKERS", str(default)))
    return max(1, min(cap, os.cpu_count() or 1, int(num_tasks)))


def _init_builder_worker(raw: Any, build_kwargs: dict[str, Any]) -> None:
    set_compute_thread_limits()
    _BUILDER_STATE.clear()
    _BUILDER_STATE["raw"] = raw
    _BUILDER_STATE["build_kwargs"] = dict(build_kwargs)


def _build_trajectory(sim_id: int) -> Any:
    state = _BUILDER_STATE
    kwargs = state["build_kwargs"]
    optional_targets = set(kwargs.get("optional_target_sim_ids", ()))
    require = int(sim_id) not in optional_targets
    return parse_trajectory_bundle(
        state["raw"][int(sim_id)]["sample"],
        sample_idx=int(sim_id),
        bandwidth=float(kwargs["bandwidth"]),
        require_targets=bool(require),
    )


def _build_shard_payload(shard_id: int, sim_ids: list[int]) -> ShardPayload:
    kwargs = _BUILDER_STATE["build_kwargs"]
    trajectories = [_build_trajectory(int(sim_id)) for sim_id in sim_ids]
    laplacian = None
    if int(kwargs.get("laplacian_eig_dim", 0)) > 0:
        laplacian = [
            pack_laplacian_for_trajectory(
                traj,
                dataset_dir=kwargs["dataset_dir"],
                laplacian_eig_dim=int(kwargs["laplacian_eig_dim"]),
                laplacian_spec=str(kwargs["laplacian_spec"]),
            )
            for traj in trajectories
        ]
    return ShardPayload(
        schema_version=CACHE_SCHEMA_VERSION,
        shard_id=int(shard_id),
        sim_ids=[int(v) for v in sim_ids],
        trajectories=trajectories,
        laplacian=laplacian,
    )


def _build_shard_worker(args: tuple[int, list[int], str]) -> tuple[int, int]:
    shard_id, sim_ids, shard_path = args
    out = Path(shard_path)
    if out.is_file():
        payload = torch.load(out, map_location="cpu", weights_only=False)
        return int(shard_id), len(payload.get("sim_ids", []))
    shard = _build_shard_payload(int(shard_id), [int(v) for v in sim_ids])
    save_shard_payload(str(out), shard)
    return int(shard_id), len(shard.sim_ids)


def build_shards(
    raw: Any,
    sim_ids: list[int],
    *,
    dataset_dir: str | Path,
    split_seed: int,
    bandwidth: float,
    optional_target_sim_ids: set[int] | frozenset[int] | None = None,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    trajectories_per_shard: int = TRAJECTORIES_PER_SHARD,
) -> list[Path]:
    from pdebench.dataset.plaid_elpl_v3.paths import shard_dir, shard_path

    ordered = sorted(int(v) for v in sim_ids)
    partitions = partition_sim_ids(ordered, trajectories_per_shard=trajectories_per_shard)
    root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    root.mkdir(parents=True, exist_ok=True)

    build_kwargs = dict(
        bandwidth=float(bandwidth),
        optional_target_sim_ids=sorted(int(v) for v in (optional_target_sim_ids or set())),
        dataset_dir=str(dataset_dir),
        laplacian_eig_dim=int(laplacian_eig_dim),
        laplacian_spec=str(laplacian_spec),
    )
    tasks = [
        (
            shard_id,
            shard_sims,
            str(
                shard_path(
                    dataset_dir,
                    split_seed=split_seed,
                    shard_id=shard_id,
                    laplacian_eig_dim=laplacian_eig_dim,
                    laplacian_spec=laplacian_spec,
                )
            ),
        )
        for shard_id, shard_sims in enumerate(partitions)
    ]

    num_workers = _precompute_workers(len(tasks))
    if num_workers <= 1:
        _BUILDER_STATE.clear()
        _BUILDER_STATE["raw"] = raw
        _BUILDER_STATE["build_kwargs"] = dict(build_kwargs)
        for task in tqdm(tasks, desc="elpl_v3 shards", ncols=90):
            _build_shard_worker(task)
    else:
        print(
            f"Using {num_workers} elpl_v3 shard worker(s) for {len(tasks)} shard(s); "
            f"trajectories_per_shard={trajectories_per_shard}."
        )
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=min(num_workers, len(tasks)),
            initializer=_init_builder_worker,
            initargs=(raw, build_kwargs),
            maxtasksperchild=4,
        ) as pool:
            for _shard_id, _count in tqdm(
                pool.imap(_build_shard_worker, tasks, chunksize=1),
                total=len(tasks),
                desc="elpl_v3 shards",
                ncols=90,
            ):
                del _shard_id, _count

    return [Path(task[2]) for task in tasks]
