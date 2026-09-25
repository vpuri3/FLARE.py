"""Trajectory parsing helpers for el-pl v3."""

from __future__ import annotations

from typing import Any

import torch

from pdebench.dataset.plaid_core import parse_plaid_elpl_trajectory
from pdebench.dataset.plaid_elpl_v3.schema import TrajectoryBundle, trajectory_from_numpy


def parse_trajectory_bundle(
    sample_bytes: bytes,
    *,
    sample_idx: int,
    bandwidth: float,
    require_targets: bool = True,
) -> TrajectoryBundle:
    del bandwidth
    payload = parse_plaid_elpl_trajectory(
        sample_bytes,
        sample_idx=sample_idx,
        require_targets=require_targets,
    )
    return trajectory_from_numpy(
        sim_id=int(payload["sample_id"]),
        pos=payload["pos"],
        cells=payload["cells"],
        sdf=payload["sdf"],
        proj=payload["proj"],
        u_traj=payload["u_traj"],
        times=payload["times"],
        boundary_ids=payload["boundary_ids"],
        boundary_tags=tuple(payload["boundary_tags"]),
        timestep_list=list(payload["timestep_list"]),
    )


def load_shard_payload(path: str) -> Any:
    from pdebench.dataset.plaid_elpl_v3.schema import ShardPayload

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return ShardPayload.from_dict(payload)


def save_shard_payload(path: str, shard: Any) -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(shard.to_dict(), tmp)
    tmp.replace(out)
