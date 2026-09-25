"""Dataclasses and serialization for el-pl v3 shard payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pdebench.dataset.plaid_elpl_v3.constants import CACHE_SCHEMA_VERSION


@dataclass
class LaplacianBundle:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "eigenvalues": self.eigenvalues.detach().cpu(),
            "eigenvectors": self.eigenvectors.detach().cpu(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LaplacianBundle":
        return cls(
            eigenvalues=torch.as_tensor(payload["eigenvalues"], dtype=torch.float32),
            eigenvectors=torch.as_tensor(payload["eigenvectors"], dtype=torch.float32),
        )


@dataclass
class TrajectoryBundle:
    sim_id: int
    pos: torch.Tensor
    cells: torch.Tensor
    sdf: torch.Tensor
    proj: torch.Tensor
    u_traj: torch.Tensor
    times: torch.Tensor
    boundary_ids: torch.Tensor
    boundary_tags: tuple[str, ...]
    edge_index: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    timestep_list: list[float] = field(default_factory=list)

    def ensure_edges(self, *, bandwidth: float) -> None:
        if self.edge_index is not None and self.edge_attr is not None:
            return
        from pdebench.dataset.plaid_core import build_edge_attr, build_edge_index_from_cells

        edge_index = build_edge_index_from_cells(self.cells.detach().cpu().numpy())
        edge_attr = build_edge_attr(self.pos, edge_index, bandwidth=float(bandwidth)).float()
        self.edge_index = edge_index.long()
        self.edge_attr = edge_attr.float()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos.detach().cpu(),
            "cells": self.cells.detach().cpu(),
            "sdf": self.sdf.detach().cpu(),
            "proj": self.proj.detach().cpu(),
            "U": self.u_traj.detach().cpu(),
            "times": self.times.detach().cpu(),
            "boundary_ids": self.boundary_ids.detach().cpu(),
            "boundary_tags": list(self.boundary_tags),
            "timestep_list": list(self.timestep_list),
        }

    @classmethod
    def from_dict(cls, sim_id: int, payload: dict[str, Any]) -> "TrajectoryBundle":
        u = payload.get("U", payload.get("u_traj"))
        edge_index = payload.get("edge_index")
        edge_attr = payload.get("edge_attr")
        return cls(
            sim_id=int(sim_id),
            pos=torch.as_tensor(payload["pos"], dtype=torch.float32),
            cells=torch.as_tensor(payload["cells"], dtype=torch.long),
            sdf=torch.as_tensor(payload["sdf"], dtype=torch.float32),
            proj=torch.as_tensor(payload["proj"], dtype=torch.float32),
            u_traj=torch.as_tensor(u, dtype=torch.float32),
            times=torch.as_tensor(payload["times"], dtype=torch.float32),
            boundary_ids=torch.as_tensor(payload["boundary_ids"], dtype=torch.long),
            boundary_tags=tuple(payload.get("boundary_tags", ())),
            edge_index=None if edge_index is None else torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr=None if edge_attr is None else torch.as_tensor(edge_attr, dtype=torch.float32),
            timestep_list=[float(v) for v in payload.get("timestep_list", payload["times"].tolist())],
        )


@dataclass
class ShardPayload:
    schema_version: int
    shard_id: int
    sim_ids: list[int]
    trajectories: list[TrajectoryBundle]
    laplacian: list[LaplacianBundle | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        lap = None
        if self.laplacian is not None:
            lap = [item.to_dict() if item is not None else None for item in self.laplacian]
        return {
            "schema_version": int(self.schema_version),
            "shard_id": int(self.shard_id),
            "sim_ids": [int(v) for v in self.sim_ids],
            "trajectories": [traj.to_dict() for traj in self.trajectories],
            "laplacian": lap,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardPayload":
        sim_ids = [int(v) for v in payload["sim_ids"]]
        trajectories = [
            TrajectoryBundle.from_dict(sim_id, traj_payload)
            for sim_id, traj_payload in zip(sim_ids, payload["trajectories"], strict=True)
        ]
        laplacian = None
        if payload.get("laplacian") is not None:
            laplacian = [
                LaplacianBundle.from_dict(item) if item is not None else None
                for item in payload["laplacian"]
            ]
        return cls(
            schema_version=int(payload.get("schema_version", CACHE_SCHEMA_VERSION)),
            shard_id=int(payload["shard_id"]),
            sim_ids=sim_ids,
            trajectories=trajectories,
            laplacian=laplacian,
        )


@dataclass(frozen=True)
class ManifestRow:
    global_idx: int
    sim_id: int
    step_idx: int
    t0: float
    t1: float


def trajectory_from_numpy(
    *,
    sim_id: int,
    pos: np.ndarray,
    cells: np.ndarray,
    sdf: np.ndarray,
    proj: np.ndarray,
    u_traj: np.ndarray,
    times: np.ndarray,
    boundary_ids: np.ndarray,
    boundary_tags: tuple[str, ...],
    timestep_list: list[float],
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
) -> TrajectoryBundle:
    return TrajectoryBundle(
        sim_id=int(sim_id),
        pos=torch.from_numpy(pos).float(),
        cells=torch.from_numpy(cells).long(),
        sdf=torch.from_numpy(sdf).float(),
        proj=torch.from_numpy(proj).float(),
        u_traj=torch.from_numpy(u_traj).float(),
        times=torch.from_numpy(times).float(),
        boundary_ids=torch.from_numpy(boundary_ids).long(),
        boundary_tags=boundary_tags,
        edge_index=edge_index.long() if edge_index is not None else None,
        edge_attr=edge_attr.float() if edge_attr is not None else None,
        timestep_list=list(timestep_list),
    )
