"""Runtime dataset for el-pl v3 transition manifest."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from pdebench.dataset.plaid_elpl_v3.assemble import assemble_transition_graph
from pdebench.dataset.plaid_elpl_v3.norm import ElPlNormStats
from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload
from pdebench.dataset.plaid_elpl_v3.schema import ShardPayload


class ElPlShardCache:
    def __init__(self, *, max_open: int = 8):
        self.max_open = max(1, int(max_open))
        self._cache: OrderedDict[int, ShardPayload] = OrderedDict()

    def get(self, shard_id: int, shard_path: str | Path) -> ShardPayload:
        key = int(shard_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        payload = load_shard_payload(str(shard_path))
        self._cache[key] = payload
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_open:
            self._cache.popitem(last=False)
        return payload


class ElPlTransitionDataset(Dataset):
    """Transition indexer over the shared el-pl trajectory store.

    ``__len__`` is ``transition_length(num_sims, num_steps)`` for the split
    (``#sims × (T − 1)``), driven by the transition manifest rows.
    """
    def __init__(
        self,
        *,
        manifest: pd.DataFrame,
        sim_to_shard: pd.DataFrame,
        shard_root: str | Path,
        stats: ElPlNormStats,
        use_sdf_features: bool,
        graph_backend: str,
        target_fields: tuple[str, ...],
        bandwidth: float,
        laplacian_eig_dim: int,
        laplacian_spec: str,
        max_open_shards: int | None = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.sim_to_shard = sim_to_shard.set_index("sim_id")
        self.shard_root = Path(shard_root)
        self.stats = stats
        self.use_sdf_features = bool(use_sdf_features)
        self.graph_backend = str(graph_backend)
        self.target_fields = tuple(target_fields)
        self.bandwidth = float(bandwidth)
        self.laplacian_eig_dim = int(laplacian_eig_dim)
        self.laplacian_spec = str(laplacian_spec)
        max_open = max_open_shards
        if max_open is None:
            max_open = int(os.environ.get("PLAID_ELPL_OPEN_SHARD_LRU", "8"))
        self.shard_cache = ElPlShardCache(max_open=max_open)

    def __len__(self) -> int:
        return len(self.manifest)

    def _shard_path(self, shard_id: int) -> Path:
        return self.shard_root / f"shard_{int(shard_id):04d}.pt"

    def __getitem__(self, idx: int | torch.Tensor) -> Any:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        row = self.manifest.iloc[int(idx)]
        sim_id = int(row["sim_id"])
        step_idx = int(row["step_idx"])
        mapping = self.sim_to_shard.loc[sim_id]
        shard_id = int(mapping["shard_id"])
        local_idx = int(mapping["local_idx"])
        shard = self.shard_cache.get(shard_id, self._shard_path(shard_id))
        traj = shard.trajectories[local_idx]
        lap = None
        if shard.laplacian is not None:
            lap = shard.laplacian[local_idx]
        return assemble_transition_graph(
            traj,
            step_idx=step_idx,
            stats=self.stats,
            use_sdf_features=self.use_sdf_features,
            graph_backend=self.graph_backend,
            target_fields=self.target_fields,
            laplacian=lap,
            bandwidth=self.bandwidth,
            laplacian_eig_dim=self.laplacian_eig_dim,
            laplacian_spec=self.laplacian_spec,
        )
