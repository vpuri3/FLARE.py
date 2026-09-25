"""Runtime dataset for el-pl terminal manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from pdebench.dataset.plaid_elpl_terminal.assemble import assemble_terminal_graph
from pdebench.dataset.plaid_elpl_terminal.norm import ElPlTerminalNormStats, YFieldNormalizer
from pdebench.dataset.plaid_elpl_v3.dataset import ElPlShardCache


class ElPlTerminalDataset(Dataset):
    """Terminal indexer over the shared el-pl trajectory store.

    ``__len__`` is ``terminal_length(num_sims)`` (one sample per simulation).
    """
    def __init__(
        self,
        *,
        manifest: pd.DataFrame,
        sim_to_shard: pd.DataFrame,
        shard_root: str | Path,
        stats: ElPlTerminalNormStats,
        runtime_y_normalizer: YFieldNormalizer,
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
        self.runtime_y_normalizer = runtime_y_normalizer
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
        mapping = self.sim_to_shard.loc[sim_id]
        shard_id = int(mapping["shard_id"])
        local_idx = int(mapping["local_idx"])
        shard = self.shard_cache.get(shard_id, self._shard_path(shard_id))
        traj = shard.trajectories[local_idx]
        lap = None
        if shard.laplacian is not None:
            lap = shard.laplacian[local_idx]
        return assemble_terminal_graph(
            traj,
            stats=self.stats,
            runtime_y_normalizer=self.runtime_y_normalizer,
            use_sdf_features=self.use_sdf_features,
            graph_backend=self.graph_backend,
            target_fields=self.target_fields,
            laplacian=lap,
            bandwidth=self.bandwidth,
            laplacian_eig_dim=self.laplacian_eig_dim,
            laplacian_spec=self.laplacian_spec,
        )
