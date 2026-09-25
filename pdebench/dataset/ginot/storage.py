from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from pdebench.dataset.ginot.types import MANIFEST_NAME


class ShardedMicroPucFixedSequence:
    """Lazy indexable sequence over sharded fixed Micro-PUC arrays."""

    def __init__(self, root: str | Path, key: str, cache_size: int = 32):
        self.root = Path(root)
        self.key = str(key)
        self.manifest = json.loads((self.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.shard_size = int(self.manifest["shard_size"])
        self.num_samples = int(self.manifest["num_samples"])
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        shard = torch.load(self.root / "shards" / f"shard_{int(shard_id):06d}.pt", map_location="cpu", weights_only=True)
        self._cache[shard_id] = shard
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0:
            idx += self.num_samples
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)
        shard_id = idx // self.shard_size
        offset = idx % self.shard_size
        shard = self._load_shard(shard_id)
        return shard[self.key][offset]
