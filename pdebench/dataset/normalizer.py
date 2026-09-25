"""Shared mean/std normalizer used by GINOT and PLAID adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MeanStdNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(device=x.device, dtype=x.dtype) + self.mean.to(device=x.device, dtype=x.dtype)

    def to(self, device) -> "MeanStdNormalizer":
        return MeanStdNormalizer(mean=self.mean.to(device), std=self.std.to(device))


class NodeFeatureNormalizer(MeanStdNormalizer):
    """PLAID node-feature normalizer (mutable ``to``, ``from_tensors`` factory)."""

    @classmethod
    def from_tensors(cls, tensors: list[torch.Tensor]) -> "NodeFeatureNormalizer":
        if len(tensors) == 0:
            raise ValueError("Cannot build normalizer from an empty tensor list.")
        cat = torch.cat(tensors, dim=0)
        mean = cat.mean(dim=0, keepdim=True)
        std = cat.std(dim=0, keepdim=True) + 1e-8
        return cls(mean=mean, std=std)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
