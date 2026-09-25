from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch


class SampleKind(str, Enum):
    STATIC = "static"
    TRANSITION = "transition"
    TERMINAL = "terminal"


@dataclass
class FeatureRequest:
    edges: bool = False
    boundary: bool = False
    laplacian_k: int = 0
    laplacian_spec: str = "graph"
    pos_domain: bool = False


@dataclass
class LossSpec:
    mask: Optional[str] = None
    extras: tuple[str, ...] = ()


@dataclass
class Sample:
    pos: torch.Tensor
    y: torch.Tensor
    sample_id: str
    kind: SampleKind
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    feats: Optional[torch.Tensor] = None
    boundary_pos: Optional[torch.Tensor] = None
    state_in: Optional[torch.Tensor] = None
    context: Optional[torch.Tensor] = None
    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    laplacian_eig: Optional[torch.Tensor] = None
    laplacian_eigvals: Optional[torch.Tensor] = None
    extras: dict[str, Any] = field(default_factory=dict)
