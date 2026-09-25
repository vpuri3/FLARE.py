"""GeoTransolver adapter: pack GINOT flat graphs into PhysicsNeMo GALE core.

Default path is the vendored irregular-mesh GeoTransolver with multi-scale
ball-query local features (``include_local_features=True``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .geotransolver_pn import GeoTransolverCore
from .utils import graph_node_input

__all__ = [
    "GeoTransolverConfig",
    "GeoTransolverModel",
]

# Far pad for unused slots so radius search never pairs them with real nodes.
_GEO_PAD = 1.0e4


@dataclass
class GeoTransolverConfig:
    model: str = "geo_transolver"
    num_blocks: int = 2
    channel_dim: int = 128
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False
    mlp_ratio: float = 4.0
    num_slices: int = 128
    geometry_dim: int = 3
    global_dim: Optional[int] = None
    use_geo: bool = True
    include_local_features: bool = True
    concat_local_features: bool = True
    ball_radii: tuple[float, ...] = (0.05, 0.25)
    ball_k: int = 16
    # Per-radius neighbor counts; empty => broadcast ``ball_k`` to each radius.
    ball_ks: tuple[int, ...] = (8, 32)
    n_hidden_local: int = 32
    state_mixing_mode: str = "weighted"


def _as_xyz(pos: torch.Tensor, geometry_dim: int = 3) -> torch.Tensor:
    """Pad / truncate coordinates to ``geometry_dim`` (BQWarp requires 3)."""
    if pos.ndim != 2:
        raise ValueError(f"pos must be [N, D], got {tuple(pos.shape)}")
    d = pos.shape[-1]
    if d == geometry_dim:
        return pos
    if d > geometry_dim:
        return pos[:, :geometry_dim]
    pad = pos.new_zeros(pos.shape[0], geometry_dim - d)
    return torch.cat([pos, pad], dim=-1)


@torch.compiler.disable
def _pack_batch(
    values: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack flat ``[N, C]`` into padded ``[B, Nmax, C]`` with a bool mask."""
    if batch_index.ndim != 1 or batch_index.shape[0] != values.shape[0]:
        raise ValueError(
            f"batch_index must be [N] matching values; got {tuple(batch_index.shape)} vs {tuple(values.shape)}"
        )
    device = values.device
    batch_ids = batch_index.to(device=device, dtype=torch.long)
    num_graphs = int(batch_ids.max().item()) + 1 if batch_ids.numel() else 1
    counts = torch.bincount(batch_ids, minlength=num_graphs)
    n_max = int(counts.max().item()) if counts.numel() else 0
    feat_dim = values.shape[-1]
    packed = values.new_full((num_graphs, n_max, feat_dim), float(pad_value))
    mask = torch.zeros(num_graphs, n_max, dtype=torch.bool, device=device)

    sort_idx = torch.argsort(batch_ids, stable=True)
    sorted_ids = batch_ids[sort_idx]
    change = torch.ones_like(sorted_ids, dtype=torch.bool)
    change[1:] = sorted_ids[1:] != sorted_ids[:-1]
    start_pos = torch.zeros_like(sorted_ids)
    start_pos[change] = torch.arange(sorted_ids.shape[0], device=device)[change]
    start_pos = start_pos.cummax(0).values
    within = torch.arange(sorted_ids.shape[0], device=device) - start_pos
    within_unsort = torch.empty_like(within)
    within_unsort[sort_idx] = within

    packed[batch_ids, within_unsort] = values
    mask[batch_ids, within_unsort] = True
    return packed, mask, counts


@torch.compiler.disable
def _unpack_batch(packed: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    parts = [packed[b, : int(n)] for b, n in enumerate(counts.tolist()) if int(n) > 0]
    if not parts:
        return packed.new_zeros((0, packed.shape[-1]))
    return torch.cat(parts, dim=0)


@torch.compiler.disable
def _extract_global_embedding(
    feats: torch.Tensor | None,
    batch_index: torch.Tensor,
    global_dim: int,
    num_graphs: int,
) -> torch.Tensor | None:
    """Take last ``global_dim`` feat channels (constant per sample) as ``[B, 1, G]``."""
    if feats is None or feats.numel() == 0 or feats.shape[-1] < global_dim:
        return None
    globals_flat = feats[:, -global_dim:]
    first = torch.zeros(num_graphs, dtype=torch.long, device=batch_index.device)
    order = torch.argsort(batch_index, stable=True)
    sorted_ids = batch_index[order]
    change = torch.ones_like(sorted_ids, dtype=torch.bool)
    change[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = order[change]
    ids = batch_index[starts]
    first[ids] = starts
    return globals_flat[first].unsqueeze(1)


@torch.compiler.disable
def _prepare_gale_batch(
    *,
    node_input: torch.Tensor,
    pos: torch.Tensor,
    feats: torch.Tensor | None,
    batch_index: torch.Tensor | None,
    geometry_dim: int,
    global_dim: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Host-side batch prep (disabled under compile to avoid graph breaks).

    Returns ``(functional_b, geometry_b, global_embedding, counts_or_none, single_graph)``.
    When ``single_graph`` is True, ``counts_or_none`` is None and outputs are ``[1,N,*]``.
    """
    geometry = _as_xyz(pos, geometry_dim)

    is_single = batch_index is None or batch_index.numel() == 0
    if not is_single:
        is_single = int(batch_index.max().item()) == 0

    if is_single:
        functional_b = node_input.unsqueeze(0)
        geometry_b = geometry.unsqueeze(0)
        global_embedding = None
        if global_dim is not None and feats is not None and feats.shape[-1] >= global_dim:
            global_embedding = feats[0, -global_dim:].view(1, 1, global_dim)
        return functional_b, geometry_b, global_embedding, None, True

    assert batch_index is not None
    batch_index = batch_index.to(device=node_input.device, dtype=torch.long)
    functional_b, _, counts = _pack_batch(node_input, batch_index)
    geometry_b, _, _ = _pack_batch(geometry, batch_index, pad_value=_GEO_PAD)
    global_embedding = None
    if global_dim is not None:
        global_embedding = _extract_global_embedding(
            feats, batch_index, global_dim, functional_b.shape[0]
        )
    return functional_b, geometry_b, global_embedding, counts, False


class GeoTransolverModel(nn.Module):
    """Thin adapter around vendored PhysicsNeMo ``GeoTransolverCore``."""

    def __init__(self, config: GeoTransolverConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        space_dim = int(metadata.get("space_dim", min(3, in_dim)))
        self.space_dim = space_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        geometry_dim = int(getattr(config, "geometry_dim", 3) or 3)
        if geometry_dim != 3:
            raise ValueError(f"geometry_dim must be 3 for ball-query GALE; got {geometry_dim}")
        self.geometry_dim = geometry_dim

        global_dim = getattr(config, "global_dim", None)
        self.global_dim = int(global_dim) if global_dim is not None else None

        raw_radii = config.ball_radii or (0.05, 0.25)
        if isinstance(raw_radii, str):
            raw_radii = tuple(p.strip() for p in raw_radii.split(",") if p.strip())
        radii = tuple(float(r) for r in raw_radii)

        raw_ks = getattr(config, "ball_ks", ()) or ()
        if isinstance(raw_ks, str):
            raw_ks = tuple(p.strip() for p in raw_ks.split(",") if p.strip())
        ball_ks = tuple(int(k) for k in raw_ks)
        if not ball_ks:
            ball_ks = tuple(int(config.ball_k) for _ in radii)
        if len(ball_ks) != len(radii):
            raise ValueError(f"ball_ks length {len(ball_ks)} != ball_radii length {len(radii)}")

        self.use_geo = bool(getattr(config, "use_geo", True))
        include_local = bool(getattr(config, "include_local_features", True)) and self.use_geo
        concat_local = bool(getattr(config, "concat_local_features", True)) and self.use_geo
        act = "gelu" if config.act is None else str(config.act)

        self.core = GeoTransolverCore(
            functional_dim=in_dim,
            out_dim=out_dim,
            geometry_dim=geometry_dim,
            global_dim=self.global_dim,
            n_layers=int(config.num_blocks),
            n_hidden=int(config.channel_dim),
            dropout=0.0,
            n_head=int(config.num_heads),
            act=act,
            mlp_ratio=float(config.mlp_ratio),
            slice_num=int(config.num_slices),
            use_te=False,
            plus=False,
            include_local_features=include_local,
            concat_local_features=concat_local,
            radii=list(radii),
            neighbors_in_radius=list(ball_ks),
            n_hidden_local=int(getattr(config, "n_hidden_local", 32)),
            state_mixing_mode=str(getattr(config, "state_mixing_mode", "weighted")),
            use_geo=self.use_geo,
        )
        self.include_local_features = include_local
        self.concat_local_features = concat_local
        self.ball_radii = radii
        self.ball_ks = ball_ks

    def forward(
        self,
        data=None,
        *,
        pos: torch.Tensor | None = None,
        feats: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        edge_index=None,
        edge_attr=None,
        mask=None,
        **kwargs,
    ):
        del edge_index, edge_attr, mask, kwargs  # GALE path does not use mesh edges.
        if data is not None:
            raise ValueError(f"Unsupported graph data type for GeoTransolverModel: {type(data)}")
        if pos is None:
            raise ValueError("GeoTransolver requires pos.")

        node_input = graph_node_input(pos, feats)
        if node_input.shape[-1] != self.in_dim:
            raise ValueError(
                f"GeoTransolver expected node features with {self.in_dim} channels, "
                f"got {node_input.shape[-1]}."
            )

        # Keep inputs on the module device (no host sync).
        device = self.core.preprocess[0].layers[0].weight.device
        if node_input.device != device:
            pos = pos.to(device=device, non_blocking=True)
            if feats is not None:
                feats = feats.to(device=device, non_blocking=True)
            if batch_index is not None:
                batch_index = batch_index.to(device=device, non_blocking=True)
            node_input = graph_node_input(pos, feats)

        functional_b, geometry_b, global_embedding, counts, single = _prepare_gale_batch(
            node_input=node_input,
            pos=pos,
            feats=feats,
            batch_index=batch_index,
            geometry_dim=self.geometry_dim,
            global_dim=self.global_dim,
        )
        # Same tensor for positions and geometry ⇒ multi-scale BQ runs once per radius.
        out_b = self.core(
            local_embedding=functional_b,
            local_positions=geometry_b,
            global_embedding=global_embedding,
            geometry=geometry_b,
        )
        if single:
            return out_b[0]
        return _unpack_batch(out_b, counts)
