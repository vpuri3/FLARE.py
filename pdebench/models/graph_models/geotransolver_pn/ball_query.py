# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-PyTorch ball query matching PhysicsNeMo BQWarp's pad-to-K contract.

Uses a vectorized GPU path: dense ``cdist``+``topk`` for modest N, and a
grid-hashed candidate gather for larger clouds (avoids full ``N²`` topk).

``torch.compile`` is disabled here (same rationale as ``knn_graph``): neighbor
construction is data-dependent and must not graph-break the training step.
"""
from __future__ import annotations

import torch
from torch import nn

# Below this Q*P product, dense cdist+topk is typically fastest on H100.
# Bumper meshes are ~14k nodes (≈2e8 pairs); dense stays faster than the grid path.
_DENSE_PAIR_LIMIT = 400_000_000


def _gather_neighbors(
    points: torch.Tensor,
    idx: torch.Tensor,
    ok: torch.Tensor,
) -> torch.Tensor:
    """Gather ``points`` by ``idx``; zero slots where ``ok`` is False."""
    bsz, n_q, k = idx.shape
    safe_idx = torch.where(ok, idx, torch.zeros_like(idx))
    batch_ix = torch.arange(bsz, device=points.device)[:, None, None].expand(bsz, n_q, k)
    neighbor_coords = points[batch_ix, safe_idx]
    return neighbor_coords * ok.to(dtype=points.dtype).unsqueeze(-1)


def _dense_radius_search(
    queries: torch.Tensor,
    points: torch.Tensor,
    radius: float,
    k: int,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, n_q, _ = queries.shape
    n_p = points.shape[1]
    k_eff = min(k, n_p)

    dists = torch.cdist(queries, points)
    in_ball = dists <= float(radius)
    if key_mask is not None:
        in_ball = in_ball & key_mask[:, None, :].to(dtype=torch.bool)
    if query_mask is not None:
        in_ball = in_ball & query_mask[:, :, None].to(dtype=torch.bool)

    sort_key = dists.masked_fill(~in_ball, float("inf"))
    _, idx = torch.topk(sort_key, k=k_eff, dim=-1, largest=False)
    if k > n_p:
        pad = idx.new_zeros(bsz, n_q, k - n_p)
        idx = torch.cat([idx, pad], dim=-1)
    ok = torch.gather(in_ball, 2, idx.clamp(max=n_p - 1))
    if k > n_p:
        ok = torch.cat([ok[:, :, :n_p], ok.new_zeros(bsz, n_q, k - n_p)], dim=-1)
    mapping = torch.where(ok, idx, idx.new_full(idx.shape, -1))
    neighbors = _gather_neighbors(points, idx.clamp(max=max(n_p - 1, 0)), ok)
    return mapping, neighbors


def _grid_radius_search(
    queries: torch.Tensor,
    points: torch.Tensor,
    radius: float,
    k: int,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid-hash ball query: distance only against points in the 3³ neighborhood."""
    bsz, n_q, _ = queries.shape
    n_p = points.shape[1]
    device = queries.device
    cell_size = float(radius)

    # Shared origin per batch element.
    origin = points.amin(dim=1, keepdim=True)  # (B,1,3)
    p_cell = torch.div(points - origin, cell_size, rounding_mode="floor").to(torch.int64)
    q_cell = torch.div(queries - origin, cell_size, rounding_mode="floor").to(torch.int64)
    # Shift to non-negative.
    shift = torch.minimum(p_cell.amin(dim=1), q_cell.amin(dim=1))  # (B,3)
    p_cell = p_cell - shift[:, None, :]
    q_cell = q_cell - shift[:, None, :]
    extents = torch.maximum(p_cell.amax(dim=1), q_cell.amax(dim=1)) + 1  # (B,3)

    # Process each batch row (B is almost always 1 for bumper); keep ops vectorized in N.
    mappings = []
    neighbors_out = []
    offs = queries.new_tensor(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=torch.int64,
    )  # (27, 3)

    for b in range(bsz):
        ex, ey, ez = (int(v) for v in extents[b].tolist())
        n_cells = max(ex * ey * ez, 1)

        def _hash(cell: torch.Tensor) -> torch.Tensor:
            return cell[..., 0] * (ey * ez) + cell[..., 1] * ez + cell[..., 2]

        ph = _hash(p_cell[b])  # (P,)
        if key_mask is not None:
            ph = torch.where(key_mask[b], ph, ph.new_full((), fill_value=n_cells))

        order = torch.argsort(ph)
        ph_sorted = ph[order]
        # searchsorted ranges per cell id
        cell_ids = torch.arange(n_cells, device=device, dtype=ph.dtype)
        starts = torch.searchsorted(ph_sorted, cell_ids, right=False)
        ends = torch.searchsorted(ph_sorted, cell_ids, right=True)

        qc = q_cell[b][:, None, :] + offs[None, :, :]  # (Q, 27, 3)
        qc[..., 0].clamp_(0, max(ex - 1, 0))
        qc[..., 1].clamp_(0, max(ey - 1, 0))
        qc[..., 2].clamp_(0, max(ez - 1, 0))
        qh = _hash(qc)  # (Q, 27)

        # Cap candidates per query from the 27 cells.
        # Gather via per-offset ranges would be variable-length; instead take up to
        # ``cand_cap`` points by sampling cell contents with a fixed window.
        cand_cap = max(k * 8, 64)
        # Build candidate index list: for each of 27 cells, take up to cand_cap/27 points.
        per_cell = max(cand_cap // 27, k)
        cand = points.new_zeros(n_q, 27 * per_cell, dtype=torch.long)
        cand_ok = torch.zeros(n_q, 27 * per_cell, dtype=torch.bool, device=device)
        for oi in range(27):
            cells = qh[:, oi]  # (Q,)
            st = starts[cells]
            en = ends[cells]
            # Take first ``per_cell`` indices in each cell (sorted order ≈ spatial).
            offsets = torch.arange(per_cell, device=device)[None, :]  # (1, per_cell)
            idx_in_sorted = st[:, None] + offsets
            valid = idx_in_sorted < en[:, None]
            # Clamp for gather then mask.
            idx_clamped = idx_in_sorted.clamp(max=max(n_p - 1, 0))
            point_idx = order[idx_clamped]
            sl = slice(oi * per_cell, (oi + 1) * per_cell)
            cand[:, sl] = point_idx
            cand_ok[:, sl] = valid

        if query_mask is not None:
            cand_ok = cand_ok & query_mask[b][:, None]

        # Distances to candidates only: (Q, C)
        pts = points[b]  # (P, 3)
        cand_pts = pts[cand.clamp(max=max(n_p - 1, 0))]  # (Q, C, 3)
        dists = (queries[b][:, None, :] - cand_pts).pow(2).sum(-1).sqrt()
        dists = dists.masked_fill(~cand_ok, float("inf"))
        in_ball = cand_ok & (dists <= float(radius))
        sort_key = dists.masked_fill(~in_ball, float("inf"))
        k_eff = min(k, sort_key.shape[-1])
        _, local_idx = torch.topk(sort_key, k=k_eff, dim=-1, largest=False)
        if k > k_eff:
            local_idx = torch.cat(
                [local_idx, local_idx.new_zeros(n_q, k - k_eff)],
                dim=-1,
            )
        ok = torch.gather(in_ball, 1, local_idx.clamp(max=sort_key.shape[-1] - 1))
        if k > k_eff:
            ok = torch.cat([ok[:, :k_eff], ok.new_zeros(n_q, k - k_eff)], dim=-1)
        chosen = torch.gather(cand, 1, local_idx.clamp(max=cand.shape[-1] - 1))
        mapping_b = torch.where(ok, chosen, chosen.new_full(chosen.shape, -1))
        neigh_b = pts[chosen.clamp(max=max(n_p - 1, 0))] * ok.unsqueeze(-1).to(pts.dtype)
        mappings.append(mapping_b)
        neighbors_out.append(neigh_b)

    mapping = torch.stack(mappings, dim=0)
    neighbors = torch.stack(neighbors_out, dim=0)
    return mapping, neighbors


@torch.compiler.disable
def radius_search(
    queries: torch.Tensor,
    points: torch.Tensor,
    radius: float,
    max_points: int | None,
    *,
    return_points: bool = True,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Find points within ``radius`` of each query; pad/truncate to ``max_points``."""
    if queries.ndim != 3 or points.ndim != 3:
        raise ValueError(
            f"queries/points must be (B, N, 3); got {tuple(queries.shape)}, {tuple(points.shape)}"
        )
    if queries.shape[-1] != 3 or points.shape[-1] != 3:
        raise ValueError("Last dimension of queries and points must be 3.")
    if max_points is None or int(max_points) <= 0:
        raise ValueError("max_points (neighbors_in_radius) must be a positive int.")

    k = int(max_points)
    n_q = queries.shape[1]
    n_p = points.shape[1]
    pairs = n_q * n_p
    if pairs <= _DENSE_PAIR_LIMIT or n_q == 0 or n_p == 0:
        mapping, neighbors = _dense_radius_search(queries, points, radius, k, query_mask, key_mask)
    else:
        mapping, neighbors = _grid_radius_search(queries, points, radius, k, query_mask, key_mask)

    if not return_points:
        return mapping
    return mapping, neighbors


class BQWarp(nn.Module):
    """BQWarp-compatible ball query (pure PyTorch)."""

    def __init__(self, radius: float = 0.25, neighbors_in_radius: int | None = 10):
        super().__init__()
        self.radius = float(radius)
        self.neighbors_in_radius = neighbors_in_radius

    def forward(
        self,
        x: torch.Tensor,
        p_grid: torch.Tensor,
        reverse_mapping: bool = True,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if p_grid.shape[-1] != x.shape[-1] or x.shape[-1] != 3:
            raise ValueError("The last dimension of p_grid and x must be 3")
        if p_grid.ndim != 3:
            raise ValueError("p_grid must be 3D (B, N, 3) for the irregular-mesh path")

        if reverse_mapping:
            queries, points = x, p_grid
            q_mask, k_mask = query_mask, key_mask
        else:
            queries, points = p_grid, x
            q_mask, k_mask = key_mask, query_mask

        return radius_search(
            queries,
            points,
            self.radius,
            self.neighbors_in_radius,
            return_points=True,
            query_mask=q_mask,
            key_mask=k_mask,
        )
