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

"""Irregular-mesh GeoTransolver core (PhysicsNeMo GALE stack)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .context_projector import GlobalContextBuilder
from .gale import GALE_block
from .pn_compat import Mlp

__all__ = ["GeoTransolverCore"]


def _normalize_dim(x: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return tuple(int(v) for v in x)
    raise TypeError(f"Invalid dim specifier {x!r}")


def _normalize_tensor(
    x: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, Sequence):
        return tuple(x)
    raise TypeError("Invalid tensor structure")


class GeoTransolverCore(nn.Module):
    """PhysicsNeMo-style GeoTransolver for irregular meshes (no TE / OOD / structured)."""

    def __init__(
        self,
        functional_dim: int | tuple[int, ...],
        out_dim: int | tuple[int, ...],
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int | float = 4,
        slice_num: int = 32,
        use_te: bool = False,
        plus: bool = False,
        include_local_features: bool = False,
        concat_local_features: bool = True,
        radii: list[float] | None = None,
        neighbors_in_radius: list[int] | None = None,
        n_hidden_local: int = 32,
        state_mixing_mode: str = "weighted",
        use_geo: bool = True,
    ) -> None:
        super().__init__()
        if use_te:
            raise ValueError("use_te=True is not supported in the vendored port.")
        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]
        if n_hidden % n_head != 0:
            raise ValueError(f"n_hidden % n_head == 0 required; got {n_hidden}, {n_head}")

        functional_dims = _normalize_dim(functional_dim)
        out_dims = _normalize_dim(out_dim)
        if len(functional_dims) != len(out_dims):
            raise ValueError(
                f"functional_dim and out_dim length mismatch: {functional_dims} vs {out_dims}"
            )

        self.use_geo = bool(use_geo)
        self.include_local_features = bool(include_local_features) and self.use_geo
        self.concat_local_features = bool(concat_local_features) and self.use_geo
        self.radii = list(radii) if self.include_local_features else []
        self.n_hidden = int(n_hidden)
        self.n_hidden_local = int(n_hidden_local)

        self.context_builder: GlobalContextBuilder | None = None
        context_dim = 0
        if self.use_geo:
            self.context_builder = GlobalContextBuilder(
                functional_dims=functional_dims,
                geometry_dim=geometry_dim,
                global_dim=global_dim,
                radii=list(radii),
                neighbors_in_radius=list(neighbors_in_radius),
                n_hidden_local=n_hidden_local,
                n_hidden=n_hidden,
                n_head=n_head,
                dropout=dropout,
                slice_num=slice_num,
                use_te=False,
                plus=plus,
                include_local_features=self.include_local_features,
            )
            context_dim = self.context_builder.get_context_dim()

        self.preprocess = nn.ModuleList(
            [
                Mlp(
                    in_features=f,
                    hidden_features=n_hidden * 2,
                    out_features=n_hidden,
                    act_layer=act,
                    drop=0.0,
                    final_dropout=False,
                )
                for f in functional_dims
            ]
        )

        effective_hidden = (
            n_hidden + n_hidden_local * len(self.radii)
            if self.include_local_features and self.concat_local_features
            else n_hidden
        )
        self.effective_hidden = int(effective_hidden)

        self.blocks = nn.ModuleList(
            [
                GALE_block(
                    num_heads=n_head,
                    hidden_dim=self.effective_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    use_te=False,
                    plus=plus,
                    context_dim=context_dim,
                    state_mixing_mode=state_mixing_mode,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_mlp_out = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(self.effective_hidden), nn.Linear(self.effective_hidden, o))
                for o in out_dims
            ]
        )

    def forward(
        self,
        local_embedding: torch.Tensor | tuple[torch.Tensor, ...],
        local_positions: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        global_embedding: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        single_input = isinstance(local_embedding, torch.Tensor)
        local_embedding = _normalize_tensor(local_embedding)
        if local_positions is not None:
            local_positions = _normalize_tensor(local_positions)

        embedding_states = None
        local_embedding_bq = None
        if self.context_builder is not None:
            embedding_states, local_embedding_bq, _geo_ctx = self.context_builder.build_context(
                local_embedding, local_positions, geometry, global_embedding
            )

        x = [self.preprocess[i](le) for i, le in enumerate(local_embedding)]
        if self.concat_local_features and local_embedding_bq is not None:
            x = [torch.cat([x[i], local_embedding_bq[i]], dim=-1) for i in range(len(x))]

        for block in self.blocks:
            x = block(tuple(x), embedding_states)

        x = [self.ln_mlp_out[i](x[i]) for i in range(len(x))]
        return x[0] if single_input else tuple(x)
