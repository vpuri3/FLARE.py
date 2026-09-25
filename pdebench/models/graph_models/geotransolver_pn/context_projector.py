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

"""Context projectors and multi-scale geometric feature extractors (irregular mesh)."""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .ball_query import BQWarp
from .pn_compat import Mlp, _compute_slices_from_projections, _project_input

__all__ = [
    "ContextProjector",
    "GeometricFeatureProcessor",
    "MultiScaleFeatureExtractor",
    "GlobalContextBuilder",
]


class _SliceToContextMixin:
    plus: bool

    def _init_slice_components(
        self,
        dim_head: int,
        slice_num: int,
        heads: int,
        use_te: bool,
        plus: bool,
    ) -> None:
        del use_te
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)
        if plus:
            self.proj_temperature = nn.Sequential(
                nn.Linear(dim_head, slice_num),
                nn.GELU(),
                nn.Linear(slice_num, 1),
                nn.GELU(),
            )

    def _compute_slices(
        self,
        slice_projections: torch.Tensor,
        fx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proj_temp = getattr(self, "proj_temperature", None) if self.plus else None
        return _compute_slices_from_projections(
            slice_projections,
            fx,
            self.temperature,
            self.plus,
            proj_temperature=proj_temp,
        )


class ContextProjector(_SliceToContextMixin, nn.Module):
    """Project context features onto physics slices ``(B, H, S, D)``."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = False,
        plus: bool = False,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        del concrete_dropout
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.use_te = False
        self.dropout = nn.Dropout(dropout)
        self.in_project_x = nn.Linear(dim, inner_dim)
        if not plus:
            self.in_project_fx = nn.Linear(dim, inner_dim)
        self.softmax = nn.Softmax(dim=-1)
        self._init_slice_components(dim_head, slice_num, heads, False, plus)
        self.output_dropout = None

    def project_input_onto_slices(self, x: torch.Tensor):
        fx = None if self.plus else self.in_project_fx
        return _project_input(
            x,
            self.in_project_x,
            self.heads,
            self.dim_head,
            "B N (H D) -> B N H D",
            project_fx=fx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.plus:
            projected_x = self.project_input_onto_slices(x)
            feature_projection = projected_x
        else:
            projected_x, feature_projection = self.project_input_onto_slices(x)
        slice_projections = self.in_project_slice(projected_x)
        _, slice_tokens = self._compute_slices(slice_projections, feature_projection)
        return slice_tokens


class GeometricFeatureProcessor(nn.Module):
    """Single-scale BQWarp + MLP over flattened neighbor coordinates."""

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.bq_warp = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)
        self.mlp = Mlp(
            in_features=feature_dim * neighbors_in_radius,
            hidden_features=[hidden_dim, hidden_dim // 2],
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(self, query_points: torch.Tensor, key_features: torch.Tensor) -> torch.Tensor:
        _, neighbors = self.bq_warp(query_points, key_features)
        neighbors_flat = rearrange(neighbors, "b n k c -> b n (k c)")
        return torch.nn.functional.tanh(self.mlp(neighbors_flat))


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-radius geometric processors + per-scale context tokenizers."""

    def __init__(
        self,
        geometry_dim: int,
        radii: list[float],
        neighbors_in_radius: list[int],
        hidden_dim: int,
        n_head: int,
        dim_head: int,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = False,
        plus: bool = False,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.num_scales = len(radii)
        self.processors = nn.ModuleList(
            [
                GeometricFeatureProcessor(radii[i], neighbors_in_radius[i], geometry_dim, hidden_dim)
                for i in range(self.num_scales)
            ]
        )
        self.tokenizers = nn.ModuleList(
            [
                ContextProjector(
                    hidden_dim,
                    n_head,
                    dim_head,
                    dropout,
                    slice_num,
                    use_te=False,
                    plus=plus,
                    concrete_dropout=False,
                )
                for _ in range(self.num_scales)
            ]
        )

    def extract_context_features(
        self,
        spatial_coords: torch.Tensor,
        geometry: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            tokenizer(processor(spatial_coords, geometry))
            for processor, tokenizer in zip(self.processors, self.tokenizers)
        ]

    def extract_local_features(
        self,
        spatial_coords: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [processor(geometry, spatial_coords) for processor in self.processors],
            dim=-1,
        )

    def extract_context_and_local(
        self,
        spatial_coords: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run each scale once when spatial==geometry (bumper / GALE default)."""
        # Same storage ⇒ BQ(query=geo, key=geo) is identical in both directions.
        same = spatial_coords is geometry
        context_feats: list[torch.Tensor] = []
        local_parts: list[torch.Tensor] = []
        for processor, tokenizer in zip(self.processors, self.tokenizers):
            ctx_feat = processor(spatial_coords, geometry)
            context_feats.append(tokenizer(ctx_feat))
            if same:
                local_parts.append(ctx_feat)
            else:
                local_parts.append(processor(geometry, spatial_coords))
        return context_feats, torch.cat(local_parts, dim=-1)


class GlobalContextBuilder(nn.Module):
    """Build concatenated GALE context and optional multi-scale local features."""

    def __init__(
        self,
        functional_dims: tuple[int, ...],
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        radii: list[float] | None = None,
        neighbors_in_radius: list[int] | None = None,
        n_hidden_local: int = 32,
        n_hidden: int = 256,
        n_head: int = 8,
        dropout: float = 0.0,
        slice_num: int = 32,
        use_te: bool = False,
        plus: bool = False,
        include_local_features: bool = False,
        structured_shape: tuple[int, ...] | None = None,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        del concrete_dropout
        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]
        if structured_shape is not None:
            raise ValueError("structured_shape is not supported in the vendored irregular-mesh port.")

        dim_head = n_hidden // n_head
        context_dim = 0
        self.structured_shape = None

        use_local_bq = geometry_dim is not None and include_local_features
        if use_local_bq:
            self.local_extractors = nn.ModuleList(
                [
                    MultiScaleFeatureExtractor(
                        geometry_dim,
                        radii,
                        neighbors_in_radius,
                        n_hidden_local,
                        n_head,
                        dim_head,
                        dropout,
                        slice_num,
                        use_te=False,
                        plus=plus,
                    )
                    for _ in functional_dims
                ]
            )
            context_dim += dim_head * len(radii) * len(functional_dims)
        else:
            self.local_extractors = None

        if geometry_dim is not None:
            self.geometry_tokenizer = ContextProjector(
                geometry_dim,
                n_head,
                dim_head,
                dropout,
                slice_num,
                use_te=False,
                plus=plus,
            )
            context_dim += dim_head
        else:
            self.geometry_tokenizer = None

        if global_dim is not None:
            self.global_tokenizer = ContextProjector(
                global_dim,
                n_head,
                dim_head,
                dropout,
                slice_num,
                use_te=False,
                plus=plus,
            )
            context_dim += dim_head
        else:
            self.global_tokenizer = None

        self._context_dim = context_dim

    def get_context_dim(self) -> int:
        return self._context_dim

    def build_context(
        self,
        local_embeddings: tuple[torch.Tensor, ...],
        local_positions: tuple[torch.Tensor, ...] | None,
        geometry: torch.Tensor | None = None,
        global_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None, torch.Tensor | None]:
        if len(local_embeddings) == 0:
            raise ValueError("Expected non-empty tuple of local embeddings")

        context_parts: list[torch.Tensor] = []
        local_features = None
        geometry_context_detached: torch.Tensor | None = None

        if local_positions is None and self.local_extractors is not None:
            raise ValueError("Local positions are required if local features are enabled.")

        if self.local_extractors is not None and geometry is not None:
            local_features = []
            for i, _embedding in enumerate(local_embeddings):
                spatial_coords = local_positions[i]
                ctx_feats, local_feat = self.local_extractors[i].extract_context_and_local(
                    spatial_coords, geometry
                )
                context_parts.extend(ctx_feats)
                local_features.append(local_feat)

        if self.geometry_tokenizer is not None and geometry is not None:
            geometry_context = self.geometry_tokenizer(geometry)
            geometry_context_detached = geometry_context.detach()
            context_parts.append(geometry_context)

        if self.global_tokenizer is not None and global_embedding is not None:
            context_parts.append(self.global_tokenizer(global_embedding))

        context = torch.cat(context_parts, dim=-1) if context_parts else None
        return context, local_features, geometry_context_detached
