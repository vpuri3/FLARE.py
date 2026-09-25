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

"""Local shims for PhysicsNeMo GeoTransolver dependencies (no nvidia-physicsnemo)."""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch
from einops import rearrange
from torch import nn

__all__ = [
    "Mlp",
    "PhysicsAttentionIrregularMesh",
    "_project_input",
    "_compute_slices_from_projections",
]


_ACT = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _resolve_activation(act_layer: nn.Module | type[nn.Module] | str) -> nn.Module:
    if isinstance(act_layer, str):
        key = act_layer.lower()
        if key not in _ACT:
            raise ValueError(f"Unknown activation {act_layer!r}")
        return _ACT[key]()
    if isinstance(act_layer, nn.Module):
        return act_layer
    return act_layer()


class Mlp(nn.Module):
    """PhysicsNeMo-compatible MLP (list/int hidden sizes, no TE)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | list[int] | None = None,
        out_features: int | None = None,
        act_layer: nn.Module | type[nn.Module] | str = nn.GELU,
        drop: float = 0.0,
        final_dropout: bool = True,
        bias: bool = True,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
        use_te: bool = False,
    ):
        super().__init__()
        del use_te, use_batchnorm, spectral_norm  # unsupported in local shim
        out_features = out_features or in_features
        if hidden_features is None:
            hidden_features = [in_features]
        elif isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers: list[nn.Module] = []
        dims = [in_features, *hidden_features, out_features]
        n_layers = len(dims) - 1
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            is_last = i == n_layers - 1
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if not is_last:
                layers.append(_resolve_activation(act_layer))
            if drop != 0 and (not is_last or final_dropout):
                layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _project_input(
    x: torch.Tensor,
    project_x: nn.Module,
    heads: int,
    dim_head: int,
    pattern: str,
    project_fx: nn.Module | None = None,
):
    px = rearrange(project_x(x), pattern, H=heads, D=dim_head)
    if project_fx is None:
        return px
    return px, rearrange(project_fx(x), pattern, H=heads, D=dim_head)


def _compute_slices_from_projections(
    slice_projections: torch.Tensor,
    fx: torch.Tensor,
    temperature: torch.Tensor,
    plus: bool,
    proj_temperature: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plus and proj_temperature is not None:
        temp = temperature + proj_temperature(fx)
        clamped = torch.clamp(temp, min=0.01).to(slice_projections.dtype)
        # Gumbel-softmax stand-in: soft temperature softmax (plus path unused by default).
        slice_weights = torch.nn.functional.softmax(slice_projections / clamped, dim=-1)
    else:
        clamped = torch.clamp(temperature, min=0.5, max=5).to(slice_projections.dtype)
        slice_weights = torch.nn.functional.softmax(slice_projections / clamped, dim=-1)
    slice_weights = slice_weights.to(slice_projections.dtype)
    slice_norm = slice_weights.sum(1) + 1e-2
    normed_weights = slice_weights / slice_norm[:, None, :, :]
    slice_token = torch.matmul(normed_weights.permute(0, 2, 3, 1), fx.permute(0, 2, 1, 3))
    return slice_weights, slice_token


class PhysicsAttentionBase(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        slice_num: int,
        use_te: bool,
        plus: bool,
    ):
        super().__init__()
        if use_te:
            raise ValueError("Transformer Engine path is disabled in the vendored port; set use_te=False.")
        inner_dim = dim_head * heads
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = False
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)
        if plus:
            self.proj_temperature = nn.Sequential(
                nn.Linear(self.dim_head, slice_num),
                nn.GELU(),
                nn.Linear(slice_num, 1),
                nn.GELU(),
            )
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.qkv_project = nn.Linear(dim_head, 3 * dim_head, bias=False)
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def project_input_onto_slices(self, x: torch.Tensor):
        ...

    def _compute_slices_from_projections(
        self,
        slice_projections: torch.Tensor,
        fx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proj_temp = getattr(self, "proj_temperature", None) if self.plus else None
        return _compute_slices_from_projections(
            slice_projections, fx, self.temperature, self.plus, proj_temp
        )

    def _compute_slice_attention_sdpa(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_project(slice_tokens)
        qkv = rearrange(qkv, "b h s (t d) -> b h s t d", t=3, d=self.dim_head)
        q, k, v = qkv.unbind(3)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    def _compute_slice_attention_te(self, slice_tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_slice_attention_sdpa(slice_tokens)

    def _project_attention_outputs(
        self,
        out_slice_token: torch.Tensor,
        slice_weights: torch.Tensor,
    ) -> torch.Tensor:
        out_x = torch.einsum("bths,bhsd->bthd", slice_weights, out_slice_token)
        out_x = rearrange(out_x, "b t h d -> b t (h d)")
        return self.out_dropout(self.out_linear(out_x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.plus:
            x_mid = self.project_input_onto_slices(x)
            fx_mid = x_mid
        else:
            x_mid, fx_mid = self.project_input_onto_slices(x)
        slice_projections = self.in_project_slice(x_mid)
        slice_weights, slice_token = self._compute_slices_from_projections(slice_projections, fx_mid)
        out_slice = self._compute_slice_attention_sdpa(slice_token)
        return self._project_attention_outputs(out_slice, slice_weights)


class PhysicsAttentionIrregularMesh(PhysicsAttentionBase):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = False,
        plus: bool = False,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)
        inner_dim = dim_head * heads
        self.in_project_x = nn.Linear(dim, inner_dim)
        if not plus:
            self.in_project_fx = nn.Linear(dim, inner_dim)

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
