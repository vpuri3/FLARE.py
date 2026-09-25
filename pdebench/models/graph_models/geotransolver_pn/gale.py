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

"""GALE attention and transformer block (irregular mesh only)."""
from __future__ import annotations

import torch
from torch import nn

from .pn_compat import Mlp, PhysicsAttentionIrregularMesh

__all__ = ["GALE", "GALE_block"]


def _mix_self_and_cross(
    self_attn: torch.Tensor,
    cross_attn: torch.Tensor,
    mode: str,
    state_mixing: nn.Parameter | None = None,
    concat_project: nn.Module | None = None,
) -> torch.Tensor:
    if mode == "weighted":
        w = torch.sigmoid(state_mixing)
        return w * self_attn + (1 - w) * cross_attn
    if mode == "concat_project":
        return concat_project(torch.cat([self_attn, cross_attn], dim=-1))
    raise ValueError(f"Invalid state_mixing_mode: {mode!r}")


def _gale_compute_slice_attention_cross(
    module: nn.Module,
    slice_tokens: list[torch.Tensor],
    context: torch.Tensor,
) -> list[torch.Tensor]:
    q_input = torch.cat(slice_tokens, dim=-2)
    q = module.cross_q(q_input)
    k = module.cross_k(context)
    v = module.cross_v(context)
    cross_attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    return list(torch.split(cross_attention, slice_tokens[0].shape[-2], dim=-2))


def _gale_forward_impl(
    module: nn.Module,
    x: tuple[torch.Tensor, ...],
    context: torch.Tensor | None,
) -> list[torch.Tensor]:
    if len(x) == 0:
        raise ValueError("Expected non-empty tuple of input tensors")
    if module.plus:
        x_mid = [module.project_input_onto_slices(_x) for _x in x]
        fx_mid = list(x_mid)
    else:
        x_mid, fx_mid = zip(*[module.project_input_onto_slices(_x) for _x in x])
    slice_projections = [module.in_project_slice(_x_mid) for _x_mid in x_mid]
    slice_weights, slice_tokens = zip(
        *[
            module._compute_slices_from_projections(proj, _fx_mid)
            for proj, _fx_mid in zip(slice_projections, fx_mid)
        ]
    )
    self_slice_token = [module._compute_slice_attention_sdpa(_st) for _st in slice_tokens]
    if context is not None:
        cross_slice_token = [
            module.compute_slice_attention_cross([_st], context)[0] for _st in slice_tokens
        ]
        out_slice_token = [
            _mix_self_and_cross(
                sst,
                cst,
                module.state_mixing_mode,
                state_mixing=getattr(module, "state_mixing", None),
                concat_project=getattr(module, "concat_project", None),
            )
            for sst, cst in zip(self_slice_token, cross_slice_token)
        ]
    else:
        out_slice_token = self_slice_token
    return [
        module._project_attention_outputs(ost, sw)
        for ost, sw in zip(out_slice_token, slice_weights)
    ]


def _gale_cross_init(
    module: nn.Module,
    dim_head: int,
    context_dim: int,
    state_mixing_mode: str = "weighted",
) -> None:
    module.cross_q = nn.Linear(dim_head, dim_head)
    module.cross_k = nn.Linear(context_dim, dim_head)
    module.cross_v = nn.Linear(context_dim, dim_head)
    module.state_mixing_mode = state_mixing_mode
    if state_mixing_mode == "weighted":
        module.state_mixing = nn.Parameter(torch.tensor(0.0))
    elif state_mixing_mode == "concat_project":
        module.concat_project = nn.Sequential(
            nn.Linear(2 * dim_head, dim_head),
            nn.GELU(),
        )
    else:
        raise ValueError(
            f"Invalid state_mixing_mode: {state_mixing_mode!r}. "
            "Expected 'weighted' or 'concat_project'."
        )


class GALE(PhysicsAttentionIrregularMesh):
    """Geometry-Aware Latent Embeddings: slice self-attn + cross-attn mix."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
    ) -> None:
        del concrete_dropout
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te=False, plus=plus)
        if context_dim > 0:
            _gale_cross_init(self, dim_head, context_dim, state_mixing_mode)

    def compute_slice_attention_cross(
        self,
        slice_tokens: list[torch.Tensor],
        context: torch.Tensor,
    ) -> list[torch.Tensor]:
        return _gale_compute_slice_attention_cross(self, slice_tokens, context)

    def forward(
        self,
        x: tuple[torch.Tensor, ...],
        context: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return _gale_forward_impl(self, x, context)


class GALE_block(nn.Module):
    """Pre-LN GALE attention + MLP residual block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int | float = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        use_te: bool = False,
        plus: bool = False,
        context_dim: int = 0,
        spatial_shape: tuple[int, ...] | None = None,
        attention_type: str = "GALE",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
    ) -> None:
        super().__init__()
        del last_layer, out_dim, concrete_dropout
        if spatial_shape is not None:
            raise ValueError("structured spatial_shape is not supported in the vendored port.")
        if attention_type != "GALE":
            raise ValueError(f"Only attention_type='GALE' is supported; got {attention_type!r}.")
        if use_te:
            raise ValueError("use_te=True is not supported in the vendored port.")

        self.ln_1 = nn.LayerNorm(hidden_dim)
        dim_head = hidden_dim // num_heads
        self.Attn = GALE(
            hidden_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            slice_num=slice_num,
            use_te=False,
            plus=plus,
            context_dim=context_dim,
            state_mixing_mode=state_mixing_mode,
        )
        self.ln_mlp1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Mlp(
                in_features=hidden_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                out_features=hidden_dim,
                act_layer=act,
                use_te=False,
            ),
        )
        self.attn_dropout = None
        self.ffn_dropout = None

    def forward(
        self,
        fx: tuple[torch.Tensor, ...],
        global_context: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if len(fx) == 0:
            raise ValueError("Expected non-empty tuple of input tensors")
        normed = [self.ln_1(_fx) for _fx in fx]
        attn = self.Attn(tuple(normed), global_context)
        fx_out = [attn[i] + fx[i] for i in range(len(fx))]
        fx_out = [self.ln_mlp1(_fx) + _fx for _fx in fx_out]
        return fx_out
