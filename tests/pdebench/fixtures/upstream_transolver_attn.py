"""Single-process reference from Transolver++ upstream attention."""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor | float = 1, hard: bool = False) -> torch.Tensor:
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class UpstreamTransolverPPAttention(nn.Module):
    """Physics_Attention_1D_Eidetic with upstream all-reduces removed."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, slice_num: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        x_mid = self.in_project_x(x).reshape(batch_size, num_tokens, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()
        temperature = torch.clamp(self.proj_temperature(x_mid) + self.bias, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))
        out_slice_token = F.scaled_dot_product_attention(
            self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        )
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        return self.to_out(rearrange(out_x, "b h n d -> b n (h d)"))


class UpstreamTransolver3Attention(nn.Module):
    """Physics_Attention_Irregular_Mesh from Transolver-3 upstream."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, slice_num: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        self.in_project = nn.Linear(dim, 2 * heads * dim_head)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_out_linear = nn.Linear(heads * dim_head, dim)

        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)
        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5

        raw_states = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        raw_states = raw_states / slice_norm.transpose(-1, -2)

        w_fx = self.in_project.weight[: self.heads * self.dim_head].view(self.heads, self.dim_head, x.size(-1))
        b_fx = self.in_project.bias[: self.heads * self.dim_head].view(self.heads, self.dim_head)
        slice_token = torch.einsum("bhgc, hdc -> bhgd", raw_states, w_fx) + b_fx.view(1, self.heads, 1, self.dim_head)

        return self.chunk_deslice_to_out(x, self.slice_attend(slice_token), slice_weights)

    def slice_attend(self, slice_token: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            self.to_q(slice_token),
            self.to_k(slice_token),
            self.to_v(slice_token),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

    def _get_fused_weight_slice(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_in = self.in_project.weight[self.heads * self.dim_head :].view(self.heads, self.dim_head, -1)
        fused_w = torch.matmul(self.in_project_slice.weight, w_in)
        b_in = self.in_project.bias[self.heads * self.dim_head :].view(self.heads, self.dim_head)
        fused_b = torch.matmul(self.in_project_slice.weight, b_in.unsqueeze(-1)).squeeze(-1)
        return fused_w, fused_b + self.in_project_slice.bias

    def chunk_weights(
        self, x: torch.Tensor, fused_w: torch.Tensor | None = None, fused_b: torch.Tensor | None = None
    ) -> torch.Tensor:
        if fused_w is None:
            fused_w, fused_b = self._get_fused_weight_slice()
        logits = torch.einsum("bnc, hgc -> bhng", x, fused_w)
        logits = logits + fused_b.view(1, self.heads, 1, self.slice_num)
        return F.softmax(logits / self.temperature, dim=-1)

    def chunk_deslice_to_out(
        self, x: torch.Tensor, out_slice_token: torch.Tensor, slice_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if slice_weights is None:
            slice_weights = self.chunk_weights(x)
        w_out = self.to_out_linear.weight.view(-1, self.heads, self.dim_head).permute(1, 2, 0)
        projected_slices = torch.einsum("bhgd, hdc -> bhgc", out_slice_token, w_out)
        out_x = torch.einsum("bhng, bhgc -> bnc", slice_weights, projected_slices)
        return self.dropout(out_x + self.to_out_linear.bias)
