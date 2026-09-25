import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

__all__ = [
    "PhysicsAttention",
    "TransolverBlock",
]


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class PhysicsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length) or attention_mask.dtype != torch.bool:
                raise ValueError(
                    "attention_mask must be a boolean tensor with shape [B, N]. "
                    f"Got {attention_mask.shape}, {attention_mask.dtype}."
                )
            valid = attention_mask[:, None, :, None].to(dtype=x.dtype)
        else:
            valid = None

        fx_mid = (
            self.in_project_fx(x)
            .reshape(batch_size, sequence_length, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(batch_size, sequence_length, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_logits = self.in_project_slice(x_mid) / temperature
        slice_weights = F.softmax(slice_logits.float(), dim=-1).to(dtype=x_mid.dtype)
        if valid is not None:
            slice_weights = slice_weights * valid
            fx_mid = fx_mid * valid
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5).unsqueeze(-1)

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots.float(), dim=-1).to(dtype=dots.dtype)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        out_x = self.to_out(out_x)
        if attention_mask is not None:
            out_x = out_x * attention_mask.unsqueeze(-1).to(dtype=out_x.dtype)
        return out_x


class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ) -> None:
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Activation {act} is not implemented.")
        activation = ACTIVATION[act]
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for linear in self.linears:
            if self.res:
                x = linear(x) + x
            else:
                x = linear(x)
        return self.linear_post(x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int,
        act: str | None = None,
        rmsnorm: bool = False,
        mlp_ratio: float = 4.0,
        num_slices: int = 64,
        rope=None,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or channel_dim % num_heads != 0:
            raise ValueError(
                f"channel_dim must be divisible by a positive num_heads. Got {channel_dim} and {num_heads}."
            )
        if num_slices <= 0:
            raise ValueError(f"num_slices must be positive. Got {num_slices}.")
        if rope is not None:
            raise ValueError("RoPE is not supported by TransolverBlock; use absolute or sinusoidal positions.")

        activation = act or "gelu"
        norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.ln_1 = norm(channel_dim)
        self.Attn = PhysicsAttention(
            channel_dim,
            heads=num_heads,
            dim_head=channel_dim // num_heads,
            dropout=0.0,
            slice_num=num_slices,
        )
        self.ln_2 = norm(channel_dim)
        self.mlp = MLP(
            channel_dim,
            int(channel_dim * mlp_ratio),
            channel_dim,
            n_layers=0,
            res=False,
            act=activation,
        )

    def initialize_weights(self) -> None:
        def init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        self.apply(init)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.Attn(self.ln_1(x), attention_mask=attention_mask) + x
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = self.mlp(self.ln_2(x)) + x
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x
