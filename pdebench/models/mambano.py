#
# Adapted from:
# https://github.com/weili419/Mamba-Neural-Operator/blob/main/MambaNOModule.py
import math
from functools import partial
from typing import Callable, Sequence

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ModuleNotFoundError as exc:
    _MAMBANO_IMPORT_ERROR = exc

    def selective_scan_fn(*args, **kwargs):
        raise ModuleNotFoundError(
            "MambaNO is unavailable because optional dependencies are missing "
            "(mamba-ssm/causal-conv1d)."
        ) from _MAMBANO_IMPORT_ERROR

__all__ = [
    "MambaNO_Structured_Mesh_2D",
]

@dataclass
class MambaNOConfig:
    model: str = "mambano"
    num_blocks: int = 8
    channel_dim: int = 64
    mambano_d_state: int = 16
    mambano_drop_path_rate: float = 0.1
    mambano_use_checkpoint: bool = False



class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        mid = max(in_channels, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.norm(self.conv1(x)))
        x = self.conv2(x)
        return self.act(x + r)


class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, out_channels: int, out_size: tuple[int, int] | None):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj(x))
        if self.out_size is not None and tuple(x.shape[-2:]) != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Parameter(torch.empty(4, self.dt_rank + 2 * d_state, self.d_inner))
        nn.init.xavier_uniform_(self.x_proj)

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_min, dt_max, dt_init_floor) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([w for (w, _) in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([b for (_, b) in dt_projs], dim=0))

        self.A_logs = self._A_log_init(d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _dt_init(dt_rank: int, d_inner: int, dt_scale: float, dt_min: float, dt_max: float, dt_init_floor: float):
        weight = torch.empty(d_inner, dt_rank)
        nn.init.uniform_(weight, -dt_rank ** -0.5 * dt_scale, dt_rank ** -0.5 * dt_scale)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        bias = dt + torch.log(-torch.expm1(-dt))
        return weight, bias

    @staticmethod
    def _A_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        return nn.Parameter(A_log)

    @staticmethod
    def _D_init(d_inner: int, copies: int = 1, merge: bool = True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        return nn.Parameter(D)

    def _forward_core(self, x: torch.Tensor):
        B, _, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.contiguous().view(B, -1, L), x.transpose(2, 3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        invwh_y = inv_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y1, y2, y3, y4 = self._forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.ssm = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.ssm(self.norm(x)))


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else float(drop_path),
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        if self.downsample is not None:
            x = self.downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class VSSLayerUp(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        upsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.upsample = upsample
        self.blocks = nn.ModuleList(
            [
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else float(drop_path),
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class MambaNOBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: tuple[int, int],
        embed_dim: int = 32,
        depths: Sequence[int] = (2, 2, 2, 2),
        d_state: int = 16,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_size = in_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.dims_encoder = [int(embed_dim * (2 ** i)) for i in range(self.num_layers)]
        self.dims_decoder = list(reversed(self.dims_encoder))
        self.encoder_sizes = self._build_sizes(in_size, self.num_layers)
        self.decoder_sizes = list(reversed(self.encoder_sizes))

        self.patch_embed = ConvProjection(in_channels, self.embed_dim, kernel_size=3)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = list(reversed(dpr))

        self.stage_res_blocks = nn.ModuleList(
            [
                nn.ModuleList([ResidualConvBlock(self.dims_encoder[i]) for _ in range(depths[i])])
                for i in range(self.num_layers)
            ]
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            downsample = None
            if i < self.num_layers - 1:
                downsample = PatchMerging2D(
                    dim=self.dims_encoder[i],
                    out_channels=self.dims_encoder[i + 1],
                    out_size=self.encoder_sizes[i + 1],
                )
            self.layers.append(
                VSSLayer(
                    dim=self.dims_encoder[i],
                    depth=depths[i],
                    d_state=d_state,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=nn.LayerNorm,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            upsample = None
            if i != 0:
                upsample = PatchMerging2D(
                    dim=self.dims_decoder[i - 1],
                    out_channels=self.dims_decoder[i],
                    out_size=self.decoder_sizes[i],
                )
            self.layers_up.append(
                VSSLayerUp(
                    dim=self.dims_decoder[i],
                    depth=depths[i],
                    d_state=d_state,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr_decoder[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=nn.LayerNorm,
                    upsample=upsample,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.final_conv = ConvProjection(self.embed_dim, self.embed_dim, kernel_size=3)
        self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _build_sizes(in_size: tuple[int, int], levels: int) -> list[tuple[int, int]]:
        sizes = [in_size]
        h, w = in_size
        for _ in range(levels - 1):
            h = max(1, (h + 1) // 2)
            w = max(1, (w + 1) // 2)
            sizes.append((h, w))
        return sizes

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def _forward_features(self, x: torch.Tensor):
        skip_list = []
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            y = x.permute(0, 3, 1, 2)
            for block in self.stage_res_blocks[i]:
                y = block(y)
            y = y.permute(0, 2, 3, 1)
            skip_list.append(y)
            x = self.layers[i](y)
        return x, skip_list

    def _forward_features_up(self, x: torch.Tensor, skip_list: list[torch.Tensor]):
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-i])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_list = self._forward_features(x)
        x = self._forward_features_up(x, skip_list)
        x = self.final_conv(x.permute(0, 3, 1, 2).contiguous())
        return self.head(x)


class MambaNO_Structured_Mesh_2D(nn.Module):
    def __init__(self, config: MambaNOConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        space_dim = int(metadata.get("space_dim", metadata.get("c_in", 1)))
        fun_dim = int(metadata.get("fun_dim", 0))
        out_dim = int(metadata.get("c_out", 1))
        n_layers = int(config.num_blocks)
        n_hidden = int(config.channel_dim)
        dropout = float(getattr(config, "dropout", 0.0))
        H = int(metadata.get("H", 64))
        W = int(metadata.get("W", 64))
        d_state = int(config.mambano_d_state)
        drop_path_rate = float(config.mambano_drop_path_rate)
        use_checkpoint = bool(config.mambano_use_checkpoint)
        depths = getattr(config, "depths", None)
        self.__name__ = "MambaNO_Structured_Mesh_2D"
        self.H = H
        self.W = W
        if depths is None:
            if isinstance(n_layers, int):
                stage_depth = max(1, n_layers // 4)
                depths = [stage_depth] * 4
            else:
                depths = [2, 2, 2, 2]

        self.model = MambaNOBackbone(
            in_channels=space_dim + fun_dim,
            out_channels=out_dim,
            in_size=(H, W),
            embed_dim=n_hidden,
            depths=depths,
            d_state=d_state,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, f=None):
        B, N, C = x.shape
        expected = self.H * self.W
        if N != expected:
            raise ValueError(f"Expected N=H*W={expected}, got N={N}.")
        if f is not None:
            x = torch.cat((x, f), dim=-1)
        x = x.view(B, self.H, self.W, C if f is None else x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        y = self.model(x)
        return y.permute(0, 2, 3, 1).contiguous().view(B, N, -1)
