from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attn import AttentionConfig, Transformer, TransformerConfig
from .layers.magno import MAGNOConfig, MAGNODecoder, MAGNOEncoder

__all__ = [
    "GAOT",
    "GAOT_PDEBench",
    "GAOT_Structured_Mesh_2D",
    "GAOTConfig",
    "GAOTArgs",
    "GAOTMagnoConfig",
    "GAOTTransformerConfig",
]


class GAOT(nn.Module):
    """
    Upstream GAOT architecture (kept structurally aligned with upstream):
    MAGNO Encoder + Vision Transformer + MAGNO Decoder.
    """

    def __init__(self, input_size: int, output_size: int, config):
        nn.Module.__init__(self)

        coord_dim = config.args.magno.coord_dim
        if coord_dim not in [2, 3]:
            raise ValueError(f"coord_dim must be 2 or 3, got {coord_dim}")

        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = coord_dim
        self.node_latent_size = config.args.magno.lifting_channels
        self.patch_size = config.args.transformer.patch_size

        latent_tokens_size = config.latent_tokens_size
        if coord_dim == 2:
            if len(latent_tokens_size) != 2:
                raise ValueError(f"For 2D, latent_tokens_size must have 2 dimensions, got {len(latent_tokens_size)}")
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1]
            self.D = None
        else:
            if len(latent_tokens_size) != 3:
                raise ValueError(f"For 3D, latent_tokens_size must have 3 dimensions, got {len(latent_tokens_size)}")
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1]
            self.D = latent_tokens_size[2]

        self.encoder = self.init_encoder(input_size, self.node_latent_size, config.args.magno)
        self.processor = self.init_processor(self.node_latent_size, config.args.transformer)
        self.decoder = self.init_decoder(output_size, self.node_latent_size, config.args.magno)

    def init_encoder(self, input_size, latent_size, config):
        return MAGNOEncoder(
            in_channels=input_size,
            out_channels=latent_size,
            config=config,
        )

    def init_processor(self, node_latent_size, config):
        if self.coord_dim == 2:
            patch_volume = self.patch_size * self.patch_size
        else:
            patch_volume = self.patch_size * self.patch_size * self.patch_size

        self.patch_linear = nn.Linear(
            patch_volume * node_latent_size,
            patch_volume * node_latent_size,
        )

        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        return Transformer(
            input_size=node_latent_size * patch_volume,
            output_size=node_latent_size * patch_volume,
            config=config,
        )

    def init_decoder(self, output_size, latent_size, config):
        return MAGNODecoder(
            in_channels=latent_size,
            out_channels=output_size,
            config=config,
        )

    def _get_patch_positions(self):
        P = self.patch_size

        if self.coord_dim == 2:
            num_patches_H = self.H // P
            num_patches_W = self.W // P
            positions = torch.stack(
                torch.meshgrid(
                    torch.arange(num_patches_H, dtype=torch.float32),
                    torch.arange(num_patches_W, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
        else:
            num_patches_H = self.H // P
            num_patches_W = self.W // P
            num_patches_D = self.D // P
            positions = torch.stack(
                torch.meshgrid(
                    torch.arange(num_patches_H, dtype=torch.float32),
                    torch.arange(num_patches_W, dtype=torch.float32),
                    torch.arange(num_patches_D, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

    def encode(self, x_coord: torch.Tensor, pndata: torch.Tensor, latent_tokens_coord: torch.Tensor, encoder_nbrs: list):
        return self.encoder(
            x_coord=x_coord,
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs,
        )

    def process(self, rndata: Optional[torch.Tensor] = None, condition: Optional[float] = None) -> torch.Tensor:
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        P = self.patch_size

        if self.coord_dim == 2:
            H, W = self.H, self.W
            assert n_regional_nodes == H * W, f"n_regional_nodes ({n_regional_nodes}) != H*W ({H}*{W})"
            assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"

            num_patches_H = H // P
            num_patches_W = W // P

            rndata = rndata.view(batch_size, H, W, C)
            rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
            rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
            rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)

        else:
            H, W, D = self.H, self.W, self.D
            assert n_regional_nodes == H * W * D, f"n_regional_nodes ({n_regional_nodes}) != H*W*D ({H}*{W}*{D})"
            assert H % P == 0 and W % P == 0 and D % P == 0, f"H({H}), W({W}), D({D}) must be divisible by P({P})"

            num_patches_H = H // P
            num_patches_W = W // P
            num_patches_D = D // P

            rndata = rndata.view(batch_size, H, W, D, C)
            rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, num_patches_D, P, C)
            rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            rndata = rndata.view(batch_size, num_patches_H * num_patches_W * num_patches_D, P * P * P * C)

        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)

        if self.positional_embedding_name == "absolute":
            patch_volume = P ** self.coord_dim
            pos_emb = self._compute_absolute_embeddings(pos, patch_volume * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
        elif self.positional_embedding_name == "rope":
            relative_positions = pos
        else:
            relative_positions = None

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        if self.coord_dim == 2:
            rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
            rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
            rndata = rndata.view(batch_size, H * W, C)
        else:
            rndata = rndata.view(batch_size, num_patches_H, num_patches_W, num_patches_D, P, P, P, C)
            rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
            rndata = rndata.view(batch_size, H * W * D, C)

        return rndata

    def decode(self, latent_tokens_coord: torch.Tensor, rndata: torch.Tensor, query_coord: torch.Tensor, decoder_nbrs: list):
        return self.decoder(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata,
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs,
        )

    def forward(
        self,
        latent_tokens_coord: torch.Tensor,
        xcoord: torch.Tensor,
        pndata: torch.Tensor,
        query_coord: Optional[torch.Tensor] = None,
        encoder_nbrs: Optional[list] = None,
        decoder_nbrs: Optional[list] = None,
        condition: Optional[float] = None,
    ) -> torch.Tensor:
        rndata = self.encode(
            x_coord=xcoord,
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs,
        )

        rndata = self.process(rndata=rndata, condition=condition)

        if query_coord is None:
            query_coord = xcoord
        output = self.decode(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata,
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs,
        )
        return output

    def autoregressive_predict(
        self,
        x_batch: torch.Tensor,
        time_indices: np.ndarray,
        t_values: np.ndarray,
        stats: Dict,
        stepper_mode: str = "output",
        latent_tokens_coord: Optional[torch.Tensor] = None,
        fixed_coord: Optional[torch.Tensor] = None,
        encoder_nbrs: Optional[List] = None,
        decoder_nbrs: Optional[List] = None,
        use_conditional_norm: bool = False,
    ) -> torch.Tensor:
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)
        predictions = []
        u_mean = stats["u"]["mean"].to(x_batch.device)
        u_std = stats["u"]["std"].to(x_batch.device)

        start_times_mean = stats["start_time"]["mean"]
        start_times_std = stats["start_time"]["std"]
        time_diffs_mean = stats["time_diffs"]["mean"]
        time_diffs_std = stats["time_diffs"]["std"]

        u_dim = stats["u"]["mean"].shape[0]
        c_dim = stats["c"]["mean"].shape[0] if "c" in stats else 0

        c_features = x_batch[..., u_dim:u_dim + c_dim] if c_dim > 0 else None
        current_u = x_batch[..., :u_dim]

        is_variable_coords = encoder_nbrs is not None and decoder_nbrs is not None

        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx - 1]
            t_out_idx = time_indices[idx]
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]

            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std

            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, dtype=x_batch.dtype, device=x_batch.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm, dtype=x_batch.dtype, device=x_batch.device)

            input_features = [current_u]
            if c_features is not None:
                input_features.append(c_features)
            input_features.extend([start_time_expanded, time_diff_expanded])
            x_input = torch.cat(input_features, dim=-1)

            with torch.no_grad():
                if use_conditional_norm:
                    if is_variable_coords:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input[..., :-1],
                            condition=x_input[..., 0, -2:-1],
                            encoder_nbrs=encoder_nbrs,
                            decoder_nbrs=decoder_nbrs,
                        )
                    else:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input[..., :-1],
                            condition=x_input[..., 0, -2:-1],
                        )
                else:
                    if is_variable_coords:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input,
                            encoder_nbrs=encoder_nbrs,
                            decoder_nbrs=decoder_nbrs,
                        )
                    else:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input,
                        )

                pred_denorm = self._process_autoregressive_prediction(
                    pred, current_u, u_mean, u_std, time_diff, stats, stepper_mode
                )
                predictions.append(pred_denorm)
                current_u = (pred_denorm - u_mean) / u_std

        return torch.stack(predictions, dim=1)

    def _process_autoregressive_prediction(
        self,
        pred: torch.Tensor,
        current_u: torch.Tensor,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        time_diff: float,
        stats: Dict,
        stepper_mode: str,
    ) -> torch.Tensor:
        if stepper_mode == "output":
            pred_denorm = pred * u_std + u_mean
        elif stepper_mode == "residual":
            res_mean = stats["res"]["mean"].to(pred.device)
            res_std = stats["res"]["std"].to(pred.device)
            pred_denorm_res = pred * res_std + res_mean
            current_u_denorm = current_u * u_std + u_mean
            pred_denorm = current_u_denorm + pred_denorm_res
        elif stepper_mode == "time_der":
            der_mean = stats["der"]["mean"].to(pred.device)
            der_std = stats["der"]["std"].to(pred.device)
            pred_denorm_der = pred * der_std + der_mean
            current_u_denorm = current_u * u_std + u_mean
            time_diff_tensor = torch.tensor(time_diff, dtype=pred.dtype, device=pred.device)
            pred_denorm = current_u_denorm + time_diff_tensor * pred_denorm_der
        else:
            raise ValueError(f"Unsupported stepper_mode: {stepper_mode}")
        return pred_denorm


@dataclass
class GAOTMagnoConfig:
    coord_dim: int = 2
    lifting_channels: int = 32
    use_geoembed: bool = True
    neighbor_search_method: str = "auto"
    use_torch_scatter: bool = True


@dataclass
class GAOTTransformerConfig:
    num_layers: int = 3
    patch_size: int = 2
    ffn_multiplier: float = 4.0
    num_heads: int = 8


@dataclass
class GAOTArgs:
    magno: GAOTMagnoConfig = field(default_factory=GAOTMagnoConfig)
    transformer: GAOTTransformerConfig = field(default_factory=GAOTTransformerConfig)


@dataclass
class GAOTConfig:
    args: GAOTArgs = field(default_factory=GAOTArgs)
    latent_tokens_size: List[int] = field(default_factory=lambda: [64, 64])


class GAOT_Structured_Mesh_2D(nn.Module):
    """
    Thin PDEBench wrapper around upstream GAOT.
    Input convention: first `coord_dim` channels are coordinates, remaining
    channels are physical node features.
    """

    def __init__(self, config: GAOTConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        space_dim = int(metadata.get("space_dim", metadata.get("c_in", 1)))
        out_dim = int(metadata.get("c_out", 1))
        coord_dim = int(config.args.magno.coord_dim)
        n_layers = int(config.args.transformer.num_layers)
        n_hidden = int(config.args.magno.lifting_channels)
        n_head = int(config.args.transformer.num_heads)
        mlp_ratio = float(config.args.transformer.ffn_multiplier)
        H = int(metadata.get("H", 64))
        W = int(metadata.get("W", 64))
        patch_size = int(config.args.transformer.patch_size)
        gaot_use_geoembed = bool(config.args.magno.use_geoembed)
        gaot_neighbor_search_method = config.args.magno.neighbor_search_method
        gaot_use_torch_scatter = bool(config.args.magno.use_torch_scatter)
        if coord_dim not in [2, 3]:
            raise ValueError(f"GAOT supports coord_dim in {{2,3}}, got {coord_dim}")
        if space_dim < coord_dim:
            raise ValueError(f"space_dim ({space_dim}) must be >= coord_dim ({coord_dim})")
        if coord_dim != 2:
            raise ValueError("GAOT_Structured_Mesh_2D expects coord_dim=2.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.H = H
        self.W = W
        self.coord_dim = coord_dim
        self.feature_dim = max(1, space_dim - coord_dim)
        self.patch_size = patch_size

        latent_h = max(patch_size, (H // patch_size) * patch_size)
        latent_w = max(patch_size, (W // patch_size) * patch_size)
        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError("latent grid must be divisible by patch_size.")
        self.latent_h = latent_h
        self.latent_w = latent_w

        attn_hidden = n_hidden * (patch_size * patch_size)
        attn_ffn_multiplier = max(1, int(round(mlp_ratio)))
        attn_cfg = AttentionConfig(
            num_heads=n_head,
            num_kv_heads=n_head,
            use_conditional_norm=False,
            cond_norm_hidden_size=4,
            atten_dropout=0.0,
        )
        tr_cfg = TransformerConfig(
            patch_size=patch_size,
            hidden_size=attn_hidden,
            use_attn_norm=True,
            use_ffn_norm=True,
            norm_eps=1e-6,
            num_layers=n_layers,
            positional_embedding="absolute",
            use_long_range_skip=True,
            ffn_multiplier=attn_ffn_multiplier,
            attn_config=attn_cfg,
        )
        magno_cfg = MAGNOConfig(
            coord_dim=coord_dim,
            radius=0.033,
            hidden_size=n_hidden,
            mlp_layers=3,
            lifting_channels=n_hidden,
            scales=[1.0],
            use_scale_weights=False,
            use_attention=True,
            attention_type="cosine",
            use_geoembed=gaot_use_geoembed,
            embedding_method="statistical",
            pooling="max",
            transform_type="linear",
            sampling_strategy=None,
            max_neighbors=None,
            sample_ratio=None,
            node_embedding=False,
            neighbor_search_method=gaot_neighbor_search_method,
            use_torch_scatter=gaot_use_torch_scatter,
            neighbor_strategy="radius",
            precompute_edges=False,
        )
        cfg = GAOTConfig(
            args=GAOTArgs(magno=magno_cfg, transformer=tr_cfg),
            latent_tokens_size=[latent_h, latent_w],
        )
        self.core = GAOT(
            input_size=self.feature_dim,
            output_size=out_dim,
            config=cfg,
        )

    def _split(self, x: torch.Tensor):
        coord = x[..., : self.coord_dim]
        feat = x[..., self.coord_dim :]
        if feat.shape[-1] == 0:
            feat = torch.zeros(*coord.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return coord, feat

    def _latent_coords_from_query_coords(self, query_coord: torch.Tensor) -> torch.Tensor:
        # query_coord: [B, H*W, coord_dim]
        b, n, c = query_coord.shape
        if n != self.H * self.W:
            raise ValueError(f"Expected N=H*W={self.H*self.W}, got N={n}")
        grid = query_coord.view(b, self.H, self.W, c).permute(0, 3, 1, 2).contiguous()
        latent_grid = F.adaptive_avg_pool2d(grid, output_size=(self.latent_h, self.latent_w))
        latent = latent_grid.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        # upstream GAOT expects latent_tokens_coord as [num_latent, coord_dim]
        return latent[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coord, feat = self._split(x)
        latent_tokens_coord = self._latent_coords_from_query_coords(coord)
        return self.core(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord,
            pndata=feat,
            query_coord=coord,
            encoder_nbrs=None,
            decoder_nbrs=None,
            condition=None,
        )


class GAOT_PDEBench(nn.Module):
    """
    PDEBench wrapper around upstream GAOT core for point-cloud style inputs.
    """

    def __init__(self, config: GAOTConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        space_dim = int(metadata.get("space_dim", metadata.get("c_in", 1)))
        out_dim = int(metadata.get("c_out", 1))
        coord_dim = int(config.args.magno.coord_dim)
        n_layers = int(config.args.transformer.num_layers)
        n_hidden = int(config.args.magno.lifting_channels)
        n_head = int(config.args.transformer.num_heads)
        mlp_ratio = float(config.args.transformer.ffn_multiplier)
        patch_size = int(config.args.transformer.patch_size)
        latent_h, latent_w = [int(v) for v in config.latent_tokens_size]
        gaot_use_geoembed = bool(config.args.magno.use_geoembed)
        gaot_neighbor_search_method = config.args.magno.neighbor_search_method
        gaot_use_torch_scatter = bool(config.args.magno.use_torch_scatter)
        if coord_dim != 2:
            raise ValueError("GAOT_PDEBench currently supports coord_dim=2.")
        if space_dim < coord_dim:
            raise ValueError(f"space_dim ({space_dim}) must be >= coord_dim ({coord_dim})")
        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError("latent_h and latent_w must be divisible by patch_size.")

        self.coord_dim = coord_dim
        self.feature_dim = max(1, space_dim - coord_dim)
        self.latent_h = latent_h
        self.latent_w = latent_w

        attn_hidden = n_hidden * (patch_size * patch_size)
        attn_ffn_multiplier = max(1, int(round(mlp_ratio)))
        attn_cfg = AttentionConfig(
            num_heads=n_head,
            num_kv_heads=n_head,
            use_conditional_norm=False,
            cond_norm_hidden_size=4,
            atten_dropout=0.0,
        )
        tr_cfg = TransformerConfig(
            patch_size=patch_size,
            hidden_size=attn_hidden,
            use_attn_norm=True,
            use_ffn_norm=True,
            norm_eps=1e-6,
            num_layers=n_layers,
            positional_embedding="absolute",
            use_long_range_skip=True,
            ffn_multiplier=attn_ffn_multiplier,
            attn_config=attn_cfg,
        )
        magno_cfg = MAGNOConfig(
            coord_dim=coord_dim,
            radius=0.033,
            hidden_size=n_hidden,
            mlp_layers=3,
            lifting_channels=n_hidden,
            scales=[1.0],
            use_scale_weights=False,
            use_attention=True,
            attention_type="cosine",
            use_geoembed=gaot_use_geoembed,
            embedding_method="statistical",
            pooling="max",
            transform_type="linear",
            sampling_strategy=None,
            max_neighbors=None,
            sample_ratio=None,
            node_embedding=False,
            neighbor_search_method=gaot_neighbor_search_method,
            use_torch_scatter=gaot_use_torch_scatter,
            neighbor_strategy="radius",
            precompute_edges=False,
        )
        cfg = GAOTConfig(
            args=GAOTArgs(magno=magno_cfg, transformer=tr_cfg),
            latent_tokens_size=[latent_h, latent_w],
        )
        self.core = GAOT(
            input_size=self.feature_dim,
            output_size=out_dim,
            config=cfg,
        )

    def _split(self, x: torch.Tensor):
        coord = x[..., : self.coord_dim]
        feat = x[..., self.coord_dim :]
        if feat.shape[-1] == 0:
            feat = torch.zeros(*coord.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return coord, feat

    def _make_latent_grid(self, coord: torch.Tensor) -> torch.Tensor:
        # coord: [B, N, 2], build latent tokens in data bounding box
        c0 = coord[0]
        mins = c0.min(dim=0).values
        maxs = c0.max(dim=0).values
        tx = torch.linspace(mins[0], maxs[0], self.latent_h, device=coord.device, dtype=coord.dtype)
        ty = torch.linspace(mins[1], maxs[1], self.latent_w, device=coord.device, dtype=coord.dtype)
        gx, gy = torch.meshgrid(tx, ty, indexing="ij")
        latent = torch.stack([gx, gy], dim=-1).reshape(-1, 2).contiguous()
        return latent

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        coord, feat = self._split(x)
        latent_tokens_coord = self._make_latent_grid(coord)
        return self.core(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord,
            pndata=feat,
            query_coord=coord,
            encoder_nbrs=None,
            decoder_nbrs=None,
            condition=None,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or bool(mask.all()):
            return self._forward_dense(x)

        outputs = None
        for batch_idx in range(x.shape[0]):
            valid = mask[batch_idx]
            if not bool(valid.any()):
                continue
            y = self._forward_dense(x[batch_idx : batch_idx + 1, valid])
            if outputs is None:
                outputs = torch.zeros(*x.shape[:-1], self.core.output_size, dtype=y.dtype, device=y.device)
            outputs[batch_idx, valid] = y.squeeze(0)
        if outputs is None:
            outputs = torch.zeros(*x.shape[:-1], self.core.output_size, dtype=x.dtype, device=x.device)
        return outputs
