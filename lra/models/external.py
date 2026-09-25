#
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "EXTERNAL_MODEL_TYPES",
    "ExternalModelWrapper",
]


def _require_transformers():
    try:
        from transformers import FunnelBaseModel, FunnelConfig, ReformerConfig, ReformerModel
    except ImportError as exc:
        raise ImportError(
            "External LRA backends require `transformers`. Install with `uv sync --extra lra_external`."
        ) from exc
    return FunnelBaseModel, FunnelConfig, ReformerConfig, ReformerModel


def _map_hidden_act(act: Optional[str]) -> str:
    if act in [None, "gelu"]:
        return "gelu"
    if act in ["silu", "swish"]:
        return "silu"
    if act == "relu":
        return "relu"
    return "gelu"


def _factor_pair(n: int) -> list[int]:
    root = int(math.sqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            return [a, n // a]
    return [1, n]


def _largest_divisor_at_most(n: int, cap: int) -> int:
    upper = min(n, cap)
    for d in range(upper, 0, -1):
        if n % d == 0:
            return d
    return 1


def _default_funnel_block_sizes(num_layers: int) -> list[int]:
    n_stages = 3 if num_layers >= 3 else num_layers
    base = num_layers // n_stages
    rem = num_layers % n_stages
    return [base + (1 if i < rem else 0) for i in range(n_stages)]


def _default_reformer_attn_layers(num_layers: int) -> list[str]:
    return ["local" if i % 2 == 0 else "lsh" for i in range(num_layers)]


class _FunnelEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        channel_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        act: Optional[str],
        emb_drop: float,
        attn_drop: float,
        proj_drop: float,
        pad_id: Optional[int],
        pool: str,
        funnel_block_sizes: Optional[List[int]],
    ) -> None:
        super().__init__()
        FunnelBaseModel, FunnelConfig, _, _ = _require_transformers()

        head_dim = channel_dim // num_heads
        block_sizes = funnel_block_sizes or _default_funnel_block_sizes(num_blocks)
        cfg = FunnelConfig(
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            d_model=channel_dim,
            n_head=num_heads,
            d_head=head_dim,
            d_inner=int(channel_dim * mlp_ratio),
            hidden_act=_map_hidden_act(act),
            hidden_dropout=max(emb_drop, proj_drop),
            attention_dropout=attn_drop,
            activation_dropout=proj_drop,
            pad_token_id=pad_id,
            separate_cls=(pool == "cls"),
            truncate_seq=True,
            max_position_embeddings=max_length,
        )
        self.model = FunnelBaseModel(cfg)
        self.hidden_size = channel_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state


class _ReformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        channel_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        act: Optional[str],
        emb_drop: float,
        attn_drop: float,
        proj_drop: float,
        pad_id: Optional[int],
        pool: str,
        reformer_attn_layers: Optional[List[str]],
        reformer_num_hashes: int,
        reformer_local_chunk_length: Optional[int],
        reformer_lsh_chunk_length: Optional[int],
        reformer_axial_pos_shape: Optional[List[int]],
    ) -> None:
        super().__init__()
        del pool
        _, _, ReformerConfig, ReformerModel = _require_transformers()

        head_dim = channel_dim // num_heads
        axial_pos_shape = reformer_axial_pos_shape or _factor_pair(max_length)
        local_chunk = reformer_local_chunk_length or _largest_divisor_at_most(max_length, 64)
        lsh_chunk = reformer_lsh_chunk_length or _largest_divisor_at_most(max_length, 64)
        attn_layers = reformer_attn_layers or _default_reformer_attn_layers(num_blocks)

        cfg = ReformerConfig(
            vocab_size=vocab_size,
            hidden_size=channel_dim,
            attention_head_size=head_dim,
            num_attention_heads=num_heads,
            num_hashes=reformer_num_hashes,
            attn_layers=attn_layers,
            feed_forward_size=int(channel_dim * mlp_ratio),
            hidden_act=_map_hidden_act(act),
            hidden_dropout_prob=max(emb_drop, proj_drop),
            local_attention_probs_dropout_prob=attn_drop,
            lsh_attention_probs_dropout_prob=attn_drop,
            pad_token_id=0 if pad_id is None else pad_id,
            max_position_embeddings=max_length,
            axial_pos_shape=axial_pos_shape,
            axial_pos_embds_dim=[channel_dim // 2, channel_dim - (channel_dim // 2)],
            local_attn_chunk_length=local_chunk,
            lsh_attn_chunk_length=lsh_chunk,
            is_decoder=False,
        )
        self.model = ReformerModel(cfg)
        # HF Reformer returns token embeddings concatenated with axial position
        # embeddings, so the output width is larger than config.hidden_size.
        self.hidden_size = channel_dim + sum(cfg.axial_pos_embds_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state


EXTERNAL_MODEL_TYPES = {
    "funnel_hf": _FunnelEncoder,
    "reformer_hf": _ReformerEncoder,
}


class ExternalModelWrapper(nn.Module):
    def __init__(
        self,
        *,
        task: str,
        vocab_size: int,
        num_labels: int,
        max_length: int = 256,
        pool: str = "mean",
        pad_id: int = None,
        emb_drop: float = 0.0,
        cls_drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pos_embed: str = "abs",
        num_blocks: int = 4,
        backend: str = "funnel_hf",
        channel_dim: int = 128,
        num_heads: int = 4,
        act: str = None,
        rmsnorm: bool = False,
        mlp_ratio: float = 4.0,
        funnel_block_sizes: Optional[List[int]] = None,
        reformer_attn_layers: Optional[List[str]] = None,
        reformer_num_hashes: int = 1,
        reformer_local_chunk_length: Optional[int] = None,
        reformer_lsh_chunk_length: Optional[int] = None,
        reformer_axial_pos_shape: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        del pos_embed, rmsnorm

        self.task = task
        self.max_length = max_length
        self.pool = pool
        self.pad_id = pad_id

        if self.task in [
            "sudoku",
            "match2",
            "match3",
            "binary_relation_composition",
            "quotient_binary_relation_composition",
        ]:
            self.pool = None
        else:
            assert pool in ["mean", "max", "cls"]

        encoder_vocab_size = vocab_size + (1 if pad_id is not None else 0)
        self.cls_token_id = None
        encoder_max_length = max_length
        if self.pool == "cls":
            self.cls_token_id = encoder_vocab_size
            encoder_vocab_size += 1
            encoder_max_length += 1
        self.encoder_input_length = encoder_max_length
        self.requires_fixed_length = backend == "reformer_hf"

        Encoder = EXTERNAL_MODEL_TYPES.get(backend, None)
        if Encoder is None:
            raise NotImplementedError(
                f"External backend {backend} not implemented. Available: {list(EXTERNAL_MODEL_TYPES.keys())}."
            )

        encoder_kwargs = dict(
            vocab_size=encoder_vocab_size,
            max_length=encoder_max_length,
            channel_dim=channel_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act=act,
            emb_drop=emb_drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pad_id=pad_id,
            pool=self.pool or "mean",
        )
        if backend == "funnel_hf":
            encoder_kwargs["funnel_block_sizes"] = funnel_block_sizes
        elif backend == "reformer_hf":
            encoder_kwargs["reformer_attn_layers"] = reformer_attn_layers
            encoder_kwargs["reformer_num_hashes"] = reformer_num_hashes
            encoder_kwargs["reformer_local_chunk_length"] = reformer_local_chunk_length
            encoder_kwargs["reformer_lsh_chunk_length"] = reformer_lsh_chunk_length
            encoder_kwargs["reformer_axial_pos_shape"] = reformer_axial_pos_shape

        self.encoder = Encoder(
            **encoder_kwargs,
        )

        hidden_size = self.encoder.hidden_size
        Norm = nn.LayerNorm
        norm_after_pool = self.pool in ["mean", "max"]
        self.final_norm = Norm(hidden_size) if norm_after_pool else nn.Identity()

        cls_dim = hidden_size * 2 if self.task == "retrieval" else hidden_size
        if task in ["image", "pathfinder32", "text", "pathfinder128"]:
            cls_proj = nn.Sequential(
                nn.Linear(cls_dim, cls_dim),
                nn.GELU(),
                nn.Linear(cls_dim, num_labels),
            )
        else:
            cls_proj = nn.Linear(cls_dim, num_labels)

        self.cls_proj = nn.Sequential(
            nn.Dropout(cls_drop),
            Norm(cls_dim) if norm_after_pool else nn.Identity(),
            cls_proj,
        )

    def _prepend_cls(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pool != "cls":
            return input_ids, attention_mask

        cls_tokens = torch.full(
            (input_ids.size(0), 1),
            fill_value=self.cls_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([cls_tokens, input_ids], dim=1)
        if attention_mask is not None:
            cls_mask = torch.ones(
                input_ids.size(0),
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        return input_ids, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_B, N = input_ids.shape

        if self.task == "retrieval":
            assert N == 2 * self.max_length, (
                f"Sequence length must be 2 * max_length for retrieval. Got {N} and {self.max_length}."
            )
            input_ids = input_ids.reshape(2 * original_B, self.max_length)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(2 * original_B, self.max_length)

        input_ids, attention_mask = self._prepend_cls(input_ids, attention_mask)
        if self.requires_fixed_length and input_ids.size(1) != self.encoder_input_length:
            pad_len = self.encoder_input_length - input_ids.size(1)
            if pad_len < 0:
                raise ValueError(
                    f"Input length {input_ids.size(1)} exceeds configured encoder length {self.encoder_input_length}."
                )
            pad_value = 0 if self.pad_id is None else self.pad_id
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_value)
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.size(0),
                    input_ids.size(1) - pad_len,
                    dtype=torch.bool,
                    device=input_ids.device,
                )
            attention_mask = F.pad(attention_mask, (0, pad_len), value=False)
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.final_norm(x)

        if self.pool == "mean":
            if attention_mask is None:
                x = x.mean(dim=1)
            else:
                weights = attention_mask.to(x.dtype).unsqueeze(-1)
                x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.pool == "max":
            if attention_mask is None:
                x = x.max(dim=1).values
            else:
                x = x.masked_fill(~attention_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        elif self.pool == "cls":
            x = x[:, 0]

        if self.task == "retrieval":
            x = x.reshape(original_B, -1)

        return self.cls_proj(x)
