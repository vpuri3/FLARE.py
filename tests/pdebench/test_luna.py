from __future__ import annotations

import pytest
import torch

from pdebench.models.luna import LunaConfig


def test_luna_config_defaults() -> None:
    cfg = LunaConfig()
    assert cfg.model == "luna"
    assert cfg.num_blocks == 8
    assert cfg.channel_dim == 128
    assert cfg.num_heads == 8
    assert cfg.act is None
    assert cfg.rmsnorm is False
    assert cfg.out_proj_norm is True
    assert cfg.num_layers_in_out_proj == 2
    assert cfg.num_layers_ffn == 0
    assert cfg.ffn_mlp_ratio == 2.0
    assert cfg.qk_norm is False
    assert cfg.num_latents == 64
    for removed in (
        "num_layers_k_proj",
        "num_layers_v_proj",
        "k_proj_mlp_ratio",
        "v_proj_mlp_ratio",
        "attn_scale",
        "encoder_cp_backend",
    ):
        assert removed not in LunaConfig.__dataclass_fields__


def test_luna_attn_scale_is_inv_sqrt_head_dim() -> None:
    from pdebench.models.luna import LunaEncoderAttention

    attn = LunaEncoderAttention(channel_dim=32, num_heads=4, num_latents=8)
    assert attn.attn_scale == pytest.approx(attn.head_dim ** -0.5)


def test_luna_kv_projections_are_linear() -> None:
    from pdebench.models.luna import LunaEncoderAttention

    attn = LunaEncoderAttention(channel_dim=32, num_heads=4, num_latents=8)
    for name in ("pq_proj", "pk_proj", "pv_proj", "q_proj", "k_proj", "v_proj", "out_proj"):
        assert isinstance(getattr(attn, name), torch.nn.Linear)


def test_luna_encoder_attention_pack_unpack_shapes() -> None:
    from pdebench.models.luna import LunaEncoderAttention

    attn = LunaEncoderAttention(channel_dim=32, num_heads=4, num_latents=8)
    x = torch.randn(2, 16, 32)
    p = torch.randn(2, 8, 32)
    yx, yp = attn(x, p)
    assert yx.shape == (2, 16, 32)
    assert yp.shape == (2, 8, 32)


def _meta():
    return {"c_in": 4, "c_out": 3, "dataset": "elasticity"}


def _cfg(**kwargs) -> LunaConfig:
    base = dict(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_latents=8,
        num_layers_in_out_proj=-1,
    )
    base.update(kwargs)
    return LunaConfig(**base)


def test_luna_model_forward_shape() -> None:
    from pdebench.models.luna import LunaModel

    model = LunaModel(_cfg(), metadata=_meta())
    y = model(torch.randn(2, 16, 4))
    assert y.shape == (2, 16, 3)


def test_luna_model_carries_packed_stream() -> None:
    from pdebench.models.luna import LunaModel

    model = LunaModel(_cfg(), metadata=_meta())
    assert model.packed_embed.shape == (8, 32)
    x = model.in_proj(torch.randn(2, 16, 4))
    p = model.packed_embed.unsqueeze(0).expand(2, -1, -1)
    x2, p2 = model.blocks[0](x, p)
    assert x2.shape == (2, 16, 32)
    assert p2.shape == (2, 8, 32)
    assert not torch.equal(p2, p)


def test_luna_in_model_config_map() -> None:
    from pdebench.config import MODEL_CONFIG_BY_MODEL
    from pdebench.models.luna import LunaConfig as CfgFromConfig

    assert MODEL_CONFIG_BY_MODEL["luna"] is CfgFromConfig


def test_luna_factory_resolves_model_class() -> None:
    from pdebench.config import Config, DatasetConfig, OptimizerConfig, RunConfig, TrainingConfig
    from pdebench.models import model_factory

    cfg = Config(
        run=RunConfig(),
        dataset=DatasetConfig(dataset="elasticity"),
        training=TrainingConfig(),
        optimizer=OptimizerConfig(),
        scheduler={},
        model={"model": "luna"},
    )
    metadata = {"c_in": 2, "c_out": 1, "dataset": "elasticity", "space_dim": 2}
    cfg, _c_in, _c_out, model_name, Model = model_factory._resolve_model_spec(cfg, metadata)
    assert model_name == "Luna"
    assert Model.__name__ == "LunaModel"
    assert cfg.model.model == "luna"


def test_luna_in_mesh_sequence_models() -> None:
    from pdebench.dataset.mesh_runtime import MESH_SEQUENCE_MODELS

    assert "luna" in MESH_SEQUENCE_MODELS
