from __future__ import annotations

import torch
from torch import nn

from pdebench.config import MODEL_CONFIG_BY_MODEL, Config
from pdebench.models.abupt_mixer import ABUPTTransformerBlock
from pdebench.models.abupt_surface_mixer import ABUPTSurfaceMixerConfig, ABUPTSurfaceMixerModel
from pdebench.models.model_factory import _resolve_model_spec


def test_surface_mixer_is_registered_and_factory_resolves_it() -> None:
    assert MODEL_CONFIG_BY_MODEL["abupt_surface_mixer"] is ABUPTSurfaceMixerConfig
    cfg = Config(model={"model": "abupt_surface_mixer", "num_blocks": 3})
    resolved, c_in, c_out, model_name, model_ctor = _resolve_model_spec(
        cfg,
        {"dataset": "drivaerml_surface", "c_in": 6, "c_out": 4, "space_dim": 3, "fun_dim": 0},
    )
    assert resolved.model.model == "abupt_surface_mixer"
    assert (c_in, c_out) == (6, 4)
    assert model_name == "AB-UPT-Surface-Mixer"
    assert model_ctor is ABUPTSurfaceMixerModel


def test_surface_mixer_has_backbone_equivalent_depth_and_surface_shape() -> None:
    model = ABUPTSurfaceMixerModel(
        ABUPTSurfaceMixerConfig(channel_dim=24, num_heads=3, num_blocks=3),
        metadata={"c_in": 6, "c_out": 4},
    ).eval()
    assert len(model.blocks) == 3
    with torch.no_grad():
        output = model(torch.rand(2, 11, 6))
    assert output.shape == (2, 11, 4)
    assert torch.isfinite(output).all()


def test_surface_mixer_has_only_surface_blocks_without_geometry_or_perceiver() -> None:
    for depth in (2, 8):
        model = ABUPTSurfaceMixerModel(ABUPTSurfaceMixerConfig(channel_dim=24, num_heads=3, num_blocks=depth))

        assert len(model.blocks) == depth
        assert all(isinstance(block, ABUPTTransformerBlock) for block in model.blocks)
        assert all(block.kind == "s" for block in model.blocks)
        assert not hasattr(model, "geometry")


def test_surface_mixer_matches_rmsnorm_and_ffn_configuration() -> None:
    config = ABUPTSurfaceMixerConfig(
        channel_dim=24,
        num_heads=3,
        num_blocks=2,
        rmsnorm=True,
        num_layers_ffn=0,
        ffn_mlp_ratio=2.0,
    )
    model = ABUPTSurfaceMixerModel(config, metadata={"c_in": 6, "c_out": 4})
    block = model.blocks[1]
    assert isinstance(block, ABUPTTransformerBlock)
    assert isinstance(block.norm1, nn.RMSNorm)
    assert isinstance(block.norm2, nn.RMSNorm)
    assert block.mlp.fc1.out_features == 48
    assert len(block.mlp.fcs) == 0
