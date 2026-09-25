from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from lra.__main__ import Config, make_model
from lra.models.backends import FLAREPP
from pdebench.models.flarepp import FLAREPPMixer


def _metadata() -> dict[str, object]:
    return {
        "task": "text",
        "vocab_size": 128,
        "num_labels": 3,
        "binary_classification": False,
        "max_length": 64,
    }


def test_flarepp_constructor_defaults_match_pdebench() -> None:
    signature = inspect.signature(FLAREPP)
    assert "encoder_cp_backend" not in signature.parameters
    assert "qk_norm" not in signature.parameters
    assert "qk0_norm" not in signature.parameters
    assert "q_norm" not in signature.parameters
    assert "attn_scale" not in signature.parameters
    assert signature.parameters["num_heads"].default == 8
    assert signature.parameters["num_latents"].default == 32
    assert signature.parameters["k_norm"].default is True
    assert signature.parameters["share_k0_v0"].default is True
    assert signature.parameters["rmsnorm"].default is False
    assert signature.parameters["q_fixed_norm"].default is True
    assert signature.parameters["gate_logit_init"].default == pytest.approx(0.25)


def test_flarepp_latent_layout_init_and_scale_match_pdebench() -> None:
    torch.manual_seed(0)
    mixer = FLAREPP(channel_dim=32, num_heads=4, num_latents=512)
    assert mixer.latent_q0.shape == (4, 512, 8)
    assert mixer.latent_q_fixed.shape == (4, 512, 8)
    assert mixer.latent_q0.detach().std().item() == pytest.approx(0.02, abs=0.002)
    assert mixer.attn_scale == pytest.approx(8**-0.5)


def test_flarepp_norm_gates_match_pdebench() -> None:
    on = FLAREPP(channel_dim=32, num_heads=4, k_norm=True, q_fixed_norm=True, rmsnorm=True)
    assert isinstance(on.q0_norm, nn.RMSNorm)
    assert on.q0_norm.elementwise_affine is True
    assert isinstance(on.k0_norm, nn.RMSNorm)
    assert on.k0_norm.elementwise_affine is False
    assert isinstance(on.v0_norm, nn.RMSNorm)
    assert isinstance(on.k_norm, nn.RMSNorm)
    assert on.k_norm.elementwise_affine is True
    assert isinstance(on.q_fixed_norm, nn.RMSNorm)
    assert on.q_fixed_norm.elementwise_affine is False

    off = FLAREPP(channel_dim=32, num_heads=4, k_norm=False, q_fixed_norm=False, rmsnorm=False)
    assert isinstance(off.k_norm, nn.Identity)
    assert isinstance(off.q_fixed_norm, nn.Identity)
    assert isinstance(off.q0_norm, nn.LayerNorm)


def test_flarepp_projections_match_pdebench() -> None:
    mixer = FLAREPP(channel_dim=32, num_heads=4)
    assert mixer.share_k0_v0 is True
    assert mixer.v0_proj is None
    for name in ("k0_proj", "k_proj", "v_proj"):
        proj = getattr(mixer, name)
        assert isinstance(proj, nn.Linear)
        assert proj.bias is not None


def test_flarepp_matches_pdebench_without_lra_adapters() -> None:
    torch.manual_seed(1)
    reference = FLAREPPMixer(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        k_norm=True,
        share_k0_v0=True,
        rmsnorm=True,
        q_fixed_norm=True,
        gate_logit_init=0.25,
    ).eval()
    actual = FLAREPP(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        k_norm=True,
        share_k0_v0=True,
        rmsnorm=True,
        q_fixed_norm=True,
        gate_logit_init=0.25,
    ).eval()
    ref_sd = {
        key: value
        for key, value in reference.state_dict().items()
        if not key.startswith("encoder_cp")
    }
    actual.load_state_dict(ref_sd, strict=True)
    x = torch.randn(2, 17, 32)
    with torch.no_grad():
        expected = reference(x)
        result = actual(x)
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_flarepp_retains_lra_mask_support() -> None:
    torch.manual_seed(2)
    mixer = FLAREPP(channel_dim=32, num_heads=4, num_latents=8).eval()
    x = torch.randn(2, 17, 32)
    changed = x.clone()
    changed[:, -3:] = torch.randn_like(changed[:, -3:]) * 100
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[:, -3:] = False
    with torch.no_grad():
        expected = mixer(x, attention_mask=mask)
        result = mixer(changed, attention_mask=mask)
    assert torch.allclose(result[:, :-3], expected[:, :-3], atol=1e-6, rtol=1e-6)


def test_flarepp_config_defaults_match_pdebench() -> None:
    cfg = Config()
    assert cfg.k_norm is True
    assert cfg.share_k0_v0 is True
    assert cfg.q_fixed_norm is True
    assert cfg.gate_logit_init == pytest.approx(0.25)
    assert not hasattr(cfg, "qk0_norm")
    assert not hasattr(cfg, "qk_norm")


def test_make_model_passes_flarepp_specific_controls() -> None:
    cfg = Config(
        model_type="flarepp",
        num_blocks=1,
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        pos_embed="abs",
        k_norm=False,
        share_k0_v0=False,
        q_fixed_norm=False,
        gate_logit_init=-0.5,
        rmsnorm=True,
    )
    model = make_model(cfg, _metadata(), GLOBAL_RANK=1)
    mixer = model.blocks[0].att
    assert isinstance(mixer.k_norm, nn.Identity)
    assert isinstance(mixer.q_fixed_norm, nn.Identity)
    assert mixer.share_k0_v0 is False
    assert mixer.v0_proj is not None
    assert torch.equal(mixer.gate_logit, torch.full((4,), -0.5))


def test_all_lra_flarepp_launches_use_parity_parameters() -> None:
    run_script = Path(__file__).parents[2] / "out" / "lra" / "run.sh"
    commands = [
        "torchrun " + chunk
        for chunk in run_script.read_text().split("torchrun ")[1:]
        if "--model_type flarepp" in chunk.split("\n\n", 1)[0]
    ]
    assert len(commands) == 5
    for command in commands:
        stanza = command.split("\n\n", 1)[0]
        assert "--qk0_norm" not in stanza
        assert "--qk_norm" not in stanza
        assert "--num_layers_q_proj" not in stanza
        assert "--num_layers_k_proj" not in stanza
        assert "--num_layers_v_proj" not in stanza
        assert "--num_layers_kv_proj" not in stanza
        assert "--q_proj_mlp_ratio" not in stanza
        assert "--k_proj_mlp_ratio" not in stanza
        assert "--v_proj_mlp_ratio" not in stanza
        assert "--kv_proj_mlp_ratio" not in stanza
        assert "--attn_scale" not in stanza
        assert "--q_norm" not in stanza
        assert "--k_norm" not in stanza


def test_pathfinder_flarepp_launch_uses_best_stable_sweep_config() -> None:
    run_script = (Path(__file__).parents[2] / "out" / "lra" / "run.sh").read_text()
    pathfinder = run_script.split(": <<'PATHFINDER_DISABLED'", maxsplit=1)[1]
    pathfinder = pathfinder.rsplit("PATHFINDER_DISABLED", maxsplit=1)[0]
    flarepp = pathfinder.split("# FLAREPP", maxsplit=1)[1].split("# PERFORMER", maxsplit=1)[0]

    assert "LR=3e-4" in flarepp
    assert "WEIGHT_DECAY=1e-4" in flarepp
    assert "--learning_rate ${LR}" in flarepp
    assert "--weight_decay ${WEIGHT_DECAY}" in flarepp
