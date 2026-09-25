import pytest

from pdebench.config import _coerce_model_config
from pdebench.models.mixer_backbone import (
    FLAREMixerConfig,
    FLAREPPAblationsMixerConfig,
    FLAREPPMixerConfig,
    MHAMixerConfig,
    MixerBackboneConfig,
    SimplifiedFLAREPPMixerConfig,
)


def test_coerce_mixer_flare_from_dict():
    cfg = _coerce_model_config({
        "model": "mixer_backbone",
        "num_blocks": 4,
        "mixer": {"kind": "flare", "num_latents": 32, "qk_norm": True},
    })
    assert isinstance(cfg, MixerBackboneConfig)
    assert cfg.num_blocks == 4
    assert isinstance(cfg.mixer, FLAREMixerConfig)
    assert cfg.mixer.num_latents == 32
    assert cfg.mixer.qk_norm is True


def test_coerce_mixer_simplifiedflarepp_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "mixer_backbone",
        "mixer.kind": "simplifiedflarepp",
        "mixer.num_latents": 16,
        "mixer.qk0_norm": True,
        "mixer.share_k0_v0": True,
    })
    assert isinstance(cfg.mixer, SimplifiedFLAREPPMixerConfig)
    assert cfg.mixer.num_latents == 16
    assert cfg.mixer.share_k0_v0 is True
    assert not hasattr(cfg.mixer, "use_gate")


def test_coerce_mixer_ablations_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "mixer_backbone",
        "mixer.kind": "flarepp_ablations",
        "mixer.q_fixed_norm": True,
        "mixer.use_gate": False,
        "mixer.k_elementwise_affine": True,
    })
    assert isinstance(cfg.mixer, FLAREPPAblationsMixerConfig)
    assert cfg.mixer.k_elementwise_affine is True
    assert cfg.mixer.use_gate is False


def test_coerce_mixer_flarepp_from_flattened_cli_keys():
    cfg = _coerce_model_config({
        "model": "mixer_backbone",
        "mixer.kind": "flarepp",
        "mixer.num_latents": 32,
        "mixer.k_norm": True,
        "mixer.share_k0_v0": False,
        "mixer.gate_logit_init": 0.25,
        "mixer.q_fixed_norm": True,
    })
    assert isinstance(cfg.mixer, FLAREPPMixerConfig)
    assert cfg.mixer.kind == "flarepp"
    assert cfg.mixer.num_latents == 32
    assert not hasattr(cfg.mixer, "qk0_norm")
    assert cfg.mixer.k_norm is True
    assert cfg.mixer.share_k0_v0 is False
    assert cfg.mixer.gate_logit_init == pytest.approx(0.25)
    assert cfg.mixer.q_fixed_norm is True
    assert not hasattr(cfg.mixer, "convex_mode")
    assert not hasattr(cfg.mixer, "use_gate")
    assert not hasattr(cfg.mixer, "qk_norm")
    assert not hasattr(cfg.mixer, "q_fixed_elementwise_affine")


def test_coerce_mixer_mha_from_dict():
    cfg = _coerce_model_config({
        "model": "mixer_backbone",
        "mixer": {"kind": "mha", "qk_norm": True},
    })
    assert isinstance(cfg.mixer, MHAMixerConfig)
    assert cfg.mixer.qk_norm is True


def test_coerce_mixer_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown mixer kind"):
        _coerce_model_config({
            "model": "mixer_backbone",
            "mixer": {"kind": "not_a_mixer"},
        })


def test_coerce_mixer_default_when_omitted():
    cfg = _coerce_model_config({"model": "mixer_backbone"})
    assert isinstance(cfg.mixer, FLAREMixerConfig)
    assert cfg.mixer.kind == "flare"
