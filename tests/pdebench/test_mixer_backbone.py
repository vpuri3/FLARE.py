from __future__ import annotations

import math
import re
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import pdebench
from pdebench.config import (
    MODEL_CONFIG_BY_MODEL,
    Config,
    DatasetConfig,
    MixerBackboneConfig,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
)
from pdebench.config import (
    MixerBackboneConfig as CfgFromConfig,
)
from pdebench.dataset.mesh_runtime import MESH_SEQUENCE_MODELS
from pdebench.models import model_factory
from pdebench.models.mixer_backbone import (
    MIXER_BY_KIND,
    FLAREMixer,
    FLAREMixerConfig,
    FLAREPPAblations,
    FLAREPPAblationsMixerConfig,
    FLAREPPMixer,
    FLAREPPMixerConfig,
    MHAMixer,
    MHAMixerConfig,
    MixerBackboneModel,
    MixerConfig,
    SimplifiedFLAREPPMixer,
    SimplifiedFLAREPPMixerConfig,
    Transolver3MixerConfig,
    TransolverPPMixerConfig,
    build_mixer,
)


def _meta():
    return {"c_in": 4, "c_out": 3, "dataset": "elasticity"}


def _backbone(**kwargs) -> MixerBackboneConfig:
    base = dict(channel_dim=32, num_blocks=2, num_heads=4, num_layers_ffn=2, mlp_ratio_ffn=1.0, rmsnorm=True)
    base.update(kwargs)
    return MixerBackboneConfig(**base)


def test_build_mixer_dispatches_by_kind():
    bb = _backbone(rmsnorm=True)
    mha = build_mixer(MHAMixerConfig(), bb)
    assert isinstance(mha, MHAMixer)
    pp = build_mixer(SimplifiedFLAREPPMixerConfig(num_latents=8), bb)
    assert isinstance(pp, SimplifiedFLAREPPMixer)
    assert pp.num_latents == 8


def test_flare_default_config_flow():
    bb = _backbone(num_heads=None, rmsnorm=True)

    flare = FLAREMixer(FLAREMixerConfig(), bb)
    assert flare.num_latents == 64
    assert flare.num_heads == 4
    assert isinstance(flare.q_norm, torch.nn.Identity)
    assert isinstance(flare.k_norm, torch.nn.Identity)

    simplifiedflarepp = SimplifiedFLAREPPMixer(SimplifiedFLAREPPMixerConfig(), bb)
    assert simplifiedflarepp.num_latents == 64
    assert simplifiedflarepp.num_heads == 4
    assert isinstance(simplifiedflarepp.q0_norm, torch.nn.RMSNorm)
    assert isinstance(simplifiedflarepp.k0_norm, torch.nn.RMSNorm)
    assert not hasattr(simplifiedflarepp, "gate_logit")

    ablations = FLAREPPAblations(FLAREPPAblationsMixerConfig(), bb)
    assert ablations.num_heads == 4


def _cfg(**kwargs) -> MixerBackboneConfig:
    mixer_arg = kwargs.pop("mixer", "flare")
    if isinstance(mixer_arg, str):
        if mixer_arg not in MIXER_BY_KIND:
            raise ValueError(f"unknown mixer kind {mixer_arg!r}")
        mixer_cls = MIXER_BY_KIND[mixer_arg][0]
        mixer_field_names = {f.name for f in fields(mixer_cls)} - {"kind"}
        mixer_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in mixer_field_names}
        mixer: MixerConfig = mixer_cls(**mixer_kwargs)
    else:
        mixer = mixer_arg

    base = dict(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_layers_ffn=2,
        mlp_ratio_ffn=1.0,
        mixer=mixer,
    )
    base.update(kwargs)
    return MixerBackboneConfig(**base)


@pytest.mark.parametrize("mixer", ["mha", "flare", "simplifiedflarepp", "transolver", "transolverpp", "transolver3"])
def test_forward_shape(mixer: str) -> None:
    model = MixerBackboneModel(_cfg(mixer=mixer), metadata=_meta())
    x = torch.randn(2, 16, 4)
    y = model(x)
    assert y.shape == (2, 16, 3)


def test_invalid_mixer_raises() -> None:
    cfg = _cfg()
    cfg.mixer = SimpleNamespace(kind="not_a_mixer")  # type: ignore[assignment]
    allowed_kinds = re.escape(str(sorted(MIXER_BY_KIND)))
    with pytest.raises(
        TypeError,
        match=rf"mixer\.kind='not_a_mixer'.*allowed mixer kinds: {allowed_kinds}",
    ):
        MixerBackboneModel(cfg, metadata=_meta())


def test_mha_has_no_latent_parameters() -> None:
    model = MixerBackboneModel(_cfg(mixer="mha"), metadata=_meta())
    names = {n for n, _ in model.named_parameters()}
    assert not any("latent_q" in n for n in names)
    assert not any("in_project_slice" in n for n in names)


def test_flare_uses_num_latents() -> None:
    model = MixerBackboneModel(_cfg(mixer="flare", num_latents=8), metadata=_meta())
    latents = [p for n, p in model.named_parameters() if "latent_q" in n]
    assert len(latents) == 2  # one per block
    # latent_q is stored as [H, M, D]
    assert latents[0].shape == (4, 8, 8)


def test_simplifiedflarepp_uses_num_latents_and_k0_v0_proj() -> None:
    model = MixerBackboneModel(_cfg(mixer="simplifiedflarepp", num_latents=8), metadata=_meta())
    names = {n for n, _ in model.named_parameters()}
    latents = [p for n, p in model.named_parameters() if "latent_q0" in n]
    assert len(latents) == 2  # one per block
    assert latents[0].shape == (4, 8, 8)
    assert any("k0_proj" in n for n in names)
    assert any("v0_proj" in n for n in names)


def test_simplifiedflarepp_defaults_match_qk0_norm_ablation_preset() -> None:
    from pdebench.models.mixer_backbone import ResidualMLP, SimplifiedFLAREPPMixer

    mixer = SimplifiedFLAREPPMixer(SimplifiedFLAREPPMixerConfig(num_latents=8), _backbone(rmsnorm=False))
    for name in ("k0_proj", "v0_proj", "k_proj", "v_proj"):
        proj = getattr(mixer, name)
        assert isinstance(proj, ResidualMLP)
        assert proj.num_layers == -1
        assert proj.residual
    # Defaults match the structural parts of flarepp_ablations ABLATION_CONFIG=qk0_norm:
    # qk0_norm=true, qk_norm=false, share_k0_v0=false
    assert isinstance(mixer.q0_norm, torch.nn.LayerNorm)
    assert isinstance(mixer.k0_norm, torch.nn.LayerNorm)
    assert isinstance(mixer.q_norm, torch.nn.Identity)
    assert isinstance(mixer.k_norm, torch.nn.Identity)
    assert isinstance(mixer.out_proj, torch.nn.Linear)
    assert mixer.share_k0_v0 is False
    assert not hasattr(mixer, "use_gate")
    assert not hasattr(mixer, "gate_logit")
    assert not hasattr(mixer, "latent_q_fixed")
    assert not hasattr(mixer, "v0_norm")
    assert not hasattr(mixer, "encoder_cp")
    assert not hasattr(mixer, "set_context_parallel")


@pytest.mark.parametrize(
    ("mixer_cls", "projection_names"),
    [
        ("FLAREMixer", ("k_proj", "v_proj")),
        ("SimplifiedFLAREPPMixer", ("k0_proj", "v0_proj", "k_proj", "v_proj")),
    ],
)
def test_flare_projections_use_fixed_residual_mlp_formula(mixer_cls: str, projection_names: tuple[str, ...]) -> None:
    from pdebench.models import mixer_backbone

    cls = getattr(mixer_backbone, mixer_cls)
    config_cls = FLAREMixerConfig if mixer_cls == "FLAREMixer" else SimplifiedFLAREPPMixerConfig
    mixer = cls(config_cls(num_latents=2), _backbone(channel_dim=4, num_heads=1, rmsnorm=False))
    x = torch.tensor([[[1.0, -2.0, 0.5, -0.25]]])
    for name in projection_names:
        proj = getattr(mixer, name)
        assert isinstance(proj, mixer_backbone.ResidualMLP)
        assert proj.num_layers == -1
        assert proj.residual
        with torch.no_grad():
            proj.fc.weight.copy_(0.25 * torch.eye(4))
            proj.fc.bias.zero_()
        torch.testing.assert_close(proj(x), x + proj.fc(x))


def test_simplifiedflarepp_qk0_norm_gates_hop1_independent_of_qk_norm() -> None:
    """qk0_norm gates hop-1 q0/k0 norms; qk_norm gates hop-2 q/k."""
    from pdebench.models.mixer_backbone import SimplifiedFLAREPPMixer

    off = SimplifiedFLAREPPMixer(
        SimplifiedFLAREPPMixerConfig(num_latents=8, qk0_norm=False, qk_norm=False),
        _backbone(rmsnorm=True),
    )
    assert isinstance(off.q0_norm, torch.nn.Identity)
    assert isinstance(off.k0_norm, torch.nn.Identity)
    assert isinstance(off.q_norm, torch.nn.Identity)
    assert isinstance(off.k_norm, torch.nn.Identity)

    hop1 = SimplifiedFLAREPPMixer(
        SimplifiedFLAREPPMixerConfig(num_latents=8, qk0_norm=True, qk_norm=False),
        _backbone(rmsnorm=True),
    )
    assert isinstance(hop1.q0_norm, torch.nn.RMSNorm)
    assert isinstance(hop1.k0_norm, torch.nn.RMSNorm)
    assert hop1.q0_norm.normalized_shape == (hop1.head_dim,)
    assert hop1.k0_norm.normalized_shape == (hop1.head_dim,)
    assert isinstance(hop1.q_norm, torch.nn.Identity)
    assert isinstance(hop1.k_norm, torch.nn.Identity)

    hop2 = SimplifiedFLAREPPMixer(
        SimplifiedFLAREPPMixerConfig(num_latents=8, qk0_norm=False, qk_norm=True),
        _backbone(rmsnorm=True),
    )
    assert isinstance(hop2.q0_norm, torch.nn.Identity)
    assert isinstance(hop2.k0_norm, torch.nn.Identity)
    assert isinstance(hop2.q_norm, torch.nn.RMSNorm)
    assert isinstance(hop2.k_norm, torch.nn.RMSNorm)

    ln = SimplifiedFLAREPPMixer(
        SimplifiedFLAREPPMixerConfig(num_latents=8, qk0_norm=True, qk_norm=False),
        _backbone(rmsnorm=False),
    )
    assert isinstance(ln.q0_norm, torch.nn.LayerNorm)
    assert isinstance(ln.k0_norm, torch.nn.LayerNorm)


def test_simplifiedflarepp_share_k0_v0_drops_v0_proj_and_forwards() -> None:
    from pdebench.models.mixer_backbone import ResidualMLP, SimplifiedFLAREPPMixer

    shared = SimplifiedFLAREPPMixer(
        SimplifiedFLAREPPMixerConfig(num_latents=8, share_k0_v0=True),
        _backbone(rmsnorm=False),
    )
    assert shared.share_k0_v0 is True
    assert shared.v0_proj is None
    assert isinstance(shared.k0_proj, ResidualMLP)
    y = shared(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)

    model = MixerBackboneModel(_cfg(mixer="simplifiedflarepp", share_k0_v0=True), metadata=_meta())
    names = {n for n, _ in model.named_parameters()}
    assert any("k0_proj" in n for n in names)
    assert not any("v0_proj" in n for n in names)
    assert model(torch.randn(2, 16, 4)).shape == (2, 16, 3)

    with pytest.raises(TypeError, match="share_k0_v0"):
        SimplifiedFLAREPPMixer(
            SimplifiedFLAREPPMixerConfig(num_latents=8, share_k0_v0=1),  # type: ignore[arg-type]
            _backbone(rmsnorm=False),
        )


@pytest.mark.parametrize("mixer", ["flare", "simplifiedflarepp"])
def test_latent_q_normal_init_std(mixer: str) -> None:
    torch.manual_seed(0)
    model = MixerBackboneModel(_cfg(mixer=mixer), metadata=_meta())
    key = "latent_q0" if mixer == "simplifiedflarepp" else "latent_q"
    latent = next(p for n, p in model.named_parameters() if key in n)
    assert abs(latent.detach().std().item() - 0.02) < 0.01


@pytest.mark.parametrize("mixer", ["transolver", "transolverpp", "transolver3"])
def test_transolver_mixers_use_num_latents_as_slices(mixer: str) -> None:
    model = MixerBackboneModel(_cfg(mixer=mixer, num_latents=8), metadata=_meta())
    slices = [p for n, p in model.named_parameters() if "in_project_slice" in n and n.endswith("weight")]
    assert len(slices) == 2
    # Linear(dim_head, slice_num): weight shape [slice_num, dim_head]
    assert slices[0].shape[0] == 8


def test_transolverpp_matches_upstream_attention() -> None:
    from pdebench.models.mixer_backbone import TransolverPPMixer
    from tests.pdebench.fixtures.upstream_transolver_attn import UpstreamTransolverPPAttention

    kwargs = dict(dim=32, heads=4, dim_head=8, dropout=0.0, slice_num=8)
    torch.manual_seed(0)
    ours = TransolverPPMixer(TransolverPPMixerConfig(num_latents=8), _backbone())
    torch.manual_seed(0)
    ref = UpstreamTransolverPPAttention(**kwargs)
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 16, 32)
    torch.manual_seed(123)
    y_ours = ours(x)
    torch.manual_seed(123)
    y_ref = ref(x)
    torch.testing.assert_close(y_ours, y_ref, rtol=0.0, atol=0.0)


def test_transolver3_matches_upstream_attention() -> None:
    from pdebench.models.mixer_backbone import Transolver3Mixer
    from tests.pdebench.fixtures.upstream_transolver_attn import UpstreamTransolver3Attention

    kwargs = dict(dim=32, heads=4, dim_head=8, dropout=0.0, slice_num=8)
    torch.manual_seed(0)
    ours = Transolver3Mixer(Transolver3MixerConfig(num_latents=8), _backbone())
    torch.manual_seed(0)
    ref = UpstreamTransolver3Attention(**kwargs)
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_ours = ours(x)
        y_ref = ref(x)
    torch.testing.assert_close(y_ours, y_ref, rtol=0.0, atol=0.0)


def test_mixer_backbone_config_default_mixer_is_flare():
    cfg = MixerBackboneConfig()
    assert cfg.model == "mixer_backbone"
    assert isinstance(cfg.mixer, FLAREMixerConfig)
    assert cfg.mixer.kind == "flare"
    assert cfg.mixer.num_latents == 64
    assert cfg.mixer.qk_norm is False
    flat_names = {f.name for f in fields(MixerBackboneConfig)}
    assert "num_latents" not in flat_names
    assert "use_gate" not in flat_names
    assert "qk0_norm" not in flat_names


def test_simplifiedflarepp_and_ablations_config_defaults():
    pp = SimplifiedFLAREPPMixerConfig()
    assert pp.qk0_norm is True
    assert pp.share_k0_v0 is False
    assert not hasattr(pp, "use_gate")
    assert not hasattr(pp, "gate_logit_init")
    assert not hasattr(pp, "v0_norm")
    ab = FLAREPPAblationsMixerConfig()
    assert ab.use_gate is False and ab.v0_norm is False
    assert ab.q_fixed_norm is True and ab.v_use_residual is True
    assert ab.q0_norm is False and ab.q_fixed_elementwise_affine is False
    mha = MHAMixerConfig()
    assert mha.qk_norm is False


def test_config_defaults() -> None:
    cfg = MixerBackboneConfig()
    assert cfg.model == "mixer_backbone"
    assert cfg.channel_dim == 128
    assert cfg.num_heads == 8
    assert cfg.num_layers_in_out_proj == 2
    assert cfg.out_proj_norm is True
    assert cfg.num_layers_ffn == 0
    assert cfg.mlp_ratio_ffn == 2.0
    assert cfg.rmsnorm is None
    assert cfg.diagnostics is False
    assert "attn_scale" not in MixerBackboneConfig.__dataclass_fields__
    assert "encoder_cp_backend" not in MixerBackboneConfig.__dataclass_fields__


def test_config_rmsnorm_default_is_none() -> None:
    assert MixerBackboneConfig().rmsnorm is None


def test_mixer_backbone_rmsnorm_auto_true_under_mp_bf16() -> None:
    cfg = _make_cfg(mixer="flare", channel_dim=32, num_blocks=1, num_heads=4, num_latents=8)
    cfg.training.mixed_precision = True
    cfg.training.amp_dtype = "bf16"
    metadata = dict(c_in=4, c_out=3, space_dim=2, fun_dim=0)
    model_factory._resolve_model_spec(cfg, metadata)
    assert cfg.model.rmsnorm is True


def test_mixer_backbone_rmsnorm_explicit_false_wins() -> None:
    cfg = _make_cfg(
        mixer="flare", channel_dim=32, num_blocks=1, num_heads=4, num_latents=8, rmsnorm=False,
    )
    cfg.training.mixed_precision = True
    cfg.training.amp_dtype = "bf16"
    metadata = dict(c_in=4, c_out=3, space_dim=2, fun_dim=0)
    model_factory._resolve_model_spec(cfg, metadata)
    assert cfg.model.rmsnorm is False


def _make_cfg(**model_kwargs):
    model = _cfg(**model_kwargs)
    return Config(
        run=RunConfig(),
        dataset=DatasetConfig(dataset="elasticity"),
        training=TrainingConfig(),
        optimizer=OptimizerConfig(),
        model=model,
        use_puri2025flare_config=False,
    )


def test_public_export() -> None:
    assert hasattr(pdebench, "MixerBackboneModel")
    assert MixerBackboneConfig.model == "mixer_backbone"


def test_registered_in_model_config_map() -> None:
    assert MODEL_CONFIG_BY_MODEL["mixer_backbone"] is CfgFromConfig


def test_in_mesh_sequence_models() -> None:
    assert "mixer_backbone" in MESH_SEQUENCE_MODELS


@pytest.mark.parametrize("mixer", ["mha", "flare", "simplifiedflarepp", "transolver", "transolverpp", "transolver3"])
def test_resolve_model_spec_mixer_backbone(mixer: str) -> None:
    cfg = _make_cfg(mixer=mixer, channel_dim=32, num_blocks=1, num_heads=4)
    metadata = dict(c_in=4, c_out=3, space_dim=2, fun_dim=0)
    cfg_out, c_in, c_out, model_name, model_ctor = model_factory._resolve_model_spec(cfg, metadata)
    assert cfg_out is cfg
    assert (c_in, c_out) == (4, 3)
    assert model_name == "MixerBackbone"
    assert model_ctor is pdebench.MixerBackboneModel


def test_mixer_block_ffn_has_no_inner_residual() -> None:
    """Block is x+ffn(norm(x)); FFN must not also use I+W residuals."""
    from pdebench.models.mixer_backbone import MixerBackboneBlock

    block = MixerBackboneBlock(
        mixer=FLAREMixerConfig(num_latents=8),
        backbone_config=_backbone(num_layers_ffn=2, mlp_ratio_ffn=1.0),
    )
    assert block.ffn.input_residual is False
    assert block.ffn.output_residual is False


@pytest.mark.parametrize("mixer_cls", ["FLAREMixer", "SimplifiedFLAREPPMixer"])
def test_mixer_attn_scale_is_inv_sqrt_head_dim(mixer_cls: str) -> None:
    from pdebench.models import mixer_backbone

    config_cls = FLAREMixerConfig if mixer_cls == "FLAREMixer" else SimplifiedFLAREPPMixerConfig
    mixer = getattr(mixer_backbone, mixer_cls)(config_cls(num_latents=8), _backbone(rmsnorm=False))
    assert mixer.attn_scale == pytest.approx(mixer.head_dim ** -0.5)


def test_mha_mixer_class_name() -> None:
    from pdebench.models.mixer_backbone import MHAMixer, MixerBackboneBlock

    block = MixerBackboneBlock(
        mixer=MHAMixerConfig(),
        backbone_config=_backbone(num_layers_ffn=0),
    )
    assert isinstance(block.mixer, MHAMixer)


def _shared_state_dict(src: torch.nn.Module, dst: torch.nn.Module) -> dict[str, torch.Tensor]:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {k: src_sd[k] for k in dst_sd if k in src_sd}
    missing = sorted(set(dst_sd) - set(shared))
    extra = sorted(set(src_sd) - set(dst_sd))
    # flarepp keeps encoder_cp buffers/params that mixer_backbone does not.
    assert missing == [], f"dst missing shared keys: {missing}"
    assert all(k.startswith("encoder_cp") for k in extra), f"unexpected extra keys: {extra}"
    return shared


@pytest.mark.parametrize("rmsnorm", [False, True])
@pytest.mark.parametrize(
    "k_norm,q_fixed_norm,share_k0_v0",
    [
        (True, True, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_mixer_backbone_anchored_matches_package_flarepp(
    rmsnorm: bool, k_norm: bool, q_fixed_norm: bool, share_k0_v0: bool,
) -> None:
    from pdebench.models.flarepp import FLAREPPMixer as PackageFLAREPPMixer

    anchored_config = FLAREPPMixerConfig(
        num_latents=8,
        k_norm=k_norm,
        q_fixed_norm=q_fixed_norm,
        share_k0_v0=share_k0_v0,
        gate_logit_init=0.25,
    )
    bb = _backbone(
        channel_dim=32,
        num_heads=4,
        rmsnorm=rmsnorm,
        diagnostics=False,
    )
    torch.manual_seed(0)
    anchored = FLAREPPMixer(anchored_config, bb)
    torch.manual_seed(0)
    package = PackageFLAREPPMixer(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        rmsnorm=rmsnorm,
        k_norm=k_norm,
        q_fixed_norm=q_fixed_norm,
        share_k0_v0=share_k0_v0,
        gate_logit_init=0.25,
        encoder_cp_backend="flash",
    )

    assert anchored.attn_scale == pytest.approx(package.attn_scale)
    package.load_state_dict(_shared_state_dict(anchored, package), strict=False)

    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_anchored = anchored(x)
        y_package = package(x)
    torch.testing.assert_close(y_anchored, y_package, rtol=0.0, atol=0.0)


def test_ablations_q_fixed_norm_defaults_and_applies_before_gate() -> None:
    from pdebench.models.mixer_backbone import FLAREPPAblations

    on = FLAREPPAblations(
        FLAREPPAblationsMixerConfig(num_latents=8, use_gate=True),
        _backbone(rmsnorm=True),
    )
    assert isinstance(on.q_fixed_norm, torch.nn.RMSNorm)
    assert on.q_fixed_norm.elementwise_affine is False

    off = FLAREPPAblations(
        FLAREPPAblationsMixerConfig(num_latents=8, use_gate=True, q_fixed_norm=False),
        _backbone(rmsnorm=True),
    )
    assert isinstance(off.q_fixed_norm, torch.nn.Identity)

    # Norm is applied to q_fixed before the gate mix (smoke: gated forward runs).
    y = on(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)

    cfg = MixerBackboneConfig(mixer=FLAREPPAblationsMixerConfig())
    assert cfg.mixer.q_fixed_norm is True
    assert cfg.mixer.q_fixed_elementwise_affine is False


#======================================================================#
# Diagnostics (LP screening wave)
#======================================================================#

_FLAREPP_HOP1_KEYS = frozenset({
    "rms_q_dynamic", "rms_k0", "rms_v0", "rms_k", "rms_v",
    "synth_attn_entropy", "synth_n_eff", "latent_q_offdiag_cos", "gate_value",
})
_STREAM_KEYS = frozenset({"residual_stream_rms", "mixer_out_rms", "mixer_over_stream"})


def test_diagnostics_default_false_is_noop() -> None:
    model = MixerBackboneModel(_cfg(mixer="flarepp_ablations"), metadata=_meta())
    assert model.diagnostics is False
    y = model(torch.randn(2, 16, 4))
    assert y.shape == (2, 16, 3)
    assert model.last_diagnostics is None
    for block in model.blocks:
        assert block.last_diagnostics is None
        assert block.mixer.last_diagnostics is None


def test_diagnostics_true_populates_hop1_keys() -> None:
    model = MixerBackboneModel(
        _cfg(mixer="flarepp_ablations", diagnostics=True, use_gate=True),
        metadata=_meta(),
    )
    assert model.diagnostics is True
    y = model(torch.randn(2, 16, 4))
    assert y.shape == (2, 16, 3)

    diagnostics = model.last_diagnostics
    assert diagnostics is not None
    assert len(diagnostics["blocks"]) == len(model.blocks)

    for block_diag in diagnostics["blocks"]:
        assert _STREAM_KEYS <= block_diag.keys()
        assert _FLAREPP_HOP1_KEYS <= block_diag.keys()
        for key in _STREAM_KEYS | (_FLAREPP_HOP1_KEYS - {"gate_value"}):
            assert math.isfinite(block_diag[key]), f"{key}={block_diag[key]!r} is not finite"

    mean = diagnostics["mean"]
    assert _STREAM_KEYS <= mean.keys()
    assert _FLAREPP_HOP1_KEYS <= mean.keys()
    for key in _STREAM_KEYS | (_FLAREPP_HOP1_KEYS - {"gate_value"}):
        assert math.isfinite(mean[key]), f"mean {key}={mean[key]!r} is not finite"
    assert math.isfinite(mean["gate_value"])


@pytest.mark.parametrize("mixer", ["simplifiedflarepp", "flarepp"])
def test_slim_flarepp_mixers_have_no_mixer_diagnostics(mixer: str) -> None:
    model = MixerBackboneModel(_cfg(mixer=mixer, diagnostics=True), metadata=_meta())
    model(torch.randn(2, 16, 4))
    for block in model.blocks:
        assert not hasattr(block.mixer, "diagnostics")
        assert not hasattr(block.mixer, "last_diagnostics")
    # Block/model stream diagnostics still populate.
    assert model.last_diagnostics is not None
    for block_diag in model.last_diagnostics["blocks"]:
        assert _STREAM_KEYS <= block_diag.keys()
        assert not (_FLAREPP_HOP1_KEYS & block_diag.keys())


@pytest.mark.parametrize("mixer", ["mha", "flare"])
def test_diagnostics_true_non_flarepp_mixers_have_only_stream_keys(mixer: str) -> None:
    """MHA/FLARE have no hop-1 synthesis; only block-level stream diagnostics apply."""
    model = MixerBackboneModel(_cfg(mixer=mixer, diagnostics=True), metadata=_meta())
    model(torch.randn(2, 16, 4))
    for block_diag in model.last_diagnostics["blocks"]:
        assert set(block_diag.keys()) == _STREAM_KEYS
        for key in _STREAM_KEYS:
            assert math.isfinite(block_diag[key])


def test_diagnostics_flag_does_not_change_forward_output() -> None:
    torch.manual_seed(0)
    model_off = MixerBackboneModel(_cfg(mixer="flarepp_ablations", diagnostics=False), metadata=_meta())
    torch.manual_seed(0)
    model_on = MixerBackboneModel(_cfg(mixer="flarepp_ablations", diagnostics=True), metadata=_meta())

    x = torch.randn(2, 16, 4)
    with torch.no_grad():
        y_off = model_off(x)
        y_on = model_on(x)
    torch.testing.assert_close(y_off, y_on, rtol=0.0, atol=0.0)


def test_diagnostics_callback_registered_and_writes_jsonl(tmp_path) -> None:
    from pdebench.callbacks import MixerDiagnosticsCallback

    case_dir = tmp_path / "run"
    model = MixerBackboneModel(_cfg(mixer="flarepp_ablations", diagnostics=True), metadata=_meta())
    model(torch.randn(2, 16, 4))

    class _FakeTrainer:
        GLOBAL_RANK = 0
        step = 1
        epoch = 0
        train_loss_per_batch = [0.5]
        grad_norm_per_step = [1.25]
        time_per_step = [0.01]
        stats_every = 10
        log_rank_every_steps = 0

        def __init__(self, model):
            self.model = model

    trainer = _FakeTrainer(model)
    callback = MixerDiagnosticsCallback(str(case_dir))
    callback(trainer)

    jsonl_path = case_dir / "diagnostics.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1

    import json
    record = json.loads(lines[0])
    assert record["step"] == 1
    assert record["train_loss"] == 0.5
    assert record["grad_norm"] == 1.25
    assert any(key.startswith("mean_") for key in record)
    assert callback.first_nonfinite_step is None

    summary_path = case_dir / "mixer_diagnostics_summary.json"
    assert summary_path.exists()


def test_diagnostics_callback_flags_first_nonfinite_step(tmp_path) -> None:
    from pdebench.callbacks import MixerDiagnosticsCallback

    case_dir = tmp_path / "run"
    model = MixerBackboneModel(_cfg(mixer="simplifiedflarepp", diagnostics=True), metadata=_meta())
    model(torch.randn(2, 16, 4))

    class _FakeTrainer:
        GLOBAL_RANK = 0
        step = 7
        epoch = 0
        train_loss_per_batch = [float("nan")]
        grad_norm_per_step = [1.0]
        time_per_step = [0.01]
        stats_every = 10
        log_rank_every_steps = 0

        def __init__(self, model):
            self.model = model

    trainer = _FakeTrainer(model)
    callback = MixerDiagnosticsCallback(str(case_dir))
    callback(trainer)

    assert callback.first_nonfinite_step == 7


def test_flarepp_config_defaults() -> None:
    cfg = FLAREPPMixerConfig()
    assert cfg.kind == "flarepp"
    assert cfg.num_latents == 64
    assert cfg.k_norm is True
    assert cfg.share_k0_v0 is True
    assert cfg.gate_logit_init == pytest.approx(0.25)
    assert cfg.q_fixed_norm is True
    assert not hasattr(cfg, "qk0_norm")
    assert not hasattr(cfg, "convex_mode")
    assert not hasattr(cfg, "use_gate")
    assert not hasattr(cfg, "qk_norm")
    assert not hasattr(cfg, "q_norm")
    assert not hasattr(cfg, "v0_norm")
    assert not hasattr(cfg, "q_fixed_elementwise_affine")


def test_build_mixer_flarepp() -> None:
    mixer = build_mixer(FLAREPPMixerConfig(num_latents=8), _backbone())
    assert isinstance(mixer, FLAREPPMixer)
    assert "flarepp" in MIXER_BY_KIND


def test_flarepp_always_on_gate_and_norms() -> None:
    mixer = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8),
        _backbone(rmsnorm=True),
    )
    assert isinstance(mixer.gate_logit, torch.nn.Parameter)
    assert mixer.gate_logit.shape == (mixer.num_heads,)
    assert torch.allclose(mixer.gate_logit, torch.full_like(mixer.gate_logit, 0.25))
    assert isinstance(mixer.latent_q_fixed, torch.nn.Parameter)
    assert mixer.latent_q_fixed.shape == (mixer.num_heads, 8, mixer.head_dim)
    assert isinstance(mixer.v0_norm, torch.nn.RMSNorm)
    assert mixer.v0_norm.elementwise_affine is False
    assert isinstance(mixer.q_fixed_norm, torch.nn.RMSNorm)
    assert mixer.q_fixed_norm.elementwise_affine is False
    assert isinstance(mixer.q0_norm, torch.nn.RMSNorm)
    assert mixer.q0_norm.elementwise_affine is True
    assert isinstance(mixer.k0_norm, torch.nn.RMSNorm)
    assert mixer.k0_norm.elementwise_affine is False
    assert isinstance(mixer.k_norm, torch.nn.RMSNorm)
    assert not hasattr(mixer, "q_norm")
    assert not hasattr(mixer, "use_gate")
    assert mixer.share_k0_v0 is True
    assert mixer.v0_proj is None
    for name in ("k0_proj", "k_proj", "v_proj"):
        proj = getattr(mixer, name)
        assert isinstance(proj, torch.nn.Linear)
        assert proj.bias is not None
        assert getattr(proj, "_skip_backbone_weight_init", False) is True
    y = mixer(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)


@pytest.mark.parametrize("gate_logit_init", [0.0, 0.25])
def test_flarepp_forward_dtype_and_shape(gate_logit_init: float) -> None:
    mixer = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, gate_logit_init=gate_logit_init),
        _backbone(rmsnorm=True),
    )
    x = torch.randn(2, 16, 32)
    y = mixer(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not hasattr(mixer, "convex_mode")


def test_flarepp_k_norm_false_keeps_hop1_norms() -> None:
    mixer = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, k_norm=False),
        _backbone(rmsnorm=True),
    )
    assert isinstance(mixer.q0_norm, torch.nn.RMSNorm)
    assert mixer.q0_norm.elementwise_affine is True
    assert isinstance(mixer.k0_norm, torch.nn.RMSNorm)
    assert mixer.k0_norm.elementwise_affine is False
    assert isinstance(mixer.k_norm, torch.nn.Identity)
    assert isinstance(mixer.v0_norm, torch.nn.RMSNorm)
    assert isinstance(mixer.q_fixed_norm, torch.nn.RMSNorm)


def test_flarepp_share_k0_v0_skips_v0_norm() -> None:
    shared = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, share_k0_v0=True),
        _backbone(rmsnorm=True),
    )
    calls: list[int] = []
    orig = shared.v0_norm.forward

    def _spy(x, *args, **kwargs):
        calls.append(1)
        return orig(x, *args, **kwargs)

    shared.v0_norm.forward = _spy  # type: ignore[method-assign]
    y = shared(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)
    assert calls == []

    separate = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, share_k0_v0=False),
        _backbone(rmsnorm=True),
    )
    calls.clear()
    orig2 = separate.v0_norm.forward

    def _spy2(x, *args, **kwargs):
        calls.append(1)
        return orig2(x, *args, **kwargs)

    separate.v0_norm.forward = _spy2  # type: ignore[method-assign]
    y2 = separate(torch.randn(2, 16, 32))
    assert y2.shape == (2, 16, 32)
    assert calls == [1]


def test_flarepp_q_fixed_norm_toggle() -> None:
    on = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, q_fixed_norm=True),
        _backbone(rmsnorm=True),
    )
    assert isinstance(on.q_fixed_norm, torch.nn.RMSNorm)
    assert on.q_fixed_norm.elementwise_affine is False

    off = FLAREPPMixer(
        FLAREPPMixerConfig(num_latents=8, q_fixed_norm=False),
        _backbone(rmsnorm=True),
    )
    assert isinstance(off.q_fixed_norm, torch.nn.Identity)
    y = off(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)
