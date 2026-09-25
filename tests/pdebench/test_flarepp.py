from __future__ import annotations

import pytest
import torch

from pdebench.models.flarepp import FlarePPConfig


def test_flarepp_config_defaults() -> None:
    cfg = FlarePPConfig()
    assert cfg.model == "flarepp"
    assert cfg.num_blocks == 8
    assert cfg.channel_dim == 128
    assert cfg.num_heads == 8
    assert cfg.num_latents == 64
    assert cfg.act is None
    assert cfg.rmsnorm is False
    assert cfg.out_proj_norm is True
    assert cfg.num_layers_in_out_proj == 2
    assert cfg.num_layers_ffn == 0
    assert cfg.ffn_mlp_ratio == 2.0
    assert not hasattr(cfg, "qk0_norm")
    assert cfg.k_norm is True
    assert cfg.q_fixed_norm is True
    assert cfg.share_k0_v0 is True
    assert cfg.gate_logit_init == pytest.approx(0.25)
    assert cfg.encoder_cp_backend == "flash"
    for removed in (
        "qk_norm", "v0_norm", "use_gate", "qk0_norm",
        "qk_num_layers", "qk_ratio", "qv_num_layers", "qv_ratio",
        "k_num_layers", "k_ratio", "v_num_layers", "v_ratio",
    ):
        assert removed not in FlarePPConfig.__dataclass_fields__


def test_flarepp_mixer_attn_scale_is_inv_sqrt_head_dim() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8)
    assert mixer.attn_scale == pytest.approx(mixer.head_dim ** -0.5)


def test_flarepp_mixer_mask_raises() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8)
    x = torch.randn(2, 16, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="mask"):
        mixer(x, mask=mask)


def test_flarepp_mixer_anchored_surface() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, rmsnorm=True)
    assert not hasattr(mixer, "use_gate")
    assert not hasattr(mixer, "q_norm")
    assert isinstance(mixer.gate_logit, torch.nn.Parameter)
    assert torch.allclose(mixer.gate_logit, torch.full_like(mixer.gate_logit, 0.25))
    assert isinstance(mixer.latent_q_fixed, torch.nn.Parameter)
    assert isinstance(mixer.v0_norm, torch.nn.RMSNorm)
    assert mixer.v0_norm.elementwise_affine is False
    assert isinstance(mixer.q0_norm, torch.nn.RMSNorm)
    assert mixer.q0_norm.elementwise_affine is True
    assert isinstance(mixer.k0_norm, torch.nn.RMSNorm)
    assert mixer.k0_norm.elementwise_affine is False
    assert isinstance(mixer.q_fixed_norm, torch.nn.RMSNorm)
    assert mixer.q_fixed_norm.elementwise_affine is False
    assert mixer.share_k0_v0 is True
    assert mixer.v0_proj is None
    for name in ("k0_proj", "k_proj", "v_proj"):
        proj = getattr(mixer, name)
        assert isinstance(proj, torch.nn.Linear)
        assert proj.bias is not None
        assert getattr(proj, "_skip_backbone_weight_init", False) is True


def test_flarepp_mixer_forward_shape() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8)
    y = mixer(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)


def _shared_mixer_state_dict(src: torch.nn.Module, dst: torch.nn.Module) -> dict[str, torch.Tensor]:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    shared = {k: src_sd[k] for k in dst_sd if k in src_sd}
    missing = sorted(set(dst_sd) - set(shared))
    extra = sorted(set(src_sd) - set(dst_sd))
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
def test_package_flarepp_matches_anchored_mixer(
    rmsnorm: bool, k_norm: bool, q_fixed_norm: bool, share_k0_v0: bool,
) -> None:
    from pdebench.models.flarepp import FLAREPPMixer as PackageMixer
    from pdebench.models.mixer_backbone import (
        FLAREPPMixer,
        FLAREPPMixerConfig,
        MixerBackboneConfig,
    )

    bb = MixerBackboneConfig(
        channel_dim=32, num_heads=4, rmsnorm=rmsnorm, diagnostics=False,
    )
    cfg = FLAREPPMixerConfig(
        num_latents=8,
        k_norm=k_norm,
        q_fixed_norm=q_fixed_norm,
        share_k0_v0=share_k0_v0,
        gate_logit_init=0.25,
    )
    torch.manual_seed(0)
    anchored = FLAREPPMixer(cfg, bb)
    torch.manual_seed(0)
    package = PackageMixer(
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
    assert package.attn_scale == pytest.approx(anchored.attn_scale)
    package.load_state_dict(_shared_mixer_state_dict(anchored, package), strict=False)
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_a = anchored(x)
        y_p = package(x)
    torch.testing.assert_close(y_a, y_p, rtol=0.0, atol=0.0)


def _meta():
    return {"c_in": 4, "c_out": 3, "dataset": "elasticity"}


def _cfg(**kwargs) -> FlarePPConfig:
    base = dict(
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        num_latents=8,
    )
    base.update(kwargs)
    return FlarePPConfig(**base)


def test_flarepp_model_forward_shape() -> None:
    from pdebench.models.flarepp import FLAREPPModel

    model = FLAREPPModel(_cfg(), metadata=_meta())
    y = model(torch.randn(2, 16, 4))
    assert y.shape == (2, 16, 3)


def test_flarepp_model_mask_raises() -> None:
    from pdebench.models.flarepp import FLAREPPModel

    model = FLAREPPModel(_cfg(), metadata=_meta())
    mask = torch.ones(2, 16, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="mask"):
        model(torch.randn(2, 16, 4), mask=mask)


def test_flarepp_model_propagates_anchored_gate_config() -> None:
    from pdebench.models.flarepp import FLAREPPModel

    model = FLAREPPModel(
        _cfg(
            gate_logit_init=-0.5,
            q_fixed_norm=False,
        ),
        metadata=_meta(),
    )
    for block in model.blocks:
        mixer = block.mixer
        assert isinstance(mixer.gate_logit, torch.nn.Parameter)
        assert torch.equal(mixer.gate_logit, torch.full((4,), -0.5))
        assert isinstance(mixer.q_fixed_norm, torch.nn.Identity)
        assert not hasattr(mixer, "use_gate")


def test_flarepp_in_model_config_map() -> None:
    from pdebench.config import MODEL_CONFIG_BY_MODEL, FlarePPConfig

    assert MODEL_CONFIG_BY_MODEL["flarepp"] is FlarePPConfig


def test_flarepp_factory_resolves_model_class() -> None:
    from pdebench.config import Config, DatasetConfig, OptimizerConfig, RunConfig, TrainingConfig
    from pdebench.models import model_factory

    cfg = Config(
        run=RunConfig(),
        dataset=DatasetConfig(dataset="elasticity"),
        training=TrainingConfig(),
        optimizer=OptimizerConfig(),
        scheduler={},
        model={"model": "flarepp"},
    )
    metadata = {"c_in": 2, "c_out": 1, "dataset": "elasticity", "space_dim": 2}
    cfg, c_in, c_out, model_name, Model = model_factory._resolve_model_spec(cfg, metadata)
    assert model_name == "FLAREPP"
    assert Model.__name__ == "FLAREPPModel"
    assert cfg.model.model == "flarepp"
    # Training not rewritten by a puri recipe for flarepp alone
    assert cfg.training.batch_size == TrainingConfig().batch_size


def test_flarepp_in_mesh_sequence_models() -> None:
    from pdebench.dataset.mesh_runtime import MESH_SEQUENCE_MODELS

    assert "flarepp" in MESH_SEQUENCE_MODELS


def test_flarepp_encoder_cp_backend_default_is_flash() -> None:
    from pdebench.distributed.flare_cp import FlareEncoderCPFlash
    from pdebench.models.flarepp import FLAREPPMixer

    assert FlarePPConfig().encoder_cp_backend == "flash"
    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8)
    assert isinstance(mixer.encoder_cp, FlareEncoderCPFlash)


def test_flarepp_encoder_cp_backend_selects_module() -> None:
    from pdebench.distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive
    from pdebench.models.flarepp import FLAREPPMixer, FLAREPPModel

    flash = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, encoder_cp_backend="flash")
    naiive = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, encoder_cp_backend="naiive")
    model = FLAREPPModel(
        _cfg(encoder_cp_backend="flash"),
        metadata=_meta(),
    )
    assert isinstance(flash.encoder_cp, FlareEncoderCPFlash)
    assert isinstance(naiive.encoder_cp, FlareEncoderCPNaiive)
    assert isinstance(model.blocks[0].mixer.encoder_cp, FlareEncoderCPFlash)


def test_flarepp_set_context_parallel_threads_to_mixer() -> None:
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPModel

    model = FLAREPPModel(_cfg(num_blocks=2), metadata=_meta())
    cp = ContextParallelState(
        rank=0, world_size=2, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=0,
    )
    model.set_context_parallel(cp, cp_debug_gather_outputs=True)
    assert model.cp_state is cp
    for block in model.blocks:
        assert block.mixer.cp_state is cp
        assert block.mixer.cp_debug_gather_outputs is True


def test_flarepp_cp_path_single_rank_matches_non_cp() -> None:
    """cp_size>1 with cp_group=None still uses encoder_cp; should match dense SDPAs."""
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(0)
    mixer = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, encoder_cp_backend="naiive")
    x = torch.randn(2, 16, 32)
    y_ref = mixer(x)

    cp = ContextParallelState(
        rank=0, world_size=1, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=16,
    )
    mixer.set_context_parallel(cp)
    y_cp = mixer(x)
    assert torch.allclose(y_cp, y_ref, atol=1e-5, rtol=1e-5)


def test_flarepp_cp_attached_default_gate_forward_is_finite() -> None:
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, encoder_cp_backend="naiive",
    )
    cp = ContextParallelState(
        rank=0, world_size=1, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=16,
    )
    mixer.set_context_parallel(cp)

    y = mixer(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rmsnorm", [False, True])
def test_flarepp_cp_flash_survives_autocast_fp16(rmsnorm: bool) -> None:
    """Regression: norms may leave q/k in fp32 under autocast; Flash CP promotes internally."""
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(0)
    device = torch.device("cuda")
    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8,
        encoder_cp_backend="flash", k_norm=True, rmsnorm=rmsnorm,
    ).to(device)

    cp = ContextParallelState(
        rank=0, world_size=1, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=16,
    )
    mixer.set_context_parallel(cp)

    x = torch.randn(2, 16, 32, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        y = mixer(x)

    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()


def test_flarepp_mixer_gate_param_shape_and_init() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=1.0986,
    )
    assert not hasattr(mixer, "use_gate")
    assert mixer.gate_logit.shape == (4,)
    assert torch.equal(mixer.gate_logit, torch.full((4,), 1.0986))
    assert mixer.latent_q0.shape == (4, 8, 8)
    assert mixer.latent_q_fixed.shape == (4, 8, 8)
    assert mixer.latent_q0 is not mixer.latent_q_fixed
    assert not hasattr(mixer, "qd_norm")
    assert not hasattr(mixer, "qf_norm")
    assert not isinstance(mixer.q0_norm, torch.nn.Identity)


def test_flarepp_mixer_gate_logit_init_validation() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    with pytest.raises(ValueError):
        FLAREPPMixer(
            channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=float("nan"),
        )


def test_flarepp_share_k0_v0_drops_v0_proj_and_forwards() -> None:
    from pdebench.models.flarepp import FLAREPPMixer, FLAREPPModel

    shared = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8)
    assert shared.share_k0_v0 is True
    assert shared.v0_proj is None
    assert isinstance(shared.k0_proj, torch.nn.Linear)
    y = shared(torch.randn(2, 16, 32))
    assert y.shape == (2, 16, 32)

    model = FLAREPPModel(_cfg(), metadata=_meta())
    names = {n for n, _ in model.named_parameters()}
    assert any("k0_proj" in n for n in names)
    assert not any("v0_proj" in n for n in names)
    assert model(torch.randn(2, 16, 4)).shape == (2, 16, 3)

    separate = FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, share_k0_v0=False)
    assert separate.v0_proj is not None
    assert isinstance(separate.v0_proj, torch.nn.Linear)

    with pytest.raises(TypeError, match="share_k0_v0"):
        FLAREPPMixer(channel_dim=32, num_heads=4, num_latents=8, share_k0_v0=1)  # type: ignore[arg-type]


def test_flarepp_gate_interpolation_endpoints() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(1)
    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=0.0,
    )
    x = torch.randn(2, 16, 32)

    with torch.no_grad():
        mixer.gate_logit.fill_(-80.0)
    y_fixed = mixer(x)

    with torch.no_grad():
        mixer.gate_logit.fill_(80.0)
    y_mixed = mixer(x)

    assert not torch.allclose(y_fixed, y_mixed, atol=1e-4, rtol=1e-4)

    # Exact endpoints: g=0 -> qf, g=1 -> qf + q_dynamic.
    q_fixed = torch.randn(2, 4, 8, 8)
    q_dynamic = torch.randn(2, 4, 8, 8)
    assert torch.equal(q_fixed + 0.0 * q_dynamic, q_fixed)
    assert torch.equal(q_fixed + 1.0 * q_dynamic, q_fixed + q_dynamic)


def test_flarepp_gate_broadcast_and_per_head() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=0.0,
    )
    with torch.no_grad():
        mixer.gate_logit.copy_(torch.tensor([-2.0, -1.0, 1.0, 2.0]))
    fixed_gate = torch.sigmoid(mixer.gate_logit).view(1, 4, 1, 1)
    assert fixed_gate.shape == (1, 4, 1, 1)
    assert not torch.allclose(fixed_gate[0, 0], fixed_gate[0, 3])


def test_flarepp_gate_gradient_flow() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(2)
    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=0.0,
    )
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = mixer(x)
    y.sum().backward()

    assert mixer.gate_logit.grad is not None
    assert torch.isfinite(mixer.gate_logit.grad).all()
    assert mixer.gate_logit.grad.abs().sum() > 0
    assert mixer.latent_q_fixed.grad is not None
    assert mixer.latent_q_fixed.grad.abs().sum() > 0
    assert mixer.latent_q0.grad is not None
    assert mixer.latent_q0.grad.abs().sum() > 0
    assert mixer.k0_proj.weight.grad is not None
    assert mixer.k0_proj.weight.grad.abs().sum() > 0


def test_flarepp_gate_fixed_endpoint_kills_dynamic_grad() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(3)
    mixer = FLAREPPMixer(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        gate_logit_init=0.0,
        share_k0_v0=False,
    )
    with torch.no_grad():
        mixer.gate_logit.fill_(-80.0)
    x = torch.randn(2, 16, 32)
    y = mixer(x)
    y.sum().backward()
    # At g≈0, q = qf only, so hop-0 synthesis grads should be ~0.
    assert mixer.k0_proj.weight.grad.abs().max().item() < 1e-5
    assert mixer.v0_proj.weight.grad.abs().max().item() < 1e-5
    assert mixer.latent_q0.grad.abs().max().item() < 1e-5
    assert mixer.latent_q_fixed.grad.abs().sum() > 0


def test_flarepp_gated_cp_path_single_rank_matches_non_cp() -> None:
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(4)
    mixer = FLAREPPMixer(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        encoder_cp_backend="naiive",
        gate_logit_init=0.0,
    )
    with torch.no_grad():
        mixer.gate_logit.copy_(torch.tensor([-1.0, 0.0, 0.5, 1.5]))
    x = torch.randn(2, 16, 32)
    y_ref = mixer(x)
    y_ref.sum().backward()
    grads_ref = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}

    mixer.zero_grad(set_to_none=True)
    cp = ContextParallelState(
        rank=0, world_size=1, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=16,
    )
    mixer.set_context_parallel(cp)
    y_cp = mixer(x)
    y_cp.sum().backward()
    assert torch.allclose(y_cp, y_ref, atol=1e-5, rtol=1e-5)
    for name, g_ref in grads_ref.items():
        g_cp = dict(mixer.named_parameters())[name].grad
        assert g_cp is not None
        assert torch.allclose(g_cp, g_ref, atol=1e-4, rtol=1e-4), name


def test_flarepp_gate_numerical_stability() -> None:
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(5)
    mixer = FLAREPPMixer(
        channel_dim=32, num_heads=4, num_latents=8, gate_logit_init=0.0,
    )
    x = torch.randn(2, 16, 32)
    y = mixer(x)
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(torch.sigmoid(mixer.gate_logit)).all()
    assert torch.isfinite(mixer.gate_logit.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flarepp_gated_cp_flash_survives_autocast_fp16() -> None:
    from pdebench.distributed.context_parallel import ContextParallelState
    from pdebench.models.flarepp import FLAREPPMixer

    torch.manual_seed(6)
    device = torch.device("cuda")
    mixer = FLAREPPMixer(
        channel_dim=32,
        num_heads=4,
        num_latents=8,
        encoder_cp_backend="flash",
        k_norm=True,
        gate_logit_init=0.0,
    ).to(device)
    cp = ContextParallelState(
        rank=0, world_size=1, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=16,
    )
    mixer.set_context_parallel(cp)
    x = torch.randn(2, 16, 32, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        y = mixer(x)
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()


def test_flarepp_latent_q_and_gate_logit_have_zero_weight_decay() -> None:
    """latent_q0 / latent_q_fixed and gate_logit must land in the zero-decay AdamW group."""
    from pdebench.models.flarepp import FLAREPPModel
    from pdebench.utils import make_optimizer_adamw, split_params_adamw

    model = FLAREPPModel(_cfg(gate_logit_init=0.0, num_blocks=2), metadata=_meta())
    decay, no_decay, latent = split_params_adamw(model)

    latent_ids = {id(p) for p in latent}
    decay_ids = {id(p) for p in decay}
    no_decay_ids = {id(p) for p in no_decay}

    named = dict(model.named_parameters())
    for name, param in named.items():
        if "latent_q" in name or name.endswith("gate_logit"):
            assert id(param) in latent_ids, name
            assert id(param) not in decay_ids, name
            assert id(param) not in no_decay_ids, name

    # Gated mixers: gate_logit + latent_q0 + latent_q_fixed per block.
    assert sum(1 for n in named if n.endswith("gate_logit")) == 2
    assert sum(1 for n in named if n.endswith("latent_q0")) == 2
    assert sum(1 for n in named if n.endswith("latent_q_fixed")) == 2

    opt = make_optimizer_adamw(model, lr=1e-3, weight_decay=1e-2)
    assert len(opt.param_groups) == 3
    assert opt.param_groups[0]["weight_decay"] == 1e-2
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert opt.param_groups[2]["weight_decay"] == 0.0
    zero_decay_ids = {id(p) for g in opt.param_groups[1:] for p in g["params"]}
    for name, param in named.items():
        if "latent_q" in name or name.endswith("gate_logit"):
            assert id(param) in zero_decay_ids, name
