import pytest
import torch

import pdebench.callbacks as callbacks
from pdebench.distributed.context_parallel import ContextParallelState
from pdebench.distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive
from pdebench.distributed.utils import shard_sequence_tensor
from pdebench.models.flare import FLARE, FlareConfig, FLAREModel


def test_ahmedml_metric_pairs_reduce_before_sqrt(monkeypatch):
    pred = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 3.0, 4.0, 0.0]]])
    cp_state = ContextParallelState(rank=0, world_size=2, cp_group=object(), cp_size=2, cp_rank=0, seq_start=0, seq_end=1)
    calls = []

    def fake_all_reduce(value, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group is cp_state.cp_group
        calls.append(value.clone())
        value.add_(torch.tensor([3.0, 7.0], dtype=value.dtype))

    monkeypatch.setattr(callbacks.dist, "all_reduce", fake_all_reduce)
    reduced = callbacks._reduce_ahmedml_surface_metric_sums(
        callbacks.ahmedml_surface_metric_sums(pred, target), cp_state
    )

    assert len(calls) == 3
    assert tuple(float(v) for v in reduced["pressure_rel_l2"]) == pytest.approx((4.0, 8.0))


def test_surface_stats_cp_does_not_change_joint_metrics(monkeypatch):
    """CP only shards evaluation; joint metrics must match the non-CP reference."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pdebench

    torch.manual_seed(0)
    n = 32
    pred = torch.randn(1, n, 4)
    target = pred + 0.1 * torch.randn(1, n, 4)
    x = torch.cat([pred, torch.randn(1, n, 2)], dim=-1)
    batch = (x, target)

    class SliceModel(torch.nn.Module):
        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return x_in[..., :4]

    ref_fun = callbacks.make_drivaerml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()},
        cp_state=None,
    )
    trainer_ref = SimpleNamespace(
        model=SliceModel(),
        auto_cast=nullcontext(),
        move_to_device=lambda b: b,
        preprocess_fn_=None,
    )
    ref_loss, ref_stats = ref_fun(trainer_ref, [batch], split="test")

    cp_state = ContextParallelState(
        rank=0, world_size=2, cp_group=object(), cp_size=2, cp_rank=0, seq_start=0, seq_end=0
    )
    reduce_calls = {"n": 0}
    # Other shard contributions: MSE sum/count first, then Rel-L2 (num, den) pairs.
    other = slice(n // 2, None)
    sq_other = (pred[:, other] - target[:, other]).float().pow(2)
    pending = [
        sq_other.sum(),
        torch.tensor(float(sq_other.numel())),
    ]
    for _name, (num, den) in callbacks.drivaerml_surface_metric_sums(pred[:, other], target[:, other]).items():
        pending.append(torch.stack((num, den)))

    def fake_all_reduce(value, op, group):
        assert op in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX)
        assert group is cp_state.cp_group
        reduce_calls["n"] += 1
        if op == torch.distributed.ReduceOp.SUM and value.dtype.is_floating_point:
            value.add_(pending.pop(0))

    monkeypatch.setattr(callbacks.dist, "all_reduce", fake_all_reduce)

    def preprocess(batch_in):
        x_b, y_b = batch_in
        return x_b[:, : x_b.shape[1] // 2], y_b[:, : y_b.shape[1] // 2]

    statsfun = callbacks.make_drivaerml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()},
        cp_state=cp_state,
    )
    trainer = SimpleNamespace(
        model=SliceModel(),
        auto_cast=nullcontext(),
        move_to_device=lambda b: b,
        preprocess_fn_=preprocess,
    )
    got_loss, got = statsfun(trainer, [batch], split="test")

    assert reduce_calls["n"] == 2 + 3 + 2  # MSE sum/count + 3 Rel-L2 pairs + min/max sync
    assert pending == []
    assert not any(key.startswith("ts3_") for key in got)
    assert got_loss == pytest.approx(ref_loss, rel=1e-5, abs=1e-6)
    assert got["mse"] == pytest.approx(ref_stats["mse"], rel=1e-5, abs=1e-6)
    for key in ("full_rel_l2", "pressure_rel_l2", "wall_shear_rel_l2"):
        assert got[key] == pytest.approx(ref_stats[key], rel=1e-5, abs=1e-6), f"CP changed joint {key}"


def test_ahmedml_surface_stats_cp_does_not_change_joint_metrics(monkeypatch):
    """AhmedML fullbatch CP sharding must not change MSE or Rel-L2 metrics."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pdebench

    torch.manual_seed(0)
    n = 32
    pred = torch.randn(1, n, 4)
    target = pred + 0.1 * torch.randn(1, n, 4)
    x = torch.cat([pred, torch.randn(1, n, 2)], dim=-1)
    batch = (x, target)

    class SliceModel(torch.nn.Module):
        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return x_in[..., :4]

    ref_fun = callbacks.make_ahmedml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()},
        cp_state=None,
    )
    trainer_ref = SimpleNamespace(
        model=SliceModel(),
        auto_cast=nullcontext(),
        move_to_device=lambda b: b,
        preprocess_fn_=None,
    )
    ref_loss, ref_stats = ref_fun(trainer_ref, [batch], split="test")
    assert not any(key.startswith("ts3_") for key in ref_stats)

    cp_state = ContextParallelState(
        rank=0, world_size=2, cp_group=object(), cp_size=2, cp_rank=0, seq_start=0, seq_end=0
    )
    reduce_calls = {"n": 0}
    other = slice(n // 2, None)
    sq_other = (pred[:, other] - target[:, other]).float().pow(2)
    pending = [
        sq_other.sum(),
        torch.tensor(float(sq_other.numel())),
    ]
    for _name, (num, den) in callbacks.ahmedml_surface_metric_sums(pred[:, other], target[:, other]).items():
        pending.append(torch.stack((num, den)))

    def fake_all_reduce(value, op, group):
        assert op in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX)
        assert group is cp_state.cp_group
        reduce_calls["n"] += 1
        if op == torch.distributed.ReduceOp.SUM and value.dtype.is_floating_point:
            value.add_(pending.pop(0))

    monkeypatch.setattr(callbacks.dist, "all_reduce", fake_all_reduce)

    def preprocess(batch_in):
        x_b, y_b = batch_in
        return x_b[:, : x_b.shape[1] // 2], y_b[:, : y_b.shape[1] // 2]

    statsfun = callbacks.make_ahmedml_surface_statsfun(
        {"y_normalizer": pdebench.IdentityNormalizer()},
        cp_state=cp_state,
    )
    trainer = SimpleNamespace(
        model=SliceModel(),
        auto_cast=nullcontext(),
        move_to_device=lambda b: b,
        preprocess_fn_=preprocess,
    )
    got_loss, got = statsfun(trainer, [batch], split="test")

    assert reduce_calls["n"] == 2 + 3 + 2  # MSE sum/count + 3 Rel-L2 pairs + min/max sync
    assert pending == []
    assert not any(key.startswith("ts3_") for key in got)
    assert got_loss == pytest.approx(ref_loss, rel=1e-5, abs=1e-6)
    assert got["mse"] == pytest.approx(ref_stats["mse"], rel=1e-5, abs=1e-6)
    for key in ("full_rel_l2", "pressure_rel_l2", "wall_shear_rel_l2"):
        assert got[key] == pytest.approx(ref_stats[key], rel=1e-5, abs=1e-6), f"CP changed joint {key}"


def test_surface_cp_count_sync_detects_desync(monkeypatch):
    cp_state = ContextParallelState(
        rank=0, world_size=2, cp_group=object(), cp_size=2, cp_rank=0, seq_start=0, seq_end=0
    )
    seen = []

    def fake_all_reduce(value, op, group):
        seen.append(op)
        if op == torch.distributed.ReduceOp.MIN:
            value.copy_(torch.tensor([1], dtype=value.dtype, device=value.device))
        elif op == torch.distributed.ReduceOp.MAX:
            value.copy_(torch.tensor([2], dtype=value.dtype, device=value.device))

    monkeypatch.setattr(callbacks.dist, "all_reduce", fake_all_reduce)
    with pytest.raises(RuntimeError, match="CP desync"):
        callbacks._assert_surface_stats_cp_counts_synced(1, cp_state, torch.device("cpu"))
    assert seen == [torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX]


def test_flare_encoder_cp_backend_selects_module():
    flash = FLARE(channel_dim=64, num_heads=4, num_latents=8, encoder_cp_backend="flash")
    naiive = FLARE(channel_dim=64, num_heads=4, num_latents=8, encoder_cp_backend="naiive")
    model = FLAREModel(
        FlareConfig(num_blocks=1, channel_dim=64, num_heads=4, num_latents=8, encoder_cp_backend="flash"),
        metadata={"c_in": 2, "c_out": 1},
    )

    assert isinstance(flash.encoder_cp, FlareEncoderCPFlash)
    assert isinstance(naiive.encoder_cp, FlareEncoderCPNaiive)
    assert isinstance(model.blocks[0].att.encoder_cp, FlareEncoderCPFlash)


def test_flare_encoder_cp_backend_default_is_flash():
    assert FlareConfig().encoder_cp_backend == "flash"
    att = FLARE(channel_dim=64, num_heads=4, num_latents=8)
    assert isinstance(att.encoder_cp, FlareEncoderCPFlash)


def test_shard_sequence_tensor_bounds():
    x = torch.arange(2 * 10 * 3).view(2, 10, 3)

    cp0 = ContextParallelState(rank=0, world_size=2, cp_group=None, cp_size=2, cp_rank=0, seq_start=0, seq_end=0)
    cp1 = ContextParallelState(rank=1, world_size=2, cp_group=None, cp_size=2, cp_rank=1, seq_start=0, seq_end=0)

    x0 = shard_sequence_tensor(x, cp0, seq_dim=1)
    x1 = shard_sequence_tensor(x, cp1, seq_dim=1)

    assert x0.shape[1] == 5
    assert x1.shape[1] == 5
    assert cp0.seq_start == 0 and cp0.seq_end == 5
    assert cp1.seq_start == 5 and cp1.seq_end == 10
    assert torch.equal(torch.cat([x0, x1], dim=1), x)


def test_flare_encoder_cp_matches_dense_softmax_single_rank():
    torch.manual_seed(0)
    b, h, m, n, d = 2, 3, 5, 17, 7

    q = torch.randn(b, h, m, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    scale = d ** -0.5

    encoder = FlareEncoderCPNaiive()
    z_cp = encoder(q_latent=q, k_local=k, v_local=v, cp_group=None, scale=scale)
    z_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    assert torch.allclose(z_cp, z_ref, atol=1e-5, rtol=1e-5)


def test_flare_encoder_cp_fp16_autocast_large_n_stays_finite():
    """u = exp(s)@v is O(N); under AMP fp16 that overflows unless encode is true fp32."""
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    b, h, m, n, d = 1, 2, 4, 200_000, 8
    # Flat scores → exp≈1 over N tokens → |u| ≈ N·|v| ≫ fp16 max without fp32 accumulators.
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.zeros(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.full((b, h, n, d), 0.4, device=device, dtype=torch.float16)
    scale = d ** -0.5

    encoder = FlareEncoderCPNaiive()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        z = encoder(q_latent=q, k_local=k, v_local=v, cp_group=None, scale=scale)

    assert z.dtype == torch.float16
    assert torch.isfinite(z).all()
    z_ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=scale
    ).to(dtype=torch.float16)
    assert torch.allclose(z, z_ref, atol=2e-3, rtol=2e-3)


def test_flare_flash_cp_packed_mass_numerator_sum_matches_separate():
    """Packed [numerator | mass] SUM must match two separate SUMs."""
    torch.manual_seed(0)
    b, h, m, d = 2, 4, 8, 16
    # Two fake CP shards
    out0 = torch.randn(b, h, m, d)
    out1 = torch.randn(b, h, m, d)
    lse0 = torch.randn(b, h, m)
    lse1 = torch.randn(b, h, m)
    lse_max = torch.maximum(lse0, lse1)

    from pdebench.distributed.flare_cp import merge_flash_cp_lse_stats

    z_pack, lse_pack = merge_flash_cp_lse_stats(
        out_shards=(out0, out1),
        lse_shards=(lse0, lse1),
        lse_max=lse_max,
        packed=True,
    )
    z_sep, lse_sep = merge_flash_cp_lse_stats(
        out_shards=(out0, out1),
        lse_shards=(lse0, lse1),
        lse_max=lse_max,
        packed=False,
    )
    assert torch.allclose(z_pack, z_sep, atol=0.0, rtol=0.0)
    assert torch.allclose(lse_pack, lse_sep, atol=0.0, rtol=0.0)


def test_flare_encoder_cp_flash_rejects_fp32_inputs():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    b, h, m, n, d = 1, 2, 4, 16, 16
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="fp16/bf16"):
        FlareEncoderCPFlash()(q, k, v, cp_group=None, scale=d ** -0.5)


def test_flare_encoder_cp_flash_rejects_mixed_dtypes():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    b, h, m, n, d = 1, 2, 4, 16, 16
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="fp16/bf16"):
        FlareEncoderCPFlash()(q, k, v, cp_group=None, scale=d ** -0.5)


def test_promote_for_flash_casts_fp32_with_fp16_sibling():
    from pdebench.distributed.flare_cp import promote_for_flash

    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    q = torch.randn(1, 2, 4, 8, device=device, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 8, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 8, device=device, dtype=torch.float32)
    q2, k2, v2 = promote_for_flash(q, k, v)
    assert q2.dtype == k2.dtype == v2.dtype == torch.float16


def test_flare_encoder_cp_flash_matches_naiive_single_rank():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Flash-legal shapes: head_dim multiple of 8
    b, h, m, n, d = 2, 4, 8, 64, 16
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    scale = d ** -0.5

    from pdebench.distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive

    z_naiive = FlareEncoderCPNaiive()(q, k, v, cp_group=None, scale=scale)
    z_flash = FlareEncoderCPFlash()(q, k, v, cp_group=None, scale=scale)
    assert z_flash.dtype == q.dtype
    assert torch.allclose(z_flash.float(), z_naiive.float(), atol=2e-2, rtol=2e-2)


def test_flare_encoder_cp_flash_grads_match_naiive_single_rank():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(1)
    device = torch.device("cuda")
    b, h, m, n, d = 2, 4, 8, 64, 16
    scale = d ** -0.5
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16, requires_grad=True)

    from pdebench.distributed.flare_cp import FlareEncoderCPFlash, FlareEncoderCPNaiive

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    z_n = FlareEncoderCPNaiive()(q, k, v, None, scale)
    z_f = FlareEncoderCPFlash()(q2, k2, v2, None, scale)
    g = torch.randn_like(z_n)
    dq_n, dk_n, dv_n = torch.autograd.grad(z_n, (q, k, v), g)
    dq_f, dk_f, dv_f = torch.autograd.grad(z_f, (q2, k2, v2), g.clone())
    assert torch.allclose(dq_f.float(), dq_n.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(dk_f.float(), dk_n.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(dv_f.float(), dv_n.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_flare_encoder_cp_flash_amp_finite(amp_dtype):
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, h, m, n, d = 1, 4, 8, 128, 16
    q = torch.randn(b, h, m, d, device=device, dtype=amp_dtype, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=amp_dtype, requires_grad=True)
    v = torch.randn(b, h, n, d, device=device, dtype=amp_dtype, requires_grad=True)
    scale = d ** -0.5

    from pdebench.distributed.flare_cp import FlareEncoderCPFlash

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        z = FlareEncoderCPFlash()(q, k, v, None, scale)
        loss = z.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(z).all()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


def test_flare_encoder_cp_flash_under_torch_compile():
    """Flash CP custom autograd must graph-break cleanly under torch.compile."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, h, m, n, d = 1, 4, 8, 64, 16
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16, requires_grad=True)
    scale = d ** -0.5

    from pdebench.distributed.flare_cp import FlareEncoderCPFlash

    enc = torch.compile(FlareEncoderCPFlash())
    z = enc(q, k, v, None, scale)
    z.float().pow(2).mean().backward()
    assert torch.isfinite(z).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
