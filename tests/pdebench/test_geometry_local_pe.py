import torch

from pdebench.models.graph_models.glt import GeoTransolverPE, GeoTransolverPEConfig, build_pe


def test_geo_transolver_pe_encode_local_shape_and_finite():
    pe = GeoTransolverPE(
        GeoTransolverPEConfig(radii=(0.05, 0.25), neighbors_in_radius=(8, 32), n_hidden_local=16)
    )
    b, n = 2, 12
    pos = torch.randn(b, n, 3) * 0.1
    local = pe.encode_local(pos)
    assert local.shape == (b, n, pe.out_dim)
    assert pe.out_dim == 16 * 2
    assert torch.isfinite(local).all()


def test_geo_transolver_pe_empty_ball_finite():
    """Isolated points with tiny radius → zero neighbor pack → finite tanh(MLP(0))."""
    pe = GeoTransolverPE(GeoTransolverPEConfig(radii=(1e-6,), neighbors_in_radius=(4,), n_hidden_local=8))
    pos = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]])
    local = pe.encode_local(pos)
    assert local.shape == (1, 2, 8)
    assert torch.isfinite(local).all()


def test_geo_transolver_pe_registry_feature_request_and_packed_forward():
    assert GeoTransolverPEConfig().kind == "geo_transolver_pe"
    cfg = GeoTransolverPEConfig(radii=(0.2,), neighbors_in_radius=(4,), n_hidden_local=8)
    fr = cfg.to_feature_request()
    assert fr.laplacian_k == 0
    assert fr.edges is False

    pe = build_pe(cfg, pos_dim=3, act="gelu")
    assert pe.out_dim == 8
    n0, n1 = 6, 5
    n = n0 + n1
    pos = torch.randn(n, 3) * 0.05
    cu = torch.tensor([0, n0, n], dtype=torch.int32)
    out = pe(
        pos=pos,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        cu_seqlens=cu,
        num_total_nodes=n,
    )
    assert out.shape == (n, pe.out_dim)
    assert torch.isfinite(out).all()


def test_glt_with_geo_transolver_pe_constructs_and_forwards():
    from pdebench.models.graph_models.glt import GLT, GLTConfig

    model = GLT(
        GLTConfig(
            pe=GeoTransolverPEConfig(radii=(0.2,), neighbors_in_radius=(4,), n_hidden_local=8),
            pe_inject_mode="concat_input",
            attn_type="linear",
        ),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )
    n0, n1 = 6, 5
    n = n0 + n1
    pos = torch.randn(n, 3) * 0.05
    cu = torch.tensor([0, n0, n], dtype=torch.int32)
    out = model(
        pos=pos,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        use_flash_varlen=True,
        cu_seqlens=cu,
        max_seqlen=n0,
        num_total_nodes=n,
    )
    assert out.shape == (n, 1)
    assert torch.isfinite(out).all()
