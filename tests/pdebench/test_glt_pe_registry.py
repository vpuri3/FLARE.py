import pytest
import torch

from pdebench.dataset.sample import FeatureRequest
from pdebench.models.graph_models.glt import (
    GRAPH_PE_BY_KIND,
    GeoTransolverPEConfig,
    MultiscaleHopPEConfig,
    NonePE,
    NonePEConfig,
    RawEigenPEConfig,
    SpectralFilterPEConfig,
    build_pe,
)


def test_none_pe_in_registry_and_feature_request():
    assert "none" in GRAPH_PE_BY_KIND
    cfg = NonePEConfig()
    assert cfg.kind == "none"
    fr = cfg.to_feature_request()
    assert fr.laplacian_k == 0
    assert fr.edges is False


def test_build_none_pe_forward_empty_last_dim():
    pe = build_pe(NonePEConfig(), pos_dim=3, act="gelu")
    assert isinstance(pe, NonePE)
    assert pe.out_dim == 0
    n0, n1 = 4, 3
    n = n0 + n1
    pos = torch.randn(n, 3)
    cu_seqlens = torch.tensor([0, n0, n], dtype=torch.int32)
    out = pe(
        pos=pos,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        cu_seqlens=cu_seqlens,
        num_total_nodes=n,
    )
    assert out.shape == (n, 0)
    assert out.dtype == pos.dtype


def test_raw_eigen_config_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="num_eigenmodes"):
        RawEigenPEConfig(num_eigenmodes=0)


def test_raw_eigen_feature_request():
    cfg = RawEigenPEConfig(num_eigenmodes=64, laplacian_spec="graph")
    fr = cfg.to_feature_request()
    assert isinstance(fr, FeatureRequest)
    assert fr.laplacian_k == 64
    assert fr.laplacian_spec == "graph"


def test_spe_feature_request_and_build_out_dim():
    cfg = SpectralFilterPEConfig(num_eigenmodes=8, filter_type="band", mode="query", poly_order=2)
    fr = cfg.to_feature_request()
    assert fr.laplacian_k == 8
    pe = build_pe(cfg, pos_dim=3, act="gelu")
    assert pe.out_dim == pe.num_total_filters * pe.query_dim  # band: F=K


def test_build_pe_rejects_unknown_config_type():
    class Dummy:
        kind = "nope"
    try:
        build_pe(Dummy(), pos_dim=3, act="gelu")  # type: ignore[arg-type]
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_spe_forward_uses_shared_graph_pe_contract():
    """SpectralFilterPE.forward must accept the same kwargs as RawEigenPE.forward (GLT calls both this way)."""
    k, pos_dim = 8, 3
    n0, n1 = 6, 5
    n = n0 + n1
    pe = build_pe(SpectralFilterPEConfig(num_eigenmodes=k), pos_dim=pos_dim, act="gelu")

    pos = torch.randn(n, pos_dim)
    cu_seqlens = torch.tensor([0, n0, n], dtype=torch.int32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    topology_features = torch.randn(n, k)
    topology_eigenvalues = torch.randn(2, k)

    out = pe(
        pos=pos,
        edge_index=edge_index,
        cu_seqlens=cu_seqlens,
        max_seqlen=n0,
        num_total_nodes=n,
        topology_features=topology_features,
        topology_eigenvalues=topology_eigenvalues,
        batch_index=None,
    )
    assert out.shape == (n, pe.out_dim)
    assert torch.isfinite(out).all()


def test_glt_with_spe_config_constructs_and_forwards():
    from pdebench.models.graph_models.glt import GLT, GLTConfig

    model = GLT(
        GLTConfig(pe=SpectralFilterPEConfig(num_eigenmodes=8), pe_inject_mode="concat_qk", attn_type="linear"),
        metadata=dict(c_in=3, c_out=1, pos_dim=3),
    )

    n0, n1, k = 6, 5, 8
    n = n0 + n1
    pos = torch.randn(n, 3)
    cu_seqlens = torch.tensor([0, n0, n], dtype=torch.int32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    topology_features = torch.randn(n, k)
    topology_eigenvalues = torch.randn(2, k)

    out = model(
        pos=pos,
        edge_index=edge_index,
        use_flash_varlen=True,
        cu_seqlens=cu_seqlens,
        max_seqlen=n0,
        topology_features=topology_features,
        topology_eigenvalues=topology_eigenvalues,
    )
    assert out.shape == (n, 1)
    assert torch.isfinite(out).all()


def test_geo_transolver_pe_in_registry():
    assert "geo_transolver_pe" in GRAPH_PE_BY_KIND
    cfg_cls, mod_cls, _ = GRAPH_PE_BY_KIND["geo_transolver_pe"]
    assert cfg_cls is GeoTransolverPEConfig
    pe = build_pe(GeoTransolverPEConfig(), pos_dim=3, act="gelu")
    assert isinstance(pe, mod_cls)
    assert pe.out_dim == 32 * 2


def test_multiscale_hop_pe_in_registry():
    assert "multiscale_hop_pe" in GRAPH_PE_BY_KIND
    cfg_cls, mod_cls, _ = GRAPH_PE_BY_KIND["multiscale_hop_pe"]
    assert cfg_cls is MultiscaleHopPEConfig
    pe = build_pe(MultiscaleHopPEConfig(), pos_dim=3, act="gelu")
    assert isinstance(pe, mod_cls)
    assert pe.out_dim == 192
    assert pe.pointnets[0][0].in_features == 5

    pe_2d = build_pe(MultiscaleHopPEConfig(), pos_dim=2, act="gelu")
    assert pe_2d.pointnets[0][0].in_features == 4
