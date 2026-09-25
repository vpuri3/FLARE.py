import pytest
import torch

from pdebench.models.graph_models.glt.pe_hop import (
    DEFAULT_HOP_SCALES,
    MAX_HOP,
    MultiscaleHopPE,
    MultiscaleHopPEConfig,
    build_multiscale_hop_neighborhoods,
)


def _path_edge_index(n: int) -> torch.Tensor:
    # undirected path 0-1-...-(n-1)
    src = torch.arange(n - 1)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def test_hop_scales_on_path_graph():
    n = 12
    edge_index = _path_edge_index(n)
    scales = build_multiscale_hop_neighborhoods(edge_index, n)
    assert len(scales) == 3
    # Center 0: neighbors at hops 1..MAX_HOP are nodes 1..MAX_HOP
    for s_idx, (lo, hi) in enumerate(DEFAULT_HOP_SCALES):
        c, nb, hops = scales[s_idx].center_index, scales[s_idx].neighbor_index, scales[s_idx].hop_distance
        mask = c == 0
        got = set(zip(nb[mask].tolist(), hops[mask].tolist()))
        expect = {(j, j) for j in range(lo, hi + 1)}
        assert got == expect
    # Distance > MAX_HOP excluded: center 0 has no pair with node 11
    all_nb0 = torch.cat([s.neighbor_index[s.center_index == 0] for s in scales])
    assert 11 not in set(all_nb0.tolist())
    # Center never appears as its own neighbor
    for s in scales:
        assert not (s.center_index == s.neighbor_index).any()
    assert MAX_HOP == 10


def test_hop_builder_batch_isolation():
    # Two disjoint edges plus a cross-batch edge; BFS must not cross graphs.
    # graph0: 0-1, graph1: 2-3
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    scales = build_multiscale_hop_neighborhoods(edge_index, 4, batch)
    s1 = scales[0]
    pairs = set(zip(s1.center_index.tolist(), s1.neighbor_index.tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (2, 3) in pairs and (3, 2) in pairs
    assert (0, 2) not in pairs and (0, 3) not in pairs
    assert (1, 2) not in pairs and (2, 1) not in pairs


def test_config_feature_request_and_out_dim():
    cfg = MultiscaleHopPEConfig()
    assert cfg.kind == "multiscale_hop_pe"
    feature_request = cfg.to_feature_request()
    assert feature_request.edges is True and feature_request.laplacian_k == 0
    assert cfg.out_dim == 192
    pe = MultiscaleHopPE(cfg)
    assert pe.out_dim == 192


def test_config_rejects_non_default_hidden_dim():
    with pytest.raises(ValueError, match="pointnet_hidden_dim must be 32"):
        MultiscaleHopPEConfig(pointnet_hidden_dim=16)


def test_forward_shape_path_graph():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    n = 12
    pos = torch.randn(n, 3)
    edge_index = _path_edge_index(n)
    cu_seqlens = torch.tensor([0, n], dtype=torch.int32)
    out = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu_seqlens, num_total_nodes=n)
    assert out.shape == (n, 192)
    assert torch.isfinite(out).all()


def test_forward_shape_path_graph_2d():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig(), pos_dim=2)
    n = 12
    pos = torch.randn(n, 2)
    edge_index = _path_edge_index(n)
    cu_seqlens = torch.tensor([0, n], dtype=torch.int32)

    out = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu_seqlens, num_total_nodes=n)

    assert pe.pointnets[0][0].in_features == 4
    assert out.shape == (n, 192)
    assert torch.isfinite(out).all()


def test_empty_scale3_on_small_diameter_finite():
    # Triangle diameter 1 → scales 2 and 3 are empty.
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    out = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu_seqlens, num_total_nodes=3)
    assert out.shape == (3, 192)
    assert torch.isfinite(out).all()


def test_translation_invariance():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    pe.eval()
    n = 8
    pos = torch.randn(n, 3)
    edge_index = _path_edge_index(n)
    cu = torch.tensor([0, n], dtype=torch.int32)

    out0 = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=n)
    out1 = pe(pos=pos + 3.5, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=n)

    assert torch.allclose(out0, out1, atol=1e-5, rtol=1e-5)


def test_permutation_equivariance():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    pe.eval()
    n = 8
    pos = torch.randn(n, 3)
    edge_index = _path_edge_index(n)
    cu = torch.tensor([0, n], dtype=torch.int32)
    perm = torch.randperm(n)
    pos_p = pos[perm]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    edge_p = inv[edge_index]

    out = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=n)
    out_p = pe(pos=pos_p, edge_index=edge_p, cu_seqlens=cu, num_total_nodes=n)

    assert torch.allclose(out_p, out[perm], atol=1e-4, rtol=1e-4)


def test_batch_isolation_positions():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    pe.eval()
    edge0 = _path_edge_index(3)
    edge1 = _path_edge_index(3) + 3
    edge_index = torch.cat([edge0, edge1], dim=1)
    pos = torch.randn(6, 3)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)

    out0 = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=6)
    pos2 = pos.clone()
    pos2[3:] += torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    out1 = pe(pos=pos2, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=6)

    assert torch.allclose(out0[:3], out1[:3], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out0[3:], out1[3:], atol=1e-3)


def test_mean_max_matches_loop_reference():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig(normalize_by_mean_edge_length=False))
    pe.eval()
    n = 4
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [5.0, 1.0, 0.0]])
    edge_index = _path_edge_index(n)
    cu = torch.tensor([0, n], dtype=torch.int32)

    out = pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=n)
    neighborhood = build_multiscale_hop_neighborhoods(edge_index, n)[0]
    deltas = pos[neighborhood.neighbor_index] - pos[neighborhood.center_index]
    distances = torch.linalg.vector_norm(deltas, dim=-1, keepdim=True)
    hops = neighborhood.hop_distance.to(dtype=pos.dtype).unsqueeze(-1) / MAX_HOP
    messages = pe.pointnets[0](torch.cat([deltas, distances, hops], dim=-1))

    pooled = torch.zeros(n, 64)
    for center in range(n):
        center_messages = messages[neighborhood.center_index == center]
        if center_messages.numel():
            pooled[center, :32] = center_messages.mean(dim=0)
            pooled[center, 32:] = center_messages.max(dim=0).values
    reference = pe.norms[0](pooled)

    assert torch.allclose(out[:, :64], reference, atol=1e-6, rtol=1e-6)


def test_gradients_finite():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    n = 8
    pos = torch.randn(n, 3, requires_grad=True)
    edge_index = _path_edge_index(n)
    cu = torch.tensor([0, n], dtype=torch.int32)

    pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=n).sum().backward()

    assert pos.grad is not None and torch.isfinite(pos.grad).all()
    for parameter in pe.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_diagnostics_keys():
    pe = MultiscaleHopPE(MultiscaleHopPEConfig())
    n = 12

    out, diagnostics = pe(
        pos=torch.randn(n, 3),
        edge_index=_path_edge_index(n),
        cu_seqlens=torch.tensor([0, n], dtype=torch.int32),
        num_total_nodes=n,
        return_diagnostics=True,
    )

    assert out.shape == (n, 192)
    assert len(diagnostics) == 3
    expected_keys = {
        "num_pairs",
        "mean_degree",
        "median_degree",
        "p95_degree",
        "p99_degree",
        "max_degree",
        "empty_fraction",
    }
    for diagnostic in diagnostics:
        assert expected_keys <= diagnostic.keys()


def test_no_dense_nxn_in_builder(monkeypatch):
    num_nodes = 8

    def shape_from_call(args, kwargs):
        shape = args[0] if args else kwargs.get("size")
        if isinstance(shape, (tuple, list, torch.Size)):
            return tuple(shape)
        return (shape,) if shape is not None else ()

    def guard_allocation(factory):
        def guarded(*args, **kwargs):
            assert shape_from_call(args, kwargs) != (num_nodes, num_nodes)
            return factory(*args, **kwargs)

        return guarded

    monkeypatch.setattr(torch, "zeros", guard_allocation(torch.zeros))
    monkeypatch.setattr(torch, "full", guard_allocation(torch.full))
    monkeypatch.setattr(torch, "empty", guard_allocation(torch.empty))

    scales = build_multiscale_hop_neighborhoods(_path_edge_index(num_nodes), num_nodes)

    assert len(scales) == len(DEFAULT_HOP_SCALES)
