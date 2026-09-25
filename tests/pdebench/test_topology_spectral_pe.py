import torch

from pdebench.models.graph_models.glt import (
    SpectralFilterPE,
    _normalize_pe,
    _resolve_packed_indices,
)


def _spe_inputs(seed: int = 0):
    torch.manual_seed(seed)
    n0, n1, k, pos_dim = 6, 5, 4, 2
    n = n0 + n1
    v0, _ = torch.linalg.qr(torch.randn(n0, k))
    v1, _ = torch.linalg.qr(torch.randn(n1, k))
    eigenvectors = torch.cat([v0, v1], dim=0)
    pos = torch.randn(n, pos_dim)
    cu_seqlens = torch.tensor([0, n0, n], dtype=torch.int32)
    eigenvalues = torch.tensor(
        [[0.0, 1.5, 1.5, 4.0], [0.0, 0.7, 2.3, 5.1]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 6, 7, 8], [1, 2, 0, 4, 3, 7, 8, 9]],
        dtype=torch.long,
    )
    return eigenvectors, pos, cu_seqlens, eigenvalues, edge_index, (n0, n1, k, pos_dim)


def _make_spe(k: int, pos_dim: int, *, spe_mode: str = "query") -> SpectralFilterPE:
    torch.manual_seed(123)
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="band",
        spe_mode=spe_mode,
        band_sigma_init=1e-3,
        hidden_dim=16,
        poly_order=1,
    )
    for p in spe.parameters():
        if p.ndim > 0:
            torch.nn.init.normal_(p, std=0.3)
    return spe.eval()


def _call_spe(
    spe: SpectralFilterPE,
    eigenvectors: torch.Tensor,
    *,
    pos: torch.Tensor,
    cu_seqlens: torch.Tensor,
    eigenvalues: torch.Tensor,
    edge_index: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Call SpectralFilterPE via the shared GraphPE contract (kwargs-only, topology_* inputs)."""
    return spe(
        pos=pos,
        edge_index=edge_index,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        topology_features=eigenvectors,
        topology_eigenvalues=eigenvalues,
    )


def test_spe_output_shape() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = _make_spe(k, pos_dim)
    out = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
    expected_dim = k * (pos_dim + 1)
    assert out.shape == (n0 + n1, expected_dim)
    assert spe.out_feature_dim == expected_dim


def _check_invariance() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = _make_spe(k, pos_dim)
    theta = 0.7
    q = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
        ]
    )
    rot = ev.clone()
    rot[:n0, 1:3] = ev[:n0, 1:3] @ q.T
    flip = ev.clone()
    flip[:, 0] *= -1.0
    flip[:, 3] *= -1.0
    with torch.no_grad():
        base = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
        rotated = _call_spe(spe, rot, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
        flipped = _call_spe(spe, flip, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
        lam_pert = lam + 1e-3 * torch.randn_like(lam)
        pert = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam_pert, edge_index=edge_index, max_seqlen=n0)
    assert torch.allclose(base, flipped, atol=1e-4), "sign invariance"
    assert torch.allclose(base, rotated, atol=1e-4), "basis invariance"
    assert (base - pert).abs().max() < 1e-1, "stability"


def test_spe_query_invariance_and_stability() -> None:
    _check_invariance()


def test_spe_spatial_filter_shape() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="spatial",
        spe_mode="query",
        num_filters=6,
        hidden_dim=32,
        poly_order=1,
    ).eval()
    assert spe.num_total_filters == 6
    expected_dim = 6 * (pos_dim + 1)
    assert spe.out_feature_dim == expected_dim
    with torch.no_grad():
        out = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
    assert out.shape == (n0 + n1, expected_dim)


def test_spe_spatial_sign_invariance() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="spatial",
        spe_mode="query",
        num_filters=6,
        hidden_dim=32,
        poly_order=1,
    )
    for p in spe.parameters():
        if p.ndim > 0:
            torch.nn.init.normal_(p, std=0.3)
    spe.eval()

    flip = ev.clone()
    flip[:, 0] *= -1.0
    flip[:, 3] *= -1.0
    with torch.no_grad():
        base = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
        flipped = _call_spe(spe, flip, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
    assert torch.allclose(base, flipped, atol=1e-4), "spatial: sign invariance"


def test_spe_spatial_mode_permutation_invariant() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="spatial",
        spe_mode="query",
        num_filters=6,
        hidden_dim=32,
        poly_order=1,
    )
    for p in spe.parameters():
        if p.ndim > 0:
            torch.nn.init.normal_(p, std=0.3)
    spe.eval()

    perm = torch.tensor([2, 0, 3, 1])
    ev_perm = ev[:, perm].clone()
    lam_perm = lam[:, perm].clone()

    with torch.no_grad():
        base = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
        permuted = _call_spe(
            spe, ev_perm, pos=pos, cu_seqlens=cu, eigenvalues=lam_perm, edge_index=edge_index, max_seqlen=n0
        )
    assert torch.allclose(base, permuted, atol=1e-4), "spatial: mode-permutation invariance"


def test_spe_band_filter_query() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="band",
        spe_mode="query",
        band_sigma_init=1e-3,
        poly_order=1,
    ).eval()
    assert spe.out_feature_dim == k * (pos_dim + 1)
    with torch.no_grad():
        out = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
    assert out.shape == (n0 + n1, k * (pos_dim + 1))


def test_spe_multihop_shape_and_node_rms() -> None:
    ev, pos, cu, lam, edge_index, (n0, n1, k, pos_dim) = _spe_inputs()
    num_hops = 3
    spe = SpectralFilterPE(
        k,
        pos_dim=pos_dim,
        filter_type="band",
        spe_mode="multihop",
        num_hops=num_hops,
        band_sigma_init=1e-3,
        poly_order=1,
    ).eval()
    expected_dim = k * (num_hops + 1)
    assert spe.out_feature_dim == expected_dim
    assert spe.norm_mode == "node_rms"
    with torch.no_grad():
        out = _call_spe(spe, ev, pos=pos, cu_seqlens=cu, eigenvalues=lam, edge_index=edge_index, max_seqlen=n0)
    assert out.shape == (n0 + n1, expected_dim)
    assert torch.isfinite(out).all()


def test_normalize_pe_graph_rms_and_node_rms() -> None:
    pe = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    graph_rms = _normalize_pe(pe, cu_seqlens, "graph_rms")
    for start, end in ((0, 2), (2, 4)):
        block = graph_rms[start:end]
        rms_per_channel = block.square().mean(dim=0).sqrt()
        assert torch.allclose(rms_per_channel, torch.ones(2), atol=1e-5)

    node_rms = _normalize_pe(pe, cu_seqlens, "node_rms")
    per_node = node_rms.square().mean(dim=-1).sqrt()
    assert torch.allclose(per_node, torch.ones(4), atol=1e-5)


def test_resolve_packed_indices_uses_batch_index() -> None:
    lengths = torch.tensor([3, 2, 4], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    batch_index = torch.repeat_interleave(torch.arange(3, dtype=torch.long), lengths)
    num_nodes = int(cu_seqlens[-1])

    packed = _resolve_packed_indices(cu_seqlens, num_nodes, cu_seqlens.device, batch_index=batch_index)
    assert torch.equal(packed.batch_index, batch_index)
    assert torch.equal(packed.local_index, torch.arange(num_nodes, dtype=torch.long) - cu_seqlens[batch_index])
