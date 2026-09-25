import logging
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pdebench.dataset.pos_domain import PosDomain
from pdebench.models.graph_models.glt.pe_dist import (
    ProbeDistGraphPE,
    ProbeDistPE,
    ProbeDistPEConfig,
    _build_probe_dist_pe,
)
from pdebench.models.graph_models.glt.registry import GRAPH_PE_BY_KIND, build_pe


def _unit_box(d: int, scale=None, shift=None):
    domain_min = torch.zeros(d)
    domain_max = torch.ones(d)
    scale = torch.ones(d) if scale is None else scale
    shift = torch.zeros(d) if shift is None else shift
    return domain_min, domain_max, scale, shift


def _chain(dim: int = 2):
    pos = torch.zeros(4, dim)
    pos[:, 0] = torch.arange(4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    cu = torch.tensor([0, 4], dtype=torch.int32)
    return pos, edge_index, cu


def _euclidean_pe(num_probes, dim, box, **kwargs):
    return ProbeDistPE(num_probes, dim, *box, geodesic_feats=False, **kwargs)


def _geodesic_pe(num_probes, dim, box, **kwargs):
    return ProbeDistPE(num_probes, dim, *box, geodesic_feats=True, **kwargs)


def _geo_slice(pe: ProbeDistPE, out: torch.Tensor) -> torch.Tensor:
    return out[:, pe.num_probes * (pe.dim + 1) :]


def _fixed_geodesic_probe(*, num_anchor_candidates: int, max_geodesic_hops: int, probe_x: float, distance_cap: float = 2.0):
    pe = _geodesic_pe(
        1,
        2,
        (torch.zeros(2), torch.tensor([3.0, 1.0]), torch.ones(2), torch.zeros(2)),
        num_anchor_candidates=num_anchor_candidates,
        max_geodesic_hops=max_geodesic_hops,
        distance_cap=distance_cap,
    )
    with torch.no_grad():
        location = torch.tensor([[probe_x, 0.5]])
        unit_location = (location - pe.domain_min) / (pe.domain_max - pe.domain_min)
        pe.probe_logits.copy_(torch.logit(unit_location.clamp(pe.init_eps, 1.0 - pe.init_eps)))
    return pe


class _RejectNodePairAllocation(TorchDispatchMode):
    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        tensors = result if isinstance(result, (tuple, list)) else (result,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.shape == (self.num_nodes, self.num_nodes):
                raise AssertionError("allocated a node-pair [N,N] tensor")
        return result


def test_rejects_euclidean_feats_false():
    try:
        ProbeDistPE(2, 2, *_unit_box(2), euclidean_feats=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "euclidean_feats" in str(exc).lower()


def test_euclidean_only_2d_shape():
    d, k, n = 2, 4, 6
    pe = _euclidean_pe(k, d, _unit_box(d))
    out = pe(torch.rand(n, d))
    assert pe.output_dim == 3 * k
    assert out.shape == (n, 3 * k)


# --- Euclidean behavior (kind=probe_dist, geodesic_feats=False) ---


def test_euclidean_manual_anisotropic_distance():
    d, k = 3, 2
    scale = torch.tensor([10.0, 2.0, 0.5])
    shift = torch.tensor([1.0, 2.0, 3.0])
    box = (torch.zeros(d), torch.ones(d), scale, shift)
    pe = _euclidean_pe(k, d, box)
    with torch.no_grad():
        pe.probe_logits.fill_(torch.log(torch.tensor(0.25 / 0.75)))
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = pe(pos)
    a = pe.probe_locations()
    ell = pe.reference_length
    delta = (pos[0] - a[0]) * scale
    r = torch.linalg.vector_norm(delta)
    feat = torch.cat([delta / ell, (r / ell).reshape(1)], dim=0)
    got = out[0, 0 : (d + 1)]
    assert torch.allclose(got, feat, atol=1e-5, rtol=1e-5)


def test_euclidean_output_is_shift_invariant():
    d, k = 3, 3
    scale = torch.tensor([2.0, 3.0, 4.0])
    pe0 = _euclidean_pe(k, d, _unit_box(d, scale=scale, shift=torch.zeros(d)))
    pe1 = _euclidean_pe(k, d, _unit_box(d, scale=scale, shift=torch.tensor([10.0, -5.0, 2.0])))
    with torch.no_grad():
        pe1.probe_logits.copy_(pe0.probe_logits)
    pos = torch.rand(7, d)
    assert torch.allclose(pe0(pos), pe1(pos), atol=1e-5)


def test_euclidean_probe_locations_stay_in_domain_after_optimizer_steps():
    d, k = 2, 5
    pe = _euclidean_pe(k, d, _unit_box(d))
    opt = torch.optim.SGD(pe.parameters(), lr=1.0)
    pos = torch.rand(8, d)
    for _ in range(20):
        opt.zero_grad()
        pe(pos).sum().backward()
        opt.step()
    a = pe.probe_locations()
    assert torch.all(a >= 0.0 - 1e-6)
    assert torch.all(a <= 1.0 + 1e-6)


def test_euclidean_gradient_flows_to_probe_logits():
    d, k = 3, 4
    pe = _euclidean_pe(k, d, _unit_box(d))
    out = pe(torch.rand(6, d))
    out.sum().backward()
    g = pe.probe_logits.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_euclidean_has_no_cross_node_dependence_or_node_pair_allocation():
    d, k = 2, 5
    pe = _euclidean_pe(k, d, _unit_box(d))
    pos_a = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    pos_b = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.3, 0.4]])
    out_a = pe(pos_a)
    out_b = pe(pos_b)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[2], atol=1e-6)

    with _RejectNodePairAllocation(pos_b.shape[0]):
        pe(pos_b)


# --- Geodesic behavior (kind=probe_dist, geodesic_feats=True) ---


def test_geodesic_soft_attachment_backpropagates_to_probe_logits():
    pe = _fixed_geodesic_probe(num_anchor_candidates=2, max_geodesic_hops=3, probe_x=0.4)
    pos, edge_index, cu = _chain()
    geo = _geo_slice(pe, pe(pos, edge_index, cu))
    loss = geo[:, 0].square().sum()
    loss.backward()
    assert pe.probe_logits.grad is not None
    assert torch.isfinite(pe.probe_logits.grad).all()
    assert pe.probe_logits.grad.abs().sum() > 0


def test_geodesic_coincident_probe_anchor_has_zero_off_mesh_distance():
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    with torch.no_grad():
        pe.probe_logits.copy_(torch.logit(torch.tensor([[pe.init_eps, pe.init_eps]])))
    pos = pe.probe_locations().detach().clone()
    edge_index = torch.empty((2, 0), dtype=torch.long)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    geo = _geo_slice(pe, pe(pos, edge_index, cu))
    assert geo[0, 1] == 0


def test_geodesic_hop_budget_caps_unreached_nodes():
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0, distance_cap=2.0)
    pos, edge_index, cu = _chain()
    mesh = _geo_slice(pe, pe(pos, edge_index, cu))[:, 0]
    assert torch.allclose(mesh, torch.tensor([0.0, 1 / 3, 2.0, 2.0]), atol=1e-6)


def test_geodesic_packed_graphs_do_not_share_anchors_or_edges():
    pe = _fixed_geodesic_probe(num_anchor_candidates=2, max_geodesic_hops=3, probe_x=0.4)
    graph_1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    graph_2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pos = torch.cat((graph_1, graph_2), dim=0)
    edge_index = torch.tensor([[0, 1, 2, 4, 5], [1, 2, 3, 5, 6]])
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)
    expected = _geo_slice(pe, pe(pos, edge_index, cu))[:4]

    translated = pos.clone()
    translated[4:] += torch.tensor([100.0, -40.0])
    actual = _geo_slice(pe, pe(translated, edge_index, cu))[:4]

    assert torch.allclose(actual, expected)


def test_geodesic_anisotropic_scale_weights_edges_in_physical_space():
    pe = _geodesic_pe(
        1,
        2,
        (torch.zeros(2), torch.ones(2), torch.tensor([2.0, 3.0]), torch.zeros(2)),
        num_anchor_candidates=1,
        max_geodesic_hops=2,
    )
    with torch.no_grad():
        pe.probe_logits.fill_(torch.logit(torch.tensor(pe.init_eps)))
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    cu = torch.tensor([0, 3], dtype=torch.int32)

    mesh = _geo_slice(pe, pe(pos, edge_index, cu))[:, 0]
    diagonal = torch.tensor(13.0).sqrt()
    assert torch.allclose(mesh, torch.tensor([0.0, 2.0, 5.0]) / diagonal, atol=1e-6)


def test_geodesic_rejects_edges_that_cross_packed_graphs():
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="crosses packed graphs"):
        pe(pos, edge_index, cu)


@pytest.mark.parametrize("edge_index", [torch.tensor([[-1], [0]]), torch.tensor([[0], [4]])])
def test_geodesic_rejects_out_of_range_edge_endpoints(edge_index):
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    pos, _, cu = _chain()
    with pytest.raises(ValueError, match="out-of-range"):
        pe(pos, edge_index, cu)


# --- Input validation ---


@pytest.mark.parametrize("dim", [1, 4])
def test_rejects_invalid_dim(dim):
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        ProbeDistPE(2, dim, *_unit_box(dim))


@pytest.mark.parametrize("num_probes", [0, -1, 1.5, True])
def test_rejects_invalid_num_probes(num_probes):
    with pytest.raises(ValueError, match="num_probes must be a positive integer"):
        ProbeDistPE(num_probes, 2, *_unit_box(2))


def test_rejects_nonfinite_pos():
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    pos, edge_index, cu = _chain()
    pos[2, 0] = torch.nan
    with pytest.raises(ValueError, match="pos must contain only finite values"):
        pe(pos, edge_index, cu)


@pytest.mark.parametrize(
    "cu",
    [
        torch.tensor([1, 4]),
        torch.tensor([0, 3]),
        torch.tensor([0, 3, 2, 4]),
        torch.tensor([0, 2, 2, 4]),
    ],
)
def test_rejects_malformed_cu_seqlens(cu):
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    pos, edge_index, _ = _chain()
    with pytest.raises(ValueError, match="cu_seqlens|packed graphs"):
        pe(pos, edge_index, cu)


def test_rejects_multinode_graph_with_only_self_edges():
    pe = _fixed_geodesic_probe(num_anchor_candidates=1, max_geodesic_hops=1, probe_x=0.0)
    pos, _, cu = _chain()
    edge_index = torch.arange(4).repeat(2, 1)
    with pytest.raises(ValueError, match="no usable non-self edges"):
        pe(pos, edge_index, cu)


def test_rejects_mixed_finite_and_nonfinite_physical_edge_lengths():
    pe = _geodesic_pe(
        1,
        2,
        (torch.zeros(2), torch.ones(2), torch.tensor([torch.finfo(torch.float32).max, 1.0]), torch.zeros(2)),
        num_anchor_candidates=1,
        max_geodesic_hops=1,
    )
    pos = torch.tensor([[0.0, 0.0], [1e-38, 0.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    cu = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="physical edge lengths must contain only finite values"):
        pe(pos, edge_index, cu)


def test_rejects_overflowed_physical_probe_locations():
    huge_scale = torch.tensor([torch.finfo(torch.float32).max, 1.0])
    pe = _geodesic_pe(
        1,
        2,
        (torch.tensor([2.0, 0.0]), torch.tensor([3.0, 1.0]), huge_scale, torch.zeros(2)),
    )
    pos = torch.tensor([[0.0, 0.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="physical probe locations must contain only finite values"):
        pe(pos, edge_index, cu)


# --- torch.compile regression ---


def test_torch_compile_repeated_execution_has_no_item_graph_break(caplog):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    pe = _fixed_geodesic_probe(num_anchor_candidates=2, max_geodesic_hops=3, probe_x=0.4)
    pos, edge_index, cu = _chain()
    compiled = torch.compile(pe, backend="eager")
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        first = compiled(pos, edge_index, cu)
        stats_after_first = dict(torch._dynamo.utils.counters.get("stats", {}))
        second = compiled(pos, edge_index, cu)
        stats_after_second = dict(torch._dynamo.utils.counters.get("stats", {}))
    diagnostic = "\n".join([str(item.message) for item in caught] + [record.message for record in caplog.records])
    assert torch.allclose(first, second)
    assert "Tensor.item" not in diagnostic
    assert "Graph break from `Tensor.item()`" not in diagnostic
    graph_breaks = torch._dynamo.utils.counters.get("graph_break", {})
    assert not any("Tensor.item" in str(reason) for reason in graph_breaks)
    # The geodesic path is an intentional eager leaf reached via
    # torch.compiler.disable, so the first call may split into a couple of
    # graph segments around it, but the repeated call must be a pure cache
    # hit: no new graphs and no new frames traced the second time.
    assert stats_after_second == stats_after_first


# --- Wrapper/builder error paths ---


def test_graph_wrapper_validates_packed_node_count():
    core = _geodesic_pe(
        2,
        2,
        (torch.zeros(2), torch.tensor([3.0, 1.0]), torch.ones(2), torch.zeros(2)),
        num_anchor_candidates=2,
        max_geodesic_hops=3,
    )
    graph_pe = ProbeDistGraphPE(core)
    pos, edge_index, cu = _chain()
    with pytest.raises(ValueError, match="packed node count"):
        graph_pe(pos=pos, edge_index=edge_index, cu_seqlens=cu, num_total_nodes=3)


def test_build_pe_requires_pos_domain():
    with pytest.raises(ValueError, match="requested pos_domain"):
        build_pe(ProbeDistPEConfig(), pos_dim=3, pos_domain=None)


def test_builder_requires_pos_domain():
    with pytest.raises(ValueError, match="requires PosDomain"):
        _build_probe_dist_pe(ProbeDistPEConfig(), pos_dim=2, act="gelu", pos_domain=None)


def test_builder_validates_pos_dim_against_pos_domain():
    cfg = ProbeDistPEConfig(num_probes=2)
    pos_domain = _toy_pos_domain(2)
    with pytest.raises(ValueError, match="pos_dim 3 != PosDomain dim 2"):
        _build_probe_dist_pe(cfg, pos_dim=3, act="gelu", pos_domain=pos_domain)


@pytest.mark.parametrize("field", ["num_probes", "num_anchor_candidates", "max_geodesic_hops"])
@pytest.mark.parametrize("invalid_count", [1.5, True])
def test_builder_rejects_fractional_and_boolean_counts(field, invalid_count):
    config = ProbeDistPEConfig()
    setattr(config, field, invalid_count)
    pos_domain = PosDomain(
        shift=torch.zeros(1, 2),
        scale=torch.ones(1, 2),
        normalized_pos_expanse=torch.tensor([[0.0, 3.0], [0.0, 1.0]]),
    )

    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        _build_probe_dist_pe(config, pos_dim=2, act="gelu", pos_domain=pos_domain)


# --- Config / registry / launcher wiring ---


def _toy_pos_domain(d: int = 3) -> PosDomain:
    return PosDomain(
        shift=torch.zeros(1, d),
        scale=torch.ones(1, d),
        normalized_pos_expanse=torch.stack([torch.zeros(d), torch.ones(d)], dim=-1),
    )


def test_config_defaults_and_feature_request():
    cfg = ProbeDistPEConfig()
    assert cfg.kind == "probe_dist"
    assert cfg.euclidean_feats is True
    assert cfg.geodesic_feats is False
    req = cfg.to_feature_request()
    assert req.pos_domain is True
    assert req.edges is False
    cfg_g = ProbeDistPEConfig(geodesic_feats=True)
    assert cfg_g.to_feature_request().edges is True


def test_config_rejects_euclidean_feats_false():
    try:
        ProbeDistPEConfig(euclidean_feats=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "euclidean_feats" in str(exc).lower()


def test_config_rejects_invalid_kind():
    try:
        ProbeDistPEConfig(kind="raw_eigen")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "kind" in str(exc).lower()
        assert "probe_dist" in str(exc)


def test_registry_contains_probe_dist_and_build_pe():
    assert GRAPH_PE_BY_KIND["probe_dist"][0] is ProbeDistPEConfig
    pe = build_pe(ProbeDistPEConfig(num_probes=4), pos_dim=3, pos_domain=_toy_pos_domain(3))
    assert isinstance(pe, ProbeDistGraphPE)
    assert pe.out_dim == 16


def test_canonical_launcher_wires_probe_dist_options():
    launcher = Path("out/pdebench/run_glt.sh").read_text()
    assert 'GLT_PE=probe_dist' in launcher or 'probe_dist' in launcher
    branch = launcher.split('if [[ "${GLT_PE}" == "probe_dist" ]]; then', maxsplit=1)[1].split("fi", maxsplit=1)[0]
    assert "--model.pe.num_probes" in branch
    assert "--model.pe.geodesic_feats" in branch
    assert "learned_probe_euclidean_dist_pe" not in launcher
    assert '== "geodesic_dist"' not in launcher
