# tests/pdebench/test_pos_domain.py
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pdebench.dataset.normalizer import MeanStdNormalizer
from pdebench.dataset.pos_domain import (
    PosDomain,
    compute_pos_domain_from_dataloader,
    extract_batch_query_pos,
    maybe_attach_pos_domain,
    resolve_pos_normalizer_shift_scale,
)
from pdebench.dataset.sample import FeatureRequest
from pdebench.models.graph_models.glt.backbone import GLT, GLTConfig
from pdebench.models.graph_models.glt.pe_eigen import RawEigenPEConfig, SpectralFilterPEConfig
from pdebench.models.graph_models.glt.pe_hop import MultiscaleHopPEConfig
from pdebench.models.graph_models.glt.pe_other import GeoTransolverPEConfig, NonePEConfig
from pdebench.models.graph_models.glt.registry import build_pe


class _FlatPosDataset(Dataset):
    def __init__(self, chunks: list[torch.Tensor]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"flat_pos": self.chunks[idx]}


def test_extract_flat_pos():
    pos = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    out = extract_batch_query_pos({"flat_pos": pos})
    assert torch.equal(out, pos)


def test_extract_padded_pos_uses_mask():
    pos = torch.zeros(2, 3, 2)
    pos[0, 0] = torch.tensor([1.0, 2.0])
    pos[0, 1] = torch.tensor([3.0, 4.0])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    out = extract_batch_query_pos({"pos": pos, "mask": mask})
    assert out.shape == (2, 2)
    assert torch.equal(out[0], torch.tensor([1.0, 2.0]))


def test_compute_pos_domain_from_dataloader_aabb():
    # batch0: (0,0), (1,2); batch1: (-1,5), (0.5, -3)
    ds = _FlatPosDataset(
        [
            torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
            torch.tensor([[-1.0, 5.0], [0.5, -3.0]]),
        ]
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=lambda xs: xs[0])
    shift = torch.zeros(1, 2)
    scale = torch.ones(1, 2)
    domain = compute_pos_domain_from_dataloader(loader, shift=shift, scale=scale)
    assert isinstance(domain, PosDomain)
    assert torch.allclose(domain.normalized_pos_expanse[:, 0], torch.tensor([-1.0, -3.0]))
    assert torch.allclose(domain.normalized_pos_expanse[:, 1], torch.tensor([1.0, 5.0]))
    assert torch.equal(domain.shift, shift)
    assert torch.equal(domain.scale, scale)


def test_compute_pos_domain_empty_loader_raises():
    ds = _FlatPosDataset([])
    loader = DataLoader(ds, batch_size=1, collate_fn=lambda xs: xs[0])
    try:
        compute_pos_domain_from_dataloader(loader, shift=torch.zeros(1, 2), scale=torch.ones(1, 2))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower() or "no query" in str(exc).lower()


def test_resolve_prefers_pos_normalizer():
    pos_n = MeanStdNormalizer(mean=torch.zeros(1, 2), std=torch.ones(1, 2) * 2)
    x_n = MeanStdNormalizer(mean=torch.ones(1, 2), std=torch.ones(1, 2))
    shift, scale = resolve_pos_normalizer_shift_scale({"pos_normalizer": pos_n, "x_normalizer": x_n})
    assert torch.equal(scale, pos_n.std)


def test_maybe_attach_pos_domain_respects_flag():
    pos_n = MeanStdNormalizer(mean=torch.zeros(1, 2), std=torch.ones(1, 2))
    ds = _FlatPosDataset([torch.tensor([[0.0, 0.0], [1.0, 1.0]])])
    meta = {"pos_normalizer": pos_n, "train_collate_fn": lambda xs: xs[0]}
    out = maybe_attach_pos_domain(
        dict(meta),
        ds,
        batch_size=1,
        feature_request=FeatureRequest(pos_domain=False),
    )
    assert "pos_domain" not in out
    out = maybe_attach_pos_domain(
        dict(meta),
        ds,
        batch_size=1,
        feature_request=FeatureRequest(pos_domain=True),
    )
    assert "pos_domain" in out
    assert torch.allclose(out["pos_domain"].normalized_pos_expanse[:, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(out["pos_domain"].normalized_pos_expanse[:, 1], torch.tensor([1.0, 1.0]))


def test_maybe_attach_missing_normalizer_raises():
    ds = _FlatPosDataset([torch.tensor([[0.0, 0.0]])])
    try:
        maybe_attach_pos_domain(
            {},
            ds,
            batch_size=1,
            feature_request=FeatureRequest(pos_domain=True),
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_pe_requires_pos_domain_when_requested(monkeypatch):
    cfg = NonePEConfig()
    real_req = cfg.to_feature_request

    def _req():
        req = real_req()
        return FeatureRequest(
            edges=req.edges,
            boundary=req.boundary,
            laplacian_k=req.laplacian_k,
            laplacian_spec=req.laplacian_spec,
            pos_domain=True,
        )

    monkeypatch.setattr(cfg, "to_feature_request", _req)
    try:
        build_pe(cfg, pos_dim=2, pos_domain=None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pos_domain" in str(exc).lower()

    domain = PosDomain(
        shift=torch.zeros(1, 2),
        scale=torch.ones(1, 2),
        normalized_pos_expanse=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )
    pe = build_pe(cfg, pos_dim=2, pos_domain=domain)
    assert pe is not None


def test_glt_passes_pos_domain_from_metadata(monkeypatch):
    domain = PosDomain(
        shift=torch.zeros(1, 2),
        scale=torch.ones(1, 2),
        normalized_pos_expanse=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )
    cfg = GLTConfig(pe=NonePEConfig(), channel_dim=32, num_blocks=1, num_heads=4)
    monkeypatch.setattr(
        cfg.pe,
        "to_feature_request",
        lambda: FeatureRequest(pos_domain=True),
    )
    try:
        GLT(cfg, metadata={"c_in": 2, "c_out": 1, "pos_dim": 2})
        assert False, "expected ValueError"
    except ValueError:
        pass
    model = GLT(cfg, metadata={"c_in": 2, "c_out": 1, "pos_dim": 2, "pos_domain": domain})
    assert model.pe is not None


@pytest.mark.parametrize(
    "cfg_factory",
    [
        NonePEConfig,
        RawEigenPEConfig,
        SpectralFilterPEConfig,
        GeoTransolverPEConfig,
        MultiscaleHopPEConfig,
    ],
)
def test_existing_pe_configs_do_not_request_pos_domain(cfg_factory):
    cfg = cfg_factory()
    assert cfg.to_feature_request().pos_domain is False


def test_probe_dist_pe_requests_pos_domain():
    from pdebench.models.graph_models.glt.pe_dist import ProbeDistPEConfig

    assert ProbeDistPEConfig().to_feature_request().pos_domain is True
