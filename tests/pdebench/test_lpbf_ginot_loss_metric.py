from __future__ import annotations

import torch
import torch_geometric as pyg

import pdebench
from pdebench.dataset.ginot.forward import ginot_per_graph_channel_rel_l2
from pdebench.dataset.lpbf import (
    _LPBF_LEGACY_BUGGY_NORMALIZER,
    LPBF_POS_SCALE,
    lpbf_metadata_normalizers,
    lpbf_warped_rel_l2,
    make_lpbf_y_normalizer,
)


def _tiny_lpbf_dataset() -> pyg.data.Dataset:
    graphs: list[pyg.data.Data] = []
    for disp_z in ([0.01, 0.02, 0.03], [0.02, 0.03, 0.04]):
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [30.0, 0.0, 60.0],
                [-30.0, -30.0, 30.0],
            ],
            dtype=torch.float32,
        )
        disp = torch.zeros(3, 3, dtype=torch.float32)
        disp[:, 2] = torch.tensor(disp_z, dtype=torch.float32)
        graph = pyg.data.Data(pos=pos, disp=disp)
        graph.x = graph.pos
        graph.y = graph.disp[:, 2]
        graphs.append(graph)

    class _TinyLPBF(pyg.data.Dataset):
        def len(self) -> int:
            return len(graphs)

        def get(self, idx: int) -> pyg.data.Data:
            return graphs[int(idx)]

    return _TinyLPBF()


def test_lpbf_flare_parity_uses_warped_decode_rel_l2_not_physical() -> None:
    """LPBF warped decode Rel-L2 (legacy_warped) must match RelL2Loss on decoded tensors."""
    train = _tiny_lpbf_dataset()
    _, y_n = lpbf_metadata_normalizers(train)

    y_raw = train[0].y.reshape(-1, 1)
    y_enc = y_n.encode(y_raw)
    batch_index = torch.zeros(y_raw.shape[0], dtype=torch.long)
    zero = torch.zeros_like(y_raw)

    warped = ginot_per_graph_channel_rel_l2(
        y_n.decode(zero),
        y_n.decode(y_raw),
        batch_index=batch_index,
        num_graphs=1,
    )[0]
    physical = ginot_per_graph_channel_rel_l2(
        y_n.decode(zero),
        y_raw,
        batch_index=batch_index,
        num_graphs=1,
    )[0]
    encoded_decode = ginot_per_graph_channel_rel_l2(
        y_n.decode(torch.zeros_like(y_enc)),
        y_n.decode(y_enc),
        batch_index=batch_index,
        num_graphs=1,
    )[0]

    lf = pdebench.RelL2Loss()
    flare_loss = lf(
        y_n.decode(torch.zeros_like(y_raw.unsqueeze(0))),
        y_n.decode(y_raw.unsqueeze(0)),
    )

    assert torch.allclose(flare_loss, warped, atol=1e-5)
    assert torch.allclose(physical, encoded_decode, atol=1e-5)
    assert float(physical) > float(warped)


def test_lpbf_y_normalizer_uses_disp_stats_not_random_init() -> None:
    train = _tiny_lpbf_dataset()
    x_n, meta_y = lpbf_metadata_normalizers(train)
    yn = make_lpbf_y_normalizer(train)
    if _LPBF_LEGACY_BUGGY_NORMALIZER:
        assert float(meta_y.mean) != float(yn.mean)
        assert float(x_n.mean) == float(yn.mean)
        assert float(x_n.std) == float(yn.std)
    else:
        assert float(meta_y.mean) == float(yn.mean)
        assert float(meta_y.std) == float(yn.std)
        random_yn = pdebench.UnitGaussianNormalizer(torch.rand(3, 1))
        assert float(random_yn.mean) != float(yn.mean)


def test_lpbf_warped_rel_l2_uses_varlen_batch_index_for_flat_tensors() -> None:
    train = _tiny_lpbf_dataset()
    _, yn = lpbf_metadata_normalizers(train)
    y0 = train[0].y.reshape(-1, 1)
    y1 = train[1].y.reshape(-1, 1)
    y = torch.cat([y0, y1], dim=0)
    zero = torch.zeros_like(y)
    batch_index = torch.cat(
        [
            torch.zeros(y0.shape[0], dtype=torch.long),
            torch.ones(y1.shape[0], dtype=torch.long),
        ]
    )
    indexed = ginot_per_graph_channel_rel_l2(
        yn.decode(zero),
        yn.decode(y),
        batch_index=batch_index,
        num_graphs=2,
    ).mean()
    warped = lpbf_warped_rel_l2(zero, y, yn, batch_index=batch_index, num_graphs=2)
    pooled = lpbf_warped_rel_l2(zero, y, yn)
    assert torch.allclose(warped, indexed, atol=1e-5)
    assert float(pooled) != float(indexed)


def test_lpbf_warped_rel_l2_matches_flare_rel_l2_for_single_graph() -> None:
    train = _tiny_lpbf_dataset()
    _, yn = lpbf_metadata_normalizers(train)
    y = train[0].y.unsqueeze(0)
    zero = torch.zeros_like(y)
    flare = pdebench.RelL2Loss()(yn.decode(zero), yn.decode(y))
    shared = lpbf_warped_rel_l2(zero, y, yn)
    assert torch.allclose(flare, shared, atol=1e-2)


def test_lpbf_pos_scale_matches_finaltime_transform() -> None:
    assert LPBF_POS_SCALE.tolist() == [30.0, 30.0, 60.0]
