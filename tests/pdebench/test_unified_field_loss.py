"""Unified field Rel-L2: padded+mask, packed/varlen, and LPBF decode-both."""

from __future__ import annotations

import torch

from pdebench.dataset.loss import (
    channel_mean_rel_l2,
    compute_field_loss,
    compute_packed_loss,
    field_rel_l2,
    packed_channel_mean_rel_l2,
)
from pdebench.dataset.sample import LossSpec
from pdebench.utils import UnitGaussianNormalizer


class _IdentityNormalizer:
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device):
        return self


def test_field_rel_l2_padded_matches_channel_mean() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 7, 3)
    target = torch.randn(2, 7, 3) + 1.0
    assert torch.allclose(field_rel_l2(pred, target), channel_mean_rel_l2(pred, target))


def test_field_rel_l2_padded_mask() -> None:
    pred = torch.randn(1, 5, 2)
    target = torch.randn(1, 5, 2) + 2.0
    mask = torch.tensor([[True, True, False, True, False]])
    assert torch.allclose(
        field_rel_l2(pred, target, mask=mask),
        channel_mean_rel_l2(pred, target, mask=mask),
    )


def test_field_rel_l2_padded_mask_drops_empty_graphs_and_clamps_zero_target() -> None:
    pred = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[False, False], [True, True]])

    got = field_rel_l2(pred, target, mask=mask)
    packed = packed_channel_mean_rel_l2(
        pred.reshape(4, 1),
        target.reshape(4, 1),
        batch_index=torch.tensor([0, 0, 1, 1]),
        num_graphs=2,
        mask=mask.reshape(4),
    )

    assert torch.isfinite(got)
    assert torch.allclose(got, packed)


def test_field_rel_l2_padded_all_empty_mask_returns_nan() -> None:
    pred = torch.ones(2, 3, 1)
    target = torch.ones_like(pred)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    assert torch.isnan(field_rel_l2(pred, target, mask=mask))


def test_field_rel_l2_packed_via_batch_index() -> None:
    pred = torch.randn(2, 4, 2)
    target = torch.randn(2, 4, 2) + 1.0
    padded = channel_mean_rel_l2(pred, target)
    flat_p = pred.reshape(8, 2)
    flat_t = target.reshape(8, 2)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    packed = field_rel_l2(flat_p, flat_t, batch_index=batch_index, num_graphs=2)
    assert torch.allclose(padded, packed, rtol=1e-5, atol=1e-6)


def test_field_rel_l2_packed_via_cu_seqlens() -> None:
    pred = torch.randn(9, 2)
    target = torch.randn(9, 2) + 1.0
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    via_cu = field_rel_l2(pred, target, cu_seqlens=cu)
    batch_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    via_bi = packed_channel_mean_rel_l2(pred, target, batch_index, num_graphs=2)
    assert torch.allclose(via_cu, via_bi, rtol=1e-5, atol=1e-6)


def test_compute_field_loss_decodes_both_sides() -> None:
    torch.manual_seed(1)
    physical = torch.randn(2, 6, 1) * 2.0 + 3.0
    yn = UnitGaussianNormalizer(physical)
    pred_enc = yn.encode(physical + 0.1)
    target_enc = yn.encode(physical)
    got = compute_field_loss(pred_enc, target_enc, yn, LossSpec())
    expect = channel_mean_rel_l2(yn.decode(pred_enc), yn.decode(target_enc))
    assert torch.allclose(got, expect)


def test_compute_field_loss_packed_with_mask_key() -> None:
    pred = torch.randn(6, 2)
    target = torch.randn(6, 2) + 1.0
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1])
    free = torch.tensor([True, False, True, True, True, False])
    got = compute_field_loss(
        pred,
        target,
        _IdentityNormalizer(),
        LossSpec(mask="free_mask"),
        masks={"free_mask": free},
        batch_index=batch_index,
        num_graphs=2,
    )
    expect = packed_channel_mean_rel_l2(pred, target, batch_index, 2, mask=free)
    assert torch.allclose(got, expect)


def test_packed_per_graph_matches_mean_and_ginot_wrapper() -> None:
    from pdebench.dataset.ginot.forward import ginot_per_graph_channel_rel_l2
    from pdebench.dataset.loss import packed_per_graph_channel_mean_rel_l2
    from pdebench.dataset.plaid_datasets import NodeFeatureNormalizer

    yh = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 2.0]])
    y = torch.tensor([[100.0, 1.0], [0.0, 1.0], [1.0, 4.0]])
    batch_index = torch.tensor([0, 0, 1])
    per_graph = packed_per_graph_channel_mean_rel_l2(yh, y, batch_index, num_graphs=2)
    assert torch.allclose(per_graph.mean(), packed_channel_mean_rel_l2(yh, y, batch_index, 2))
    assert torch.allclose(
        per_graph,
        ginot_per_graph_channel_rel_l2(yh, y, batch_index=batch_index, num_graphs=2),
    )
    yn = NodeFeatureNormalizer(mean=torch.zeros(1, 2), std=torch.ones(1, 2))
    loss = compute_packed_loss(
        yh,
        y,
        yn,
        LossSpec(),
        batch_index=batch_index,
        num_graphs=2,
    )
    assert torch.allclose(loss, per_graph.mean())
    assert torch.allclose(
        loss,
        compute_field_loss(yh, y, yn, LossSpec(), batch_index=batch_index, num_graphs=2),
    )
