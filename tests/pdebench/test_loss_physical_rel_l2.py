from __future__ import annotations

import torch

from pdebench.dataset.loss import channel_mean_rel_l2, compute_loss
from pdebench.dataset.sample import LossSpec
from pdebench.utils import RelL2Loss, UnitGaussianNormalizer


class _IdentityNormalizer:
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device):
        return self


def test_c_out_1_matches_rell2loss() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 10, 1)
    target = torch.randn(2, 10, 1) + 1.0  # avoid near-zero denom

    got = channel_mean_rel_l2(pred, target)
    expect = RelL2Loss()(pred, target)
    assert torch.allclose(got, expect)

    via_compute = compute_loss(pred, target, _IdentityNormalizer(), LossSpec())
    assert torch.allclose(via_compute, expect)


def test_decode_before_rel_l2() -> None:
    torch.manual_seed(1)
    physical = torch.randn(2, 8, 2) * 3.0 + 5.0
    y_normalizer = UnitGaussianNormalizer(physical)
    pred_enc = y_normalizer.encode(physical + 0.25)
    target_enc = y_normalizer.encode(physical)

    physical_loss = compute_loss(pred_enc, target_enc, y_normalizer, LossSpec())
    encoded_loss = channel_mean_rel_l2(pred_enc, target_enc)

    assert not torch.allclose(physical_loss, encoded_loss)
    assert torch.allclose(
        physical_loss,
        channel_mean_rel_l2(y_normalizer.decode(pred_enc), y_normalizer.decode(target_enc)),
    )


def test_multi_channel_means_per_channel_rel_l2() -> None:
    # Hand-crafted so per-channel Rel-L2s are easy to check.
    pred = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 2.0], [1.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[100.0, 1.0], [0.0, 1.0]],
            [[1.0, 4.0], [1.0, 4.0]],
        ],
        dtype=torch.float32,
    )

    got = channel_mean_rel_l2(pred, target)

    # Graph 0: ch0 → 1; ch1 → 1/√2. Graph 1: ch0 → 0; ch1 → 0.5.
    g0 = (1.0 + (1.0 / (2.0**0.5))) / 2.0
    g1 = (0.0 + 0.5) / 2.0
    expect = torch.tensor((g0 + g1) / 2.0)
    assert torch.allclose(got, expect)

    # RelL2Loss pools all channels together — must differ for C>1.
    pooled = RelL2Loss()(pred, target)
    assert not torch.allclose(got, pooled)


def test_loss_spec_mask_excludes_nodes() -> None:
    torch.manual_seed(2)
    pred = torch.randn(1, 6, 2)
    target = torch.randn(1, 6, 2) + 2.0
    free_mask = torch.tensor([[True, True, False, True, False, True]])

    unmasked = channel_mean_rel_l2(pred, target)
    masked = channel_mean_rel_l2(pred, target, mask=free_mask)
    assert not torch.allclose(unmasked, masked)

    # Masked loss equals Rel-L2 on the kept nodes only.
    kept_pred = pred[:, free_mask[0], :]
    kept_target = target[:, free_mask[0], :]
    assert torch.allclose(masked, channel_mean_rel_l2(kept_pred, kept_target))

    via_spec = compute_loss(
        pred,
        target,
        _IdentityNormalizer(),
        LossSpec(mask="free_mask"),
        masks={"free_mask": free_mask},
    )
    assert torch.allclose(via_spec, masked)


def test_packed_channel_mean_matches_padded():
    from pdebench.dataset.loss import channel_mean_rel_l2, packed_channel_mean_rel_l2
    import torch
    pred = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 3)
    padded = channel_mean_rel_l2(pred, target)
    flat_p = pred.reshape(10, 3)
    flat_t = target.reshape(10, 3)
    batch_index = torch.tensor([0]*5 + [1]*5)
    packed = packed_channel_mean_rel_l2(flat_p, flat_t, batch_index, num_graphs=2)
    assert torch.allclose(padded, packed, rtol=1e-5, atol=1e-6)


def test_compute_packed_loss_decodes():
    from pdebench.dataset.loss import compute_packed_loss
    from pdebench.dataset.sample import LossSpec
    import torch
    class Norm:
        def decode(self, x):
            return x * 2
        def to(self, device):
            return self
    pred = torch.ones(4, 2)
    target = torch.ones(4, 2) * 0.5
    batch_index = torch.tensor([0, 0, 1, 1])
    loss = compute_packed_loss(pred, target, Norm(), LossSpec(), batch_index, 2)
    assert torch.isfinite(loss)
