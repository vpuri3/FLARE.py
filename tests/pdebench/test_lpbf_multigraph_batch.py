"""LPBF multi-graph collate (padded+mask / flat varlen) and batch loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_geometric as pyg

from pdebench.dataset.lpbf import (
    lpbf_collate_padded,
    lpbf_collate_varlen,
    lpbf_flare_batch_loss,
    lpbf_warped_rel_l2,
    make_lpbf_collate_fn,
    resolve_lpbf_batch_format,
)
from pdebench.utils import UnitGaussianNormalizer


def _graph(n: int, z: float) -> pyg.data.Data:
    x = torch.randn(n, 3)
    y = torch.full((n, 1), z, dtype=torch.float32)
    return pyg.data.Data(x=x, y=y, pos=x.clone())


class _IdentityY:
    def decode(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def to(self, device):
        return self


class _EchoModel(nn.Module):
    """Returns zeros shaped like a FLARE padded/varlen forward would."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        self.calls = []

    def forward(self, x, mask=None, use_flash_varlen=False, cu_seqlens=None, max_seqlen=None):
        self.calls.append(
            {
                "mask": mask,
                "use_flash_varlen": use_flash_varlen,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
            }
        )
        if x.ndim == 3:
            return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


def test_resolve_lpbf_batch_format_defaults() -> None:
    assert resolve_lpbf_batch_format(mixed_precision=True) == "varlen"
    assert resolve_lpbf_batch_format(mixed_precision=False) == "padded"
    assert resolve_lpbf_batch_format(mixed_precision=True, explicit="padded") == "padded"
    assert resolve_lpbf_batch_format(mixed_precision=True, use_context_parallel=True) == "padded"
    assert (
        resolve_lpbf_batch_format(
            mixed_precision=True,
            explicit="padded",
            use_context_parallel=True,
        )
        == "padded"
    )
    with pytest.raises(ValueError, match="incompatible with context parallel"):
        resolve_lpbf_batch_format(
            mixed_precision=True,
            explicit="varlen",
            use_context_parallel=True,
        )


def test_lpbf_collate_padded_masks_shorter_graphs() -> None:
    batch = lpbf_collate_padded([_graph(3, 0.1), _graph(5, 0.2)])
    assert batch["x"].shape == (2, 5, 3)
    assert batch["y"].shape == (2, 5, 1)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].shape == (2, 5)
    assert bool(batch["mask"][0].sum() == 3)
    assert bool(batch["mask"][1].all())
    assert batch["num_graphs"] == 2
    assert batch["format"] == "padded"


def test_lpbf_collate_varlen_is_flat_with_cu_seqlens() -> None:
    batch = lpbf_collate_varlen([_graph(3, 0.1), _graph(5, 0.2)])
    assert batch["x"].shape == (8, 3)
    assert batch["y"].shape == (8, 1)
    assert torch.equal(batch["cu_seqlens"], torch.tensor([0, 3, 8], dtype=torch.int32))
    assert batch["max_seqlen"] == 5
    assert batch["num_graphs"] == 2
    assert batch["format"] == "varlen"
    assert torch.equal(batch["batch_index"], torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]))


def test_make_lpbf_collate_fn_dispatches() -> None:
    padded = make_lpbf_collate_fn("padded")([_graph(2, 0.0), _graph(2, 1.0)])
    varlen = make_lpbf_collate_fn("varlen")([_graph(2, 0.0), _graph(2, 1.0)])
    assert padded["format"] == "padded"
    assert varlen["format"] == "varlen"


def test_lpbf_flare_batch_loss_padded_matches_per_graph_mean() -> None:
    model = _EchoModel()
    yn = _IdentityY()
    g0, g1 = _graph(4, 1.0), _graph(3, 2.0)
    batch = lpbf_collate_padded([g0, g1])
    loss = lpbf_flare_batch_loss(model, batch, yn)

    # Echo model predicts 0; warped rel-L2 of zeros vs targets.
    l0 = lpbf_warped_rel_l2(torch.zeros(1, 4, 1), g0.y.unsqueeze(0), yn)
    l1 = lpbf_warped_rel_l2(torch.zeros(1, 3, 1), g1.y.unsqueeze(0), yn)
    expect = (l0 + l1) / 2
    assert torch.allclose(loss, expect, rtol=1e-5, atol=1e-6)


def test_lpbf_flare_batch_loss_varlen_matches_padded() -> None:
    model = _EchoModel()
    yn = _IdentityY()
    graphs = [_graph(4, 1.0), _graph(3, 2.0)]
    loss_p = lpbf_flare_batch_loss(model, lpbf_collate_padded(graphs), yn)
    loss_v = lpbf_flare_batch_loss(model, lpbf_collate_varlen(graphs), yn)
    assert torch.allclose(loss_p, loss_v, rtol=1e-5, atol=1e-6)


def test_lpbf_flare_batch_loss_padded_format_uses_mask_not_flash_varlen() -> None:
    model = _EchoModel()
    yn = _IdentityY()
    batch = lpbf_collate_padded([_graph(4, 1.0), _graph(3, 2.0)])

    loss = lpbf_flare_batch_loss(model, batch, yn)

    assert torch.isfinite(loss)
    assert len(model.calls) == 1
    assert model.calls[0]["mask"] is not None
    assert model.calls[0]["use_flash_varlen"] is False
    assert model.calls[0]["cu_seqlens"] is None


def test_lpbf_flare_batch_loss_rejects_unknown_dict_format() -> None:
    model = _EchoModel()
    yn = _IdentityY()
    batch = lpbf_collate_padded([_graph(2, 1.0)])
    batch["format"] = "legacy"

    import pytest

    with pytest.raises(ValueError, match="LPBF batch format"):
        lpbf_flare_batch_loss(model, batch, yn)


def test_lpbf_warped_rel_l2_uses_unified_field_loss() -> None:
    from pdebench.dataset.loss import compute_field_loss
    from pdebench.dataset.sample import LossSpec

    torch.manual_seed(0)
    yh = torch.randn(2, 5, 1)
    y = torch.randn(2, 5, 1) + 1.0
    yn = UnitGaussianNormalizer(y)
    got = lpbf_warped_rel_l2(yh, y, yn)
    expect = compute_field_loss(yh, y, yn, LossSpec())
    assert torch.allclose(got, expect)
