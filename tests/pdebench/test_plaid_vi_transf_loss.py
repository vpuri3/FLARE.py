"""Vi-Transf λ-blended PLAID train loss."""

from __future__ import annotations

import types

import torch

from pdebench.dataset.mesh_runtime import mesh_batch_plaid_scaled_mse


def test_plaid_scaled_mse_lbda_one_is_field_only() -> None:
    yh = torch.zeros(4, 3)
    y = torch.ones(4, 2)
    batch_index = torch.tensor([0, 0, 1, 1])
    batch = types.SimpleNamespace()
    loss, field_mse, scalar_mse = mesh_batch_plaid_scaled_mse(
        yh, y, batch, batch_index=batch_index, num_graphs=2, field_dim=2, scalar_dim=0, lbda=1.0
    )
    assert torch.allclose(loss, field_mse)
    assert float(scalar_mse) == 0.0
    assert torch.allclose(field_mse, torch.nn.functional.mse_loss(yh[:, :2], y))


def test_plaid_scaled_mse_lbda_half_blends_scalar() -> None:
    yh = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    y = torch.zeros(4, 2)
    batch_index = torch.tensor([0, 0, 1, 1])
    batch = types.SimpleNamespace(output_scalars=torch.tensor([[1.0], [1.0]], dtype=torch.float32))
    loss, field_mse, scalar_mse = mesh_batch_plaid_scaled_mse(
        yh, y, batch, batch_index=batch_index, num_graphs=2, field_dim=2, scalar_dim=1, lbda=0.5
    )
    # scalar preds are 2 and 4 → MSE vs 1 is ((1)^2 + (3)^2)/2 = 5
    expect_scalar = torch.tensor(5.0)
    expect_field = torch.nn.functional.mse_loss(yh[:, :2], y)
    expect = 0.5 * expect_field + 0.5 * expect_scalar
    assert torch.allclose(scalar_mse, expect_scalar)
    assert torch.allclose(field_mse, expect_field)
    assert torch.allclose(loss, expect)
