"""Static PLAID must not double-concat graph scalars on the mesh GLT path."""

from __future__ import annotations

import types

import torch

from pdebench.dataset.mesh_runtime import _mesh_glt_forward


class _CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_feats = None

    def forward(self, pos, feats=None, **kwargs):
        del kwargs
        self.last_feats = None if feats is None else feats.detach().clone()
        n = pos.shape[0]
        return torch.zeros(n, 1)


def _pyg_like_batch(*, x: torch.Tensor, input_scalars: torch.Tensor | None, with_context: bool):
    n = x.shape[0]
    batch = types.SimpleNamespace(
        x=x,
        y=torch.zeros(n, 1),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        ptr=torch.tensor([0, n], dtype=torch.int32),
        batch=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        input_scalars=input_scalars,
    )
    if with_context:
        batch.context = input_scalars
    return batch


def test_static_plaid_does_not_reconcat_input_scalars() -> None:
    # Scalars already baked into x (static PLAID); input_scalars present but no context.
    x = torch.randn(4, 6)  # pos(2) + feats(4) including baked scalars
    scalars = torch.randn(1, 2)
    model = _CaptureModel()
    _mesh_glt_forward(model, _pyg_like_batch(x=x, input_scalars=scalars, with_context=False))
    assert model.last_feats is not None
    assert model.last_feats.shape[-1] == 4  # x[:, 2:] only


def test_elpl_concats_graph_level_context() -> None:
    x = torch.randn(4, 4)  # pos(2) + node feats(2); time NOT in x
    scalars = torch.tensor([[0.25]])
    model = _CaptureModel()
    _mesh_glt_forward(model, _pyg_like_batch(x=x, input_scalars=scalars, with_context=True))
    assert model.last_feats is not None
    assert model.last_feats.shape[-1] == 3  # node feats(2) + time(1)
    assert torch.allclose(model.last_feats[:, -1], torch.full((4,), 0.25))
