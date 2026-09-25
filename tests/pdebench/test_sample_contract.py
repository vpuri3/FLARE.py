from __future__ import annotations

import torch

from pdebench.dataset.sample import FeatureRequest, LossSpec, Sample, SampleKind


def test_feature_request_defaults() -> None:
    req = FeatureRequest()
    assert req.edges is False
    assert req.boundary is False
    assert req.laplacian_k == 0
    assert req.laplacian_spec == "graph"


def test_sample_with_tensors_and_kind() -> None:
    pos = torch.randn(8, 3)
    y = torch.randn(8, 2)
    sample = Sample(
        pos=pos,
        y=y,
        sample_id="elasticity/0",
        kind=SampleKind.STATIC,
    )
    assert sample.pos is pos
    assert sample.y is y
    assert sample.sample_id == "elasticity/0"
    assert sample.kind is SampleKind.STATIC
    assert sample.kind == "static"
    assert sample.edge_index is None
    assert sample.edge_attr is None
    assert sample.feats is None
    assert sample.boundary_pos is None
    assert sample.state_in is None
    assert sample.context is None
    assert sample.masks == {}
    assert sample.laplacian_eig is None
    assert sample.laplacian_eigvals is None
    assert sample.extras == {}


def test_sample_kind_values() -> None:
    assert SampleKind.STATIC == "static"
    assert SampleKind.TRANSITION == "transition"
    assert SampleKind.TERMINAL == "terminal"


def test_loss_spec_mask_and_extras() -> None:
    empty = LossSpec()
    assert empty.mask is None
    assert empty.extras == ()

    spec = LossSpec(mask="free_mask", extras=("bc_l2",))
    assert spec.mask == "free_mask"
    assert spec.extras == ("bc_l2",)
