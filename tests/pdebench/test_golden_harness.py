from __future__ import annotations

import torch

from pdebench.dataset.sample import Sample, SampleKind
from tests.pdebench.goldens.helpers import assert_sample_roundtrip, load_manifest


def test_manifest_loads() -> None:
    manifest = load_manifest()
    assert "version" in manifest
    assert isinstance(manifest.get("entries"), list)


def test_assert_sample_roundtrip_identity() -> None:
    sample = Sample(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        y=torch.tensor([[1.0], [2.0]]),
        sample_id="0",
        kind=SampleKind.STATIC,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    # identity converter pair
    assert_sample_roundtrip(
        sample,
        to_legacy=lambda s: s,
        from_legacy=lambda s: s,
        fields=("pos", "y", "edge_index", "sample_id", "kind"),
    )
