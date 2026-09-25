from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from pdebench.dataset.sample import Sample, SampleKind
from pdebench.dataset.sample_bridge import sample_to_pyg_data
from pdebench.dataset.sample_collate import collate_plaid_static, collate_samples_plaid_static


def _tiny_sample(sample_id: str, num_nodes: int) -> Sample:
    return Sample(
        pos=torch.arange(num_nodes * 2, dtype=torch.float32).reshape(num_nodes, 2),
        y=torch.arange(num_nodes, dtype=torch.float32).reshape(num_nodes, 1) + 0.5,
        sample_id=sample_id,
        kind=SampleKind.STATIC,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long) % num_nodes,
        edge_attr=torch.ones(2, 1, dtype=torch.float32),
        feats=torch.ones(num_nodes, 3, dtype=torch.float32),
    )


def _tiny_samples() -> list[Sample]:
    return [_tiny_sample("1", 2), _tiny_sample("2", 3)]


def test_collate_samples_plaid_static_matches_batch_from_data_list() -> None:
    samples = _tiny_samples()
    legacy_data_list = [sample_to_pyg_data(sample) for sample in samples]
    expected = Batch.from_data_list(legacy_data_list)

    got = collate_samples_plaid_static(samples)

    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(got, field), getattr(expected, field)), field


def test_collate_plaid_static_on_samples_matches_data_list_path() -> None:
    samples = _tiny_samples()
    legacy_data_list = [sample_to_pyg_data(sample) for sample in samples]
    expected = Batch.from_data_list(legacy_data_list)

    got = collate_plaid_static(samples)

    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(got, field), getattr(expected, field)), field


def test_collate_plaid_static_on_data_list_matches_batch_from_data_list() -> None:
    samples = _tiny_samples()
    legacy_data_list = [sample_to_pyg_data(sample) for sample in samples]
    expected = Batch.from_data_list(legacy_data_list)

    got = collate_plaid_static(legacy_data_list)

    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(got, field), getattr(expected, field)), field


def test_collate_plaid_static_sample_and_data_paths_agree() -> None:
    samples = _tiny_samples()
    legacy_data_list = [sample_to_pyg_data(sample) for sample in samples]

    from_samples = collate_plaid_static(samples)
    from_data = collate_plaid_static(legacy_data_list)

    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(from_samples, field), getattr(from_data, field)), field


def test_collate_plaid_static_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        collate_plaid_static([])
    with pytest.raises(ValueError):
        collate_samples_plaid_static([])


def test_collate_plaid_static_rejects_unknown_item_type() -> None:
    with pytest.raises(TypeError):
        collate_plaid_static([{"pos": torch.zeros(1, 2)}])


def test_collate_plaid_static_single_item_batch() -> None:
    sample = _tiny_sample("7", 4)
    got = collate_plaid_static([sample])
    expected = Batch.from_data_list([sample_to_pyg_data(sample)])
    for field in ("x", "y", "pos", "edge_index", "batch"):
        assert torch.equal(getattr(got, field), getattr(expected, field)), field
    assert got.num_graphs == 1
