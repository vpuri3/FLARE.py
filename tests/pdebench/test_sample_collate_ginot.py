from __future__ import annotations

import pytest
import torch

from pdebench.dataset.ginot.collate import ginot_collate_fn
from pdebench.dataset.sample import Sample, SampleKind
from pdebench.dataset.sample_bridge import ginot_dict_to_sample, sample_to_ginot_dict
from pdebench.dataset.sample_collate import collate_ginot, collate_samples_ginot


def _tiny_ginot_dict(sample_id: int, num_nodes: int, num_boundary_nodes: int) -> dict:
    return {
        "pos": torch.arange(num_nodes * 2, dtype=torch.float32).reshape(num_nodes, 2),
        "boundary_pos": torch.arange(num_boundary_nodes * 2, dtype=torch.float32).reshape(num_boundary_nodes, 2) + 100.0,
        "y": torch.arange(num_nodes, dtype=torch.float32).reshape(num_nodes, 1) + 0.5,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long) % num_nodes,
        "edge_attr": torch.ones(2, 3, dtype=torch.float32),
        "sample_id": torch.tensor(sample_id, dtype=torch.long),
    }


def _tiny_ginot_dicts() -> list[dict]:
    return [_tiny_ginot_dict(0, 2, 1), _tiny_ginot_dict(1, 3, 2)]


def _tiny_samples() -> list[Sample]:
    return [ginot_dict_to_sample(d, kind=SampleKind.STATIC) for d in _tiny_ginot_dicts()]


def _assert_batches_equal(got: dict, expected: dict) -> None:
    assert set(got) == set(expected), (set(got), set(expected))
    for key, expected_val in expected.items():
        got_val = got[key]
        if torch.is_tensor(expected_val):
            assert got_val.shape == expected_val.shape, key
            assert torch.equal(got_val, expected_val), key
        else:
            assert got_val == expected_val, key


def test_collate_samples_ginot_matches_ginot_collate_fn_on_dicts() -> None:
    dicts = _tiny_ginot_dicts()
    samples = _tiny_samples()
    expected = ginot_collate_fn(dicts)

    got = collate_samples_ginot(samples)

    _assert_batches_equal(got, expected)
    for field in ("pos", "y", "mask"):
        assert torch.equal(got[field], expected[field]), field


def test_collate_samples_ginot_forwards_kwargs() -> None:
    dicts = _tiny_ginot_dicts()
    samples = _tiny_samples()
    kwargs = dict(pad_to_nodes=5, pad_to_boundary_nodes=4, use_flash_varlen=False, include_padded_boundary=True)
    expected = ginot_collate_fn(dicts, **kwargs)

    got = collate_samples_ginot(samples, **kwargs)

    _assert_batches_equal(got, expected)


def test_collate_samples_ginot_flash_varlen_matches() -> None:
    dicts = _tiny_ginot_dicts()
    samples = _tiny_samples()
    expected = ginot_collate_fn(dicts, use_flash_varlen=True)

    got = collate_samples_ginot(samples, use_flash_varlen=True)

    _assert_batches_equal(got, expected)


def test_collate_ginot_on_samples_matches_dict_path() -> None:
    dicts = _tiny_ginot_dicts()
    samples = _tiny_samples()
    expected = ginot_collate_fn(dicts)

    got = collate_ginot(samples)

    _assert_batches_equal(got, expected)


def test_collate_ginot_on_dict_list_matches_ginot_collate_fn() -> None:
    dicts = _tiny_ginot_dicts()
    expected = ginot_collate_fn(dicts)

    got = collate_ginot(dicts)

    _assert_batches_equal(got, expected)


def test_collate_ginot_sample_and_dict_paths_agree() -> None:
    dicts = _tiny_ginot_dicts()
    samples = _tiny_samples()

    from_samples = collate_ginot(samples)
    from_dicts = collate_ginot(dicts)

    _assert_batches_equal(from_samples, from_dicts)


def test_collate_ginot_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        collate_ginot([])
    with pytest.raises(ValueError):
        collate_samples_ginot([])


def test_collate_ginot_rejects_unknown_item_type() -> None:
    with pytest.raises(TypeError):
        collate_ginot([object()])


def test_sample_to_ginot_dict_sample_id_is_scalar_long_tensor() -> None:
    sample = _tiny_samples()[0]
    out = sample_to_ginot_dict(sample)
    assert torch.is_tensor(out["sample_id"])
    assert out["sample_id"].dtype == torch.long
    assert out["sample_id"].reshape(()).item() == 0
