from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.data import Dataset

from pdebench.dataset.ginot.collate import ginot_collate_fn
from pdebench.dataset.sample import FeatureRequest, Sample, SampleKind
from pdebench.dataset.sample_bridge import ginot_dict_to_sample, sample_to_ginot_dict
from tests.pdebench.goldens.helpers import assert_sample_roundtrip


def _tiny_ginot_dict() -> dict:
    return {
        "pos": torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32),
        "y": torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32),
        "boundary_pos": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        "sample_id": torch.tensor(7, dtype=torch.long),
        "edge_cache_key": "7:2",
        "graph_cache_dir": "/tmp/cache",
    }


def test_ginot_dict_roundtrip_preserves_core_fields() -> None:
    d = _tiny_ginot_dict()
    sample = ginot_dict_to_sample(d, kind=SampleKind.STATIC)
    assert sample.sample_id == "7"
    assert sample.kind == SampleKind.STATIC
    assert_sample_roundtrip(
        sample,
        to_legacy=lambda s: sample_to_ginot_dict(s),
        from_legacy=lambda d: ginot_dict_to_sample(d, kind=SampleKind.STATIC),
        fields=("pos", "y", "edge_index", "edge_attr", "boundary_pos", "sample_id", "kind"),
    )


def test_ginot_dict_to_sample_puts_non_sample_keys_in_extras() -> None:
    d = _tiny_ginot_dict()
    sample = ginot_dict_to_sample(d, kind=SampleKind.STATIC)
    assert sample.extras == {"edge_cache_key": "7:2", "graph_cache_dir": "/tmp/cache"}

    back = sample_to_ginot_dict(sample)
    assert back["edge_cache_key"] == "7:2"
    assert back["graph_cache_dir"] == "/tmp/cache"


def test_ginot_dict_to_sample_missing_sample_id_defaults_to_zero() -> None:
    d = {
        "pos": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        "y": torch.tensor([[0.5], [1.5]], dtype=torch.float32),
    }
    sample = ginot_dict_to_sample(d, kind=SampleKind.STATIC)
    assert sample.sample_id == "0"


def test_ginot_dict_to_sample_falls_back_to_idx_key() -> None:
    d = {
        "pos": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "y": torch.tensor([[0.5]], dtype=torch.float32),
        "idx": 3,
    }
    sample = ginot_dict_to_sample(d, kind=SampleKind.STATIC)
    assert sample.sample_id == "3"


class _TinyGinotDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_wrapper_getitem_roundtrips() -> None:
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    base = _TinyGinotDataset([_tiny_ginot_dict()])
    wrapped = RoundTripGinotDataset(base, kind=SampleKind.STATIC)
    out = wrapped[0]
    assert torch.equal(out["pos"], base[0]["pos"])
    assert torch.equal(out["y"], base[0]["y"])
    assert out["edge_cache_key"] == base[0]["edge_cache_key"]


def test_ginot_canonical_adapter_wraps_all_ginot_families(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    for name in (
        "poisson_unstructured",
        "bracket_lug",
        "micro_puc",
        "micro_puc_fixed",
        "deform_plate",
    ):
        train_ds = _TinyGinotDataset([_tiny_ginot_dict()])
        test_ds = _TinyGinotDataset([_tiny_ginot_dict()])

        def _fake_ginot_loader(**kwargs):
            return train_ds, test_ds, {"meta": 1}

        monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            ginot_max_samples=0,
            ginot_use_flash_varlen=False,
            feature_request=FeatureRequest(),
        )

        assert isinstance(train, RoundTripGinotDataset)
        assert isinstance(test, RoundTripGinotDataset)
        assert meta["sample_bridge"] == "c2a_ginot"
        assert torch.equal(train[0]["pos"], train_ds[0]["pos"])
        assert torch.equal(test[0]["pos"], test_ds[0]["pos"])


def test_ginot_canonical_adapter_skips_wrap_when_sample_bridge_disabled(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    monkeypatch.setenv("PDEBENCH_SAMPLE_BRIDGE", "0")

    for name in (
        "poisson_unstructured",
        "bracket_lug",
        "micro_puc",
        "micro_puc_fixed",
        "deform_plate",
    ):
        train_ds = _TinyGinotDataset([_tiny_ginot_dict()])
        test_ds = _TinyGinotDataset([_tiny_ginot_dict()])

        def _fake_ginot_loader(**kwargs):
            return train_ds, test_ds, {"meta": 1}

        monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            ginot_max_samples=0,
            ginot_use_flash_varlen=False,
            feature_request=FeatureRequest(),
        )

        assert not isinstance(train, RoundTripGinotDataset)
        assert not isinstance(test, RoundTripGinotDataset)
        assert train is train_ds
        assert test is test_ds
        assert "sample_bridge" not in meta


def test_ginot_canonical_adapter_skips_wrap_for_non_dataset_sentinels(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter

    def _fake_ginot_loader(**kwargs):
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    adapter = get_adapter("poisson_unstructured")
    train, test, meta = adapter.load(
        "unused",
        mesh_split_seed=0,
        ginot_max_samples=0,
        ginot_use_flash_varlen=False,
        feature_request=FeatureRequest(),
    )

    assert (train, test) == ("train", "test")
    assert meta == {"meta": 1, "sample_bridge": "c2a_ginot"}


def test_lpbf_adapter_wraps_ginot_dict_path(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    train_ds = _TinyGinotDataset([_tiny_ginot_dict()])
    test_ds = _TinyGinotDataset([_tiny_ginot_dict()])

    def _fake_lpbf_loader(**kwargs):
        assert kwargs["include_edges"] is True
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_lpbf_loader)

    adapter = get_adapter("lpbf")
    train, test, meta = adapter.load(
        "unused",
        ginot_use_flash_varlen=True,
        mesh_split_seed=0,
        ginot_max_samples=0,
        feature_request=FeatureRequest(edges=True),
    )

    assert isinstance(train, RoundTripGinotDataset)
    assert isinstance(test, RoundTripGinotDataset)
    assert meta["sample_bridge"] == "c2a_ginot"
    assert torch.equal(train[0]["pos"], train_ds[0]["pos"])


def test_lpbf_adapter_skips_wrap_for_pyg_flare_path(monkeypatch) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    marker = object()

    def _fake_lpbf_loader(**kwargs):
        assert kwargs["include_edges"] is False
        return marker, marker, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_lpbf_loader)

    adapter = get_adapter("lpbf")
    train, test, meta = adapter.load(
        "unused",
        ginot_use_flash_varlen=False,
        mesh_split_seed=0,
        ginot_max_samples=0,
        feature_request=FeatureRequest(),
    )

    assert train is marker
    assert test is marker
    assert not isinstance(train, RoundTripGinotDataset)
    assert "sample_bridge" not in meta


def test_wrapper_yield_sample_mode_returns_sample_not_dict() -> None:
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    base = _TinyGinotDataset([_tiny_ginot_dict()])
    wrapped = RoundTripGinotDataset(base, kind=SampleKind.STATIC, yield_sample=True)
    out = wrapped[0]
    assert isinstance(out, Sample)
    assert torch.equal(out.pos, base[0]["pos"])
    assert torch.equal(out.y, base[0]["y"])


_GINOT_SAMPLE_COLLATE_NAMES = (
    "poisson_unstructured",
    "poisson_structured",
    "bracket_lug",
    "micro_puc",
    "micro_puc_fixed",
    "deform_plate",
)


def test_ginot_canonical_adapter_sample_collate_default_off(monkeypatch) -> None:
    """PDEBENCH_SAMPLE_COLLATE unset: dataset still yields dict (C2a only, no C4)."""
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter

    monkeypatch.delenv("PDEBENCH_SAMPLE_COLLATE", raising=False)

    train_ds = _TinyGinotDataset([_tiny_ginot_dict()])
    test_ds = _TinyGinotDataset([_tiny_ginot_dict()])

    def _fake_ginot_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1, "train_collate_fn": ginot_collate_fn, "eval_collate_fn": ginot_collate_fn}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    adapter = get_adapter("poisson_unstructured")
    train, _test, meta = adapter.load(
        "unused",
        mesh_split_seed=0,
        ginot_max_samples=0,
        ginot_use_flash_varlen=False,
        feature_request=FeatureRequest(),
    )

    assert "sample_collate" not in meta
    assert isinstance(train[0], dict)
    assert meta["train_collate_fn"] is ginot_collate_fn


@pytest.mark.parametrize("env_value", ["ginot", "1", "poisson_unstructured,bracket_lug"])
def test_ginot_canonical_adapter_sample_collate_enabled_via_env(monkeypatch, env_value) -> None:
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_collate import collate_samples_ginot

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", env_value)

    for name in ("poisson_unstructured", "bracket_lug"):
        dicts = [_tiny_ginot_dict(), _tiny_ginot_dict()]
        train_ds = _TinyGinotDataset(dicts)
        test_ds = _TinyGinotDataset([_tiny_ginot_dict()])
        legacy_collate_fn = partial(
            ginot_collate_fn, pad_to_nodes=5, pad_to_boundary_nodes=3, use_flash_varlen=False,
            include_padded_boundary=True,
        )

        def _fake_ginot_loader(**kwargs):
            return train_ds, test_ds, {
                "meta": 1, "train_collate_fn": legacy_collate_fn, "eval_collate_fn": legacy_collate_fn,
            }

        monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

        adapter = get_adapter(name)
        train, test, meta = adapter.load(
            "unused",
            mesh_split_seed=0,
            ginot_max_samples=0,
            ginot_use_flash_varlen=False,
            feature_request=FeatureRequest(),
        )

        assert meta["sample_collate"] is True
        assert meta["collate_fn_name"] == "ginot_sample"
        assert meta["train_collate_fn"].func is collate_samples_ginot
        assert meta["train_collate_fn"].keywords == legacy_collate_fn.keywords
        assert meta["eval_collate_fn"] is meta["train_collate_fn"]
        assert isinstance(train[0], Sample)
        assert isinstance(test[0], Sample)

        got = meta["train_collate_fn"]([train[0], train[1]])
        expected = legacy_collate_fn(dicts)
        assert set(got) == set(expected)
        for field in ("pos", "y", "mask"):
            assert torch.equal(got[field], expected[field]), field


def test_ginot_canonical_adapter_sample_collate_scoped_to_ginot_families_only(monkeypatch) -> None:
    """PDEBENCH_SAMPLE_COLLATE=1 must not affect LPBF (out of this task's scope)."""
    import pdebench.dataset.utils as dataset_utils
    from pdebench.dataset.adapters import get_adapter
    from pdebench.dataset.sample_wrappers import RoundTripGinotDataset

    monkeypatch.setenv("PDEBENCH_SAMPLE_COLLATE", "1")

    train_ds = _TinyGinotDataset([_tiny_ginot_dict()])
    test_ds = _TinyGinotDataset([_tiny_ginot_dict()])

    def _fake_lpbf_loader(**kwargs):
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_lpbf_dataset", _fake_lpbf_loader)

    adapter = get_adapter("lpbf")
    train, _test, meta = adapter.load(
        "unused",
        ginot_use_flash_varlen=True,
        mesh_split_seed=0,
        ginot_max_samples=0,
        feature_request=FeatureRequest(edges=True),
    )

    assert "sample_collate" not in meta
    assert isinstance(train, RoundTripGinotDataset)
    assert isinstance(train[0], dict)
