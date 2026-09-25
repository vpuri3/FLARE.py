"""Sample-native collates for canonical PLAID mesh-static and GINOT families (C4).

``collate_samples_plaid_static`` converts each ``Sample`` to a PyG ``Data`` via
``sample_bridge.sample_to_pyg_data`` and batches with ``Batch.from_data_list``,
producing the same ``Batch`` that ``torch_geometric.loader.DataLoader``'s
``Collater`` builds from the equivalent legacy ``Data`` list (see
``tests/pdebench/test_sample_collate_plaid.py`` for the equality goldens).

``collate_plaid_static`` is the dispatch entry point training wires up as
``metadata["train_collate_fn"]`` / ``metadata["eval_collate_fn"]``: it detects
whether the incoming batch already holds ``Sample`` or legacy PyG ``Data``
items so callers do not need to know which mode the dataset behind them is in
(``RoundTripPygDataset`` with ``yield_sample=False`` still yields ``Data``;
only families with ``PDEBENCH_SAMPLE_COLLATE`` enabled yield ``Sample``).

``collate_samples_ginot`` / ``collate_ginot`` are the analogous pair for the
canonical GINOT families: convert each ``Sample`` to the legacy GINOT ``dict``
via ``sample_bridge.sample_to_ginot_dict`` and delegate to the existing
``ginot_collate_fn`` (see ``tests/pdebench/test_sample_collate_ginot.py`` for
the equality goldens vs. ``ginot_collate_fn`` on legacy dicts).
"""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Batch, Data

from pdebench.dataset.ginot.collate import ginot_collate_fn
from pdebench.dataset.sample import Sample
from pdebench.dataset.sample_bridge import sample_to_ginot_dict, sample_to_pyg_data


def collate_samples_plaid_static(samples: list[Sample]) -> Batch:
    """Convert each ``Sample`` to PyG ``Data`` then batch with ``Batch.from_data_list``."""
    if not samples:
        raise ValueError("collate_samples_plaid_static received an empty batch.")
    data_list = [sample_to_pyg_data(sample) for sample in samples]
    return Batch.from_data_list(data_list)


def collate_plaid_static(batch: list[Sample] | list[Data]) -> Batch:
    """Dispatch to the ``Sample`` or legacy ``Data`` collate path based on item type."""
    if not batch:
        raise ValueError("collate_plaid_static received an empty batch.")
    first = batch[0]
    if isinstance(first, Sample):
        return collate_samples_plaid_static(batch)
    if isinstance(first, Data):
        return Batch.from_data_list(batch)
    raise TypeError(f"collate_plaid_static expects Sample or PyG Data items, got {type(first)!r}")


def collate_samples_ginot(samples: list[Sample], **kwargs: Any) -> dict[str, Any]:
    """Convert each ``Sample`` to a legacy GINOT dict then run ``ginot_collate_fn``."""
    if not samples:
        raise ValueError("collate_samples_ginot received an empty batch.")
    dict_list = [sample_to_ginot_dict(sample) for sample in samples]
    return ginot_collate_fn(dict_list, **kwargs)


def collate_ginot(batch: list[Sample] | list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Dispatch to the ``Sample`` or legacy ``dict`` collate path based on item type."""
    if not batch:
        raise ValueError("collate_ginot received an empty batch.")
    first = batch[0]
    if isinstance(first, Sample):
        return collate_samples_ginot(batch, **kwargs)
    if isinstance(first, dict):
        return ginot_collate_fn(batch, **kwargs)
    raise TypeError(f"collate_ginot expects Sample or dict items, got {type(first)!r}")
