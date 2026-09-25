"""Dataset wrappers that round-trip legacy items through ``Sample`` (C2a/C4).

``RoundTripPygDataset`` converts each ``__getitem__`` PyG ``Data`` through
``pyg_data_to_sample`` / ``sample_to_pyg_data`` so downstream training code keeps
consuming legacy ``Data`` objects while the ``Sample`` bridge stays exercised on
every access. With ``yield_sample=True`` (C4 Sample collate), ``__getitem__``
stops at the ``Sample`` instead of converting back to ``Data`` — the
``Sample`` -> ``Data`` -> ``Batch`` conversion moves to collate time via
``pdebench.dataset.sample_collate.collate_plaid_static``.

``RoundTripGinotDataset`` does the same for the ``dict`` items GINOT datasets
(``GinotDataset`` / ``GraphCacheDataset``) produce, via ``ginot_dict_to_sample`` /
``sample_to_ginot_dict``, with the same C4 ``yield_sample`` escape hatch.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from pdebench.dataset.sample import SampleKind
from pdebench.dataset.sample_bridge import (
    ginot_dict_to_sample,
    pyg_data_to_sample,
    sample_to_ginot_dict,
    sample_to_pyg_data,
)


class RoundTripPygDataset(Dataset):
    """C2a: legacy PyG -> Sample -> PyG on every __getitem__ (C4: optionally stop at Sample).

    ``yield_sample=False`` (default, C2a) round-trips back to ``Data`` so
    training keeps consuming legacy PyG batches unchanged. ``yield_sample=True``
    (C4) returns the intermediate ``Sample`` directly; pair with
    ``pdebench.dataset.sample_collate.collate_plaid_static`` as the
    ``DataLoader`` collate function.
    """

    def __init__(self, base: Dataset, *, kind: SampleKind = SampleKind.STATIC, yield_sample: bool = False):
        self.base = base
        self.kind = kind
        self.yield_sample = yield_sample

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        sample = pyg_data_to_sample(data, kind=self.kind)
        if self.yield_sample:
            return sample
        return sample_to_pyg_data(sample)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class RoundTripGinotDataset(Dataset):
    """C2a: legacy GINOT dict -> Sample -> dict on every __getitem__ (C4: optionally stop at Sample).

    ``yield_sample=False`` (default, C2a) round-trips back to ``dict`` so
    training keeps consuming legacy GINOT batches unchanged. ``yield_sample=True``
    (C4) returns the intermediate ``Sample`` directly; pair with
    ``pdebench.dataset.sample_collate.collate_ginot`` as the ``DataLoader``
    collate function.
    """

    def __init__(self, base: Dataset, *, kind: SampleKind = SampleKind.STATIC, yield_sample: bool = False):
        self.base = base
        self.kind = kind
        self.yield_sample = yield_sample

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        sample = ginot_dict_to_sample(item, kind=self.kind)
        if self.yield_sample:
            return sample
        return sample_to_ginot_dict(sample)

    def __getattr__(self, name: str):
        return getattr(self.base, name)
