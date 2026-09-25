"""Converters between legacy dataset item formats and the canonical ``Sample`` facade.

C2a (PLAID static): ``pyg_data_to_sample`` / ``sample_to_pyg_data`` round-trip the
fields ``load_mesh_static_dataset`` graphs carry (``pos``, ``y``, ``x`` <-> ``feats``,
``edge_index``, ``edge_attr``, ``context``, laplacian caches) so ``RoundTripPygDataset``
can wrap PLAID datasets without changing on-disk cache layout.

C2a (GINOT small): ``ginot_dict_to_sample`` / ``sample_to_ginot_dict`` round-trip the
``dict`` items ``GinotDataset`` / ``GraphCacheDataset`` produce so ``RoundTripGinotDataset``
can wrap ``poisson_unstructured`` without changing the on-disk cache layout. Keys the
``Sample`` dataclass does not model (``edge_cache_key``, ``graph_cache_dir``, ``cells``,
and anything ``attach_sample_feats`` adds) round-trip through ``Sample.extras``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from pdebench.dataset.sample import Sample, SampleKind

if TYPE_CHECKING:
    from torch_geometric.data import Data

_PYG_EXTRA_KEYS = (
    "input_scalars",
    "output_scalars",
    "context",
    "laplacian_eig",
    "laplacian_eigvals",
)


def _sample_id_from_pyg(data: Data) -> str:
    sid = getattr(data, "sample_id", None)
    if sid is None:
        return "0"
    if torch.is_tensor(sid):
        return str(int(sid.reshape(-1)[0].item()))
    return str(sid)


def pyg_data_to_sample(data: Data, *, kind: SampleKind = SampleKind.STATIC) -> Sample:
    pos = data.pos
    y = data.y
    if y is None:
        y = torch.zeros(pos.shape[0], 1, dtype=pos.dtype)
    feats = getattr(data, "x", None)
    extras: dict[str, Any] = {}
    for key in _PYG_EXTRA_KEYS:
        if key in ("laplacian_eig", "laplacian_eigvals"):
            continue
        if hasattr(data, key) and getattr(data, key) is not None:
            extras[key] = getattr(data, key)
    return Sample(
        pos=pos,
        y=y,
        sample_id=_sample_id_from_pyg(data),
        kind=kind,
        edge_index=getattr(data, "edge_index", None),
        edge_attr=getattr(data, "edge_attr", None),
        feats=feats,
        context=getattr(data, "context", None),
        laplacian_eig=getattr(data, "laplacian_eig", None),
        laplacian_eigvals=getattr(data, "laplacian_eigvals", None),
        extras=extras,
    )


def sample_to_pyg_data(sample: Sample) -> Data:
    from torch_geometric.data import Data

    kwargs: dict[str, Any] = {
        "pos": sample.pos,
        "y": sample.y,
        "sample_id": torch.tensor([int(sample.sample_id)], dtype=torch.long),
    }
    if sample.feats is not None:
        kwargs["x"] = sample.feats
    if sample.edge_index is not None:
        kwargs["edge_index"] = sample.edge_index
    if sample.edge_attr is not None:
        kwargs["edge_attr"] = sample.edge_attr
    if sample.context is not None:
        kwargs["context"] = sample.context
    if sample.laplacian_eig is not None:
        kwargs["laplacian_eig"] = sample.laplacian_eig
    if sample.laplacian_eigvals is not None:
        kwargs["laplacian_eigvals"] = sample.laplacian_eigvals
    for key, value in sample.extras.items():
        kwargs[key] = value
    return Data(**kwargs)


_GINOT_KNOWN_KEYS = (
    "pos",
    "y",
    "edge_index",
    "edge_attr",
    "feats",
    "boundary_pos",
    "laplacian_eig",
    "laplacian_eigvals",
    "sample_id",
    "idx",
)


def _sample_id_from_ginot(d: dict[str, Any]) -> str:
    sid = d.get("sample_id", d.get("idx", 0))
    if torch.is_tensor(sid):
        return str(int(sid.reshape(-1)[0].item()))
    return str(sid)


def ginot_dict_to_sample(d: dict[str, Any], *, kind: SampleKind = SampleKind.STATIC) -> Sample:
    extras = {key: value for key, value in d.items() if key not in _GINOT_KNOWN_KEYS}
    return Sample(
        pos=d["pos"],
        y=d["y"],
        sample_id=_sample_id_from_ginot(d),
        kind=kind,
        edge_index=d.get("edge_index"),
        edge_attr=d.get("edge_attr"),
        feats=d.get("feats"),
        boundary_pos=d.get("boundary_pos"),
        laplacian_eig=d.get("laplacian_eig"),
        laplacian_eigvals=d.get("laplacian_eigvals"),
        extras=extras,
    )


def sample_to_ginot_dict(sample: Sample) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pos": sample.pos,
        "y": sample.y,
        # ``ginot_collate_fn`` calls ``sample["sample_id"].reshape(())``, so this
        # must round-trip as the scalar long tensor ``GinotDataset``/``GraphCacheDataset``
        # produce, not the canonical ``Sample.sample_id`` string.
        "sample_id": torch.tensor(int(sample.sample_id), dtype=torch.long),
    }
    for attr in ("edge_index", "edge_attr", "feats", "boundary_pos", "laplacian_eig", "laplacian_eigvals"):
        val = getattr(sample, attr)
        if val is not None:
            out[attr] = val
    out.update(sample.extras)
    return out
