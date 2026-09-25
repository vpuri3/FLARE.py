from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from pdebench.dataset.ginot.bumper_beam import (
    BUMPER_BEAM_CACHE_TAG,
    compute_bumper_beam_normalizers,
    preload_bumper_beam_store,
)
from pdebench.dataset.ginot.collate import ginot_collate_fn
from pdebench.dataset.ginot.dataset import GinotDataset, GraphCacheDataset
from pdebench.dataset.ginot.deform_plate import (
    DeformPlateSplitSpec,
    compute_deform_plate_normalizers,
    resolve_deform_plate_splits,
)
from pdebench.dataset.ginot.features import active_feats_dim, compute_feats_normalizer
from pdebench.dataset.ginot.graph_cache import (
    ensure_graph_cache,
    graph_cache_dir,
    graph_cache_root,
    is_sharded_lmdb_graph_cache,
    load_graph_cache_normalizers,
    load_graph_cache_sample,
    open_graph_cache_split,
)
from pdebench.dataset.ginot.io import expected_micro_puc_fixed_samples
from pdebench.dataset.ginot.types import (
    MICRO_PUC_MESH_SAMPLES,
    GinotRawDataset,
    StandardNormalizer,
)
from pdebench.dataset.ginot.utils import (
    as_float_array,
    compute_minmax_normalizer,
    compute_normalizer,
    fractional_split_indices,
    identity_normalizer,
    num_indexable_samples,
)
from pdebench.dataset.lpbf import compute_lpbf_normalizers, resolve_lpbf_splits


@dataclass(frozen=True)
class _SplitPolicy:
    test_fraction: float | None = None
    train_size: int | None = None
    test_size: int | None = None
    cache_tag: str = "default"
    split_label: str | None = None
    validate: Callable[[GinotRawDataset, int], None] | None = None


def _validate_micro_puc(raw: GinotRawDataset, num_samples: int) -> None:
    if raw.micro_puc_mesh_idx is None:
        raise ValueError("micro_puc is missing micro_puc_mesh_idx mapping.")
    if num_samples < len(raw.micro_puc_mesh_idx):
        raise ValueError(
            f"micro_puc has {len(raw.micro_puc_mesh_idx)} rows but only {num_samples} indexable samples."
        )


def _validate_micro_puc_fixed(_raw: GinotRawDataset, num_samples: int) -> None:
    expected = expected_micro_puc_fixed_samples()
    if num_samples != expected:
        raise ValueError(
            f"micro_puc_fixed should contain {expected} x/y-periodic samples; found {num_samples}."
        )


_MINMAX_POSITION_DATASETS = frozenset({
    "poisson_unstructured",
    "poisson_structured",
    "bracket_lug",
})


# Random fractional splits (sklearn train_test_split, fixed seed). All listed datasets use the
# full indexable sample count.
_SPLIT_POLICIES: dict[str, _SplitPolicy] = {
    "poisson_unstructured": _SplitPolicy(test_fraction=0.2, split_label="80_20"),
    "poisson_structured": _SplitPolicy(test_fraction=0.2, split_label="80_20"),
    "bracket_lug": _SplitPolicy(test_fraction=0.2, split_label="80_20"),
    "micro_puc": _SplitPolicy(
        test_fraction=0.2,
        cache_tag="full73879",
        split_label="full73879_80_20",
        validate=_validate_micro_puc,
    ),
    "micro_puc_fixed": _SplitPolicy(
        test_fraction=0.2,
        cache_tag="full73879_xyperiodic",
        split_label="full73879_xyperiodic_80_20",
        validate=_validate_micro_puc_fixed,
    ),
    "bumper_beam": _SplitPolicy(
        test_fraction=0.2,
        cache_tag=BUMPER_BEAM_CACHE_TAG,
        split_label=BUMPER_BEAM_CACHE_TAG,
    ),
    "deform_plate": _SplitPolicy(
        cache_tag=DeformPlateSplitSpec().cache_tag,
        split_label=DeformPlateSplitSpec().split_label,
    ),
    "lpbf": _SplitPolicy(
        cache_tag="hf_train_test",
        split_label="hf_train_test",
    ),
}


def _cap_split_indices(
    train_ids: list[int],
    test_ids: list[int],
    *,
    max_samples: int,
) -> tuple[list[int], list[int]]:
    cap = int(max_samples)
    if cap < 0:
        raise ValueError("max_samples must be >= 0.")
    if cap == 0:
        return train_ids, test_ids
    return train_ids[:cap], test_ids[:cap]


def _max_samples_allowed(dataset_name: str) -> bool:
    return dataset_name == "micro_puc_fixed"


def _validate_max_samples(dataset_name: str, max_samples: int) -> None:
    cap = int(max_samples)
    if cap < 0:
        raise ValueError("max_samples must be >= 0.")
    if cap > 0 and not _max_samples_allowed(dataset_name):
        raise ValueError(
            "max_samples > 0 is only supported for micro_puc_fixed."
        )


def _split_train_test(
    dataset_name: str,
    num_samples: int,
    split_seed: int,
    *,
    data_root: str | None = None,
) -> tuple[list[int], list[int]]:
    if dataset_name == "deform_plate":
        if data_root is None:
            raise ValueError("data_root is required for deform_plate split resolution.")
        return resolve_deform_plate_splits(data_root, split_seed)

    if dataset_name == "lpbf":
        return resolve_lpbf_splits(data_root or "", split_seed)

    policy = _SPLIT_POLICIES.get(dataset_name)
    if policy is None:
        raise ValueError(
            f"No GINOT train/test split policy for dataset {dataset_name!r}. "
            f"Supported: {sorted(_SPLIT_POLICIES)}."
        )
    if policy.test_fraction is None:
        raise ValueError(
            f"GINOT split policy for {dataset_name!r} must set test_fraction "
            "(fixed train_size/test_size splits are not supported)."
        )
    return fractional_split_indices(num_samples, split_seed, policy.test_fraction)


def _build_normalizers(
    dataset_name: str,
    raw: GinotRawDataset,
    train_ids: list[int],
) -> tuple[StandardNormalizer, StandardNormalizer, StandardNormalizer, StandardNormalizer | None]:
    def pos_getter(r, i):
        return as_float_array(r.query_points[i], dims=r.space_dim)

    def boundary_getter(r, i):
        return as_float_array(r.point_clouds[i], dims=r.space_dim)

    def target_getter(r, i):
        return as_float_array(r.targets[i])

    if dataset_name == "bumper_beam":
        return compute_bumper_beam_normalizers(raw, train_ids)

    if dataset_name == "deform_plate":
        pos_normalizer, boundary_pos_normalizer, y_normalizer, _scale = compute_deform_plate_normalizers(
            raw, train_ids
        )
        feats_normalizer = None
        return pos_normalizer, boundary_pos_normalizer, y_normalizer, feats_normalizer

    if dataset_name == "lpbf":
        pos_normalizer, boundary_pos_normalizer, y_normalizer = compute_lpbf_normalizers(raw, train_ids)
        return pos_normalizer, boundary_pos_normalizer, y_normalizer, None

    if dataset_name in _MINMAX_POSITION_DATASETS:
        pos_normalizer = compute_minmax_normalizer(raw, train_ids, (pos_getter, boundary_getter))
        boundary_pos_normalizer = pos_normalizer
    else:
        pos_normalizer = (
            compute_normalizer(raw, train_ids, pos_getter)
            if raw.normalize_pos
            else identity_normalizer(raw.space_dim)
        )
        boundary_pos_normalizer = (
            compute_normalizer(raw, train_ids, boundary_getter)
            if raw.normalize_boundary_pos
            else identity_normalizer(raw.space_dim)
        )
    y_normalizer = raw.target_normalizer if raw.target_normalizer is not None else compute_normalizer(
        raw, train_ids, target_getter
    )
    feats_normalizer = compute_feats_normalizer(raw, train_ids)
    return pos_normalizer, boundary_pos_normalizer, y_normalizer, feats_normalizer


def _resolve_ginot_normalizers(
    *,
    dataset_name: str,
    raw: GinotRawDataset,
    full_train_ids: list[int],
    split_seed: int,
    cache_tag: str,
    include_edges: bool,
) -> tuple[StandardNormalizer, StandardNormalizer, StandardNormalizer, StandardNormalizer | None]:
    train_cache_probe = graph_cache_dir(
        raw=raw,
        dataset_name=dataset_name,
        split_name="train",
        split_seed=split_seed,
        dataset_split=cache_tag,
    )
    if include_edges and is_sharded_lmdb_graph_cache(train_cache_probe):
        # LPBF caches store raw targets; y_normalizer is always rebuilt from lpbf_metadata_normalizers.
        if dataset_name == "lpbf":
            return _build_normalizers(dataset_name, raw, full_train_ids)
        return load_graph_cache_normalizers(train_cache_probe)
    return _build_normalizers(dataset_name, raw, full_train_ids)


def _mesh_split_note(
    dataset_name: str,
    num_samples: int,
    split_seed: int,
    train_count: int,
    test_count: int,
    raw: GinotRawDataset,
) -> str:
    if dataset_name == "micro_puc":
        return (
            f"micro_puc uses all {num_samples} rows; each row maps through sample_ids.npy to one of "
            f"{MICRO_PUC_MESH_SAMPLES} canonical meshes in mesh_cells10K.pkl. Using an 80/20 split "
            f"with seed={split_seed}: {train_count} train samples and {test_count} test samples."
        )
    if dataset_name == "micro_puc_fixed":
        return (
            f"micro_puc_fixed contains {len(raw.query_points)} full-row samples with canonical x/y-periodic "
            f"mesh quotients; using an 80/20 split with seed={split_seed}: "
            f"{train_count} train samples and {test_count} test samples."
        )
    if dataset_name == "deform_plate":
        return (
            f"deform_plate uses official train/valid split ({DeformPlateSplitSpec().split_label}): "
            f"{train_count} train samples and {test_count} validation samples from valid.tfrecord."
        )
    if dataset_name == "bumper_beam":
        return (
            f"bumper_beam combines all {num_samples} curated runs and uses an 80/20 split "
            f"with seed={split_seed}: {train_count} train samples and {test_count} test samples."
        )
    return (
        f"Using GINOT split with seed={split_seed}: {train_count} train samples and {test_count} test samples."
    )


@dataclass(frozen=True)
class GinotPrecomputeSplit:
    indices: list[int]
    graph_cache_dir: str


def _resolve_graph_cache_dir(
    *,
    raw: GinotRawDataset,
    dataset_name: str,
    split_name: str,
    indices: list[int],
    split_seed: int,
    dataset_split: str,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    feats_normalizer: StandardNormalizer | None = None,
) -> str:
    return ensure_graph_cache(
        raw=raw,
        dataset_name=dataset_name,
        split_name=split_name,
        indices=indices,
        split_seed=split_seed,
        dataset_split=dataset_split,
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
    )


def load_ginot_precompute_context(
    dataset_name: str,
    data_root: str,
    *,
    split_seed: int = 0,
    max_samples: int = 0,
) -> tuple[GinotPrecomputeSplit, GinotPrecomputeSplit, GinotRawDataset]:
    """Lightweight train/test handles for static-cache precompute (no GinotDataset length scan)."""
    from pdebench.dataset.ginot import _load_raw_dataset

    dataset_name = dataset_name.lower()
    _validate_max_samples(dataset_name, max_samples)
    raw = _load_raw_dataset(dataset_name, data_root)
    num_samples = num_indexable_samples(raw)

    policy = _SPLIT_POLICIES.get(dataset_name)
    if policy is not None and policy.validate is not None:
        policy.validate(raw, num_samples)

    train_ids, test_ids = _split_train_test(dataset_name, num_samples, split_seed, data_root=data_root)
    full_train_ids = train_ids
    train_ids, test_ids = _cap_split_indices(train_ids, test_ids, max_samples=max_samples)
    cache_tag = policy.cache_tag if policy is not None else "default"
    pos_normalizer, boundary_pos_normalizer, y_normalizer, feats_normalizer = _resolve_ginot_normalizers(
        dataset_name=dataset_name,
        raw=raw,
        full_train_ids=full_train_ids,
        split_seed=split_seed,
        cache_tag=cache_tag,
        include_edges=True,
    )

    train_graph_cache_dir = _resolve_graph_cache_dir(
        raw=raw,
        dataset_name=dataset_name,
        split_name="train",
        indices=train_ids,
        split_seed=split_seed,
        dataset_split=cache_tag,
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
    )
    if test_ids:
        test_graph_cache_dir = _resolve_graph_cache_dir(
            raw=raw,
            dataset_name=dataset_name,
            split_name="test",
            indices=test_ids,
            split_seed=split_seed,
            dataset_split=cache_tag,
            pos_normalizer=pos_normalizer,
            boundary_pos_normalizer=boundary_pos_normalizer,
            y_normalizer=y_normalizer,
            feats_normalizer=feats_normalizer,
        )
    else:
        test_graph_cache_dir = ""

    return (
        GinotPrecomputeSplit(indices=train_ids, graph_cache_dir=train_graph_cache_dir),
        GinotPrecomputeSplit(indices=test_ids, graph_cache_dir=test_graph_cache_dir),
        raw,
    )


def load_ginot_dataset(
    dataset_name: str,
    data_root: str,
    split_seed: int = 0,
    max_samples: int = 0,
    include_edges: bool = False,
    use_flash_varlen: bool = False,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    include_padded_boundary: bool = False,
    *,
    bucketed_batches: bool | None = None,
    build_collate_metadata: bool = True,
):
    dataset_name = dataset_name.lower()
    _validate_max_samples(dataset_name, max_samples)
    from pdebench.dataset.ginot import _load_raw_dataset

    raw = _load_raw_dataset(dataset_name, data_root)
    num_samples = num_indexable_samples(raw) if include_edges else len(raw.query_points)

    policy = _SPLIT_POLICIES.get(dataset_name)
    if policy is not None and policy.validate is not None:
        policy.validate(raw, num_samples)

    train_ids, test_ids = _split_train_test(dataset_name, num_samples, split_seed, data_root=data_root)
    full_train_ids = train_ids
    full_test_ids = test_ids
    train_ids, test_ids = _cap_split_indices(train_ids, test_ids, max_samples=max_samples)
    cache_tag = policy.cache_tag if policy is not None else "default"
    pos_normalizer, boundary_pos_normalizer, y_normalizer, feats_normalizer = _resolve_ginot_normalizers(
        dataset_name=dataset_name,
        raw=raw,
        full_train_ids=full_train_ids,
        split_seed=split_seed,
        cache_tag=cache_tag,
        include_edges=include_edges,
    )
    test_count = len(test_ids)
    train_bucketed = include_edges if bucketed_batches is None else bool(bucketed_batches)
    if int(max_samples) > 0:
        train_bucketed = False

    # Raw (no-edge) training reads VTPs every step; warm the bumper decode cache
    # here. Edge training uses LMDB once built — preload only during cache build
    # (see ensure_graph_cache), not on every warm-cache startup.
    if not include_edges:
        preload_bumper_beam_store(raw)

    if include_edges:
        train_cache_dir = _resolve_graph_cache_dir(
            raw=raw,
            dataset_name=dataset_name,
            split_name="train",
            indices=train_ids,
            split_seed=split_seed,
            dataset_split=cache_tag,
            pos_normalizer=pos_normalizer,
            boundary_pos_normalizer=boundary_pos_normalizer,
            y_normalizer=y_normalizer,
            feats_normalizer=feats_normalizer,
        )
        train_split = open_graph_cache_split(train_cache_dir, train_ids)
        train_mean_nodes = float(sum(train_split.node_lengths) / len(train_split.node_lengths))
        train_data = GraphCacheDataset(
            train_split,
            dataset_name=dataset_name,
            raw=raw,
            feats_normalizer=feats_normalizer,
            y_normalizer=y_normalizer,
            laplacian_eig_dim=laplacian_eig_dim,
            laplacian_spec=laplacian_spec,
            bucketed_batches=train_bucketed,
        )
        if test_ids:
            test_cache_dir = _resolve_graph_cache_dir(
                raw=raw,
                dataset_name=dataset_name,
                split_name="test",
                indices=test_ids,
                split_seed=split_seed,
                dataset_split=cache_tag,
                pos_normalizer=pos_normalizer,
                boundary_pos_normalizer=boundary_pos_normalizer,
                y_normalizer=y_normalizer,
                feats_normalizer=feats_normalizer,
            )
            test_split = open_graph_cache_split(test_cache_dir, test_ids)
            test_data = GraphCacheDataset(
                test_split,
                dataset_name=dataset_name,
                raw=raw,
                feats_normalizer=feats_normalizer,
                y_normalizer=y_normalizer,
                laplacian_eig_dim=laplacian_eig_dim,
                laplacian_spec=laplacian_spec,
                bucketed_batches=False,
            )
        else:
            test_data = None
        first_sample = load_graph_cache_sample(train_cache_dir, train_ids[0])
        first_pos = first_sample["pos"].float()
        first_y = first_sample["y"].float()
        pad_to_nodes = train_split.max_node_length
        pad_to_boundary_nodes = train_split.max_boundary_length
        max_length = pad_to_nodes
        train_graph_cache_dir = train_cache_dir
    else:
        dataset_kwargs = dict(
            dataset_name=dataset_name,
            raw=raw,
            pos_normalizer=pos_normalizer,
            boundary_pos_normalizer=boundary_pos_normalizer,
            y_normalizer=y_normalizer,
            feats_normalizer=feats_normalizer,
            include_edges=False,
        )
        train_data = GinotDataset(**dataset_kwargs, indices=train_ids, bucketed_batches=train_bucketed)
        test_data = (
            GinotDataset(**dataset_kwargs, indices=test_ids, bucketed_batches=False)
            if test_ids
            else None
        )
        first_pos = torch.from_numpy(as_float_array(raw.query_points[train_ids[0]], dims=raw.space_dim)).float()
        first_y = torch.from_numpy(as_float_array(raw.targets[train_ids[0]])).float()
        pad_to_nodes = int(first_pos.shape[0])
        first_boundary_pos = torch.from_numpy(
            as_float_array(raw.point_clouds[train_ids[0]], dims=raw.space_dim)
        )
        pad_to_boundary_nodes = int(first_boundary_pos.shape[0])
        max_length = pad_to_nodes
        train_graph_cache_dir = None
        train_mean_nodes = None

    if build_collate_metadata:
        train_collate_fn = partial(
            ginot_collate_fn,
            pad_to_nodes=pad_to_nodes,
            pad_to_boundary_nodes=pad_to_boundary_nodes,
            use_flash_varlen=use_flash_varlen,
            include_padded_boundary=include_padded_boundary,
        )
    else:
        train_collate_fn = None

    pos_dim = int(first_pos.shape[-1])
    feats_dim = int(active_feats_dim(raw))
    metadata = dict(
        dataset=dataset_name,
        x_normalizer=pos_normalizer,
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
        pos_dim=pos_dim,
        feats_dim=feats_dim,
        c_in=pos_dim + feats_dim,
        c_edge=pos_dim + 1,
        c_out=first_y.shape[-1],
        space_dim=pos_dim,
        fun_dim=feats_dim,
        boundary_dim=pos_dim,
        time_cond=False,
        max_length=max_length,
        target_fields=list(raw.target_fields),
        mesh_split_note=_mesh_split_note(
            dataset_name, num_samples, split_seed, len(train_data), test_count, raw
        ),
        ginot_micro_puc_mesh_samples=(
            MICRO_PUC_MESH_SAMPLES
            if dataset_name == "micro_puc"
            else len(raw.query_points) if dataset_name == "micro_puc_fixed" else None
        ),
        ginot_micro_puc_num_rows=len(raw.query_points) if dataset_name == "micro_puc" else None,
        ginot_micro_puc_split=policy.split_label if dataset_name in _SPLIT_POLICIES else None,
        mesh_split_sizes=dict(train=len(train_data), val=test_count, test=test_count),
        mesh_test_data=test_data,
        ginot_include_edges=include_edges,
        ginot_graph_cache_dir=train_graph_cache_dir,
        ginot_graph_cache_root=graph_cache_root(raw) if include_edges else None,
        ginot_train_mean_nodes_per_run=train_mean_nodes,
        ginot_train_mean_nodes_per_batch_bs1=train_mean_nodes,
        ginot_use_flash_varlen=use_flash_varlen,
        ginot_bucketed_batches=train_bucketed,
        ginot_max_samples=int(max_samples),
        ginot_split_train_size=len(full_train_ids),
        ginot_split_test_size=len(full_test_ids),
        train_collate_fn=train_collate_fn,
        eval_collate_fn=train_collate_fn,
        ginot=True,
    )
    return train_data, test_data, metadata
