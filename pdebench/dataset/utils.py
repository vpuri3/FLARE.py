import os

import mlutils

# The load_* names below are not called directly in this module: adapters.py
# resolves them lazily off `pdebench.dataset.utils` (via `_utils()`/`getattr`)
# so that tests can `monkeypatch.setattr(dataset_utils, "load_...", fake)`.
# Keep these imports as module attributes even though ruff sees them as unused.
from .ahmedml import load_ahmedml_surface_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .drivaerml import load_drivaerml_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .drivaerml_surface import load_drivaerml_surface_dataset  # noqa: F401
from .fno import load_darcy_dataset, load_navier_stokes_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .geo_fno import (  # noqa: F401 — re-export for adapter monkeypatching
    load_airfoil_steady_dataset,
    load_elasticity_dataset,
    load_pipe_dataset,
    load_plasticity_dataset,
)
from .ginot import GINOT_DATASETS, load_ginot_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .lpbf import load_lpbf_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .nasa_crm import load_nasa_crm_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .plaid_datasets import PLAID_DATASETS, load_mesh_static_dataset  # noqa: F401 — re-export for adapter monkeypatching
from .registry import resolve_dataset_name
from .sample import FeatureRequest
from .shapenet_car import load_shapenet_car_dataset  # noqa: F401 — re-export for adapter monkeypatching

DISTRIBUTED = mlutils.is_torchrun()
GLOBAL_RANK = int(os.environ["RANK"]) if DISTRIBUTED else 0


def uses_ginot_pipeline(dataset_name: str, model_type: str | None = None) -> bool:
    del model_type
    name = resolve_dataset_name(dataset_name.lower())
    if name in PLAID_DATASETS:
        return False
    return name in GINOT_DATASETS


def resolve_use_flash_varlen(
    *,
    model_type: str,
    dataset_name: str,
    mixed_precision: bool,
    use_context_parallel: bool = False,
) -> bool:
    """Whether training should use packed flash-attn varlen sequences.

    Enable only when the active pipeline actually emits packed varlen batches:

    - GLT always packs via ``cu_seqlens`` / ``ptr``
    - FLARE / GITO under mixed precision on GINOT or LPBF varlen paths

    Dense sequence datasets (e.g. ``nasa_crm``, ``ahmedml_surface``, ``shapenet_car``)
    keep padded ``[B, N, C]`` tensors even with mixed precision, so flash-varlen
    stays off. Context parallel shards the sequence dim of those dense tensors and
    is incompatible with packed flash-attn varlen.
    """
    from .lpbf import LPBF_DATASETS

    if use_context_parallel:
        if model_type == "glt":
            raise RuntimeError(
                "use_context_parallel is incompatible with GLT "
                "(GLT requires packed flash-attn varlen)."
            )
        return False
    if model_type == "glt":
        return True
    if not bool(mixed_precision):
        return False
    if model_type not in {"flare", "gito"}:
        return False
    if uses_ginot_pipeline(dataset_name, model_type):
        return True
    name = resolve_dataset_name(dataset_name.lower())
    return name in LPBF_DATASETS


def compile_stats_model_for_dataset(dataset_name: str, model_type: str | None, compile_model: bool) -> bool:
    """Whether full-batch stats should run on the compiled model (not the eager fallback)."""
    if not bool(compile_model):
        return False
    name = resolve_dataset_name(dataset_name.lower())
    # Full-mesh Rel-L2 (~8.6M pts) is a different shape than amortized train (100k);
    # keep stats eager to avoid inductor recompile / addr2line stalls per mesh.
    if name == "drivaerml_surface":
        return False
    if name == "lpbf" and model_type == "glt":
        return False
    return not uses_ginot_pipeline(dataset_name, model_type)


# ======================================================================#
def load_dataset(
    dataset_name: str,
    DATADIR_BASE: str,
    PROJDIR: str,
    force_reload: bool = False,
    mesh: bool = False,
    cells: bool = False,
    max_steps: int = None,
    init_step: int = None,
    init_case: int = None,
    exclude: bool = True,
    train_rollout_noise: float = 0.0,
    mesh_split_seed: int = 0,
    ginot_max_samples: int = 0,
    plaid_max_samples: int = 0,
    mesh_graph_backend: str = "pyg",
    ginot_use_flash_varlen: bool = False,
    plaid_use_sdf_features: bool = True,
    plaid_load_public_test: bool = False,
    plaid_terminal_y_norm: str = "asinh_iqr",
    plaid_terminal_target_fields: list[str] | None = ["U_x"],
    model_type: str | None = None,
    feature_request: FeatureRequest | None = None,
    subset_size: int = 100_000,
    iid_samples: bool = True,
):
    """Load a dataset by name.

    Features (edges / boundary / Laplacian eigenmodes) come only from
    ``feature_request``; pass a ``FeatureRequest`` to request them. When
    ``feature_request`` is ``None``, defaults to ``FeatureRequest()`` (no
    edges, no boundary, no Laplacian eigenmodes).

    Returns:
        tuple: (train_data, test_data, metadata)
    """
    del train_rollout_noise

    dataset_name = resolve_dataset_name(dataset_name.lower())
    fr = feature_request if feature_request is not None else FeatureRequest()

    # Lazy import: keeps adapters.py free of a hard import-time dependency on
    # utils, and lets tests monkeypatch ``pdebench.dataset.adapters.get_adapter``.
    import pdebench.dataset.adapters as adapters

    adapter = adapters.get_adapter(dataset_name)
    return adapter.load(
        DATADIR_BASE,
        mesh_split_seed=mesh_split_seed,
        ginot_max_samples=ginot_max_samples,
        plaid_max_samples=plaid_max_samples,
        mesh_graph_backend=mesh_graph_backend,
        ginot_use_flash_varlen=ginot_use_flash_varlen,
        plaid_use_sdf_features=plaid_use_sdf_features,
        plaid_load_public_test=plaid_load_public_test,
        plaid_terminal_y_norm=plaid_terminal_y_norm,
        plaid_terminal_target_fields=plaid_terminal_target_fields,
        feature_request=fr,
        subset_size=subset_size,
        iid_samples=iid_samples,
    )
