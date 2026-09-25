"""Load a trained run the way ``pdebench/__main__.py`` does under ``run.evaluate``.

The model is rebuilt from the experiment's own ``config.yaml`` through
``make_model``, never from a vendored copy of the architecture. The vendored
copy in ``figs/pdebench_vis.ipynb`` is already stale relative to
``pdebench/models/flare.py``, which is exactly the failure this avoids.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

import pdebench
from pdebench.config import Config
from pdebench.dataset.registry import resolve_dataset_name
from pdebench.dataset.sample import FeatureRequest
from pdebench.distributed import ContextParallelState, build_context_parallel_state
from pdebench.models.model_factory import make_model

PROJDIR = Path(__file__).resolve().parents[2]
CASEDIR = PROJDIR / "out" / "pdebench"

# Full-mesh evaluation datasets: metadata carries the complete surfaces under
# these keys, while train/test datasets hold amortized subsamples.
_RUN_DATA_KEY = {
    "ahmedml_surface": "ahmedml_test_run_data",
    "drivaerml_surface": "drivaerml_test_run_data",
}


@dataclass
class Run:
    """One loaded checkpoint plus everything needed to evaluate it."""

    exp_name: str
    dataset: str
    model: torch.nn.Module
    metadata: dict
    test_data: torch.utils.data.Dataset
    cfg: Config
    ckpt_path: Path
    case_dir: Path
    cp_state: Optional[ContextParallelState] = None
    cp_sequence_dim: int = 1

    @property
    def mixed_precision(self) -> bool:
        return bool(self.cfg.training.mixed_precision)


def latest_checkpoint(case_dir: Path) -> Path:
    """Newest ``ckptNN/model.pt`` in an experiment directory."""
    candidates = []
    for entry in sorted(case_dir.iterdir()):
        match = re.fullmatch(r"ckpt(\d+)", entry.name)
        if match and (entry / "model.pt").is_file():
            candidates.append((int(match.group(1)), entry / "model.pt"))
    if not candidates:
        raise FileNotFoundError(f"no ckptNN/model.pt under {case_dir}")
    return max(candidates)[1]


def _strip_wrappers(state: dict) -> dict:
    """Undo ``torch.compile`` / DDP key prefixes."""
    for prefix in ("_orig_mod.", "module."):
        while any(key.startswith(prefix) for key in state):
            state = {
                (key[len(prefix) :] if key.startswith(prefix) else key): value
                for key, value in state.items()
            }
    return state


def _remap_plain_proj_to_fc(state: dict) -> dict:
    """Map ``*.k0_proj.weight`` → ``*.k0_proj.fc.weight`` for MLP-wrapped projs."""
    stems = ("k0_proj", "v0_proj", "k_proj", "v_proj", "q_proj", "q0_proj")
    remapped: dict = {}
    for key, value in state.items():
        new_key = key
        for stem in stems:
            if key.endswith(f"{stem}.weight"):
                new_key = key[: -len("weight")] + "fc.weight"
                break
            if key.endswith(f"{stem}.bias"):
                new_key = key[: -len("bias")] + "fc.bias"
                break
        remapped[new_key] = value
    return remapped


def _load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load a checkpoint, remapping plain Linear proj keys only if needed."""
    state = _strip_wrappers(state)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError as exc:
        msg = str(exc)
        if "Missing key" not in msg or ".fc.weight" not in msg:
            raise
    model.load_state_dict(_remap_plain_proj_to_fc(state))


def _unwrap_dataset(dataset):
    """Reach the concrete dataset under Subset / wrapper layers."""
    seen = set()
    while True:
        if id(dataset) in seen:
            return dataset
        seen.add(id(dataset))
        inner = getattr(dataset, "dataset", None)
        if inner is None or inner is dataset:
            return dataset
        dataset = inner


def _attach_vis_geometry(metadata: dict, dataset_name: str, *datasets) -> None:
    """Record the coordinate min-max the 3D loaders normalized with.

    ``pdebench.vis.extract`` needs it to put the geometry back into physical
    units; per-axis min-max otherwise renders a stretched body.
    """
    if dataset_name not in ("nasa_crm", "ahmedml_surface", "drivaerml_surface") and not dataset_name.startswith(
        "drivaerml_"
    ):
        return
    for candidate in datasets:
        if candidate is None:
            continue
        obj = _unwrap_dataset(candidate)
        lo, hi = getattr(obj, "xyz_min", None), getattr(obj, "xyz_max", None)
        if lo is not None and hi is not None:
            metadata["vis_xyz_min"] = lo
            metadata["vis_xyz_max"] = hi
            return
    raise AttributeError(
        f"{dataset_name}: no dataset object exposed xyz_min / xyz_max; cannot restore physical coordinates"
    )


def case_roots(extra: list[str] | None = None) -> list[Path]:
    """Directories searched for experiment dirs, most specific first.

    Runs do not necessarily live under this checkout: on a shared cluster the
    checkpoints behind a table may sit in a collaborator's tree. Roots come
    from ``--case-root``, then ``PDEBENCH_CASE_ROOT`` (colon-separated), then
    this repo's own ``out/pdebench``.
    """
    roots = [Path(p).expanduser() for p in (extra or [])]
    env = os.environ.get("PDEBENCH_CASE_ROOT", "")
    roots += [Path(p).expanduser() for p in env.split(os.pathsep) if p]
    roots.append(CASEDIR)

    seen, unique = set(), []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_case_dir(exp_name: str, roots: list[Path] | None = None) -> Path:
    """Find the experiment directory for ``exp_name``.

    ``exp_name`` may be an absolute or relative path to a case directory, or a
    bare name to look up under the search roots. Ambiguity is an error rather
    than a silent first-match: two roots holding the same experiment name is
    exactly the situation where the wrong checkpoint gets plotted.
    """
    candidate = Path(exp_name).expanduser()
    if candidate.is_dir() and (candidate / "config.yaml").is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        raise FileNotFoundError(f"{candidate}/config.yaml not found")

    search = roots if roots is not None else case_roots()
    hits = [root / exp_name for root in search if (root / exp_name / "config.yaml").is_file()]
    if not hits:
        listed = "\n  ".join(str(root) for root in search)
        raise FileNotFoundError(f"no {exp_name}/config.yaml under any case root:\n  {listed}")
    if len(hits) > 1:
        listed = "\n  ".join(str(hit) for hit in hits)
        raise FileNotFoundError(
            f"{exp_name!r} is ambiguous across case roots; pass the full path instead:\n  {listed}"
        )
    return hits[0].resolve()


def _nest_legacy_mixer_model(model: dict) -> dict:
    """Rewrite flat ``mixer_backbone`` YAML into nested ``model.mixer``.

    Older case dirs wrote mixer kwargs (``num_latents``, ``qk_norm``, …) as
    siblings of ``mixer: flare``. Current ``Config`` nests them under
    ``mixer: {kind, …}``. Without this rewrite, stage A cannot load those
    checkpoints at all.
    """
    from dataclasses import fields as dc_fields

    from pdebench.config import MIXER_BY_KIND, MixerBackboneConfig

    if not isinstance(model, dict) or model.get("model") != "mixer_backbone":
        return model
    mixer = model.get("mixer")
    if isinstance(mixer, dict):
        return model
    kind = mixer if isinstance(mixer, str) else "flare"
    if kind not in MIXER_BY_KIND:
        raise ValueError(f"unknown mixer kind {kind!r} in legacy config")
    mixer_cls, _ = MIXER_BY_KIND[kind]
    mixer_fields = {f.name for f in dc_fields(mixer_cls)} - {"kind"}
    backbone_fields = {f.name for f in dc_fields(MixerBackboneConfig)}
    nested = {"kind": kind}
    out: dict = {}
    for key, value in model.items():
        if key == "mixer":
            continue
        if key in mixer_fields:
            nested[key] = value
        elif key in backbone_fields:
            out[key] = value
        # Drop leftover flat ablation keys that do not belong on this mixer kind.
    out["mixer"] = nested
    return out


def load_run(
    exp_name: str,
    device: torch.device,
    *,
    data_root: str | None = None,
    roots: list[Path] | None = None,
    cp_size: int | None = None,
) -> Run:
    """Rebuild one experiment's model and test data from its case directory."""
    case_dir = resolve_case_dir(exp_name, roots)
    config_file = case_dir / "config.yaml"

    with open(config_file) as handle:
        raw = yaml.safe_load(handle)
    if isinstance(raw.get("model"), dict):
        raw["model"] = _nest_legacy_mixer_model(raw["model"])
    cfg = Config(**raw)

    dataset_name = resolve_dataset_name(cfg.dataset.dataset.lower())
    root = data_root or cfg.dataset.data_root or str(PROJDIR / "data")

    load_kwargs = dict(
        mesh=False,
        model_type=cfg.model.model,
        feature_request=FeatureRequest(edges=False, boundary=False),
    )
    if dataset_name in ("ahmedml_surface", "drivaerml_surface"):
        load_kwargs["subset_size"] = cfg.dataset.subset_size
    if dataset_name == "drivaerml_surface":
        load_kwargs["iid_samples"] = cfg.dataset.iid_samples

    train_data, test_data, metadata = pdebench.load_dataset(dataset_name, root, str(PROJDIR), **load_kwargs)
    metadata["dataset"] = dataset_name
    metadata["model"] = cfg.model.model

    # Full-mesh surfaces where the train/test datasets are amortized subsamples.
    run_key = _RUN_DATA_KEY.get(dataset_name)
    if run_key is not None:
        run_data = metadata.get(run_key)
        if run_data is None:
            raise KeyError(f"{dataset_name}: metadata has no {run_key} for full-mesh visualization")
        test_data = run_data

    _attach_vis_geometry(metadata, dataset_name, test_data, train_data)

    cfg, model = make_model(cfg, metadata, 0)
    ckpt_path = latest_checkpoint(case_dir)
    snapshot = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    _load_model_state(model, snapshot["model_state"])
    del snapshot

    cp_state = None
    cp_sequence_dim = int(cfg.training.cp_sequence_dim)
    if cfg.training.use_context_parallel:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                f"{exp_name} was trained with use_context_parallel=True "
                f"(cp_size={cfg.training.context_parallel_size}); stage A must be "
                "launched with torchrun so the sequence can be sharded the same way "
                "as full-mesh eval during training."
            )
        requested_cp = int(cfg.training.context_parallel_size if cp_size is None else cp_size)
        if requested_cp < 1:
            raise ValueError(f"context_parallel_size must be >= 1, got {requested_cp}")
        cp_state = build_context_parallel_state(requested_cp)
        if not hasattr(model, "set_context_parallel"):
            raise TypeError(f"{cfg.model.model} does not expose set_context_parallel")
        model.set_context_parallel(
            cp_state=cp_state,
            cp_debug_gather_outputs=bool(cfg.training.cp_debug_gather_outputs),
        )

    model.to(device).eval()

    return Run(
        exp_name=exp_name,
        dataset=dataset_name,
        model=model,
        metadata=metadata,
        test_data=test_data,
        cfg=cfg,
        ckpt_path=ckpt_path,
        case_dir=case_dir,
        cp_state=cp_state,
        cp_sequence_dim=cp_sequence_dim,
    )


def config_digest(case_dir: Path) -> str:
    """Content hash of an experiment's config, for the manifest."""
    import hashlib

    return hashlib.sha256((case_dir / "config.yaml").read_bytes()).hexdigest()[:16]


def git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJDIR), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def nasa_crm_connectivity(data_root: str) -> "tuple[object, str]":
    """Load NASA-CRM surface triangles, if the connectivity file is present.

    The layout of ``connectivity_NASA-CRM.h5`` is not documented in this repo,
    so this probes for a plausible ``[T, 3]`` integer array and reports what it
    found rather than guessing silently. Run ``dump.py --probe-connectivity``
    to print the file's tree if this returns nothing.
    """
    import h5py
    import numpy as np

    candidates = (
        os.path.join(data_root, "NASA_CRM", "connectivity_NASA-CRM.h5"),
        os.path.join(data_root, "NASA-CRM", "connectivity_NASA-CRM.h5"),
        os.path.join(data_root, "connectivity_NASA-CRM.h5"),
    )
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        return None, f"connectivity file not found under {data_root}"

    found: list[tuple[str, object]] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.shape[1] in (3, 4):
            if np.issubdtype(obj.dtype, np.integer):
                found.append((name, obj[()]))

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)

    if not found:
        # Real layout: Connectivity/<face_id> as 1D int vectors of mixed length
        # (mostly quads). Not a single [T,3]/[T,4] array — triangles optional.
        return None, (
            f"no [T,3]/[T,4] integer array in {path}; "
            "file uses per-face 1D Connectivity/<id> datasets (mixed 3–8-gons); "
            "dumping without triangles"
        )
    name, tri = found[0]
    return tri, f"{path}::{name} shape={tri.shape}"
