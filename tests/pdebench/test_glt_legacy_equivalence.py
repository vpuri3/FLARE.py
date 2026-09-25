"""Construction / rejection checks for glt.GLT vs legacy topology.GLT config grid.

Loads ``topology.py`` from base commit ``ff4800f2`` and checks that invalid
(mode, K, SPE) combos still reject on both stacks. Valid combos assert the new
GLT builds and runs a finite forward. Weight-level mathematical equivalence was
dropped when GLT switched ResidualMLP projections to ``nn.Linear``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pdebench.models.graph_models.glt import GLT as NewGLT
from pdebench.models.graph_models.glt import GLTConfig as NewGLTConfig
from pdebench.models.graph_models.glt import NonePEConfig, RawEigenPEConfig, SpectralFilterPEConfig

_BASE_COMMIT = "ff4800f2"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_MODULE_NAME = "pdebench.models.graph_models._topology_legacy_ff4800f2"


def _load_legacy_topology():
    if _LEGACY_MODULE_NAME in sys.modules:
        return sys.modules[_LEGACY_MODULE_NAME]
    code = subprocess.check_output(
        ["git", "show", f"{_BASE_COMMIT}:pdebench/models/graph_models/topology.py"],
        cwd=_REPO_ROOT,
    )
    path = Path("/tmp") / f"{_LEGACY_MODULE_NAME.split('.')[-1]}.py"
    path.write_bytes(code)
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "pdebench.models.graph_models"
    sys.modules[_LEGACY_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _dummy_batch(*, n0: int = 5, n1: int = 4, pos_dim: int = 3, k: int):
    """Two packed graphs with a line edge graph and optional Laplacian features."""
    n_tot = n0 + n1
    pos = torch.randn(n_tot, pos_dim)
    # Line edges within each graph.
    e0 = torch.stack(
        [torch.arange(n0 - 1), torch.arange(1, n0)],
        dim=0,
    )
    e1 = torch.stack(
        [torch.arange(n1 - 1) + n0, torch.arange(1, n1) + n0],
        dim=0,
    )
    edge_index = torch.cat([e0, e1], dim=1)
    cu_seqlens = torch.tensor([0, n0, n_tot], dtype=torch.int32)
    max_seqlen = max(n0, n1)
    if k <= 0:
        topology_features = None
        topology_eigenvalues = None
    else:
        # Packed [N_tot, K] eigenvectors (need not be exact Laplacian eigenpairs).
        topology_features = torch.randn(n_tot, k)
        topology_features = topology_features / topology_features.norm(dim=0, keepdim=True).clamp_min(1e-6)
        topology_eigenvalues = torch.stack(
            [
                torch.linspace(0.1, 1.0, k),
                torch.linspace(0.2, 1.2, k),
            ],
            dim=0,
        )
    return {
        "pos": pos,
        "edge_index": edge_index,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "num_total_nodes": n_tot,
        "topology_features": topology_features,
        "topology_eigenvalues": topology_eigenvalues,
        "use_flash_varlen": True,
    }


def _new_config(*, mode: int, k: int, spe: bool) -> NewGLTConfig:
    pe_inject_mode = "concat_input" if mode == 0 else "concat_qk"
    if spe:
        pe = SpectralFilterPEConfig(num_eigenmodes=k, filter_type="band", mode="query")
    elif k <= 0:
        pe = NonePEConfig()
    else:
        pe = RawEigenPEConfig(num_eigenmodes=k)
    return NewGLTConfig(
        pe_inject_mode=pe_inject_mode,
        pe_update=False,
        pe=pe,
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        attn_type="linear",
        mlp_ratio=2.0,
    )


def _legacy_config(legacy, *, mode: int, k: int, spe: bool):
    return legacy.GLTConfig(
        glt_mode=mode,
        glt_update_c=False,
        pe_spe=spe,
        topology_num_eigenmodes=k,
        topology_laplacian_spec="graph",
        pe_spe_filter_type="band",
        pe_spe_mode="query",
        channel_dim=32,
        num_blocks=2,
        num_heads=4,
        attn_type="linear",
        mlp_ratio=2.0,
        num_layers_in_out_proj=1,
    )


def _combo_is_valid(*, mode: int, k: int, spe: bool) -> bool:
    if k <= 0 and mode != 0:
        return False
    if k <= 0 and spe:
        return False
    return True


def _forward(model, batch: dict) -> torch.Tensor:
    return model(**batch)


@pytest.fixture(scope="module")
def legacy_topology():
    return _load_legacy_topology()


@pytest.mark.parametrize("mode", [0, 2])
@pytest.mark.parametrize("k", [0, 32])
@pytest.mark.parametrize("spe", [False, True])
def test_glt_legacy_vs_new_config_grid(legacy_topology, mode: int, k: int, spe: bool):
    meta = dict(c_in=3, c_out=1, pos_dim=3)
    valid = _combo_is_valid(mode=mode, k=k, spe=spe)

    if not valid:
        with pytest.raises(ValueError):
            legacy_topology.GLT(_legacy_config(legacy_topology, mode=mode, k=k, spe=spe), metadata=meta)
        with pytest.raises(ValueError):
            NewGLT(_new_config(mode=mode, k=k, spe=spe), metadata=meta)
        return

    new = NewGLT(_new_config(mode=mode, k=k, spe=spe), metadata=meta)
    torch.manual_seed(123)
    batch = _dummy_batch(k=k)
    new.eval()
    with torch.no_grad():
        y_new = _forward(new, batch)

    assert y_new.shape == (batch["num_total_nodes"], 1)
    assert torch.isfinite(y_new).all()
