"""PLAID tensile/hyperelasticity target-scalar parsing."""

from __future__ import annotations

import numpy as np

from pdebench.dataset.plaid_core import target_scalar_names_for_dataset
from pdebench.dataset.plaid_datasets import PLAID_SPECS


def test_tensile_spec_includes_vi_transf_scalars() -> None:
    assert PLAID_SPECS["plaid_tensile2d"].benchmark_scalar_outputs == (
        "max_von_mises",
        "max_U2_top",
        "max_sig22_top",
    )


def test_target_scalar_names_for_tensile_and_hyperelasticity() -> None:
    assert target_scalar_names_for_dataset("plaid_tensile2d") == (
        "max_von_mises",
        "max_U2_top",
        "max_sig22_top",
    )
    assert target_scalar_names_for_dataset("plaid_hyperelasticity") == ("effective_energy",)
    assert target_scalar_names_for_dataset("plaid_el_pl_dynamics") == ()


def test_tensile_target_scalars_extracted_from_scalar_dict() -> None:
    """Mirror parse_plaid_sample_bytes scalar extraction without a full CGNS tree."""
    names = target_scalar_names_for_dataset("plaid_tensile2d")
    scalar_dict = {
        "P": 1.0,
        "p1": 0.1,
        "max_von_mises": 10.0,
        "max_U2_top": 0.2,
        "max_sig22_top": 3.0,
    }
    assert all(name in scalar_dict for name in names)
    target_scalars = np.array([scalar_dict[name] for name in names], dtype=np.float32)
    assert target_scalars.shape == (3,)
    np.testing.assert_allclose(target_scalars, [10.0, 0.2, 3.0])
