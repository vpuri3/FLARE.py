from __future__ import annotations

import pytest

from pdebench.dataset.registry import RegistryError, resolve_dataset_name


def test_alias_tensile() -> None:
    assert resolve_dataset_name("tensile2d") == "plaid_tensile2d"


def test_alias_hyperelasticity() -> None:
    assert resolve_dataset_name("hyperelasticity") == "plaid_hyperelasticity"


def test_deleted_names_raise() -> None:
    for name in [
        "jeb",
        "plaid_2d_profile",
        "plaid_vki_ls59",
        "plaid_rotor37",
        "airfoil_dynamic",
        "cylinder_flow",
    ]:
        with pytest.raises(RegistryError):
            resolve_dataset_name(name)


def test_poisson_structured_is_canonical() -> None:
    assert resolve_dataset_name("poisson_structured") == "poisson_structured"


def test_canonical_lpbf() -> None:
    assert resolve_dataset_name("lpbf") == "lpbf"
