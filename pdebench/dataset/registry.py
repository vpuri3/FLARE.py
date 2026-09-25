"""Dataset name registry: aliases, deleted names, and canonical identifiers."""

from __future__ import annotations


class RegistryError(ValueError):
    """Raised when a dataset name is unknown, deleted, or otherwise invalid."""


DELETED: frozenset[str] = frozenset(
    {
        "jeb",
        "plaid_2d_profile",
        "plaid_vki_ls59",
        "plaid_rotor37",
        "airfoil_dynamic",
        "cylinder_flow",
    }
)

ALIASES: dict[str, str] = {
    "tensile2d": "plaid_tensile2d",
    "hyperelasticity": "plaid_hyperelasticity",
}

CANONICAL: frozenset[str] = frozenset(
    {
        "poisson_unstructured",
        "poisson_structured",
        "bracket_lug",
        "micro_puc",
        "micro_puc_fixed",
        "deform_plate",
        "bumper_beam",
        "lpbf",
        "plaid_tensile2d",
        "plaid_hyperelasticity",
        "plaid_el_pl_dynamics",
        "plaid_elpl_terminal",
    }
)


def resolve_dataset_name(name: str) -> str:
    """Map a user-facing dataset name to its canonical registry key.

    Raises:
        RegistryError: if ``name`` was removed from the supported set.
    """
    if name in DELETED:
        raise RegistryError(f"{name} was removed; use a supported dataset name")
    return ALIASES.get(name, name)
