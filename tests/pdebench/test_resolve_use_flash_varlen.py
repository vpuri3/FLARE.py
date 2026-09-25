"""Flash-varlen enablement: only packed varlen pipelines, never with context parallel."""

from __future__ import annotations

import pytest

from pdebench.dataset.utils import resolve_use_flash_varlen


@pytest.mark.parametrize(
    ("model_type", "dataset_name", "mixed_precision", "use_context_parallel", "expected"),
    [
        # Dense sequence datasets stay padded even under mixed precision.
        ("flare", "nasa_crm", True, False, False),
        ("flare", "ahmedml_surface", True, False, False),
        ("flare", "shapenet_car", True, False, False),
        ("flare", "elasticity", True, False, False),
        # Context parallel forces flash-varlen off for FLARE.
        ("flare", "nasa_crm", True, True, False),
        ("flare", "lpbf", True, True, False),
        ("flare", "bracket_lug", True, True, False),
        # GINOT / LPBF FLARE under mixed precision packs varlen.
        ("flare", "bracket_lug", True, False, True),
        ("flare", "lpbf", True, False, True),
        ("gito", "micro_puc_fixed", True, False, True),
        # Without mixed precision, FLARE/GITO stay padded.
        ("flare", "bracket_lug", False, False, False),
        ("flare", "lpbf", False, False, False),
        # GLT always packs (non-CP).
        ("glt", "bracket_lug", False, False, True),
        ("glt", "plaid_tensile2d", True, False, True),
        ("glt", "lpbf", False, False, True),
        # Other models never opt into this flag via the shared gate.
        ("flarepp", "nasa_crm", True, False, False),
        ("transolver", "elasticity", True, False, False),
    ],
)
def test_resolve_use_flash_varlen(
    model_type: str,
    dataset_name: str,
    mixed_precision: bool,
    use_context_parallel: bool,
    expected: bool,
) -> None:
    assert (
        resolve_use_flash_varlen(
            model_type=model_type,
            dataset_name=dataset_name,
            mixed_precision=mixed_precision,
            use_context_parallel=use_context_parallel,
        )
        is expected
    )


def test_resolve_use_flash_varlen_glt_rejects_context_parallel() -> None:
    with pytest.raises(RuntimeError, match="incompatible with GLT"):
        resolve_use_flash_varlen(
            model_type="glt",
            dataset_name="bracket_lug",
            mixed_precision=True,
            use_context_parallel=True,
        )
