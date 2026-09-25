from __future__ import annotations


def test_plaid_laplacian_helpers_reexport() -> None:
    from pdebench.dataset import plaid_laplacian as pl
    from pdebench.dataset import plaid_datasets as pd

    assert pl._ensure_plaid_laplacian is not None
    assert pd._ensure_plaid_laplacian is pl._ensure_plaid_laplacian
