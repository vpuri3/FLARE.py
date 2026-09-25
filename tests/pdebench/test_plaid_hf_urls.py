"""PLAID HF Space URL resolution (no network)."""

from __future__ import annotations

import os

from pdebench.dataset.plaid_hf import PLAID_HF_SPACE_URLS, resolve_plaid_hf_benchmark_url


def test_plaid_hf_url_map_covers_in_repo_datasets() -> None:
    expected = {
        "plaid_tensile2d": "https://plaidcompetitions-tensile2dbenchmark.hf.space",
        "plaid_hyperelasticity": "https://plaidcompetitions-2dmultiscalehyperelasticitybenchmark.hf.space",
        "plaid_el_pl_dynamics": "https://plaidcompetitions-2delastoplastodynamics.hf.space",
        "plaid_elpl_terminal": "https://plaidcompetitions-2delastoplastodynamics.hf.space",
    }
    assert PLAID_HF_SPACE_URLS == expected
    for name, url in expected.items():
        assert resolve_plaid_hf_benchmark_url(name) == url


def test_plaid_hf_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PLAID_HF_BENCHMARK_URL", "https://example.hf.space/")
    assert resolve_plaid_hf_benchmark_url("plaid_tensile2d") == "https://example.hf.space"
    monkeypatch.delenv("PLAID_HF_BENCHMARK_URL", raising=False)
    assert resolve_plaid_hf_benchmark_url("plaid_tensile2d") == PLAID_HF_SPACE_URLS["plaid_tensile2d"]
    # Unknown dataset falls back to hyperelasticity default when no override.
    assert resolve_plaid_hf_benchmark_url("unknown") == PLAID_HF_SPACE_URLS["plaid_hyperelasticity"]
    assert "PLAID_HF_BENCHMARK_URL" not in os.environ
