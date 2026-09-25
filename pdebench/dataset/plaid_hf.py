"""PLAID Hugging Face Space URL resolution for hidden-test submit."""

from __future__ import annotations

import os

# Verified against https://huggingface.co/PLAIDcompetitions (Space subdomain = id lowercased).
PLAID_HF_SPACE_URLS: dict[str, str] = {
    "plaid_tensile2d": "https://plaidcompetitions-tensile2dbenchmark.hf.space",
    "plaid_hyperelasticity": "https://plaidcompetitions-2dmultiscalehyperelasticitybenchmark.hf.space",
    "plaid_el_pl_dynamics": "https://plaidcompetitions-2delastoplastodynamics.hf.space",
    "plaid_elpl_terminal": "https://plaidcompetitions-2delastoplastodynamics.hf.space",
}

_DEFAULT_HF_URL = PLAID_HF_SPACE_URLS["plaid_hyperelasticity"]


def resolve_plaid_hf_benchmark_url(dataset_name: str | None = None) -> str:
    """Return HF Space base URL for PLAID submit.

    Precedence: ``PLAID_HF_BENCHMARK_URL`` env override, then per-dataset map,
    then hyperelasticity default.
    """
    override = os.environ.get("PLAID_HF_BENCHMARK_URL")
    if override:
        return override.rstrip("/")
    if dataset_name:
        key = str(dataset_name).strip().lower()
        if key in PLAID_HF_SPACE_URLS:
            return PLAID_HF_SPACE_URLS[key]
    return _DEFAULT_HF_URL
