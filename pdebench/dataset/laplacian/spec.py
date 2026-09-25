"""Laplacian operator names, spec parsing, and cache-path slugs."""

from __future__ import annotations

LAPLACIAN_OPERATORS = {
    "graph",
    "edge",
    "fem_u",
    "fem_v",
}
DEFAULT_LAPLACIAN_EIGENVECTORS = 64
DEFAULT_LAPLACIAN_SPECS = "graph:64"

# Per-dataset spectral precompute specs. FEM is omitted for micro_puc_fixed (2-node periodic
# edges are not valid 2D FEM elements; precompute would fail or waste GPU time).
DATASET_LAPLACIAN_SPECS: dict[str, str] = {
    "bumper_beam": "graph:32",
    "micro_puc_fixed": "graph:64",
    "micro_puc": "graph:64",
    "poisson_unstructured": "graph:64",
    "poisson_structured": "graph:64",
    "bracket_lug": "graph:64",
    "deform_plate": "graph:64",
}


def default_laplacian_specs_for_dataset(dataset_name: str) -> str:
    return DATASET_LAPLACIAN_SPECS.get(str(dataset_name).lower(), DEFAULT_LAPLACIAN_SPECS)


def parse_laplacian_spec(spec: str | None, default_dim: int) -> list[tuple[str, int]]:
    spec = "graph" if spec is None or str(spec).strip() == "" else str(spec).strip()
    parts = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, count = part.split(":", 1)
            count_int = int(count)
        else:
            name = part
            count_int = int(default_dim)
        name = name.strip().lower().replace("-", "_")
        if name not in LAPLACIAN_OPERATORS:
            raise ValueError(f"Unsupported Laplacian cache operator '{name}'.")
        if count_int < 0:
            raise ValueError(f"Negative Laplacian feature count in '{part}'.")
        if count_int > 0:
            parts.append((name, count_int))
    return parts


def _cache_operator_name(name: str) -> str:
    return "fem" if str(name).startswith("fem_") else str(name)


def _cache_spec_for_part(name: str, count: int) -> str:
    cache_name = _cache_operator_name(name)
    if cache_name == "fem":
        return f"fem_v:{int(count)}"
    return f"{cache_name}:{int(count)}"


def laplacian_spec_dim(spec: str | None, default_dim: int) -> int:
    return sum(count for _, count in parse_laplacian_spec(spec, default_dim))


def laplacian_spec_slug(spec: str | None, default_dim: int) -> str:
    parts = parse_laplacian_spec(spec, default_dim)
    if not parts:
        return "none0"
    return "_".join(f"{_cache_operator_name(name)}{count}" for name, count in parts)


def laplacian_spec_entries(spec: str | None, default_dim: int) -> list[str]:
    return [f"{name}:{count}" for name, count in parse_laplacian_spec(spec, default_dim)]


def resolve_laplacian_specs_for_dataset(
    dataset_name: str,
    specs_arg: str | None,
    laplacian_dim: int,
) -> list[str]:
    """Resolve CLI --laplacian-specs or fall back to per-dataset defaults."""
    raw = default_laplacian_specs_for_dataset(dataset_name) if specs_arg is None else str(specs_arg)
    return laplacian_spec_entries(raw, laplacian_dim)


def laplacian_cache_spec_entry(spec: str | None, default_dim: int) -> str:
    name, count = single_laplacian_spec_part(spec, default_dim)
    return _cache_spec_for_part(name, count)


def single_laplacian_spec_part(spec: str | None, default_dim: int) -> tuple[str, int]:
    parts = parse_laplacian_spec(spec, default_dim)
    if len(parts) != 1:
        raise ValueError(
            f"Expected exactly one Laplacian operator per spec entry, got {parts!r} from {spec!r}."
        )
    return parts[0]
