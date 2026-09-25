"""Unified PDEBench dataset visualization CLI.

Resolves canonical/alias dataset names via the registry, then delegates to the
existing GINOT/PLAID visualization helpers.

Usage:
  python -m pdebench.dataset.visualize --dataset plaid_tensile2d --mode raw
  python -m pdebench.dataset.visualize --dataset poisson_unstructured --spectral
"""

from __future__ import annotations

import sys
from collections.abc import Collection
from pathlib import Path

from pdebench.dataset.registry import RegistryError, resolve_dataset_name


def _rewrite_argv_datasets(available: Collection[str] | None = None) -> None:
    """Resolve --dataset values in-place before ginot.visualize parses args."""
    if "--dataset" not in sys.argv:
        return
    idx = sys.argv.index("--dataset")
    end = idx + 1
    while end < len(sys.argv) and not sys.argv[end].startswith("-"):
        end += 1
    raw = sys.argv[idx + 1 : end]
    resolved: list[str] = []
    for name in raw:
        try:
            resolved_name = resolve_dataset_name(name)
        except RegistryError as exc:
            raise SystemExit(str(exc)) from exc
        if available is not None and resolved_name not in available:
            supported = ", ".join(sorted(available))
            raise SystemExit(
                f"Unsupported visualization dataset {resolved_name!r}. "
                f"Supported visualization datasets: {supported}"
            )
        resolved.append(resolved_name)
    sys.argv = sys.argv[: idx + 1] + resolved + sys.argv[end:]


def _ensure_default_outdir() -> None:
    if "--outdir" in sys.argv:
        return
    # Insert before other flags so argparse sees the default unified tree.
    sys.argv.extend(["--outdir", str(Path("out/pdebench/dataset_viz"))])


def main() -> None:
    # Lazy import: visualize pulls matplotlib / heavy deps.
    from pdebench.dataset import plaid_visualize
    from pdebench.dataset.ginot import visualize as ginot_visualize

    available = set(ginot_visualize.DATASETS) | set(plaid_visualize.DATASETS)
    _rewrite_argv_datasets(available)
    _ensure_default_outdir()

    if "--dataset" not in sys.argv:
        raise SystemExit("Missing --dataset")
    idx = sys.argv.index("--dataset")
    end = idx + 1
    while end < len(sys.argv) and not sys.argv[end].startswith("-"):
        end += 1
    names = sys.argv[idx + 1 : end]
    if not names:
        raise SystemExit("Missing --dataset")
    if all(n in plaid_visualize.DATASETS for n in names):
        plaid_visualize.main()
    elif all(n in set(ginot_visualize.DATASETS) for n in names):
        ginot_visualize.main()
    else:
        raise SystemExit(
            f"Visualization datasets must be all-PLAID or all-GINOT in one invocation; got {names}"
        )


if __name__ == "__main__":
    main()
