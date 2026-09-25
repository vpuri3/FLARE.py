"""Shim: Laplacian precompute lives in ``pdebench.dataset.laplacian.precompute``."""

from __future__ import annotations

import warnings

warnings.warn(
    "pdebench.dataset.ginot.precompute is deprecated; use "
    "python -m pdebench.dataset.laplacian.precompute",
    DeprecationWarning,
    stacklevel=2,
)

from pdebench.dataset.laplacian.precompute import *  # noqa: E402, F403
from pdebench.dataset.laplacian.precompute import _parse_args, main  # noqa: E402, F401

if __name__ == "__main__":
    main()
