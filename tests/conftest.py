from __future__ import annotations

import os
import sys

# Ensure repo-local packages (pdebench, lra) import in all pytest entry modes.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
