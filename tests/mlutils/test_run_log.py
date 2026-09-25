from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys

from mlutils.run_log import _Tee


def _make_spawn_lock_with_tee_stderr(log_path: str) -> None:
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            mp.get_context("spawn").Lock()
        finally:
            sys.stderr = original_stderr


def test_run_log_tee_exposes_fileno_for_spawn_resource_tracker(tmp_path) -> None:
    code = (
        "from tests.mlutils.test_run_log import _make_spawn_lock_with_tee_stderr; "
        f"_make_spawn_lock_with_tee_stderr({str(tmp_path / 'run.log')!r})"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

