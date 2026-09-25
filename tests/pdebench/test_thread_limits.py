from __future__ import annotations

import os

import torch


def test_set_thread_limits_sets_env_and_torch_threads(monkeypatch) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    from pdebench.dataset.thread_limits import set_compute_thread_limits

    set_compute_thread_limits()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert torch.get_num_threads() == 1
