"""Distributed Flash CP checks — use the standalone torchrun harness.

Pytest+NCCL under this repo's NFS/agent cpuset is flaky. Canonical command:

    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTHONPATH=/project/community/vedantpu/FLARE-dev.py
    torchrun --standalone --nproc_per_node=1 tests/pdebench/run_flare_encoder_cp_flash_dist.py
    torchrun --standalone --nproc_per_node=2 tests/pdebench/run_flare_encoder_cp_flash_dist.py
    torchrun --standalone --nproc_per_node=4 tests/pdebench/run_flare_encoder_cp_flash_dist.py

This module stays collectable but skips outside that harness so plain `pytest` is quiet.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Use tests/pdebench/run_flare_encoder_cp_flash_dist.py under torchrun "
        f"(WORLD_SIZE={os.environ.get('WORLD_SIZE', 'unset')})."
    ),
)


def test_flash_cp_distributed_harness_placeholder():
    raise AssertionError("unreachable — module is always skipped")
