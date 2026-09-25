from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


RUN_TRAINING_SMOKE = os.environ.get("PDEBENCH_RUN_TRAINING_SMOKE") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not RUN_TRAINING_SMOKE,
    reason="Set PDEBENCH_RUN_TRAINING_SMOKE=1 to run GPU PDEBench training smoke tests.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PDEBench training smoke requires CUDA.")
def test_flare_elasticity_run_comp_10_step_smoke(tmp_path: Path) -> None:
    exp_name = f"pytest_smoke_flare_elasticity_steps10_{os.getpid()}"
    cmd = [
        sys.executable,
        "-m",
        "pdebench",
        "--dataset.dataset",
        "elasticity",
        "--run.train",
        "true",
        "--model.model",
        "flare",
        "--training.epochs",
        "0",
        "--training.steps",
        "10",
        "--training.stats_every",
        "5",
        "--optimizer.weight_decay",
        "1e-5",
        "--training.batch_size",
        "2",
        "--model.channel_dim",
        "64",
        "--model.num_latents",
        "64",
        "--model.num_blocks",
        "8",
        "--model.num_heads",
        "8",
        "--run.seed",
        "0",
        "--run.exp_name",
        exp_name,
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    log_path = tmp_path / "flare_elasticity_smoke.log"
    log_path.write_text(result.stdout)
    tail = "\n".join(result.stdout.splitlines()[-80:])

    assert result.returncode == 0, f"smoke failed; log={log_path}\n{tail}"
    assert "Traceback" not in result.stdout
    assert "[Step 10 / 10]" in result.stdout
    assert not re.search(r"LOSS\s+(nan|inf)", result.stdout, re.IGNORECASE)
    assert not re.search(r"GNORM:\s+(nan|inf)", result.stdout, re.IGNORECASE)
