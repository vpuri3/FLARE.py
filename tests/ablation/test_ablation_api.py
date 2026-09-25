from __future__ import annotations

import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = REPO_ROOT / "ablation"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)


def _has(name: str) -> bool:
    return (ABLATION_DIR / name).exists()


def test_ablation_python_scripts_compile() -> None:
    for path in sorted(ABLATION_DIR.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize(
    "script,extra_args",
    [
        pytest.param("ablate_num_blocks.py", []),
        pytest.param("ablate_num_heads.py", []),
        pytest.param("ablate_num_layers.py", []),
        pytest.param(
            "ablate_reviews.py",
            [],
            marks=pytest.mark.skipif(not _has("ablate_reviews.py"), reason="script not present in this repo"),
        ),
        pytest.param("time_memory_bwd.py", []),
        pytest.param("time_memory_bwd_flarepp.py", []),
        pytest.param("cp_scaling_bwd.py", []),
    ],
)
def test_argparse_scripts_help(script: str, extra_args: list[str]) -> None:
    proc = _run([sys.executable, str(ABLATION_DIR / script), "--help", *extra_args])
    assert proc.returncode == 0, proc.stderr


def test_num_blocks_cli_noop() -> None:
    proc = _run([
        sys.executable,
        str(ABLATION_DIR / "ablate_num_blocks.py"),
        "--dataset",
        "elasticity",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "No action specified" in proc.stdout


@pytest.mark.parametrize(
    "script",
    [
        "ablate_num_heads.py",
        "ablate_num_layers.py",
        "time_memory_bwd.py",
        "time_memory_bwd_flarepp.py",
        "cp_scaling_bwd.py",
    ],
)
def test_cli_noop(script: str) -> None:
    proc = _run([sys.executable, str(ABLATION_DIR / script)])
    assert proc.returncode == 0, proc.stderr
    assert "No action specified" in proc.stdout


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            "run_comp.sh",
            marks=pytest.mark.skipif(not _has("run_comp.sh"), reason="script not present in this repo"),
        ),
        pytest.param(
            "ablate_latent_blocks.sh",
            marks=pytest.mark.skipif(not _has("ablate_latent_blocks.sh"), reason="script not present in this repo"),
        ),
        pytest.param(
            "ablate_shared_latents.sh",
            marks=pytest.mark.skipif(not _has("ablate_shared_latents.sh"), reason="script not present in this repo"),
        ),
    ],
)
def test_shell_launchers_use_new_api(script: str) -> None:
    text = (ABLATION_DIR / script).read_text(encoding="utf-8")
    assert re.search(r"--model_type\s+[0-9]+", text) is None
    assert "--shared_latents" not in text
    assert "--num_latent_blocks" not in text


def test_utils_runner_uses_new_api_flags() -> None:
    text = (ABLATION_DIR / "utils.py").read_text(encoding="utf-8")
    assert ("--num_layers_ffn" in text) or ("num_layers_mlp" in text)
