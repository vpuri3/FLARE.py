from __future__ import annotations

from pathlib import Path

HARNESS = Path("out/pdebench/run_glt.sh").read_text()


def test_local_staging_defaults_off_and_has_local_root_default():
    assert 'STAGE_DATA_TO_LOCAL="${STAGE_DATA_TO_LOCAL:-false}"' in HARNESS
    assert 'PDEBENCH_LOCAL_DATA_ROOT="${PDEBENCH_LOCAL_DATA_ROOT:-/tmp/pdebench-data}"' in HARNESS


def test_harness_invokes_stager_before_data_root_argument_is_built():
    stage_call = "python -m pdebench.dataset.local_stage"
    assert stage_call in HARNESS
    assert HARNESS.index(stage_call) < HARNESS.index('BASE_ARGS=(')
    assert HARNESS.index('DATA_ROOT="${PDEBENCH_LOCAL_DATA_ROOT}"') < HARNESS.index('BASE_ARGS=(')


def test_harness_passes_active_eigen_k_to_stager():
    assert '--laplacian-k "${STAGE_LAPLACIAN_K}"' in HARNESS
    assert 'STAGE_LAPLACIAN_K="${GLT_PE_NUM_EIGENMODES:-64}"' in HARNESS


def test_harness_rejects_unsupported_staged_dataset():
    assert 'Local staging supports only micro_puc and micro_puc_fixed' in HARNESS
