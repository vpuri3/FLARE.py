from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from pdebench.dataset.adapters import get_adapter
from pdebench.dataset.ginot.graph_cache import graph_cache_num_shards
from pdebench.dataset.ginot.loader import _split_train_test
from pdebench.dataset.ginot.stats import field_squared_error_sum
from pdebench.dataset.ginot.types import GINOT_DATASETS
from pdebench.dataset.registry import CANONICAL

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bumper_beam_is_a_canonical_ginot_dataset() -> None:
    assert "bumper_beam" in CANONICAL
    assert "bumper_beam" in GINOT_DATASETS
    assert get_adapter("bumper_beam").name == "bumper_beam"


def test_training_entrypoint_preserves_canonical_dataset_metadata() -> None:
    source = (REPO_ROOT / "pdebench" / "__main__.py").read_text()

    assert 'metadata["dataset"] = resolved_dataset' in source
    assert 'metadata["dataset"] = dataset' not in source


def test_public_load_dataset_routes_bumper_beam_through_ginot_adapter(monkeypatch, tmp_path: Path) -> None:
    import pdebench.dataset.utils as dataset_utils

    calls: dict[str, object] = {}

    def _fake_ginot_loader(**kwargs):
        calls.update(kwargs)
        return "train", "test", {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_ginot_dataset", _fake_ginot_loader)

    train, test, metadata = dataset_utils.load_dataset(
        "BUMPER_BEAM",
        DATADIR_BASE=str(tmp_path),
        PROJDIR=str(tmp_path),
        mesh_split_seed=7,
    )

    assert (train, test) == ("train", "test")
    assert metadata == {"meta": 1, "sample_bridge": "c2a_ginot"}
    assert calls["dataset_name"] == "bumper_beam"
    assert calls["split_seed"] == 7


def test_bumper_beam_split_is_deterministic_104_27() -> None:
    first = _split_train_test("bumper_beam", 131, 0)
    second = _split_train_test("bumper_beam", 131, 0)
    assert first == second
    assert tuple(map(len, first)) == (104, 27)
    assert not set(first[0]) & set(first[1])


def test_bumper_beam_uses_small_graph_cache_shard_count() -> None:
    assert graph_cache_num_shards("bumper_beam", 104) == 4


def test_bumper_field_squared_error_is_channelwise() -> None:
    pred = torch.tensor([[1.0, 3.0], [4.0, 8.0]])
    target = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
    squared_sum, count = field_squared_error_sum(pred, target)
    torch.testing.assert_close(squared_sum, torch.tensor([5.0, 13.0]))
    assert count == 2


def test_bumper_download_spec_is_curated_only() -> None:
    from scripts.download_pdebench_dataset import SNAPSHOT_SPECS, get_tasks, hf_download_workers

    spec = SNAPSHOT_SPECS["bumper_beam"]
    assert spec.repo_id == "AIRBORNEPANDA/BumperBeamCrashExample"
    assert spec.local_subdir == "bumper_beam"
    assert spec.allow_patterns == [
        "CURATED_DATA_VTP/GLOBAL_FEATURES.json",
        "CURATED_DATA_VTP/TRAINING_DATA/*.vtp",
        "CURATED_DATA_VTP/VALIDATION_DATA/*.vtp",
    ]
    assert spec.expected_files == ["CURATED_DATA_VTP/GLOBAL_FEATURES.json"]
    assert any(task.key == "bumper_beam" and not task.default_yes for task in get_tasks())
    assert hf_download_workers({"PDEBENCH_HF_DOWNLOAD_WORKERS": "12"}) == 12
    assert hf_download_workers({"PDEBENCH_HF_DOWNLOAD_WORKERS": "0"}) == 1


def _write_fake_bumper_snapshot(dst_dir: Path, *, train_count: int, validation_count: int) -> None:
    curated = dst_dir / "CURATED_DATA_VTP"
    (curated / "TRAINING_DATA").mkdir(parents=True, exist_ok=True)
    (curated / "VALIDATION_DATA").mkdir(parents=True, exist_ok=True)
    metadata = {}
    run_id = 1
    for split_name, count in (("TRAINING_DATA", train_count), ("VALIDATION_DATA", validation_count)):
        for _ in range(count):
            (curated / split_name / f"Run{run_id}.vtp").write_text("vtp", encoding="utf-8")
            metadata[f"Run{run_id}"] = {}
            run_id += 1
    (curated / "GLOBAL_FEATURES.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_bumper_snapshot_download_requires_exact_131_curated_vtps(monkeypatch, tmp_path: Path) -> None:
    import scripts.download_pdebench_dataset as download_script

    spec = download_script.SNAPSHOT_SPECS["bumper_beam"]

    def _fake_snapshot_download(**kwargs):
        _write_fake_bumper_snapshot(Path(kwargs["local_dir"]), train_count=104, validation_count=26)
        return kwargs["local_dir"]

    monkeypatch.setattr(download_script, "snapshot_download", _fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match=r"131 VTP"):
        download_script.download_snapshot_dataset(tmp_path, spec)


def test_bumper_snapshot_download_accepts_exact_131_curated_vtps(monkeypatch, tmp_path: Path) -> None:
    import scripts.download_pdebench_dataset as download_script

    spec = download_script.SNAPSHOT_SPECS["bumper_beam"]

    def _fake_snapshot_download(**kwargs):
        _write_fake_bumper_snapshot(Path(kwargs["local_dir"]), train_count=104, validation_count=27)
        return kwargs["local_dir"]

    monkeypatch.setattr(download_script, "snapshot_download", _fake_snapshot_download)

    download_script.download_snapshot_dataset(tmp_path, spec)


def _run_launcher_and_capture_python_args(
    tmp_path: Path,
    *,
    dataset: str,
    glt_pe_num_eigenmodes: int | None = None,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    args_path = tmp_path / "python-args.bin"
    launcher_env = os.environ.copy()
    for key in (
        "AMP_DTYPE",
        "BATCH_SIZE",
        "EPOCH",
        "STEPS",
        "GLT_PE",
        "GLT_PE_NUM_EIGENMODES",
        "GLT_PE_RADII",
        "GLT_PE_NEIGHBORS",
        "GLT_PE_N_HIDDEN_LOCAL",
        "GLT_PE_POINTNET_HIDDEN",
        "GLT_PE_NORMALIZE_EDGE_LEN",
        "GLT_NUM_BLOCKS",
        "GLT_CHANNEL_DIM",
        "GLT_NUM_HEADS",
        "NUM_BLOCKS",
        "NUM_CHANNELS",
        "NUM_HEADS",
        "CHANNEL_DIM",
        "MODEL",
        "GEO_BALL_RADII",
        "GEO_BALL_KS",
        "GEO_NUM_SLICES",
        "EXTRA_ARGS",
        "OPTIMIZER",
        "MIN_LR",
        "SCHEDULE",
        "EMA",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "COMPILE_MODEL",
    ):
        launcher_env.pop(key, None)
    launcher_env.update(
        DATASET=dataset,
        TORCHRUN_NPROC="1",
    )
    if glt_pe_num_eigenmodes is not None:
        launcher_env["GLT_PE_NUM_EIGENMODES"] = str(glt_pe_num_eigenmodes)
    if env_overrides is not None:
        launcher_env.update(env_overrides)
    script = f"""
set -euo pipefail
python() {{
  printf '%s\\0' "$@" > "{args_path}"
}}
export -f python
bash out/pdebench/run_glt.sh >/dev/null
"""
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=REPO_ROOT,
        env=launcher_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return [part.decode("utf-8") for part in args_path.read_bytes().split(b"\0") if part]


def _arg_value(args: list[str], key: str) -> str:
    for i, token in enumerate(args):
        if token == key:
            return args[i + 1]
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    raise ValueError(f"{key!r} is not in list")


def test_launcher_glt_pe_none_omits_eigen_flags(tmp_path: Path) -> None:
    args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={"GLT_PE": "none"},
    )
    assert _arg_value(args, "--model.pe.kind") == "none"
    assert "--model.pe.num_eigenmodes" not in args
    assert "--model.pe.laplacian_spec" not in args
    assert "--model.pe.filter_type" not in args


def test_launcher_glt_geo_transolver_pe_flags(tmp_path: Path) -> None:
    args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={
            "GLT_PE": "geo_transolver_pe",
            "GLT_PE_RADII": "[0.1,0.3]",
            "GLT_PE_NEIGHBORS": "[4,16]",
            "GLT_PE_N_HIDDEN_LOCAL": "24",
        },
    )
    assert _arg_value(args, "--model.pe.kind") == "geo_transolver_pe"
    assert _arg_value(args, "--model.pe.radii") == "[0.1,0.3]"
    assert _arg_value(args, "--model.pe.neighbors_in_radius") == "[4,16]"
    assert _arg_value(args, "--model.pe.n_hidden_local") == "24"
    assert "--model.pe.num_eigenmodes" not in args
    assert "--model.pe.laplacian_spec" not in args


def test_launcher_glt_multiscale_hop_pe_flags(tmp_path: Path) -> None:
    args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={
            "GLT_PE": "multiscale_hop_pe",
            "GLT_PE_POINTNET_HIDDEN": "32",
            "GLT_PE_NORMALIZE_EDGE_LEN": "true",
        },
    )
    assert _arg_value(args, "--model.pe.kind") == "multiscale_hop_pe"
    assert _arg_value(args, "--model.pe.pointnet_hidden_dim") == "32"
    assert _arg_value(args, "--model.pe.normalize_by_mean_edge_length") == "true"
    assert "--model.pe.num_eigenmodes" not in args


def test_bumper_launcher_effective_defaults_and_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "EXTRA_ARGS",
        "--training.amp_dtype fp32 --training.batch_size 9 --training.epochs 7",
    )
    bumper_args = _run_launcher_and_capture_python_args(tmp_path, dataset="bumper_beam")
    assert _arg_value(bumper_args, "--dataset.dataset") == "bumper_beam"
    assert _arg_value(bumper_args, "--training.amp_dtype") == "bf16"
    assert _arg_value(bumper_args, "--training.batch_size") == "1"
    assert _arg_value(bumper_args, "--training.epochs") == "2000"
    assert _arg_value(bumper_args, "--training.compile_model") == "true"
    assert _arg_value(bumper_args, "--training.ema") == "true"
    assert _arg_value(bumper_args, "--optimizer.optimizer") == "adam"
    assert _arg_value(bumper_args, "--optimizer.learning_rate") == "1e-3"
    assert _arg_value(bumper_args, "--optimizer.weight_decay") == "1e-5"
    assert _arg_value(bumper_args, "--scheduler.schedule") == "OneCycleLR"
    assert _arg_value(bumper_args, "--model.num_blocks") == "8"
    assert _arg_value(bumper_args, "--model.channel_dim") == "128"
    assert _arg_value(bumper_args, "--model.num_heads") == "8"
    assert _arg_value(bumper_args, "--model.pe.kind") == "raw_eigen"
    assert _arg_value(bumper_args, "--model.pe.num_eigenmodes") == "64"
    assert _arg_value(bumper_args, "--model.pe_inject_mode") == "concat_input"

    geo_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={"MODEL": "geo_transolver"},
    )
    assert _arg_value(geo_args, "--model.model") == "geo_transolver"
    assert _arg_value(geo_args, "--model.num_blocks") == "8"
    assert _arg_value(geo_args, "--model.channel_dim") == "128"
    assert _arg_value(geo_args, "--model.num_heads") == "8"
    assert _arg_value(geo_args, "--model.num_slices") == "128"
    assert _arg_value(geo_args, "--training.epochs") == "2000"
    assert _arg_value(geo_args, "--optimizer.optimizer") == "adam"
    assert _arg_value(geo_args, "--scheduler.schedule") == "OneCycleLR"
    assert _arg_value(geo_args, "--model.use_geo") == "true"
    assert _arg_value(geo_args, "--model.include_local_features") == "true"
    assert _arg_value(geo_args, "--model.concat_local_features") == "true"
    assert _arg_value(geo_args, "--model.geometry_dim") == "3"
    assert _arg_value(geo_args, "--model.global_dim") == "3"
    assert _arg_value(geo_args, "--model.ball_radii") == "[0.05,0.25]"
    assert _arg_value(geo_args, "--model.ball_ks") == "[8,32]"

    ts_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={"MODEL": "transolver"},
    )
    assert _arg_value(ts_args, "--model.model") == "transolver"
    assert _arg_value(ts_args, "--model.num_blocks") == "8"
    assert _arg_value(ts_args, "--model.channel_dim") == "128"
    assert _arg_value(ts_args, "--model.num_heads") == "8"
    assert _arg_value(ts_args, "--model.num_slices") == "128"
    assert _arg_value(ts_args, "--training.epochs") == "2000"
    assert _arg_value(ts_args, "--optimizer.optimizer") == "adam"

    override_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        glt_pe_num_eigenmodes=17,
        env_overrides={
            "AMP_DTYPE": "fp16",
            "BATCH_SIZE": "3",
            "EPOCH": "17",
        },
    )
    assert _arg_value(override_args, "--training.amp_dtype") == "fp16"
    assert _arg_value(override_args, "--training.batch_size") == "3"
    assert _arg_value(override_args, "--training.epochs") == "17"
    assert _arg_value(override_args, "--model.pe.num_eigenmodes") == "17"

    step_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={"EPOCH": "17", "STEPS": "9"},
    )
    assert _arg_value(step_args, "--training.steps") == "9"
    assert _arg_value(step_args, "--training.epochs") == "0"

    bracket_args = _run_launcher_and_capture_python_args(tmp_path, dataset="bracket_lug")
    assert _arg_value(bracket_args, "--training.amp_dtype") == "fp16"
    assert _arg_value(bracket_args, "--training.epochs") == "100"
    assert _arg_value(bracket_args, "--model.pe.num_eigenmodes") == "64"


def test_launcher_wires_gito(tmp_path: Path) -> None:
    gito_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={"MODEL": "gito"},
    )
    assert _arg_value(gito_args, "--model.model") == "gito"
    assert _arg_value(gito_args, "--model.num_blocks_hgt") == "8"
    assert _arg_value(gito_args, "--model.num_blocks_self_attn") == "0"
    assert _arg_value(gito_args, "--model.channel_dim") == "128"
    assert _arg_value(gito_args, "--model.num_heads") == "8"
    assert _arg_value(gito_args, "--run.exp_name") == "bumper_beam_GITO_B8_C128_H8"

    override_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bracket_lug",
        env_overrides={
            "MODEL": "gito",
            "GITO_NUM_BLOCKS": "3",
            "GITO_CHANNEL_DIM": "64",
            "GITO_NUM_HEADS": "4",
            "GITO_RMSNORM": "true",
        },
    )
    assert _arg_value(override_args, "--model.num_blocks_hgt") == "3"
    assert _arg_value(override_args, "--model.num_blocks_self_attn") == "0"
    assert _arg_value(override_args, "--model.channel_dim") == "64"
    assert _arg_value(override_args, "--model.num_heads") == "4"
    assert _arg_value(override_args, "--model.rmsnorm") == "true"
    assert _arg_value(override_args, "--run.exp_name") == "bracket_lug_GITO_B3_C64_H4"


def test_bumper_launcher_can_disable_geo_local_feature_concatenation(tmp_path: Path) -> None:
    geo_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={
            "MODEL": "geo_transolver",
            "GEO_CONCAT_LOCAL_FEATURES": "false",
        },
    )
    assert _arg_value(geo_args, "--model.concat_local_features") == "false"


def test_bumper_launcher_can_disable_use_geo(tmp_path: Path) -> None:
    geo_args = _run_launcher_and_capture_python_args(
        tmp_path,
        dataset="bumper_beam",
        env_overrides={
            "MODEL": "geo_transolver",
            "GEO_USE_GEO": "false",
        },
    )
    assert _arg_value(geo_args, "--model.use_geo") == "false"


def test_bumper_laplacian_precompute_defaults(monkeypatch) -> None:
    import pdebench.dataset.laplacian.precompute as laplacian_precompute
    from pdebench.dataset.laplacian.spec import DATASET_LAPLACIAN_SPECS

    assert DATASET_LAPLACIAN_SPECS["bumper_beam"] == "graph:32"

    monkeypatch.setattr(
        "sys.argv",
        ["precompute"],
    )
    args = laplacian_precompute._parse_args()
    assert "bumper_beam" in args.datasets.split(",")

    launcher = Path("out/pdebench/run_glt.sh").read_text()
    assert "bumper_beam)" in launcher
    assert "python -m pdebench.dataset.laplacian.precompute --datasets bumper_beam" in launcher
