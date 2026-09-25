from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_case_stem_flare_and_flarepp():
    from ablation.sweep_flarepp_ahmedml import _case_stem

    assert _case_stem("flare", 8, 64) == (
        "ahmedml_surface_flare_cp2_C128_B8_H8_M64_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd1e-4"
    )
    assert _case_stem("flarepp", 12, 32) == (
        "ahmedml_surface_flarepp_cp2_C128_B12_H8_M32_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd1e-4"
    )


def test_collect_data_grid_shape(tmp_path):
    from ablation import sweep_flarepp_ahmedml as harness

    df = harness.collect_data(casedir=str(tmp_path))
    assert len(df) == len(harness.MODELS) * len(harness.NUM_BLOCKS) * len(harness.NUM_LATENTS)
    assert set(df["mixer"].unique()) == {"flare", "flarepp"}
    assert set(df["num_blocks"].unique()) == {2, 4, 8, 12}
    assert set(df["num_latents"].astype(int).unique()) == {32, 64, 128}
    assert set(df["precision"].unique()) == {"fp16"}
    assert bool(df["rmsnorm"].all())
    assert float(df["weight_decay"].iloc[0]) == pytest.approx(1e-4)
    assert int(df["context_parallel_size"].iloc[0]) == 2
    assert int(df["complete"].sum()) == 0


def test_collect_data_reads_ckpt10(tmp_path):
    from ablation import sweep_flarepp_ahmedml as harness

    stem = harness._case_stem("flarepp", 4, 128)
    case = tmp_path / f"{stem}_01"
    ckpt = case / "ckpt10"
    ckpt.mkdir(parents=True)
    (ckpt / "stats.json").write_text(
        json.dumps({"train_loss": 0.011, "test_loss": 0.022}),
        encoding="utf-8",
    )
    df = harness.collect_data(casedir=str(tmp_path))
    sub = df[(df["mixer"] == "flarepp") & (df["num_blocks"] == 4) & (df["num_latents"] == 128)]
    assert len(sub) == 1
    row = sub.iloc[0]
    assert bool(row["complete"]) is True
    assert Path(row["case_path"]) == case
    assert row["train_mse"] == pytest.approx(0.011)
    assert row["test_mse"] == pytest.approx(0.022)


def test_plot_results_writes_csv_and_pdf(tmp_path, monkeypatch):
    from ablation import sweep_flarepp_ahmedml as harness

    monkeypatch.setattr(harness, "FIGDIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    df = harness.collect_data(casedir=str(tmp_path / "empty"))
    harness.plot_results(df)
    assert (tmp_path / "sweep_flarepp_ahmedml_surface.pdf").is_file()
    assert (tmp_path / "sweep_flarepp_ahmedml_surface.csv").is_file()


def test_cli_noop():
    proc = subprocess.run(
        [sys.executable, "-m", "ablation.sweep_flarepp_ahmedml"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "No action specified" in proc.stdout
