from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_case_stem_flare_and_flarepp():
    from ablation.sweep_flarepp_drivaerml_surface import _case_stem

    assert _case_stem("flare", 16, 64) == (
        "drivaerml_surface_flare_cp2_C128_B16_H8_M64_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd1e-4"
    )
    assert _case_stem("flarepp", 20, 32) == (
        "drivaerml_surface_flarepp_cp2_C128_B20_H8_M32_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd1e-4"
    )


def test_collect_data_grid_includes_b16_b20(tmp_path):
    from ablation import sweep_flarepp_drivaerml_surface as harness

    df = harness.collect_data(casedir=str(tmp_path))
    assert set(df["mixer"].unique()) == {"flare", "flarepp", "abupt_surface_only"}
    assert set(df["num_blocks"].unique()) == {2, 4, 8, 12, 16, 20}
    assert set(df["num_latents"].dropna().astype(int).unique()) == {32, 64, 128}
    assert len(df) == 2 * 6 * 3 + 6
    assert int(df["context_parallel_size"].iloc[0]) == 2
    assert int(df["complete"].sum()) == 0


def test_collect_data_reads_ckpt10(tmp_path):
    from ablation import sweep_flarepp_drivaerml_surface as harness

    stem = harness._case_stem("flarepp", 16, 128)
    case = tmp_path / f"{stem}_01"
    ckpt = case / "ckpt10"
    ckpt.mkdir(parents=True)
    (ckpt / "stats.json").write_text(
        json.dumps({"train_loss": 0.011, "test_loss": 0.022}),
        encoding="utf-8",
    )
    df = harness.collect_data(casedir=str(tmp_path))
    sub = df[(df["mixer"] == "flarepp") & (df["num_blocks"] == 16) & (df["num_latents"] == 128)]
    assert len(sub) == 1
    row = sub.iloc[0]
    assert bool(row["complete"]) is True
    assert Path(row["case_path"]) == case
    assert row["train_mse"] == pytest.approx(0.011)
    assert row["test_mse"] == pytest.approx(0.022)


def test_plot_results_replaces_stale_existing_csv_rows(tmp_path, monkeypatch):
    from ablation import sweep_flarepp_drivaerml_surface as harness

    monkeypatch.setattr(harness, "FIGDIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    csv_path = tmp_path / "sweep_flarepp_drivaerml_surface.csv"
    existing = pd.DataFrame(
        [
            {
                "outer_label": "old-label",
                "channel_dim": 128,
                "num_heads": 8,
                "num_layers_in_out_proj": 2,
                "out_proj_norm": True,
                "num_layers_ffn": 0,
                "mlp_ratio_ffn": 2.0,
                "weight_decay": 0.0001,
                "precision": "fp16",
                "rmsnorm": True,
                "context_parallel_size": 2,
                "mixer": "flare",
                "num_latents": 32,
                "num_blocks": 2,
                "series": "flare M=32",
                "case_path": "/old/sbandred/path",
                "train_mse": 0.029599707204857866,
                "test_mse": 0.029330539740622042,
                "complete": True,
            }
        ]
    )
    existing.to_csv(csv_path, index=False)

    df = harness.collect_data(casedir=str(tmp_path / "empty"))
    harness.plot_results(df)
    out = pd.read_csv(csv_path)
    old = out[(out["mixer"] == "flare") & (out["num_blocks"] == 2) & (out["num_latents"] == 32)]
    assert len(old) == 1
    assert pd.isna(old.iloc[0]["case_path"])
    assert pd.isna(old.iloc[0]["train_mse"])
    assert bool(old.iloc[0]["complete"]) is False
    assert ((out["num_blocks"] == 16) & (out["mixer"] == "flare") & (out["num_latents"] == 32)).any()
    assert ((out["num_blocks"] == 20) & (out["mixer"] == "flarepp") & (out["num_latents"] == 128)).any()


def test_plot_results_updates_row_when_new_run_complete(tmp_path, monkeypatch):
    from ablation import sweep_flarepp_drivaerml_surface as harness

    monkeypatch.setattr(harness, "FIGDIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    csv_path = tmp_path / "sweep_flarepp_drivaerml_surface.csv"
    pd.DataFrame(
        [
            {
                "mixer": "flare",
                "num_latents": 32,
                "num_blocks": 16,
                "series": "flare M=32",
                "case_path": "",
                "train_mse": float("nan"),
                "test_mse": float("nan"),
                "complete": False,
            }
        ]
    ).to_csv(csv_path, index=False)

    stem = harness._case_stem("flare", 16, 32)
    case = tmp_path / stem
    ckpt = case / "ckpt10"
    ckpt.mkdir(parents=True)
    (ckpt / "stats.json").write_text(
        json.dumps({"train_loss": 0.005, "test_loss": 0.006}),
        encoding="utf-8",
    )
    df = harness.collect_data(casedir=str(tmp_path))
    harness.plot_results(df)
    out = pd.read_csv(csv_path)
    row = out[(out["mixer"] == "flare") & (out["num_blocks"] == 16) & (out["num_latents"] == 32)].iloc[0]
    assert bool(row["complete"]) is True
    assert row["train_mse"] == pytest.approx(0.005)
    assert Path(row["case_path"]) == case


def test_cli_noop():
    proc = subprocess.run(
        [sys.executable, "-m", "ablation.sweep_flarepp_drivaerml_surface"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "No action specified" in proc.stdout
