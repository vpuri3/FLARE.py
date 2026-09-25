from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_case_stem_flare_and_flarepp():
    from ablation.sweep_flarepp_nasa_crm import _case_stem

    assert _case_stem("flare", 8, 64) == (
        "nasa_crm_flare_cp4_C128_B8_H8_M64_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd5e-2"
    )
    assert _case_stem("flarepp", 16, 32) == (
        "nasa_crm_flarepp_cp4_C128_B16_H8_M32_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16_rmsnorm_wd5e-2"
    )


def test_collect_data_reads_stats_mse(tmp_path):
    from ablation import sweep_flarepp_nasa_crm as harness

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
