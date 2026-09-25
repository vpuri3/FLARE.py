from __future__ import annotations

import json

import pytest


def _write_stats(case, *, train_mse: float, test_mse: float) -> None:
    stats = case / "ckpt10" / "stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "train_loss": train_mse,
                "test_loss": test_mse,
                "train_stats": {"mse": train_mse, "full_rel_l2": 0.1},
                "test_stats": {
                    "mse": test_mse,
                    "full_rel_l2": 0.2,
                    "pressure_rel_l2": 0.3,
                    "wall_shear_rel_l2": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )


def test_collect_data_uses_completed_rerun_and_b2_legacy_case(tmp_path):
    from ablation import sweep_abupt_drivaerml_surface as harness

    stale = tmp_path / harness._grid_case_stem(32, 2)
    stale.mkdir()
    completed_rerun = tmp_path / f"{harness._grid_case_stem(32, 2)}_01"
    _write_stats(completed_rerun, train_mse=0.01, test_mse=0.02)
    legacy = tmp_path / harness._legacy_b2_case_stem(256)
    _write_stats(legacy, train_mse=0.03, test_mse=0.04)

    df = harness.collect_data(casedir=tmp_path)

    assert len(df) == 50
    assert set(df["num_surface_anchors"]) == {32, 64, 128, 256, 512}
    assert set(df["num_blocks"]) == {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
    rerun = df[(df["num_surface_anchors"] == 32) & (df["num_blocks"] == 2)].iloc[0]
    assert rerun["case_path"] == str(completed_rerun)
    assert bool(rerun["complete"]) is True
    assert rerun["test_mse"] == pytest.approx(0.02)
    assert rerun["test_full_rel_l2"] == pytest.approx(0.2)
    b2_legacy = df[(df["num_surface_anchors"] == 256) & (df["num_blocks"] == 2)].iloc[0]
    assert b2_legacy["case_path"] == str(legacy)
    assert b2_legacy["test_mse"] == pytest.approx(0.04)


def test_plot_results_writes_mse_curve_and_csv(tmp_path, monkeypatch):
    from ablation import sweep_abupt_drivaerml_surface as harness

    case = tmp_path / harness._grid_case_stem(32, 2)
    _write_stats(case, train_mse=0.01, test_mse=0.02)
    monkeypatch.setattr(harness, "FIGDIR", str(tmp_path / "figs"))

    df = harness.collect_data(casedir=tmp_path)
    csv_path, pdf_path = harness.write_results(df)

    assert csv_path.name == "sweep_abupt_drivaerml_surface.csv"
    assert pdf_path.name == "sweep_abupt_drivaerml_surface.pdf"
    assert csv_path.is_file()
    assert pdf_path.is_file()
