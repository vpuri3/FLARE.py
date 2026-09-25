from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARD_LAUNCHER = REPO_ROOT / "out" / "pdebench" / "run_flarepp_standard.sh"


def _mixer_default_branch(source: str, mixer: str) -> str:
    start = source.index(f"  {mixer})")
    end = source.index("  ;;", start)
    return source[start:end]


def test_standard_launcher_accepts_luna():
    source = STANDARD_LAUNCHER.read_text(encoding="utf-8")
    assert "luna" in source
    branch = _mixer_default_branch(source, "flare|luna")
    assert 'QK_NORM="${QK_NORM:-false}"' in branch
    assert "--model.model luna" in source
    assert "--model.mixer.kind" in source
    assert "--model.ffn_mlp_ratio" in source
    model_luna = source[source.index('if [[ "${MIXER}" == "luna" ]]; then') :]
    luna_args = model_luna[: model_luna.index("else")]
    for flag in (
        "--model.model luna",
        "--model.num_latents",
        "--model.qk_norm",
        "--model.num_layers_ffn",
        "--model.ffn_mlp_ratio",
        "--model.out_proj_norm",
    ):
        assert flag in luna_args
    assert "--model.mixer.kind" not in luna_args
    assert 'extras="${extras}_qknorm_${QK_NORM}"' in source


def test_standard_launcher_pipe_default_num_blocks_is_2():
    source = STANDARD_LAUNCHER.read_text(encoding="utf-8")
    branch = _mixer_default_branch(source, "pipe")
    assert 'DEFAULT_NUM_BLOCKS=2' in branch
    ela = _mixer_default_branch(source, "elasticity|airfoil_steady")
    assert 'DEFAULT_NUM_BLOCKS=8' in ela


def test_case_stem_luna_fp32():
    from ablation.sweep_flarepp import _case_stem, _mixer_extras

    extras = _mixer_extras("luna")
    assert extras == "_qknorm_false"
    stem = _case_stem("elasticity", "luna", 64, 8, "fp32")
    assert stem == (
        "elasticity_luna_64_C128_B8_H8_IO_2_FFN_0_OPN_1_MR_2p0_fp32_qknorm_false"
    )


def test_standard_launcher_accepts_flarepp():
    source = STANDARD_LAUNCHER.read_text(encoding="utf-8")
    assert "flarepp" in source
    branch = _mixer_default_branch(source, "flarepp")
    assert "QK0_NORM=" not in branch
    assert 'K_NORM="${K_NORM:-true}"' in branch
    assert 'SHARE_K0_V0="${SHARE_K0_V0:-true}"' in branch
    assert 'GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-0.25}"' in branch
    assert 'Q_FIXED_NORM="${Q_FIXED_NORM:-true}"' in branch
    assert "CONVEX_MODE=" not in branch
    assert "USE_GATE=" not in branch
    assert "QK_NORM=" not in branch


def test_standard_launcher_passes_anchored_mixer_flags():
    source = STANDARD_LAUNCHER.read_text(encoding="utf-8")
    # MODEL_ARGS branch (second occurrence after defaults case); find MODEL_ARGS case block
    model_case = source.index("case \"${MIXER}\" in", source.index("MODEL_ARGS=("))
    branch_start = source.index("  flarepp)", model_case)
    branch_end = source.index("  ;;", branch_start)
    branch = source[branch_start:branch_end]
    for flag in (
        "--model.mixer.num_latents",
        "--model.mixer.k_norm",
        "--model.mixer.share_k0_v0",
        "--model.mixer.gate_logit_init",
        "--model.mixer.q_fixed_norm",
    ):
        assert flag in branch
    assert "--model.mixer.qk0_norm" not in branch
    assert "--model.mixer.convex_mode" not in branch
    assert "--model.mixer.use_gate" not in branch
    assert "--model.mixer.qk_norm" not in branch
    # Exp-name tag stays fixed for stem continuity with prior non-convex runs.
    assert 'extras="${extras}_convex_false"' in source
    # Default on: only tag when q_fixed_norm is off.
    assert 'extras="${extras}_qfnorm0"' in source


def test_case_stem_mha_fp32():
    from ablation.sweep_flarepp import _case_stem

    stem = _case_stem("elasticity", "mha", None, 8, "fp32")
    assert stem == (
        "elasticity_mha_C128_B8_H8_IO_2_FFN_0_OPN_1_MR_2p0_fp32_qknorm_false"
    )


def test_case_stem_simplifiedflarepp_fp16():
    from ablation.sweep_flarepp import _case_stem

    stem = _case_stem("darcy", "simplifiedflarepp", 64, 4, "fp16")
    assert stem == (
        "darcy_simplifiedflarepp_64_C128_B4_H8_IO_2_FFN_0_OPN_1_MR_2p0_amp_fp16"
        "_qknorm_false_qk0_true_v0_false_sharek0v0_false_gate_false"
    )


def test_case_stem_flarepp_fp32():
    from ablation.sweep_flarepp import _case_stem

    stem = _case_stem("elasticity", "flarepp", 64, 8, "fp32")
    assert stem == (
        "elasticity_flarepp_64_C128_B8_H8_IO_2_FFN_0_OPN_1_MR_2p0_fp32"
        "_k_true_sharek0v0_true_gate_0.25_convex_false"
    )


def test_case_stem_transolver3():
    from ablation.sweep_flarepp import _case_stem

    stem = _case_stem("pipe", "transolver3", 128, 2, "fp32")
    assert stem == "pipe_transolver3_128_C128_B2_H8_IO_2_FFN_0_OPN_1_MR_2p0_fp32"


def test_collect_data_unknown_dataset():
    from ablation.sweep_flarepp import collect_data

    with pytest.raises(ValueError, match="unsupported dataset"):
        collect_data("lpbf")


def test_cli_noop():
    proc = subprocess.run(
        [sys.executable, "-m", "ablation.sweep_flarepp"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "No action specified" in proc.stdout


def test_plot_results_2x2(monkeypatch, tmp_path):
    import ablation.sweep_flarepp as harness

    monkeypatch.setattr(harness, "FIGDIR", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    empty = tmp_path / "flarepp_standard"
    empty.mkdir()
    df = harness.collect_data("elasticity", casedir=str(empty))
    assert set(df["precision"].unique()) == {"fp32", "fp16"}
    harness.plot_results(df, "elasticity")
    assert (tmp_path / "sweep_flarepp_elasticity.pdf").is_file()
    assert (tmp_path / "sweep_flarepp_elasticity.csv").is_file()


def test_collect_data_casedir(tmp_path):
    import ablation.sweep_flarepp as harness

    empty = tmp_path / "flarepp_standard"
    empty.mkdir()
    df = harness.collect_data("elasticity", casedir=str(empty))
    assert len(df) == len(harness.MIXER_SPECS) * len(harness.NUM_BLOCKS) * len(harness.PRECISIONS)
    assert int(df["complete"].sum()) == 0


def test_best_rel_error_falls_back_to_stats_when_rel_error_missing(tmp_path):
    from ablation.sweep_flarepp import _best_rel_error_across_ckpts

    case = tmp_path / "elasticity_mha_C128_B2"
    for i, (tr, te) in enumerate([(0.05, 0.06), (0.02, 0.04), (0.03, 0.05)]):
        ckpt = case / f"ckpt{i:02d}"
        ckpt.mkdir(parents=True)
        (ckpt / "stats.json").write_text(
            json.dumps(
                {
                    "train_loss": tr,
                    "test_loss": te,
                    "train_loss_ema": tr - 0.001,
                    "test_loss_ema": te - 0.001,
                }
            )
        )
    (case / "ckpt10").mkdir()
    (case / "ckpt10" / "stats.json").write_text(
        json.dumps({"train_loss": 0.025, "test_loss": 0.045})
    )

    assert _best_rel_error_across_ckpts(str(case), "train") == pytest.approx(0.019)
    assert _best_rel_error_across_ckpts(str(case), "test") == pytest.approx(0.039)


def test_best_rel_error_prefers_rel_error_json_over_stats(tmp_path):
    from ablation.sweep_flarepp import _best_rel_error_across_ckpts

    case = tmp_path / "darcy_mha_C128_B2"
    ckpt = case / "ckpt10"
    ckpt.mkdir(parents=True)
    (ckpt / "stats.json").write_text(
        json.dumps({"train_loss": 0.99, "test_loss": 0.98})
    )
    (ckpt / "rel_error.json").write_text(
        json.dumps({"train_rel_error": 0.01, "test_rel_error": 0.02})
    )

    assert _best_rel_error_across_ckpts(str(case), "train") == pytest.approx(0.01)
    assert _best_rel_error_across_ckpts(str(case), "test") == pytest.approx(0.02)
