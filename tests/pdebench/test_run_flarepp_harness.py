from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "out" / "pdebench" / "run_flarepp.sh"


def _flarepp_branch(text: str) -> str:
    # Extract the else/flarepp MODEL_ARGS region after MODEL == flare branch.
    idx = text.index("# flarepp")
    return text[idx:]


def test_run_flarepp_defaults_match_anchored() -> None:
    text = LAUNCHER.read_text()
    assert "DEFAULT_NUM_BLOCKS=8" in text
    assert "DEFAULT_CHANNEL_DIM=128" in text
    assert "DEFAULT_NUM_HEADS=8" in text
    assert "DEFAULT_NUM_LATENTS=128" in text
    assert "DEFAULT_SHARE_K0_V0_FLAREPP=true" in text
    assert "DEFAULT_SHARE_K0_V0_FLAREPP=false" not in text
    assert 'GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-0.25}"' in text
    branch = _flarepp_branch(text)
    assert "--model.qk0_norm" not in branch
    assert "--model.k_norm" in branch
    assert "--model.q_fixed_norm" in branch
    assert "--model.gate_logit_init" in branch
    assert "--model.share_k0_v0" in branch
    assert "DEFAULT_QK0_NORM_FLAREPP" not in text
    assert "--model.use_gate" not in branch
    assert "--model.qk_norm" not in branch
    assert "--model.v0_norm" not in branch
    assert "--model.k_num_layers" not in branch
    assert "--model.v_num_layers" not in branch
    assert "gate_tag" not in branch
