from pathlib import Path

import pytest
import torch
from torch import nn

from lra.__main__ import Config, make_model
from lra.models.transolver import PhysicsAttention, TransolverBlock
from lra.models.wrapper import ModelWrapper
from pdebench.models.transolver import PhysicsAttention as PDEPhysicsAttention
from pdebench.models.transolver import Transolver_block as PDETransolverBlock


@pytest.mark.parametrize("use_mask", [False, True])
def test_physics_attention_matches_pdebench(use_mask: bool) -> None:
    torch.manual_seed(0)
    reference = PDEPhysicsAttention(
        dim=32,
        heads=4,
        dim_head=8,
        dropout=0.0,
        slice_num=8,
    ).eval()
    actual = PhysicsAttention(
        dim=32,
        heads=4,
        dim_head=8,
        dropout=0.0,
        slice_num=8,
    ).eval()
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(2, 17, 32)
    mask = None
    if use_mask:
        mask = torch.ones(2, 17, dtype=torch.bool)
        mask[:, -3:] = False

    with torch.no_grad():
        expected = reference(x, mask=mask)
        result = actual(x, attention_mask=mask)

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("use_mask", [False, True])
def test_transolver_block_matches_pdebench(use_mask: bool) -> None:
    torch.manual_seed(1)
    reference = PDETransolverBlock(
        num_heads=4,
        hidden_dim=32,
        dropout=0.0,
        act="gelu",
        mlp_ratio=2.0,
        slice_num=8,
        last_layer=False,
    ).eval()
    actual = TransolverBlock(
        channel_dim=32,
        num_heads=4,
        act="gelu",
        mlp_ratio=2.0,
        num_slices=8,
    ).eval()
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(2, 17, 32)
    mask = None
    if use_mask:
        mask = torch.ones(2, 17, dtype=torch.bool)
        mask[:, -3:] = False

    with torch.no_grad():
        expected = reference(x, mask=mask)
        result = actual(x, attention_mask=mask)

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_physics_attention_excludes_masked_values() -> None:
    torch.manual_seed(2)
    attention = PhysicsAttention(dim=32, heads=4, dim_head=8, slice_num=8).eval()
    x = torch.randn(2, 11, 32)
    changed = x.clone()
    changed[:, -2:] = 1000.0
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[:, -2:] = False

    with torch.no_grad():
        baseline = attention(x, attention_mask=mask)
        result = attention(changed, attention_mask=mask)

    assert torch.allclose(result[:, :-2], baseline[:, :-2], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(result[:, -2:]) == 0


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (torch.ones(2, 8), "boolean"),
        (torch.ones(2, 7, dtype=torch.bool), "shape"),
    ],
)
def test_physics_attention_rejects_invalid_mask(mask: torch.Tensor, match: str) -> None:
    attention = PhysicsAttention(dim=32, heads=4, dim_head=8, slice_num=8)
    with pytest.raises(ValueError, match=match):
        attention(torch.randn(2, 8, 32), attention_mask=mask)


def test_transolver_block_validates_configuration() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransolverBlock(channel_dim=30, num_heads=4)
    with pytest.raises(ValueError, match="num_slices"):
        TransolverBlock(channel_dim=32, num_heads=4, num_slices=0)
    with pytest.raises(ValueError, match="RoPE"):
        TransolverBlock(channel_dim=32, num_heads=4, rope=object())


def _make_wrapper(**kwargs) -> ModelWrapper:
    defaults = {
        "task": "text",
        "vocab_size": 128,
        "num_labels": 3,
        "max_length": 64,
        "backend": "transolver",
        "channel_dim": 32,
        "num_heads": 4,
        "num_blocks": 2,
        "pos_embed": "abs",
        "num_slices": 8,
        "mlp_ratio": 2.0,
    }
    return ModelWrapper(**(defaults | kwargs))


def test_model_wrapper_builds_transolver_backend() -> None:
    model = _make_wrapper().eval()
    with torch.no_grad():
        result = model(torch.randint(0, 128, (2, 31)))
    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()
    assert all(isinstance(block, TransolverBlock) for block in model.blocks)


def test_model_wrapper_rejects_transolver_rope() -> None:
    with pytest.raises(ValueError, match="RoPE"):
        _make_wrapper(pos_embed="rope", num_blocks=1)


def test_make_model_passes_num_slices() -> None:
    cfg = Config(
        model_type="transolver",
        num_blocks=1,
        channel_dim=32,
        num_heads=4,
        pos_embed="abs",
        num_slices=7,
    )
    metadata = {
        "task": "text",
        "vocab_size": 128,
        "num_labels": 3,
        "binary_classification": False,
        "max_length": 64,
    }
    model = make_model(cfg, metadata, GLOBAL_RANK=1)
    assert isinstance(model.blocks[0], TransolverBlock)
    assert model.blocks[0].Attn.in_project_slice.out_features == 7


def test_make_model_rejects_transolver_trm() -> None:
    cfg = Config(
        model_type="transolver",
        trm=True,
        num_blocks=1,
        channel_dim=32,
        num_heads=4,
        pos_embed="abs",
    )
    metadata = {
        "task": "text",
        "vocab_size": 128,
        "num_labels": 3,
        "binary_classification": False,
        "max_length": 64,
    }
    with pytest.raises(NotImplementedError, match="Transolver is not supported with TRM"):
        make_model(cfg, metadata, GLOBAL_RANK=1)


def test_wrapper_calls_backend_initialization_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = TransolverBlock.initialize_weights

    def spy(block: TransolverBlock) -> None:
        nonlocal calls
        calls += 1
        original(block)

    monkeypatch.setattr(TransolverBlock, "initialize_weights", spy)
    _make_wrapper(num_blocks=2)
    assert calls == 2


@pytest.mark.parametrize("pool", ["mean", "max"])
def test_pooling_ignores_masked_tokens(pool: str) -> None:
    torch.manual_seed(3)
    model = ModelWrapper(
        task="listops",
        vocab_size=128,
        num_labels=3,
        max_length=8,
        pool=pool,
        backend="transformer",
        channel_dim=32,
        num_heads=4,
        num_blocks=0,
        pos_embed="abs",
    ).eval()
    short_ids = torch.tensor([[1, 2, 3, 4]])
    padded_ids = torch.tensor([[1, 2, 3, 4, 99, 100]])
    short_mask = torch.ones(1, 4, dtype=torch.bool)
    padded_mask = torch.tensor([[True, True, True, True, False, False]])

    with torch.no_grad():
        expected = model(short_ids, attention_mask=short_mask)
        result = model(padded_ids, attention_mask=padded_mask)

    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_retrieval_reshapes_attention_mask() -> None:
    model = _make_wrapper(
        task="retrieval",
        max_length=8,
        num_blocks=1,
        pool="mean",
    ).eval()
    input_ids = torch.randint(0, 128, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.bool)
    attention_mask[:, [6, 7, 14, 15]] = False

    with torch.no_grad():
        result = model(input_ids, attention_mask=attention_mask)

    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()


def test_transolver_adds_absolute_positions_once() -> None:
    class CaptureBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input: torch.Tensor | None = None

        def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
            self.input = x.detach().clone()
            return x

    model = _make_wrapper(num_blocks=1).eval()
    first = CaptureBlock()
    second = CaptureBlock()
    model.blocks = nn.ModuleList([first, second])

    with torch.no_grad():
        model(torch.randint(0, 128, (1, 8)))

    assert first.input is not None
    assert second.input is not None
    assert torch.equal(second.input, first.input)


def _assert_transolver_launch(section: str, task: str, expected_fragments: tuple[str, ...]) -> None:
    assert "--model_type transolver" in section, task
    assert "--epochs ${EPOCHS} --steps ${STEPS}" in section, task
    shared_model_args = "--num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS}"
    assert shared_model_args in section, task
    assert "--learning_rate ${TRANSOLVER_LR}" in section, task
    assert "--weight_decay ${TRANSOLVER_WEIGHT_DECAY}" in section, task
    assert "--num_slices ${TRANSOLVER_NUM_SLICES}" in section, task
    assert "--pos_embed abs" in section, task
    for fragment in expected_fragments:
        assert fragment in section, f"{task}: {fragment}"


def test_run_script_has_transolver_for_existing_task_sections() -> None:
    run_script = Path("out/lra/run.sh").read_text()
    active_script, disabled_pathfinder = run_script.split(": <<'PATHFINDER_DISABLED'", maxsplit=1)
    disabled_pathfinder = disabled_pathfinder.rsplit("PATHFINDER_DISABLED", maxsplit=1)[0]

    active_tasks = ["listops", "image", "retrieval", "text"]
    expected_by_task = {
        "listops": (
            "TRANSOLVER_LR=1e-3",
            "TRANSOLVER_WEIGHT_DECAY=1e-5",
            "TRANSOLVER_NUM_SLICES=128",
            "--exp_name ${TASK}/transolver_tuned",
        ),
        "image": (
            "TRANSOLVER_LR=5e-4",
            "TRANSOLVER_WEIGHT_DECAY=1e-1",
            "TRANSOLVER_EMB_DROP=0.05",
            "TRANSOLVER_NUM_SLICES=64",
            "--emb_drop ${TRANSOLVER_EMB_DROP}",
            "--exp_name ${TASK}/transolver_tuned",
        ),
        "retrieval": (
            "TRANSOLVER_LR=8e-4",
            "TRANSOLVER_WEIGHT_DECAY=1e-4",
            "TRANSOLVER_NUM_SLICES=128",
            "--exp_name ${TASK}/transolver_tuned",
        ),
        "text": (
            "TRANSOLVER_LR=2e-5",
            "TRANSOLVER_WEIGHT_DECAY=1e-4",
            "TRANSOLVER_NUM_SLICES=64",
            "--pool ${POOL}",
            "--exp_name ${TASK}/transolver_tuned",
        ),
    }
    starts = [(task, active_script.index(f"TASK={task}")) for task in active_tasks]
    for index, (task, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(active_script)
        _assert_transolver_launch(active_script[start:end], task, expected_by_task[task])

    assert "TASK=pathfinder32" not in active_script
    assert "TASK=pathfinder32" in disabled_pathfinder
    assert disabled_pathfinder.count("--model_type transolver") == 1
    assert "TRANSOLVER_LR=1e-3" in disabled_pathfinder
    assert "TRANSOLVER_WEIGHT_DECAY=1.5e-4" in disabled_pathfinder
    assert "TRANSOLVER_NUM_SLICES=64" in disabled_pathfinder
    assert "--schedule OneCycleLR" in disabled_pathfinder
    assert "--pool cls" in disabled_pathfinder
    assert "--rmsnorm false" in disabled_pathfinder
    assert "--ema false" in disabled_pathfinder
    assert "--emb_drop 0.0" in disabled_pathfinder
    assert "--exp_name ${TASK}/transolver_pf_s64_lr1e3_cls" in disabled_pathfinder
    assert "TASK=pathfinder128" not in run_script
