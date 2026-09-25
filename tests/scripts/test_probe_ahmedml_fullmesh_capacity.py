from __future__ import annotations

import argparse
import json
import os

import pytest
import torch

from scripts import probe_ahmedml_fullmesh_capacity as probe


def test_audit_event_is_one_atomic_stderr_write(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = []
    monkeypatch.setattr(os, "write", lambda fd, data: writes.append((fd, data)) or len(data))

    probe.audit_event("ready", world_size=2)

    assert writes == [(2, b"AHMEDML_PROBE event=ready rank=0 world_size=2\n")]


def test_parse_args_defaults_to_eight_blocks() -> None:
    args = probe.parse_args(["--model", "flare", "--data-root", "/data", "--run-id", "run_7"])

    assert args.num_blocks == 8


@pytest.mark.parametrize("model", ["bad", "glt"])
def test_parse_args_rejects_unsupported_models(model: str) -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(["--model", model, "--data-root", "/data", "--run-id", "run_7"])


def test_parse_args_rejects_non_eight_block_probe() -> None:
    with pytest.raises(SystemExit):
        probe.parse_args(
            ["--model", "flare", "--data-root", "/data", "--run-id", "run_7", "--num-blocks", "4"]
        )


def test_make_run_dataset_selects_exactly_one_run(tmp_path) -> None:
    args = argparse.Namespace(data_root=tmp_path, run_id="run_17")

    dataset = probe.make_run_dataset(args)

    assert len(dataset) == 1
    assert dataset.source_run_ids == ("run_17",)


def test_result_schema_is_json_serializable() -> None:
    result = probe.make_result(model="flarepp", world_size=4, run_id="run_17", point_count=123)

    assert result == {
        "model": "flarepp",
        "world_size": 4,
        "cp_size": 4,
        "run_id": "run_17",
        "point_count": 123,
        "inference_status": "ERROR",
        "training_status": "ERROR",
        "metrics": {},
        "rank_memory": [],
    }
    probe.dumps_result(result)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (torch.OutOfMemoryError("out of memory"), "OOM"),
        (RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"), "OOM"),
        (RuntimeError("shape mismatch"), "ERROR"),
    ],
)
def test_classify_exception(error: BaseException, expected: str) -> None:
    assert probe.classify_exception(error) == expected


def test_phase_runner_returns_pass_and_classifies_failures() -> None:
    assert probe.run_phase(lambda: {"value": 1}) == ("PASS", {"value": 1}, None)
    status, value, message = probe.run_phase(lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    assert (status, value) == ("OOM", None)
    assert message == "CUDA out of memory"


@pytest.mark.parametrize("model_name", ["flare", "flarepp"])
def test_build_model_uses_factory_with_eight_blocks_and_keeps_stdout_clean(
    model_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def factory(cfg, metadata, rank):
        print("factory diagnostic")
        calls.append((cfg, metadata, rank))
        return cfg, torch.nn.Linear(6, 4)

    args = argparse.Namespace(model=model_name, data_root="/data", num_blocks=8)
    cfg, model = probe.build_probe_model(args, 19, 0, factory=factory)

    assert isinstance(model, torch.nn.Linear)
    assert cfg.model.model == model_name
    assert cfg.model.num_blocks == 8
    assert calls[0][1]["max_length"] == 19
    assert capsys.readouterr().out == ""


def test_cpu_workload_runs_one_inference_and_one_training_step() -> None:
    class CountingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(6, 4, bias=False)
            torch.nn.init.zeros_(self.linear.weight)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.linear(x)

    model = CountingModel()
    before = model.linear.weight.detach().clone()
    x = torch.ones(1, 5, 6)
    target = torch.ones(1, 5, 4)

    inference, training = probe.run_workload(
        model,
        x,
        target,
        cp_state=None,
        autocast=lambda: torch.autocast("cpu", enabled=False),
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=0.1),
    )

    assert inference[0] == "PASS"
    assert set(inference[1]) == {"full_rel_l2", "pressure_rel_l2", "wall_shear_rel_l2"}
    assert training[0] == "PASS"
    assert model.calls == 2
    assert not torch.equal(model.linear.weight, before)


def test_workload_emits_auditable_phase_markers(capfd: pytest.CaptureFixture[str]) -> None:
    model = torch.nn.Linear(6, 4, bias=False)
    x = torch.ones(1, 5, 6)
    target = torch.ones(1, 5, 4)

    probe.run_workload(
        model,
        x,
        target,
        cp_state=None,
        autocast=lambda: torch.autocast("cpu", enabled=False),
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=0.1),
    )

    stderr = capfd.readouterr().err
    assert stderr.count("event=inference_forward_start") == 1
    assert stderr.count("event=inference_forward_complete") == 1
    assert stderr.count("event=physical_metrics_complete") == 1
    assert stderr.count("event=training_forward_start") == 1
    assert stderr.count("event=training_forward_complete") == 1
    assert stderr.count("event=backward_complete") == 1
    assert stderr.count("event=optimizer_step_complete") == 1
    assert "model_call=1 full_mesh=true" in stderr
    assert "model_call=2 full_mesh=true" in stderr


def test_global_phase_promotes_peer_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    def gather(outcomes, local):
        outcomes[:] = [local, ("OOM", None, "peer allocation failed")]

    monkeypatch.setattr(probe.dist, "all_gather_object", gather)

    outcome = probe._global_phase(("PASS", {"ok": True}, None), 2)

    assert outcome == ("OOM", None, "peer allocation failed")


def test_gather_memory_includes_every_reporting_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    def gather(memories, local):
        memories[:] = [local, {"rank": 1, "peak_allocated_bytes": 30, "peak_reserved_bytes": 40}]

    monkeypatch.setattr(probe.dist, "all_gather_object", gather)
    local = {"rank": 0, "peak_allocated_bytes": 10, "peak_reserved_bytes": 20}

    assert probe.gather_rank_memory(local, 2) == [
        local,
        {"rank": 1, "peak_allocated_bytes": 30, "peak_reserved_bytes": 40},
    ]


def test_main_stdout_is_exactly_one_json_object(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    expected = probe.make_result(model="flare", world_size=1, run_id="run_7", point_count=5)
    expected["inference_status"] = "PASS"
    expected["training_status"] = "PASS"
    monkeypatch.setattr(probe, "probe", lambda args: expected)

    assert probe.main(["--model", "flare", "--data-root", "/data", "--run-id", "run_7"]) == 0

    stdout = capsys.readouterr().out
    assert stdout.count("\n") == 1
    assert json.loads(stdout) == expected


def test_main_destroys_initialized_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = probe.make_result(model="flare", world_size=2, run_id="run_7", point_count=5)
    expected["inference_status"] = "PASS"
    expected["training_status"] = "PASS"
    destroyed = []
    monkeypatch.setattr(probe, "probe", lambda args: expected)
    monkeypatch.setattr(probe.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(probe.dist, "destroy_process_group", lambda: destroyed.append(True))

    assert probe.main(["--model", "flare", "--data-root", "/data", "--run-id", "run_7"]) == 0
    assert destroyed == [True]
