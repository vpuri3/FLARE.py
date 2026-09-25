"""Unit tests for offline dataloader/train bench helpers (no GPU/data required)."""

import json

from pdebench.dataset.bench_dataloader import (
    WAIT_FRACTION_FAIL,
    _write_report,
    parse_timeline,
    summarize_timeline,
    summarize_timing,
    verdict,
)


def test_fail_when_wait_over_10pct():
    assert verdict(wait_s=0.2, step_s=1.0) == "FAIL"


def test_pass_at_10pct_or_below():
    assert verdict(wait_s=0.1, step_s=1.0) == "PASS"
    assert verdict(wait_s=0.05, step_s=1.0) == "PASS"


def test_threshold_constant():
    assert WAIT_FRACTION_FAIL == 0.10


def test_summarize_timing_uses_latter_half():
    # 50 slow head steps (compile-like), 50 fast steady steps
    steps = [1.0] * 50 + [0.10] * 50
    waits = [0.05] * 50 + [0.005] * 50
    out = summarize_timing(steps, waits, total_wall_s=60.0, metric_tail=50)
    assert out["num_steps_recorded"] == 100
    assert abs(out["time_per_step_s"] - 0.10) < 1e-9
    assert abs(out["data_wait_per_step_s"] - 0.005) < 1e-9
    assert out["verdict"] == "PASS"  # 0.005/0.10 = 5%
    # head excess ≈ 50*((1.0+0.05)-(0.10+0.005)) = 50*0.945
    assert abs(out["compile_overhead_s"] - 50 * 0.945) < 1e-6


def test_parse_timeline_derives_setup_and_startup_durations():
    text = """
    dataset_load_start: elapsed=1.000s delta=0.000s
    dataset_loaded: elapsed=3.500s delta=2.500s
    model_build_start: elapsed=3.500s delta=0.000s
    model_built: elapsed=5.000s delta=1.500s
    trainer_construct_start: elapsed=6.000s delta=1.000s
    trainer_compile: elapsed=10.000s delta=4.000s
    trainer_train_enter: elapsed=10.200s delta=0.200s
    trainer_dataloader: elapsed=10.700s delta=0.500s
    trainer_statistics_start: elapsed=10.700s delta=0.000s
    trainer_statistics_done: elapsed=12.200s delta=1.500s
    trainer_first_batch_ready: elapsed=13.200s delta=1.000s
    trainer_first_train_step: elapsed=13.400s delta=0.200s
    """

    markers = parse_timeline(text)
    out = summarize_timeline(markers)

    assert markers["trainer_compile"] == 10.0
    assert out["dataset_init_s"] == 2.5
    assert out["model_init_s"] == 1.5
    assert out["compile_setup_s"] == 4.0
    assert out["dataloader_setup_s"] == 0.5
    assert out["startup_stats_s"] == 1.5
    assert out["first_batch_ready_s"] == 3.0
    assert abs(out["first_train_step_ready_s"] - 3.2) < 1e-9


def test_summarize_timing_reports_warmup_and_steady_state_distribution():
    out = summarize_timing(
        [2.0, 2.0, 1.0, 2.0],
        [0.2, 0.1, 0.1, 0.2],
        total_wall_s=8.0,
        metric_tail=2,
    )

    assert out["warmup_steps"] == 2
    assert out["steady_state_steps"] == 2
    assert out["warmup_mean_step_s"] == 2.0
    assert abs(out["warmup_mean_total_step_s"] - 2.15) < 1e-9
    assert abs(out["steady_state_mean_total_step_s"] - 1.65) < 1e-9
    assert abs(out["warmup_overhead_s"] - 1.0) < 1e-9
    assert out["compile_overhead_s"] == out["warmup_overhead_s"]
    assert abs(out["warmup_to_steady_step_ratio"] - 2.15 / 1.65) < 1e-9
    assert abs(out["steady_state_step_std_s"] - 0.55) < 1e-9
    assert abs(out["steady_state_step_p95_s"] - 2.145) < 1e-9
    assert abs(out["steady_state_step_p99_s"] - 2.189) < 1e-9
    assert abs(out["wait_to_step_ratio"] - 0.1) < 1e-9
    assert out["wait_fraction"] == out["wait_to_step_ratio"]


def test_short_run_with_large_metric_tail_is_strict_json():
    out = summarize_timing([0.1], [0.01], total_wall_s=0.2, metric_tail=50)

    assert out["warmup_steps"] == 0
    assert out["steady_state_steps"] == 1
    assert out["warmup_overhead_s"] is None
    assert out["mean_step_head_s"] is None
    json.dumps(out, allow_nan=False)


def test_mismatched_timing_arrays_expose_dropped_sample_count():
    out = summarize_timing([1.0, 1.0, 1.0], [0.1, 0.1], total_wall_s=3.3, metric_tail=2)

    assert out["num_steps_recorded"] == 2
    assert out["timing_samples_dropped"] == 1


def test_report_json_and_markdown_include_timing_schema(tmp_path):
    report = {
        "schema_version": 2,
        "dataset": "bracket_lug",
        **summarize_timing([2.0, 1.0], [0.2, 0.1], total_wall_s=3.3, metric_tail=1),
        **summarize_timeline({
            "trainer_construct_start": 1.0,
            "trainer_compile": 2.0,
        }),
        "subprocess_wall_s": 3.3,
    }

    _write_report(tmp_path, "bracket_lug", "host", report)
    saved = json.loads((tmp_path / "bracket_lug_host.json").read_text())
    markdown = (tmp_path / "bracket_lug_host.md").read_text()

    assert saved["schema_version"] == 2
    assert saved["compile_setup_s"] == 1.0
    assert saved["warmup_overhead_s"] == saved["compile_overhead_s"]
    assert "warmup_overhead_s=" in markdown
    assert "warmup_mean_total_step_s=" in markdown
    assert "steady_state_mean_total_step_s=" in markdown
    assert "compile_setup_s=" in markdown
