from __future__ import annotations

import json

from mlutils.metrics import format_metric, format_scalar, metric_is_set, normalize_metric, unset_metric
from mlutils.run_timer import RunTimer


def test_unset_metric_is_none_not_nan() -> None:
    assert unset_metric() is None
    assert not metric_is_set(None)
    assert not metric_is_set(float("nan"))
    assert metric_is_set(1.0)
    assert normalize_metric(float("nan")) is None
    assert format_metric(None) == "null"
    assert format_metric(0.5) == "5.000000e-01"
    assert format_scalar(0.5, precision=4) == "5.0000e-01"


def test_run_timer_records_marks() -> None:
    timer = RunTimer(enabled=True, rank=0, log_rank=0, print_marks=False)
    timer.mark("a")
    timer.mark("b")
    assert len(timer.marks) == 2
    assert timer.marks[0][0] == "a"
    assert timer.marks[1][0] == "b"


def test_stats_json_null_for_unset_loss() -> None:
    payload = {"train_loss": None, "test_loss": None}
    text = json.dumps(payload)
    assert "null" in text
    assert "NaN" not in text
