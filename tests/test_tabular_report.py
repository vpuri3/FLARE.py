from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.tabular_report import markdown_table

REPO = Path(__file__).resolve().parents[1]


def test_markdown_table_minimal() -> None:
    md = markdown_table([["a", "b"], ["1", "2.5"]])
    assert md.splitlines() == [
        "| a | b |",
        "| --- | --- |",
        "| 1 | 2.5 |",
    ]


def test_collect_and_pivot(tmp_path: Path) -> None:
    for topo, pe, loss in [(0, "GRMS", 0.42), (0, "PENODE", 0.51), (1, "GRMS", 0.39)]:
        exp = tmp_path / f"TOPO{topo}_{pe}_20EP" / "ckpt1"
        exp.mkdir(parents=True)
        (exp / "stats.json").write_text(json.dumps({"train_loss": loss}), encoding="utf-8")

    collect = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/tabular_report.py"),
            "collect",
            "--root",
            str(tmp_path),
            "--glob",
            "*_20EP",
            "--columns",
            "exp",
            "train_loss",
            "--format",
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TOPO0_GRMS_20EP" in collect.stdout

    md = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/tabular_report.py"),
            "md",
        ],
        input=collect.stdout,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "| exp | train_loss |" in md.stdout
    assert "| TOPO0_GRMS_20EP | 0.42 |" in md.stdout
