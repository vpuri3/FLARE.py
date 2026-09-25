#!/usr/bin/env python3
"""Small CLI for collecting rows and emitting GitHub-flavored markdown tables.

Avoid one-off agent-generated Python for tabular summaries. Prefer:

  python scripts/tabular_report.py md < data.tsv
  python scripts/tabular_report.py collect --root DIR --glob 'PAT' ...
  python scripts/tabular_report.py pivot --tsv data.tsv --row R --col C --value V
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def markdown_table(rows: Sequence[Sequence[str]]) -> str:
    """Render a rectangular table as a minimal pipe markdown table (no cell padding)."""
    if not rows:
        return ""
    width = len(rows[0])
    if width == 0:
        return ""
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {i} has {len(row)} columns, expected {width}")
    header = [str(c) for c in rows[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _read_tsv_rows(stream: Iterable[str]) -> list[list[str]]:
    reader = csv.reader(stream, delimiter="\t")
    return [list(row) for row in reader if row]


def _read_csv_rows(stream: Iterable[str]) -> list[list[str]]:
    reader = csv.reader(stream)
    return [list(row) for row in reader if row]


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_latest_json(parent: Path, pattern: str) -> Path | None:
    matches = sorted(parent.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def _extract_field(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def cmd_md(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    stream = text.splitlines()
    rows = _read_csv_rows(stream) if args.csv else _read_tsv_rows(stream)
    if not rows:
        print("(empty table)", file=sys.stderr)
        return 1
    out = markdown_table(rows)
    if args.title:
        out = f"### {args.title}\n\n{out}"
    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"collect: not a directory: {root}", file=sys.stderr)
        return 1

    id_key, *metric_keys = args.columns
    header = [id_key, *metric_keys]
    body: list[list[str]] = []
    missing: list[str] = []

    for child in sorted(root.glob(args.glob)):
        if not child.is_dir():
            continue
        label = child.name
        if args.id_regex:
            m = re.search(args.id_regex, label)
            if m:
                label = m.group(1) if m.lastindex == 1 else "/".join(m.groups())
        stats_path = _pick_latest_json(child, args.stats_glob)
        if stats_path is None:
            missing.append(label)
            row = [label, *["—" for _ in metric_keys]]
            body.append(row)
            continue
        data = _load_json(stats_path)
        row = [label]
        for key in metric_keys:
            row.append(_fmt_cell(_extract_field(data, key)))
        body.append(row)

    rows = [header, *body]
    tsv = "\n".join("\t".join(r) for r in rows)
    if args.format == "tsv":
        if args.output:
            Path(args.output).write_text(tsv + "\n", encoding="utf-8")
        print(tsv)
        return 0

    md = markdown_table(rows)
    if args.title:
        md = f"### {args.title}\n\n{md}"
    if missing:
        md += f"\n\nMissing ({len(missing)}): " + ", ".join(missing[:20])
        if len(missing) > 20:
            md += f", … (+{len(missing) - 20} more)"
    if args.output:
        Path(args.output).write_text(md + "\n", encoding="utf-8")
    print(md)
    if missing:
        print(f"collect: {len(missing)} dirs missing {args.stats_glob}", file=sys.stderr)
    return 0


def cmd_pivot(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    rows = _read_tsv_rows(text.splitlines())
    if not rows:
        print("pivot: empty input", file=sys.stderr)
        return 1
    header = rows[0]
    try:
        row_i = header.index(args.row)
        col_i = header.index(args.col)
        val_i = header.index(args.value)
    except ValueError as exc:
        print(f"pivot: column not found: {exc}", file=sys.stderr)
        return 1

    grid: dict[str, dict[str, str]] = defaultdict(dict)
    row_labels: list[str] = []
    col_labels: list[str] = []
    for row in rows[1:]:
        rlab = row[row_i]
        clab = row[col_i]
        if rlab not in grid:
            row_labels.append(rlab)
        if clab not in col_labels:
            col_labels.append(clab)
        grid[rlab][clab] = row[val_i]

    out_header = [args.row, *col_labels]
    out_rows = [out_header]
    for rlab in row_labels:
        out_rows.append([rlab, *[grid[rlab].get(clab, "—") for clab in col_labels]])

    md = markdown_table(out_rows)
    if args.title:
        md = f"### {args.title}\n\n{md}"
    if args.output:
        Path(args.output).write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_md = sub.add_parser("md", help="Format TSV/CSV stdin or file as a markdown table")
    p_md.add_argument("--input", type=str, default=None, help="Input file (default: stdin)")
    p_md.add_argument("--csv", action="store_true", help="Input is comma-separated (default: TSV)")
    p_md.add_argument("--title", type=str, default=None)
    p_md.add_argument("--output", type=str, default=None)

    p_collect = sub.add_parser("collect", help="Walk experiment dirs and read JSON metrics")
    p_collect.add_argument("--root", type=Path, required=True)
    p_collect.add_argument("--glob", default="*", help="Glob under --root (default: *)")
    p_collect.add_argument("--stats-glob", default="ckpt*/stats.json")
    p_collect.add_argument(
        "--columns",
        nargs="+",
        required=True,
        metavar="COL",
        help="First column is the row id (dirname); rest are dotted JSON paths, e.g. train_loss test_stats.rel_l2",
    )
    p_collect.add_argument("--id-regex", type=str, default=None, help="Optional regex; group(s) replace row id")
    p_collect.add_argument("--format", choices=("md", "tsv"), default="md")
    p_collect.add_argument("--title", type=str, default=None)
    p_collect.add_argument("--output", type=str, default=None)

    p_pivot = sub.add_parser("pivot", help="Pivot a TSV on row/col/value columns")
    p_pivot.add_argument("--input", type=str, default=None, help="TSV file (default: stdin)")
    p_pivot.add_argument("--row", required=True)
    p_pivot.add_argument("--col", required=True)
    p_pivot.add_argument("--value", required=True)
    p_pivot.add_argument("--title", type=str, default=None)
    p_pivot.add_argument("--output", type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "md":
        return cmd_md(args)
    if args.command == "collect":
        return cmd_collect(args)
    if args.command == "pivot":
        return cmd_pivot(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
