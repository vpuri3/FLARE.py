#!/usr/bin/env python3
"""Build deform_plate ginot_cache from TFRecords (runs in the tfrecord-only sub-env)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GINOT_DIR = ROOT / "pdebench/dataset/ginot"


def _load_module(module_name: str, filename: str):
    path = GINOT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_core():
    _load_module("deform_plate_tfrecord_io", "deform_plate_tfrecord_io.py")
    return _load_module("deform_plate_core", "deform_plate_core.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deform_plate GINOT cache from TFRecords.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    core = _bootstrap_core()
    cache_root = core.build_deform_plate_cache(args.data_root, overwrite=bool(args.overwrite))
    print(f"Wrote deform_plate GINOT cache under {cache_root}")


if __name__ == "__main__":
    main()
