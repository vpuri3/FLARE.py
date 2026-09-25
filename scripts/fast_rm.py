#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def get_available_workers() -> int:
    slurm_cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus_on_node:
        try:
            return max(1, int(slurm_cpus_on_node))
        except ValueError:
            pass
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            return max(1, int(slurm_cpus_per_task))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast multi-threaded recursive remove for large cache trees.")
    p.add_argument("paths", nargs="+", help="Files or directories to remove.")
    p.add_argument("--workers", type=int, default=None, help="Worker threads (default: all available CPUs).")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    p.add_argument("--missing-ok", action="store_true", help="Ignore paths that do not exist.")
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Maximum queued unlink jobs (default: 16 * workers, minimum 256).",
    )
    return p.parse_args()


def remove_file(path: Path) -> int:
    try:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink()
        return 1
    except FileNotFoundError:
        return 0


def collect_paths(root: Path) -> tuple[list[Path], list[tuple[int, Path]]]:
    files: list[Path] = []
    dirs: list[tuple[int, Path]] = []
    if root.is_dir() and not root.is_symlink():
        for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
            base = Path(dirpath)
            files.extend(base / name for name in filenames)
            depth = len(base.parts)
            dirs.append((depth, base))
            for name in dirnames:
                child = base / name
                if child.is_symlink():
                    files.append(child)
    else:
        files.append(root)
    return files, dirs


def drain_done(done: set[cf.Future[int]], pbar) -> tuple[int, int]:
    removed = 0
    failed = 0
    for future in done:
        try:
            removed += int(future.result())
        except OSError:
            failed += 1
        if pbar:
            pbar.update(1)
    return removed, failed


def main() -> int:
    args = parse_args()
    workers = args.workers if args.workers is not None else get_available_workers()
    workers = max(1, int(workers))

    roots = [Path(os.path.expanduser(path)) for path in args.paths]
    missing = [path for path in roots if not path.exists()]
    if missing and not args.missing_ok:
        for path in missing:
            print(f"Missing path: {path}", file=sys.stderr)
        return 2

    files: list[Path] = []
    dirs: list[tuple[int, Path]] = []
    for root in roots:
        if root.exists():
            root_files, root_dirs = collect_paths(root)
            files.extend(root_files)
            dirs.extend(root_dirs)

    total = len(files) + len(dirs)
    if total == 0:
        print("Nothing to remove.")
        return 0
    if tqdm is None and not args.no_progress:
        print("tqdm is required for progress; pass --no-progress to disable it.", file=sys.stderr)
        return 3

    print(f"Using {workers} worker threads; removing {len(files)} file(s) and {len(dirs)} dir(s).")
    pbar = None if args.no_progress else tqdm(total=total, desc="remove", ncols=80)
    removed = 0
    failed = 0

    max_inflight = args.max_inflight if args.max_inflight is not None else max(256, workers * 16)
    futures: set[cf.Future[int]] = set()

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for path in files:
            futures.add(ex.submit(remove_file, path))
            if len(futures) >= max_inflight:
                done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                removed_delta, failed_delta = drain_done(done, pbar)
                removed += removed_delta
                failed += failed_delta
        while futures:
            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            removed_delta, failed_delta = drain_done(done, pbar)
            removed += removed_delta
            failed += failed_delta

    for _, path in sorted(dirs, reverse=True):
        try:
            path.rmdir()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            failed += 1
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    print(f"done: removed={removed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
