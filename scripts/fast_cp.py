#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import os
import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def get_available_workers() -> int:
    """Return all CPUs available to this process (Slurm/cpuset-aware when possible)."""
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

def parse_args():
    p = argparse.ArgumentParser(
        description="Fast multi-threaded directory copy (IO-bound)."
    )
    p.add_argument("src", nargs="?", default="@data/imagenet",
                   help="Source directory (default: @data/imagenet)")
    p.add_argument("dst", nargs="?", default="/tmp/imagenet",
                   help="Destination directory (default: /tmp/imagenet)")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker threads (default: all available CPUs on this node)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip files that already exist with matching size")
    p.add_argument("--follow-symlinks", action="store_true",
                   help="Follow symlinks instead of copying them")
    return p.parse_args()

def get_size(path: Path, follow_symlinks: bool) -> int:
    st = path.stat() if follow_symlinks else path.lstat()
    return st.st_size

def should_skip(src_path: Path, dst_path: Path, follow_symlinks: bool) -> bool:
    if not dst_path.exists():
        return False
    try:
        return get_size(dst_path, follow_symlinks) == get_size(src_path, follow_symlinks)
    except OSError:
        return False

def copy_one(src_path: Path, dst_path: Path, follow_symlinks: bool, skip_existing: bool):
    if skip_existing and should_skip(src_path, dst_path, follow_symlinks):
        return 0
    size = get_size(src_path, follow_symlinks)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path, follow_symlinks=follow_symlinks)
    return size


def copy_tree(
    src: Path,
    dst: Path,
    *,
    workers: int | None = None,
    follow_symlinks: bool = False,
    skip_existing: bool = False,
) -> tuple[int, int, int]:
    """Copy a directory tree concurrently and return copied, skipped, failed counts."""
    workers = get_available_workers() if workers is None else max(1, int(workers))
    copied = 0
    skipped = 0
    failed = 0
    max_inflight = max(32, workers * 4)
    futures: set[cf.Future] = set()

    def collect(done: set[cf.Future]) -> None:
        nonlocal copied, skipped, failed
        for future in done:
            try:
                if future.result() > 0:
                    copied += 1
                else:
                    skipped += 1
            except Exception:
                failed += 1

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        for root, _dirs, files in os.walk(src, followlinks=follow_symlinks):
            relative_root = Path(root).relative_to(src)
            destination_root = dst / relative_root
            destination_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                source_path = Path(root) / name
                destination_path = destination_root / name
                futures.add(
                    executor.submit(copy_one, source_path, destination_path, follow_symlinks, skip_existing)
                )
                if len(futures) >= max_inflight:
                    done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                    collect(done)
        if futures:
            done, _pending = cf.wait(futures)
            collect(done)
    return copied, skipped, failed

def main():
    args = parse_args()
    src = Path(os.path.expanduser(args.src))
    dst = Path(os.path.expanduser(args.dst))

    if not src.exists():
        print(f"Source does not exist: {src}", file=sys.stderr)
        return 2
    if not src.is_dir():
        print(f"Source is not a directory: {src}", file=sys.stderr)
        return 2

    workers = args.workers
    if workers is None:
        workers = get_available_workers()

    print(f"Using {workers} worker threads")

    if tqdm is None and not args.no_progress:
        print("tqdm is required for the progress bar. Install it or pass --no-progress.", file=sys.stderr)
        return 3

    copied = 0
    skipped = 0
    failed = 0
    total = 0
    total_bytes = 0
    copied_bytes = 0

    if not args.no_progress:
        for root, dirs, files in os.walk(src, followlinks=args.follow_symlinks):
            rel_root = Path(root).relative_to(src)
            dst_root = dst / rel_root
            for name in files:
                s = Path(root) / name
                d = dst_root / name
                if args.skip_existing and should_skip(s, d, args.follow_symlinks):
                    continue
                try:
                    total_bytes += get_size(s, args.follow_symlinks)
                except OSError:
                    failed += 1

        if total_bytes == 0:
            print("Nothing to copy (all files already present or size 0).")
            return 0

        pbar = tqdm(
            total=total_bytes,
            ncols=80,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="copy",
        )
    else:
        pbar = None

    max_inflight = max(32, workers * 4)
    futures = set()

    def submit(executor, s: Path, d: Path):
        nonlocal total, total_bytes
        futures.add(executor.submit(copy_one, s, d, args.follow_symlinks, args.skip_existing))
        total += 1

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for root, dirs, files in os.walk(src, followlinks=args.follow_symlinks):
            rel_root = Path(root).relative_to(src)
            dst_root = dst / rel_root
            if not dst_root.exists():
                dst_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                s = Path(root) / name
                d = dst_root / name
                if args.skip_existing and should_skip(s, d, args.follow_symlinks):
                    skipped += 1
                    continue
                submit(ex, s, d)
                if len(futures) >= max_inflight:
                    done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        try:
                            bytes_copied = f.result()
                            if bytes_copied > 0:
                                copied += 1
                                copied_bytes += bytes_copied
                                if pbar:
                                    pbar.update(bytes_copied)
                            else:
                                skipped += 1
                        except Exception:
                            failed += 1
                    if pbar:
                        done_gb = copied_bytes / (1024 ** 3)
                        left_gb = max(0, total_bytes - copied_bytes) / (1024 ** 3)
                        pbar.set_postfix_str(f"GB done {done_gb:.2f} | left {left_gb:.2f}")

        while futures:
            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for f in done:
                try:
                    bytes_copied = f.result()
                    if bytes_copied > 0:
                        copied += 1
                        copied_bytes += bytes_copied
                        if pbar:
                            pbar.update(bytes_copied)
                    else:
                        skipped += 1
                except Exception:
                    failed += 1
            if pbar:
                done_gb = copied_bytes / (1024 ** 3)
                left_gb = max(0, total_bytes - copied_bytes) / (1024 ** 3)
                pbar.set_postfix_str(f"GB done {done_gb:.2f} | left {left_gb:.2f}")

    if pbar:
        pbar.close()

    print(f"done: total={total} copied={copied} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
