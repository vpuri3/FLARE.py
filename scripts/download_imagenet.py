#!/usr/bin/env python3
"""
Download ImageNet-1k dataset in WebDataset (tar) format for timm training.

This script downloads ImageNet-1k tar files from HuggingFace and stores them
in the directory structure expected by timm's WDS format.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import pathlib
import tarfile
from typing import List, Optional, Tuple

# IMPORTANT: Set cache paths BEFORE importing datasets/huggingface_hub
# These libraries read environment variables at import time
def set_cache_path_early(BASE_DIR: str):
    """Set cache paths before importing libraries that use them"""
    CACHE_BASE = os.path.join(BASE_DIR, "cache")
    env_vars = {
        "PIP_CACHE_DIR": os.path.join(CACHE_BASE, "pip"),
        "UV_CACHE_DIR": os.path.join(CACHE_BASE, "uv"),
        "XDG_CACHE_HOME": CACHE_BASE,
        "HF_HOME": os.path.join(CACHE_BASE, "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
        "TORCH_HOME": os.path.join(CACHE_BASE, "torch"),
        "WANDB_CACHE_DIR": os.path.join(CACHE_BASE, "wandb"),
        "TRITON_CACHE_DIR": os.path.join(CACHE_BASE, "triton"),
        "DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "MPLCONFIGDIR": os.path.join(CACHE_BASE, "matplotlib"),
        # Additional HuggingFace cache variables
        "HF_DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "HF_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
    }
    
    for var, path in env_vars.items():
        os.environ[var] = str(path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    return env_vars['DATASETS_CACHE'], env_vars['HF_HOME']

# Set cache paths immediately based on script location
dotdot = lambda dir: os.path.abspath(os.path.join(dir, '..'))
_script_dir = os.path.dirname(os.path.abspath(__file__))
_PROJDIR = dotdot(_script_dir)
_DATASETS_CACHE, _HF_HOME = set_cache_path_early(dotdot(_PROJDIR))

try:
    from tqdm.auto import tqdm
    from huggingface_hub import HfApi, hf_hub_download, login, whoami
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install tqdm huggingface_hub")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MAX_WORKERS = 32


def cap_workers(n: int) -> int:
    """Clamp worker count to [1, MAX_WORKERS]."""
    return max(1, min(int(n), MAX_WORKERS))

def get_available_workers() -> int:
    """Return all CPUs available to this process (Slurm/cpuset-aware when possible)."""
    slurm_cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus_on_node:
        try:
            return cap_workers(int(slurm_cpus_on_node))
        except ValueError:
            pass
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            return cap_workers(int(slurm_cpus_per_task))
        except ValueError:
            pass
    try:
        # Prefer affinity so we respect scheduler/cgroup CPU limits on shared nodes.
        return cap_workers(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return cap_workers(cpu_count())


def verify_tar_file(tar_path: str, *, require_image: bool = True) -> Tuple[bool, str]:
    """
    Verify a WebDataset shard tar for basic integrity.

    - Detects truncation / corruption (tarfile.ReadError, etc).
    - Optionally checks that at least one image file exists in the tar.

    Note: This does NOT fully validate per-sample completeness (e.g. every sample has an image),
    but it will reliably catch truncated shards like "unexpected end of data".
    """
    try:
        has_image = False
        # Iterate through the full archive to ensure end-of-archive can be read.
        with tarfile.open(tar_path, mode="r:*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = (m.name or "").lower()
                if name.endswith(IMG_EXTS):
                    has_image = True
        if require_image and not has_image:
            return False, "no_image_files_found"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def verify_tar_files(
    tar_paths: List[str],
    *,
    num_workers: int,
    require_image: bool = True,
):
    """Verify many tar files in parallel. Returns (ok_paths, bad_paths_with_reason)."""
    ok: List[str] = []
    bad: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_tar_file, p, require_image=require_image): p
            for p in tar_paths
        }
        with tqdm(total=len(tar_paths), desc="Verifying tars", unit="file") as pbar:
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    is_ok, reason = fut.result()
                except Exception as e:
                    is_ok, reason = False, f"{type(e).__name__}: {e}"
                if is_ok:
                    ok.append(p)
                else:
                    bad.append((p, reason))
                pbar.update(1)

    return ok, bad


def download_tar_file(repo_id, filename, local_dir, hf_token=None, retries=3):
    """Download a single tar file from HuggingFace"""
    local_path = os.path.join(local_dir, filename)
    
    # Skip if already downloaded
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size > 0:  # File exists and is not empty
            return True, local_path, file_size
    
    for attempt in range(retries):
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type='dataset',
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=hf_token,
                resume_download=True,
            )
            file_size = os.path.getsize(downloaded_path)
            return True, downloaded_path, file_size
        except Exception as e:
            if attempt == retries - 1:
                return False, None, 0
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False, None, 0


def download_tar_files(repo_id, tar_files, local_dir, hf_token=None, num_workers=None):
    """Download multiple tar files in parallel"""
    os.makedirs(local_dir, exist_ok=True)
    num_workers = cap_workers(num_workers or get_available_workers())
    
    print(f"Downloading {len(tar_files)} tar files to {local_dir}")
    print(f"Using {num_workers} parallel workers")
    print("")
    
    downloaded = 0
    skipped = 0
    failed = 0
    total_size = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks
        futures = {executor.submit(download_tar_file, repo_id, tar_file, local_dir, hf_token): tar_file 
                   for tar_file in tar_files}
        
        # Process completed downloads with progress bar
        with tqdm(total=len(tar_files), desc="Downloading tars", unit="file", unit_scale=True) as pbar:
            for future in as_completed(futures):
                tar_file = futures[future]
                try:
                    success, path, size = future.result()
                    if success:
                        if path and os.path.exists(path) and os.path.getsize(path) == size:
                            # File was already there
                            if size > 0:
                                skipped += 1
                                total_size += size
                            else:
                                downloaded += 1
                                total_size += size
                        else:
                            downloaded += 1
                            total_size += size
                    else:
                        failed += 1
                        print(f"\n⚠️  Failed to download: {tar_file}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'downloaded': downloaded,
                        'skipped': skipped,
                        'failed': failed,
                        'size': f'{total_size/1024/1024/1024:.1f}GB'
                    })
                except Exception as e:
                    failed += 1
                    print(f"\n⚠️  Error downloading {tar_file}: {e}")
                    pbar.update(1)
    
    elapsed = time.time() - start_time
    print("")
    print(f"✅ Download complete!")
    print(f"   Downloaded: {downloaded} files")
    print(f"   Skipped (already exists): {skipped} files")
    print(f"   Failed: {failed} files")
    print(f"   Total size: {total_size/1024/1024/1024:.2f} GB")
    print(f"   Time: {elapsed/60:.1f} minutes")
    
    return downloaded, skipped, failed


def get_hf_token_from_default_location():
    """Try to get HuggingFace token from default cache location"""
    # Default token locations (in order of preference)
    default_token_locations = [
        os.path.expanduser('~/.cache/huggingface/token'),
        os.path.expanduser('~/.huggingface/token'),
        os.path.expanduser('~/.cache/huggingface/hub/token'),
    ]
    
    # Try to read token from default locations
    for token_file in default_token_locations:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    if token and len(token) > 10:  # Basic validation
                        return token
            except Exception:
                pass
    
    return None


def check_hf_authentication(hf_token=None):
    """Check if user is authenticated with HuggingFace"""
    # If no token provided, try to get from default location
    if not hf_token:
        hf_token = get_hf_token_from_default_location()
    
    try:
        # Try to get current user info
        user_info = whoami(token=hf_token)
        if user_info:
            print(f"✅ Authenticated with HuggingFace as: {user_info.get('name', 'user')}")
            return True, hf_token
    except Exception:
        pass
    
    # If not authenticated, try to login with token
    if hf_token:
        try:
            login(token=hf_token)
            user_info = whoami()
            if user_info:
                print(f"✅ Authenticated with HuggingFace as: {user_info.get('name', 'user')}")
                return True, hf_token
        except Exception as e:
            print(f"⚠️  Failed to authenticate with provided token: {e}")
    
    print("❌ Not authenticated with HuggingFace!")
    print("")
    print("This dataset requires HuggingFace authentication.")
    print("Please authenticate using one of the following methods:")
    print("")
    print("Option 1: Use HuggingFace CLI")
    print("  Run: huggingface-cli login")
    print("  Or:  uv run hf login")
    print("")
    print("Option 2: Set environment variable")
    print("  export HF_TOKEN=your_token_here")
    print("  Get your token from: https://huggingface.co/settings/tokens")
    print("")
    print("Option 3: Pass token as argument")
    print("  python download_imagenet.py --hf-token your_token_here")
    print("")
    return False, None


def set_cache_path(BASE_DIR: str):
    """Set cache paths (called again in main for consistency)"""
    CACHE_BASE = os.path.join(BASE_DIR, "cache")
    env_vars = {
        "PIP_CACHE_DIR": os.path.join(CACHE_BASE, "pip"),
        "UV_CACHE_DIR": os.path.join(CACHE_BASE, "uv"),
        "XDG_CACHE_HOME": CACHE_BASE,
        "HF_HOME": os.path.join(CACHE_BASE, "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
        "TORCH_HOME": os.path.join(CACHE_BASE, "torch"),
        "WANDB_CACHE_DIR": os.path.join(CACHE_BASE, "wandb"),
        "TRITON_CACHE_DIR": os.path.join(CACHE_BASE, "triton"),
        "DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "MPLCONFIGDIR": os.path.join(CACHE_BASE, "matplotlib"),
        "HF_DATASETS_CACHE": os.path.join(CACHE_BASE, "datasets"),
        "HF_HUB_CACHE": os.path.join(CACHE_BASE, "huggingface"),
    }

    for var, path in env_vars.items():
        os.environ[var] = str(path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return env_vars['DATASETS_CACHE'], env_vars['HF_HOME']


def main():
    parser = argparse.ArgumentParser(description='Download ImageNet-1k dataset in WebDataset (tar) format')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Base data directory (default: $PROJ_DIR/data)')
    parser.add_argument('--imagenet-dir', type=str, default=None,
                       help='ImageNet directory (default: $DATA_DIR/imagenet)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help=f'Number of parallel workers (default: auto, max {MAX_WORKERS})')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace token (or use HF_TOKEN/HUGGING_FACE_HUB_TOKEN env var)')
    parser.add_argument('--skip-auth-check', action='store_true',
                       help='Skip HuggingFace authentication check (not recommended)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing/downloaded tar shards (detect truncation/corruption)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing tar shards; do not download anything')
    parser.add_argument('--force', action='store_true',
                       help='Delete existing tar shards before downloading (full re-download)')
    parser.add_argument('--repair-bad', action='store_true',
                       help='If verification finds bad shards, delete and re-download them')
    parser.add_argument('--verify-workers', type=int, default=None,
                       help=f'Parallel workers for verification (default: same as --num-workers, max {MAX_WORKERS})')
    parser.add_argument('--verify-require-image', action='store_true', default=True,
                       help='(default true) Also fail shards that contain no .jpg/.png/.webp members')
    
    args = parser.parse_args()

    # Determine project directory
    dotdot = lambda dir: os.path.abspath(os.path.join(dir, '..'))
    PROJDIR = dotdot(os.path.dirname(__file__))
    
    # Get token BEFORE setting custom HF_HOME (so we can read from default location)
    if not args.hf_token:
        args.hf_token = get_hf_token_from_default_location()
    
    # Now set cache paths
    DATASETS_CACHE_DIR, HF_HOME_DIR = set_cache_path(dotdot(PROJDIR))
    CACHE_BASE = os.path.join(dotdot(PROJDIR), "cache")
    
    print(f"Setting cache base to: {CACHE_BASE}")
    print(f"  DATASETS_CACHE: {DATASETS_CACHE_DIR}")
    print(f"  HF_HOME: {HF_HOME_DIR}")
    print("")
    
    # Verify cache directories are set (critical for avoiding home dir writes)
    assert os.environ.get('DATASETS_CACHE') == DATASETS_CACHE_DIR, \
        f"DATASETS_CACHE not set correctly! Expected: {DATASETS_CACHE_DIR}"
    assert os.environ.get('HF_HOME') == HF_HOME_DIR, \
        f"HF_HOME not set correctly! Expected: {HF_HOME_DIR}"
    
    # Verify we're not using home directory for caches
    home_dir = os.path.expanduser('~')
    if DATASETS_CACHE_DIR.startswith(home_dir):
        print(f"⚠️  WARNING: Cache directory is in home directory: {DATASETS_CACHE_DIR}")
        print("   This may cause disk quota issues. Consider using a different location.")
        print("")
    else:
        print(f"✅ Cache directories are outside home directory (good!)")
        print("")
    
    # Determine directories (honor CLI args)
    if args.imagenet_dir:
        IMAGENET_DIR = os.path.abspath(os.path.expanduser(args.imagenet_dir))
        DATADIR = os.path.abspath(os.path.join(IMAGENET_DIR, os.pardir))
    else:
        DATADIR = os.path.abspath(os.path.expanduser(args.data_dir)) if args.data_dir else os.path.join(PROJDIR, 'data')
        IMAGENET_DIR = os.path.join(DATADIR, 'imagenet')
    
    # Configuration
    num_workers = cap_workers(args.num_workers) if args.num_workers else get_available_workers()
    verify_workers = cap_workers(args.verify_workers) if args.verify_workers else num_workers
    # Use token from args (which may have been retrieved from default location)
    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    print("="*60)
    print("ImageNet-1k WebDataset (Tar) Download Script")
    print("="*60)
    print(f"Target directory: {IMAGENET_DIR}")
    print(f"Using {num_workers} workers for parallel downloads")
    if args.verify or args.verify_only:
        print(f"Verification enabled (workers={verify_workers}, repair_bad={args.repair_bad})")
    print(f"Cache directory: {DATASETS_CACHE_DIR}")
    print(f"HF_HOME: {HF_HOME_DIR}")
    print("")
    
    if args.verify_only:
        if not os.path.exists(IMAGENET_DIR):
            print(f"❌ ImageNet directory does not exist: {IMAGENET_DIR}")
            return 1
        existing_tars = sorted([str(p) for p in Path(IMAGENET_DIR).glob("*.tar")])
        if not existing_tars:
            print(f"❌ No .tar files found in: {IMAGENET_DIR}")
            return 1
        print(f"Found {len(existing_tars)} tar files. Starting verification...")
        ok, bad = verify_tar_files(
            existing_tars,
            num_workers=verify_workers,
            require_image=bool(args.verify_require_image),
        )
        print("")
        print("=== verification summary ===")
        print(f"ok:  {len(ok)}")
        print(f"bad: {len(bad)}")
        if bad:
            print("")
            print("Bad shards:")
            for p, reason in sorted(bad, key=lambda x: x[0]):
                print(f"  - {os.path.basename(p)} :: {reason}")

        if bad and args.repair_bad:
            print("")
            print("Repair requested: deleting bad shards...")
            repo_id = 'timm/imagenet-1k-wds'
            bad_names = [os.path.basename(p) for p, _ in bad]
            for p, _reason in bad:
                try:
                    os.remove(p)
                except OSError:
                    pass
            print("NOTE: --verify-only does not download. Re-run without --verify-only to repair.")

        return 0 if not bad else 2

    # Check HuggingFace authentication and get token
    if not args.skip_auth_check:
        print("Checking HuggingFace authentication...")
        is_authenticated, retrieved_token = check_hf_authentication(hf_token)
        if retrieved_token and not hf_token:
            # Use the token we found from default location
            hf_token = retrieved_token
            print(f"✅ Using token from default HuggingFace cache location")
        if not is_authenticated:
            print("\n⚠️  Authentication check failed. Attempting to continue anyway...")
            print("   If download fails, please authenticate using the instructions above.")
            print("")
        else:
            print("")
    
    # Check if tar files already exist
    print(f"Checking if tar files already exist at {IMAGENET_DIR}...")
    tar_files_exist = False
    if os.path.exists(IMAGENET_DIR):
        existing_tars = list(Path(IMAGENET_DIR).glob('*.tar'))
        if existing_tars:
            tar_files_exist = True
            print(f"✅ Found {len(existing_tars)} existing tar files")
            print(f"   Sample files: {existing_tars[0].name}, {existing_tars[1].name if len(existing_tars) > 1 else ''}")
            print("")
        if args.force and tar_files_exist:
            print("⚠️  --force specified: deleting existing tar shards for full re-download...")
            for p in existing_tars:
                try:
                    os.remove(p)
                except OSError:
                    pass
            tar_files_exist = False
    
    # Get list of tar files from HuggingFace
    repo_id = 'timm/imagenet-1k-wds'
    print(f"Fetching list of files from {repo_id}...")
    
    try:
        api = HfApi()
        all_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset', token=hf_token)
        tar_files = sorted([f for f in all_files if f.endswith('.tar')])
        info_files = [f for f in all_files if f.endswith('_info.json') or f == 'info.json']
        
        print(f"✅ Found {len(tar_files)} tar files in repository")
        print(f"   Training files: {len([f for f in tar_files if 'train' in f])}")
        print(f"   Validation files: {len([f for f in tar_files if 'val' in f])}")
        print(f"   Info files: {info_files}")
        print("")
        
        # Download info file first (required for timm WDS reader)
        if info_files:
            info_file = info_files[0]  # Usually '_info.json'
            info_path = os.path.join(IMAGENET_DIR, info_file)
            if not os.path.exists(info_path):
                print(f"Downloading info file: {info_file}...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=info_file,
                        repo_type='dataset',
                        local_dir=IMAGENET_DIR,
                        local_dir_use_symlinks=False,
                        token=hf_token,
                    )
                    print(f"✅ Downloaded info file: {info_file}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not download info file: {e}")
                    print("   The dataset may still work, but split information might be missing.")
            else:
                print(f"✅ Info file already exists: {info_file}")
            print("")
        
        if tar_files_exist:
            # Check which files need to be downloaded
            existing_tar_names = {f.name for f in Path(IMAGENET_DIR).glob('*.tar')}
            missing_tars = [f for f in tar_files if f not in existing_tar_names]
            if missing_tars:
                print(f"📥 Need to download {len(missing_tars)} missing tar files")
                tar_files = missing_tars
            else:
                print("✅ All tar files already downloaded!")
                print(f"Dataset is ready at: {IMAGENET_DIR}")
                # Do NOT return early here: if --verify is enabled we still want
                # to verify (and optionally repair) the existing shards.
                tar_files = []
        
        # Download tar files
        if tar_files:
            print("="*60)
            print("Downloading tar files")
            print("="*60)
            overall_start = time.time()
            
            downloaded, skipped, failed = download_tar_files(
                repo_id=repo_id,
                tar_files=tar_files,
                local_dir=IMAGENET_DIR,
                hf_token=hf_token,
                num_workers=num_workers
            )
            
            total_time = time.time() - overall_start
            print("")
            print("="*60)
            print("✅ ImageNet-1k tar download complete!")
            print("="*60)
            print(f"Dataset location: {IMAGENET_DIR}")
            print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
            print("")
            print("The dataset is now ready for use with timm's WDS format:")
            print(f"  --dataset 'wds/' --data-dir {IMAGENET_DIR}")
            print("")
            
            if failed > 0:
                print(f"⚠️  Warning: {failed} files failed to download. You may need to run the script again.")
                return 1

            if args.verify:
                print("")
                print("="*60)
                print("Verifying downloaded/existing tar files")
                print("="*60)
                existing_tars = sorted([str(p) for p in Path(IMAGENET_DIR).glob("*.tar")])
                ok, bad = verify_tar_files(
                    existing_tars,
                    num_workers=verify_workers,
                    require_image=bool(args.verify_require_image),
                )
                print("")
                print("=== verification summary ===")
                print(f"ok:  {len(ok)}")
                print(f"bad: {len(bad)}")
                if bad:
                    print("")
                    print("Bad shards:")
                    for p, reason in sorted(bad, key=lambda x: x[0]):
                        print(f"  - {os.path.basename(p)} :: {reason}")

                if bad and args.repair_bad:
                    print("")
                    print("Repair requested: deleting and re-downloading bad shards...")
                    bad_names = [os.path.basename(p) for p, _ in bad]
                    for p, _reason in bad:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    downloaded, skipped, failed = download_tar_files(
                        repo_id=repo_id,
                        tar_files=bad_names,
                        local_dir=IMAGENET_DIR,
                        hf_token=hf_token,
                        num_workers=num_workers
                    )
                    if failed > 0:
                        return 1
                    # Re-verify repaired shards
                    repaired_paths = [os.path.join(IMAGENET_DIR, n) for n in bad_names]
                    _, bad2 = verify_tar_files(
                        repaired_paths,
                        num_workers=verify_workers,
                        require_image=bool(args.verify_require_image),
                    )
                    if bad2:
                        print("")
                        print("❌ Some shards are still failing verification after repair:")
                        for p, reason in sorted(bad2, key=lambda x: x[0]):
                            print(f"  - {os.path.basename(p)} :: {reason}")
                        return 1
                elif bad and not args.repair_bad:
                    # Make verification failures obvious in exit code so batch scripts can catch it.
                    return 2
            
            return 0
        else:
            print("✅ All tar files are already downloaded!")
            if args.verify:
                existing_tars = sorted([str(p) for p in Path(IMAGENET_DIR).glob("*.tar")])
                if existing_tars:
                    ok, bad = verify_tar_files(
                        existing_tars,
                        num_workers=verify_workers,
                        require_image=bool(args.verify_require_image),
                    )
                    if bad:
                        print("")
                        print("❌ Verification found bad shards:")
                        for p, reason in sorted(bad, key=lambda x: x[0]):
                            print(f"  - {os.path.basename(p)} :: {reason}")
                        if args.repair_bad:
                            print("")
                            print("Repair requested: deleting and re-downloading bad shards...")
                            bad_names = [os.path.basename(p) for p, _ in bad]
                            for p, _reason in bad:
                                try:
                                    os.remove(p)
                                except OSError:
                                    pass
                            downloaded, skipped, failed = download_tar_files(
                                repo_id=repo_id,
                                tar_files=bad_names,
                                local_dir=IMAGENET_DIR,
                                hf_token=hf_token,
                                num_workers=num_workers
                            )
                            if failed > 0:
                                return 1
                            return 0
                        return 2
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("Partial download may be available. You can resume by running the script again.")
        return 1
    except Exception as e:
        import traceback
        print(f"\n❌ Error downloading from HuggingFace: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
