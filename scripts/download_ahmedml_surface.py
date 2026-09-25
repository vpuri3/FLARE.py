from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO = "neashton/ahmedml"
ALLOW_PATTERNS = ["run_*/boundary_*.vtp"]
DEFAULT_MAX_WORKERS = 32


def download_ahmedml_surface(data_root: Path, *, max_workers: int = DEFAULT_MAX_WORKERS) -> Path:
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(data_root),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=max_workers,
        resume_download=True,
    )
    return data_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download AhmedML surface boundary VTPs from Hugging Face.")
    p.add_argument("--data-root", type=Path, default=Path("data/AhmedML/raw"))
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = download_ahmedml_surface(args.data_root, max_workers=args.max_workers)
    print(f"Download complete under {root}")


if __name__ == "__main__":
    main()
