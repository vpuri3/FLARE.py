#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import select
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


@dataclass(frozen=True)
class SnapshotSpec:
    key: str
    repo_id: str
    local_subdir: str
    allow_patterns: list[str]
    expected_files: list[str]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    description: str
    default_yes: bool


@dataclass(frozen=True)
class ZenodoArchiveSpec:
    key: str
    record_id: str
    filename: str
    expected_paths: list[str]


@dataclass(frozen=True)
class MeshgraphnetsGcsSpec:
    """Official DeepMind MeshGraphNets GCS release (Pfaff et al., arXiv:2010.03409)."""

    key: str
    dataset_name: str
    local_subdir: str
    base_url: str
    expected_files: dict[str, int]


MESHGRAPHNETS_GCS_SPECS: dict[str, MeshgraphnetsGcsSpec] = {
    "deforming_plate": MeshgraphnetsGcsSpec(
        key="deforming_plate",
        dataset_name="deforming_plate",
        local_subdir="deforming_plate",
        base_url="https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/",
        expected_files={
            "meta.json": 878,
            "train.tfrecord": 9_906_860_823,
            "valid.tfrecord": 793_258_168,
            "test.tfrecord": 805_682_868,
        },
    ),
}


SNAPSHOT_SPECS: dict[str, SnapshotSpec] = {
    "bumper_beam": SnapshotSpec(
        key="bumper_beam",
        repo_id="AIRBORNEPANDA/BumperBeamCrashExample",
        local_subdir="bumper_beam",
        allow_patterns=[
            "CURATED_DATA_VTP/GLOBAL_FEATURES.json",
            "CURATED_DATA_VTP/TRAINING_DATA/*.vtp",
            "CURATED_DATA_VTP/VALIDATION_DATA/*.vtp",
        ],
        expected_files=["CURATED_DATA_VTP/GLOBAL_FEATURES.json"],
    ),
    "tensile2d": SnapshotSpec(
        key="tensile2d",
        repo_id="PLAID-datasets/Tensile2d",
        local_subdir=os.path.join("plaid", "Tensile2d"),
        allow_patterns=["README.md", "data/all_samples-*"],
        expected_files=[
            "README.md",
            "data/all_samples-00000-of-00002.parquet",
            "data/all_samples-00001-of-00002.parquet",
        ],
    ),
    "plaid_hyperelasticity": SnapshotSpec(
        key="plaid_hyperelasticity",
        repo_id="PLAID-datasets/2D_Multiscale_Hyperelasticity",
        local_subdir=os.path.join("plaid", "2D_Multiscale_Hyperelasticity"),
        allow_patterns=["README.md", "data/all_samples-*"],
        expected_files=[
            "README.md",
            "data/all_samples-00000-of-00002.parquet",
            "data/all_samples-00001-of-00002.parquet",
        ],
    ),
    "plaid_el_pl_dynamics": SnapshotSpec(
        key="plaid_el_pl_dynamics",
        repo_id="PLAID-datasets/2D_ElastoPlastoDynamics",
        local_subdir=os.path.join("plaid", "2D_ElastoPlastoDynamics"),
        allow_patterns=["README.md", "data/all_samples-*"],
        expected_files=["README.md"],
    ),
}


GINOT_ARCHIVE_SPECS: dict[str, ZenodoArchiveSpec] = {
    "poisson": ZenodoArchiveSpec(
        key="poisson",
        record_id="15293036",
        filename="poisson.zip",
        expected_paths=[
            "poisson/poisson_geo_struc_msh.pkl",
            "poisson/poisson_geo_unstruc_msh.pkl",
        ],
    ),
    "bracket_lug": ZenodoArchiveSpec(
        key="bracket_lug",
        record_id="15293036",
        filename="PLASTIC_LUG.zip",
        expected_paths=[
            "PLASTIC_LUG/input_params.npy",
            "PLASTIC_LUG/LUG_cells.pkl",
            "PLASTIC_LUG/LUG_node_S_PC.pkl",
        ],
    ),
    "micro_puc": ZenodoArchiveSpec(
        key="micro_puc",
        record_id="15121966",
        filename="PeriodUnitCell.zip",
        expected_paths=[
            "PeriodUnitCell/mises_disp_laststep.pkl",
            "PeriodUnitCell/points_cloud.pkl",
            "PeriodUnitCell/mesh_coords.pkl",
            "PeriodUnitCell/mesh_cells10K.pkl",
            "PeriodUnitCell/sample_ids.npy",
        ],
    ),
}


GINOT_TASK_TO_ARCHIVE = {
    "poisson_unstructured": "poisson",
    "bracket_lug": "bracket_lug",
    "micro_puc": "micro_puc",
}


DRIVAERML_POINT_COUNTS = ["10k", "40k", "50k", "100k", "200k", "300k", "400k", "500k", "1m"]
DRIVAERML_OTHER_POINT_COUNTS = ["10k", "50k", "100k", "200k", "300k", "400k", "500k"]


def get_tasks() -> list[TaskSpec]:
    tasks: list[TaskSpec] = [
        TaskSpec(key="fno", description="FNO", default_yes=True),
        TaskSpec(key="geo-fno", description="Geo-FNO", default_yes=True),
        TaskSpec(key="drivaerml-40k", description="DrivAerML 40k", default_yes=True),
        TaskSpec(key="drivaerml-1m", description="DrivAerML 1m", default_yes=True),
        TaskSpec(
            key="drivaerml-other",
            description="DrivAerML (10k, 50k, 100k, 200k, 300k, 400k, 500k)",
            default_yes=False,
        ),
    ]

    tasks.extend(
        [
            TaskSpec(key="shapenet-car", description="ShapeNet-Car", default_yes=False),
            TaskSpec(key="bumper_beam", description="PhysicsNeMo/OpenRadioss bumper-beam crash VTP", default_yes=False),
            TaskSpec(key="tensile2d", description="PLAID Tensile2d", default_yes=True),
            TaskSpec(key="plaid_hyperelasticity", description="PLAID 2D Multiscale Hyperelasticity", default_yes=True),
            TaskSpec(
                key="plaid_el_pl_dynamics",
                description="PLAID 2D ElastoPlastoDynamics (static mesh GLT / legacy GINOT)",
                default_yes=False,
            ),
            TaskSpec(key="poisson_unstructured", description="GINOT Poisson unstructured mesh", default_yes=True),
            TaskSpec(key="bracket_lug", description="GINOT bracket lug", default_yes=True),
            TaskSpec(key="micro_puc", description="GINOT micro-PUC", default_yes=True),
            TaskSpec(
                key="deforming_plate",
                description="MeshGraphNets deforming_plate (DeepMind GCS / Pfaff et al. 2010.03409)",
                default_yes=False,
            ),
        ]
    )
    return tasks


def parse_args(task_keys: list[str]):
    proj_dir = Path(__file__).resolve().parent.parent
    machine = socket.gethostname()
    if machine == "eagle":
        # VDEL Eagle - 1 node: 4x 2080Ti 11 GB
        default_data_root = "/mnt/hdd1/vedantpu/data/"
    else:
        default_data_root = str(proj_dir / "data")

    parser = argparse.ArgumentParser(
        description=(
            "Unified dataset downloader. Prompts for each selected dataset with [Y/n] or [y/N] defaults and "
            "auto-selects the default after 30 seconds."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=task_keys + ["all"],
        help="Dataset key(s) to consider. If omitted, all datasets are considered.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="Root directory where datasets are stored.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Prompt timeout for each dataset. Uses per-dataset default on timeout.",
    )
    return parser.parse_args()


def configure_cache_dirs(proj_dir: Path):
    cache_base = (proj_dir.parent / "cache").resolve()
    print(f"Setting cache base to: {cache_base}")

    env_map = {
        "PIP_CACHE_DIR": cache_base / "pip",
        "UV_CACHE_DIR": cache_base / "uv",
        "XDG_CACHE_HOME": cache_base,
        "HF_HOME": cache_base / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_base / "huggingface",
        "TORCH_HOME": cache_base / "torch",
        "WANDB_CACHE_DIR": cache_base / "wandb",
        "TRITON_CACHE_DIR": cache_base / "triton",
        "DATASETS_CACHE": cache_base / "datasets",
        "MPLCONFIGDIR": cache_base / "matplotlib",
        "HF_DATASETS_CACHE": cache_base / "datasets",
        "HF_HUB_CACHE": cache_base / "huggingface",
    }

    for key, path in env_map.items():
        os.environ[key] = str(path)

    for path in set(env_map.values()):
        path.mkdir(parents=True, exist_ok=True)


def verify_snapshot_dataset(dst_dir: Path, spec: SnapshotSpec) -> list[str]:
    missing = []
    for rel_path in spec.expected_files:
        abs_path = dst_dir / rel_path
        if not abs_path.exists():
            missing.append(rel_path)
    if spec.key == "bumper_beam":
        train_vtps = list((dst_dir / "CURATED_DATA_VTP" / "TRAINING_DATA").glob("*.vtp"))
        validation_vtps = list((dst_dir / "CURATED_DATA_VTP" / "VALIDATION_DATA").glob("*.vtp"))
        total_vtps = len(train_vtps) + len(validation_vtps)
        if total_vtps != 131:
            missing.append(
                "expected 131 VTP files across CURATED_DATA_VTP/TRAINING_DATA and "
                f"CURATED_DATA_VTP/VALIDATION_DATA, found {total_vtps}"
            )
    return missing


def prompt_yes_no(description: str, default_yes: bool, timeout_seconds: int) -> bool:
    prompt = "[Y/n]" if default_yes else "[y/N]"
    default_label = "yes" if default_yes else "no"
    sys.stdout.write(
        f"Download {description}? {prompt} (auto-default: {default_label} in {timeout_seconds}s): "
    )
    sys.stdout.flush()

    # Keep waiting until timeout even if select reports spurious readability.
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print()
            print(f"  -> timed out after {timeout_seconds}s; using default '{default_label}'")
            return default_yes

        try:
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
        except (OSError, ValueError):
            ready = []

        if not ready:
            continue

        raw = sys.stdin.readline()
        if raw == "":
            # EOF/no real input event; keep waiting until timeout.
            continue

        answer = raw.strip().lower()
        if answer == "":
            return default_yes
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print(f"  -> unrecognized response '{answer}'; using default '{default_label}'")
        return default_yes


def hf_download_workers(environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    return max(1, int(env.get("PDEBENCH_HF_DOWNLOAD_WORKERS", "16")))


def download_snapshot_dataset(data_root: Path, spec: SnapshotSpec):
    dst_dir = data_root / spec.local_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{spec.key}]")
    print(f"  repo: {spec.repo_id}")
    print(f"  dst : {dst_dir}")

    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        local_dir=str(dst_dir),
        allow_patterns=spec.allow_patterns,
        max_workers=hf_download_workers(),
    )

    missing = verify_snapshot_dataset(dst_dir, spec)
    if missing:
        raise FileNotFoundError(f"Download incomplete for {spec.key}. Missing files: {missing}")

    if spec.key == "bumper_beam":
        print("  status: OK (GLOBAL_FEATURES.json + 131 curated VTPs present)")
    else:
        print(f"  status: OK ({len(spec.expected_files)} required files present)")


def verify_expected_paths(data_root: Path, expected_paths: list[str]) -> list[str]:
    return [rel_path for rel_path in expected_paths if not (data_root / rel_path).exists()]


def verify_meshgraphnets_gcs_dataset(dst_dir: Path, spec: MeshgraphnetsGcsSpec) -> list[str]:
    missing: list[str] = []
    for filename, expected_size in spec.expected_files.items():
        path = dst_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            missing.append(f"{filename} (expected {expected_size} bytes, got {actual_size})")
    return missing


def _download_meshgraphnets_gcs_file(url: str, dst: Path) -> None:
    if shutil.which("wget"):
        subprocess.run(
            ["wget", "-c", "-O", str(dst), url],
            check=True,
        )
        return

    print("  wget not found; falling back to urllib (no resume)")
    urllib.request.urlretrieve(url, dst)


def download_meshgraphnets_gcs_dataset(data_root: Path, spec: MeshgraphnetsGcsSpec):
    dst_dir = data_root / spec.local_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{spec.key}]")
    print("  source: DeepMind MeshGraphNets GCS (Pfaff et al., arXiv:2010.03409)")
    print(f"  bucket: {spec.base_url}")
    print(f"  dst   : {dst_dir}")
    print(
        "  refs  : https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets"
        " | https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/deforming_plate"
    )

    missing_before = verify_meshgraphnets_gcs_dataset(dst_dir, spec)
    if not missing_before:
        print(f"  status: OK ({len(spec.expected_files)} files present with expected sizes)")
        return

    for filename, expected_size in spec.expected_files.items():
        dst = dst_dir / filename
        if dst.is_file() and dst.stat().st_size == expected_size:
            print(f"  skip : {filename} ({expected_size} bytes)")
            continue

        url = f"{spec.base_url.rstrip('/')}/{filename}"
        print(f"  fetch: {filename} ({expected_size} bytes)")
        _download_meshgraphnets_gcs_file(url, dst)
        actual_size = dst.stat().st_size
        if actual_size != expected_size:
            raise FileNotFoundError(
                f"Download incomplete for {spec.key}/{filename}: "
                f"expected {expected_size} bytes, got {actual_size}"
            )

    missing_after = verify_meshgraphnets_gcs_dataset(dst_dir, spec)
    if missing_after:
        raise FileNotFoundError(f"Download incomplete for {spec.key}. Missing or wrong-size files: {missing_after}")
    print(f"  status: OK ({len(spec.expected_files)} files present with expected sizes)")


def download_zenodo_archive(data_root: Path, spec: ZenodoArchiveSpec):
    missing_before = verify_expected_paths(data_root, spec.expected_paths)
    print(f"\n[{spec.key}]")
    print(f"  zenodo record: {spec.record_id}")
    print(f"  file         : {spec.filename}")
    print(f"  dst          : {data_root}")
    if not missing_before:
        print(f"  status       : OK ({len(spec.expected_paths)} required files already present)")
        return

    archive_path = data_root / spec.filename
    url = f"https://zenodo.org/records/{spec.record_id}/files/{spec.filename}?download=1"
    print(f"  downloading  : {url}")
    urllib.request.urlretrieve(url, archive_path)

    print(f"  extracting   : {archive_path.name} -> {data_root}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(data_root)
    archive_path.unlink(missing_ok=True)

    missing_after = verify_expected_paths(data_root, spec.expected_paths)
    if missing_after:
        raise FileNotFoundError(f"Download incomplete for {spec.key}. Missing files: {missing_after}")
    print(f"  status       : OK ({len(spec.expected_paths)} required files present)")


def extract_tar_gz(archive_path: Path, dst_dir: Path):
    print(f"  extracting {archive_path.name} -> {dst_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)


def hf_download_archive(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            resume_download=True,
        )
    )
    return archive_path


def download_drivaerml_variant(data_root: Path, points: str):
    dst_dir = data_root / "DrivAerML"
    filename = f"drivaerml_surface_presampled_{points}.tar.gz"
    print(f"\n[drivaerml-{points}]")
    print("  repo: vedantpuri/PDESurrogates")
    print(f"  file: {filename}")
    print(f"  dst : {dst_dir}")

    archive_path = hf_download_archive(
        repo_id="vedantpuri/PDESurrogates",
        filename=filename,
        local_dir=dst_dir,
    )
    extract_tar_gz(archive_path, dst_dir)
    archive_path.unlink(missing_ok=True)
    print("  status: OK")


def download_pd_surrogates_archive(data_root: Path, key: str, filename: str):
    print(f"\n[{key}]")
    print("  repo: vedantpuri/PDESurrogates")
    print(f"  file: {filename}")
    print(f"  dst : {data_root}")

    archive_path = hf_download_archive(
        repo_id="vedantpuri/PDESurrogates",
        filename=filename,
        local_dir=data_root,
    )
    extract_tar_gz(archive_path, data_root)
    archive_path.unlink(missing_ok=True)
    print("  status: OK")


def download_shapenet_car(data_root: Path):
    dst_dir = data_root / "ShapeNet-Car"
    dst_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip"
    zip_path = dst_dir / "mlcfd_data.zip"

    print("\n[shapenet-car]")
    print(f"  url: {zip_url}")
    print(f"  dst: {dst_dir}")

    print("  downloading zip archive...")
    urllib.request.urlretrieve(zip_url, zip_path)

    print("  extracting zip archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

    macosx_dir = dst_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    training_dir = dst_dir / "mlcfd_data" / "training_data"
    for idx in range(9):
        tar_name = f"param{idx}.tar.gz"
        tar_path = training_dir / tar_name
        if tar_path.exists():
            extract_tar_gz(tar_path, training_dir)

    for bad_path in [
        "param2/854bb96a96a4d1b338acbabdc1252e2f",
        "param2/85bb9748c3836e566f81b21e2305c824",
        "param5/9ec13da6190ab1a3dd141480e2c154d3",
        "param8/c5079a5b8d59220bc3fb0d224baae2a",
    ]:
        target = training_dir / bad_path
        if target.exists():
            shutil.rmtree(target)

    if zip_path.exists():
        zip_path.unlink()

    for idx in range(9):
        tar_path = training_dir / f"param{idx}.tar.gz"
        if tar_path.exists():
            tar_path.unlink()

    print("  status: OK")


def list_tree(root_dir: Path, depth: int = 3):
    root_dir = root_dir.resolve()
    print(f"\nDirectory summary under {root_dir}:")
    if not root_dir.exists():
        print("  (missing)")
        return

    for current, dirs, _ in os.walk(root_dir):
        current_path = Path(current)
        rel = current_path.relative_to(root_dir)
        level = 0 if rel == Path(".") else len(rel.parts)
        if level > depth:
            dirs[:] = []
            continue

        indent = "  " * level
        name = "." if rel == Path(".") else current_path.name
        print(f"{indent}{name}/")


def run_task(task: TaskSpec, data_root: Path):
    if task.key == "fno":
        download_pd_surrogates_archive(data_root, key="fno", filename="FNO.tar.gz")
        return

    if task.key == "geo-fno":
        download_pd_surrogates_archive(data_root, key="geo-fno", filename="Geo-FNO.tar.gz")
        return

    if task.key == "drivaerml-other":
        for points in DRIVAERML_OTHER_POINT_COUNTS:
            download_drivaerml_variant(data_root, points)
        return

    if task.key.startswith("drivaerml-"):
        points = task.key.split("drivaerml-", maxsplit=1)[1]
        if points not in DRIVAERML_POINT_COUNTS:
            raise ValueError(f"Unknown DrivAerML point count: {points}")
        download_drivaerml_variant(data_root, points)
        return

    if task.key == "shapenet-car":
        download_shapenet_car(data_root)
        return

    if task.key in GINOT_TASK_TO_ARCHIVE:
        archive_key = GINOT_TASK_TO_ARCHIVE[task.key]
        download_zenodo_archive(data_root, GINOT_ARCHIVE_SPECS[archive_key])
        return

    if task.key in SNAPSHOT_SPECS:
        download_snapshot_dataset(data_root, SNAPSHOT_SPECS[task.key])
        return

    if task.key in MESHGRAPHNETS_GCS_SPECS:
        download_meshgraphnets_gcs_dataset(data_root, MESHGRAPHNETS_GCS_SPECS[task.key])
        return

    raise ValueError(f"Unknown task key: {task.key}")


def resolve_selected_tasks(args, tasks: list[TaskSpec]) -> list[TaskSpec]:
    task_by_key = {task.key: task for task in tasks}
    if "all" in args.dataset:
        return tasks

    selected = []
    seen = set()
    for key in args.dataset:
        if key in seen:
            continue
        seen.add(key)
        selected.append(task_by_key[key])
    return selected


def main():
    tasks = get_tasks()
    task_keys = [task.key for task in tasks]
    args = parse_args(task_keys)

    proj_dir = Path(__file__).resolve().parent.parent
    configure_cache_dirs(proj_dir)

    selected_tasks = resolve_selected_tasks(args, tasks)
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Using data root: {data_root}")
    print("Selected dataset candidates:")
    for task in selected_tasks:
        default_label = "yes" if task.default_yes else "no"
        print(f"  - {task.key} (default: {default_label})")

    downloaded = []
    skipped = []

    for task in selected_tasks:
        should_download = prompt_yes_no(
            description=task.description,
            default_yes=task.default_yes,
            timeout_seconds=args.timeout_seconds,
        )
        if not should_download:
            print(f"  -> skipping {task.key}")
            skipped.append(task.key)
            continue

        run_task(task, data_root)
        downloaded.append(task.key)

    print("\nSummary:")
    print(f"  downloaded ({len(downloaded)}): {', '.join(downloaded) if downloaded else '(none)'}")
    print(f"  skipped    ({len(skipped)}): {', '.join(skipped) if skipped else '(none)'}")
    list_tree(data_root, depth=4)


if __name__ == "__main__":
    main()
