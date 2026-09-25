"""Dump visualization bundles for one dataset from a pair of trained runs.

Stage A of the manuscript figure pipeline. Two passes:

1. score the full test split -- per-sample scalars only, which is what backs
   the reproduction check, the case selection, and the error-distribution
   figure;
2. re-run only the handful of selected cases to get their predictions, and
   write one ``.npz`` per case plus a ``manifest.json``.

The split is deliberate. Scoring everything is cheap and is the thing that
proves the checkpoint reproduces the manuscript table; *retaining* everything is
not, since one AhmedML prediction is ~16 MB and only four get rendered.

Usage
-----
    python -m pdebench.vis.dump \
        --dataset elasticity \
        --runs flare=<exp_name> flarepp=<exp_name> \
        --reported flare=0.0064 flarepp=0.0038

    python -m pdebench.vis.dump --dataset nasa_crm --probe-connectivity

``--reported`` is optional but strongly recommended: it makes the run assert
that the checkpoint reproduces the manuscript table before any figure is drawn
from it. Values are fractions, not percentages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from mlutils.utils import dist_finalize, dist_setup, is_torchrun
from pdebench.vis import extract, select
from pdebench.vis.runner import (
    CASEDIR,
    PROJDIR,
    Run,
    case_roots,
    config_digest,
    git_sha,
    load_run,
    nasa_crm_connectivity,
)

DEFAULT_OUTDIR = CASEDIR / "vis_cache"


def _rank() -> int:
    return int(os.environ["RANK"]) if is_torchrun() else 0


def _is_main_rank() -> bool:
    return _rank() == 0


def _log(msg: str) -> None:
    if _is_main_rank():
        print(msg, flush=True)


def _parse_kv(pairs: list[str], what: str) -> dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--{what} expects key=value entries, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _autocast(run: Run, device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=run.mixed_precision and device.type == "cuda")


def _sample_metrics_from_phys(run: Run, yh_phys: torch.Tensor, y_phys: torch.Tensor) -> dict[str, float]:
    """Per-sample Rel-L2, reducing across the CP group when the mesh is sharded."""
    if run.cp_state is not None and run.cp_state.cp_size > 1:
        from pdebench.callbacks import (
            _reduce_surface_metric_sums,
            _surface_metric_sums,
            _surface_rel_l2_from_sums,
        )

        rels = _surface_rel_l2_from_sums(
            _reduce_surface_metric_sums(_surface_metric_sums(yh_phys, y_phys), run.cp_state)
        )
        return {"rel_l2": rels["full_rel_l2"], **rels}
    return select.sample_metrics(
        run.dataset,
        yh_phys[0].detach().cpu().numpy(),
        y_phys[0].detach().cpu().numpy(),
    )


@torch.no_grad()
def score_run(run: Run, device: torch.device, *, limit: int = 0) -> list[dict[str, float]]:
    """Per-sample metrics over the test split. Scalars only.

    Predictions are deliberately *not* retained: at 1M points and four fields a
    single AhmedML prediction is ~16 MB, so holding the split would cost many
    gigabytes of host RAM to serve the four cases that actually get rendered.
    ``predict_cases`` re-runs those few samples instead.

    ``limit`` truncates the split for smoke tests. It invalidates the
    reproduction check, so the caller must disable that gate when using it.
    """
    # Prefer a few worker processes when the allocation has spare CPUs; override
    # with VIS_NUM_WORKERS. Keep pin_memory on for CUDA so host→device copies overlap.
    num_workers = int(os.environ.get("VIS_NUM_WORKERS", "4"))
    loader = torch.utils.data.DataLoader(
        run.test_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    y_normalizer = run.metadata["y_normalizer"].to(device)

    records: list[dict[str, float]] = []
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        if run.cp_state is not None:
            from pdebench.distributed import shard_batch

            x, y = shard_batch((x, y), run.cp_state, seq_dim=run.cp_sequence_dim)
        with _autocast(run, device):
            yh = run.model(x)
        records.append(
            _sample_metrics_from_phys(
                run,
                y_normalizer.decode(yh.float()),
                y_normalizer.decode(y.float()),
            )
        )
        del x, y, yh
        if device.type == "cuda" and (index + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return records


@torch.no_grad()
def predict_cases(run: Run, indices: list[int], device: torch.device) -> dict[int, np.ndarray]:
    """Second pass: predictions in physical units for the selected cases only.

    Same model, same weights, same autocast setting as ``score_run``, so the
    rendered field is the scored field.
    """
    y_normalizer = run.metadata["y_normalizer"].to(device)
    out: dict[int, np.ndarray] = {}
    for index in indices:
        x, _ = run.test_data[index]
        x = x.unsqueeze(0).to(device)
        if run.cp_state is not None:
            from pdebench.distributed import gather_sequence_tensor, shard_sequence_tensor

            x = shard_sequence_tensor(x, run.cp_state, seq_dim=run.cp_sequence_dim)
            with _autocast(run, device):
                yh = run.model(x)
            yh = gather_sequence_tensor(yh.float(), run.cp_state, seq_dim=1)
        else:
            with _autocast(run, device):
                yh = run.model(x)
        out[index] = y_normalizer.decode(yh.float())[0].cpu().numpy().astype(np.float32)
        del x, yh
    return out


def probe_connectivity(data_root: str) -> None:
    """Print the NASA-CRM connectivity file's tree so we can map it once."""
    import h5py

    for candidate in (
        Path(data_root) / "NASA_CRM" / "connectivity_NASA-CRM.h5",
        Path(data_root) / "NASA-CRM" / "connectivity_NASA-CRM.h5",
        Path(data_root) / "connectivity_NASA-CRM.h5",
    ):
        if candidate.is_file():
            print(f"# {candidate}")
            # This file stores one 1D face dataset per cell under Connectivity/
            # (~4.5e5 entries). Do not visititems the whole tree — summarize.
            with h5py.File(candidate, "r") as handle:
                print(f"  top keys: {list(handle.keys())}")
                if "Connectivity" in handle and hasattr(handle["Connectivity"], "keys"):
                    group = handle["Connectivity"]
                    n_items = len(group)
                    print(f"  Connectivity/ n_items={n_items}")
                    shape_counts: dict[tuple[int, ...], int] = {}
                    samples = []
                    # Full scan of ~4.5e5 faces is slow; sample for the shape summary.
                    for i, (name, obj) in enumerate(group.items()):
                        if not hasattr(obj, "shape"):
                            continue
                        shape = tuple(obj.shape)
                        if i < 5000 or i % 50 == 0:
                            shape_counts[shape] = shape_counts.get(shape, 0) + 1
                        if len(samples) < 5:
                            samples.append((name, shape, str(obj.dtype), obj[()].tolist()))
                        if i >= 20000:
                            break
                    for name, shape, dtype, values in samples:
                        print(f"  sample Connectivity/{name}  {shape}  {dtype}  {values}")
                    print(
                        f"  shape_hist (sampled): "
                        f"{dict(sorted(shape_counts.items(), key=lambda kv: -kv[1]))}"
                    )
                    print("  note: no single [T,3]/[T,4] array; faces are per-cell 1D datasets")
                else:
                    handle.visititems(
                        lambda name, obj: print(
                            f"  {name}  {getattr(obj, 'shape', '')}  {getattr(obj, 'dtype', '')}"
                        )
                    )
            return
    print(f"connectivity_NASA-CRM.h5 not found under {data_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=extract.DATASETS)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[],
        metavar="MODEL=EXP",
        help="model label to experiment directory, e.g. flare=exp01 flarepp=exp02",
    )
    parser.add_argument(
        "--reported",
        nargs="*",
        default=[],
        metavar="MODEL=ERROR",
        help="manuscript test error as a fraction, for the reproduction check",
    )
    parser.add_argument("--key-model", default="flarepp", help="model whose error drives case selection")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--case-root",
        nargs="*",
        default=[],
        metavar="DIR",
        help=(
            "directories to search for experiment dirs, in order. Also read from "
            "PDEBENCH_CASE_ROOT (colon-separated); this repo's out/pdebench is always last. "
            "An entry in --runs may also be a full path to a case directory."
        ),
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--probe-connectivity", action="store_true")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="relative tolerance for the reproduction check (default 1%)",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="write bundles even if a checkpoint fails to reproduce its reported error",
    )
    parser.add_argument(
        "--extra-indices",
        type=int,
        nargs="*",
        default=[],
        help="additional test-split indices to bundle, on top of the selected cases",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N test samples (smoke test; disables the reproduction gate)",
    )
    parser.add_argument(
        "--cp-size",
        type=int,
        default=0,
        help="override context-parallel size (0 = use the run config)",
    )
    args = parser.parse_args()

    if is_torchrun():
        dist_setup()

    try:
        _run_dump(args)
    finally:
        dist_finalize()


def _run_dump(args) -> None:
    data_root = args.data_root or str(PROJDIR / "data")

    if args.probe_connectivity:
        if _is_main_rank():
            probe_connectivity(data_root)
        return

    if not args.runs:
        raise SystemExit("--runs is required unless --probe-connectivity is given")

    runs = _parse_kv(args.runs, "runs")
    reported = {k: float(v) for k, v in _parse_kv(args.reported, "reported").items()}
    if args.key_model not in runs:
        raise SystemExit(f"--key-model {args.key_model!r} is not among --runs {sorted(runs)}")

    if is_torchrun():
        device = torch.device(int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = select.primary_metric(args.dataset)
    roots = case_roots(args.case_root)
    _log(f"dataset={args.dataset} device={device} primary metric={metric}")
    _log("case roots:\n  " + "\n  ".join(str(root) for root in roots))

    scored: dict[str, dict[str, np.ndarray]] = {}
    loaded: dict[str, Run] = {}
    check_lines: list[str] = []
    mismatched = False

    # Pass 1 -- score the split. Scalars only; see score_run.
    for model, exp_name in runs.items():
        _log(f"\n--- {model}: {exp_name} ---")
        run = load_run(
            exp_name,
            device,
            data_root=args.data_root,
            roots=roots,
            cp_size=(args.cp_size or None),
        )
        if run.dataset != args.dataset:
            raise SystemExit(f"{exp_name} was trained on {run.dataset!r}, not {args.dataset!r}")
        _log(f"checkpoint: {run.ckpt_path}")
        if run.cp_state is not None:
            _log(f"context parallel: cp_size={run.cp_state.cp_size} seq_dim={run.cp_sequence_dim}")

        errors = select.per_sample_errors(score_run(run, device, limit=args.limit))
        scored[model] = errors
        loaded[model] = run

        if args.limit:
            _log(f"scored {errors[metric].size} of {len(run.test_data)} samples (--limit)")
        line = select.check_against_reported(
            args.dataset,
            model,
            metric,
            float(errors[metric].mean()),
            None if args.limit else reported.get(model),
            tol=args.tol,
        )
        _log(line)
        check_lines.append(line)
        if "MISMATCH" in line:
            mismatched = True

        run.model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if mismatched and not args.allow_mismatch:
        raise SystemExit(
            "\nA checkpoint does not reproduce its reported error. Refusing to write bundles.\n"
            "Resolve the discrepancy, or pass --allow-mismatch to write anyway (the manifest\n"
            "records the mismatch either way)."
        )

    key_errors = scored[args.key_model][metric]
    other = next((m for m in runs if m != args.key_model), None)
    cases = select.select_cases(key_errors, scored[other][metric] if other else None)
    for position, index in enumerate(args.extra_indices):
        if not 0 <= index < key_errors.size:
            raise SystemExit(f"--extra-indices {index} is outside the scored range [0, {key_errors.size})")
        cases[f"extra{position}"] = index
    _log(f"\nselected cases: {cases}")

    # Pass 2 -- predictions for the handful of cases that get rendered.
    indices = sorted(set(cases.values()))
    preds: dict[str, dict[int, np.ndarray]] = {}
    for model, run in loaded.items():
        run.model.to(device)
        preds[model] = predict_cases(run, indices, device)
        run.model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tri, tri_note = (None, "not applicable")
    if args.dataset == "nasa_crm":
        tri, tri_note = nasa_crm_connectivity(data_root)
        _log(f"connectivity: {tri_note}")

    if not _is_main_rank():
        return

    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    key_run = loaded[args.key_model]
    y_normalizer_cpu = key_run.metadata["y_normalizer"].to("cpu")
    written = []
    for label, index in cases.items():
        x, y = key_run.test_data[index]
        # UnitGaussianNormalizer.decode always returns a leading batch dim.
        y_ref = y_normalizer_cpu.decode(y.float().unsqueeze(0) if y.ndim == 2 else y.float())[0].numpy()
        bundle = extract.extract_bundle(
            args.dataset,
            x=x,
            y_ref=y_ref,
            predictions={model: preds[model][index] for model in runs},
            metadata=key_run.metadata,
            tri=tri,
        )
        path = outdir / f"{label}.npz"
        np.savez_compressed(path, **bundle)
        written.append(str(path))
        errs = {m: float(scored[m][metric][index]) for m in runs}
        print(f"  {label:11s} idx={index:4d}  {metric}: " + "  ".join(f"{m}={100 * e:.3f}%" for m, e in errs.items()))

    manifest = {
        "dataset": args.dataset,
        "primary_metric": metric,
        "key_model": args.key_model,
        "cases": cases,
        "selection_rule": {
            "median": "n//2-th order statistic of the key model's per-sample error",
            "max": "largest per-sample error of the key model",
            "best_gain": "largest (other - key) error difference",
            "worst_gain": "smallest (other - key) error difference",
            "extraN": "explicitly requested via --extra-indices",
        },
        "runs": {
            model: {
                "exp_name": exp_name,
                "case_dir": str(loaded[model].case_dir),
                "checkpoint": str(loaded[model].ckpt_path),
                "config_sha256_16": config_digest(loaded[model].case_dir),
                "mixed_precision": loaded[model].mixed_precision,
                "test_mean": {k: float(v.mean()) for k, v in scored[model].items()},
                "reported": reported.get(model),
            }
            for model, exp_name in runs.items()
        },
        "per_sample_errors": {
            model: {k: v.tolist() for k, v in errors.items()} for model, errors in scored.items()
        },
        "field_names": list(extract.field_names(args.dataset)),
        "n_test": int(len(key_run.test_data)),
        "n_scored": int(key_errors.size),
        "connectivity": tri_note,
        "reproduction_check": check_lines,
        "flare_dev_git_sha": git_sha(),
        "bundles": written,
    }
    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {len(written)} bundles and {manifest_path}")


if __name__ == "__main__":
    main()
