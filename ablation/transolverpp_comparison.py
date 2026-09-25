from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable
from urllib.request import urlopen

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdebench.models.transolver_plus import Physics_Attention_1D_Eidetic as LocalPhysicsAttention

OFFICIAL_URL = "https://raw.githubusercontent.com/thuml/Transolver_plus/main/models/Transolver_plus.py"
DEFAULT_OUTPUT = Path("out/pdebench/transolverpp_comparison.md")


@dataclass(frozen=True)
class ParamSummary:
    name: str
    shape: tuple[int, ...]
    count: int


@dataclass(frozen=True)
class TensorDiff:
    max_abs_diff: float
    mean_abs_diff: float
    rms_diff: float
    allclose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the local PDEBench Transolver++ PhysicsAttention block against the official GitHub version.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown report destination.")
    parser.add_argument("--official-url", type=str, default=OFFICIAL_URL, help="Official upstream Python source URL.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--tokens", type=int, default=17, help="Token count.")
    parser.add_argument("--dim", type=int, default=32, help="Model dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Attention head count.")
    parser.add_argument("--slice-num", type=int, default=8, help="Slice count.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Tensor dtype.")
    return parser.parse_args()


def fetch_official_source(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def load_official_module(source: str) -> ModuleType:
    with tempfile.TemporaryDirectory(prefix="transolverpp_official_") as tmpdir:
        source_path = Path(tmpdir) / "Transolver_plus.py"
        source_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("transolverpp_official", source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec for official Transolver++ source.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def patch_official_single_process(module: ModuleType) -> None:
    def _identity_all_reduce(tensor: torch.Tensor, op=None, group=None):
        del op, group
        return tensor

    module.dist_nn.all_reduce = _identity_all_reduce


def instantiate_attention(
    cls: type[torch.nn.Module],
    *,
    dim: int,
    heads: int,
    slice_num: int,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if dim % heads != 0:
        raise ValueError(f"dim={dim} must be divisible by heads={heads}")
    module = cls(dim=dim, heads=heads, dim_head=dim // heads, dropout=0.0, slice_num=slice_num).eval()
    return module.to(dtype=dtype)


def summarize_params(module: torch.nn.Module) -> list[ParamSummary]:
    return [
        ParamSummary(name=name, shape=tuple(param.shape), count=param.numel())
        for name, param in module.named_parameters()
    ]


def compare_tensors(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> TensorDiff:
    diff = (a - b).detach()
    abs_diff = diff.abs()
    max_abs_diff = abs_diff.max().item() if abs_diff.numel() else 0.0
    mean_abs_diff = abs_diff.mean().item() if abs_diff.numel() else 0.0
    rms_diff = math.sqrt(diff.pow(2).mean().item()) if diff.numel() else 0.0
    return TensorDiff(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rms_diff=rms_diff,
        allclose=torch.allclose(a, b, atol=atol, rtol=rtol),
    )


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    materialized_rows = [[str(cell) for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialized_rows)
    return "\n".join(lines)


def format_shape(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    atol = 1e-6 if dtype == torch.float32 else 1e-12
    rtol = 1e-5 if dtype == torch.float32 else 1e-12

    official_source = fetch_official_source(args.official_url)
    official_module = load_official_module(official_source)
    patch_official_single_process(official_module)
    OfficialPhysicsAttention = official_module.Physics_Attention_1D_Eidetic

    torch.manual_seed(args.seed)
    official_attention = instantiate_attention(
        OfficialPhysicsAttention,
        dim=args.dim,
        heads=args.heads,
        slice_num=args.slice_num,
        dtype=dtype,
    )
    torch.manual_seed(args.seed + 1)
    local_attention = instantiate_attention(
        LocalPhysicsAttention,
        dim=args.dim,
        heads=args.heads,
        slice_num=args.slice_num,
        dtype=dtype,
    )
    local_attention.load_state_dict(official_attention.state_dict(), strict=True)

    official_params = summarize_params(official_attention)
    local_params = summarize_params(local_attention)
    param_structure_match = official_params == local_params
    structure_repr_match = repr(official_attention) == repr(local_attention)

    torch.manual_seed(args.seed + 2)
    base_input = torch.randn(args.batch, args.tokens, args.dim, dtype=dtype)
    upstream_grad = torch.randn_like(base_input)

    official_input = base_input.clone().requires_grad_(True)
    local_input = base_input.clone().requires_grad_(True)

    torch.manual_seed(args.seed + 3)
    official_output = official_attention(official_input)
    torch.manual_seed(args.seed + 3)
    local_output = local_attention(local_input)
    output_diff = compare_tensors(local_output, official_output, atol=atol, rtol=rtol)

    official_attention.zero_grad(set_to_none=True)
    local_attention.zero_grad(set_to_none=True)
    official_input.grad = None
    local_input.grad = None

    official_output.backward(upstream_grad)
    local_output.backward(upstream_grad)

    input_grad_diff = compare_tensors(local_input.grad, official_input.grad, atol=atol, rtol=rtol)

    grad_rows = []
    overall_grad_max = 0.0
    overall_grad_mean_sum = 0.0
    grad_names_match = True
    official_named_grads = dict(official_attention.named_parameters())
    local_named_grads = dict(local_attention.named_parameters())
    if official_named_grads.keys() != local_named_grads.keys():
        grad_names_match = False

    for name in official_named_grads:
        official_grad = official_named_grads[name].grad
        local_grad = local_named_grads[name].grad
        grad_diff = compare_tensors(local_grad, official_grad, atol=atol, rtol=rtol)
        overall_grad_max = max(overall_grad_max, grad_diff.max_abs_diff)
        overall_grad_mean_sum += grad_diff.mean_abs_diff
        grad_rows.append(
            [
                name,
                format_shape(tuple(local_named_grads[name].shape)),
                f"{local_named_grads[name].numel()}",
                f"{grad_diff.max_abs_diff:.3e}",
                f"{grad_diff.mean_abs_diff:.3e}",
                f"{grad_diff.rms_diff:.3e}",
                str(grad_diff.allclose),
            ]
        )

    param_rows = [
        [summary.name, format_shape(summary.shape), summary.count]
        for summary in local_params
    ]

    report_lines = [
        "# Transolver++ PhysicsAttention Comparison",
        "",
        f"- Official source: `{args.official_url}`",
        f"- Local source: `pdebench/models/transolver_plus.py`",
        f"- Output report: `{args.output}`",
        f"- Device: `cpu`",
        f"- Dtype: `{args.dtype}`",
        f"- Seed: `{args.seed}`",
        f"- Input shape: `{tuple(base_input.shape)}`",
        "",
        "## Instantiated modules",
        "",
        "### Local PDEBench implementation",
        "```python",
        repr(local_attention),
        "```",
        "",
        "### Official GitHub implementation",
        "```python",
        repr(official_attention),
        "```",
        "",
        "## Structure checks",
        "",
        f"- `repr(local) == repr(official)`: `{structure_repr_match}`",
        f"- Matching named-parameter structure: `{param_structure_match}`",
        f"- Local parameter count: `{sum(item.count for item in local_params)}`",
        f"- Official parameter count: `{sum(item.count for item in official_params)}`",
        "",
        markdown_table(["Parameter", "Shape", "Count"], param_rows),
        "",
        "## Forward comparison",
        "",
        f"- Output shape: `{tuple(local_output.shape)}`",
        f"- `torch.allclose(local, official, atol={atol}, rtol={rtol})`: `{output_diff.allclose}`",
        f"- Max abs diff: `{output_diff.max_abs_diff:.3e}`",
        f"- Mean abs diff: `{output_diff.mean_abs_diff:.3e}`",
        f"- RMS diff: `{output_diff.rms_diff:.3e}`",
        "",
        "## Backward comparison",
        "",
        f"- Upstream gradient shape: `{tuple(upstream_grad.shape)}`",
        f"- Input gradient allclose: `{input_grad_diff.allclose}`",
        f"- Input grad max abs diff: `{input_grad_diff.max_abs_diff:.3e}`",
        f"- Input grad mean abs diff: `{input_grad_diff.mean_abs_diff:.3e}`",
        f"- Matching gradient parameter names: `{grad_names_match}`",
        f"- Max parameter-grad abs diff across all parameters: `{overall_grad_max:.3e}`",
        f"- Sum of per-parameter mean abs diffs: `{overall_grad_mean_sum:.3e}`",
        "",
        markdown_table(
            ["Parameter", "Shape", "Count", "Grad max abs diff", "Grad mean abs diff", "Grad RMS diff", "Allclose"],
            grad_rows,
        ),
        "",
        "## Notes",
        "",
        "- The official implementation unconditionally calls `torch.distributed.nn.all_reduce`; this script patches that call to an identity function so the single-process comparison matches the local non-context-parallel execution path.",
        "- Both forwards are run with the same RNG seed immediately before execution so the stochastic Gumbel-softmax path receives identical random samples.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {args.output}")


if __name__ == "__main__":
    main()
