# FLARE.py: FLARE and Experimental Attention Models

<p align="center">
<a href="http://arxiv.org/abs/2508.12594" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2508.12594-b31b1b.svg" /></a>
<a href="https://huggingface.co/papers/2508.12594" alt="HuggingFace">
    <img src="https://img.shields.io/badge/🤗_HuggingFace-2508.12594-ffbd00.svg" /></a>
</p>

This repository is centered on **FLARE (Fast Low-rank Attention Routing Engine)** and also serves as a sandbox for efficient and higher-order attention models across:

- `pdebench/`: neural PDE surrogate training/evaluation
- `lra/`: Long Range Arena style sequence modeling

## FLARE

FLARE is a low-rank attention routing mechanism that keeps global communication while reducing token-mixing cost from quadratic to near-linear in sequence length. Instead of constructing a full `N x N` attention map, each head routes information through a fixed set of latent tokens (`M << N`) via a gather-scatter attention pattern implemented with fused SDPA primitives. In practice, this gives strong scaling behavior on large unstructured meshes while preserving the flexibility of attention-based models.

## Highlights

- **FLARE-first repo** with reproducible PDEBench and LRA training workflows.
- **Broad experimental model zoo** for linear, low-rank, and higher-order attention ideas.
- **Performance-oriented kernels**, including custom Triton implementations in `lra/models/triton/`.

FLARE exhibits excellent scaling and can tackle problems with millions of tokens on a single GPU.
We present time and memory requirements of different attention schemes.
On an input sequence of one million tokens, FLARE (red) is over $200\times$ faster than vanilla attention, while consuming marginally more memory.
All models are implemented with flash attention (Dao et al., 2022), and the memory upper bound on a single H100 80GB GPU is depicted with a dashed line.
Note that the curves for FLARE are somewhat overlapping.

<p align="center">
  <img src="figs/time_memory_bwd.png" alt="FLARE scaling" width="100%">
</p>

The implementation of FLARE is straightforward and employs highly optimized fused self-attention kernels.

```python
import torch.nn.functional as F
def flare_multihead_mixer(q, k, v):
    """
    Arguments:
    q: Query tensor [H, M, D]
    k: Key tensor [B, H, N, D]
    v: Value tensor [B, H, N, D]
    Returns:
    y: Output tensor [B, H, N, D]
    """

    z = F.scaled_dot_product_attention(q, k, v, scale=1.0)
    y = F.scaled_dot_product_attention(k, q, z, scale=1.0)

    return y
```

## Blog posts

Detailed write-ups of FLARE and related attention mechanisms:

- [Scaling attention to 1M tokens on a single GPU](https://vpuri3.github.io/blog/scaling-attention-to-1m-tokens-on-a-single-gpu/) — the FLARE gather–scatter mechanism, PDE benchmark results, and scaling analysis.
- [From Encoder to Decoder: Extending FLARE to Memory-Efficient Causal Attention](https://vpuri3.github.io/blog/from-encoder-to-decoder-extending-flare-to-memory-efficient-causal-attention/) — causal FLARE for language modeling: recurrent decode, stable prefill, and training/inference tradeoffs.
- [Higher-Order Attention in Linear Time](https://vpuri3.github.io/blog/adventures-in-high-order-attention/) — linear attention bottlenecks, multilinear memories, Strassen-style mixing, and triple/quad attention.
- [Triple Attention in Triton](https://vpuri3.github.io/blog/triple-attention-in-triton-building-a-third-order-memory-in-linear-time/) — third-order memory in linear time, with a fused Triton kernel compared to the einsum reference.

## Benchmark dataset of additive manufacturing (AM) simulations.

We simulate the LPBF process on selected geometries from the Autodesk segementation dataset (Lambourne et al., 2021) to generate a benchmark dataset for AM calculations.
Several geometries are presented in this gallery.
The color indicates Z (vertical) displacement field.

<p align="center">
  <img src="figs/lpbf_gallery.jpg" alt="FLARE Architecture" width="100%">
</p>

## 🏗️ Codebase Architecture

This codebase implements the FLARE architecture and is built upon the [`mlutils.py`](https://github.com/vpuri3/mlutils.py/tree/master) framework, which provides foundational ML training infrastructure with multi-GPU support, extendable trainer classes, and callback systems.

- `pdebench/`: FLARE and baseline/experimental PDE surrogate models
- `lra/`: transformer backends for long-context tasks, including multiple linear-attention families
- `ablation/`: timing/memory/scaling scripts
- `am/`: additive manufacturing data/model tooling
- `mlutils/`: shared training utilities and trainer/callback infrastructure

## Model Zoo

| Model type | Available in | Status | Notes | Citation / Source |
|---|---|---|---|---|
| `flare` | `pdebench`, `lra` | Stable | Main FLARE backend | FLARE paper: https://arxiv.org/abs/2508.12594 |
| `transformer` | `pdebench`, `lra` | Stable | Vanilla softmax self-attention baseline | Transformer: https://arxiv.org/abs/1706.03762 |
| `linformer` | `pdebench`, `lra` | Stable | Sequence-projected efficient attention | Linformer: https://arxiv.org/abs/2006.04768 |
| `linear` | `pdebench`, `lra` | Stable | Kernelized linear attention baseline | Linear Transformers: https://arxiv.org/abs/2006.16236 |
| `flare_experimental` | `pdebench` | Experimental | FLARE variant with alternative projection stack | In-repo experimental model |
| `flare_ablations` | `pdebench` | Experimental | Large/ablation-oriented FLARE variants | In-repo ablation model |
| `transolver` | `pdebench` | Stable | PDE transformer baseline | Transolver: https://arxiv.org/abs/2402.02366 |
| `transolver++` | `pdebench` | Stable | Transolver++ implementation path | Upstream impl: https://github.com/thuml/Transolver_plus |
| `lno` | `pdebench` | Stable | Latent Neural Operator baseline | Upstream impl: https://github.com/L-I-M-I-T/LatentNeuralOperator |
| `gnot` | `pdebench` | Stable | Geometry-aware neural operator baseline | Upstream impl: https://github.com/HaoZhongkai/GNOT |
| `perceiverio` | `pdebench` | Stable | Latent cross-attention baseline | Perceiver IO: https://arxiv.org/abs/2107.14795 |
| `lamo` | `pdebench` | Stable (optional deps) | Latent SSM-style model | Upstream impl: https://github.com/M3RG-IITD/LaMO |
| `upt` | `pdebench` | Experimental / WIP | Placeholder in current CLI (`NotImplemented`) | UPT tutorial source: https://github.com/BenediktAlkin/upt-tutorial |
| `loopy` | `pdebench` | Experimental (code present) | Currently disabled in CLI path | In-repo experimental model |
| `unloopy` | `pdebench` | Experimental (code present) | Currently disabled in CLI path | In-repo experimental model |
| `performer` | `lra` | Stable | FAVOR+ random-feature attention | Performer: https://arxiv.org/abs/2009.14794 |
| `multilinear` | `lra` | Experimental | Higher-order multilinear state backend | In-repo experimental backend |
| `triple` | `lra` | Experimental | Third-order state mixer (`use_triton` optional) | In-repo experimental backend |
| `quad` | `lra` | Experimental | Fourth-order multilinear attention backend | In-repo experimental backend |
| `strassen` | `lra` | Experimental | Strassen-style structured higher-order backend | In-repo experimental backend |
| `third_order` | `lra` | Experimental | Third-order attention variants | In-repo experimental backend |
| `ema` | `lra` | Experimental | MEGA-style EMA + gated attention block | MEGA: https://arxiv.org/abs/2209.10655 |

## Linear Attention Frameworks in This Repo

The repo intentionally spans multiple efficient-attention families:

- **Low-rank latent routing**: FLARE (`flare`) routes through latent tokens with SDPA gather-scatter.
- **Sequence projection**: Linformer (`linformer`) compresses token dimension before attention.
- **Kernelized state-based linear attention**: `linear` and `performer` use associative state accumulation to avoid `N x N` attention maps.
- **Higher-order multilinear attention**: `multilinear`, `triple`, `third_order` (plus `quad`/`strassen`) explore simplex/multilinear memory states.
- **EMA-gated hybrids**: `ema` combines moving-average memory with attention/gating.

These design directions are discussed in detail in the blog post:

- **Higher-Order Attention in Linear Time: Multilinear Memories and Simplex Mixing**  
  https://vpuri3.github.io/blog/adventures-in-high-order-attention/

The post frames the following tradeoff: linear-time models are efficient because they compress interactions into feature-space states, but this can weaken token-specific routing compared to softmax attention. The higher-order variants in this repo are aligned with that exploration.

## Triton Implementations (`lra/models/triton/`)

- `lra/models/triton/triple.py`:
  - Fused Triton path for triple/higher-order style attention primitives.
  - Implements tiled/chunked state construction and output passes.
  - Includes custom autograd backward path for the fused kernels.

- `lra/models/triton/causal_linear.py`:
  - Fused causal linear attention implementation.
  - Uses chunked KV-state construction + prefix-state accumulation + fused chunk output.
  - Designed for high-throughput linear-time causal attention.

These kernels are experimental performance paths; keep a non-Triton fallback in workflows where portability is needed.

## Installation

```bash
git clone https://github.com/vpuri3/FLARE.py.git
cd FLARE.py
chmod +x scripts/install.sh
./scripts/install.sh
```

What this script does:

- creates/uses a Python 3.11 `uv` environment
- installs CUDA-specific PyTorch
- installs project dependencies via `uv sync --extra dev --extra test`
- optionally installs PyG / vision / TensorFlow / AM-SDF extras
- optionally installs FlashAttention and LaTeX tooling

## Datasets

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

This script downloads supported PDEBench-related datasets into `data/`.

## Usage

### PDEBench

Train:

```bash
uv run python -m pdebench \
  --train true \
  --dataset elasticity \
  --exp_name flare_elas \
  --model_type flare \
  --epochs 100
```

Evaluate:

```bash
uv run python -m pdebench \
  --evaluate true \
  --exp_name flare_elas
```

### LRA

Train:

```bash
uv run python -m lra \
  --train true \
  --task listops \
  --exp_name lra_flare \
  --model_type flare
```

Evaluate:

```bash
uv run python -m lra \
  --evaluate true \
  --exp_name lra_flare
```

## Reproducibility Notes

- Outputs are stored under `out/pdebench/` and `out/lra/`.
- Config snapshots are saved per experiment directory.
- For model-specific flags:

```bash
uv run python -m pdebench --help
uv run python -m lra --help
```

## Cite FLARE

```bibtex
@misc{puri2025flarefastlowrankattention,
      title={FLARE: Fast Low-rank Attention Routing Engine},
      author={Vedant Puri and Yichi Zhang and Yuze Zhang and Carl E. Rasmussen and Jinkyoo Park and Xiaoyu Song and C. Karen Liu and Tarek I. Zohdi and Somdatta Goswami},
      year={2025},
      eprint={2508.12594},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.12594}
}
```
