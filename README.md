# 🎇 FLARE: Fast Low-rank Attention Routing Engine

<p align="center">
<a href="http://arxiv.org/abs/2508.12594" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2508.12594-b31b1b.svg" /></a>
<a href="https://huggingface.co/papers/2508.12594" alt="HuggingFace">
    <img src="https://img.shields.io/badge/🤗_HuggingFace-2508.12594-ffbd00.svg" /></a>
</p>

## Abstract
The quadratic complexity of self-attention limits its applicability and scalability on large unstructured meshes.
We introduce **Fast Low-rank Attention Routing Engine (FLARE)**, a linear complexity self-attention mechanism that routes attention through fixed-length latent sequences.
Each attention head performs global communication among $N$ tokens by projecting the input sequence onto a fixed length latent sequence of $M \ll N$ tokens using learnable query tokens.
By routing attention through a bottleneck sequence, FLARE learns a low-rank form of attention that can be applied at $\mathcal{O}(NM)$ cost.
FLARE not only scales to unprecedented problem sizes, but also delivers superior accuracy compared to state-of-the-art neural PDE surrogates across diverse benchmarks.
We also release a new additive manufacturing dataset to spur further research.
Our code is available at
[https://github.com/vpuri3/FLARE.py](https://github.com/vpuri3/FLARE.py).

Please note that this repository is a cleaned-up version of the internal research repository we use. In case you encounter any problems with it, please don't hesitate to contact me.

## Highlights

- **Linear complexity token mixing.** FLARE is an efficient self-attention mechanism designed to learn on long sequences such as point clouds.
By replacing full self-attention with low-rank projections and reconstructions, FLARE achieves linear complexity in the number of points.
- **Superior accuracy.** Across multiple PDE benchmarks, FLARE achieves superior predictive accuracy compared to leading neural surrogate models, despite operating with fewer parameters, and at much lower computational complexity.
- **Scalability.** FLARE is built entirely from standard fused attention primitives, ensuring high GPU utilization and ease of integration into existing transformer architectures.
As such, FLARE enables end-to-end training on unstructured meshes with one million points (see Figure below) without distributed computing (Luo et al., 2025) or memory offloading – the largest scale demonstrated for transformer-based PDE surrogates.

## FLARE

FLARE is a simple yet powerful mechanism designed to break the scalability barrier in PDE surrogate learning.
FLARE is built on the argument that projecting input sequences onto shorter latent sequences, and then unprojecting to the original sequence length, is equivalent to constructing a low-rank form of attention with rank at most equal to the number of latent tokens (see figure below).

Furthermore, we argue that multiple simultaneous low-rank projections could collectively capture a full attention pattern.
Unlike Transolver which shares projection weights across heads, or LNO which applies only a single projection, our design allocates a distinct slice of the latent tokens to each head resulting in distinct projection matrices for each head.
This allows each head to learn independent attention relationships, opening up a key direction of scaling and exploration, wherein each head may specialize in distinct routing patterns.

<p align="center">
  <img src="figs/FLARE.png" alt="FLARE Architecture" width="100%">
</p>

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

## Benchmark dataset of additive manufacturing (AM) simulations.

We simulate the LPBF process on selected geometries from the Autodesk segementation dataset (Lambourne et al., 2021) to generate a benchmark dataset for AM calculations.
Several geometries are presented in this gallery.
The color indicates Z (vertical) displacement field.

<p align="center">
  <img src="figs/lpbf_gallery.jpg" alt="FLARE Architecture" width="100%">
</p>

## 🏗️ Codebase Architecture

This codebase implements the FLARE architecture and is built upon the [`mlutils.py`](https://github.com/vpuri3/mlutils.py/tree/master) framework, which provides foundational ML training infrastructure with multi-GPU support, extendable trainer classes, and callback systems.

The project is organized into several key packages:

### **`pdebench/`** - Main PDE Benchmarking Framework
- **Models**: Implementation of FLARE alongside state-of-the-art neural PDE surrogates
  - `flare.py`: Core FLARE architecture with linear complexity attention
  - `transolver.py`: Transolver baseline model
  - `lno.py`: Linear Neural Operator
  - `transformer.py`: Standard transformer architectures
  - `gnot.py`: Geometry-aware Neural Operator
  - `perceiver.py`: PerceiverIO architecture
- **Datasets**: Comprehensive PDE dataset loading and preprocessing
  - `utils.py`: Dataset utilities and transformations
- **Callbacks**: Training monitoring, evaluation, and visualization

#### **`am/`** - Additive Manufacturing Specialization
- **Models**: Specialized architectures for AM simulations
  - `meshGNN.py`: Graph neural networks for mesh data
- **Datasets**: LPBF (Laser Powder Bed Fusion) data processing
  - `sdf.py`: Signed distance function utilities
  - `extraction.py`: Feature extraction from AM simulations
  - `filtering.py`: Data filtering and preprocessing
- **Visualization**: 3D visualization tools for AM geometries

#### **`mlutils/`** - Core ML Framework (from mlutils.py)
- `trainer.py`: Distributed training with checkpointing and restart capabilities
- `callbacks.py`: Extensible callback system for monitoring and analysis
- `utils.py`: General ML utilities and helper functions

#### **`ablation/`** - Performance Analysis Suite
- Scaling experiments: `scale_dml.py`, `time_memory_*.py`
- Architecture ablations: `ablate_num_heads.py`, `ablate_num_layers.py`, `ablate_num_blocks.py`
- Memory and timing benchmarks with Flash Attention comparisons

### 🚀 Key Features

**Scalable Training Infrastructure**
- Multi-GPU/multi-node training with `torchrun`
- Automatic checkpointing and restart capabilities
- Mixed precision training (FP16/FP32)
- Comprehensive logging and monitoring

**Flexible Model Zoo**
- FLARE and many of the state-of-the-art neural PDE surrogates
- Modular architecture for easy experimentation

### 💻 Installation

Clone the repository and run the installation script:

```bash
git clone https://github.com/vpuri3/FLARE.py.git
cd FLARE.py
chmod +x scripts/install.sh
./scripts/install.sh
```

The installer will:
- Set up Python 3.11 virtual environment with `uv`
- Install PyTorch with CUDA support
- Install all required dependencies
- Optionally install Flash Attention for optimal performance
- Optionally install LaTeX for publication-quality plots


### 📊 Datasets

This codebase supports a variety of PDE datasets. You can download them using the built-in dataset utility:

```bash
git clone https://github.com/vpuri3/FLARE.py.git
cd FLARE.py
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

### 🎯 Usage

**Training**

Single GPU training:
```bash
uv run python -m pdebench --train true --dataset elasticity --exp_name flare_elas --model_type 2 --epochs 100 ...
```

Multi-GPU training:
```bash
uv run torchrun --nproc-per-node 2 -m pdebench --train true --dataset flare_darcy --exp_name flare_elasticity --model_type 2 --epochs 100 ...
```

Training hyperparameters can be modified with the following command-line arguments:

```
$ uv run python -m pdebench --help
usage: __main__.py [-h] [--config CONFIG] [--print_config[=flags]] [--train {true,false}]
                   [--evaluate {true,false}] [--restart {true,false}] [--exp_name EXP_NAME]
                   [--seed SEED] [--dataset DATASET] [--num_workers NUM_WORKERS] [--epochs EPOCHS]
                   [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY]
                   [--learning_rate LEARNING_RATE] [--schedule SCHEDULE]
                   [--one_cycle_pct_start ONE_CYCLE_PCT_START]
                   [--one_cycle_div_factor ONE_CYCLE_DIV_FACTOR]
                   [--one_cycle_final_div_factor ONE_CYCLE_FINAL_DIV_FACTOR]
                   [--one_cycle_three_phase {true,false}] [--opt_beta1 OPT_BETA1]
                   [--opt_beta2 OPT_BETA2] [--opt_eps OPT_EPS] [--clip_grad_norm CLIP_GRAD_NORM]
                   [--optimizer OPTIMIZER] [--mixed_precision {true,false}]
                   [--attn_backend ATTN_BACKEND] [--timing_only {true,false}] [--model_type MODEL_TYPE]
                   [--conv2d {true,false}] [--unified_pos {true,false}] [--act ACT]
                   [--channel_dim CHANNEL_DIM] [--num_blocks NUM_BLOCKS] [--num_heads NUM_HEADS]
                   [--num_latents NUM_LATENTS] [--num_layers_kv_proj NUM_LAYERS_KV_PROJ]
                   [--num_layers_mlp NUM_LAYERS_MLP] [--num_layers_in_out_proj NUM_LAYERS_IN_OUT_PROJ]
                   [--mlp_ratio MLP_RATIO] [--kv_proj_ratio KV_PROJ_RATIO]
                   [--in_out_proj_ratio IN_OUT_PROJ_RATIO] [--out_proj_ln {true,false}]
```

Each training run will create a directory in `out/pdebench` where it would store checkpoints.

```
$ tree out/pdebench/ -L 2
out/pdebench
├── flare_elas
│   ├── ckpt01
│   ├── ...
│   ├── ckpt10
│   ├── config.yaml
│   ├── grad_norm.png
│   ├── learning_rate.png
│   ├── losses.png
│   ├── rel_error.json
│   └── model_stats.json
└── flare_darcy
    ├── ckpt01
    ├── ...
    ├── ckpt10
    ├── config.yaml
    ├── grad_norm.png
    ├── learning_rate.png
    ├── losses.png
    ├── rel_error.json
    └── model_stats.json
```

**Evaluation**

Load and evaluate a trained model:
```bash
python -m pdebench --eval true --exp_name flare_elasticity
```

**Configuration**

All experiments are managed through YAML configuration files with comprehensive command-line override support. Results are automatically organized in the `out/` directory with:
- Model checkpoints
- Training logs and metrics
- Evaluation results and visualizations
- Configuration snapshots

### 📊 Datasets

**PDE Benchmarks**
- Supports multiple standard PDE benchmark datasets
- Scalable data loading for large mesh datasets
- Flexible preprocessing and augmentation pipelines

**Additive Manufacturing Dataset**
- New benchmark dataset with LPBF simulations
- Generated on Autodesk segmentation geometries
- Includes displacement fields and thermal histories

### 🔄 Reproducibility

We are committed to ensuring the reproducibility of our research results. Our main results can be reproduced by running the script:

```
chmod +x ./out/pdebench/run_comp.sh
./out/pdebench/run_comp.sh
```

### 🔬 Research Applications

- **Neural PDE Surrogates**: Fast approximation of expensive PDE solvers
- **Point Cloud Processing**: Large-scale geometric deep learning
- **Scientific Computing**: Scalable transformer architectures for irregular data

## Bibtex
```
@misc{puri2025flare,
      title={{FLARE}: {F}ast {L}ow-rank {A}ttention {R}outing {E}ngine}, 
      author={Vedant Puri and Aditya Joglekar and Kevin Ferguson and Yu-hsuan Chen and Yongjie Jessica Zhang and Levent Burak Kara},
      year={2025},
      eprint={2508.12594},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.12594}, 
}
```
