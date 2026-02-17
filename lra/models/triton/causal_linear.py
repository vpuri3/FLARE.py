#
# Causal Linear Implementation with Triton Fused Kernel
#
# Usage:
#   out = CausalLinear.apply(Q, K, V)
#
#
import math
import os

import torch
import torch.nn.functional as F
import torch.autograd as autograd
from einops import rearrange

import triton
import triton.language as tl
import triton.testing

#======================================================================#
# Causal Linear Implementation
#======================================================================#
class CausalLinear(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        """
        Forward pass of causal Linear.

        Args:
            Q: [B, H, N, Dk] - queries from input sequence
            K: [B, H, N, Dk] - keys from input sequence  
            V: [B, H, N, Dv] - values from input sequence

        Returns:
            Y: [B, H, N, Dv] - output sequence
        """
        B, H, N, Dk, Dv = *Q.size(), V.size(-1)
        scale = N ** -0.5

        assert Q.size() == K.size(), f"Q, K must have the same shape. Got Q.shape={Q.shape}, K.shape={K.shape}"
        assert V.size() == (B, H, N, Dv), f"V must have shape for {B, H, N, Dv}. Got V.shape={V.shape}"

        device = Q.device
        dtype = Q.dtype
        BH = B * H

        O = torch.empty((B, H, N, Dv), device=device, dtype=dtype)
        O.zero_()

        # BLOCK_D should be at least max(Dk, Dv) and a power of 2, rounded up to next multiple of 16
        BLOCK_D = max(Dk, Dv, 16)

        if BLOCK_D % 16 != 0:
            BLOCK_D = ((BLOCK_D + 15) // 16) * 16

        CHUNK_N = 64

        # Ensure block sizes are positive
        assert BLOCK_D > 0, f"BLOCK_D must be positive. Got BLOCK_D={BLOCK_D}."
        assert BLOCK_D >= Dk and BLOCK_D >= Dv
        assert CHUNK_N > 0, "Chunk size must be positive."

        assert BLOCK_D % 16 == 0, f"BLOCK_D must be divisible by 16. Got BLOCK_D={BLOCK_D}."
        assert CHUNK_N % 16 == 0, f"CHUNK_N must be divisible by 16. Got CHUNK_N={CHUNK_N}."

        # Flatten batch and heads to launch Triton kernels over BH
        Qf = Q.reshape(B * H, N, Dk).contiguous()
        Kf = K.reshape(B * H, N, Dk).contiguous()
        Vf = V.reshape(B * H, N, Dv).contiguous()
        Of = O.view(BH, N, Dv)

        Q_stride = Qf.stride()
        K_stride = Kf.stride()
        V_stride = Vf.stride()
        O_stride = Of.stride()

        # Ensure N is divisible by CHUNK_N (no padding needed for now)
        assert N % CHUNK_N == 0, f"N ({N}) must be divisible by CHUNK_N ({CHUNK_N})"
        NUM_CHUNKS = N // CHUNK_N
        
        #---------------------------------------------------------------#
        # Phase 1: Compute chunk-wise S matrices: KVc = Kc.mT @ Vc
        #---------------------------------------------------------------#
        # Reshape into chunks: [BH, NUM_CHUNKS, CHUNK_N, Dk/Dv]
        K_chunks = Kf.view(BH, NUM_CHUNKS, CHUNK_N, Dk).contiguous() * scale
        V_chunks = Vf.view(BH, NUM_CHUNKS, CHUNK_N, Dv).contiguous() * scale

        # Allocate output tensors
        KVc = torch.zeros((BH, NUM_CHUNKS, Dk, Dv), device=device, dtype=torch.float32)
        Yc = torch.zeros((BH, NUM_CHUNKS, CHUNK_N, Dv), device=device, dtype=torch.float32)

        # Strides for chunked tensors
        Kc_stride = K_chunks.stride()
        Vc_stride = V_chunks.stride()
        KVc_stride = KVc.stride()
        
        # Launch Phase 1 kernel: Compute KVc = Kc.mT @ Vc
        grid_kv = (BH, NUM_CHUNKS)
        linear_chunk_kv[grid_kv](
            K_chunks, V_chunks, KVc,
            Kc_stride[0], Kc_stride[1], Kc_stride[2], Kc_stride[3],
            Vc_stride[0], Vc_stride[1], Vc_stride[2], Vc_stride[3],
            KVc_stride[0], KVc_stride[1], KVc_stride[2], KVc_stride[3],
            BH, NUM_CHUNKS, Dk, Dv,
            BLOCK_D=BLOCK_D,
            CHUNK_N=CHUNK_N,
        )
        
        #---------------------------------------------------------------#
        # Phase 2: Compute prefix S matrices (cumsum, then shift)
        #---------------------------------------------------------------#
        Sc = KVc.cumsum(dim=1)
        Sc = torch.cat([torch.zeros_like(Sc[:, :1]), Sc[:, :-1]], dim=1)
        Sc = Sc.contiguous()
        Sc_stride = Sc.stride()

        #---------------------------------------------------------------#
        # Phase 3: Fused inter- and intra-chunk contribution computation
        #---------------------------------------------------------------#
        Q_chunks = Qf.view(BH, NUM_CHUNKS, CHUNK_N, Dk).contiguous()
        Qc_stride = Q_chunks.stride()
        Yc_stride = Yc.stride()

        use_tensor_cores = Q_chunks.dtype in (torch.float16, torch.bfloat16)
        grid_out = (BH, NUM_CHUNKS)
        linear_chunk_out[grid_out](
            Q_chunks, K_chunks, V_chunks, Sc, Yc,
            Qc_stride[0], Qc_stride[1], Qc_stride[2], Qc_stride[3],
            Kc_stride[0], Kc_stride[1], Kc_stride[2], Kc_stride[3],
            Vc_stride[0], Vc_stride[1], Vc_stride[2], Vc_stride[3],
            Sc_stride[0], Sc_stride[1], Sc_stride[2], Sc_stride[3],
            Yc_stride[0], Yc_stride[1], Yc_stride[2], Yc_stride[3],
            BH, NUM_CHUNKS, Dk, Dv,
            BLOCK_D=BLOCK_D,
            CHUNK_N=CHUNK_N,
            USE_TENSOR_CORES=use_tensor_cores,
        )

        #---------------------------------------------------------------#
        # Phase 4: Reshape back to [B, H, N, Dv]
        #---------------------------------------------------------------#
        O = Yc.to(dtype).view(BH, N, Dv).view(B, H, N, Dv)  # [B, H, N, Dv]

        O = O.view(B, H, N, Dv)
        return O

#======================================================================#
# Triton forward kernels
#======================================================================#
@triton.jit
def linear_chunk_kv(
    Kc_ptr, Vc_ptr, KVc_ptr,
    stride_kc_bh, stride_kc_chunk, stride_kc_n, stride_kc_dk,
    stride_vc_bh, stride_vc_chunk, stride_vc_n, stride_vc_dv,
    stride_kvc_bh, stride_kvc_chunk, stride_kvc_dk, stride_kvc_dv,
    BH, NUM_CHUNKS, Dk, Dv,
    BLOCK_D: tl.constexpr,
    CHUNK_N: tl.constexpr,
):
    """
    Phase 1: Compute chunk-wise S matrices: KVc = Kc.mT @ Vc
    Each kernel handles one (bh, chunk) pair and computes [Dk, Dv] output.
    """
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH or chunk_idx >= NUM_CHUNKS:
        return

    dk_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_D)
    n_offsets = tl.arange(0, CHUNK_N)
    
    mask_dk = dk_offsets < Dk
    mask_dv = dv_offsets < Dv
    mask_n = n_offsets < CHUNK_N
    mask_kv = mask_dk[:, None] & mask_dv[None, :]
    mask_k_entries = mask_n[:, None] & mask_dk[None, :]
    mask_v_entries = mask_n[:, None] & mask_dv[None, :]

    # Base pointers for this batch/head and chunk
    Kc_base = Kc_ptr + pid_bh * stride_kc_bh + chunk_idx * stride_kc_chunk
    Vc_base = Vc_ptr + pid_bh * stride_vc_bh + chunk_idx * stride_vc_chunk

    # Load K chunk: [CHUNK_N, Dk]
    Kc = tl.load(
        Kc_base + n_offsets[:, None] * stride_kc_n + dk_offsets[None, :] * stride_kc_dk,
        mask=mask_k_entries,
        other=0.0,
    ).to(tl.float32)  # [CHUNK_N, BLOCK_D]

    # Load V chunk: [CHUNK_N, Dv]
    Vc = tl.load(
        Vc_base + n_offsets[:, None] * stride_vc_n + dv_offsets[None, :] * stride_vc_dv,
        mask=mask_v_entries,
        other=0.0,
    ).to(tl.float32)  # [CHUNK_N, BLOCK_D]

    # Compute KVc = Kc.mT @ Vc: [Dk, CHUNK_N] @ [CHUNK_N, Dv] = [Dk, Dv]
    KVc = tl.dot(tl.trans(Kc), Vc)  # [BLOCK_D, BLOCK_D]

    # Store output: [Dk, Dv]
    KVc_base = KVc_ptr + pid_bh * stride_kvc_bh + chunk_idx * stride_kvc_chunk
    tl.store(
        KVc_base + dk_offsets[:, None] * stride_kvc_dk + dv_offsets[None, :] * stride_kvc_dv,
        KVc.to(KVc_ptr.dtype.element_ty),
        mask=mask_kv,
    )


@triton.jit
def linear_chunk_out(
    Qc_ptr, Kc_ptr, Vc_ptr, Sc_ptr, Yc_ptr,
    stride_qc_bh, stride_qc_chunk, stride_qc_n, stride_qc_dk,
    stride_kc_bh, stride_kc_chunk, stride_kc_n, stride_kc_dk,
    stride_vc_bh, stride_vc_chunk, stride_vc_n, stride_vc_dv,
    stride_sc_bh, stride_sc_chunk, stride_sc_dk, stride_sc_dv,
    stride_yc_bh, stride_yc_chunk, stride_yc_n, stride_yc_dv,
    BH, NUM_CHUNKS, Dk, Dv,
    BLOCK_D: tl.constexpr,
    CHUNK_N: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    """
    Phase 3: Fused kernel that loads each chunk of Q, K, V once,
    applies the prefix state (Sc) and causal intra-chunk computation,
    and writes the final chunk output.
    """
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH or chunk_idx >= NUM_CHUNKS:
        return

    dk_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_D)
    n_offsets = tl.arange(0, CHUNK_N)

    mask_dk = dk_offsets < Dk
    mask_dv = dv_offsets < Dv
    mask_n = n_offsets < CHUNK_N
    mask_q_entries = mask_n[:, None] & mask_dk[None, :]
    mask_v_entries = mask_n[:, None] & mask_dv[None, :]
    mask_sc = mask_dk[:, None] & mask_dv[None, :]
    mask_y_entries = mask_n[:, None] & mask_dv[None, :]

    Qc_base = Qc_ptr + pid_bh * stride_qc_bh + chunk_idx * stride_qc_chunk
    Kc_base = Kc_ptr + pid_bh * stride_kc_bh + chunk_idx * stride_kc_chunk
    Vc_base = Vc_ptr + pid_bh * stride_vc_bh + chunk_idx * stride_vc_chunk
    Sc_base = Sc_ptr + pid_bh * stride_sc_bh + chunk_idx * stride_sc_chunk

    Qc = tl.load(
        Qc_base + n_offsets[:, None] * stride_qc_n + dk_offsets[None, :] * stride_qc_dk,
        mask=mask_q_entries,
        other=0.0,
    )  # [CHUNK_N, BLOCK_D]

    Sc = tl.load(
        Sc_base + dk_offsets[:, None] * stride_sc_dk + dv_offsets[None, :] * stride_sc_dv,
        mask=mask_sc,
        other=0.0,
    )  # [BLOCK_D, BLOCK_D]

    if USE_TENSOR_CORES:
        Qc_inter = Qc
        Sc_inter = Sc.to(Qc.dtype)
    else:
        Qc_inter = Qc.to(tl.float32)
        Sc_inter = Sc.to(tl.float32)

    Yc_inter = tl.dot(Qc_inter, Sc_inter, out_dtype=tl.float32)  # [CHUNK_N, BLOCK_D]

    Qc_fp32 = Qc.to(tl.float32)
    Kc = tl.load(
        Kc_base + n_offsets[:, None] * stride_kc_n + dk_offsets[None, :] * stride_kc_dk,
        mask=mask_q_entries,
        other=0.0,
    ).to(tl.float32)  # [CHUNK_N, BLOCK_D]
    Vc = tl.load(
        Vc_base + n_offsets[:, None] * stride_vc_n + dv_offsets[None, :] * stride_vc_dv,
        mask=mask_v_entries,
        other=0.0,
    ).to(tl.float32)  # [CHUNK_N, BLOCK_D]

    QKc = tl.dot(Qc_fp32, tl.trans(Kc))  # [CHUNK_N, CHUNK_N]
    causal_mask = n_offsets[:, None] >= n_offsets[None, :]
    QKc = tl.where(causal_mask, QKc, 0.0)
    Yc_intra = tl.dot(QKc, Vc)  # [CHUNK_N, BLOCK_D]

    Yc_total = Yc_inter + Yc_intra

    Yc_out_base = Yc_ptr + pid_bh * stride_yc_bh + chunk_idx * stride_yc_chunk
    tl.store(
        Yc_out_base + n_offsets[:, None] * stride_yc_n + dv_offsets[None, :] * stride_yc_dv,
        Yc_total.to(Yc_ptr.dtype.element_ty),
        mask=mask_y_entries,
    )

#======================================================================#
# PyTorch Implementations
#======================================================================#

def causal_SDPA(Q, K, V):
    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    return Y

def linear_noncausal(Q, K, V):
    B, H, N, Dk, Dv = *K.size(), V.size(-1)
    assert Q.size() == K.size(), f"Q, K must have the same shape. Got Q.shape={Q.shape}, K.shape={K.shape}"

    scale = N ** -0.5
    K = K * scale
    V = V * scale

    S = K.mT @ V  # [B, H, Dk, Dv]
    Y = Q @ S     # [B, H, N, Dv]
    return Y

def linear_causal_reference(Q, K, V):
    B, H, N, Dk, Dv = *K.size(), V.size(-1)
    assert Q.size() == K.size()

    scale = N ** -0.5

    Y = torch.zeros_like(V)
    S = torch.zeros((B, H, Dv, Dk), device=Q.device, dtype=torch.float32)

    for t in range(N):
        qt = Q[:, :, t, :] # [B, H, Dk]
        kt = K[:, :, t, :].to(torch.float32) * scale # [B, H, Dk]
        vt = V[:, :, t, :].to(torch.float32) * scale # [B, H, Dv]

        S += vt.view(B, H, Dv, 1) @ kt.view(B, H, 1, Dk) # vt @ kt.mT
        yt = (S @ qt.view(B, H, Dk, 1)).view(B, H, Dv)

        Y[:, :, t, :] = yt

    return Y

def linear_causal_pytorch(Q, K, V):
    """
    PyTorch implementation matching linear_causal_reference exactly.
    """
    B, H, N, Dk, Dv = *K.size(), V.size(-1)
    assert Q.size() == K.size()

    CHUNK_SIZE = 128
    NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)
    scale = N ** -0.5

    K = K.to(torch.float32) * scale
    V = V.to(torch.float32) * scale

    # [B H NUM_CHUNKS CHUNK_SIZE D]
    Qc, Kc, Vc = [rearrange(Z, 'b h (n c) d -> b h n c d', c=CHUNK_SIZE) for Z in [Q, K, V]]

    ##############################################################

    ###
    # Compute chunk-wise S matrices
    ###

    VKc = Vc.mT @ Kc # [B H NUM_CHUNKS Dv Dk]
    # Accumulate prefix S matrices
    Sc = VKc.cumsum(dim=2) # [B H NUM_CHUNKS Dv Dk]
    # prepend 0 to Sc for first chunk and discard last chunk
    Sc = torch.cat([torch.zeros_like(Sc[:, :, :1]), Sc[:, :, :-1]], dim=2) # [B H NUM_CHUNKS Dk Dv]

    ###
    # at this point, Sc contains the prefix S matrices for each chunk
    # so we can compute contribution of all previous chunks
    ###

    Yc_inter = Qc @ Sc.to(Q.dtype).mT # [B H NUM_CHUNKS CHUNK_SIZE Dv]

    ###
    # now we compute intra-chunk contribution for each chunk like
    # Y = (Q @ K^T) @ V
    # with a causal mask applied to the upper triangular part of the matrix
    ###

    A_intra = Qc.to(torch.float32) @ Kc.mT # [B H NUM_CHUNKS CHUNK_SIZE CHUNK_SIZE]
    A_intra = A_intra.masked_fill_(torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=bool, device=Q.device), diagonal=1), 0.)
    Yc_intra = A_intra @ Vc # [B H NUM_CHUNKS CHUNK_SIZE Dv]

    ##############################################################

    Yc = Yc_inter + Yc_intra
    Y = rearrange(Yc, 'b h n c d -> b h (n c) d')

    return Y

#======================================================================#
# Testing scripts
#======================================================================#
def optimize_for_h100():
    """Apply H100-specific optimizations for maximum performance"""

    # Environment variables for H100 optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # H100-specific PyTorch backend optimizations
    # Enable TF32 tensor cores: FP32 operations use TF32 (10-bit mantissa, 8-bit exponent)
    # This provides ~6× better accuracy than BF16 with only ~1% performance cost
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # FP8 support (only available in newer PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_fp8_e4m3fn = True
        torch.backends.cuda.matmul.allow_fp8_e5m2 = True
    except AttributeError:
        print("FP8 support not available in this PyTorch version")

    # Performance vs reproducibility settings (choose one)
    torch.backends.cudnn.benchmark = True  # Enable for better performance
    torch.backends.cudnn.deterministic = False  # Disable for better performance

    # Memory management optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        torch.cuda.empty_cache()

    print("H100 optimizations applied successfully!")

#======================================================================#
def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function using Triton's memory profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        peak_mem = (torch.cuda.max_memory_allocated() - start_mem) / 1e9
        return result, peak_mem
    else:
        return func(*args, **kwargs), 0

def compute_errors(y_pred, y_ref, name):
    """Compute detailed error metrics between predicted and reference outputs."""
    y_pred_f = y_pred.float()
    y_ref_f = y_ref.float()

    abs_err = (y_pred_f - y_ref_f).abs()
    ref_magnitude = y_ref_f.abs()
    rel_err = abs_err / (ref_magnitude + 1e-8)

    atol = 1e-5 if y_pred.dtype == torch.float32 else 1e-3
    rtol = 1e-5 if y_pred.dtype == torch.float32 else 1e-3
    allclose = torch.allclose(y_pred_f, y_ref_f, atol=atol, rtol=rtol)

    return {
        f'{name}_max_abs_err': abs_err.max().item(),
        f'{name}_mean_abs_err': abs_err.mean().item(),
        f'{name}_max_rel_err': rel_err.max().item(),
        f'{name}_mean_rel_err': rel_err.mean().item(),
        f'{name}_allclose': allclose,
    }

def main(B: int = 1, H: int = 8, N: int = 4096, Dk: int = 128, Dv: int = 32, dtype: str = 'float32'):
    device = torch.device('cuda')
    dtype = getattr(torch, dtype)

    torch.manual_seed(0)
    Q = torch.randn(B, H, N, Dk, device=device, dtype=dtype)
    K = torch.randn(B, H, N, Dk, device=device, dtype=dtype)
    V = torch.randn(B, H, N, Dv, device=device, dtype=dtype)

    print("="*120)
    print(f"Testing Causal Linear Forward Pass (B={B}, H={H}, N={N}, Dk={Dk}, Dv={Dv})")
    print("="*120)

    #======================================================================#
    # Benchmark reference implementation
    #======================================================================#
    print("Measuring reference implementation...", end=" ", flush=True)

    if N > 32768:
        import warnings
        warnings.warn(f"Reference implementation skipped for N={N} > 32768 (too slow). Returning NaNs.")
        Y_reference = torch.full((B, H, N, Dv), float('nan'), device=device, dtype=dtype)
        ref_mem = 0.0
        ref_ms = float('inf')
        print("Skipped (N > 32768)")
    else:
        # Memory measurement
        Y_reference, ref_mem = measure_memory(linear_causal_reference, Q, K, V)

        # Timing using triton.testing.do_bench
        ref_ms = triton.testing.do_bench(lambda: linear_causal_reference(Q, K, V), warmup=2, rep=2)

        print("Done")

    #======================================================================#
    # Benchmark PyTorch implementation
    #======================================================================#
    print("Measuring PyTorch implementation...", end=" ", flush=True)

    # Memory measurement
    Y_pytorch, pytorch_mem = measure_memory(linear_causal_pytorch, Q, K, V)

    # Timing using triton.testing.do_bench
    pytorch_ms = triton.testing.do_bench(lambda: linear_causal_pytorch(Q, K, V), warmup=2, rep=2)

    # Compute errors for PyTorch (skip if reference is NaN)
    if torch.isnan(Y_reference).any():
        pytorch_errors = {
            'pytorch_max_abs_err': float('nan'),
            'pytorch_mean_abs_err': float('nan'),
            'pytorch_max_rel_err': float('nan'),
            'pytorch_mean_rel_err': float('nan'),
            'pytorch_allclose': False,
        }
    else:
        pytorch_errors = compute_errors(Y_pytorch, Y_reference, 'pytorch')

    print("Done")

    #======================================================================#
    # Benchmark Triton implementation
    #======================================================================#
    print("Measuring Triton implementation...", end=" ", flush=True)

    # Memory measurement
    Y_triton, triton_mem = measure_memory(CausalLinear.apply, Q, K, V)

    # Timing using triton.testing.do_bench
    triton_ms = triton.testing.do_bench(lambda: CausalLinear.apply(Q, K, V))

    # Compute errors for Triton (skip if reference is NaN)
    if torch.isnan(Y_reference).any():
        triton_errors = {
            'triton_max_abs_err': float('nan'),
            'triton_mean_abs_err': float('nan'),
            'triton_max_rel_err': float('nan'),
            'triton_mean_rel_err': float('nan'),
            'triton_allclose': False,
        }
    else:
        triton_errors = compute_errors(Y_triton, Y_reference, 'triton')

    print("Done")

    #======================================================================#
    # Benchmark Linear noncausal implementation
    #======================================================================#
    print("Measuring Linear noncausal implementation...", end=" ", flush=True)

    # Memory measurement
    _, noncausal_mem = measure_memory(linear_noncausal, Q, K, V)

    # Timing using triton.testing.do_bench
    noncausal_ms = triton.testing.do_bench(lambda: linear_noncausal(Q, K, V))

    print("Done")

    #======================================================================#
    # Benchmark Causal SDPA implementation
    #======================================================================#
    print("Measuring Causal SDPA implementation...", end=" ", flush=True)

    Q_, K_, V_ = [torch.rand_like(V) for _ in range(3)]

    # Memory measurement
    _, SDPA_mem = measure_memory(causal_SDPA, Q_, K_, V_)

    # Timing using triton.testing.do_bench
    SDPA_ms = triton.testing.do_bench(lambda: causal_SDPA(Q_, K_, V_))

    print("Done")

    #======================================================================#
    # Print results table
    #======================================================================#
    # Calculate speedups
    has_nan_ref = torch.isnan(Y_reference).any()
    pytorch_speedup = ref_ms / pytorch_ms if pytorch_ms > 0 and ref_ms != float('inf') else float('inf')
    noncausal_speedup = ref_ms / noncausal_ms if noncausal_ms > 0 and ref_ms != float('inf') else float('inf')
    triton_speedup = ref_ms / triton_ms if triton_ms > 0 and ref_ms != float('inf') else float('inf')
    SDPA_speedup = ref_ms / SDPA_ms if SDPA_ms > 0 and ref_ms != float('inf') else float('inf')

    print("\n" + "="*120)
    print("FORWARD PASS performance summary")
    print("="*120)
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Speedup':<12} {'Memory (GB)':<15} {'Allclose':<10} "
          f"{'Abs Err (mean/max)':<25} {'Rel Err (mean/max)':<25}")
    print("-"*120)

    # Reference row (baseline)
    ref_ms_str = f"{ref_ms:.2f}" if ref_ms != float('inf') else "N/A"
    print(f"{'Reference':<20} {ref_ms_str:<15} {'1.00x':<12} {ref_mem:<15.2e} {'N/A':<10} "
          f"{'N/A':<25} {'N/A':<25}")

    # PyTorch row
    pytorch_allclose_str = "✓" if pytorch_errors['pytorch_allclose'] else "✗"
    if has_nan_ref or math.isnan(pytorch_errors['pytorch_mean_abs_err']) or math.isnan(pytorch_errors['pytorch_max_abs_err']):
        pytorch_abs_err_str = "N/A"
        pytorch_rel_err_str = "N/A"
    else:
        pytorch_abs_err_str = f"{pytorch_errors['pytorch_mean_abs_err']:.2e}/{pytorch_errors['pytorch_max_abs_err']:.2e}"
        pytorch_rel_err_str = f"{pytorch_errors['pytorch_mean_rel_err']:.2e}/{pytorch_errors['pytorch_max_rel_err']:.2e}"
    pytorch_speedup_str = f"{pytorch_speedup:.2f}x" if pytorch_speedup != float('inf') and not math.isnan(pytorch_speedup) else "N/A"
    print(f"{'PyTorch':<20} {pytorch_ms:<15.2f} {pytorch_speedup_str:<12} {pytorch_mem:<15.2e} {pytorch_allclose_str:<10} "
          f"{pytorch_abs_err_str:<25} {pytorch_rel_err_str:<25}")

    # Triton row
    triton_allclose_str = "✓" if triton_errors['triton_allclose'] else "✗"
    if has_nan_ref or math.isnan(triton_errors['triton_mean_abs_err']) or math.isnan(triton_errors['triton_max_abs_err']):
        triton_abs_err_str = "N/A"
        triton_rel_err_str = "N/A"
    else:
        triton_abs_err_str = f"{triton_errors['triton_mean_abs_err']:.2e}/{triton_errors['triton_max_abs_err']:.2e}"
        triton_rel_err_str = f"{triton_errors['triton_mean_rel_err']:.2e}/{triton_errors['triton_max_rel_err']:.2e}"
    triton_speedup_str = f"{triton_speedup:.2f}x" if triton_speedup != float('inf') and not math.isnan(triton_speedup) else "N/A"
    print(f"{'Triton':<20} {triton_ms:<15.2f} {triton_speedup_str:<12} {triton_mem:<15.2e} {triton_allclose_str:<10} "
          f"{triton_abs_err_str:<25} {triton_rel_err_str:<25}")

    # Linear noncausal row
    noncausal_speedup_str = f"{noncausal_speedup:.2f}x" if noncausal_speedup != float('inf') and not math.isnan(noncausal_speedup) and not has_nan_ref else "N/A"
    print(f"{'Linear Noncausal':<20} {noncausal_ms:<15.2f} {noncausal_speedup_str:<12} {noncausal_mem:<15.2e} {'N/A':<10} "
          f"{'N/A':<25} {'N/A':<25}")

    # Causal SDPA row
    SDPA_speedup_str = f"{SDPA_speedup:.2f}x" if SDPA_speedup != float('inf') and not math.isnan(SDPA_speedup) and not has_nan_ref else "N/A"
    print(f"{'Causal SDPA':<20} {SDPA_ms:<15.2f} {SDPA_speedup_str:<12} {SDPA_mem:<15.2e} {'N/A':<10} "
          f"{'N/A':<25} {'N/A':<25}")

    print("="*120 + "\n")

    return

if __name__ == "__main__":
    # Triton cache directory for faster recompilation
    dotdot = lambda dir: os.path.abspath(os.path.join(dir, '..'))
    PROJDIR = dotdot(dotdot(dotdot(os.path.dirname(__file__))))
    CACHE_DIR = os.path.join(dotdot(PROJDIR), 'cache', 'triton')
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = CACHE_DIR
    # shutil.rmtree(CACHE_DIR, ignore_errors=True)

    optimize_for_h100()

    main(B=1, H=8, N=2048, Dk=128, Dv=32, dtype='float32')
    # main(B=1, H=8, N=16384, Dk=128, Dv=32, dtype='float32')
    # main(B=1, H=8, N=16384, Dk=128, Dv=32, dtype='float16')

    # main(B=1, H=8, N=65536, Dk=32, Dv=32, dtype='float16')
#======================================================================#
#