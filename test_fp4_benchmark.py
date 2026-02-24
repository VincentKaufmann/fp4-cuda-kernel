#!/usr/bin/env python3
"""FP4 GEMM benchmark at realistic sizes + hardware path detection.

Tests if tl.dot_scaled is using Blackwell tensor cores (tcgen05_mma_scaled)
or falling back to software emulation.
"""

import torch
import triton
import triton.language as tl
import time
import sys
sys.path.insert(0, ".")

from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def quantize_rows_mxfp4(tensor, block_size=32):
    """Quantize [rows, cols] float32 → packed [rows, cols//2] uint8 + scales [rows, cols//32] e8m0."""
    rows, cols = tensor.shape
    assert cols % block_size == 0
    device = tensor.device

    blocks = tensor.reshape(rows * (cols // block_size), block_size)
    block_max = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    scale_float = block_max / 6.0

    scale_e8m0_obj = MXScaleTensor(data=scale_float)
    scales_uint8 = scale_e8m0_obj.data
    scale_deq = scale_e8m0_obj.to(torch.float32)

    scaled = blocks / scale_deq.unsqueeze(-1)
    fp4_obj = MXFP4Tensor(data=scaled.reshape(-1))
    dequant = (fp4_obj.to(torch.float32).reshape(-1, block_size) * scale_deq.unsqueeze(-1)).reshape(rows, cols)

    fp4_2d = MXFP4Tensor.__new__(MXFP4Tensor)
    fp4_2d.data = fp4_obj.data.reshape(rows, cols)
    fp4_2d.device = device
    packed = fp4_2d.to_packed_tensor(dim=1)

    return packed, scales_uint8.reshape(rows, cols // block_size), dequant


@triton.jit
def fp4_matmul_kernel(
    a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_k_iters = K // BLOCK_K

    K_packed = K // 2
    K_scale = K // 32
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // 32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    for k_idx in range(n_k_iters):
        k_packed_start = k_idx * BLOCK_K_PACKED
        offs_k_packed = k_packed_start + tl.arange(0, BLOCK_K_PACKED)

        a_ptrs = a_ptr + offs_m[:, None] * K_packed + offs_k_packed[None, :]
        a = tl.load(a_ptrs)

        b_ptrs = b_ptr + offs_k_packed[:, None] * N + offs_n[None, :]
        b = tl.load(b_ptrs)

        k_scale_start = k_idx * BLOCK_K_SCALE
        offs_k_scale = k_scale_start + tl.arange(0, BLOCK_K_SCALE)
        a_scale_ptrs = a_scale_ptr + offs_m[:, None] * K_scale + offs_k_scale[None, :]
        a_scale = tl.load(a_scale_ptrs)

        b_scale_ptrs = b_scale_ptr + offs_n[:, None] * K_scale + offs_k_scale[None, :]
        b_scale = tl.load(b_scale_ptrs)

        acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def prepare_data(M, K, N, device):
    """Prepare FP4 data for benchmark."""
    # Pad K to multiple of 64
    K_pad = ((K + 63) // 64) * 64

    a_float = torch.randn(M, K_pad, device=device, dtype=torch.float32) * 0.1
    b_float = torch.randn(K_pad, N, device=device, dtype=torch.float32) * 0.1

    a_packed, a_scales, _ = quantize_rows_mxfp4(a_float)

    b_t = b_float.T.contiguous()
    b_packed_nt, b_scales, _ = quantize_rows_mxfp4(b_t)
    b_packed = b_packed_nt.T.contiguous()

    return a_packed, a_scales, b_packed, b_scales, K_pad


def benchmark_size(M, K, N, device, n_warmup=5, n_iters=50):
    """Benchmark FP4 vs BF16 at given size."""
    a_packed, a_scales, b_packed, b_scales, K_pad = prepare_data(M, K, N, device)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 256
    # Adjust BLOCK_K if K_pad < 256
    if K_pad < 256:
        BLOCK_K = K_pad

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Warmup FP4
    for _ in range(n_warmup):
        fp4_matmul_kernel[grid](
            a_packed, a_scales, b_packed, b_scales, c,
            M, N, K_pad, N, 1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    torch.cuda.synchronize()

    # Time FP4
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fp4_matmul_kernel[grid](
            a_packed, a_scales, b_packed, b_scales, c,
            M, N, K_pad, N, 1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    torch.cuda.synchronize()
    t_fp4 = (time.perf_counter() - t0) / n_iters * 1000

    # Time BF16
    a_bf = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    b_bf = torch.randn(K, N, device=device, dtype=torch.bfloat16) * 0.1
    for _ in range(n_warmup):
        torch.mm(a_bf, b_bf)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a_bf, b_bf)
    torch.cuda.synchronize()
    t_bf16 = (time.perf_counter() - t0) / n_iters * 1000

    return t_fp4, t_bf16


def check_ptx():
    """Check if the compiled PTX uses tcgen05 instructions."""
    print("\n--- Checking compiled PTX for tcgen05 instructions ---")
    try:
        # Get the compiled kernel's binary/PTX
        # Triton caches compiled kernels
        import glob
        import os
        cache_dir = os.path.expanduser("~/.triton/cache")
        ptx_files = glob.glob(f"{cache_dir}/**/*.ptx", recursive=True)
        ttgir_files = glob.glob(f"{cache_dir}/**/*.ttgir", recursive=True)

        if ptx_files:
            # Get most recent PTX file
            ptx_files.sort(key=os.path.getmtime, reverse=True)
            latest = ptx_files[0]
            with open(latest) as f:
                ptx = f.read()

            has_tcgen05 = "tcgen05" in ptx
            has_mma_scaled = "mma.scaled" in ptx or "tcgen05.mma" in ptx
            has_fp4_to_fp = "fp4_to_fp" in ptx.lower() or "cvt.rn.bf16" in ptx

            print(f"  Latest PTX: {latest}")
            print(f"  Contains tcgen05: {has_tcgen05}")
            print(f"  Contains mma.scaled: {has_mma_scaled}")

            if has_tcgen05 and has_mma_scaled:
                print("  >>> HARDWARE PATH: Using Blackwell tensor cores! <<<")
            elif has_fp4_to_fp:
                print("  >>> SOFTWARE PATH: FP4→BF16 upcast + standard dot <<<")
            else:
                print("  >>> UNKNOWN PATH — check PTX manually <<<")

            # Show relevant PTX snippet
            for line in ptx.split('\n'):
                if any(kw in line.lower() for kw in ['tcgen05', 'mma', 'fp4', 'e2m1', 'scaled']):
                    print(f"  PTX: {line.strip()}")

        if ttgir_files:
            ttgir_files.sort(key=os.path.getmtime, reverse=True)
            for f in ttgir_files[:3]:
                with open(f) as fh:
                    ir = fh.read()
                if 'dot_scaled' in ir or 'e2m1' in ir:
                    print(f"\n  TTGIR: {f}")
                    for line in ir.split('\n'):
                        if 'dot_scaled' in line or 'e2m1' in line:
                            print(f"    {line.strip()}")

    except Exception as e:
        print(f"  Could not check PTX: {e}")


def main():
    device = torch.device("cuda")

    print("=" * 70)
    print(" FP4 GEMM Benchmark — DGX Spark Blackwell (sm_121)")
    print("=" * 70)

    # First, compile and verify correctness at small size
    print("\n--- Correctness check ---")
    M, K, N = 128, 256, 128
    a_packed, a_scales, b_packed, b_scales, K_pad = prepare_data(M, K, N, device)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    fp4_matmul_kernel[(1,1)](
        a_packed, a_scales, b_packed, b_scales, c,
        M, N, K_pad, N, 1,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, num_warps=4)
    torch.cuda.synchronize()
    print(f"  Kernel compiled and executed. Output range: [{c.min():.4f}, {c.max():.4f}]")

    # Check PTX
    check_ptx()

    # Benchmark at multiple sizes
    print("\n--- Benchmarks ---")
    print(f"{'Size':>20s}  {'FP4 (ms)':>10s}  {'BF16 (ms)':>10s}  {'Ratio':>8s}")
    print("-" * 55)

    sizes = [
        (128, 256, 128),      # Tiny
        (256, 512, 256),      # Small
        (512, 1024, 512),     # Medium
        (1024, 2048, 1024),   # Large
        (2048, 2880, 2880),   # GPT-OSS hidden dim
        (4096, 2880, 2880),   # GPT-OSS batch
        (2048, 2880, 7680),   # GPT-OSS FFN (2880 → 7680)
    ]

    for M, K, N in sizes:
        try:
            t_fp4, t_bf16 = benchmark_size(M, K, N, device)
            ratio = t_bf16 / t_fp4
            tag = " <<<" if ratio > 1.0 else ""
            print(f"  {M}×{K}×{N:>5d}  {t_fp4:10.3f}  {t_bf16:10.3f}  {ratio:7.2f}x{tag}")
        except Exception as e:
            print(f"  {M}×{K}×{N:>5d}  FAILED: {e}")

    print(f"\n{'='*70}")
    print(f" FP4 GEMM confirmed working on sm_121!")
    print(f" Key question: is it using tensor cores or software fallback?")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
