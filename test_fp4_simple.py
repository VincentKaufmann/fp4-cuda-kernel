#!/usr/bin/env python3
"""Minimal FP4 GEMM test using Triton tl.dot_scaled on DGX Spark (sm_121).

The "simple hack": tl.dot_scaled with "e2m1" format compiles to PTX directly,
bypassing the cuBLAS/CUTLASS arch restriction that blocks FP4 on sm_121.

Key insight from Triton semantics:
  lhs: [M, K//2] uint8    (FP4 packed along K)
  rhs: [K//2, N] uint8    (FP4 packed along K)  ← NOT transposed!
  lhs_scale: [M, K//32] uint8 e8m0
  rhs_scale: [N, K//32] uint8 e8m0  ← note: [N, ...] not [K//32, N]
"""

import torch
import triton
import triton.language as tl
import time
import sys
sys.path.insert(0, ".")

from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def quantize_rows_mxfp4(tensor, block_size=32):
    """Quantize a 2D float tensor to MXFP4 row-by-row along last dim.

    Args:
        tensor: [rows, cols] float32, cols must be multiple of block_size

    Returns:
        packed: [rows, cols//2] uint8
        scales: [rows, cols//block_size] uint8 e8m0
        dequant: [rows, cols] float32 for reference
    """
    rows, cols = tensor.shape
    assert cols % block_size == 0
    device = tensor.device

    # Per-block scaling
    blocks = tensor.reshape(rows * (cols // block_size), block_size)
    block_max = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    FP4_MAX = 6.0
    scale_float = block_max / FP4_MAX

    # E8M0 encode scales
    scale_e8m0_obj = MXScaleTensor(data=scale_float)
    scales_uint8 = scale_e8m0_obj.data  # flat

    # Dequantize E8M0 for correct scaling
    scale_deq = scale_e8m0_obj.to(torch.float32)

    # Scale values and convert to FP4
    scaled = blocks / scale_deq.unsqueeze(-1)
    fp4_obj = MXFP4Tensor(data=scaled.reshape(-1))

    # Dequantize for reference
    dequant_unscaled = fp4_obj.to(torch.float32)
    dequant = (dequant_unscaled.reshape(-1, block_size) * scale_deq.unsqueeze(-1)).reshape(rows, cols)

    # Pack along columns (dim=1)
    fp4_2d = MXFP4Tensor.__new__(MXFP4Tensor)
    fp4_2d.data = fp4_obj.data.reshape(rows, cols)
    fp4_2d.device = device
    packed = fp4_2d.to_packed_tensor(dim=1)  # [rows, cols//2]

    scales_out = scales_uint8.reshape(rows, cols // block_size)

    return packed, scales_out, dequant


# ─── Triton FP4 GEMM Kernel ─────────────────────────────────────────────────

@triton.jit
def fp4_matmul_kernel(
    a_ptr, a_scale_ptr,
    b_ptr, b_scale_ptr,
    c_ptr,
    M, N, K,  # K is the UNPACKED reduction dimension
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # UNPACKED K block size
):
    """FP4 × FP4 matmul: C[M,N] = A[M,K] @ B[K,N]

    Data layout:
      A packed: [M, K//2] uint8, row-major (stride_ak = K//2)
      B packed: [K//2, N] uint8, row-major (stride_bk = N, stride_bn = 1)
      A_scale: [M, K//32] uint8 e8m0
      B_scale: [N, K//32] uint8 e8m0 (note: [N,...] not [K//32,...])
    """
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
        # ── Load A tile: [BLOCK_M, BLOCK_K//2] uint8 ──
        k_packed_start = k_idx * BLOCK_K_PACKED
        offs_k_packed = k_packed_start + tl.arange(0, BLOCK_K_PACKED)
        a_ptrs = a_ptr + offs_m[:, None] * K_packed + offs_k_packed[None, :]
        a = tl.load(a_ptrs)

        # ── Load B tile: [BLOCK_K//2, BLOCK_N] uint8 ──
        # B is [K//2, N] row-major, so stride along K dim is N
        b_ptrs = b_ptr + offs_k_packed[:, None] * N + offs_n[None, :]
        b = tl.load(b_ptrs)

        # ── Load A scales: [BLOCK_M, BLOCK_K//32] uint8 ──
        k_scale_start = k_idx * BLOCK_K_SCALE
        offs_k_scale = k_scale_start + tl.arange(0, BLOCK_K_SCALE)
        a_scale_ptrs = a_scale_ptr + offs_m[:, None] * K_scale + offs_k_scale[None, :]
        a_scale = tl.load(a_scale_ptrs)

        # ── Load B scales: [BLOCK_N, BLOCK_K//32] uint8 ──
        # Shape is [N, K//32] — rows are N, cols are K//32
        b_scale_ptrs = b_scale_ptr + offs_n[:, None] * K_scale + offs_k_scale[None, :]
        b_scale = tl.load(b_scale_ptrs)

        # ── FP4 scaled dot product ──
        acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


# ─── Test ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")

    print("=" * 70)
    print(" FP4 GEMM Test v3 — DGX Spark Blackwell (sm_121)")
    print(" tl.dot_scaled with e2m1, CORRECT rhs layout [K//2, N]")
    print("=" * 70)

    M, K, N = 128, 256, 128
    print(f"\nMatrix: C[{M}×{N}] = A[{M}×{K}] @ B[{K}×{N}]")

    torch.manual_seed(42)
    a_float = torch.randn(M, K, device=device, dtype=torch.float32) * 0.5
    b_float = torch.randn(K, N, device=device, dtype=torch.float32) * 0.5

    ref_bf16 = a_float @ b_float
    print(f"BF16 ref: range=[{ref_bf16.min():.4f}, {ref_bf16.max():.4f}]")

    # Quantize A: [M, K] → packed [M, K//2], scales [M, K//32]
    print("\nQuantizing A [M, K]...")
    a_packed, a_scales, a_dequant = quantize_rows_mxfp4(a_float)
    print(f"  A packed: {a_packed.shape}, A scales: {a_scales.shape}")

    # Quantize B: [K, N]
    # B is [K, N]. We need to pack along K (dim=0), not N.
    # But quantize_rows_mxfp4 packs along dim=1 (columns).
    # So we transpose, quantize (packing along cols = original rows = K), then transpose back.
    print("Quantizing B [K, N] (pack along K=dim0)...")
    b_t = b_float.T.contiguous()  # [N, K]
    b_packed_nt, b_scales, b_dequant_nt = quantize_rows_mxfp4(b_t)  # [N, K//2], [N, K//32]
    # b_packed_nt is [N, K//2] — but we need [K//2, N] for the kernel
    b_packed = b_packed_nt.T.contiguous()  # [K//2, N]
    b_dequant = b_dequant_nt.T.contiguous()  # [K, N]
    # b_scales stays [N, K//32] — correct for tl.dot_scaled rhs_scale
    print(f"  B packed: {b_packed.shape}, B scales: {b_scales.shape}")

    # Software FP4 reference
    ref_fp4_sw = a_dequant @ b_dequant
    cos_quant = torch.nn.functional.cosine_similarity(
        ref_bf16.flatten().unsqueeze(0),
        ref_fp4_sw.flatten().unsqueeze(0),
    ).item()
    print(f"\nSW FP4 quality (cos sim vs BF16): {cos_quant:.6f}")

    # Launch kernel
    c = torch.zeros(M, N, device=device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    print(f"\n--- Launching Triton FP4 Kernel ---")
    print(f"Grid: {grid}, Block: ({BLOCK_M}×{BLOCK_N}×{BLOCK_K})")

    try:
        fp4_matmul_kernel[grid](
            a_packed, a_scales,
            b_packed, b_scales,
            c,
            M, N, K,
            N, 1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        torch.cuda.synchronize()

        print(f"\n*** FP4 KERNEL EXECUTED ON SM_121! ***")
        print(f"Result range: [{c.min():.4f}, {c.max():.4f}]")
        print(f"Expected:     [{ref_fp4_sw.min():.4f}, {ref_fp4_sw.max():.4f}]")

        cos_hw_bf16 = torch.nn.functional.cosine_similarity(
            ref_bf16.flatten().unsqueeze(0), c.flatten().unsqueeze(0)).item()
        cos_hw_sw = torch.nn.functional.cosine_similarity(
            ref_fp4_sw.flatten().unsqueeze(0), c.flatten().unsqueeze(0)).item()
        rmse = ((c - ref_fp4_sw) ** 2).mean().sqrt().item()

        print(f"\n  Cos sim vs BF16:   {cos_hw_bf16:.6f}")
        print(f"  Cos sim vs SW FP4: {cos_hw_sw:.6f}")
        print(f"  RMSE vs SW FP4:    {rmse:.6f}")

        if cos_hw_sw > 0.95:
            status = "CORRECT"
        elif cos_hw_sw > 0.5:
            status = "PARTIALLY CORRECT"
        else:
            status = "INCORRECT"
            print(f"\n  First 8 values:")
            print(f"    HW:  {c[0,:8].tolist()}")
            print(f"    SW:  {ref_fp4_sw[0,:8].tolist()}")
            print(f"    Ref: {ref_bf16[0,:8].tolist()}")

        print(f"\n  Status: {status}")

        if cos_hw_sw > 0.5:
            # Benchmark
            print(f"\n--- Benchmarking (128×256×128) ---")
            for _ in range(10):
                fp4_matmul_kernel[grid](
                    a_packed, a_scales, b_packed, b_scales, c,
                    M, N, K, N, 1,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
            torch.cuda.synchronize()

            n_iters = 200
            t0 = time.perf_counter()
            for _ in range(n_iters):
                fp4_matmul_kernel[grid](
                    a_packed, a_scales, b_packed, b_scales, c,
                    M, N, K, N, 1,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
            torch.cuda.synchronize()
            t_fp4 = (time.perf_counter() - t0) / n_iters * 1000

            a_bf = a_float.bfloat16(); b_bf = b_float.bfloat16()
            for _ in range(10): torch.mm(a_bf, b_bf)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters): torch.mm(a_bf, b_bf)
            torch.cuda.synchronize()
            t_bf16 = (time.perf_counter() - t0) / n_iters * 1000

            print(f"  FP4:  {t_fp4:.4f} ms")
            print(f"  BF16: {t_bf16:.4f} ms")
            print(f"  Ratio: {t_bf16/t_fp4:.2f}x")

    except Exception as e:
        print(f"\nKernel failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
