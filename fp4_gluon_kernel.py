#!/usr/bin/env python3
"""FP4 GEMM on DGX Spark Blackwell (sm_121) via Triton Gluon tcgen05_mma_scaled.

Version 3: TMA loading + tcgen05_copy for scales (avoids tcgen05.wait::st).

The key insight: LLVM's NVPTX backend in Triton 3.6.0 doesn't have ISel patterns
for llvm.nvvm.tcgen05.wait.st. This intrinsic is generated when storing registers
to Tensor Memory (TMEM) via allocate_tensor_memory(value=...). We avoid it by:
  1. Allocating accumulator TMEM without initial value (use_acc=False handles it)
  2. Copying scales from shared memory to TMEM via tcgen05_copy (async, committed
     via mbarrier — no tcgen05.wait::st needed)

Data flow:
  Global → TMA → Shared Memory (A, B operands — stay in smem for MMA)
  Global → TMA → Shared Memory → tcgen05_copy → Tensor Memory (scales)
  MMA → Tensor Memory (accumulator) → Registers → Shared → TMA → Global (output)

Usage:
    cd ~/GPT-OSS-120B && python fp4-hack/fp4_gluon_kernel.py
"""

import torch
import time
import sys
sys.path.insert(0, ".")

import triton
import triton.language as tl

# ─── Gluon imports ─────────────────────────────────────────────────────────────

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._layouts import NVMMASharedLayout

# Host-side TMA descriptor
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# Device-side TMA + synchronization
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)

# Blackwell tensor core + tensor memory
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma_scaled,
    tcgen05_commit,
    tcgen05_copy,
)

from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


# ─── Layout helpers (from translator_helpers.py) ─────────────────────────────

@gluon.constexpr_function
def get_num_threads_per_warp():
    return ttgl.constexpr(32)


@gluon.constexpr_function
def default_blocked_layout(shape, num_warps):
    rank = len(shape)
    size_per_thread = [1 for _ in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    threads_per_warp[rank - 1] = get_num_threads_per_warp()
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    order = [i for i in range(rank - 1, -1, -1)]
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


@gluon.constexpr_function
def get_shared_memory_mma_layout(type, operand_index, allow_transpose,
                                  is_fp4_padded=False, force_transpose=False):
    if not allow_transpose:
        transposed = (operand_index == 1)
        if force_transpose:
            transposed = not transposed
    else:
        transposed = operand_index == 1

    shape = type.shape
    ele_bit_width = type.element_ty.primitive_bitwidth
    packing_factor = 2 if is_fp4_padded else 1

    contig_dim_size_in_byte = (
        (shape[0] if transposed else shape[1]) * packing_factor * ele_bit_width // 8
    )
    if contig_dim_size_in_byte >= 128 and contig_dim_size_in_byte % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size_in_byte >= 64 and contig_dim_size_in_byte % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size_in_byte >= 32 and contig_dim_size_in_byte % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for dim in shape:
        flatten_outer_dim *= dim
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0

    return ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        transposed=transposed,
        element_bitwidth=ele_bit_width,
        rank=len(shape),
        fp4_padded=is_fp4_padded,
    )


# ─── TMA Load Helpers ────────────────────────────────────────────────────────

@gluon.jit
def tma_load_smem(desc, coord):
    """TMA load: global → shared memory. Returns shared_memory_descriptor."""
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, coord, bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    return smem


@gluon.jit
def tma_store_from_regs(desc, coord, value):
    """TMA store: registers → shared → global."""
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc, coord, smem)
    tma.store_wait(0)
    smem._keep_alive()


# ─── The FP4 Gluon Kernel (TMA + tcgen05_copy) ──────────────────────────────

@gluon.jit
def fp4_gemm_tma(
    desc_a,       # TMA desc: A [M, K//2] uint8
    desc_b,       # TMA desc: B [K//2, N] uint8
    desc_as,      # TMA desc: A_scale [M, K//32] uint8
    desc_bs,      # TMA desc: B_scale [N, K//32] uint8
    c_ptr,        # Pointer: C [M, N] float32 (pointer store, avoids layout issue)
    stride_c,     # Stride for C
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K_PACKED: ttgl.constexpr,   # K//2
    BLOCK_K_SCALE: ttgl.constexpr,    # K//32
):
    """FP4 × FP4 GEMM using Blackwell tcgen05_mma_scaled via TMA loading.

    Single-tile K (no K-loop): entire K dimension in one MMA.
    Inputs via TMA — zero pointer arithmetic for loads.
    Scales via tcgen05_copy (shared→TMEM) — no tcgen05.wait::st.
    Output via pointer store — avoids TMEM→linear→blocked layout issue.
    """
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)

    # ─── TMA Load: A [BLOCK_M, BLOCK_K_PACKED] uint8 ───
    a_smem = tma_load_smem(desc_a, [pid_m * BLOCK_M, 0])

    # ─── TMA Load: B [BLOCK_K_PACKED, BLOCK_N] uint8 ───
    b_smem = tma_load_smem(desc_b, [0, pid_n * BLOCK_N])

    # ─── TMA Load: Scales to shared memory ───
    as_smem = tma_load_smem(desc_as, [pid_m * BLOCK_M, 0])
    bs_smem = tma_load_smem(desc_bs, [pid_n * BLOCK_N, 0])

    # ─── Accumulator in TMEM (NO initial value — avoids tcgen05.wait::st) ───
    M: ttgl.constexpr = BLOCK_M
    N: ttgl.constexpr = BLOCK_N
    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = ttgl.float32
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
        acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps()
    )

    # Allocate WITHOUT initial value — no tcgen05.st, no tcgen05.wait::st!
    acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_tmem_layout)

    # ─── Scales: shared memory → TMEM via tcgen05_copy ───
    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(ttgl.uint8, [BLOCK_M, BLOCK_K_SCALE], scale_layout)
    b_scale_tmem = allocate_tensor_memory(ttgl.uint8, [BLOCK_N, BLOCK_K_SCALE], scale_layout)

    tcgen05_copy(as_smem, a_scale_tmem)
    tcgen05_copy(bs_smem, b_scale_tmem)

    # Commit scale copies and wait
    scale_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(scale_bar, count=1)
    tcgen05_commit(scale_bar)
    mbarrier.wait(scale_bar, phase=0)
    mbarrier.invalidate(scale_bar)

    # ─── THE HARDWARE INSTRUCTION ───
    mma_bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)

    tcgen05_mma_scaled(
        a_smem, b_smem, acc_tmem,
        a_scale_tmem, b_scale_tmem,
        "e2m1", "e2m1",
        use_acc=False,
    )
    tcgen05_commit(mma_bar)
    mbarrier.wait(mma_bar, phase=0)
    mbarrier.invalidate(mma_bar)

    # ─── Load result from TMEM → registers (stays in linear layout) ───
    out = acc_tmem.load(tmem_reg_layout)

    # Convert to blocked layout for pointer store
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)

    # ─── Pointer-based store (only 2 aranges: M and N, no conflict) ───
    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_c + offs_n[None, :]
    ttgl.store(c_ptrs, out)


# ─── Data Preparation ────────────────────────────────────────────────────────

def quantize_rows_mxfp4(tensor, block_size=32):
    """Quantize [rows, cols] float32 → packed [rows, cols//2] uint8 + scales."""
    rows, cols = tensor.shape
    assert cols % block_size == 0
    device = tensor.device

    blocks = tensor.reshape(rows * (cols // block_size), block_size)
    block_max = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    scale_float = block_max / 6.0

    scale_obj = MXScaleTensor(data=scale_float)
    scales_uint8 = scale_obj.data
    scale_deq = scale_obj.to(torch.float32)

    scaled = blocks / scale_deq.unsqueeze(-1)
    fp4_obj = MXFP4Tensor(data=scaled.reshape(-1))
    dequant = (fp4_obj.to(torch.float32).reshape(-1, block_size) * scale_deq.unsqueeze(-1)).reshape(rows, cols)

    fp4_2d = MXFP4Tensor.__new__(MXFP4Tensor)
    fp4_2d.data = fp4_obj.data.reshape(rows, cols)
    fp4_2d.device = device
    packed = fp4_2d.to_packed_tensor(dim=1)

    return packed, scales_uint8.reshape(rows, cols // block_size), dequant


def pad_k_to_512(K):
    """Pad K to next multiple of 512 for TMA alignment (K//32 must be mult of 16)."""
    return ((K + 511) // 512) * 512


def prepare_data(M, K, N, device):
    """Prepare FP4 data with K padded for TMA alignment."""
    K_pad = pad_k_to_512(K)

    a_float = torch.randn(M, K_pad, device=device, dtype=torch.float32) * 0.5
    b_float = torch.randn(K_pad, N, device=device, dtype=torch.float32) * 0.5

    # Zero the padding region
    if K_pad > K:
        a_float[:, K:] = 0.0
        b_float[K:, :] = 0.0

    # Quantize A
    a_packed, a_scales, a_dequant = quantize_rows_mxfp4(a_float)

    # Quantize B (pack along K via transpose)
    b_t = b_float.T.contiguous()
    b_packed_nt, b_scales, b_dequant_nt = quantize_rows_mxfp4(b_t)
    b_packed = b_packed_nt.T.contiguous()
    b_dequant = b_dequant_nt.T.contiguous()

    ref_fp4_sw = a_dequant @ b_dequant

    return a_packed, a_scales, b_packed, b_scales, a_dequant, b_dequant, ref_fp4_sw, K_pad


def create_tma_descriptors(a_packed, a_scales, b_packed, b_scales, BLOCK_M, BLOCK_N, BLOCK_K_PACKED, BLOCK_K_SCALE):
    """Create host-side TMA descriptors for input tensors."""

    # A: [M, K//2] uint8, operand 0 (not transposed)
    layout_a = NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K_PACKED], tl.uint8, transposed=False
    )
    desc_a = TensorDescriptor.from_tensor(a_packed, [BLOCK_M, BLOCK_K_PACKED], layout_a)

    # B: [K//2, N] uint8, operand 1 (transposed for MMA)
    layout_b = NVMMASharedLayout.get_default_for(
        [BLOCK_K_PACKED, BLOCK_N], tl.uint8, transposed=True
    )
    desc_b = TensorDescriptor.from_tensor(b_packed, [BLOCK_K_PACKED, BLOCK_N], layout_b)

    # A_scale: [M, K//32] uint8 — TMA to shared, then tcgen05_copy to TMEM
    layout_as = NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K_SCALE], tl.uint8, transposed=False
    )
    desc_as = TensorDescriptor.from_tensor(a_scales, [BLOCK_M, BLOCK_K_SCALE], layout_as)

    # B_scale: [N, K//32] uint8
    layout_bs = NVMMASharedLayout.get_default_for(
        [BLOCK_N, BLOCK_K_SCALE], tl.uint8, transposed=False
    )
    desc_bs = TensorDescriptor.from_tensor(b_scales, [BLOCK_N, BLOCK_K_SCALE], layout_bs)

    return desc_a, desc_b, desc_as, desc_bs


# ─── Test ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")

    print("=" * 70)
    print(" FP4 GEMM — Gluon tcgen05_mma_scaled + TMA + tcgen05_copy")
    print(" DGX Spark Blackwell (sm_121)")
    print(" Avoids tcgen05.wait::st via async copy pipeline")
    print("=" * 70)

    M, K, N = 128, 256, 128
    K_pad = pad_k_to_512(K)
    print(f"\nMatrix: C[{M}×{N}] = A[{M}×{K}] @ B[{K}×{N}]")
    print(f"K padded: {K} → {K_pad} (TMA alignment)")

    torch.manual_seed(42)

    # Reference
    a_ref = torch.randn(M, K, device=device, dtype=torch.float32) * 0.5
    b_ref = torch.randn(K, N, device=device, dtype=torch.float32) * 0.5
    ref_bf16 = a_ref @ b_ref

    # FP4 data
    a_packed, a_scales, b_packed, b_scales, a_dq, b_dq, ref_fp4_sw, K_pad = prepare_data(M, K, N, device)

    print(f"\nData shapes:")
    print(f"  A packed:  {a_packed.shape}  (uint8)")
    print(f"  B packed:  {b_packed.shape}  (uint8)")
    print(f"  A scales:  {a_scales.shape}  (uint8 e8m0)")
    print(f"  B scales:  {b_scales.shape}  (uint8 e8m0)")

    cos_quant = torch.nn.functional.cosine_similarity(
        ref_bf16.flatten().unsqueeze(0),
        ref_fp4_sw.flatten().unsqueeze(0),
    ).item()
    print(f"  SW FP4 quality (cos sim vs BF16): {cos_quant:.6f}")

    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = 128, 128
    BLOCK_K_PACKED = K_pad // 2
    BLOCK_K_SCALE = K_pad // 32

    print(f"\nCreating TMA descriptors...")
    desc_a, desc_b, desc_as, desc_bs = create_tma_descriptors(
        a_packed, a_scales, b_packed, b_scales,
        BLOCK_M, BLOCK_N, BLOCK_K_PACKED, BLOCK_K_SCALE
    )

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    stride_c = N  # row-major

    print(f"\n--- Launching Gluon FP4 Kernel ---")
    print(f"Grid: {grid}, Blocks: M={BLOCK_M} N={BLOCK_N} K_packed={BLOCK_K_PACKED}")
    print(f"Strategy: TMA load → tcgen05_copy for scales → tcgen05_mma_scaled")
    print(f"Bypasses: tcgen05.wait::st (LLVM ISel gap)")
    print(f"Output: pointer store (avoids linear→blocked layout issue)")

    try:
        fp4_gemm_tma[grid](
            desc_a, desc_b, desc_as, desc_bs, c, stride_c,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K_PACKED=BLOCK_K_PACKED, BLOCK_K_SCALE=BLOCK_K_SCALE,
            num_warps=4,
        )
        torch.cuda.synchronize()

        print(f"\n{'='*70}")
        print(f" *** GLUON FP4 KERNEL EXECUTED ON SM_121! ***")
        print(f"{'='*70}")
        print(f"Result range: [{c.min():.4f}, {c.max():.4f}]")
        print(f"Expected:     [{ref_fp4_sw.min():.4f}, {ref_fp4_sw.max():.4f}]")

        cos_hw_sw = torch.nn.functional.cosine_similarity(
            ref_fp4_sw.flatten().unsqueeze(0),
            c.flatten().unsqueeze(0),
        ).item()
        cos_hw_bf16 = torch.nn.functional.cosine_similarity(
            ref_bf16.flatten().unsqueeze(0),
            c.flatten().unsqueeze(0),
        ).item()
        rmse = ((c - ref_fp4_sw) ** 2).mean().sqrt().item()

        print(f"\n  Cos sim vs SW FP4: {cos_hw_sw:.6f}")
        print(f"  Cos sim vs BF16:   {cos_hw_bf16:.6f}")
        print(f"  RMSE vs SW FP4:    {rmse:.6f}")

        if cos_hw_sw > 0.95:
            print(f"\n  ** HARDWARE FP4 IS CORRECT! **")

            # Benchmark
            print(f"\n--- Benchmarking ---")
            for _ in range(10):
                fp4_gemm_tma[grid](
                    desc_a, desc_b, desc_as, desc_bs, c, stride_c,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED, BLOCK_K_SCALE=BLOCK_K_SCALE,
                    num_warps=4)
            torch.cuda.synchronize()

            n_iters = 200
            t0 = time.perf_counter()
            for _ in range(n_iters):
                fp4_gemm_tma[grid](
                    desc_a, desc_b, desc_as, desc_bs, c, stride_c,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED, BLOCK_K_SCALE=BLOCK_K_SCALE,
                    num_warps=4)
            torch.cuda.synchronize()
            t_fp4 = (time.perf_counter() - t0) / n_iters * 1000

            a_bf = a_ref.bfloat16()
            b_bf = b_ref.bfloat16()
            for _ in range(10):
                torch.mm(a_bf, b_bf)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                torch.mm(a_bf, b_bf)
            torch.cuda.synchronize()
            t_bf16 = (time.perf_counter() - t0) / n_iters * 1000

            print(f"  FP4 (Gluon HW):  {t_fp4:.4f} ms")
            print(f"  BF16 (cuBLAS):   {t_bf16:.4f} ms")
            ratio = t_bf16 / t_fp4
            print(f"  Speedup: {ratio:.2f}x")
            if ratio > 1.0:
                print(f"\n  ** FP4 IS FASTER THAN BF16! HARDWARE TENSOR CORES CONFIRMED! **")
        else:
            print(f"\n  Results don't match — debugging needed")
            print(f"  First 8 HW:  {c[0,:8].tolist()}")
            print(f"  First 8 SW:  {ref_fp4_sw[0,:8].tolist()}")

    except Exception as e:
        print(f"\nKernel failed: {e}")
        import traceback
        traceback.print_exc()

        err = str(e)
        if "tcgen05.wait" in err.lower():
            print("\n>>> tcgen05.wait::st STILL appearing — check which allocate_tensor_memory has value=...")
        elif "Cannot select" in err:
            print(f"\n>>> LLVM ISel failure — intrinsic not supported in this LLVM build")
            # Try to extract which intrinsic
            for line in err.split('\n'):
                if 'intrinsic' in line.lower():
                    print(f"    {line.strip()}")
        elif "auto_encoding" in err.lower():
            print("\n>>> Auto-encoding error — should be impossible with TMA!")
        elif "descriptor" in err.lower() or "tma" in err.lower():
            print("\n>>> TMA descriptor error")
        else:
            print(f"\n>>> Error type: {type(e).__name__}")


if __name__ == "__main__":
    main()
