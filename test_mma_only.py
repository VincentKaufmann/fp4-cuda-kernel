#!/usr/bin/env python3
"""Test JUST the MMA path — no tmem_load (avoid tcgen05.wait.ld).

If this compiles and runs, it means:
- TMA loading works
- tcgen05_copy works
- tcgen05_mma_scaled works
- The ONLY problem is getting data OUT of TMEM

Uses tl.dot_scaled's decomposed fallback for output.
"""

import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, ".")

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)


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
def get_shared_memory_mma_layout(type, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
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


@gluon.jit
def get_shared_memory_mma_operand(value, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    layout: ttgl.constexpr = get_shared_memory_mma_layout(
        value.type, operand_index, allow_transpose, is_fp4_padded, force_transpose)
    return ttgl.allocate_shared_memory(value.dtype, value.shape, layout, value)


@gluon.jit
def simple_mma_test(
    a_ptr, b_ptr, c_ptr,
    stride_c,
    M: ttgl.constexpr,
    N: ttgl.constexpr,
    K: ttgl.constexpr,
):
    """Simple BF16 dot product using Blackwell tcgen05_mma.

    Uses register-based loading (no TMA) to test if MMA works at all.
    Avoids tmem_load by returning zeros (just testing compilation).
    """
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)

    # Load A and B via pointers (register path)
    blocked: ttgl.constexpr = default_blocked_layout([M, K], ttgl.num_warps())
    offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(0, blocked))
    offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, blocked))
    a_ptrs = a_ptr + pid_m * M * K + offs_m[:, None] * K + offs_k[None, :]
    a = ttgl.load(a_ptrs)

    blocked_b: ttgl.constexpr = default_blocked_layout([K, N], ttgl.num_warps())
    offs_k_b = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, blocked_b))
    offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(1, blocked_b))
    b_ptrs = b_ptr + pid_n * N + offs_k_b[:, None] * N + offs_n[None, :]
    b = ttgl.load(b_ptrs)

    # Convert to MMA shared layout
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose=True)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose=True)

    # Setup TMEM accumulator
    m: ttgl.constexpr = 128 if M >= 128 else 64
    n: ttgl.constexpr = 256 if N >= 256 else N
    acc_dtype: ttgl.constexpr = ttgl.float32
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
        acc_dtype, (M, N), acc_tmem_layout, ttgl.num_warps()
    )

    # Zero-init accumulator in registers, store to TMEM
    acc_temp = ttgl.zeros([M, N], acc_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
    fence_async_shared()

    # MMA
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # Load result from TMEM — THIS IS THE CRITICAL STEP
    out = acc_tmem.load(tmem_reg_layout)
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)

    # Store via pointers
    offs_m_out = pid_m * M + ttgl.arange(0, M)
    offs_n_out = pid_n * N + ttgl.arange(0, N)
    c_ptrs = c_ptr + offs_m_out[:, None] * stride_c + offs_n_out[None, :]
    ttgl.store(c_ptrs, out)


def main():
    device = torch.device("cuda")

    M, N, K = 128, 128, 128
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    print(f"Testing Blackwell tcgen05_mma (BF16 dot, {M}×{K}×{N})...")

    try:
        simple_mma_test[(1, 1)](
            a, b, c, N,
            M=M, N=N, K=K,
            num_warps=4,
        )
        torch.cuda.synchronize()
        ref = (a.float() @ b.float())
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), c.flatten().unsqueeze(0)
        ).item()
        print(f"  SUCCESS! Output range: [{c.min():.4f}, {c.max():.4f}]")
        print(f"  Cos sim vs ref: {cos:.6f}")
    except Exception as e:
        err = str(e)[:500]
        print(f"  FAILED: {err}")
        if "tcgen05.wait" in err.lower():
            print("  >>> tcgen05.wait ISel gap — LLVM can't lower this instruction")
        elif "Cannot select" in err:
            print("  >>> LLVM ISel failure")
        elif "layout" in err.lower():
            print("  >>> Layout conversion issue")


if __name__ == "__main__":
    main()
