#!/usr/bin/env python3
"""Test TMEM load layout conversion with different num_warps on sm_121."""

import torch
import triton
import sys
sys.path.insert(0, ".")

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout,
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


@gluon.jit
def test_tmem_load(c_ptr, stride_c, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
    pid_m = ttgl.program_id(0)
    pid_n = ttgl.program_id(1)

    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 256 if BLOCK_N >= 256 else BLOCK_N
    acc_dtype: ttgl.constexpr = ttgl.float32
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    tmem_reg_layout: ttgl.constexpr = get_tmem_reg_layout(
        acc_dtype, (BLOCK_M, BLOCK_N), acc_tmem_layout, ttgl.num_warps()
    )

    acc_tmem = allocate_tensor_memory(acc_dtype, [BLOCK_M, BLOCK_N], acc_tmem_layout)
    out = acc_tmem.load(tmem_reg_layout)
    ret_layout: ttgl.constexpr = default_blocked_layout([BLOCK_M, BLOCK_N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)

    offs_m = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_c + offs_n[None, :]
    ttgl.store(c_ptrs, out)


def main():
    device = torch.device("cuda")
    c = torch.zeros(128, 128, device=device, dtype=torch.float32)

    for nw in [4, 8]:
        print(f"Testing num_warps={nw}...")
        try:
            import shutil
            import os
            cache_dir = os.path.expanduser("~/.triton/cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

            test_tmem_load[(1, 1)](c, 128, BLOCK_M=128, BLOCK_N=128, num_warps=nw)
            torch.cuda.synchronize()
            print(f"  SUCCESS with num_warps={nw}!")
            print(f"  Output: [{c.min():.4f}, {c.max():.4f}]")
        except Exception as e:
            err = str(e)[:300]
            print(f"  FAILED: {err}")


if __name__ == "__main__":
    main()
