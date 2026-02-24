# FP4 GEMM Library for NVIDIA DGX Spark / GeForce RTX 50 Series

Hardware FP4 tensor core GEMM for Blackwell SM120/SM121, built on CUTLASS 3.8.

**5-9x faster than BF16 cuBLAS** at real model dimensions, with 0.991 Pearson correlation accuracy.

## Performance

Benchmarked on DGX Spark GB10 (SM121, 128 GB unified LPDDR5x):

| Matrix Size (M×N×K) | FP4 | BF16 cuBLAS | Speedup |
|---------------------|-----|-------------|---------|
| 256 × 2944 × 2944 | 0.08 ms (53 TF) | 0.33 ms (13 TF) | **4.0x** |
| 1024 × 2944 × 2944 | 0.26 ms (67 TF) | 1.37 ms (13 TF) | **5.2x** |
| 2048 × 2944 × 2944 | 0.34 ms (103 TF) | 2.40 ms (15 TF) | **7.0x** |
| 4096 × 2944 × 2944 | 0.50 ms (143 TF) | 4.66 ms (15 TF) | **9.4x** |
| 2048 × 7680 × 2944 | 0.70 ms (133 TF) | 6.05 ms (15 TF) | **8.7x** |

Peak: **143 TFLOPS** (including GPU-side quantization overhead). Raw GEMM kernel: **280 TFLOPS**.

## What This Does

Takes BF16 matrices on GPU, quantizes them to FP4 E2M1 with UE4M3 block scales **entirely on GPU**, then runs the CUTLASS block-scaled tensor core GEMM. One function call, no host roundtrips.

```python
from fp4_gemm import fp4_matmul, fp4_linear

# Like torch.mm(A, B.T) but 5-9x faster
C = fp4_matmul(A, B)  # A: [M, K] bf16, B: [N, K] bf16 → C: [M, N] bf16

# Drop-in replacement for F.linear
out = fp4_linear(x, weight, bias)
```

## Architecture

```
BF16 Input A [M, K]  ──→  GPU Quantize Kernel  ──→  FP4 packed [M, K/2] + UE4M3 scales
BF16 Input B [N, K]  ──→  GPU Quantize Kernel  ──→  FP4 packed [N, K/2] + UE4M3 scales
                                                              │
                                                              ▼
                                                    CUTLASS Block-Scaled GEMM
                                                    (mma.sync.aligned.block_scale)
                                                              │
                                                              ▼
                                                    BF16 Output D [M, N]
```

- **Quantization**: Per-block (16 elements) max-abs scaling. Scale = max/6.0, rounded to UE4M3.
- **FP4 E2M1**: 4-bit float (1 sign, 2 exponent bias=1, 1 mantissa). Values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
- **UE4M3 Scale**: 8-bit unsigned float (4 exp bias=7, 3 mantissa). Range: [0.00195, 480.0].
- **Block size**: 16 FP4 elements share 1 UE4M3 scale factor.
- **Scale layout**: CUTLASS interleaved `SfKMajorAtom` with CuTe flat coordinate decomposition.

## Requirements

- NVIDIA GPU with SM120 or SM121 (RTX 5090, DGX Spark GB10, etc.)
- CUDA Toolkit 12.8+ (12.9+ for SM121)
- CUTLASS 3.8+ (included as submodule)
- Python 3.8+, PyTorch with CUDA support

## Build

```bash
git clone --recursive https://github.com/VincentKaufmann/fp4-gemm-blackwell.git
cd fp4-gemm-blackwell

nvcc -arch=sm_121a -shared -Xcompiler -fPIC -O2 --expt-relaxed-constexpr \
  -I cutlass/include -I cutlass/tools/util/include -I cutlass/examples/common \
  -o libfp4gemm.so fp4_gemm_lib.cu
```

For RTX 5090 (SM120), use `-arch=sm_120a`.

## How It Works — Key Technical Details

### The CUTLASS Configuration

This wraps CUTLASS Example 79a with identical kernel configuration:

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using LayoutA  = cutlass::layout::RowMajor;                     // A is row-major
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using LayoutB  = cutlass::layout::ColumnMajor;                  // B is column-major
using TileShape = Shape<_128, _128, _128>;                      // Threadblock tile
using ClusterShape = Shape<_1, _1, _1>;                         // No multicast (SM121)
```

### Scale Factor Layout

The scale factors use CUTLASS's interleaved `SfKMajorAtom`:
```
Shape:  ((32, 4), (SFVecSize, 4))
Stride: ((16, 4), (0,         1))
```

The K-inner dimension (SFVecSize=16) has stride 0 (broadcast — all 16 elements share one scale). The layout is tiled across the full matrix via `tile_to_shape`.

**Critical finding**: CuTe's flat coordinate decomposition (`layout(row, k, 0)`) handles the interleaved indexing correctly. Manual hierarchical coordinate computation produces **wrong indices** and corrupts ~10% of output elements.

### GPU Quantization Kernel

One CUDA thread per 16-element scale block:
1. Read 16 BF16 values from source matrix
2. Compute max absolute value → UE4M3 scale factor
3. Divide each value by scale, round to nearest FP4 E2M1
4. Pack 2 FP4 values per byte (low nibble = even index)
5. Write scale to CUTLASS interleaved layout position

## Files

| File | Description |
|------|-------------|
| `fp4_gemm_lib.cu` | CUDA source — CUTLASS GEMM + GPU quantization kernels |
| `fp4_gemm.py` | Python ctypes wrapper with auto-padding and batching |
| `FINDINGS.md` | Full research writeup — SM121 FP4 capabilities |
| `BUGFIX.md` | Bug fix documentation and key technical discoveries |

## Limitations

- Dimensions must be multiples of 128 (auto-padded in Python API)
- Quantization happens every call (no cached FP4 weights yet)
- No gradient support (forward-only, suitable for inference and frozen-weight training)
- SM120/SM121 only (no SM100 support — different instruction set)

## Citation

If you find this useful, please cite:

```
@software{fp4_gemm_blackwell,
  author = {Koc, Vincent},
  title = {FP4 GEMM Library for Blackwell SM120/SM121},
  year = {2026},
  url = {https://github.com/VincentKaufmann/fp4-gemm-blackwell}
}
```

## License

Apache 2.0 (our code). CUTLASS is BSD-3-Clause (NVIDIA).
