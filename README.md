# FP4 CUDA Kernel for NVIDIA Blackwell (DGX Spark / RTX 50 Series)

Hardware FP4 tensor core acceleration for Blackwell SM120/SM121, built on CUTLASS 3.8.

No existing library exposes FP4 tensor cores as a callable API on consumer Blackwell (SM120/SM121). Not cuBLAS, not Triton, not bitsandbytes. CUTLASS Example 79a proves the hardware works, but it's a standalone demo that generates random data internally. I needed real FP4 for inference on my VLM project, so I built the missing pieces: a GPU-side BF16-to-FP4 quantization kernel, the CUTLASS interleaved scale layout engine, and a Python API that makes it a one-line call.

The key feature is a **pre-quantized weight cache** - quantize weights once at model load, then every inference call only quantizes activations on the fly. This hits **85-129 TFLOPS** and is **1.4-2.4x faster than BF16 `F.linear`** at inference-relevant batch sizes, with **4x memory savings** from FP4 weights.

## Performance

Benchmarked on DGX Spark GB10 (SM121, 128 GB unified LPDDR5x, 273 GB/s):

| Size (M x N x K) | FP4 Cached | BF16 F.linear | Speedup | Float32 mm |
|---|---|---|---|---|
| 256 x 2880 x 2880 | 0.050 ms (85 TF) | 0.118 ms | **2.4x** | 0.64 ms |
| 512 x 2880 x 2880 | 0.100 ms (85 TF) | 0.101 ms | **1.0x** | 0.85 ms |
| 2048 x 2880 x 7680 | 0.702 ms (129 TF) | 1.190 ms | **1.7x** | 7.12 ms |
| 2048 x 7680 x 2880 | 0.766 ms (118 TF) | 1.073 ms | **1.4x** | 6.95 ms |
| 4096 x 2880 x 2880 | 1.089 ms (62 TF) | 0.752 ms | 0.7x | 5.12 ms |

At M=4096, BF16 cuBLAS pulls ahead - the GEMM is compute-bound enough that BF16 saturates the tensor cores and FP4's activation quantization becomes overhead. At that scale, the win is **4x memory savings**, not speed.

**Accuracy**: 0.991 Pearson correlation vs float32, ~1.2% mean relative error. Same NVFP4 format NVIDIA uses in ModelOpt/TensorRT-LLM.

## Quick Start

```bash
git clone https://github.com/VincentKaufmann/fp4-cuda-kernel.git
cd fp4-cuda-kernel
./build.sh    # auto-detects GPU arch, clones CUTLASS 3.8, compiles
```

### Cached Mode (Inference)

Quantize weights once at model load, then every forward pass only quantizes activations:

```python
from fp4_gemm import fp4_quantize, fp4_cached_linear

# One-time cost at model load (milliseconds)
cache = fp4_quantize(weight)

# Every inference call - 85-129 TFLOPS
output = fp4_cached_linear(x, cache)
```

### Dynamic Mode

Both matrices quantized every call. Slower, but useful when both inputs change:

```python
from fp4_gemm import fp4_matmul, fp4_linear

C = fp4_matmul(A, B)              # A: [M, K] bf16, B: [N, K] bf16
out = fp4_linear(x, weight, bias)  # drop-in F.linear replacement
```

### Full API

```python
from fp4_gemm import fp4_quantize, fp4_cached_linear, FP4WeightCache

# Context manager with auto-cleanup
with FP4WeightCache(weight, bias=bias) as cache:
    output = cache.forward(x)

# Quantize all linear layers at model load
caches = {}
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() == 2:
        caches[name] = fp4_quantize(param.data)
```

FP4 weights are 4x smaller than BF16. A 2880x2880 weight matrix drops from 15.8 MB to 4.0 MB. Auto-padding is handled internally.

## Why This Exists

I spent two days trying to get FP4 working through existing paths before writing this:

| Approach | Result on SM121 |
|----------|----------------|
| cuBLAS FP4 | Not available (no `cublasLtMatmul` FP4 support) |
| Triton `tl.dot_scaled` | Software fallback - 100x slower than BF16 |
| Gluon / tcgen05 | SM121 lacks TMEM/tcgen05 (datacenter SM100 only) |
| bitsandbytes NF4 | Software dequant, no tensor core acceleration |
| CUTLASS Example 79a | Works! But standalone binary, not a library |
| vLLM / TensorRT-LLM | Require full serving stack, can't just call a GEMM |

One thing that tripped me up: **MXFP4 weights (E2M1 + E8M0 scales) can't be used directly on SM121 tensor cores.** SM121 uses NVFP4 with UE4M3 scale factors and a specific interleaved layout (`SfKMajorAtom`). Any MXFP4 model checkpoints need re-quantization before the hardware will accept them.

## How It Works

```
Cached path (inference):
  Load time:  BF16 Weight [N, K]  ->  GPU Quantize  ->  FP4 [N, K/2] + UE4M3 scales (stored)
  Each call:  BF16 Activation [M, K]  ->  GPU Quantize  ->  FP4 [M, K/2] + UE4M3 scales
                                                                    |
                                                                    v
                                                         CUTLASS Block-Scaled GEMM
                                                         (mma.sync.aligned.block_scale)
                                                                    |
                                                                    v
                                                         BF16 Output [M, N]
```

### The Custom Pieces

**GPU FP4 Quantization Kernel** - not in stock CUTLASS. One thread per 16-element block:
1. Read 16 BF16 values, compute max absolute value
2. Scale = max / 6.0, convert to UE4M3 (unsigned, 4 exp bias=7, 3 mantissa)
3. Divide each value by scale, round to nearest FP4 E2M1, pack pairs into bytes
4. Write scale to CUTLASS interleaved `SfKMajorAtom` position

**Scale Factor Layout** - the hardest part. I had to reverse-engineer CuTe's flat coordinate decomposition for the interleaved layout:

```
Atom Shape:  ((32, 4), (16, 4))
Atom Stride: ((16, 4), (0,  1))

index = (row % 32) * 16 + ((row / 32) % 4) * 4
      + (row / 128) * row_tile_stride
      + (k_block % 4) * 1 + (k_block / 4) * k_tile_stride
```

Getting this wrong corrupts ~10% of output elements. I tried manual hierarchical coordinates first - produces different indices. The flat decomposition is the only correct path.

**UE4M3 Conversion** - NVFP4 uses UE4M3 scale factors (not E8M0 like MXFP4). Full encoding/decoding with subnormal handling implemented in device code. This difference is underdocumented and cost me a full day of debugging.

### CUTLASS Configuration

Identical to Example 79a:
```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;  // no multicast on SM121
```

## Manual Build

```bash
git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git cutlass

# sm_120a for RTX 5090, sm_121a for DGX Spark
nvcc -arch=sm_121a -shared -Xcompiler -fPIC -O2 --expt-relaxed-constexpr \
  -I cutlass/include -I cutlass/tools/util/include -I cutlass/examples/common \
  -o libfp4gemm.so fp4_gemm_lib.cu
```

## Requirements

- NVIDIA GPU with SM120 or SM121 (RTX 5090/5080, DGX Spark GB10)
- CUDA Toolkit 12.8+ (12.9+ for SM121)
- CUTLASS 3.8+ (auto-downloaded by `build.sh`)
- Python 3.8+, PyTorch with CUDA

## Files

| File | Description |
|------|-------------|
| `fp4_gemm_lib.cu` | CUDA kernel - CUTLASS GEMM + GPU quantization + cached weight API |
| `fp4_gemm.py` | Python wrapper - dynamic, cached, auto-padding, batching, context managers |
| `build.sh` | Build script - auto-detects GPU arch, clones CUTLASS, compiles |

## Limitations

- Dimensions auto-padded to multiples of 128 (transparent in Python API)
- Forward-only, no gradient support (inference and frozen-weight training)
- SM120/SM121 only (SM100 datacenter Blackwell uses a different instruction set)
- At large M (4096+), BF16 cuBLAS is faster in wall-clock - FP4 still wins on memory

## Citation

```
@software{fp4_cuda_kernel,
  author = {Koc, Vincent},
  title = {FP4 CUDA Kernel for Blackwell SM120/SM121},
  year = {2026},
  url = {https://github.com/VincentKaufmann/fp4-cuda-kernel}
}
```

## License

Apache 2.0. CUTLASS is BSD-3-Clause (NVIDIA).
