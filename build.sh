#!/bin/bash
# Build FP4 GEMM shared library for SM120/SM121

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pull CUTLASS if needed
if [ ! -d "cutlass/include" ]; then
    echo "Cloning CUTLASS 3.8..."
    git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git cutlass
fi

# Detect architecture
ARCH="sm_121a"  # DGX Spark GB10
if [ -n "$1" ]; then
    ARCH="$1"
    echo "Using architecture: $ARCH"
else
    # Try to auto-detect
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if echo "$GPU_NAME" | grep -qi "5090\|5080\|5070"; then
        ARCH="sm_120a"
        echo "Detected RTX 50 series, using $ARCH"
    elif echo "$GPU_NAME" | grep -qi "GB10\|Spark"; then
        ARCH="sm_121a"
        echo "Detected DGX Spark, using $ARCH"
    else
        echo "Using default $ARCH (pass sm_120a for RTX 5090)"
    fi
fi

echo "Building libfp4gemm.so (arch=$ARCH)..."
nvcc -arch="$ARCH" -shared -Xcompiler -fPIC -O2 --expt-relaxed-constexpr \
    -I cutlass/include -I cutlass/tools/util/include -I cutlass/examples/common \
    -o libfp4gemm.so fp4_gemm_lib.cu

echo "Done! Library: $SCRIPT_DIR/libfp4gemm.so"
echo ""
echo "Quick test:"
echo "  python3 -c \"from fp4_gemm import fp4_matmul; import torch; A=torch.randn(128,128,dtype=torch.bfloat16,device='cuda'); B=torch.randn(128,128,dtype=torch.bfloat16,device='cuda'); C=fp4_matmul(A,B); print(f'OK: {C.shape}')\""
