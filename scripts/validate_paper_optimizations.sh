#!/usr/bin/env bash
# Usage: bash scripts/validate_paper_optimizations.sh <allocated GPU> [ablation args]
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/validate_paper_optimizations.sh <allocated GPU> [ablation args]' >&2
  exit 2
fi
selected_gpu="$1"
shift
if [[ ! "$selected_gpu" =~ ^[0-9]+$ && ! "$selected_gpu" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  printf '%s\n' 'Select exactly one GPU allocated to you.' >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$selected_gpu" ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES is set differently; preserve your allocation.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache}"
export FLASHOPW_OTT_HESSIAN_PATH="${FLASHOPW_OTT_HESSIAN_PATH:-outputs/third_party/OTT-Hessian}"
export FLASHOPW_AUTOTUNE=0 FLASHOPW_VECTOR_KERNEL=1 FLASHOPW_GRADIENT_KERNEL=1
if [[ ! -f "$FLASHOPW_OTT_HESSIAN_PATH/SinkhornHessian.py" ]]; then
  printf '%s\n' 'Run bash scripts/setup_ott_hessian.sh before validation.' >&2
  exit 2
fi
python -c 'import os, jax, ott, lineax, optax; from experiments.ott_hessian import load_hessian; load_hessian(os.environ["FLASHOPW_OTT_HESSIAN_PATH"]); devices=jax.devices(); assert any(d.platform == "gpu" for d in devices), f"JAX sees no GPU: {devices}"; print("JAX devices:", devices)'
python -m pytest -q --require-gpu tests/test_gpu.py tests/test_jax_hvp.py
python -m experiments.kernel_benchmarks "$@"
