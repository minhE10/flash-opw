#!/usr/bin/env bash
# Trial k-NN/MAP/ACC on exactly one allocated GPU; CPU references use two threads.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_opw_knn.sh <allocated GPU index or UUID> [overrides]' >&2
  exit 2
fi
selected_gpu="$1"
shift
if [[ ! "$selected_gpu" =~ ^[0-9]+$ && ! "$selected_gpu" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  printf '%s\n' 'Select exactly one GPU allocated to you.' >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$selected_gpu" ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES is already set differently. Preserve your allocation.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache}"
python -m experiments.opw_knn "$@" --device cuda
