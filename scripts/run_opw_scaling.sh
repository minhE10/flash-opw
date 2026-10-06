#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_opw_scaling.sh <allocated GPU index or UUID> [options]' >&2
  exit 2
fi
selected_gpu="$1"
shift
if [[ ! "$selected_gpu" =~ ^[0-9]+$ && ! "$selected_gpu" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  printf '%s\n' 'Select exactly one allocated GPU.' >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$selected_gpu" ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES differs; preserve your allocation.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$selected_gpu"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache}"
python -m experiments.opw_scaling "$@" --device cuda
