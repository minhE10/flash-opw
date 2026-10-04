#!/usr/bin/env bash
# Run all eight paper-style synthetic scaling panels on one allocated GPU.
# Usage: bash scripts/run_paper_benchmarks.sh <allocated-GPU-index-or-UUID> [overrides]
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_paper_benchmarks.sh <allocated GPU index or UUID> [overrides]' >&2
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache}"
export KEOPS_CACHE_FOLDER="${KEOPS_CACHE_FOLDER:-$HOME/.cache/keops2.3}"
printf 'Using Python: %s\n' "$(command -v python)"
python -m experiments.paper_benchmarks "$@"
