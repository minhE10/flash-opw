#!/usr/bin/env bash
# Paper-style Phase 1 synthetic benchmark on one allocated GPU.
# Usage: bash scripts/run_phase1.sh <allocated-GPU-index-or-UUID> [toy overrides]
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_phase1.sh <allocated GPU index or UUID> [toy overrides]' >&2
  exit 2
fi

selected_gpu="$1"
shift
if [[ ! "$selected_gpu" =~ ^[0-9]+$ && ! "$selected_gpu" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  printf '%s\n' 'Select exactly one GPU allocated to you.' >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_GPUS:-}" ]]; then
  printf '%s\n' 'Inside Slurm, preserve scheduler CUDA_VISIBLE_DEVICES and run the Python command directly.' >&2
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache}"

printf 'Using Python: %s\n' "$(command -v python)"
python -m experiments.toy \
  --device cuda \
  --datasets gaussian \
  --sizes 10000 \
  --dim 64 \
  --epsilon 0.1 \
  --iters 10 \
  --schedule symmetric \
  --precision tf32 \
  --warmups 10 \
  --repeats 50 \
  --threads 2 \
  --memory-fraction 0.45 \
  --max-dense-mib 6144 \
  --no-plots \
  "$@"
