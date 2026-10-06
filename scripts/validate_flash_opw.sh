#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/validate_flash_opw.sh <allocated GPU index or UUID>' >&2
  exit 2
fi
selected_gpu="$1"
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
python -m pytest -q --require-gpu tests/test_opw.py tests/test_retrieval.py \
  tests/test_sequence_metrics.py tests/test_sequence_data.py tests/test_opw_experiments.py \
  tests/test_opw_group1.py tests/test_opw_group2.py tests/test_opw_group3.py
