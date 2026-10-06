#!/usr/bin/env bash
# Same panels/data/timer as the current run, with untouched author Flash code.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_author_benchmarks.sh <allocated GPU index or UUID> [overrides]' >&2
  exit 2
fi
selected_gpu="$1"
shift
exec bash scripts/run_paper_benchmarks.sh "$selected_gpu" "$@" --flash-implementation author
