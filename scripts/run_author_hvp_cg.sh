#!/usr/bin/env bash
# Bound the search; keep all failed attempts and validate core at the first passing cap.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one allocated GPU}"
author_python="${AUTHOR_PYTHON:-outputs/venv-author-flashsinkhorn/bin/python}"
run_stamp="$(date +%Y%m%d_%H%M%S)_$$"

for cg_cap in 128 256 512 1024; do
  echo "[sweep] Checking double-backward HVP with CG cap ${cg_cap} (damping/tolerances unchanged)."
  run_output="outputs/author_flashsinkhorn_rtx5080_cg_${run_stamp}_${cg_cap}"
  if "$author_python" scripts/validate_author_flashsinkhorn.py \
      --gpu --suite cg --kernel-profile rtx5080 --hvp-max-cg-iter "$cg_cap" --output "$run_output"; then
    echo "[sweep] Both CG paths converged and parity passed at cap ${cg_cap}; validating core."
    "$author_python" scripts/validate_author_flashsinkhorn.py \
      --gpu --suite core --kernel-profile rtx5080 --hvp-max-cg-iter "$cg_cap" \
      --output "outputs/author_flashsinkhorn_rtx5080_core_cg_${run_stamp}_${cg_cap}"
    exit 0
  else
    attempt_exit=$?
    # Environment/source/collection problems are not evidence that CG needs more steps.
    if ! "$author_python" - "$run_output/validation.json" "$attempt_exit" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
run = json.loads(path.read_text(encoding="utf-8"))
study = run.get("cg_convergence", {})
records = study.get("records", [])
retry = (int(sys.argv[2]) == 1 and run.get("pytest_exit_code") == 1 and
         run.get("source_after_tests", {}).get("status") == "verified" and
         run.get("profile_after_tests", {}).get("status") == "profile_verified" and
         study.get("status") == "failed" and
         {record["path"] for record in records} == {"autograd", "reference"} and
         all(record["finite_output"] for record in records) and
         all(record["confirmed"] or record["cg_iters"] >= record["max_cg_iter"] for record in records) and
         any(not record["confirmed"] for record in records))
raise SystemExit(0 if retry else 1)
PY
    then
      echo "[sweep] Stopping: failure requires inspection (not a confirmed CG iteration-limit case)." >&2
      exit "$attempt_exit"
    fi
  fi
done
echo "[sweep] CG did not confirm convergence up to cap 1024. Inspect cg_convergence.json; no tolerance/damping changes made." >&2
exit 1
