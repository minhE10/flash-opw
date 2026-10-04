# Validation record

Validation date: 2026-10-03 (user timezone Asia/Jakarta).

## Windows CPU

Environment: Python 3.14.3, PyTorch 2.11.0+cpu. No local CUDA device.

```text
python -m pytest -q
38 passed, 43 skipped
```

The 43 skips are real GPU tests, **not GPU passes**. CPU tests cover:

- Direct-distance dense vs streamed shifted updates in float64, both schedules,
  half/full squared cost, rectangular clouds and partial tiles.
- Agreement with independently computed classical scaling Sinkhorn.
- Nonuniform positive marginals, early stopping and small-epsilon constant costs.
- `P V`, `P.T V`, signed multi-column values and unconverged row masses.
- First-order envelope gradients vs finite differences of converged OT.
- Hadamard-weighted transport vs a materialized dense plan.
- Paper HVP decomposition vs finite differences, for full and half squared cost.
- Differentiable scalar cost, analytic backward and PyTorch double backward.
- Invalid-input rejection and deterministic datasets.

Toy end-to-end run:

```bash
python -m experiments.toy --device cpu --sizes 32 65 --iters 100 --repeats 1 \
  --weighted --target-ratio 1.3 --output outputs/cpu-validation
```

All three datasets passed, with 32x42 and 65x84 clouds:

| Quantity | Largest observed value |
| --- | ---: |
| Absolute difference between dense and Torch-online plan elements | 6.985e-9 |
| Absolute log-plan difference | 1.908e-6 |
| Marginal L1 residual across both solvers | 3.153e-7 |

`comparison.png` was visually inspected. A second CPU run (Gaussian, d=64,
epsilon=0.05, symmetric updates, sizes 33/65, 100 iterations) also passed.
CPU timings are not evidence of Triton speedup and are not reported as such.
Generated CSV/JSON/PNG files live in ignored `outputs/` and can be reproduced.

The server shell script passes `bash -n`. Both a CUDA experiment without
explicit GPU selection and `pytest --require-gpu` on this CPU environment
fail as intended, instead of silently reporting successful GPU validation.

## Actual kernel logic on CPU (WSL)

In an isolated WSL Ubuntu environment: Python 3.14.4, Triton 3.8.0,
NumPy 2.5.3; no PyTorch or CUDA GPU needed for these developer checks.

```text
python scripts/interpret_kernels.py
PASS: 28 solver, 56 transport/adjoint and 28 Hadamard cases;
max plan error=8.55e-07
```

This executes the actual Triton kernel bodies with the Triton CPU interpreter,
checking them against an independent float64 NumPy direct-distance reference.
Cases include 1x1, 1x67, 63x1 and irregular rectangular shapes, feature dimensions
up to 129, two epsilon values, both schedules, masked feature/value tiles,
and signed 35-column transport applications in both directions. The interpreter
does not model GPU scheduling, resource limits or Tensor Core rounding.

## Offline RTX 5080 compilation

```text
python scripts/compile_kernels.py --arch 120 --output outputs/offline-compile-hvp-rtx5080.json
PASS: compiled 129 variants for sm_120
```

Triton 3.8.0 generated nonempty CUDA binaries for alternating update,
one-launch symmetric update and transport kernels in `ieee`, `tf32` and
`tf32x3`, with d in {2, 7, 64, 65, 129, 256, 512, 1024}.
The effective tile/pipeline choices all used **at most 65,536 bytes of shared
memory**. Initial larger configurations exceeded that budget; the final launch
policy caps high-dimensional tiles at 16x32 and uses one pipeline stage. The
Hadamard HVP kernel has its own stricter launch policy from d=64 because it
keeps score and weighting dot products live at the same time.

This verifies compiler acceptance and static shared-memory requirements,
**not** launch success or numerical/performance behavior on actual hardware.
The server's installed Triton version has not been supplied; keep its existing
PyTorch-compatible version and run the on-server checks below.

## Required server checks

The RTX 5080 server has not been accessed by this agent. Its actual Triton
version, JIT/runtime compatibility, kernel execution correctness, GPU memory
usage and performance must be checked there:

```bash
# Example only: replace 1 with the GPU allocated to you.
bash scripts/run_server.sh 1 --sizes 128 256 512 --weighted --target-ratio 1.3
```

Do not infer a speedup from the paper or the offline/CPU checks. The command
performs GPU smoke/parity tests before benchmarking and writes the actual
environment, timings, allocated memory and validation status into `outputs/`.
