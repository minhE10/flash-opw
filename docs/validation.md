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

## Optimization checks, 2026-10-05

These changes extend commit `fec70ce`; the original comparison
report and original benchmark log were not replaced or reinterpreted as new
measurements. No server GPU has been accessed during these checks.

Windows: Python 3.14.3, PyTorch 2.11.0+cpu. An isolated `.venv-baselines`
with JAX 0.8.2, OTT-JAX 0.5.1, Lineax 0.0.8 and Optax 0.2.6 reuses the
existing CPU PyTorch installation. With the external pinned OTT-Hessian source:

```text
FLASHOPW_OTT_HESSIAN_PATH=outputs/ott-hessian-reference
.venv-baselines/Scripts/python -m pytest -q
48 passed, 118 skipped, 1 warning
```

All 118 skips are GPU tests. The warning is an upstream OTT use of deprecated
JAX batching API. The nine JAX tests run on CPU: three custom matrix-free HVP
cases, plus six external OTT-Hessian adapter cases with epsilon 0.1/0.7,
nonuniform rectangular Gaussian inputs and uniform 37x79 inputs at d=65,
using both 12 and 50 CG steps. Paper HVP direction normalization is used in
the external tests. The external cases retain elementwise rtol=3e-3,
atol=3e-5 and also require global relative L2 error below 3e-3.
They check OTT potential conversion and absolute damping conversion.
A separate CPU test verifies that the autotuner never launches candidates
above the shared-memory budget and reuses its cached selection.

The external checkout is
`yexf308/OTT-Hessian@7eb189fe39982f587da935044480655b65939637`.
The normalized `SinkhornHessian.py` checksum is
`7cd3c27a14563e949bf2f35d5719173308df38f498975a6b950488e5cb1c5158`.
The file stays unmodified. A fixed-step adapter replaces the Lineax CG call,
while retaining upstream HessianA formulas and OTT transport geometry.
Initial testing of zero-tolerance upstream Lineax at 50 steps produced NaN
for the uniform d=65 case. The guarded fixed-step CG adapter passes that case
without reducing precision, damping, iteration budget or error thresholds.
This is not a claim about the performance of raw upstream Lineax CG.

Existing WSL environment: Python 3.14.4, Triton 3.8.0, NumPy 2.5.3.
The updated kernel interpreter ran:

```text
PASS: 32 solver, 64 matrix, 64 vector, 64 gradient and 32 Hadamard cases;
max plan error=3.77e-06
```

Cases extend to d=1024 and exercise both transport directions. Primitive
applications are compared against an independent float64 dense plan formed
from stored potentials, separating operator roundoff from finite FP32 solver
roundoff. The solver comparison still uses the independent float64
direct-distance recurrence. Warnings arise on masked padded rows, which are
never written into real outputs; all real outputs meet existing tolerances.

Offline compilation for RTX 5080 (`sm_120`) with Triton 3.8.0 passed:

```text
compile_kernels.py --arch 120 --dims 2 64 129 1024
PASS: compiled 99 variants for sm_120
compile_tuning.py --arch 120 --dims 64 1024
PASS: 134/174 candidates fit 64 KiB; others will not launch
```

The 99 default variants cover all six kernel types and IEEE/TF32/TF32x3.
The 174 tuning candidates cover all six kernels at d=64/1024 in IEEE/TF32;
40 exceed the shared-memory limit and are recorded as rejected before launch.
The maximum accepted shared-memory requirement is exactly 65,536 bytes.
This is resource/compiler evidence, not GPU correctness or speedup evidence.
Large gradient variants generate sizeable binaries and may have register
pressure or spills; the actual CUDA tuner/profile is needed to assess them.

An additional isolated `.venv-linux36` uses **Triton 3.6.0**, the server's
reported version, with Python 3.14.4 and NumPy 2.5.3. Both the final interpreter
suite and the 99 default `sm_120` compilation variants above pass with 3.6.0.
The same maximum interpreter plan error is 3.77e-06; all compiler shared-memory
requirements stay within 64 KiB. This caught and fixed a compiler compatibility
issue: the gradient's static feature-loop count needs an explicit `tl.constexpr`
instead of relying on `tl.cdiv` constant folding available in Triton 3.8.0.
Autotune's 174-candidate resource table was collected with 3.8.0; the runtime
autotuner checks metadata again with the server's installed compiler.

The new server scripts pass `bash -n`; Python compilation and `git diff --check`
pass. The transfer patch is checked and applied to an extracted pristine
`fec70ce` tree, and all changed files are compared with the local working tree.
See [server commands and interpretation](paper_reproduction.md).

## JAX GPU parity follow-up, 2026-10-05

The user supplied the server validation log after pulling `81b6bbd`:
`126 passed, 1 failed, 1 warning in 264.99s`. The only reported failure was
the custom matrix-free JAX HVP on uniform 37x79 points at d=65, epsilon 0.1,
with 12 CG steps. It differed from the CPU dense FP32 reference at 183/2405
entries, with maximum absolute difference 3.3867359e-4. This is user-supplied
GPU evidence; no remote execution was performed here. The validation script
stops on pytest failure, so this log does not establish ablation timings.

Inspection found that the custom implementation recomputed the transposed
coupling with reversed potential addition order and a different score tile
shape. On the local uniform fixture, reversed addition alone changes logits
by up to 7.6293945e-6. The correction evaluates identical source-by-target
tiles for both directions, adds source u before target v, and preserves FP32
rounding points with optimization barriers. Matmul precision is explicit
even under JAX's default precision context. The custom HVP also now uses the
same persistent device-side CG activity guard as the external adapter.
Whether these changes fully resolve the observed GPU failure needs a server
rerun; the GPU discrepancy was not reproduced on the local CPU backend.

The original unnormalized test direction and elementwise rtol=3e-3,
atol=3e-5 remain unchanged. Coverage now includes 12 and 50 CG steps, a
relative L2 requirement below 3e-3, and exact forward/transpose plan parity
for source/target tiles 4x8, 16x32 and 64x256 with partial tiles. Local checks
in the same isolated baseline environment report:

```text
FLASHOPW_OTT_HESSIAN_PATH=outputs/ott-hessian-reference
.venv-baselines/Scripts/python -m pytest -q
54 passed, 118 skipped, 1 warning in 63.83s
```

The 118 GPU tests remain skipped locally. On the failing fixture, local JAX
FP32 versus Torch FP32 maximum absolute HVP differences were 2.4795532e-5
at 12 steps and 1.2397766e-5 at 50 steps. A separate Torch float64 calculation
using the same stored FP32 points/potentials (cast to float64 without another
Sinkhorn solve) differed from Torch FP32 by at most 2.9646573e-5 and
1.1455654e-5 respectively; all entries met the unchanged test tolerances.
These are CPU diagnostics, not confirmation of GPU parity or performance.

## Confirmed server validation and ablation, 2026-10-05

The user reran `validate_paper_optimizations.sh` on allocated physical GPU 1
after pulling `283ee1c`. JAX reported `CudaDevice(id=0)` within that allocation.
The supplied log reports **133 passed, 1 warning in 49.87s**, including the
previously failing custom JAX HVP and all new reciprocity/50-step tests.
The OTT deprecation warning remained. The subsequent ablation completed and
printed the output directory:

```text
/home/doanpt/minh.nd/flash-opw/outputs/kernel_20261005T084531.029667Z
```

The following are rounded mean CUDA-event timings from the supplied console
log. Speedup is generic mean divided by tuned mean. All modes use the same
revision and input tensors; generic is the retained implementation route,
not an independently measured checkout of an older commit. No comparison to
KeOps, JAX, Tensorized or paper runtime tables is implied by these ablations.

| n=m | d | Operation | Generic ms | Specialized ms | Tuned ms | Generic/tuned | Tuned relative L2 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 64 | Vector transport | 3.389 | 2.417 | 2.179 | 1.555x | 2.07e-7 |
| 10,000 | 64 | Source gradient | 1.400 | 0.948 | 0.954 | 1.468x | 2.68e-4 |
| 10,000 | 64 | Forward + backward | 9.402 | 9.103 | 8.416 | 1.117x | 2.68e-4 |
| 10,000 | 64 | HVP | 368.687 | 266.469 | 236.366 | 1.560x | 9.44e-6 |
| 20,000 | 1024 | Vector transport | 246.663 | 239.671 | 128.272 | 1.923x | 2.16e-8 |
| 20,000 | 1024 | Source gradient | 326.351 | 82.200 | 69.621 | 4.688x | 1.37e-4 |
| 20,000 | 1024 | Forward + backward | 989.600 | 742.054 | 539.367 | 1.835x | 1.37e-4 |

The specialized wide gradient is already 3.970x faster than generic; tuning
further improves it to 4.688x. The complete wide forward/backward reduction
is 45.50%, smaller than the isolated gradient reduction because it includes
the Sinkhorn solve. Tuning has little visible effect on the d=64 gradient
(0.948 versus 0.954 ms); samples/dispersion are needed to interpret that
small difference. HVP at d=1024 was omitted by the ablation's default
`--hvp-max-d 128`, so there is no HVP result at that dimension.

All reported relative L2 errors meet the existing ablation thresholds:
5e-3 for vector/HVP and 1e-2 for TF32 gradient/forward-backward. The supplied
console log does not include environment.json, timing samples/dispersion,
coupling/CG diagnostics, selected tuning configurations or profiler traces.
Those server artifacts are still needed to investigate the performance
mechanism and run stability. The profiler warning alone does not establish
missing data; traces have not been inspected locally. The eight-panel
cross-method benchmark remains the next experiment, using the documented
unchanged mathematical protocol and a fresh output directory.
