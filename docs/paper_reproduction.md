# Paper reproduction and optimization checks

The mathematical data/settings remain uniform `[0,1]^d`, uniform weights,
full squared Euclidean cost, epsilon 0.1, 10 forward/backward iterations,
100 untimed HVP solve iterations and 50 CG steps with absolute Schur damping
1e-5. The measurement defaults remain 10 warmups and 50/30/20 repetitions.
No precision or accuracy threshold was lowered to improve a ratio.

References: [paper v3](https://arxiv.org/html/2602.03067v3),
[FlashSinkhorn source](https://github.com/ot-triton-lab/flash-sinkhorn),
[OTT-Hessian source](https://github.com/yexf308/OTT-Hessian).
The source checkouts are not established as the exact revisions used to
produce the paper tables. RTX 5080 results cannot establish A100 runtimes.

## Implemented changes

- A dedicated vector transport kernel reduces `weights * vector` directly,
  with online exponential rescaling. It handles signed vectors, transpose,
  partial tiles and single-column matrices without a padded matrix output.
- A source/target gradient kernel computes each score tile once and reuses
  it across all output feature blocks. It uses actual induced masses,
  including for unconverged potentials. Its feature accumulators can increase
  register pressure; offline compilation alone does not prove an improvement.
- Opt-in tuning searches a bounded set of row/key tiles, stages and warps,
  with unchanged dot-product precision. Compiler shared-memory metadata gates
  candidates before launch. Selected configurations, timings, registers and
  spills are recorded in `autotuning.json`; selection is cached in-process.
  Default fixed tiles and the generic transport/gradient paths remain available.
- Fixed-step CG no longer transfers residual diagnostics to the CPU when the
  caller does not request `return_info`. Diagnostics remain available outside
  measured sections.
- TF32x3 transport at small feature dimensions uses a 32-column output tile
  to fit the 64 KiB shared-memory limit. This changes tiling, not precision.

## OTT-Hessian baseline

`scripts/setup_ott_hessian.sh` clones the external baseline into ignored
`outputs/third_party/OTT-Hessian` at revision
`7eb189fe39982f587da935044480655b65939637`. No third-party code is vendored.
The adapter checks the normalized source checksum and refuses a different file.
The adapter calls its Lineax `HessianA` implementation (called
`HessianALineax` in the FlashSinkhorn reference tests).

The baseline is configured with the same shared coupling as Flash/KeOps.
OTT potentials include marginal logs: `f_OTT = epsilon*u + ||x||^2`,
`g_OTT = epsilon*v + ||y||^2`. Passing `result.f/g` directly would omit those
logs and compute a different plan. Geometry explicitly uses squared Euclidean
cost with scaling 1.0, and points/potentials stay dynamic JIT arguments.

Upstream adds `epsilon*tau2` to the Schur diagonal. The adapter passes
`tau2=damping/epsilon` to preserve the existing benchmark's absolute damping
1e-5. The adapter uses a guarded JAX CG with the same fixed-step arithmetic
and device-side breakdown conditions as Flash/KeOps for the 50-step budget.
The source's HessianA formulas and OTT transport geometry remain unchanged.
Upstream defaults instead use Lineax tolerances 1e-6 and may stop early;
forcing its tolerances to zero produced NaN in a uniform d=65 CPU fixture
after near-convergence. The fixed-step adapter avoids that failure without
changing the iteration budget or damping. It does not apply Lineax's residual
stabilization. This CG substitution is recorded in the legend, row detail,
environment and source provenance. Exact raw Lineax performance and exact
paper-table reproduction are not claimed. `load_hessian` also supports
positive tolerances when an upstream Lineax call is needed separately.

Missing dependencies/source produce a baseline failure. There is no automatic
fallback. `--jax-hvp-backend matrix-free` explicitly selects the previous
custom JAX implementation, and the plots label it accordingly. Neither baseline
is `linearize(grad(loss))`. The setup script downloads source only; benchmark
dependencies come from the isolated environment's `baselines` extra.

## Server sequence

Pull the committed changes on the server before installing dependencies:

```bash
conda activate minh
cd ~/minh.nd/flash-opw
git pull --ff-only origin main
python -m pip install -e '.[dev,plots,baselines]'
bash scripts/setup_ott_hessian.sh
```

If you already applied the transfer patch, `git pull` may report conflicting
local changes. Keep those changes and inspect `git status` before proceeding;
do not apply the patch again or discard local work.

Run real GPU correctness before the long benchmark:

```bash
bash scripts/validate_paper_optimizations.sh 1 --cases 10000:64 20000:1024 --profile
```

GPU tests cover updates across dimensions/precisions/schedules, signed vector
transport/transpose, gradients with nonuniform and unconverged masses, wide
Hadamard transport, tuned kernels and a complete fixed-step HVP. JAX tests
exercise the actual external baseline, including its potential conversion,
damping conversion and the guarded fixed 50-step protocol. `--require-gpu` prevents
an all-skipped GPU run from being accepted as validation.

The ablation runner compares `generic`, `specialized`, `tuned` in this same
revision, keeping tensors and cached HVP potentials identical. Forward+backward
includes 10 solve iterations for each mode. It reports all samples, dispersion,
relative L2 differences, actual marginal residuals and CG information. It exits
nonzero if the output comparison fails. Its baseline is the retained generic
transport route, not an independently timed checkout of an earlier commit.
HVP is omitted from dimensions above 128 by default; set `--hvp-max-d` to expand.
Profiler traces are separate untimed calls for the first case and can be viewed
in a Chrome/Perfetto trace viewer. `profile_*.txt` lists operator/device time;
traces help distinguish launches, CPU synchronization and kernel cost. Compiler
register/spill records help investigate gradient register pressure.

Then run the full protocol:

```bash
bash scripts/run_paper_benchmarks.sh 1 --autotune --block-m 64 --block-n 128 --diagnostics
```

Outputs include CSV/JSON, eight panels plus overview, environment and all samples,
`autotuning.json`, and `diagnostics.json`. Diagnostics are outside timing and
the environment records source hashes and GPU driver/clocks/temperature when
`nvidia-smi` is available. Nonfinite operation outputs keep their timing row
but are marked `nonfinite_output` and excluded from plots. Diagnostics
include Flash coupling residuals, Flash/KeOps CG information and JAX OTT-Hessian
HVP comparison against Flash using the same coupling. Diagnostic failures are
recorded separately from timing status. Forward/backward baseline parity is not
established just by these diagnostics. Ten fixed solve iterations need not
converge, so a residual is evidence to inspect rather than a reason to silently
increase the iteration budget for one method. Memory timing warms up first,
then resets the allocator peak, excluding compilation/tuning scratch.

Keep failed/OOM/estimated-memory-skipped rows. Separate runs with the custom
JAX implementation from OTT-Hessian runs. Tensorized still precomputes forward
cost outside timing. Compare paper ratios with their correct Flash denominator,
and never mix protocols or the short/full runs of the original log.

## Local developer checks

```bash
python -m pytest -q
python scripts/interpret_kernels.py
python scripts/compile_kernels.py --arch 120
python scripts/compile_tuning.py --arch 120
```

The last three commands require Linux/WSL Triton. Interpreter and offline
compilation check masks, recurrence logic and compiler/resource acceptance.
They are not GPU correctness or measured speedups. See
[validation record](validation.md) for the checks actually performed.
